"""A Bernoulli-gamma head: predict the distribution, not the conditional mean.

Every deterministic model in this study minimises a per-cell loss, and a
per-cell loss is minimised by the conditional mean. That is why they blur, why
their frequency bias sits below 1, and why detection has to be bought back with
loss weighting -- the heavy-cell weight in configuration E raises Austin's
frequency bias from 0.73 to 0.87 as a *side effect* of reweighting the squared
error, with no way to ask for a particular value.

Daily rainfall is a mixed discrete-continuous variable: an atom at zero, and a
right-skewed density above it. Modelling it as such gives three things the
conditional mean cannot:

* **POD becomes a dial.** The model emits P(Y >= t) per cell for any threshold,
  so detection is a decision on that probability rather than a property baked
  into the weights. One trained model sweeps the whole ROC curve.
* **Frequency bias becomes targetable.** Choosing the operating probability to
  match the observed event rate makes bias 1.0 by construction. On the Front
  Range, where the control sits at 0.15 even at a matched threshold, that is
  the difference between a usable product and an unusable one.
* **The mean is still available**, as p * mu, so RMSE remains comparable.

Parameterisation. The network emits three channels:

    logit_p      occurrence, P(Y >= wet_mm)
    r            log-space residual on the bilinear field
    log_k        gamma shape, cell-wise

with the conditional-on-rain mean ``mu = exp(log(bilinear + eps) + r)`` and the
gamma written in (mean, shape) form, so ``scale = mu / k``. Keeping ``mu``
residual on bilinear preserves the property the rest of the study relies on:
at initialisation the network contributes nothing and the predicted mean is the
bilinear field. ``logit_p`` is bias-initialised high rather than at zero so that
``p * mu`` starts at bilinear rather than at half of it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# p = sigmoid(6.0) = 0.9975: the untrained mean is bilinear to within 0.25 %,
# which keeps "an untrained model is the bilinear baseline" true for this head.
P_BIAS_INIT = 6.0
EPS = 1e-6
# Gamma shape is clamped rather than left free: k -> 0 sends the density to a
# spike at zero and the NLL to -inf, which is a degenerate optimum reachable by
# any cell whose observed value happens to be tiny.
LOG_K_CLAMP = (-3.0, 3.0)
R_CLAMP = (-6.0, 6.0)


class BernoulliGammaHead(nn.Module):
    """Wrap a 3-channel backbone into (p, mu, k), residual on bilinear."""

    def __init__(self, backbone: nn.Module, factor: int = 12, wet_mm: float = 0.1):
        super().__init__()
        self.net = backbone
        self.factor = factor
        self.wet_mm = wet_mm
        out = _final_conv(backbone)
        if out is not None:
            nn.init.zeros_(out.weight)
            with torch.no_grad():
                out.bias.zero_()
                out.bias[0] = P_BIAS_INIT      # occurrence starts near certain
                # channels 1 (residual) and 2 (log shape) start at zero, i.e.
                # mu = bilinear and k = 1 (an exponential tail).

    def forward(self, cond: torch.Tensor, coarse: torch.Tensor):
        from .models import bilinear_up
        base = bilinear_up(coarse, self.factor)
        base = base[..., : cond.shape[-2], : cond.shape[-1]]
        raw = self.net(torch.cat([cond, base], dim=1))
        logit_p, r, log_k = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3]
        return logit_p, r.clamp(*R_CLAMP), log_k.clamp(*LOG_K_CLAMP), base


def _final_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def params_mm(logit_p, r, log_k, base_mm):
    """Map raw outputs to (p, mu, k) with ``mu`` in mm and conditional on rain."""
    p = torch.sigmoid(logit_p)
    mu = torch.exp(torch.log(base_mm.clamp(min=EPS)) + r).clamp(min=EPS)
    k = torch.exp(log_k)
    return p, mu, k


def nll(logit_p, r, log_k, base_mm, obs_mm, mask, wet_mm: float = 0.1,
        heavy_mm: float = 0.0, heavy_weight: float = 1.0):
    """Negative log likelihood of a Bernoulli-gamma mixture.

    The occurrence term is scored on every valid cell; the intensity term only
    on wet ones, because the gamma density is not defined at zero and a dry cell
    carries no information about the intensity distribution.
    """
    p, mu, k = params_mm(logit_p, r, log_k, base_mm)
    wet = (obs_mm >= wet_mm) & (mask > 0)
    valid = mask > 0

    w = valid.float()
    if heavy_mm > 0 and heavy_weight != 1.0:
        w = w * torch.where(obs_mm >= heavy_mm, heavy_weight, 1.0)

    # with_logits rather than BCE on the sigmoid: the fused form is the only one
    # autocast allows (plain binary_cross_entropy raises "unsafe to autocast"),
    # and it is the numerically stable one, since it folds the log and the
    # sigmoid into a single log-sum-exp instead of taking log of a clamped
    # probability.
    occ = F.binary_cross_entropy_with_logits(logit_p, wet.float(), reduction="none")
    occ = (w * occ).sum() / w.sum().clamp(min=1.0)

    # Gamma(mean=mu, shape=k):  -log f = k*log(mu/k) + lgamma(k) - (k-1)log y + y*k/mu
    y = obs_mm.clamp(min=EPS)
    theta = (mu / k).clamp(min=EPS)
    gam = (k * torch.log(theta) + torch.lgamma(k)
           - (k - 1.0) * torch.log(y) + y / theta)
    ww = w * wet.float()
    gam = (ww * gam).sum() / ww.sum().clamp(min=1.0)
    return occ + gam, occ.detach(), gam.detach()


def mean_mm(logit_p, r, log_k, base_mm):
    """Unconditional mean E[Y] = p * mu -- the estimator RMSE is scored on."""
    p, mu, _ = params_mm(logit_p, r, log_k, base_mm)
    return p * mu


def exceedance(logit_p, r, log_k, base_mm, thresh_mm: float):
    """P(Y >= thresh) = p * (1 - F_gamma(thresh)).

    This is the dial. Detection at any threshold becomes a decision on this
    field rather than a consequence of how the loss was weighted, and sweeping
    the decision level traces the full POD/FAR curve from one trained model.

    The regularised lower incomplete gamma has no autograd-safe implementation
    in torch for the shape argument, so this is inference-only: it uses
    ``torch.special.gammainc``, which is forward-only w.r.t. ``k``.
    """
    p, mu, k = params_mm(logit_p, r, log_k, base_mm)
    z = (thresh_mm * k / mu.clamp(min=EPS)).clamp(min=0.0)
    cdf = torch.special.gammainc(k, z)
    return (p * (1.0 - cdf)).clamp(0.0, 1.0)


def quantile_mm(logit_p, r, log_k, base_mm, q: float, iters: int = 40):
    """Inverse CDF by bisection, for an upper-quantile product.

    Solves P(Y >= y) = 1 - q. Bisection rather than a closed form because the
    gamma quantile function is not available in torch; 40 iterations on a
    bracket of [0, 2000] mm resolves to under 2e-3 mm.
    """
    target = 1.0 - q
    lo = torch.zeros_like(base_mm)
    hi = torch.full_like(base_mm, 2000.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = _exceed_at(logit_p, r, log_k, base_mm, mid)
        hi = torch.where(above < target, mid, hi)
        lo = torch.where(above < target, lo, mid)
    return 0.5 * (lo + hi)


def _exceed_at(logit_p, r, log_k, base_mm, thresh: torch.Tensor):
    p, mu, k = params_mm(logit_p, r, log_k, base_mm)
    z = (thresh * k / mu.clamp(min=EPS)).clamp(min=0.0)
    return (p * (1.0 - torch.special.gammainc(k, z))).clamp(0.0, 1.0)


def calibrate_operating_point(prob: torch.Tensor, obs_mm: torch.Tensor, mask: torch.Tensor,
                              thresh_mm: float) -> float:
    """Pick the probability cut that makes frequency bias 1.0.

    Frequency bias is forecast count over observed count, so the cut that
    equalises them is the observed event rate's quantile of the predicted
    probability field. This is the step that makes "bias 1.0 by construction"
    literal rather than aspirational.
    """
    valid = mask > 0
    pv, ov = prob[valid], obs_mm[valid]
    rate = (ov >= thresh_mm).float().mean().item()
    if rate <= 0:
        return 1.0
    # torch.quantile refuses inputs beyond ~16M elements, and a 731-day field is
    # two orders past that. Sorting the whole thing is both allowed and exact.
    pv = pv.float().flatten().sort().values
    idx = int(round((1.0 - rate) * (pv.numel() - 1)))
    return float(pv[min(max(idx, 0), pv.numel() - 1)])
