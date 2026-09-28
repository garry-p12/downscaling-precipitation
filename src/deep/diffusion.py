"""Residual diffusion for precipitation downscaling (CorrDiff-style).

A frozen deterministic model supplies the conditional mean ``mu`` in u-space;
the diffusion model learns the distribution of the residual ``r = u - mu``
given the conditioning stack. Sampling therefore produces *ensembles* of
physically plausible 1 km fields rather than a single blurred estimate, which
is the property deterministic regression cannot have: minimising MSE forces the
conditional mean, and the conditional mean of a convective rain field has far
too little fine-scale variance.

Schedule: cosine (Nichol & Dhariwal). Objective: epsilon prediction.
Sampler: DDIM, so a small number of steps gives usable members.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import bilinear_up


def cosine_alphas(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule with the per-step betas clipped (Nichol & Dhariwal).

    The raw cosine reaches ``abar = 0`` exactly at ``t = T``. Sampling divides
    by ``sqrt(abar[t])`` to recover ``r0``, so an unclipped schedule amplifies
    the first reverse step by ~1e4 and the sample saturates. Clipping beta to
    ``max_beta`` keeps ``abar[T]`` small but finite, and leaves every timestep
    below the clip untouched.
    """
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1].clamp(min=1e-12)).clamp(0.0, max_beta)
    ab = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1.0 - betas, dim=0)])
    return ab.float()


class ResidualDiffusion:
    """Schedule, training loss and DDIM sampler for the residual model.

    Parameterisation
    ----------------
    The network predicts ``r0`` (the clean residual) rather than the noise.
    With epsilon-prediction the reverse step recovers ``r0`` as
    ``(r_t - sqrt(1-abar) * eps) / sqrt(abar)``, and the cosine schedule drives
    ``abar`` to ~1e-8 at ``t = T``: any error in ``eps`` is amplified ~1e4 and
    the first reverse step saturates, which made sampled residuals roughly
    twice as wide as the data. Predicting ``r0`` divides by ``sqrt(1 - abar)``
    instead, which is ~1 exactly where the epsilon form is worst.
    """

    def __init__(self, timesteps: int = 1000, device="cpu", res_scale: float = 1.0,
                 parameterization: str = "x0", r_clamp: float = 6.0, min_snr_gamma: float = 0.0):
        self.T = timesteps
        self.abar = cosine_alphas(timesteps).to(device)  # (T+1,)
        self.device = device
        self.res_scale = res_scale  # residuals are standardised by this before diffusion
        self.parameterization = parameterization
        self.r_clamp = r_clamp
        # min-SNR-gamma loss weighting (Hang et al. 2023). Uniform weighting over
        # timesteps is dominated by the high-noise end, where the best answer is
        # the conditional mean - which is exactly how a conditional diffusion
        # model collapses to a deterministic predictor and under-disperses.
        self.min_snr_gamma = min_snr_gamma

    def to(self, device):
        self.abar = self.abar.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------ #
    def q_sample(self, r0, t, noise):
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * r0 + (1 - a).sqrt() * noise

    def _context(self, cond, coarse, mu, factor: int):
        base = bilinear_up(coarse, factor)[..., : cond.shape[-2], : cond.shape[-1]]
        return torch.cat([cond, base, mu], dim=1)

    def _target(self, r0, noise, a):
        """Regression target for the chosen parameterisation."""
        if self.parameterization == "x0":
            return r0
        if self.parameterization == "eps":
            return noise
        if self.parameterization == "v":          # v = sqrt(abar) eps - sqrt(1-abar) r0
            return a.sqrt() * noise - (1 - a).sqrt() * r0
        raise ValueError(self.parameterization)

    def _snr_weight(self, a):
        if not self.min_snr_gamma:
            return torch.ones_like(a)
        snr = (a / (1 - a).clamp(min=1e-8)).clamp(max=1e8)
        g = self.min_snr_gamma
        if self.parameterization == "x0":
            return torch.minimum(snr, torch.full_like(snr, g))
        if self.parameterization == "eps":
            return torch.minimum(snr, torch.full_like(snr, g)) / snr
        return torch.minimum(snr, torch.full_like(snr, g)) / (snr + 1)

    def loss(self, net, cond, coarse, mu, target_u, mask, factor: int):
        """Masked MSE on the network's prediction, optionally min-SNR weighted."""
        r0 = (target_u - mu) / self.res_scale
        b = r0.shape[0]
        t = torch.randint(1, self.T + 1, (b,), device=r0.device)
        a = self.abar[t][:, None, None, None]
        noise = torch.randn_like(r0)
        rt = self.q_sample(r0, t, noise)
        pred = net(torch.cat([self._context(cond, coarse, mu, factor), rt], dim=1), t)
        target = self._target(r0, noise, a)
        w = mask * self._snr_weight(a)
        return (w * (pred - target) ** 2).sum() / w.sum().clamp(min=1.0)

    # ------------------------------------------------------------------ #
    def _to_r0_eps(self, out, r, a):
        """Convert the network output to (r0, eps) whatever it predicts."""
        if self.parameterization == "x0":
            r0 = out.clamp(-self.r_clamp, self.r_clamp)
            eps = (r - a.sqrt() * r0) / (1 - a).clamp(min=1e-8).sqrt()
        elif self.parameterization == "v":
            r0 = (a.sqrt() * r - (1 - a).sqrt() * out).clamp(-self.r_clamp, self.r_clamp)
            eps = (1 - a).sqrt() * r + a.sqrt() * out
        else:
            eps = out
            r0 = ((r - (1 - a).sqrt() * eps) / a.clamp(min=1e-8).sqrt()).clamp(-self.r_clamp, self.r_clamp)
        return r0, eps

    @torch.no_grad()
    def sample(self, net, cond, coarse, mu, factor: int, steps: int = 50, eta: float = 0.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """DDIM sampling; returns the residual in u-space (already rescaled)."""
        shape = mu.shape
        dev = mu.device
        r = torch.randn(shape, device=dev, generator=generator)
        ctx = self._context(cond, coarse, mu, factor)
        ts = torch.linspace(self.T, 1, steps, device=dev).long()
        for i, t in enumerate(ts):
            out = net(torch.cat([ctx, r], dim=1), t.repeat(shape[0]))
            a_t = self.abar[t]
            r0, eps = self._to_r0_eps(out, r, a_t)
            if i == len(ts) - 1:
                return r0 * self.res_scale
            a_prev = self.abar[ts[i + 1]]
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
            noise = torch.randn(shape, device=dev, generator=generator) if eta > 0 else 0.0
            r = a_prev.sqrt() * r0 + dir_xt + sigma * noise
        return r * self.res_scale
