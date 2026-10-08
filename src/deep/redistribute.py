"""A sharpener that can only redistribute: mass is conserved by construction.

Every other model here predicts ``bilinear(coarse) + correction``, which lets
one network do two jobs at once -- fix the block total, and decide where inside
the block the rain fell. The error budget says the first job is 96 % of the
measured gain, so the second is never isolated and never gets the capacity.

This head can only do the second. The network emits one logit per fine cell; a
softmax over each block turns those into weights summing to one, and the output
is the block total spread over its cells in those proportions. The block mean is
therefore exactly the input's block mean, whatever the network does. Two
consequences:

* the entire capacity goes to placement, because nothing else is reachable;
* the error budget becomes exact rather than inferred, since the coarse term is
  pinned to the input by construction and every error the model makes is a
  placement error.

The idea is the hard-constraint layer of Harder et al. (2023); the arrangement
here differs in being applied to a residual-free sharpener fed a separately
corrected coarse field, so the two stages can be trained and scored apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# A block of 144 cells sharing one total: a logit range of +-8 spans a weight
# ratio of e^16, far more dynamic range than rainfall within a block needs, and
# keeps the softmax away from saturation where its gradient vanishes.
LOGIT_CLAMP = 8.0


class Redistributor(nn.Module):
    """``out = block_total x softmax(logits within block)``.

    ``coarse`` is the block *mean*, so the fine field is scaled by ``factor**2``
    after the softmax: the weights sum to one over the block, and multiplying by
    the sum restores the mean.
    """

    def __init__(self, backbone: nn.Module, factor: int = 12):
        super().__init__()
        self.net = backbone
        self.factor = factor
        out = _final_conv(backbone)
        if out is not None:
            # Zero-init: every logit starts equal, every weight is 1/144, and an
            # untrained model is exactly block replication -- the same "untrained
            # model is a named baseline" property the residual models have, with
            # nearest rather than bilinear as the baseline.
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)

    def forward(self, cond: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        from .models import bilinear_up
        f = self.factor
        H, W = cond.shape[-2], cond.shape[-1]
        # Checked before anything else: a patch that is not a whole number of
        # blocks would otherwise surface as a tensor-size mismatch inside the
        # concat, which says nothing about the actual cause.
        if H % f or W % f:
            raise ValueError(f"fine field {H}x{W} is not a whole number of {f}-cell blocks; "
                             f"a patch size that is not a multiple of the factor would let "
                             f"edge cells escape the mass constraint")
        base = bilinear_up(coarse, f)[..., :H, :W]      # context channel only
        logits = self.net(torch.cat([cond, base], dim=1)).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

        b, _, Hc, Wc = logits.shape
        ny, nx = Hc // f, Wc // f
        # (b, 1, ny, f, nx, f) -> softmax over the two within-block axes
        g = logits.view(b, 1, ny, f, nx, f).permute(0, 1, 2, 4, 3, 5).reshape(b, 1, ny, nx, f * f)
        w = F.softmax(g, dim=-1)

        c = coarse[..., :ny, :nx]
        tot = (c * (f * f)).unsqueeze(-1)               # block sum, from the block mean
        out = (w * tot).reshape(b, 1, ny, nx, f, f).permute(0, 1, 2, 4, 3, 5).reshape(b, 1, Hc, Wc)
        return out


def _final_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


@torch.no_grad()
def mass_error(pred: torch.Tensor, coarse: torch.Tensor, factor: int) -> float:
    """Largest block-mean departure from the input, in mm. Should be ~0.

    Worth asserting rather than assuming: a patch size that is not a multiple of
    the factor, or an off-by-one in the reshape, breaks the constraint silently
    and the model quietly becomes an ordinary sharpener again.
    """
    b, _, H, W = pred.shape
    f = factor
    ny, nx = H // f, W // f
    blk = pred[..., : ny * f, : nx * f].reshape(b, 1, ny, f, nx, f).mean(dim=(3, 5))
    return float((blk - coarse[..., :ny, :nx]).abs().max())
