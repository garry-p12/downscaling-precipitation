"""Deep downscaling architectures: CNN (U-Net), Swin transformer, diffusion U-Net.

All models map a conditioning stack at 1 km plus the coarse precipitation field
to a 1 km precipitation field in standardised log space. Every model adds a
bilinearly upsampled coarse field internally, so the network only has to learn
the *correction*, which is what the baselines cannot do.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def unpad(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    return x[..., : hw[0], : hw[1]]


def bilinear_up(x: torch.Tensor, factor: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=8, t_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.temb = nn.Linear(t_dim, cout) if t_dim else None

    def forward(self, x, t=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.temb is not None and t is not None:
            h = h + self.temb(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class UNet(nn.Module):
    """Plain U-Net over the 1 km grid, optionally conditioned on a diffusion step."""

    def __init__(self, cin: int, cout: int = 1, base: int = 64, mults=(1, 2, 4), t_dim: int | None = None):
        super().__init__()
        self.depth = len(mults)
        # ``t`` arrives as integer diffusion steps; embed it once here so every
        # ConvBlock receives a float embedding of width ``t_dim``.
        self.t_embed = TimeEmbedding(t_dim // 4) if t_dim else None
        self.stem = nn.Conv2d(cin, base, 3, padding=1)
        chans = [base * m for m in mults]
        self.downs = nn.ModuleList()
        c = base
        for ch in chans:
            self.downs.append(nn.ModuleList([ConvBlock(c, ch, t_dim=t_dim), ConvBlock(ch, ch, t_dim=t_dim),
                                             nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
            c = ch
        self.mid1 = ConvBlock(c, c, t_dim=t_dim)
        self.mid2 = ConvBlock(c, c, t_dim=t_dim)
        self.ups = nn.ModuleList()
        for ch in reversed(chans):
            self.ups.append(nn.ModuleList([nn.ConvTranspose2d(c, ch, 4, stride=2, padding=1),
                                           ConvBlock(ch * 2, ch, t_dim=t_dim), ConvBlock(ch, ch, t_dim=t_dim)]))
            c = ch
        self.out_norm = nn.GroupNorm(8, c)
        self.out = nn.Conv2d(c, cout, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t=None):
        if self.t_embed is not None and t is not None:
            t = self.t_embed(t)
        x, hw = pad_to_multiple(x, 2 ** self.depth)
        h = self.stem(x)
        skips = []
        for b1, b2, down in self.downs:
            h = b2(b1(h, t), t)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid1(h, t), t)
        for (up, b1, b2), skip in zip(self.ups, reversed(skips)):
            h = up(h)
            h = b2(b1(torch.cat([h, skip], dim=1), t), t)
        return unpad(self.out(F.silu(self.out_norm(h))), hw)


# --------------------------------------------------------------------------- #
# Swin transformer blocks (SwinIR-style, compact implementation)
# --------------------------------------------------------------------------- #
def window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)


def window_reverse(win, ws, h, w):
    b = win.shape[0] // (h * w // ws // ws)
    x = win.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.dim, self.ws, self.heads = dim, ws, heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rpb = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.rpb, std=0.02)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Relative position bias (and the shifted-window mask) enter as an
        # additive attention mask so PyTorch can use its fused SDPA kernels,
        # which is several times faster and far lighter on memory than
        # materialising the attention matrix here.
        bias = self.rpb[self.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        if mask is None:
            attn_mask = bias.unsqueeze(0)                       # (1, heads, n, n)
        else:
            nw = mask.shape[0]
            attn_mask = (bias[None] + mask[:, None]).repeat(b_ // nw, 1, 1, 1)
        attn_mask = attn_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(b_, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, ws, heads, shift, mlp_ratio=2.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _mask(self, h, w, device):
        if self.shift == 0:
            return None
        img = torch.zeros(1, h, w, 1, device=device)
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
            for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                img[:, hs, wsl, :] = cnt
                cnt += 1
        mw = window_partition(img, self.ws).squeeze(-1)
        m = mw.unsqueeze(1) - mw.unsqueeze(2)
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def forward(self, x, hw):
        h, w = hw
        b, n, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.ws)
        win = self.attn(win, self._mask(h, w, x.device))
        x = window_reverse(win, self.ws, h, w)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(b, n, c)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    """Residual Swin transformer block: several Swin layers plus a conv."""

    def __init__(self, dim, depth, ws, heads):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, ws, heads, 0 if i % 2 == 0 else ws // 2) for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, hw):
        h, w = hw
        res = x
        for blk in self.blocks:
            x = blk(x, hw)
        b, n, c = x.shape
        x = self.conv(x.transpose(1, 2).view(b, c, h, w)).flatten(2).transpose(1, 2)
        return x + res


class SwinSR(nn.Module):
    """SwinIR-style transformer operating on the 1 km grid."""

    def __init__(self, cin: int, dim: int = 96, depths=(4, 4, 4, 4), ws: int = 8, heads: int = 6, factor: int = 12):
        super().__init__()
        self.ws, self.factor = ws, factor
        self.stem = nn.Conv2d(cin, dim, 3, padding=1)
        self.groups = nn.ModuleList([RSTB(dim, d, ws, heads) for d in depths])
        self.norm = nn.LayerNorm(dim)
        self.conv_after = nn.Conv2d(dim, dim, 3, padding=1)
        self.out = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(dim // 2, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t=None):
        x, hw0 = pad_to_multiple(x, self.ws)
        h, w = x.shape[-2:]
        f = self.stem(x)
        seq = f.flatten(2).transpose(1, 2)
        for g in self.groups:
            seq = g(seq, (h, w))
        seq = self.norm(seq)
        b, n, c = seq.shape
        y = self.conv_after(seq.transpose(1, 2).view(b, c, h, w)) + f
        return unpad(self.out(y), hw0)


# --------------------------------------------------------------------------- #
# Wrappers that add the bilinear coarse field
# --------------------------------------------------------------------------- #
class Downscaler(nn.Module):
    """Deterministic downscaler: ``u_hat = bilinear(coarse) + net([cond, bilinear])``."""

    def __init__(self, backbone: nn.Module, factor: int = 12):
        super().__init__()
        self.net = backbone
        self.factor = factor

    def forward(self, cond: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        base = bilinear_up(coarse, self.factor)
        base = base[..., : cond.shape[-2], : cond.shape[-1]]
        return base + self.net(torch.cat([cond, base], dim=1))


def build_model(name: str, n_cond: int, factor: int = 12, **kw) -> nn.Module:
    """``cnn`` (U-Net), ``swin`` (SwinIR-style) or ``diffusion`` (residual DDPM backbone)."""
    cin = n_cond + 1  # + bilinear coarse channel
    if name == "cnn":
        return Downscaler(UNet(cin, 1, base=kw.get("base", 64), mults=kw.get("mults", (1, 2, 4))), factor)
    if name == "swin":
        return Downscaler(SwinSR(cin, dim=kw.get("dim", 96), depths=kw.get("depths", (4, 4, 4, 4)),
                                 ws=kw.get("ws", 8), heads=kw.get("heads", 6), factor=factor), factor)
    if name == "diffusion":
        # + 1 for the deterministic mean, + 1 for the noisy residual
        return UNet(cin + 2, 1, base=kw.get("base", 64), mults=kw.get("mults", (1, 2, 4)), t_dim=kw.get("t_dim", 64) * 4)
    raise ValueError(f"unknown model {name}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
