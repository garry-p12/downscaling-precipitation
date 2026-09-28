"""Tests for the deep downscalers. Run: pytest -q tests/test_deep.py"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Run this file in its own process (see pytest.ini): torch's OpenMP runtime
# clashes with the one used by rasterio/xgboost in the pipeline suite.
pytestmark = pytest.mark.torch

from src.deep.data import LogNorm, dem_anomaly  # noqa: E402
from src.deep.diffusion import ResidualDiffusion, cosine_alphas  # noqa: E402
from src.deep.metrics import crps_ensemble, fss, rapsd, spectral_ratio, wet_area_ratio  # noqa: E402
from src.deep.models import WindowAttention, build_model, count_params, pad_to_multiple, unpad  # noqa: E402


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def test_lognorm_roundtrip():
    p = np.array([0.0, 0.1, 5.0, 40.0, 250.0], np.float32)
    n = LogNorm.fit(np.concatenate([p, np.zeros(500, np.float32)]))
    assert np.allclose(n.to_mm(n.to_u(p)), p, atol=1e-3)
    t = torch.tensor(p)
    assert torch.allclose(n.to_mm(n.to_u(t)), t, atol=1e-3)
    assert float(n.to_mm(n.to_u(torch.tensor([-5.0])))) == 0.0  # negatives clamp to zero


def test_dem_anomaly_is_zero_mean_and_smooth():
    rng = np.random.default_rng(0)
    dem = rng.normal(500, 50, (48, 60)).astype(np.float32)
    a = dem_anomaly(dem, 12)
    assert a.shape == dem.shape
    assert abs(float(a.mean())) < 5.0
    # a constant DEM has no anomaly
    assert np.allclose(dem_anomaly(np.full((24, 24), 7.0, np.float32), 12), 0.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def test_pad_unpad_roundtrip():
    x = torch.randn(2, 3, 420, 360)
    p, hw = pad_to_multiple(x, 16)
    assert p.shape[-2] % 16 == 0 and p.shape[-1] % 16 == 0
    assert torch.equal(unpad(p, hw), x)


@pytest.mark.parametrize("name", ["cnn", "swin"])
def test_downscaler_shapes_and_residual_init(name):
    n_cond, f, P = 18, 12, 48
    m = build_model(name, n_cond, f, base=16, dim=32, depths=(2, 2), heads=4).eval()
    cond = torch.randn(2, n_cond, P, P)
    coarse = torch.randn(2, 1, P // f, P // f)
    with torch.no_grad():
        out = m(cond, coarse)
    assert out.shape == (2, 1, P, P)
    assert count_params(m) > 0
    # The output head is zero-initialised, so an untrained model returns exactly
    # the bilinear upsampling of the coarse field — a sane starting point.
    base = torch.nn.functional.interpolate(coarse, scale_factor=f, mode="bilinear", align_corners=False)
    assert torch.allclose(out, base, atol=1e-5)


def test_swin_handles_non_multiple_of_window():
    m = build_model("swin", 4, 12, dim=32, depths=(2,), heads=4, ws=8).eval()
    cond = torch.randn(1, 4, 60, 84)  # 60 is not a multiple of 8
    coarse = torch.randn(1, 1, 5, 7)
    with torch.no_grad():
        out = m(cond, coarse)
    assert out.shape == (1, 1, 60, 84)


def test_window_attention_matches_explicit_softmax():
    """The fused-SDPA path must equal the textbook attention computation."""
    torch.manual_seed(0)
    ws, heads, dim, nw, b = 4, 2, 16, 9, 2
    att = WindowAttention(dim, ws, heads).eval()
    torch.nn.init.normal_(att.rpb, std=0.5)
    x = torch.randn(b * nw, ws * ws, dim)
    mask = torch.zeros(nw, ws * ws, ws * ws)
    mask[1:] = torch.where(torch.rand(nw - 1, ws * ws, ws * ws) < 0.3, -100.0, 0.0)
    for m in (None, mask):
        with torch.no_grad():
            got = att(x, m)
            n, c = ws * ws, dim
            qkv = att.qkv(x).reshape(b * nw, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            a = (q * att.scale) @ k.transpose(-2, -1)
            a = a + att.rpb[att.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1).unsqueeze(0)
            if m is not None:
                a = (a.view(b, nw, heads, n, n) + m[None, :, None]).view(-1, heads, n, n)
            ref = att.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(b * nw, n, c))
        assert torch.allclose(got, ref, atol=1e-5)


def test_diffusion_unet_accepts_integer_timesteps():
    net = build_model("diffusion", 6, 12, base=16)
    x = torch.randn(2, 6 + 3, 32, 32)
    t = torch.randint(1, 1000, (2,))
    assert net(x, t).shape == (2, 1, 32, 32)


# --------------------------------------------------------------------------- #
# Diffusion
# --------------------------------------------------------------------------- #
def test_cosine_schedule_monotone():
    ab = cosine_alphas(100)
    assert ab.shape == (101,)
    assert ab[0] > 0.99 and ab[-1] < 0.05
    assert bool((ab[1:] <= ab[:-1] + 1e-6).all())


def test_diffusion_loss_and_sampling_shapes():
    torch.manual_seed(0)
    n_cond, f, P = 6, 12, 24
    net = build_model("diffusion", n_cond, f, base=16)
    # The output head is zero-initialised, which under x0-parameterisation makes
    # an untrained net predict r0 = 0 for every input; perturb it so the sampler
    # is exercised the way a trained model would drive it.
    with torch.no_grad():
        net.out.weight.normal_(0, 0.05)
    diff = ResidualDiffusion(50, device="cpu", res_scale=0.5)
    cond = torch.randn(2, n_cond, P, P)
    coarse = torch.randn(2, 1, P // f, P // f)
    mu = torch.randn(2, 1, P, P)
    tgt = mu + 0.3 * torch.randn(2, 1, P, P)
    mask = torch.ones_like(tgt)
    loss = diff.loss(net, cond, coarse, mu, tgt, mask, f)
    assert loss.requires_grad and torch.isfinite(loss)
    with torch.no_grad():
        r1 = diff.sample(net, cond, coarse, mu, f, steps=4, generator=torch.Generator().manual_seed(1))
        r2 = diff.sample(net, cond, coarse, mu, f, steps=4, generator=torch.Generator().manual_seed(2))
    assert r1.shape == mu.shape and torch.isfinite(r1).all()
    assert not torch.allclose(r1, r2)  # different seeds give different members


def test_x0_parameterisation_is_stable_at_high_noise():
    """The epsilon form divides by sqrt(abar) ~ 1e-4 at t=T; x0 must not."""
    torch.manual_seed(0)
    net = build_model("diffusion", 4, 12, base=16)
    with torch.no_grad():
        net.out.weight.normal_(0, 0.05)
    cond, coarse, mu = torch.randn(1, 4, 24, 24), torch.randn(1, 1, 2, 2), torch.zeros(1, 1, 24, 24)
    x0 = ResidualDiffusion(1000, device="cpu", res_scale=1.0, parameterization="x0")
    eps = ResidualDiffusion(1000, device="cpu", res_scale=1.0, parameterization="eps")
    g = lambda: torch.Generator().manual_seed(0)  # noqa: E731
    s_x0 = x0.sample(net, cond, coarse, mu, 12, steps=8, generator=g())
    s_eps = eps.sample(net, cond, coarse, mu, 12, steps=8, generator=g())
    assert torch.isfinite(s_x0).all()
    # x0 keeps the sample inside the clamp range; eps blows past it at t=T.
    assert float(s_x0.abs().max()) <= x0.r_clamp + 1e-4
    assert float(s_x0.std()) < float(s_eps.std())


def test_masked_loss_ignores_invalid_cells():
    torch.manual_seed(0)
    net = build_model("diffusion", 4, 12, base=16)
    diff = ResidualDiffusion(50, device="cpu")
    cond, coarse = torch.randn(1, 4, 24, 24), torch.randn(1, 1, 2, 2)
    mu, tgt = torch.zeros(1, 1, 24, 24), torch.zeros(1, 1, 24, 24)
    mask = torch.zeros_like(tgt)
    mask[..., :12, :] = 1
    torch.manual_seed(3)
    a = diff.loss(net, cond, coarse, mu, tgt, mask, 12)
    tgt2 = tgt.clone()
    tgt2[..., 12:, :] = 99.0  # only masked-out cells change
    torch.manual_seed(3)
    b = diff.loss(net, cond, coarse, mu, tgt2, mask, 12)
    assert torch.allclose(a, b, atol=1e-5)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_spectral_ratio_detects_smoothing():
    from scipy import ndimage

    rng = np.random.default_rng(0)
    rough = rng.random((128, 128))
    wl, p_rough = rapsd(rough)
    _, p_smooth = rapsd(ndimage.gaussian_filter(rough, 5))
    assert spectral_ratio(wl, p_smooth, p_rough, 10.0) < 0.1
    assert 0.9 < spectral_ratio(wl, p_rough, p_rough, 10.0) < 1.1


def test_fss_ranks_by_displacement():
    from scipy import ndimage

    rng = np.random.default_rng(0)
    obs = ndimage.gaussian_filter(rng.random((160, 160)), 8)
    thr = float(np.quantile(obs, 0.9))
    perfect = fss(obs, obs, thr, 5)
    near = fss(np.roll(obs, 3, 0), obs, thr, 5)
    far = fss(np.roll(obs, 20, 0), obs, thr, 5)
    assert perfect == pytest.approx(1.0, abs=1e-6)
    assert perfect > near > far


def test_crps_rewards_calibrated_spread():
    rng = np.random.default_rng(0)
    obs = np.zeros(4000)
    sharp_right = np.zeros((20, 4000))
    biased = np.full((20, 4000), 2.0)
    spread = rng.normal(0, 2.0, (20, 4000))
    assert crps_ensemble(sharp_right, obs).mean() == pytest.approx(0.0, abs=1e-9)
    assert crps_ensemble(spread, obs).mean() < crps_ensemble(biased, obs).mean()


def test_wet_area_ratio():
    obs = np.array([0.0, 0.5, 2.0, 5.0])
    assert wet_area_ratio(obs, obs, 1.0) == pytest.approx(1.0)
    assert wet_area_ratio(np.array([2.0, 2.0, 2.0, 2.0]), obs, 1.0) == pytest.approx(2.0)
