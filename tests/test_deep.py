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


# --------------------------------------------------------------------------- #
# Spectral loss
#
# The point of this term is that it penalises a quantity no per-cell loss can
# see, and that it is the *same* quantity src.model_comparison reports -- so
# the first test here pins it to the numpy scorer, and the rest check it
# behaves like a loss.
# --------------------------------------------------------------------------- #
from src.deep.spectral import rapsd_torch, ratio_from_sums, rapsd_sums, spectral_loss  # noqa: E402
from src.deep.train import masked_loss  # noqa: E402


def _rain_like(s=96, seed=0):
    """Smooth large scales plus convective speckle, as a rain field has."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.randn(1, 1, s // 12, s // 12, generator=g)
    coarse = torch.nn.functional.interpolate(coarse, size=(s, s), mode="bilinear", align_corners=False)
    speckle = torch.randn(s, s, generator=g).clamp(min=0) ** 2
    return (coarse[0, 0] * 3 + speckle * 2).clamp(min=0) * 4


def _blur(t, k):
    f = torch.nn.functional.avg_pool2d(t, k, stride=k)
    return torch.nn.functional.interpolate(f, scale_factor=k, mode="nearest")


def test_torch_rapsd_matches_the_scored_numpy_definition():
    field = _rain_like(96).double()
    got = rapsd_torch(field[None]).numpy()[0]
    _, want = rapsd(field.numpy())
    ok = np.isfinite(want) & (want > 0)
    assert got.shape == want.shape
    assert np.allclose(got[ok], want[ok], rtol=1e-9)


def test_spectral_loss_rises_with_blur_and_is_zero_for_a_perfect_match():
    obs = _rain_like(96)[None, None]
    mask = torch.ones_like(obs)
    assert float(spectral_loss(obs.clone(), obs, mask)) == pytest.approx(0.0, abs=1e-10)
    losses = [float(spectral_loss(_blur(obs, k), obs, mask)) for k in (2, 3, 6)]
    assert losses == sorted(losses) and losses[0] > 0


def test_spectral_loss_is_differentiable_and_tolerates_gaps():
    obs = _rain_like(96)[None, None]
    mask = torch.ones_like(obs)
    mask[..., :20, :] = 0                       # a quarter of the patch invalid
    pred = (obs * 0.5).requires_grad_(True)
    loss = spectral_loss(pred, obs, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert float(pred.grad.norm()) > 0


def test_dry_patches_contribute_nothing():
    dry = torch.zeros(2, 1, 96, 96)
    assert float(spectral_loss(dry.clone(), dry, torch.ones_like(dry))) == 0.0
    assert rapsd_sums(dry.clone(), dry, torch.ones_like(dry)) is None


def test_spectral_ratio_of_a_field_against_itself_is_one():
    obs = _rain_like(96)[None, None]
    mask = torch.ones_like(obs)
    sp, so, n = rapsd_sums(obs.clone(), obs, mask)
    assert ratio_from_sums(sp, so, n, 96) == pytest.approx(1.0, rel=1e-5)
    sp, so, n = rapsd_sums(_blur(obs, 6), obs, mask)
    assert ratio_from_sums(sp, so, n, 96) < 0.5      # a blurred field is short of power


@pytest.mark.parametrize("kind", ["logmse", "mse", "hybrid", "quantile"])
def test_zero_spectral_weight_leaves_the_existing_losses_identical(kind):
    norm = LogNorm.fit(np.concatenate([np.array([0.0, 2.0, 30.0], np.float32), np.zeros(500, np.float32)]))
    tgt = _rain_like(96)[None, None] * 0.1
    pred = tgt + 0.3
    mask = torch.ones_like(tgt)
    before = float(masked_loss(pred, tgt, mask, norm, kind, 8.6))
    after = float(masked_loss(pred, tgt, mask, norm, kind, 8.6, spectral_weight=0.0))
    assert before == after


@pytest.mark.parametrize("kind", ["logmse", "mse", "hybrid", "quantile"])
def test_spectral_weight_penalises_a_blurry_field_on_top_of_any_kind(kind):
    norm = LogNorm.fit(np.concatenate([np.array([0.0, 2.0, 30.0], np.float32), np.zeros(500, np.float32)]))
    tgt = _rain_like(96)[None, None] * 0.1
    mask = torch.ones_like(tgt)
    blurry = _blur(tgt, 6)
    plain = float(masked_loss(blurry, tgt, mask, norm, kind, 8.6))
    with_spec = float(masked_loss(blurry, tgt, mask, norm, kind, 8.6, spectral_weight=0.01))
    assert with_spec > plain


# --------------------------------------------------------------------------- #
# Spectrally adjusted MSE (Subich et al. 2025)
#
# AMSE claims to be a drop-in for MSE that removes the incentive to blur. Both
# halves of that need testing: that it really is MSE (Parseval, zero iff equal,
# sees a constant bias) and that it really removes the incentive (it accepts
# incoherent detail that restores the spectrum, where MSE rejects it).
# --------------------------------------------------------------------------- #
from src.deep.spectral import amse, _binned_spectra  # noqa: E402

_AMSE_NORM = lambda s: float(s) ** 4 * (3.0 / 8.0) ** 2


def _wmse(a, b):
    px, py, cr = _binned_spectra(a[:, 0], b[:, 0])
    return float((px + py - 2 * cr).sum()) / _AMSE_NORM(a.shape[-1])


def test_binned_spectra_satisfies_parseval():
    x = _rain_like(96).double()[None, None]
    y = (_rain_like(96, seed=4).double())[None, None]
    px, py, cr = _binned_spectra(x[:, 0], y[:, 0])
    s = 96
    w = torch.hann_window(s, periodic=False, dtype=torch.float64)
    w = w[None, :, None] * w[None, None, :]
    direct = float(((((x - y)[:, 0]) * w) ** 2).sum()) * s * s
    assert float((px + py - 2 * cr).sum()) == pytest.approx(direct, rel=1e-9)


def test_amse_is_zero_only_for_a_match_and_sees_a_constant_bias():
    o = _rain_like(96).double()[None, None]
    m = torch.ones_like(o)
    assert float(amse(o.clone(), o, m)) == pytest.approx(0.0, abs=1e-8)
    # demeaning the patch would hide this entirely
    assert float(amse(o + 1.0, o, m)) > 0.1


def test_amse_charges_more_than_mse_for_a_blur():
    o = _rain_like(96).double()[None, None]
    m = torch.ones_like(o)
    for k in (2, 3, 6):
        b = _blur(o, k)
        assert float(amse(b, o, m)) > _wmse(b, o)


def test_amse_accepts_incoherent_detail_that_mse_rejects():
    """The mechanism: restoring amplitude without adding skill."""
    o = _rain_like(96).double()[None, None]
    m = torch.ones_like(o)
    base = _blur(o, 6)
    n = _rain_like(96, seed=11).double()[None, None]
    n = n - n.mean()
    n = n - torch.nn.functional.avg_pool2d(n, 6).repeat_interleave(6, -1).repeat_interleave(6, -2)
    cand = [(al, base + al * n) for al in (0.0, 0.5, 0.75, 1.0)]
    a_best = min(cand, key=lambda c: float(amse(c[1], o, m)))[0]
    m_best = min(cand, key=lambda c: _wmse(c[1], o))[0]
    assert m_best == 0.0 and a_best > 0.0
