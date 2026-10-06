"""A differentiable spectral penalty, so structure can be trained and not only scored.

``src/deep/metrics.py`` measures how much fine-scale variance a field carries
(:func:`~src.deep.metrics.rapsd`, :func:`~src.deep.metrics.spectral_ratio`), but
nothing in the training objective ever saw it: every ``--loss`` option is a
per-cell intensity loss, and a per-cell loss is minimised by the conditional
mean, which is smooth. The POWER comparison is the clearest statement of the
problem -- the XGBoost product beats bilinear on RMSE while carrying 0.06 % of
the observed power below 10 km.

This module reproduces the scoring definition in torch so the quantity trained
is the quantity reported: square crop, remove the mean, Hanning window in both
directions, ``|FFT2|**2``, radial average. The penalty is then the squared gap
between predicted and observed *log* power, averaged over the bins below the
cut-off wavelength.

Two deliberate choices:

* **Log power, not power.** The spectrum spans several orders of magnitude
  across wavelengths; a linear penalty would be entirely decided by the largest
  scales, which are exactly the ones a downscaler already gets right. In log
  space a factor-of-two error costs the same at every wavelength, which is what
  "the ratio should be 1" actually means.
* **Invalid cells are filled with the patch mean, in both fields alike.** A
  Fourier transform needs a complete rectangle. Filling introduces spectral
  leakage, but it introduces the *same* leakage into prediction and target, so
  the difference between them stays meaningful. This is the same compromise
  :func:`~src.deep.metrics.rapsd` makes when it replaces NaNs with the mean.
"""
from __future__ import annotations

import torch

# Radial bin layout depends only on the patch size, so it is built once.
_CACHE: dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]] = {}


def _radial(s: int, device: torch.device, dtype: torch.dtype):
    """Return (bin index, in-range mask, n_bins, bin centre radii) for an s x s spectrum."""
    key = (s, str(device), str(dtype))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    c = s // 2
    y, x = torch.meshgrid(torch.arange(s, device=device), torch.arange(s, device=device), indexing="ij")
    r = torch.hypot((y - c).to(dtype), (x - c).to(dtype)).reshape(-1)
    n_bins = s // 2
    edges = torch.linspace(0.5, s / 2, n_bins + 1, device=device, dtype=dtype)
    # torch's `right` is the opposite sense to numpy's: right=True gives
    # edges[i-1] <= r < edges[i], which is what np.digitize does by default
    idx = (torch.bucketize(r, edges, right=True) - 1).clamp(0, n_bins - 1)
    # radii off the end of the annulus would be clamped into a real bin, so they
    # are dropped instead of polluting the lowest and highest wavenumbers
    keep = (r >= edges[0]) & (r < edges[-1])
    centres = 0.5 * (edges[:-1] + edges[1:])
    _CACHE[key] = (idx, keep, n_bins, centres)
    return _CACHE[key]


def rapsd_torch(f: torch.Tensor) -> torch.Tensor:
    """Radially averaged power spectral density of (B, H, W) fields.

    Mirrors :func:`src.deep.metrics.rapsd`; returns (B, n_bins), bin 0 being the
    longest wavelength. Input must already be finite.
    """
    b, h, w = f.shape
    s = min(h, w)
    f = f[:, :s, :s]
    f = f - f.mean(dim=(1, 2), keepdim=True)
    win = torch.hann_window(s, periodic=False, device=f.device, dtype=f.dtype)
    f = f * win[None, :, None] * win[None, None, :]
    p = torch.fft.fftshift(torch.fft.fft2(f), dim=(-2, -1)).abs() ** 2
    idx, keep, n_bins, _ = _radial(s, f.device, f.dtype)
    p = p.reshape(b, -1)[:, keep]
    i = idx[keep]
    M = torch.zeros(i.numel(), n_bins, device=f.device, dtype=f.dtype)
    M[torch.arange(i.numel(), device=f.device), i] = 1.0
    total = p @ M                      # deterministic, unlike index_add_
    count = M.sum(0)
    return total / count.clamp(min=1.0)


def band_mask(s: int, cut_cells: float, device, dtype) -> torch.Tensor:
    """Bins whose wavelength is at or below ``cut_cells`` (1 cell = 1 km here)."""
    _, _, _, centres = _radial(s, device, dtype)
    wavelength = s / centres.clamp(min=1e-6)
    return wavelength <= cut_cells


def spectral_loss(pred: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor,
                  cut_cells: float = 10.0, min_wet_mm: float = 1.0,
                  eps: float = 1e-8, wet_ref: torch.Tensor | None = None) -> torch.Tensor:
    """Squared log-power gap below ``cut_cells``, averaged over wet patches.

    ``pred``/``obs``/``mask`` are (B, 1, H, W). Patches whose observed field
    never reaches ``min_wet_mm`` are skipped: a dry patch has no structure to
    match, and its spectrum is numerical noise that would otherwise be fitted.

    The fields may be in mm or in the standardised log space the network
    predicts in. Subich et al. (2025, ICML) note that a spectral decomposition
    suits normally-distributed fields, and that precipitation -- localised and
    non-negative -- is the variable their own spherical-harmonic loss handles
    worst. Taking the transform in log space tests whether that is our problem
    too. ``wet_ref`` then supplies the mm field used to decide which patches
    carry rain, since ``min_wet_mm`` is meaningless in log units.
    """
    p, o, m = pred[:, 0], obs[:, 0], mask[:, 0]
    ref = (wet_ref if wet_ref is not None else obs)[:, 0]
    valid = m > 0
    n_valid = valid.flatten(1).sum(1).clamp(min=1.0)
    # fill invalid cells with the patch mean, identically in both fields
    fill_p = (p * valid).flatten(1).sum(1) / n_valid
    fill_o = (o * valid).flatten(1).sum(1) / n_valid
    p = torch.where(valid, p, fill_p[:, None, None])
    o = torch.where(valid, o, fill_o[:, None, None])

    wet = (ref * valid).flatten(1).amax(1) >= min_wet_mm
    if not bool(wet.any()):
        return pred.sum() * 0.0
    p, o = p[wet], o[wet]

    sp, so = rapsd_torch(p), rapsd_torch(o)
    band = band_mask(min(p.shape[-2], p.shape[-1]), cut_cells, p.device, p.dtype)
    if not bool(band.any()):
        return pred.sum() * 0.0
    d = torch.log(sp[:, band] + eps) - torch.log(so[:, band] + eps)
    return (d ** 2).mean()


@torch.no_grad()
def rapsd_sums(pred_mm: torch.Tensor, obs_mm: torch.Tensor, mask: torch.Tensor,
               min_wet_mm: float = 1.0):
    """Accumulate RAPSD over a split, the way the scored diagnostic does.

    ``src.model_comparison`` averages the per-day spectrum first and takes the
    predicted/observed ratio second. Evaluating it the same way here means the
    number printed on the dev line is comparable with the number in the
    comparison table, rather than merely similar to it.

    Returns ``(sum_pred, sum_obs, n_days)``; ``None`` if every field was dry.
    """
    p, o, m = pred_mm[:, 0], obs_mm[:, 0], mask[:, 0]
    valid = m > 0
    n_valid = valid.flatten(1).sum(1).clamp(min=1.0)
    p = torch.where(valid, p, ((p * valid).flatten(1).sum(1) / n_valid)[:, None, None])
    o = torch.where(valid, o, ((o * valid).flatten(1).sum(1) / n_valid)[:, None, None])
    wet = (o * valid).flatten(1).amax(1) >= min_wet_mm
    if not bool(wet.any()):
        return None
    p, o = p[wet].float(), o[wet].float()
    return rapsd_torch(p).sum(0), rapsd_torch(o).sum(0), int(wet.sum())


def ratio_from_sums(sum_pred: torch.Tensor, sum_obs: torch.Tensor, n: int,
                    size: int, cut_cells: float = 10.0) -> float:
    """Mean predicted/observed power below the cut-off -- 1.0 is realistic texture."""
    if n == 0:
        return float("nan")
    band = band_mask(size, cut_cells, sum_pred.device, sum_pred.dtype)
    ok = band & (sum_obs > 0)
    if not bool(ok.any()):
        return float("nan")
    return float((sum_pred[ok] / sum_obs[ok]).mean())


# --------------------------------------------------------------------------- #
# Spectrally adjusted MSE
# --------------------------------------------------------------------------- #
_FULL: dict[tuple[int, str, str], tuple[torch.Tensor, int]] = {}
_ONEHOT: dict[tuple[int, str, str, int], torch.Tensor] = {}


def _bin_matrix(s: int, device, dtype, group: int = 1) -> torch.Tensor:
    """One-hot (numel, n_bins) matrix for deterministic radial binning.

    ``index_add_`` on CUDA accumulates with atomics, so its summation order
    varies between runs. That is the whole source of the run-to-run variation
    we measured at fixed seed: two identical spectral-penalty runs gave dev
    4.8772 and 4.8887 while the per-cell control reproduced exactly. A matmul
    against a fixed one-hot matrix sums in a fixed order and removes it, at the
    cost of holding an (numel x n_bins) matrix -- 11 MB at patch 180.
    """
    key = (s, str(device), str(dtype), group)
    hit = _ONEHOT.get(key)
    if hit is not None:
        return hit
    idx, n_tot = _radial_full(s, device, dtype, group)
    m = torch.zeros(idx.numel(), n_tot, device=device, dtype=dtype)
    m[torch.arange(idx.numel(), device=device), idx] = 1.0
    _ONEHOT[key] = m
    return m


def _radial_full(s: int, device: torch.device, dtype: torch.dtype, group: int = 1):
    """Radial bin index covering *every* mode, with one extra bin for the corners.

    :func:`_radial` drops radii beyond ``s/2`` because a radial average is only
    well defined inside the inscribed circle. That is right for a diagnostic and
    wrong for a loss: the corners are 1 - pi/4, about 21%, of the Fourier plane,
    and modes excluded from the sum are modes the loss does not constrain. They
    are collected into a final anisotropic bin so that Parseval holds and the
    loss is zero only when the fields actually match.
    """
    key = (s, str(device), str(dtype), group)
    hit = _FULL.get(key)
    if hit is not None:
        return hit
    c = s // 2
    y, x = torch.meshgrid(torch.arange(s, device=device), torch.arange(s, device=device), indexing="ij")
    r = torch.hypot((y - c).to(dtype), (x - c).to(dtype)).reshape(-1)
    n_bins = s // 2
    edges = torch.linspace(0.5, s / 2, n_bins + 1, device=device, dtype=dtype)
    idx = (torch.bucketize(r, edges, right=True) - 1).clamp(0, n_bins)
    if group > 1:
        # A 96-cell crop has very few modes in its innermost annuli, against the
        # 2l+1 per degree a spherical harmonic basis gives. Widening the bins
        # trades scale resolution for a better-determined amplitude estimate.
        n_ann = (n_bins + group - 1) // group
        ann = (idx.clamp(max=n_bins - 1) // group)
        idx = torch.where(idx == n_bins, torch.full_like(ann, n_ann), ann)
        n_bins = n_ann
    _FULL[key] = (idx, n_bins + 1)
    return _FULL[key]


def _binned_spectra(x: torch.Tensor, y: torch.Tensor, group: int = 1):
    """Per-annulus PSD of each field and their cross-spectrum.

    Returns (PSD_x, PSD_y, Cross) each (B, n_bins + 1), summed within the bin so
    that ``sum_k (PSD_x + PSD_y - 2*Cross)`` is exactly the mean squared error of
    the windowed fields (Parseval).
    """
    b, h, w = x.shape
    s = min(h, w)
    x, y = x[:, :s, :s], y[:, :s, :s]
    # No demeaning here, unlike the diagnostic in `rapsd_torch`. The zero
    # wavenumber IS the patch mean, and a loss that drops it cannot see a
    # constant bias -- the model would be free to offset every patch.
    win = torch.hann_window(s, periodic=False, device=x.device, dtype=x.dtype)
    win = win[None, :, None] * win[None, None, :]
    fx = torch.fft.fftshift(torch.fft.fft2(x * win), dim=(-2, -1))
    fy = torch.fft.fftshift(torch.fft.fft2(y * win), dim=(-2, -1))
    M = _bin_matrix(s, x.device, x.dtype, group)
    def acc(v):
        return v.reshape(b, -1) @ M
    return acc(fx.abs() ** 2), acc(fy.abs() ** 2), acc((fx * fy.conj()).real)


def amse(pred: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor,
         min_wet_mm: float = 0.0, eps: float = 1e-10,
         wet_ref: torch.Tensor | None = None, pool: str = "sample",
         group: int = 1) -> torch.Tensor:
    """Spectrally adjusted mean squared error, after Subich et al. (2025, ICML).

    Written in the spectral domain, MSE separates exactly into an amplitude
    error and a decorrelation term::

        MSE = sum_k (sqrt(PSD_x) - sqrt(PSD_y))^2
                  + 2 sqrt(PSD_x PSD_y) (1 - Coh_k)

    The blurring incentive lives in the *coupling*. Because the decorrelation
    penalty is scaled by the geometric mean of the two amplitudes, a model whose
    coherence is poor at some scale can cut its loss by shrinking its own
    amplitude there; the optimum is ``sqrt(PSD_x / PSD_y) = Coh_k``, so the
    prediction's spectrum is dragged down to its own skill. Replacing that
    prefactor with ``max(PSD_x, PSD_y)`` removes the reward -- below the target
    amplitude the term no longer depends on the prediction -- while leaving the
    amplitude term, which is minimised exactly at ``PSD_x = PSD_y``.

    The result is parameter-free: unlike the additive penalty of
    :func:`spectral_loss` there is no weight to choose and no equilibrium
    between competing terms to overshoot. It is zero iff the fields match.

    ``min_wet_mm`` defaults to 0 here, unlike :func:`spectral_loss`. That guard
    exists because a log-power gap is meaningless on an all-dry patch; AMSE has
    no logarithm and goes to zero on one by itself. As a *replacement* for MSE
    it must also keep dry areas dry, so it sees every patch.

    Two departures from the original, both from working on a patch rather than
    a sphere. A Hanning window is applied before the transform, so this is the
    MSE of the windowed patch; without it, edge discontinuities leak across all
    wavenumbers. And the decomposition is a 2-D Cartesian FFT with radial bins
    rather than spherical harmonics.
    """
    p, o, m = pred[:, 0], obs[:, 0], mask[:, 0]
    ref = (wet_ref if wet_ref is not None else obs)[:, 0]
    valid = m > 0
    n_valid = valid.flatten(1).sum(1).clamp(min=1.0)
    p = torch.where(valid, p, ((p * valid).flatten(1).sum(1) / n_valid)[:, None, None])
    o = torch.where(valid, o, ((o * valid).flatten(1).sum(1) / n_valid)[:, None, None])
    wet = (ref * valid).flatten(1).amax(1) >= min_wet_mm
    if not bool(wet.any()):
        return pred.sum() * 0.0
    p, o = p[wet], o[wet]

    px, py, cross = _binned_spectra(p, o, group)
    if pool == "batch":
        # Pool the three spectra across the batch *before* forming the loss, so
        # the max() switch sees one well-determined amplitude per bin rather
        # than one noisy estimate per crop. Averaging per-sample AMSE values
        # instead would leave the noisy comparison inside.
        px, py, cross = (v.mean(0, keepdim=True) for v in (px, py, cross))
    amp = (torch.sqrt(px + eps) - torch.sqrt(py + eps)) ** 2
    # Coh = Cross / sqrt(PSD_x PSD_y); the prefactor max(.) replaces the
    # geometric mean, which is the whole modification
    coh = cross / torch.sqrt((px + eps) * (py + eps))
    decor = 2.0 * torch.maximum(px, py) * (1.0 - coh)
    # torch's fft2 is unnormalised, so the spectral sum carries s^2 and the
    # per-pixel mean another s^2; dividing by s^4 puts AMSE in the same units
    # and roughly the same magnitude as the mm-space MSE it replaces, which is
    # what lets the existing learning rate carry over unchanged
    # s^4 undoes the unnormalised fft2 and the per-pixel mean; the window
    # factor (mean of w^2, 9/64 for a 2-D Hann) undoes the taper, so AMSE lands
    # in the same units and magnitude as the mm-space MSE it replaces
    s = min(p.shape[-2], p.shape[-1])
    win_norm = (3.0 / 8.0) ** 2
    return (amp + decor).sum(dim=1).mean() / (float(s) ** 4 * win_norm)
