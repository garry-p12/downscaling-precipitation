"""Metrics that distinguish blurry fields from realistic ones.

Deterministic RMSE rewards the conditional mean, which for convective rainfall
is a smooth field with far too little fine-scale variance. These diagnostics
expose that:

* :func:`rapsd`           radially averaged power spectral density — how much
                          variance a field carries at each wavelength
* :func:`spectral_ratio`  predicted/observed power below a wavelength cut-off
                          (1.0 = correct fine-scale variance, <1 = too smooth)
* :func:`fss`             Fractions Skill Score at a spatial scale/threshold
* :func:`crps_ensemble`   probabilistic score for the diffusion ensemble
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def rapsd(field: np.ndarray, d: float = 1.0, n_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density of a 2-D field.

    Returns ``(wavelength, power)`` with wavelength in units of ``d`` (km here).
    NaNs are replaced by the field mean before the transform.
    """
    f = np.asarray(field, dtype=np.float64)
    f = np.where(np.isfinite(f), f, np.nanmean(f) if np.isfinite(f).any() else 0.0)
    h, w = f.shape
    s = min(h, w)
    f = f[:s, :s] - f[:s, :s].mean()
    win = np.hanning(s)
    f = f * win[:, None] * win[None, :]
    p = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    cy = cx = s // 2
    y, x = np.indices((s, s))
    r = np.hypot(y - cy, x - cx)
    n_bins = n_bins or s // 2
    bins = np.linspace(0.5, s / 2, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < n_bins)
    power = np.bincount(idx[ok], weights=p.ravel()[ok], minlength=n_bins)
    count = np.bincount(idx[ok], minlength=n_bins)
    power = np.where(count > 0, power / np.maximum(count, 1), np.nan)
    k = 0.5 * (bins[:-1] + bins[1:]) / (s * d)      # cycles per km
    return 1.0 / k, power


def mean_rapsd(fields: np.ndarray, d: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Mean RAPSD over a stack of 2-D fields (skipping dry ones)."""
    acc = None
    n = 0
    for f in fields:
        if not np.isfinite(f).any() or np.nanmax(f) <= 0:
            continue
        wl, p = rapsd(f, d)
        acc = p if acc is None else acc + p
        n += 1
    return wl, (acc / max(n, 1) if acc is not None else np.zeros(1))


def spectral_ratio(wl: np.ndarray, p_pred: np.ndarray, p_obs: np.ndarray, max_wavelength: float = 10.0) -> float:
    """Mean predicted/observed power for wavelengths shorter than the cut-off."""
    m = np.isfinite(p_pred) & np.isfinite(p_obs) & (wl <= max_wavelength) & (p_obs > 0)
    return float(np.mean(p_pred[m] / p_obs[m])) if m.any() else np.nan


def fss(pred: np.ndarray, obs: np.ndarray, threshold: float, scale: int) -> float:
    """Fractions Skill Score at a neighbourhood ``scale`` (cells) and threshold (mm).

    1 = perfect, 0 = no skill. Computed over one or many (lat, lon) fields.
    """
    pred = np.atleast_3d(pred.astype(np.float64)) if pred.ndim == 2 else pred.astype(np.float64)
    obs = np.atleast_3d(obs.astype(np.float64)) if obs.ndim == 2 else obs.astype(np.float64)
    valid = np.isfinite(pred) & np.isfinite(obs)
    bp = np.where(valid & (pred >= threshold), 1.0, 0.0)
    bo = np.where(valid & (obs >= threshold), 1.0, 0.0)
    size = (1, scale, scale) if bp.ndim == 3 else (scale, scale)
    fp = ndimage.uniform_filter(bp, size=size, mode="nearest")
    fo = ndimage.uniform_filter(bo, size=size, mode="nearest")
    num = np.mean((fp - fo) ** 2)
    den = np.mean(fp**2) + np.mean(fo**2)
    return float(1 - num / den) if den > 0 else np.nan


def crps_ensemble(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS for an ensemble of shape (M, ...) against obs (...)."""
    m = members.shape[0]
    if m < 2:
        return np.abs(members[0] - obs)
    mae = np.abs(members - obs[None]).mean(0)
    s = np.sort(members, axis=0)
    w = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (members.ndim - 1))
    spread = 2.0 * (w * s).sum(0) / (m * (m - 1))
    return mae - 0.5 * spread


def wet_area_ratio(pred: np.ndarray, obs: np.ndarray, threshold: float = 1.0) -> float:
    """Predicted wet-area fraction divided by the observed one."""
    v = np.isfinite(pred) & np.isfinite(obs)
    po = np.mean(obs[v] >= threshold)
    return float(np.mean(pred[v] >= threshold) / po) if po > 0 else np.nan
