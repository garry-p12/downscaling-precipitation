"""Datasets and normalisation for deep downscaling models.

Task
----
Input  : IMERG precipitation on the 0.1 deg grid (35 x 30 for the Austin domain)
         plus static 1 km predictors (DEM, slope, aspect, impervious, land-cover
         fractions) and a day-of-year encoding.
Output : AORC precipitation on the 1/120 deg grid (420 x 360), i.e. a x12
         super-resolution problem with side information.

Everything is modelled in *standardised log space*
``u = (log1p(p) - mean) / std`` because daily precipitation is heavy tailed;
``to_mm`` inverts the transform.

Splits (strictly temporal, the 2019-2020 test years are never touched during
training or model selection):
    train 2015-01-01 .. 2017-12-31
    dev   2018-01-01 .. 2018-12-31   (early stopping / checkpoint choice)
    test  2019-01-01 .. 2020-12-31
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy import ndimage
from torch.utils.data import Dataset

STATIC_VARS = [
    "dem", "dem_anom", "slope", "aspect_sin", "aspect_cos", "imperv",
    "lc_frac_water", "lc_frac_developed", "lc_frac_barren", "lc_frac_forest",
    "lc_frac_shrub", "lc_frac_grass", "lc_frac_crop", "lc_frac_wetland",
]
DEFAULT_SPLITS = {
    "train": ("2000-06-01", "2017-12-31"),
    "dev": ("2018-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2020-12-31"),
}


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
@dataclass
class LogNorm:
    """log1p + standardisation, fitted on the training period only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, p: np.ndarray) -> "LogNorm":
        t = np.log1p(np.clip(p[np.isfinite(p)], 0, None))
        return cls(float(t.mean()), float(t.std() + 1e-6))

    def to_u(self, p):
        if isinstance(p, torch.Tensor):
            return (torch.log1p(p.clamp(min=0)) - self.mean) / self.std
        return (np.log1p(np.clip(p, 0, None)) - self.mean) / self.std

    def to_mm(self, u):
        if isinstance(u, torch.Tensor):
            return torch.expm1(u * self.std + self.mean).clamp(min=0)
        return np.clip(np.expm1(u * self.std + self.mean), 0, None)

    def to_dict(self):
        return {"mean": self.mean, "std": self.std}


def percentile_rank(values: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Per-cell climatological percentile of each day's precipitation.

    The reference distribution is built from the training days only, so the dev
    and test years contribute nothing to the feature. This is the ``imerg_pct``
    predictor that ranked third in the tree model's importance, included here so
    the networks see the same information as the baseline.
    """
    t, ny, nx = values.shape
    out = np.zeros_like(values, dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            ref = np.sort(values[train_idx, i, j])
            out[:, i, j] = np.searchsorted(ref, values[:, i, j], side="right") / max(ref.size, 1)
    return out


def dem_anomaly(dem: np.ndarray, factor: int) -> np.ndarray:
    """Elevation minus the bilinearly upsampled block-mean elevation.

    Matches ``feature_engineering.fine_features_1km``: the smooth reference is
    used (rather than a block-constant mean) so the predictor does not imprint
    the 10 km grid on the output.
    """
    h, w = dem.shape
    coarse = np.nan_to_num(dem).reshape(h // factor, factor, w // factor, factor).mean((1, 3))
    smooth = torch.nn.functional.interpolate(
        torch.from_numpy(coarse)[None, None], size=(h, w), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return (np.nan_to_num(dem) - smooth).astype(np.float32)


def fine_grid_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std of a (C, H, W) static stack, NaN-aware."""
    m = np.nanmean(x, axis=(1, 2), keepdims=True)
    s = np.nanstd(x, axis=(1, 2), keepdims=True) + 1e-6
    return m.astype(np.float32), s.astype(np.float32)


# --------------------------------------------------------------------------- #
# Bundle of everything a model needs
# --------------------------------------------------------------------------- #
class DownscalingData:
    """Loads the aligned NetCDFs and holds them as float32 arrays in RAM.

    Attributes
    ----------
    coarse : (T, ch, hc, wc) IMERG channels — precip and its 7-day mean, in u-space
    fine   : (T, H, W) AORC target in mm/day (NaNs kept; masked in the loss)
    static : (C, H, W) standardised static predictors
    doy    : (T, 2) sin/cos of day of year
    """

    def __init__(self, data_dir: str | Path, factor: int = 12, splits: dict | None = None,
                 roll_days: int = 7, use_era5: bool = True):
        d = Path(data_dir)
        self.factor = factor
        self.splits = splits or DEFAULT_SPLITS

        imerg_ds = xr.open_dataset(d / "imerg_aligned_10km.nc").load()
        imerg = imerg_ds["precip"]
        aorc = xr.open_dataset(d / "aorc_aligned_1km.nc")["precip"].load()
        nlcd = xr.open_dataset(d / "nlcd_aligned_1km.nc").load()
        self.time = imerg["time"].values
        self.lat, self.lon = aorc["lat"].values, aorc["lon"].values

        p_c = np.nan_to_num(imerg.values.astype(np.float32), nan=0.0)
        self.fine = aorc.values.astype(np.float32)
        self.shape_c = p_c.shape[1:]
        self.shape_f = self.fine.shape[1:]
        if self.shape_f != tuple(s * factor for s in self.shape_c):
            raise ValueError(f"grid mismatch: coarse {self.shape_c} x{factor} != fine {self.shape_f}")

        idx = self.split_index("train")
        self.norm = LogNorm.fit(self.fine[idx])
        roll = np.stack([p_c[max(0, t - roll_days + 1): t + 1].mean(0) for t in range(len(p_c))])
        pct = percentile_rank(p_c, idx)
        chans = [self.norm.to_u(p_c), self.norm.to_u(roll), pct * 2.0 - 1.0]
        self.coarse_names = ["imerg", "imerg_roll", "imerg_pct"]
        # Sub-daily structure from the daily granule's half-hour counters: how
        # much of the day was raining and how hard while it did. A daily total
        # alone cannot distinguish one violent hour from twelve gentle ones.
        if {"precipitation_cnt", "precipitation_cnt_cond"} <= set(imerg_ds.data_vars):
            cnt = np.nan_to_num(imerg_ds["precipitation_cnt"].values.astype(np.float32))
            wet = np.nan_to_num(imerg_ds["precipitation_cnt_cond"].values.astype(np.float32))
            wet_frac = np.clip(np.divide(wet, np.maximum(cnt, 1e-6)), 0, 1)
            hours = np.maximum(wet * 0.5, 1e-6)
            cond_int = np.clip(np.where(wet > 0, p_c / hours, 0.0), 0, 200)
            struct = {"imerg_wet_frac": wet_frac * 2.0 - 1.0,
                      "imerg_cond_intensity": self.norm.to_u(cond_int)}
            if "MWprecipitation" in imerg_ds:
                mw = np.nan_to_num(imerg_ds["MWprecipitation"].values.astype(np.float32))
                struct["imerg_mw_frac"] = np.clip(np.where(p_c > 0.1, mw / np.maximum(p_c, 0.1), 1.0), 0, 3) - 1.0
            for name, a in struct.items():
                m, sd = float(a[idx].mean()), float(a[idx].std() + 1e-6)
                chans.append(((a - m) / sd).astype(np.float32))
                self.coarse_names.append(name)
        # ERA5 environmental state (CAPE, moisture, wind) standardised on the
        # training years. These are 0.25 deg fields, so they carry large-scale
        # context for the coarse correction, not 1 km detail.
        era5_path = d / "era5_aligned_10km.nc"
        if use_era5 and era5_path.exists():
            era = xr.open_dataset(era5_path).load()
            era = era.sel(time=slice(str(self.time[0])[:10], str(self.time[-1])[:10]))
            for v in era.data_vars:
                a = np.nan_to_num(era[v].values.astype(np.float32))
                m, sd = float(a[idx].mean()), float(a[idx].std() + 1e-6)
                chans.append((a - m) / sd)
                self.coarse_names.append(v)
        self.coarse = np.stack(chans, axis=1).astype(np.float32)

        if "dem_anom" not in nlcd:
            nlcd["dem_anom"] = (("lat", "lon"), dem_anomaly(nlcd["dem"].values.astype(np.float32), factor))
        stat = np.stack([nlcd[v].values.astype(np.float32) for v in STATIC_VARS])
        self.static_mean, self.static_std = fine_grid_stats(stat)
        self.static = np.nan_to_num((stat - self.static_mean) / self.static_std).astype(np.float32)

        doy = np.asarray([np.datetime64(t, "D").astype("datetime64[D]") for t in self.time])
        doy = ((doy - doy.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1).astype(np.float32)
        ang = 2 * np.pi * doy / 365.25
        self.doy = np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)

        # Per-day, per-crop wetness used to bias patch sampling towards rain.
        self.wetness = ndimage.uniform_filter(p_c, size=(1, 3, 3), mode="nearest")

    # ------------------------------------------------------------------ #
    def split_index(self, split: str) -> np.ndarray:
        a, b = self.splits[split]
        t = self.time.astype("datetime64[D]")
        return np.where((t >= np.datetime64(a)) & (t <= np.datetime64(b)))[0]

    @property
    def n_static(self) -> int:
        return self.static.shape[0]

    @property
    def n_cond(self) -> int:
        """Channels of the conditioning stack built by :meth:`condition`."""
        return self.coarse.shape[1] + self.n_static + self.doy.shape[1]

    def condition(self, t_idx, y0: int = 0, x0: int = 0, h: int | None = None, w: int | None = None) -> np.ndarray:
        """Conditioning stack at fine resolution for days ``t_idx``.

        Coarse channels are nearest-upsampled onto the fine grid; the network's
        first layers smooth them, and a bilinear version is supplied separately
        by :func:`bilinear_up` for the residual formulation.
        """
        f = self.factor
        h = h if h is not None else self.shape_f[0]
        w = w if w is not None else self.shape_f[1]
        cy, cx, ch, cw = y0 // f, x0 // f, h // f, w // f
        c = self.coarse[t_idx][..., cy:cy + ch, cx:cx + cw]
        c = np.repeat(np.repeat(c, f, axis=-2), f, axis=-1)
        s = np.broadcast_to(self.static[:, y0:y0 + h, x0:x0 + w], (len(np.atleast_1d(t_idx)),) + (self.n_static, h, w))
        d = np.broadcast_to(self.doy[t_idx][..., None, None], (len(np.atleast_1d(t_idx)), self.doy.shape[1], h, w))
        return np.concatenate([c, s, d], axis=1).astype(np.float32)

    def target_u(self, t_idx, y0=0, x0=0, h=None, w=None) -> np.ndarray:
        h = h if h is not None else self.shape_f[0]
        w = w if w is not None else self.shape_f[1]
        return self.norm.to_u(self.fine[t_idx][..., y0:y0 + h, x0:x0 + w])

    def coarse_u(self, t_idx, y0=0, x0=0, h=None, w=None) -> np.ndarray:
        """Coarse precipitation (u-space) for the crop, at coarse resolution."""
        f = self.factor
        h = h if h is not None else self.shape_f[0]
        w = w if w is not None else self.shape_f[1]
        return self.coarse[t_idx][..., 0:1, y0 // f:(y0 + h) // f, x0 // f:(x0 + w) // f]


def bilinear_up(x: torch.Tensor, factor: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class PatchDataset(Dataset):
    """Random wet-biased crops for training.

    Each item is ``(cond, coarse, target, mask, weight)`` with
    ``cond``   (n_cond, P, P) standardised conditioning stack,
    ``coarse`` (1, P/f, P/f) coarse precipitation in u-space,
    ``target`` (1, P, P) AORC in u-space,
    ``mask``   (1, P, P) 1 where AORC is valid,
    ``weight`` scalar importance weight correcting the wet-biased proposal back
               to the uniform (evaluation) distribution.
    """

    def __init__(self, data: DownscalingData, split: str, patch: int = 96, n_per_epoch: int = 4096,
                 wet_bias: float = 0.8, seed: int = 0, augment: bool = True):
        self.d = data
        self.augment = augment
        # Channel positions of the aspect encoding, needed to flip it correctly.
        n_coarse = data.coarse.shape[1]
        self.i_aspect_sin = n_coarse + STATIC_VARS.index("aspect_sin")
        self.i_aspect_cos = n_coarse + STATIC_VARS.index("aspect_cos")
        self.idx = data.split_index(split)
        self.patch = patch
        self.cp = patch // data.factor
        self.n = n_per_epoch
        self.wet_bias = wet_bias
        self.seed = seed
        self.epoch = 0
        hc, wc = data.shape_c
        if self.cp > min(hc, wc):
            raise ValueError(f"patch {patch} exceeds the domain ({hc}x{wc} coarse cells)")
        self.n_y, self.n_x = hc - self.cp + 1, wc - self.cp + 1

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        t = int(rng.choice(self.idx))
        w = self.d.wetness[t, : self.n_y, : self.n_x]
        n_crops = self.n_y * self.n_x
        uniform_p = 1.0 / n_crops
        if rng.random() < self.wet_bias and w.max() > 0:
            p = (w + 1e-3).ravel()
            p = p / p.sum()
            k = int(rng.choice(p.size, p=p))
        else:
            p = None
            k = int(rng.integers(n_crops))
        # Importance weight: crops are drawn from a wet-biased proposal, so the
        # plain average over them estimates the wrong expectation. w = q/p with
        # q uniform (the evaluation distribution) corrects it; the mixture
        # density accounts for the (1 - wet_bias) uniform component.
        p_wet = (w + 1e-3).ravel()
        p_wet = p_wet / p_wet.sum()
        p_mix = self.wet_bias * p_wet[k] + (1.0 - self.wet_bias) * uniform_p
        weight = float(np.clip(uniform_p / max(p_mix, 1e-12), 0.05, 20.0))
        cy, cx = divmod(k, self.n_x)
        y0, x0 = cy * self.d.factor, cx * self.d.factor
        P = self.patch
        cond = self.d.condition(np.array([t]), y0, x0, P, P)[0]
        coarse = self.d.coarse_u(np.array([t]), y0, x0, P, P)[0]
        tgt = self.d.target_u(np.array([t]), y0, x0, P, P)[0]
        if self.augment:
            # Reflections quadruple the effective sample size, which matters
            # because the independent unit here is a weather day (~1100 of
            # them), not a grid cell. Aspect is a direction, so it has to be
            # transformed with the field: a north-south flip sends the aspect
            # angle t -> 180 - t (cos flips sign) and an east-west flip sends
            # t -> -t (sin flips sign).
            if rng.random() < 0.5:
                cond, coarse, tgt = cond[:, ::-1], coarse[:, ::-1], tgt[::-1]
                cond = cond.copy()
                cond[self.i_aspect_cos] *= -1.0
            if rng.random() < 0.5:
                cond, coarse, tgt = cond[..., ::-1], coarse[..., ::-1], tgt[..., ::-1]
                cond = cond.copy()
                cond[self.i_aspect_sin] *= -1.0
            cond, coarse, tgt = np.ascontiguousarray(cond), np.ascontiguousarray(coarse), np.ascontiguousarray(tgt)
        mask = np.isfinite(tgt).astype(np.float32)
        return (
            torch.from_numpy(cond),
            torch.from_numpy(coarse.astype(np.float32)),
            torch.from_numpy(np.nan_to_num(tgt)[None].astype(np.float32)),
            torch.from_numpy(mask[None]),
            torch.tensor(weight, dtype=torch.float32),
        )


class FullFieldDataset(Dataset):
    """Whole-domain fields for evaluation and inference (one item per day)."""

    def __init__(self, data: DownscalingData, split: str | None = None, t_idx: np.ndarray | None = None):
        self.d = data
        self.idx = t_idx if t_idx is not None else data.split_index(split)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = int(self.idx[i])
        cond = self.d.condition(np.array([t]))[0]
        coarse = self.d.coarse_u(np.array([t]))[0]
        tgt = self.d.target_u(np.array([t]))
        mask = np.isfinite(tgt).astype(np.float32)
        return (
            torch.from_numpy(cond),
            torch.from_numpy(coarse.astype(np.float32)),
            torch.from_numpy(np.nan_to_num(tgt).astype(np.float32)),
            torch.from_numpy(mask),
            t,
        )
