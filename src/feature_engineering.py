"""Phase 2: coarse (10 km) and fine (1 km) features and training-set assembly.

Stage 1 (10 km) rows are (day, coarse cell) pairs:
    X = IMERG dynamic features + NLCD static features aggregated to 10 km
    y = AORC block-mean precipitation at 10 km
Stage 2 (1 km, ``upsampling.method == residual``) rows are (day, fine cell):
    X = bilinearly upsampled stage-1 prediction / IMERG + 1 km NLCD features
    y = AORC 1 km - upsampled stage-1 prediction (the sub-grid residual)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from .atmos import ERA5_FEATURES
from .utils import (
    LC_GROUPS,
    LOG,
    GridPair,
    coarsen_reduce,
    open_dataset,
    save_json,
    to_netcdf,
    upsample_bilinear,
    upsample_nearest,
)

M_PER_DEG = 111_195.0


# --------------------------------------------------------------------------- #
# Feature name registry
# --------------------------------------------------------------------------- #
#: Sub-daily structure and retrieval-quality predictors derived from the extra
#: fields on the IMERG daily granules.
STRUCTURE_FEATURES = ["imerg_wet_frac", "imerg_cond_intensity", "imerg_mw_frac", "imerg_prob_liquid"]


def structure_available(imerg_ds) -> bool:
    return {"precipitation_cnt", "precipitation_cnt_cond"} <= set(getattr(imerg_ds, "data_vars", {}))


def imerg_structure_features(ds: xr.Dataset) -> xr.Dataset:
    """Sub-daily structure from the daily granule's half-hour counters.

    A daily total of 30 mm can be one violent hour or twelve gentle ones, and
    the two have very different sub-grid structure. ``precipitation_cnt_cond``
    counts the raining half-hours, so the conditional intensity
    ``total / (wet half-hours x 0.5 h)`` separates convective from stratiform
    days without needing the half-hourly product at all.
    """
    cnt = ds["precipitation_cnt"].astype("float32")
    wet = ds["precipitation_cnt_cond"].astype("float32")
    p = ds["precip"].astype("float32")
    out = xr.Dataset()
    out["imerg_wet_frac"] = (wet / cnt.where(cnt > 0)).clip(0, 1).fillna(0.0)
    hours = (wet * 0.5).where(wet > 0)
    out["imerg_cond_intensity"] = (p / hours).fillna(0.0).clip(0, 200)
    if "MWprecipitation" in ds:
        # share of the estimate from a real microwave overpass rather than
        # IR morphing - a proxy for how much to trust the retrieval
        out["imerg_mw_frac"] = (ds["MWprecipitation"] / p.where(p > 0.1)).clip(0, 3).fillna(1.0)
    if "probabilityLiquidPrecipitation" in ds:
        out["imerg_prob_liquid"] = (ds["probabilityLiquidPrecipitation"].astype("float32") / 100.0).clip(0, 1)
    for v in out.data_vars:
        out[v] = out[v].astype("float32")
    return out


def load_imerg_structure(cfg: dict, time_index) -> xr.Dataset | None:
    p = Path(cfg["paths"]["processed"]) / "imerg_aligned_10km.nc"
    if not p.exists():
        return None
    ds = open_dataset(p)
    if not structure_available(ds):
        return None
    return imerg_structure_features(ds.sel(time=time_index).load())


def source_prefix(cfg: dict) -> str:
    """Name of the coarse input, used to label its derived features.

    Defaults to ``imerg`` so existing runs are untouched; a NASA POWER run sets
    ``features.source_prefix: power`` and the feature names follow.
    """
    return str((cfg.get("features") or {}).get("source_prefix", "imerg"))


def coarse_feature_names(cfg: dict) -> list[str]:
    fcfg = cfg["features"]
    sp = source_prefix(cfg)
    names = [sp, f"{sp}_roll{fcfg.get('rolling_days', 7)}", f"{sp}_pct",
             f"{sp}_nbr{fcfg.get('neighborhood', 3)}"]
    if fcfg.get("adjacent_days", False):
        names += [f"{sp}_prev", f"{sp}_next"]
    if fcfg.get("bias_climatology", False):
        names += ["imerg_bias_clim"]
    if fcfg.get("include_doy", True):
        names += ["doy_sin", "doy_cos"]
    names += ["dem_mean", "dem_std", "slope_mean", "imperv_mean"] + [f"lc_frac_{g}" for g in LC_GROUPS]
    if fcfg.get("include_latlon", False):
        names += ["lat", "lon"]
    if era5_available(cfg):
        names += list(ERA5_FEATURES)
    if structure_features_available(cfg):
        names += list(STRUCTURE_FEATURES)
    return names


def structure_features_available(cfg: dict) -> bool:
    p = Path(cfg["paths"]["processed"]) / "imerg_aligned_10km.nc"
    if not (cfg["features"].get("use_imerg_structure", True) and p.exists()):
        return False
    try:
        return structure_available(open_dataset(p))
    except Exception:
        return False


def era5_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed"]) / "era5_aligned_10km.nc"


def era5_available(cfg: dict) -> bool:
    """ERA5 is used only when enabled *and* the aligned file exists, so the
    feature-name list can never promise columns the data cannot supply."""
    return bool(cfg["features"].get("use_era5", False)) and era5_path(cfg).exists()


def fine_static_names() -> list[str]:
    return (
        ["dem", "slope", "aspect_sin", "aspect_cos", "imperv"]
        + [f"lc_frac_{g}" for g in LC_GROUPS]
        + ["dem_x_slope", "dem_anom", "dem_x_forest", "dem_x_developed"]
    )


def fine_feature_names(cfg: dict) -> list[str]:
    fcfg = cfg["features"]
    sp = source_prefix(cfg)
    names = ["pred_bl", f"{sp}_bl", f"{sp}_roll{fcfg.get('rolling_days', 7)}_bl", f"{sp}_pct_bl"]
    if fcfg.get("include_doy", True):
        names += ["doy_sin", "doy_cos"]
    return names + fine_static_names()


# --------------------------------------------------------------------------- #
# Terrain derivatives
# --------------------------------------------------------------------------- #
def compute_derived_features(dem: xr.DataArray, lulc: xr.DataArray | None = None) -> xr.Dataset:
    """Slope (deg), aspect (deg clockwise from N) and sin/cos aspect from a DEM."""
    z = dem.values.astype(np.float64)
    z = np.where(np.isnan(z), np.nanmean(z), z)
    lat = dem.lat.values
    dlat = float(np.abs(lat[1] - lat[0]))
    dlon = float(np.abs(dem.lon.values[1] - dem.lon.values[0]))
    dy = dlat * M_PER_DEG
    dx = (dlon * M_PER_DEG * np.cos(np.deg2rad(lat)))[:, None]
    gy, gx = np.gradient(z)          # gy along lat (rows, ascending north), gx along lon
    gy = gy / dy
    gx = gx / dx
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # Aspect: direction the slope faces (downhill), clockwise from north.
    aspect = (np.degrees(np.arctan2(-gx, -gy)) + 360.0) % 360.0
    flat = np.hypot(gx, gy) < 1e-6
    aspect_rad = np.deg2rad(aspect)
    ds = xr.Dataset(
        {
            "slope": (("lat", "lon"), slope.astype(np.float32), {"units": "degrees"}),
            "aspect": (("lat", "lon"), aspect.astype(np.float32), {"units": "degrees clockwise from north"}),
            "aspect_sin": (("lat", "lon"), np.where(flat, 0.0, np.sin(aspect_rad)).astype(np.float32)),
            "aspect_cos": (("lat", "lon"), np.where(flat, 0.0, np.cos(aspect_rad)).astype(np.float32)),
        },
        coords={"lat": dem.lat, "lon": dem.lon},
    )
    return ds


# --------------------------------------------------------------------------- #
# Static features
# --------------------------------------------------------------------------- #
def fine_features_1km(nlcd: xr.Dataset, grids: GridPair) -> xr.Dataset:
    """Static 1 km predictors from the aligned NLCD/DEM dataset."""
    ds = nlcd[["dem", "slope", "aspect_sin", "aspect_cos", "imperv"] + [f"lc_frac_{g}" for g in LC_GROUPS]].copy()
    # Anomaly relative to the *bilinearly* upsampled 10 km mean DEM: this is
    # continuous across block edges, so it cannot imprint the coarse grid on
    # the 1 km output (a block-mean anomaly would).
    dem_c = coarsen_reduce(nlcd["dem"], grids, "mean")
    ds["dem_anom"] = nlcd["dem"] - upsample_bilinear(dem_c, grids, clip_zero=False)
    ds["dem_x_slope"] = nlcd["dem"] * nlcd["slope"]
    ds["dem_x_forest"] = nlcd["dem"] * nlcd["lc_frac_forest"]
    ds["dem_x_developed"] = nlcd["dem"] * nlcd["lc_frac_developed"]
    for v in ds.data_vars:
        ds[v] = ds[v].astype(np.float32)
    ds.attrs["feature_names"] = fine_static_names()
    return ds[fine_static_names()]


def coarse_static_features_10km(nlcd: xr.Dataset, grids: GridPair, cfg: dict) -> xr.Dataset:
    """NLCD/DEM features aggregated to the 10 km grid."""
    ds = xr.Dataset(
        {
            "dem_mean": coarsen_reduce(nlcd["dem"], grids, "mean"),
            "dem_std": coarsen_reduce(nlcd["dem"], grids, "std"),
            "slope_mean": coarsen_reduce(nlcd["slope"], grids, "mean"),
            "imperv_mean": coarsen_reduce(nlcd["imperv"], grids, "mean"),
        }
    )
    for g in LC_GROUPS:
        ds[f"lc_frac_{g}"] = coarsen_reduce(nlcd[f"lc_frac_{g}"], grids, "mean")
    if cfg["features"].get("include_latlon", False):
        lat2d, lon2d = np.meshgrid(grids.coarse.lat, grids.coarse.lon, indexing="ij")
        ds["lat"] = (("lat", "lon"), lat2d.astype(np.float32))
        ds["lon"] = (("lat", "lon"), lon2d.astype(np.float32))
    for v in ds.data_vars:
        ds[v] = ds[v].astype(np.float32)
    return ds


# --------------------------------------------------------------------------- #
# Dynamic (time-varying) coarse features
# --------------------------------------------------------------------------- #
def nan_uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """NaN-aware moving mean over the last two axes."""
    vals = np.where(np.isnan(arr), 0.0, arr)
    cnt = (~np.isnan(arr)).astype(np.float64)
    kw = {"size": (1,) * (arr.ndim - 2) + (size, size), "mode": "nearest"}
    s = ndimage.uniform_filter(vals, **kw)
    c = ndimage.uniform_filter(cnt, **kw)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c <= 0] = np.nan
    return out


def build_climatology(imerg: xr.DataArray, train_slice: slice) -> xr.DataArray:
    """Per-cell sorted training-period values used for percentile ranks."""
    tr = imerg.sel(time=train_slice).values.astype(np.float32)
    tr = np.sort(np.where(np.isnan(tr), np.inf, tr), axis=0)  # NaNs to the end
    clim = xr.DataArray(tr, dims=("rank", "lat", "lon"), coords={"lat": imerg.lat, "lon": imerg.lon})
    clim.attrs["train_start"] = str(train_slice.start)
    clim.attrs["train_end"] = str(train_slice.stop)
    return clim


def percentile_rank(values: np.ndarray, clim: np.ndarray) -> np.ndarray:
    """Fraction of climatology values <= each value, per cell. Shapes (t, y, x), (r, y, x)."""
    t, ny, nx = values.shape
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            c = clim[:, i, j]
            c = c[np.isfinite(c)]
            if c.size == 0:
                continue
            v = values[:, i, j]
            ok = ~np.isnan(v)
            out[ok, i, j] = np.searchsorted(c, v[ok], side="right") / c.size
    return out


def coarse_dynamic_features(imerg: xr.DataArray, clim: xr.DataArray, cfg: dict) -> xr.Dataset:
    """Time-varying 10 km predictors from IMERG (all days in ``imerg``)."""
    fcfg = cfg["features"]
    win = int(fcfg.get("rolling_days", 7))
    nb = int(fcfg.get("neighborhood", 3))
    sp = source_prefix(cfg)
    da = imerg.transpose("time", "lat", "lon").astype(np.float32)
    ds = xr.Dataset({sp: da})
    ds[f"{sp}_roll{win}"] = da.rolling(time=win, min_periods=1).mean().astype(np.float32)
    ds[f"{sp}_pct"] = (("time", "lat", "lon"), percentile_rank(da.values, clim.values))
    if fcfg.get("adjacent_days", False):
        # The previous and next day as separate channels, for the case where a
        # storm straddling a day boundary lands on different days in the input
        # and the reference. Off by default, because it does not apply here:
        # data_pipeline aggregates AORC on "the timestamp's UTC date, consistent
        # with IMERG daily files", so the two already share a boundary and there
        # is no misallocation to recover. Measured on Austin it moves stage-1
        # RMSE 4.394 -> 4.393, with the two channels taking 1.8 % of the tree's
        # importance and returning nothing for it.
        #
        # Kept because it is the right feature for an input whose day convention
        # differs from the reference's. It uses the future, so it is valid for a
        # retrospective product and not for a forecast.
        ds[f"{sp}_prev"] = da.shift(time=1).astype(np.float32)
        ds[f"{sp}_next"] = da.shift(time=-1).astype(np.float32)
        # The record's first and last day have no neighbour; fall back to the
        # day itself rather than leaving a NaN the tree would have to special-case.
        for k in (f"{sp}_prev", f"{sp}_next"):
            ds[k] = ds[k].fillna(da)
    ds[f"{sp}_nbr{nb}"] = (("time", "lat", "lon"), nan_uniform_filter(da.values.astype(np.float64), nb).astype(np.float32))
    if fcfg.get("include_doy", True):
        doy = da["time"].dt.dayofyear.values.astype(np.float32)
        ang = 2 * np.pi * doy / 365.25
        ds["doy_sin"] = ("time", np.sin(ang).astype(np.float32))
        ds["doy_cos"] = ("time", np.cos(ang).astype(np.float32))
    return ds


def load_era5(cfg: dict, time_index) -> xr.Dataset | None:
    """ERA5 environmental fields on the coarse grid, aligned to ``time_index``."""
    if not era5_available(cfg):
        return None
    return open_dataset(era5_path(cfg)).sel(time=time_index).load()


def coarse_features_10km(imerg: xr.DataArray, nlcd: xr.Dataset, clim: xr.DataArray, grids: GridPair,
                         cfg: dict, era5: xr.Dataset | None = None) -> xr.Dataset:
    """Full 10 km feature Dataset (dynamic + static), ordered as ``coarse_feature_names``."""
    dyn = coarse_dynamic_features(imerg, clim, cfg)
    stat = coarse_static_features_10km(nlcd, grids, cfg)
    ds = dyn.merge(stat)
    if cfg["features"].get("bias_climatology", False):
        # Fitted on training years only, then looked up by calendar month, so a
        # test day never sees a bias estimated from its own year.
        try:
            bc = imerg_bias_climatology(cfg, grids)
            months = ds["time"].dt.month.values
            ds["imerg_bias_clim"] = (("time", "lat", "lon"),
                                     bc.values[months - 1].astype(np.float32))
        except (FileNotFoundError, KeyError) as e:
            LOG.warning("bias climatology unavailable (%s); continuing without it", e)
    if era5_available(cfg):
        era5 = era5 if era5 is not None else load_era5(cfg, imerg["time"])
        if era5 is not None:
            ds = ds.merge(era5[[v for v in ERA5_FEATURES if v in era5]])
    if structure_features_available(cfg):
        st = load_imerg_structure(cfg, imerg["time"])
        if st is not None:
            ds = ds.merge(st[[v for v in STRUCTURE_FEATURES if v in st]])
    ds.attrs["feature_names"] = coarse_feature_names(cfg)
    return ds


def features_to_matrix(ds: xr.Dataset, names: list[str]) -> np.ndarray:
    """(time*lat*lon, n_features) float32 matrix, broadcasting static/time-only vars."""
    nt, ny, nx = ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"]
    X = np.empty((nt * ny * nx, len(names)), dtype=np.float32)
    for k, name in enumerate(names):
        da = ds[name]
        if set(da.dims) == {"time", "lat", "lon"}:
            arr = da.transpose("time", "lat", "lon").values
        elif set(da.dims) == {"lat", "lon"}:
            arr = np.broadcast_to(da.values[None], (nt, ny, nx))
        elif set(da.dims) == {"time"}:
            arr = np.broadcast_to(da.values[:, None, None], (nt, ny, nx))
        else:
            raise ValueError(f"Unexpected dims for {name}: {da.dims}")
        X[:, k] = np.asarray(arr, dtype=np.float32).reshape(-1)
    return X


# --------------------------------------------------------------------------- #
# Stage-1 training data
# --------------------------------------------------------------------------- #
def prepare_training_data(cfg: dict, grids: GridPair) -> dict:
    """Build X/y for the 10 km model with a strict temporal train/val split."""
    proc = Path(cfg["paths"]["processed"])
    tcfg = cfg["time"]
    imerg = open_dataset(proc / "imerg_aligned_10km.nc")["precip"].load()
    aorc_c = open_dataset(proc / "aorc_aligned_10km.nc")["precip"].load()
    nlcd = open_dataset(proc / "nlcd_aligned_1km.nc").load()

    train_slice = slice(tcfg["train_start"], tcfg["train_end"])
    val_slice = slice(tcfg["val_start"], tcfg["val_end"])
    clim = build_climatology(imerg, train_slice)
    to_netcdf(xr.Dataset({"imerg_sorted": clim}), proc / "imerg_climatology_10km.nc")

    feats = coarse_features_10km(imerg, nlcd, clim, grids, cfg)
    feats["target"] = aorc_c.transpose("time", "lat", "lon").astype(np.float32)
    names = coarse_feature_names(cfg)

    out = {}
    for split, sl in (("train", train_slice), ("val", val_slice)):
        sub = feats.sel(time=sl)
        X = features_to_matrix(sub, names)
        y = sub["target"].values.reshape(-1).astype(np.float32)
        nt, ny, nx = sub.sizes["time"], sub.sizes["lat"], sub.sizes["lon"]
        t_idx = np.repeat(np.arange(nt), ny * nx)
        c_idx = np.tile(np.arange(ny * nx), nt)
        ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X, y, t_idx, c_idx = X[ok], y[ok], t_idx[ok], c_idx[ok]
        np.save(proc / f"X_{split}.npy", X)
        np.save(proc / f"y_{split}.npy", y)
        np.savez(proc / f"idx_{split}.npz", time_index=t_idx, cell_index=c_idx,
                 time=sub["time"].values.astype("datetime64[ns]"))
        tag = "train" if split == "train" else "test"
        to_netcdf(sub, proc / f"features_{tag}.nc")
        out[split] = {"n_rows": int(ok.sum()), "n_days": int(nt), "start": str(sub.time.values[0])[:10],
                      "end": str(sub.time.values[-1])[:10]}
        LOG.info("%s: %d rows x %d features (%d days)", split, ok.sum(), X.shape[1], nt)

    meta = {"feature_names": names, "target": "aorc_10km_mm_day", "grids": grids.to_dict(), "splits": out,
            "fine_feature_names": fine_feature_names(cfg)}
    save_json(meta, proc / "feature_metadata.json")
    return meta


# --------------------------------------------------------------------------- #
# Stage-2 (1 km residual) features
# --------------------------------------------------------------------------- #
class FineFeatureBuilder:
    """Builds 1 km feature blocks for a run of days from coarse fields.

    Parameters
    ----------
    coarse : Dataset with ``pred`` (stage-1 prediction), ``imerg``, the rolling
        mean and ``imerg_pct`` on the coarse grid for all days of interest.
    fine_static : Dataset from :func:`fine_features_1km`.
    """

    def __init__(self, coarse: xr.Dataset, fine_static: xr.Dataset, grids: GridPair, cfg: dict):
        self.prefix = source_prefix(cfg)
        self.coarse = coarse
        self.grids = grids
        self.cfg = cfg
        self.names = fine_feature_names(cfg)
        self.win = int(cfg["features"].get("rolling_days", 7))
        self.static_names = fine_static_names()
        # (n_cells, n_static) matrix for fast gathering
        self.static = np.stack([fine_static[v].values.reshape(-1) for v in self.static_names], axis=1).astype(np.float32)
        self.include_doy = cfg["features"].get("include_doy", True)

    def dynamic_fields(self, tslice: slice) -> dict[str, np.ndarray]:
        """Bilinearly upsampled coarse fields for a time slice -> (nt, ny, nx) arrays."""
        c = self.coarse.isel(time=tslice)
        sp = self.prefix
        fields = {
            "pred_bl": upsample_bilinear(c["pred"], self.grids).values,
            f"{sp}_bl": upsample_bilinear(c[sp], self.grids).values,
            f"{sp}_roll{self.win}_bl": upsample_bilinear(c[f"{sp}_roll{self.win}"], self.grids).values,
            f"{sp}_pct_bl": upsample_bilinear(c[sp + "_pct"], self.grids, clip_zero=False).values,
        }
        if self.include_doy:
            doy = c["time"].dt.dayofyear.values.astype(np.float32)
            ang = 2 * np.pi * doy / 365.25
            fields["doy_sin"] = np.sin(ang)
            fields["doy_cos"] = np.cos(ang)
        return fields

    def matrix(self, fields: dict[str, np.ndarray], day: int, cells: np.ndarray | None = None) -> np.ndarray:
        """Feature matrix for one day (all cells or the given flat cell indices)."""
        n_cells = self.static.shape[0]
        if cells is None:
            cells = np.arange(n_cells)
        X = np.empty((len(cells), len(self.names)), dtype=np.float32)
        for k, name in enumerate(self.names):
            if name in self.static_names:
                X[:, k] = self.static[cells, self.static_names.index(name)]
            elif name in ("doy_sin", "doy_cos"):
                X[:, k] = fields[name][day]
            else:
                X[:, k] = fields[name][day].reshape(-1)[cells]
        return X


def sample_fine_training_data(cfg: dict, grids: GridPair, coarse: xr.Dataset, fine_static: xr.Dataset,
                              aorc_1km: xr.DataArray, time_slice: slice, n_samples: int, seed: int,
                              months: tuple[int, ...] | None = None) -> tuple:
    """Random (day, cell) samples of 1 km features and residual targets.

    Returns (X, y, base) where ``base`` is the bilinear stage-1 prediction so
    the caller can reconstruct absolute values (``y + base``).

    ``months`` restricts sampling to a season. Orographic forcing is seasonal --
    in Colorado the cool half-year is terrain-driven while summer is convective
    -- so a residual stage can be worth shipping in one season and not the other.
    """
    rng = np.random.default_rng(seed)
    builder = FineFeatureBuilder(coarse.sel(time=time_slice), fine_static, grids, cfg)
    times = builder.coarse["time"].values
    keep = np.ones(len(times), dtype=bool)
    if months:
        keep = np.isin(pd.DatetimeIndex(times).month, list(months))
        LOG.info("Fine sampling restricted to months %s: %d of %d days",
                 sorted(months), int(keep.sum()), len(times))
    nt = int(keep.sum())
    n_cells = builder.static.shape[0]
    per_day = int(np.ceil(n_samples / max(nt, 1)))
    chunk = int(cfg["upsampling"].get("time_chunk", 32))
    Xs, ys, bases = [], [], []
    aorc_sel = aorc_1km.sel(time=time_slice)
    n_all = len(times)
    for t0 in range(0, n_all, chunk):
        t1 = min(n_all, t0 + chunk)
        if not keep[t0:t1].any():
            continue
        fields = builder.dynamic_fields(slice(t0, t1))
        obs = aorc_sel.isel(time=slice(t0, t1)).values.reshape(t1 - t0, -1)
        for d in range(t1 - t0):
            if not keep[t0 + d]:
                continue
            cells = rng.choice(n_cells, size=min(per_day, n_cells), replace=False)
            X = builder.matrix(fields, d, cells)
            base = fields["pred_bl"][d].reshape(-1)[cells]
            y = obs[d][cells] - base
            ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            Xs.append(X[ok]); ys.append(y[ok]); bases.append(base[ok])
    X = np.concatenate(Xs); y = np.concatenate(ys); base = np.concatenate(bases)
    LOG.info("Fine samples: %d rows x %d features from %d days", len(y), X.shape[1], nt)
    return X, y.astype(np.float32), base.astype(np.float32)


def imerg_bias_climatology(cfg: dict, grids: GridPair, k: float = 200.0) -> xr.DataArray:
    """Per-cell, per-month multiplicative bias of the coarse input, from training years only.

    Section 5.7 found that what stage 1 mostly learns is local bias. This hands
    it that directly, which both shortens the path and makes pooled multi-region
    training possible -- a model trained on several domains currently has to
    rediscover each one's bias from the features, and mostly fails to.

    The estimator needs shrinkage to be usable. A cell-month cell has on the
    order of ninety training days, and the quantity is a ratio of precipitation
    sums, which are heavy-tailed: the raw ratio is dominated by whether one
    convective day happened to fall in that cell. Each cell-month is therefore
    pulled toward the domain-wide ratio for that month with weight n/(n+k),
    where n is the observed accumulation. With k = 200 mm a cell needs a few
    hundred millimetres of record before its own ratio counts for much.

    Measured on Austin, it does not work, and the reason is worth more than the
    feature: the map fitted on 2015-2017 correlates **-0.107** with the same map
    fitted on 2019-2020, and the difference between them is 1.38x the map's own
    spread across cells. IMERG's per-cell monthly bias is not a stable property
    at this resolution over a few years, so the correction does not transfer.
    The tree ranks it fifth by importance and is misled by it: stage-1 val RMSE
    4.391 -> 4.463. Off by default.

    That stage 1 still beats raw IMERG by 0.42 mm while this fails says what it
    actually learns is not a static per-cell offset but a conditional one --
    bias given intensity, season and environment -- which is the stable part.

    Returns a (month, lat, lon) field; look it up by calendar month.
    """
    proc = Path(cfg["paths"]["processed"])
    sl = slice(cfg["time"]["train_start"], cfg["time"]["train_end"])
    obs = xr.open_dataset(proc / "aorc_aligned_10km.nc")["precip"].sel(time=sl)
    inp = xr.open_dataset(proc / "imerg_aligned_10km.nc")["precip"].sel(time=sl)
    # Align on the days both carry; a day present in one only would bias the ratio.
    t = np.intersect1d(obs["time"].values, inp["time"].values)
    obs, inp = obs.sel(time=t), inp.sel(time=t)

    mo = obs["time"].dt.month
    o_sum = obs.groupby(mo).sum("time")
    i_sum = inp.groupby(mo).sum("time")
    o_sum = o_sum.rename({"month": "month"}) if "month" in o_sum.dims else o_sum
    # Domain-wide ratio for the month: the target the per-cell estimate shrinks to.
    pooled = (o_sum.sum(("lat", "lon")) / i_sum.sum(("lat", "lon")).clip(min=1e-6))
    raw = o_sum / i_sum.clip(min=1e-6)
    w = i_sum / (i_sum + k)
    bias = (w * raw + (1 - w) * pooled).astype("float32")
    # A ratio outside this range is an artefact of a near-dry cell, not a bias.
    return bias.clip(0.2, 5.0).rename("imerg_bias_clim")
