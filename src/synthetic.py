"""Synthetic raw data for end-to-end testing without downloads or credentials.

Writes the same file formats the real pipeline consumes:
* IMERG-style HDF5 daily granules (``Grid/precipitation`` (time, lon, lat))
* AORC daily NetCDF on the fine grid (``processed/aorc_1km_daily_{year}.nc``)
* 1 km DEM / land cover / impervious GeoTIFFs under ``raw/nlcd``

The 1 km "truth" has orographic enhancement, urban enhancement, seasonal and
day-to-day variability and heavy-tailed extremes; the synthetic IMERG is the
block mean of the truth with smoothing, per-cell bias, noise, drizzle false
alarms and a few missing values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from .data_pipeline import write_geotiff
from .utils import LOG, GridPair, to_netcdf


def _smooth_noise(rng, shape, sigma, seed_scale=1.0):
    n = rng.standard_normal(shape)
    s = ndimage.gaussian_filter(n, sigma, mode="reflect")
    return s / (s.std() + 1e-9) * seed_scale


def make_static(grids: GridPair, rng) -> dict:
    ny, nx = grids.fine.shape
    lat2d, lon2d = np.meshgrid(grids.fine.lat, grids.fine.lon, indexing="ij")
    lon_c = 0.5 * (grids.fine.lon[0] + grids.fine.lon[-1])
    # Escarpment rising to the west + rolling hills.
    esc = 450.0 / (1.0 + np.exp((lon2d - lon_c + 0.4) / 0.15))
    hills = _smooth_noise(rng, (ny, nx), sigma=12, seed_scale=120.0)
    fine_tex = _smooth_noise(rng, (ny, nx), sigma=2.5, seed_scale=35.0)
    dem = 150.0 + esc + hills + fine_tex
    dem = np.clip(dem, 60, None).astype(np.float32)

    lulc = np.full((ny, nx), 71, np.uint8)  # grassland default
    veg = _smooth_noise(rng, (ny, nx), sigma=8)
    lulc[(dem > 350) & (veg > -0.2)] = 42  # evergreen forest on the plateau
    lulc[(dem <= 350) & (veg > 0.6)] = 41  # deciduous patches
    lulc[(dem <= 250) & (veg < -0.7)] = 82  # cropland in the lowlands
    lulc[(dem <= 300) & (veg > 0.2) & (veg < 0.6)] = 52  # shrub
    imperv = np.zeros((ny, nx), np.float32)
    for _ in range(3):  # cities
        ci, cj = rng.integers(ny // 5, 4 * ny // 5), rng.integers(nx // 5, 4 * nx // 5)
        r = rng.uniform(6, 14)
        d2 = ((np.arange(ny)[:, None] - ci) ** 2 + (np.arange(nx)[None, :] - cj) ** 2) / r**2
        urban = np.exp(-d2)
        imperv = np.maximum(imperv, (urban * 0.85).astype(np.float32))
        lulc[urban > 0.35] = 23
        lulc[urban > 0.7] = 24
    for _ in range(4):  # lakes
        ci, cj = rng.integers(0, ny), rng.integers(0, nx)
        d2 = ((np.arange(ny)[:, None] - ci) ** 2 + (np.arange(nx)[None, :] - cj) ** 2)
        lulc[d2 < rng.uniform(2, 5) ** 2] = 11
    return {"dem": dem, "lulc": lulc, "imperv": imperv}


def make_truth(grids: GridPair, static: dict, times: np.ndarray, rng) -> np.ndarray:
    ny, nx = grids.fine.shape
    nt = len(times)
    dem = static["dem"].astype(np.float64)
    dem_anom = dem - ndimage.gaussian_filter(dem, sigma=grids.factor / 2.0, mode="nearest")
    slope_proxy = np.hypot(*np.gradient(ndimage.gaussian_filter(dem, 1)))
    slope_proxy /= slope_proxy.mean() + 1e-9
    oro = 1.0 + 0.9 * np.clip(dem_anom / 300.0, -1, 2) + 0.25 * (dem - dem.mean()) / 500.0 + 0.15 * np.clip(slope_proxy - 1, 0, 3)
    urban = 1.0 + 0.25 * static["imperv"]
    east_west = 1.0 + 0.35 * (np.arange(nx) - nx / 2) / nx  # wetter to the east
    modulation = np.clip(oro * urban * east_west[None, :], 0.2, None)

    doy = pd.DatetimeIndex(times).dayofyear.values
    season = 1.0 + 0.6 * np.sin(2 * np.pi * (doy - 110) / 365.25) + 0.4 * np.sin(4 * np.pi * (doy - 60) / 365.25)
    season = np.clip(season, 0.3, None)
    ar = np.zeros(nt)
    for t in range(1, nt):
        ar[t] = 0.55 * ar[t - 1] + rng.standard_normal() * 0.9
    wet = (ar + rng.standard_normal(nt) * 0.5) > 0.55
    intensity = np.exp(0.9 * ar + rng.standard_normal(nt) * 0.6) * season * 4.0
    heavy = rng.random(nt) < 0.02
    intensity[heavy] *= rng.uniform(3, 6, size=heavy.sum())

    truth = np.zeros((nt, ny, nx), np.float32)
    for t in range(nt):
        if not wet[t]:
            continue
        pattern = _smooth_noise(rng, (ny, nx), sigma=max(6, min(ny, nx) // 8))
        pattern = np.exp(0.8 * pattern)
        pattern[pattern < np.quantile(pattern, rng.uniform(0.0, 0.45))] = 0.0
        cells = _smooth_noise(rng, (ny, nx), sigma=2.0)
        pattern *= np.exp(0.35 * cells)
        truth[t] = np.clip(intensity[t] * pattern * modulation, 0, 400).astype(np.float32)
    return truth


def make_imerg(grids: GridPair, truth: np.ndarray, rng) -> np.ndarray:
    f = grids.factor
    nt, ny, nx = truth.shape
    coarse = truth.reshape(nt, ny // f, f, nx // f, f).mean((2, 4))
    smoothed = ndimage.uniform_filter(coarse, size=(1, 3, 3), mode="nearest")
    imerg = 0.55 * coarse + 0.45 * smoothed
    bias = np.exp(_smooth_noise(rng, coarse.shape[1:], sigma=4, seed_scale=0.2))
    noise = np.exp(rng.standard_normal(imerg.shape) * 0.3)
    imerg = imerg * bias[None] * noise
    drizzle = (rng.random(imerg.shape) < 0.05) & (imerg < 0.05)
    imerg[drizzle] = rng.uniform(0.1, 1.0, size=drizzle.sum())
    missed = (imerg < 1.0) & (rng.random(imerg.shape) < 0.3)
    imerg[missed] = 0.0
    imerg[rng.random(imerg.shape) < 0.003] = np.nan
    return imerg.astype(np.float32)


def write_imerg_granules(cfg: dict, grids: GridPair, imerg: np.ndarray, times: np.ndarray, pad: int = 2) -> int:
    """Write IMERG-style daily HDF5 granules with a padded domain."""
    import h5py

    raw = Path(cfg["paths"]["raw"]) / "imerg"
    res = grids.coarse.res
    lat = np.round(np.concatenate([grids.coarse.lat[0] - res * np.arange(pad, 0, -1), grids.coarse.lat,
                                   grids.coarse.lat[-1] + res * np.arange(1, pad + 1)]), 6)
    lon = np.round(np.concatenate([grids.coarse.lon[0] - res * np.arange(pad, 0, -1), grids.coarse.lon,
                                   grids.coarse.lon[-1] + res * np.arange(1, pad + 1)]), 6)
    n = 0
    for k, t in enumerate(times):
        ts = pd.Timestamp(t)
        d = raw / f"{ts.year}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"3B-DAY.MS.MRG.3IMERG.{ts.strftime('%Y%m%d')}-S000000-E235959.V07B.nc4"
        full = np.full((len(lat), len(lon)), np.nan, np.float32)
        full[pad:pad + len(grids.coarse.lat), pad:pad + len(grids.coarse.lon)] = imerg[k]
        full = np.where(np.isnan(full), -9999.9, full).astype(np.float32)
        with h5py.File(path, "w") as h:
            g = h.create_group("Grid")
            g.create_dataset("lat", data=lat.astype(np.float32))
            g.create_dataset("lon", data=lon.astype(np.float32))
            g.create_dataset("time", data=np.array([(ts - pd.Timestamp("1970-01-01")).days], np.int32))
            ds = g.create_dataset("precipitation", data=full.T[None], compression="gzip")
            ds.attrs["_FillValue"] = np.float32(-9999.9)
            ds.attrs["DimensionNames"] = np.bytes_("time,lon,lat")
            ds.attrs["units"] = np.bytes_("mm/day")
        n += 1
    return n


def generate(cfg: dict, grids: GridPair, seed: int = 0) -> dict:
    """Generate the full synthetic raw dataset for the configured domain/time."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(cfg["time"]["start"], cfg["time"]["end"], freq="D").values.astype("datetime64[ns]")
    LOG.info("Synthetic data: %d days, fine grid %s, coarse grid %s", len(times), grids.fine.shape, grids.coarse.shape)
    static = make_static(grids, rng)
    nlcd_dir = Path(cfg["paths"]["raw"]) / "nlcd"
    write_geotiff(static["dem"], grids.fine, nlcd_dir / "dem_1km.tif", nodata=np.nan, descriptions=["elevation_m"])
    write_geotiff(static["lulc"], grids.fine, nlcd_dir / "lulc_1km.tif", nodata=0, descriptions=["nlcd_class"])
    write_geotiff(static["imperv"], grids.fine, nlcd_dir / "impervious_1km.tif", nodata=np.nan,
                  descriptions=["impervious_fraction"])

    truth = make_truth(grids, static, times, rng)
    years = pd.DatetimeIndex(times).year
    for y in np.unique(years):
        sel = years == y
        da = xr.DataArray(truth[sel], dims=("time", "lat", "lon"),
                          coords={"time": times[sel], **grids.fine.coords()},
                          attrs={"units": "mm/day", "long_name": "synthetic AORC daily precipitation"})
        to_netcdf(xr.Dataset({"precip": da}), Path(cfg["paths"]["processed"]) / f"aorc_1km_daily_{y}.nc")
    imerg = make_imerg(grids, truth, rng)
    n = write_imerg_granules(cfg, grids, imerg, times)
    LOG.info("Synthetic: wrote %d IMERG granules, %d AORC years, NLCD rasters", n, len(np.unique(years)))
    return {"n_days": len(times), "truth_mean": float(truth.mean()), "wet_frac": float((truth > 1).mean())}
