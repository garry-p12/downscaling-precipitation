"""ERA5 atmospheric predictors for the coarse (10 km) stage.

Why these fields
----------------
IMERG tells the model *how much* the satellite thought fell; it says nothing
about the atmospheric state that produced it. Warm-season Texas rainfall is
convective, and nothing in the existing predictor set indexes instability or
moisture. The fields below are the standard environmental controls:

    cape        convective available potential energy - instability
    tcwv        total column water vapour ("PWV") - available moisture
    u10, v10    10 m wind - low-level flow and storm motion
    t2m         2 m temperature - thermodynamic state / season beyond day-of-year

From these we derive wind speed and direction (sin/cos), and a moisture-flux
proxy ``tcwv * wind_speed``.

Note the resolution asymmetry: ERA5 is 0.25 deg (~28 km), *coarser* than
IMERG's 0.1 deg. These features can therefore only sharpen the 10 km bias
correction - they carry no information about 1 km sub-grid structure. What they
can do is separate convective from stratiform regimes, which have very
different sub-grid variability, so they may matter more for the spread of the
generative model than for any conditional mean.

Source: ARCO-ERA5 (analysis-ready cloud-optimised Zarr on GCS, anonymous).
CAPE lives only in the hourly full-resolution store; the rest come from the
6-hourly store, which is ~7x cheaper to read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils import LOG, GridPair, to_netcdf

ARCO_HOURLY = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_6H = "gs://gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr"

# short name -> (ARCO variable, store, daily reducers)
ERA5_VARS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "cape": ("convective_available_potential_energy", "hourly", ("max", "mean")),
    "tcwv": ("total_column_water_vapour", "6h", ("mean", "max")),
    "u10": ("10m_u_component_of_wind", "6h", ("mean",)),
    "v10": ("10m_v_component_of_wind", "6h", ("mean",)),
    "t2m": ("2m_temperature", "6h", ("mean",)),
    # ERA5's own precipitation. The environmental predictors above describe the
    # conditions for rain without ever stating how much the reanalysis thinks
    # fell -- and over cool-season orography, where IMERG is weakest, reanalysis
    # precipitation is often the better estimate. Snowfall is carried separately
    # because frozen precipitation is exactly the regime a microwave retrieval
    # struggles with, so its share of the total is informative on its own.
    "tp": ("total_precipitation", "hourly", ("sum",)),
    "sf": ("snowfall", "hourly", ("sum",)),
}


def _open(store: str):
    return xr.open_zarr(ARCO_HOURLY if store == "hourly" else ARCO_6H,
                        chunks={"time": 24}, storage_options=dict(token="anon"))


def fetch_era5_daily(grids: GridPair, start: str, end: str, variables=None, pad_deg: float = 0.75,
                     cache_dir: str | Path | None = None) -> xr.Dataset:
    """Daily ERA5 aggregates over the domain, on ERA5's own 0.25 deg grid.

    Downloads month by month and caches each month so the (slow) transfer can be
    resumed. ERA5 longitudes are 0-360 and latitudes descend.
    """
    variables = variables or list(ERA5_VARS)
    lon_min, lat_min, lon_max, lat_max = grids.coarse.edges
    sel = dict(latitude=slice(lat_max + pad_deg, lat_min - pad_deg),
               longitude=slice(lon_min % 360 - pad_deg, lon_max % 360 + pad_deg))
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    stores = {s: _open(s) for s in {ERA5_VARS[v][1] for v in variables}}
    months = pd.period_range(start, end, freq="M")
    pieces = []
    for per in months:
        tag = per.strftime("%Y%m")
        cf = cache / f"era5_{tag}.nc" if cache else None
        cached, todo = None, variables
        if cf and cf.exists():
            cached = xr.open_dataset(cf).load()
            have = set(cached.data_vars)
            want = {v if how == "mean" else f"{v}_{how}"
                    for v in variables for how in ERA5_VARS[v][2]}
            if want <= have:
                pieces.append(cached)
                continue
            # Cached before a variable was added to ERA5_VARS. Serving it as-is
            # would drop the new predictor with no error anywhere, so fetch the
            # gap -- but only the gap: re-downloading five cached variables to
            # add two is hours of transfer for nothing.
            todo = [v for v in variables
                    if not {v if how == "mean" else f"{v}_{how}"
                            for how in ERA5_VARS[v][2]} <= have]
            LOG.info("ERA5 %s: cache missing %s, fetching those only",
                     tag, ", ".join(sorted(want - have)))
        m0 = max(pd.Timestamp(start), per.start_time)
        m1 = min(pd.Timestamp(end) + pd.Timedelta(hours=23), per.end_time)
        out = {}
        for v in todo:
            arco_name, store, reducers = ERA5_VARS[v]
            da = stores[store][arco_name].sel(time=slice(m0, m1), **sel)
            daily = da.resample(time="1D")
            for how in reducers:
                name = v if how == "mean" else f"{v}_{how}"
                out[name] = getattr(daily, how)().compute().astype("float32")
        ds = xr.Dataset(out)
        if cached is not None:
            ds = xr.merge([cached, ds])
            cached.close()
        if cf:
            ds.to_netcdf(cf, mode="w")
        pieces.append(ds)
        LOG.info("ERA5 %s: %d days x %d fields", tag, ds.sizes["time"], len(ds.data_vars))
    return xr.concat(pieces, dim="time").sortby("time")


def derive_and_regrid(ds: xr.Dataset, grids: GridPair, time_axis: np.ndarray) -> xr.Dataset:
    """Add derived fields and interpolate onto the 10 km coarse grid."""
    ds = ds.copy()
    if {"u10", "v10"} <= set(ds.data_vars):
        ds["wind_speed"] = np.hypot(ds["u10"], ds["v10"])
        ang = np.arctan2(ds["v10"], ds["u10"])
        ds["wind_sin"] = np.sin(ang)
        ds["wind_cos"] = np.cos(ang)
        ds = ds.drop_vars(["u10", "v10"])
        if "tcwv" in ds:
            # crude vertically-integrated moisture transport
            ds["moist_flux"] = ds["tcwv"] * ds["wind_speed"]
    # ERA5 accumulations are metres of water equivalent; everything else in this
    # pipeline is mm/day, and a predictor silently 1000x off is worse than one
    # that is missing.
    for v, out_name in (("tp_sum", "era5_tp"), ("sf_sum", "era5_sf")):
        if v in ds.data_vars:
            ds[out_name] = ds[v] * 1000.0
            ds = ds.drop_vars(v)
    if {"era5_tp", "era5_sf"} <= set(ds.data_vars):
        # Share of the reanalysis total falling as snow: a direct handle on the
        # regime where IMERG's error structure changes.
        ds["era5_snow_frac"] = (ds["era5_sf"] / ds["era5_tp"].where(ds["era5_tp"] > 0.1)).fillna(0.0).clip(0, 1)
    lon = ds["longitude"]
    ds = ds.assign_coords(longitude=((lon + 180) % 360) - 180).sortby("longitude").sortby("latitude")
    ds = ds.interp(latitude=grids.coarse.lat, longitude=grids.coarse.lon, method="linear",
                   kwargs={"fill_value": "extrapolate"})
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.reindex(time=time_axis)
    for v in ds.data_vars:
        ds[v] = ds[v].astype("float32")
    return ds


def build_era5_features(cfg: dict, grids: GridPair, variables=None) -> Path:
    """Fetch, derive and align ERA5 -> ``processed/era5_aligned_10km.nc``."""
    from .utils import date_range_daily

    proc = Path(cfg["paths"]["processed"])
    raw = fetch_era5_daily(grids, cfg["time"]["start"], cfg["time"]["end"], variables,
                           cache_dir=proc / "era5_cache")
    time_axis = date_range_daily(cfg["time"]["start"], cfg["time"]["end"])
    ds = derive_and_regrid(raw, grids, time_axis)
    n_missing = int(ds[list(ds.data_vars)[0]].isnull().all(("lat", "lon")).sum())
    if n_missing:
        LOG.warning("ERA5: %d of %d days missing", n_missing, len(time_axis))
    out = proc / "era5_aligned_10km.nc"
    to_netcdf(ds, out)
    LOG.info("ERA5 aligned -> %s (%d fields: %s)", out, len(ds.data_vars), ", ".join(ds.data_vars))
    return out


ERA5_FEATURES = ["cape_max", "cape", "tcwv", "tcwv_max", "t2m", "wind_speed", "wind_sin", "wind_cos",
                 "moist_flux", "era5_tp", "era5_sf", "era5_snow_frac"]
