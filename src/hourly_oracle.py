"""Hourly coarse AORC: does sub-daily position tell the sharpener where rain fell?

The sharpening step reaches 0.723 mm from a perfect *daily* 10 km field, while
plain bilinear reaches 1.065. The network therefore does learn placement -- but
a daily total is the sum of a storm that moved, and the single number per block
cannot say where within the block it rained hardest or in which direction the
swath ran.

This builds the probe. AORC is hourly, so coarsening it to 10 km *hourly* gives
a perfect-input field that still carries the within-day geometry: 24 frames per
day at 10 km. Feeding those alongside the daily total and predicting the same
1 km daily target isolates one question -- is sub-daily position the missing
placement information?

Only the coarse field is kept. The fine hourly data is 10 GB for two years and
is never needed: the target stays the 1 km daily total we already have.

    python -m src.hourly_oracle --config config_vista.yaml --years 2015-2020
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data_pipeline import open_aorc_year, snap_to_grid
from .utils import LOG, coarsen_mean, grids_from_config, load_config, setup_logging, to_netcdf


def hourly_coarse_year(cfg: dict, grids, year: int, out: Path) -> Path:
    """One year of AORC coarsened to the 10 km grid, hourly, as (time, hour, lat, lon)."""
    acfg = cfg["data"]["aorc"]
    var = acfg.get("variable", "APCP_surface")
    fine = grids.fine
    lon_min, lat_min, lon_max, lat_max = fine.edges
    pad = fine.res

    ds = open_aorc_year(acfg["bucket"], year)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    da = ds[var].sel({lat_name: slice(lat_min - pad, lat_max + pad),
                      lon_name: slice(lon_min - pad, lon_max + pad)})

    months = []
    for per in pd.period_range(f"{year}-01", f"{year}-12", freq="M"):
        sub = da.sel(time=slice(per.start_time, per.end_time)).astype("float32")
        if sub.sizes["time"] == 0:
            continue
        sub = sub.rename({lat_name: "lat", lon_name: "lon"})
        sub = snap_to_grid(sub, fine).transpose("time", "lat", "lon")
        # Coarsen before anything else: the fine hourly array is what makes this
        # expensive, and nothing downstream needs it.
        c = coarsen_mean(sub, grids).compute().astype("float32")
        months.append(c)
        LOG.info("  %s: %d hours -> %s", per, sub.sizes["time"], tuple(c.shape))
    if not months:
        raise SystemExit(f"no AORC hours for {year}")
    hourly = xr.concat(months, "time").sortby("time")

    # Reshape to one row per UTC day with 24 hour-slots. Hour h of day D is the
    # accumulation over (h-1, h], matching the daily convention used elsewhere.
    t = pd.to_datetime(hourly["time"].values)
    days = pd.DatetimeIndex(sorted(set(t.normalize())))
    ny, nx = hourly.sizes["lat"], hourly.sizes["lon"]
    arr = np.full((len(days), 24, ny, nx), np.nan, np.float32)
    di = {d: i for i, d in enumerate(days)}
    v = hourly.values
    for k, ts in enumerate(t):
        arr[di[ts.normalize()], ts.hour] = v[k]
    out_ds = xr.Dataset(
        {"precip_hourly": (("time", "hour", "lat", "lon"), arr)},
        coords={"time": days, "hour": np.arange(24),
                "lat": hourly["lat"].values, "lon": hourly["lon"].values})
    out_ds["precip_hourly"].attrs = {"units": "mm/hour", "source": "AORC, coarsened to the model grid"}
    to_netcdf(out_ds, out)
    frac = float(np.isfinite(arr).mean())
    LOG.info("%d -> %s  (%d days, %.3f of hour-slots filled)", year, out, len(days), frac)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_vista.yaml")
    ap.add_argument("--years", required=True, help="e.g. 2015-2020")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    dest = Path(a.out_dir or Path(cfg["paths"]["processed"]) / "aorc_hourly_coarse")
    dest.mkdir(parents=True, exist_ok=True)
    y0, y1 = (int(x) for x in a.years.split("-"))
    for y in range(y0, y1 + 1):
        f = dest / f"aorc_hourly_10km_{y}.nc"
        if f.exists():
            LOG.info("%d cached", y)
            continue
        hourly_coarse_year(cfg, grids, y, f)
    print("done ->", dest)


if __name__ == "__main__":
    main()
