"""Assemble an oracle data directory whose coarse input carries 24 hourly frames.

deep/data.py stacks every variable it finds in ``era5_aligned_10km.nc`` as a
coarse conditioning channel, so the 24 hourly fields can be delivered through
that slot and the model, the training loop and the scoring all stay unchanged.
Two directories come out of this, differing in exactly one thing:

    data_oracle_12          daily perfect input    (already built)
    data_oracle_12_hourly   the same, plus hr00..hr23

so the comparison between them prices sub-daily position and nothing else.

The existing ERA5 fields are dropped from the hourly directory on purpose. They
are 0.25 deg environmental context for the *coarse correction*, and the coarse
field here is already perfect -- keeping them would let the model spend
capacity on a problem that no longer exists, and would confound the comparison
with the daily oracle, which has them.

    python -m src.hourly_oracle_data --config config_vista.yaml \
        --src data_oracle_12 --out data_oracle_12_hourly
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import LOG, load_config, setup_logging, to_netcdf

LINK = ("aorc_aligned_1km.nc", "nlcd_aligned_1km.nc", "imerg_climatology_10km.nc",
        "imerg_aligned_10km.nc")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_vista.yaml")
    ap.add_argument("--src", default="data_oracle_12")
    ap.add_argument("--out", default="data_oracle_12_hourly")
    ap.add_argument("--hourly-dir", default=None)
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    src, out = Path(a.src), Path(a.out)
    hdir = Path(a.hourly_dir or Path(cfg["paths"]["processed"]) / "aorc_hourly_coarse")
    out.mkdir(parents=True, exist_ok=True)

    for n in LINK:
        s, d = src / n, out / n
        if not s.exists():
            LOG.warning("%s absent in %s", n, src)
            continue
        if d.exists() or d.is_symlink():
            d.unlink()
        d.symlink_to(s.resolve())
    LOG.info("linked %d files from %s", len(LINK), src)

    files = sorted(hdir.glob("aorc_hourly_10km_*.nc"))
    if not files:
        raise SystemExit(f"no hourly files in {hdir}")
    ds = xr.open_mfdataset(files, combine="by_coords").load()
    LOG.info("hourly coarse: %s", dict(ds.sizes))

    # Align to the days the oracle's coarse field covers, so the stacked
    # channels and the target cannot drift apart by a day.
    base = xr.open_dataset(src / "imerg_aligned_10km.nc")["precip"]
    common = np.intersect1d(base["time"].values, ds["time"].values)
    if len(common) != base.sizes["time"]:
        raise SystemExit(f"hourly covers {len(common)} of {base.sizes['time']} days; "
                         f"fetch the missing years before building this")
    ds = ds.sel(time=common)

    v = ds["precip_hourly"].values.astype(np.float32)        # (time, hour, lat, lon)
    # A missing hour means no accumulation was reported, which is zero rain,
    # not unknown rain -- leaving NaN would poison the standardisation.
    v = np.nan_to_num(v, nan=0.0)
    out_ds = xr.Dataset(
        {f"hr{h:02d}": (("time", "lat", "lon"), v[:, h]) for h in range(24)},
        coords={"time": common, "lat": ds["lat"].values, "lon": ds["lon"].values})
    dest = out / "era5_aligned_10km.nc"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    to_netcdf(out_ds, dest)
    tot = v.sum(axis=1)
    LOG.info("wrote %s: 24 channels, %d days", dest, len(common))
    LOG.info("  hourly sum vs daily coarse: max|diff| %.5f mm",
             float(np.nanmax(np.abs(tot - base.sel(time=common).values))))
    print("done ->", out)


if __name__ == "__main__":
    main()
