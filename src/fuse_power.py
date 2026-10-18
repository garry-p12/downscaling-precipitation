"""Add NASA POWER as a second precipitation estimate on the IMERG grid.

The error budget leaves one lever. 91-92 % of every product's error is in its
coarse field; architecture at that stage helps a little (4.10, 5.1) and the
sharpener cannot reach it at all. What is left is giving the coarse stage
better information, and the cleanest form of that is a second, independent
estimate of the same quantity.

IMERG and POWER qualify. IMERG is a satellite retrieval -- passive microwave
where a sensor passed, infrared infill elsewhere. POWER's PRECTOTCORR is the
MERRA-2 reanalysis, a physics model with observations assimilated, then
bias-corrected against CPC gauges. They are wrong in different ways, which is
the property that makes a combination worth more than either.

This puts POWER on the IMERG coarse grid as an extra conditioning channel. It
does not resample IMERG and does not touch the target, so a model trained on
the result is comparable with every other coarse-stage run.

Two things stated rather than hidden:

* **This adds no resolution.** POWER is 0.5 deg and the grid here is 0.1, so
  the interpolated field is smooth across each 5x5 block of IMERG cells. Its
  value is in the *values*, not the detail -- which is the whole point, since
  the budget says the values are what is wrong.
* **The southern edge is extrapolated.** POWER's cell centres start at
  28.75 N and this grid starts at 28.55, so the first two rows sit outside the
  convex hull of POWER's centres by less than half a POWER cell. They are
  filled by nearest-neighbour rather than left as NaN, and the count is
  reported so the cost is visible.

* **One caveat on independence.** POWER is bias-corrected against CPC gauges
  and AORC, our reference, also assimilates gauge data. The two therefore
  share some information, so a gain here is not quite the same as a gain from
  a fully independent source.

    python -m src.fuse_power --src data_coarse1 \\
        --power /work/.../dsc-power/imerg_aligned_10km.nc --out data_coarse1_fused
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

ERA5 = "era5_aligned_10km.nc"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--var", default="power_precip")
    a = ap.parse_args(argv)

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Everything except the predictor file is carried over untouched.
    for p in src.iterdir():
        if p.name == ERA5:
            continue
        d = out / p.name
        if d.exists() or d.is_symlink():
            d.unlink()
        d.symlink_to(p.resolve() if not p.is_symlink() else p.readlink())

    era = xr.open_dataset(src / ERA5)
    pw = xr.open_dataset(a.power)
    v = "precip" if "precip" in pw else list(pw.data_vars)[0]
    p = pw[v]

    # Align time to the predictor file, then space to its grid.
    p = p.sel(time=era.time, method="nearest")
    before = int(p.sizes["lat"]), int(p.sizes["lon"])
    # Edge-clamp rather than extrapolate. POWER's cell centres start at 28.75 N
    # and this grid starts at 28.55, so the southern rows fall outside the
    # convex hull of the centres by less than half a POWER cell. Interpolating
    # at clamped coordinates gives those rows the nearest edge value, which is
    # what a half-cell extrapolation of a 0.5 deg field would give anyway, and
    # unlike extrapolation it cannot produce values outside POWER's range.
    tgt_lat = np.clip(era.lat.values, float(p.lat.min()), float(p.lat.max()))
    tgt_lon = np.clip(era.lon.values, float(p.lon.min()), float(p.lon.max()))
    clamped = int((tgt_lat != era.lat.values).sum()) * len(tgt_lon) + \
              int((tgt_lon != era.lon.values).sum()) * len(tgt_lat)
    p = p.interp(lat=tgt_lat, lon=tgt_lon, method="linear")
    p = p.assign_coords(lat=era.lat.values, lon=era.lon.values)
    still = int(np.isnan(p.values).any(axis=0).sum())
    print(f"  POWER {before[0]}x{before[1]} -> {p.sizes['lat']}x{p.sizes['lon']}; "
          f"{clamped} cells edge-clamped, {still} empty", flush=True)
    if still:
        raise SystemExit("POWER still does not cover this grid after clamping")

    era[a.var] = p.astype("float32")
    era[a.var].attrs = {"units": "mm/day",
                        "long_name": "NASA POWER PRECTOTCORR on the IMERG grid",
                        "note": ("second, independent precipitation estimate; "
                                 "0.5 deg interpolated to this grid, so it adds "
                                 "values and not resolution")}
    enc = {k: {"zlib": True, "complevel": 4} for k in era.data_vars}
    era.to_netcdf(out / ERA5, encoding=enc)
    print(f"  wrote {out/ERA5} with {len(era.data_vars)} channels "
          f"(was {len(era.data_vars)-1})", flush=True)
    m = float(np.nanmean(era[a.var].values))
    print(f"  POWER mean on this grid {m:.3f} mm/day", flush=True)


if __name__ == "__main__":
    main()
