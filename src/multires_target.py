"""Build a training directory whose *target* sits at a coarser grid.

The RSIR proposal asks for downscaled products at 25, 10, 5 and 1 km. Our
pipeline produces one: it reads a coarse input and predicts the 1 km AORC
field. Producing the others is not a matter of averaging the 1 km output --
that answers "how good is our product when used at 10 km", which
``src.proposal_metrics`` already reports. It is a different question: *train a
model to predict the 10 km field, and see whether the downscaling problem gets
easier as the target coarsens.*

This builds the directories for that. The loader opens fixed filenames, so a
directory is made by block-averaging the target and the fine predictors onto
the requested grid and leaving the coarse input untouched:

    aorc_aligned_1km.nc   target, averaged by k      (the name is kept; the
    nlcd_aligned_1km.nc   fine predictors, averaged   content is coarser)
    imerg_aligned_10km.nc coarse input, unchanged
    era5_aligned_10km.nc  coarse predictors, unchanged

The refinement factor the loader infers then falls from 60 to 60/k. On the
POWER arm (coarse 0.5 deg, fine 1/120 deg, factor 60) the proposal's four
spacings are exactly

    k = 30 -> 0.25 deg, factor 2     the NEX-GDDP grid
    k = 12 -> 0.10 deg, factor 5     the IMERG grid
    k =  6 -> 0.05 deg, factor 10    the MERRA-2 HydroSCS grid
    k =  1 -> 1/120 deg, factor 60   what we already build

Averaging is exact: k must divide both fine dimensions, and the check is
enforced rather than assumed, because a silent truncation here would bias
every score computed downstream.

    python -m src.multires_target --src /work/.../dsc-power --k 30 \
        --out /work/.../dsc-power-f2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

# Averaged onto the new fine grid.
FINE = ("aorc_aligned_1km.nc", "nlcd_aligned_1km.nc")
# Left alone: these are on the coarse grid, which does not move.
COARSE = ("imerg_aligned_10km.nc", "era5_aligned_10km.nc",
          "imerg_climatology_10km.nc", "aorc_aligned_10km.nc")


def coarsen_ds(ds: xr.Dataset, k: int) -> xr.Dataset:
    if k == 1:
        return ds
    ny, nx = ds.sizes["lat"], ds.sizes["lon"]
    if ny % k or nx % k:
        raise SystemExit(
            f"fine grid {ny}x{nx} is not divisible by k={k}; coarsen with "
            f"boundary='exact' would drop cells and bias every later score")
    out = ds.coarsen(lat=k, lon=k, boundary="exact").mean()
    # coarsen() leaves the mean of the cell centres, which is the centre of the
    # new cell -- correct, but make it explicit for anything that reads coords.
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="an existing processed directory")
    ap.add_argument("--k", type=int, required=True, help="block size on the fine grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--link", action="store_true",
                    help="symlink the untouched coarse files instead of copying")
    a = ap.parse_args(argv)

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in FINE:
        p = src / name
        if not p.exists():
            raise SystemExit(f"missing {p}")
        # Chunked over time: 20 years of 1 km over this box is 16.9 GiB and
        # will not fit. dask streams it; the mean check below stays lazy too.
        chunks = {"time": 180} if "time" in xr.open_dataset(p).dims else None
        ds = xr.open_dataset(p, chunks=chunks)
        before = (ds.sizes["lat"], ds.sizes["lon"])
        c = coarsen_ds(ds, a.k)
        enc = {v: {"zlib": True, "complevel": 4} for v in c.data_vars}
        c.to_netcdf(out / name, encoding=enc)
        print(f"  {name:<24} {before[0]}x{before[1]} -> "
              f"{c.sizes['lat']}x{c.sizes['lon']}", flush=True)
        # Conservation check. Over a block whose k*k cells are all finite the
        # coarse value is their mean by construction, so the two domain means
        # agree exactly. Where the reference has NaNs -- its domain edge --
        # a partly valid block still gets full weight in the coarse mean while
        # its missing cells carry none in the fine mean, so the two differ by
        # construction and not because anything is wrong. The test is therefore
        # run over fully valid blocks only, where it is exact.
        if "precip" in ds:
            v = ds["precip"]
            finite = v.notnull()
            frac = finite.coarsen(lat=a.k, lon=a.k, boundary="exact").mean()
            full = (frac > 0.999)
            fine_sum = (v.fillna(0.0)
                        .coarsen(lat=a.k, lon=a.k, boundary="exact").sum())
            coarse = xr.open_dataset(out / name, chunks={"time": 180})["precip"]
            lhs = float(fine_sum.where(full).sum().compute())
            rhs = float((coarse * a.k * a.k).where(full).sum().compute())
            pct_full = float(full.mean().compute())
            if not np.isclose(lhs, rhs, rtol=1e-6):
                raise SystemExit(f"{name}: block sums differ {lhs:.4f} vs {rhs:.4f}")
            print(f"  {'':<24} mass exact over the {pct_full:.1%} of blocks that "
                  f"are fully valid", flush=True)

    for name in COARSE:
        p = src / name
        if not p.exists():
            continue
        dest = out / name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if a.link:
            dest.symlink_to(p.resolve())
        else:
            shutil.copy2(p, dest)
        print(f"  {name:<24} carried over unchanged", flush=True)

    print(f"\nwritten {out}  (loader will infer factor {60 // a.k} on the POWER arm)")


if __name__ == "__main__":
    main()
