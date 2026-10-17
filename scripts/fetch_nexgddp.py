"""Subset NEX-GDDP-CMIP6 daily precipitation to the Austin box.

Stage two of the RSIR proposal's Activity 2 needs a mapping from CMIP6 /
NEX-GDDP historical simulations to POWER-like fields, validated where both
exist and then applied to projections. This fetches the CMIP6 side.

NEX-GDDP-CMIP6 lives in a public S3 bucket as one global daily file per model,
scenario and year, each a few hundred MB at 0.25 deg. We need a 6 x 7 degree
box, so the whole file is never wanted. s3fs + h5netcdf read it lazily over
byte ranges and only the requested slice is transferred, which turns a
multi-terabyte archive into a few MB per model-year.

Run on a login node; it is network-bound, not compute-bound, and must not run
on a laptop.

    python scripts/fetch_nexgddp.py --out /work/.../nexgddp \
        --models ACCESS-CM2,MPI-ESM1-2-HR --scenario historical \
        --start 2000 --end 2014
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

BUCKET = "nex-gddp-cmip6/NEX-GDDP-CMIP6"
# The variant label differs by model; these are the ones NEX publishes.
VARIANT = {"ACCESS-CM2": "r1i1p1f1", "MPI-ESM1-2-HR": "r1i1p1f1",
           "MRI-ESM2-0": "r1i1p1f1", "NorESM2-MM": "r1i1p1f1",
           "GFDL-ESM4": "r1i1p1f1", "EC-Earth3": "r1i1p1f1",
           "CNRM-ESM2-1": "r1i1p1f2", "UKESM1-0-LL": "r1i1p1f2"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="ACCESS-CM2,MPI-ESM1-2-HR")
    ap.add_argument("--scenario", default="historical")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2014)
    ap.add_argument("--var", default="pr")
    # The POWER domain, with a degree of margin so interpolation has support.
    ap.add_argument("--bbox", default="-101,28.5,-95,35.5", help="W,S,E,N")
    a = ap.parse_args()

    import s3fs
    import xarray as xr

    W, S, E, N = [float(x) for x in a.bbox.split(",")]
    fs = s3fs.S3FileSystem(anon=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ok = fail = skip = 0

    for model in a.models.split(","):
        var = VARIANT.get(model, "r1i1p1f1")
        for year in range(a.start, a.end + 1):
            dest = out / f"{a.var}_{model}_{a.scenario}_{year}.nc"
            if dest.exists():
                skip += 1
                continue
            # NEX publishes a few model-years with a _v1.1 suffix; try both.
            stems = [f"{a.var}_day_{model}_{a.scenario}_{var}_gn_{year}",
                     f"{a.var}_day_{model}_{a.scenario}_{var}_gn_{year}_v1.1",
                     f"{a.var}_day_{model}_{a.scenario}_{var}_gr_{year}"]
            got = False
            for stem in stems:
                key = f"{BUCKET}/{model}/{a.scenario}/{var}/{a.var}/{stem}.nc"
                if not fs.exists(key):
                    continue
                try:
                    with fs.open(key, "rb") as fh:
                        ds = xr.open_dataset(fh, engine="h5netcdf")
                        # NEX longitudes run 0..360.
                        lon = ds["lon"]
                        sel = ds.sel(lat=slice(S, N),
                                     lon=slice(W % 360, E % 360))
                        if sel.sizes.get("lon", 0) == 0:
                            sel = ds.sel(lat=slice(S, N), lon=slice(W, E))
                        sel = sel.load()
                    sel.to_netcdf(dest)
                    mb = dest.stat().st_size / 1e6
                    print(f"  ok   {model:<16} {a.scenario:<11} {year}  "
                          f"{dict(sel.sizes)}  {mb:.1f} MB", flush=True)
                    ok += 1
                    got = True
                    break
                except Exception as exc:
                    print(f"  FAIL {model} {year} {stem}: {exc}", flush=True)
                    traceback.print_exc(limit=1)
                    break
            if not got:
                fail += 1
                print(f"  MISS {model:<16} {a.scenario:<11} {year}  no key found",
                      flush=True)
    print(f"\n{ok} written, {skip} already present, {fail} missing", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
