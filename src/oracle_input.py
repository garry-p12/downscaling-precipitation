"""Build a perfect-input data directory: the reference, block-averaged.

Every number in this study is a sum of two things we cannot separate from the
scores alone -- how wrong the satellite is at 10 km, and how much 1 km detail is
unrecoverable from *any* 10 km field. A model that stops improving might be out
of capacity, or it might have already extracted everything its input contains.

This builds the control for that question. It replaces the coarse precipitation
field with a block-average of AORC itself, leaving every other channel alone, so
the input is correct by construction and the only error left is the sub-grid
variance that averaging destroyed. Training on it bounds the ceiling: whatever
RMSE the oracle reaches is the floor no real input can beat, and the gap between
the oracle and the satellite run is the part of our error that is the
satellite's fault rather than the method's.

The output is a drop-in ``--data-dir``: deep/data.py reads
``imerg_aligned_10km.nc`` and infers the refinement factor from the grids, so
nothing downstream needs to know the input changed.

    python -m src.oracle_input --config config.yaml --out data_oracle_12
    python -m src.oracle_input --config config.yaml --out data_oracle_60 --factor 60
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import coarsen_mean, grids_from_config, load_config, make_grids, setup_logging

# Everything the loader opens besides the coarse precipitation field. These are
# copied rather than rebuilt: the oracle changes one input, not the feature set.
PASSTHROUGH = ("aorc_aligned_1km.nc", "nlcd_aligned_1km.nc", "era5_aligned_10km.nc",
               "dem_slope_aspect.nc", "imerg_climatology_10km.nc")


def build(cfg: dict, out: Path, factor: int | None, link: bool) -> None:
    src = Path(cfg["paths"]["processed"])
    grids = grids_from_config(cfg)
    dom = cfg["domain"]
    base_factor = int(dom.get("fine_factor", grids.factor))
    if factor is not None and factor != grids.factor:
        # A coarser oracle (0.5 deg, to bound the POWER arm) needs its own grid
        # pair. Scale the config's own coarse resolution rather than measuring
        # the fine grid: its spacing is stored truncated (0.008333), so 60 cells
        # of it come to 0.49998 and make_grids rejects the bbox as not a
        # multiple of it.
        grids = make_grids(dom["bbox"],
                           coarse_res=float(dom["coarse_res_deg"]) * factor / base_factor,
                           factor=factor)
    out.mkdir(parents=True, exist_ok=True)

    aorc = xr.open_dataset(src / "aorc_aligned_1km.nc")["precip"]
    ny, nx = aorc.sizes["lat"], aorc.sizes["lon"]
    if ny % grids.factor or nx % grids.factor:
        raise SystemExit(
            f"fine grid {ny}x{nx} is not divisible by factor {grids.factor}; "
            f"coarsen_mean uses boundary='exact' and would silently drop cells")

    print(f"coarsening {ny}x{nx} -> {ny // grids.factor}x{nx // grids.factor} "
          f"(factor {grids.factor})", flush=True)
    coarse = coarsen_mean(aorc, grids)

    # Keep the variable name and dtype the satellite file uses: the point of the
    # oracle is that nothing downstream can tell the difference.
    ds = coarse.astype("float32").rename("precip").to_dataset()
    ds["precip"].attrs = {"units": "mm/day", "long_name": "AORC block mean (perfect input oracle)",
                          "source": "aorc_aligned_1km.nc", "factor": int(grids.factor)}
    ds.attrs["note"] = ("Perfect-input oracle: this is the reference averaged to the coarse "
                        "grid, NOT a satellite product. Scores against it bound the ceiling.")
    dest = out / "imerg_aligned_10km.nc"
    ds.to_netcdf(dest)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)

    finite = np.isfinite(coarse.values)
    print(f"  coarse cells finite: {finite.mean():.3f}; "
          f"mean {np.nanmean(coarse.values):.3f} mm/day", flush=True)

    # Files already on the coarse grid cannot simply be linked when the factor
    # changes: at factor 60 the coarse grid is 7x6 while these were built for
    # 35x30, and the loader stacks them with the coarse precipitation field.
    # Linking them produced "all input arrays must have the same shape" only
    # once training started, well after the oracle itself looked fine.
    regrid = grids.factor != base_factor
    step = grids.factor // base_factor if regrid else 1

    for name in PASSTHROUGH:
        s = src / name
        if not s.exists():
            print(f"  skip {name} (absent)", flush=True)
            continue
        d = out / name
        if d.exists() or d.is_symlink():
            d.unlink()
        if regrid and "10km" in name:
            ds = xr.open_dataset(s)
            ny, nx = ds.sizes["lat"], ds.sizes["lon"]
            if ny % step or nx % step:
                raise SystemExit(f"{name}: {ny}x{nx} not divisible by {step}")
            ds = ds.coarsen(lat=step, lon=step, boundary="exact").mean()
            ds = ds.assign_coords(lat=grids.coarse.lat, lon=grids.coarse.lon)
            ds.to_netcdf(d)
            print(f"  regridded {name} {ny}x{nx} -> {ny // step}x{nx // step}", flush=True)
            continue
        if link:
            d.symlink_to(s.resolve())
        else:
            shutil.copy2(s, d)
        print(f"  {'linked' if link else 'copied'} {name}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=None,
                    help="refinement factor; default is the config's (12 for 0.1 deg)")
    ap.add_argument("--copy", action="store_true",
                    help="copy the passthrough files instead of symlinking them")
    a = ap.parse_args(argv)
    setup_logging()
    build(load_config(a.config), Path(a.out), a.factor, link=not a.copy)


if __name__ == "__main__":
    main()
