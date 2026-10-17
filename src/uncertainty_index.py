"""The colour-coded uncertainty layer the RSIR proposal asks for.

    "we will produce an uncertainty index that converts numeric expectations
     into a single color-coded layer (green = low uncertainty, yellow =
     moderate, red = high)"

What goes into it matters more than the colours. A city layer that shades a
map by how much three training seeds disagreed would be close to useless,
because seed disagreement is the *smallest* of the uncertainties in this
system -- 0.02-0.06 mm/day against a total error near 4.5. Shading by it would
tell a planner that the product is uniformly certain, which is false.

So the index combines the two terms that actually move, per cell:

* **spread** -- the standard deviation across the ensemble members at that
  cell, which is what the model knows it does not know;
* **skill**  -- the cell's own RMSE against the reference over the record,
  which is what the model does not know it does not know, and which is the
  dominant term here.

Each is converted to a percentile over the domain and the index is the worse
of the two, so a cell is green only when the ensemble agrees *and* the product
is accurate there. Cut points are the 50th and 80th percentile of that
combined score, which makes the classes read as "the better half", "the next
30 %", and "the worst fifth" -- relative, domain-specific, and honest about
being so. An absolute scale would need an accuracy target the City has not
set, and inventing one here would be worse than saying it is relative.

    python -m src.uncertainty_index --config config_vista.yaml \
        --member results/specM1/s0/cnn_1km_2019_2020.nc \
        --member results/specM1/s1/cnn_1km_2019_2020.nc \
        --member results/specM1/s2/cnn_1km_2019_2020.nc \
        --out results/uncertainty_index.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import (evaluation_grids, grids_from_config, load_config,
                    setup_logging, subset_box, to_netcdf)

CUTS = (50.0, 80.0)        # percentiles separating green | amber | red


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_vista.yaml")
    ap.add_argument("--member", action="append", required=True,
                    help="one ensemble member; repeat (>= 2 needed for spread)")
    ap.add_argument("--out", default="results/uncertainty_index.nc")
    a = ap.parse_args(argv)
    setup_logging()
    if len(a.member) < 2:
        raise SystemExit("need at least two members: spread is undefined otherwise")

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    _, box = evaluation_grids(cfg, grids)
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    src = Path(cfg["paths"]["processed"])

    o = xr.open_dataset(src / "aorc_aligned_1km.nc")["precip"].sel(time=sl)
    o = subset_box(o, box) if box else o
    O = o.values.astype(np.float32)

    M = []
    for path in a.member:
        ds = xr.open_dataset(path)
        v = "precipitation" if "precipitation" in ds else list(ds.data_vars)[0]
        p = ds[v].sel(time=sl)
        p = (subset_box(p, box) if box else p).values.astype(np.float32)
        if p.shape != O.shape:
            raise SystemExit(f"{path}: shape {p.shape} != reference {O.shape}")
        M.append(p)
    M = np.stack(M)
    print(f"{M.shape[0]} members, {O.shape[0]} days, grid {O.shape[1]}x{O.shape[2]}", flush=True)

    # ddof=1: three seeds are a sample of training runs, not the population.
    spread = np.nanmean(np.nanstd(M, axis=0, ddof=1), axis=0)
    mean = np.nanmean(M, axis=0)
    rmse = np.sqrt(np.nanmean((mean - O) ** 2, axis=0))
    print(f"  per-cell spread  mean {np.nanmean(spread):.4f}  max {np.nanmax(spread):.4f}")
    print(f"  per-cell RMSE    mean {np.nanmean(rmse):.4f}  max {np.nanmax(rmse):.4f}")
    print(f"  spread is {np.nanmean(spread)/np.nanmean(rmse):.1%} of RMSE -- the smaller term")

    def pct(x):
        f = np.isfinite(x)
        r = np.full(x.shape, np.nan, dtype=np.float32)
        r[f] = 100.0 * (np.argsort(np.argsort(x[f])) / max(f.sum() - 1, 1))
        return r

    score = np.fmax(pct(spread), pct(rmse))      # green only if both are good
    index = np.full(score.shape, np.nan, dtype=np.float32)
    index[score <= CUTS[0]] = 0                  # green
    index[(score > CUTS[0]) & (score <= CUTS[1])] = 1   # amber
    index[score > CUTS[1]] = 2                   # red
    for lbl, v in (("green", 0), ("amber", 1), ("red", 2)):
        print(f"  {lbl:<6} {np.nanmean(index == v):6.1%} of cells")

    ds = xr.Dataset(
        {"uncertainty_index": (("lat", "lon"), index),
         "ensemble_spread": (("lat", "lon"), spread.astype(np.float32)),
         "rmse": (("lat", "lon"), rmse.astype(np.float32)),
         "score_percentile": (("lat", "lon"), score.astype(np.float32))},
        coords={"lat": o.lat.values, "lon": o.lon.values})
    ds["uncertainty_index"].attrs = {
        "flag_values": "0 1 2", "flag_meanings": "green amber red",
        "long_name": "combined uncertainty class",
        "note": ("worse of two within-domain percentiles: ensemble spread and "
                 "per-cell RMSE. Relative to this domain and period, not an "
                 f"absolute accuracy standard. Cuts at {CUTS[0]}/{CUTS[1]}th."),
        "members": len(a.member)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    to_netcdf(ds, Path(a.out))
    print("written", a.out)


if __name__ == "__main__":
    main()
