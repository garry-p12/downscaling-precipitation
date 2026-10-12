"""Fix the wet/dry field by concentrating the same water into fewer cells.

MEASURED AND REJECTED. On Austin it does what it promises to the wet fraction
-- 0.359 to 0.279 against an observed 0.280, mass preserved to seven digits --
and destroys the field doing it. Redistributing a day's water into fewer cells
multiplies the survivors by total/held, and when few cells clear the cut that
factor is large: texture went from 2.6 to 10.9 on one arm, 4.9 to 219.7 on
another, and 0.019 to 532 on XGBoost, while RMSE rose by 0.2 to 4.0 mm. POD
and CSI did not move at all, so it did not even buy detection.

The synthetic test here passed because it checked wet fraction, mass, a tail
quantile and NaN handling, and never checked the spectrum -- the one thing the
operation most obviously threatens. Kept as a documented negative; moving water
after the fact is the wrong mechanism, and occurrence has to be fixed during
training, where the model can learn not to put the water there at all.

Two of the three distributional failures are the same failure seen twice. The
products rain on 57-60 % of cells where reality rains on 28 %, and they reach
only 66-83 % of the heaviest values. Both follow from hedging: a per-cell loss
is minimised by spreading a little water everywhere rather than committing a
lot somewhere, which inflates the wet area and truncates the tail at once.

So the correction is one operation, not two. Decide which cells are dry, set
them to zero, and give the water they were holding to the cells that remain.
Mass is preserved exactly, the wet fraction becomes correct by construction,
and the tail lengthens as a consequence rather than as a second fix.

The decision needs a probability of rain per cell, which the Bernoulli-gamma
head supplies directly as ``exceedance_prob``. For a deterministic product
the field's own value is used as a ranking instead, which is weaker -- it
cannot distinguish "a little rain here" from "probably no rain here" -- and is
offered so the two can be compared.

The cut is calibrated on the dev year and applied unchanged to test, so the
wet fraction is not fitted on the data it is scored against.

    python -m src.occurrence --config config_vista.yaml \
        --pred results/bg/au_s0/cnn_bg_1km_2019_2020.nc --out results/occ/bg_s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import (LOG, evaluation_grids, grids_from_config, load_config,
                    setup_logging, subset_box, to_netcdf)

WET = 0.1          # mm/day; the threshold AORC's own wet fraction is defined on


def calibrate_cut(rank: np.ndarray, target_wet_fraction: float) -> float:
    """The ranking value whose exceedance rate equals the observed wet rate."""
    v = rank[np.isfinite(rank)]
    if v.size == 0 or not (0 < target_wet_fraction < 1):
        return float("inf")
    return float(np.quantile(v, 1.0 - target_wet_fraction))


def concentrate(pred: np.ndarray, rank: np.ndarray, cut: float) -> np.ndarray:
    """Zero the cells below ``cut`` and redistribute their water, per day.

    Redistribution is proportional to what each surviving cell already held, so
    the operation changes *where* the water is, not the relative shape of what
    remains. Done per day: a day's total is a physical quantity and must not be
    moved into another day.
    """
    out = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        p, r = pred[i], rank[i]
        ok = np.isfinite(p)
        keep = ok & np.isfinite(r) & (r >= cut)
        tot = float(np.nansum(np.where(ok, p, 0.0)))
        held = float(np.nansum(np.where(keep, p, 0.0)))
        if held <= 0 or tot <= 0:
            out[i] = np.where(ok, p, np.nan)
            continue
        out[i] = np.where(keep, p * (tot / held), 0.0)
        out[i][~ok] = np.nan
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_vista.yaml")
    ap.add_argument("--pred", required=True, help="test-period prediction file")
    ap.add_argument("--dev-pred", default=None,
                    help="dev-year predictions; the cut is calibrated here. Without "
                         "it the cut comes from the test field's own ranking, which "
                         "leaks and is only for a quick look.")
    ap.add_argument("--prob-var", default="exceedance_prob",
                    help="variable to rank cells by; falls back to the value itself")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    _, box = evaluation_grids(cfg, grids)
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])

    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc")["precip"].sel(time=sl)
    if box:
        obs = subset_box(obs, box)
    o = obs.values.astype(np.float32)
    target = float(np.nanmean(o[np.isfinite(o)] >= WET))
    LOG.info("observed wet fraction on test: %.4f", target)

    ds = xr.open_dataset(a.pred)
    pv = "precipitation" if "precipitation" in ds else list(ds.data_vars)[0]
    pred = ds[pv].sel(time=sl)
    pred = (subset_box(pred, box) if box else pred)
    P = pred.values.astype(np.float32)

    has_prob = a.prob_var in ds
    R = (subset_box(ds[a.prob_var].sel(time=sl), box) if box else ds[a.prob_var].sel(time=sl)
         ).values.astype(np.float32) if has_prob else P
    LOG.info("ranking cells by %s", a.prob_var if has_prob else "the predicted value")

    # Calibrate on the dev year when it is available, so the cut never sees the
    # test field it is applied to.
    if a.dev_pred:
        dsd = xr.open_dataset(a.dev_pred)
        dv = a.prob_var if (has_prob and a.prob_var in dsd) else (
            "precipitation" if "precipitation" in dsd else list(dsd.data_vars)[0])
        rd = dsd[dv]
        rd = (subset_box(rd, box) if box else rd).values.astype(np.float32)
        # The dev year's own observed wet fraction is the right target there.
        od = xr.open_dataset(proc / "aorc_aligned_1km.nc")["precip"].sel(
            time=slice(*[str(t)[:10] for t in (dsd.time.values[0], dsd.time.values[-1])]))
        od = (subset_box(od, box) if box else od).values.astype(np.float32)
        tgt_dev = float(np.nanmean(od[np.isfinite(od)] >= WET))
        cut = calibrate_cut(rd, tgt_dev)
        LOG.info("cut %.5f calibrated on dev (target wet fraction %.4f)", cut, tgt_dev)
    else:
        cut = calibrate_cut(R, target)
        LOG.warning("cut %.5f taken from the test field itself -- leaks, diagnostic only", cut)

    out = concentrate(P, R, cut)
    before = float(np.nanmean(P[np.isfinite(P)] >= WET))
    after = float(np.nanmean(out[np.isfinite(out)] >= WET))
    mb = float(np.nansum(out) / max(np.nansum(np.where(np.isfinite(P), P, 0.0)), 1e-9))
    LOG.info("wet fraction %.4f -> %.4f (observed %.4f); mass ratio %.6f", before, after, target, mb)

    dest = Path(a.out); dest.mkdir(parents=True, exist_ok=True)
    y0, y1 = str(pred.time.values[0])[:4], str(pred.time.values[-1])[:4]
    f = dest / f"occ_1km_{y0}_{y1}.nc"
    to_netcdf(xr.Dataset({"precipitation": (pred.dims, out)}, coords=pred.coords), f)
    print("written", f)


if __name__ == "__main__":
    main()
