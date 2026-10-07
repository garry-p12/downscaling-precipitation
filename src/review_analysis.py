"""Event, distribution and season analyses requested at review.

Three questions the aggregate tables cannot answer:

* **Does it work on the storms anyone cares about?** A two-year RMSE is an
  average over 731 days, most of them dry. Reviewers ask about the events, and
  an event is scored on its own.
* **Does it reproduce the intensity distribution?** RMSE is minimised by the
  conditional mean, which is a blurred field with a truncated upper tail. A
  PDF/CDF shows that directly, where RMSE hides it.
* **Does the skill hold through the year?** Austin's rain is convective in
  summer and frontal in winter; the Front Range is orographic and snow-driven
  in the cool season. One number averages two regimes.

Every analysis here runs on the 2019-2020 test period, which is outside the
training record (train ends 2017-12-31, 2018 is the dev year).

    python -m src.review_analysis --config config.yaml --out results/review
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import evaluation_grids, grids_from_config, load_config, setup_logging, subset_box

HEAVY = 30.0
# Intensity bins for the PDF/CDF: log-spaced because daily rainfall spans three
# decades and the question is about the tail, which a linear axis hides.
BINS = np.concatenate([[0.0, 0.1], np.logspace(np.log10(0.5), np.log10(500.0), 40)])
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def load_products(cfg: dict, spec: list[str], box) -> dict[str, xr.DataArray]:
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    out = {}
    for s in spec:
        name, paths = s.split("=", 1)
        files = [p for p in paths.split(",") if p]
        das = []
        for p in files:
            ds = xr.open_dataset(p)
            v = "precipitation" if "precipitation" in ds else list(ds.data_vars)[0]
            das.append(ds[v])
        da = xr.concat(das, dim="time").sortby("time") if len(das) > 1 else das[0]
        da = da.sel(time=sl)
        out[name] = subset_box(da, box) if box else da
    return out


def event_metrics(p: np.ndarray, o: np.ndarray, heavy: float = HEAVY) -> dict:
    """Scores for a single day, on the cells where both are valid.

    ``heavy`` is the detection threshold. 30 mm is the Austin convention and
    does not travel: over the Front Range it is 6.7x rarer, and POD/FAR/CSI
    there describe the threshold rather than the model.
    """
    m = np.isfinite(p) & np.isfinite(o)
    if not m.any():
        return {}
    d = p[m] - o[m]
    pe, oe = p[m] >= heavy, o[m] >= heavy
    hits, miss, fa = int((pe & oe).sum()), int((~pe & oe).sum()), int((pe & ~oe).sum())
    return {
        "rmse": float(np.sqrt((d ** 2).mean())),
        "bias": float(d.mean()),
        "mean_pred": float(p[m].mean()), "mean_obs": float(o[m].mean()),
        "max_pred": float(p[m].max()), "max_obs": float(o[m].max()),
        # The ratio of maxima is the number a flood application actually feels:
        # a model can carry the right total and still miss the peak by half.
        "peak_ratio": float(p[m].max() / max(o[m].max(), 1e-6)),
        "pod": float(hits / max(hits + miss, 1)),
        "far": float(fa / max(hits + fa, 1)),
        "csi": float(hits / max(hits + miss + fa, 1)),
        "n_heavy_obs": int(oe.sum()), "n_heavy_pred": int(pe.sum()),
    }


def pick_events(obs: xr.DataArray, k: int = 3) -> dict[str, list[str]]:
    """Wettest days by area mean and by point maximum.

    Both lists are reported because they are different failure modes: a
    widespread frontal day tests whether the total is right, and a localised
    convective cell tests whether the peak survives downscaling at all.
    """
    v = obs.values.astype(np.float32)
    dates = np.array([str(t)[:10] for t in obs["time"].values])
    dm, dx = np.nanmean(v, axis=(1, 2)), np.nanmax(v, axis=(1, 2))
    areal = dates[np.argsort(dm)[::-1][:k]].tolist()
    local = [d for d in dates[np.argsort(dx)[::-1]] if d not in areal][:k]
    return {"areal": areal, "local": local}


def intensity_distribution(arr: np.ndarray, valid: np.ndarray) -> dict:
    """Histogram, exceedance curve and quantiles of a product's wet intensities."""
    x = arr[valid]
    x = x[np.isfinite(x)]
    hist, _ = np.histogram(x, bins=BINS)
    n = max(x.size, 1)
    # Exceedance: P(X >= b). Read on a log axis this is the tail behaviour that
    # a squared-error fit compresses.
    exceed = [float((x >= b).sum() / n) for b in BINS]
    qs = [0.5, 0.9, 0.99, 0.999, 0.9999]
    return {
        "bins": BINS.tolist(),
        "pdf": (hist / n).tolist(),
        "exceedance": exceed,
        "quantiles": {str(q): float(np.quantile(x, q)) for q in qs},
        "wet_fraction": float((x >= 0.1).mean()),
        "mean": float(x.mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--product", action="append", default=[], metavar="NAME=PATH[,PATH]")
    ap.add_argument("--out", default="results/review")
    ap.add_argument("--n-events", type=int, default=3)
    ap.add_argument("--heavy", type=float, default=HEAVY, help="detection threshold in mm")
    ap.add_argument("--match-exceedance", type=float, default=None, metavar="RATE",
                    help="pick the threshold whose observed exceedance equals RATE")
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    _, box = evaluation_grids(cfg, grids)

    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc")["precip"].sel(time=sl)
    if box:
        obs = subset_box(obs, box)
    prods = load_products(cfg, a.product, box)
    print(f"reference {obs.shape}; products: {', '.join(prods)}", flush=True)

    o = obs.values.astype(np.float32)
    heavy = a.heavy
    if a.match_exceedance is not None:
        fin = o[np.isfinite(o)]
        heavy = float(np.quantile(fin, 1.0 - a.match_exceedance))
        print(f"matched exceedance {a.match_exceedance:.3e} -> threshold {heavy:.2f} mm", flush=True)
    dates = np.array([str(t)[:10] for t in obs["time"].values])
    months = np.array([int(str(t)[5:7]) for t in obs["time"].values])
    valid = np.isfinite(o)

    out = {"period": [dates[0], dates[-1]], "n_days": int(len(dates)),
           "training_ends": cfg["time"]["train_end"], "heavy_mm": heavy,
           "events": pick_events(obs, a.n_events)}

    # ---- per-event scores -------------------------------------------------
    out["event_scores"] = {}
    for kind, days in out["events"].items():
        for d in days:
            i = int(np.where(dates == d)[0][0])
            rec = {"kind": kind, "obs_mean": float(np.nanmean(o[i])),
                   "obs_max": float(np.nanmax(o[i])), "products": {}}
            for name, da in prods.items():
                rec["products"][name] = event_metrics(da.values[i].astype(np.float32), o[i], heavy)
            out["event_scores"][d] = rec
        print(f"  scored {len(days)} {kind} events", flush=True)

    # ---- intensity distributions -----------------------------------------
    out["distribution"] = {"AORC": intensity_distribution(o, valid)}
    for name, da in prods.items():
        out["distribution"][name] = intensity_distribution(da.values.astype(np.float32), valid)
    print(f"  distributions for {len(out['distribution'])} fields", flush=True)

    # ---- seasonal stratification -----------------------------------------
    out["seasonal"] = {}
    for season, mm in SEASONS.items():
        sel = np.isin(months, mm)
        rec = {"n_days": int(sel.sum()), "obs_mean": float(np.nanmean(o[sel])), "products": {}}
        for name, da in prods.items():
            rec["products"][name] = event_metrics(da.values[sel].astype(np.float32), o[sel], heavy)
        out["seasonal"][season] = rec
        print(f"  {season}: {int(sel.sum())} days", flush=True)

    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / "review_analysis.json"
    json.dump(out, open(f, "w"), indent=2)
    print("written", f)


if __name__ == "__main__":
    main()
