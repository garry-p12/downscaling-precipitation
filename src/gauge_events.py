"""Gauge-based error analysis at storm events, and as a time series.

The domain-wide gauge validation answers "is the product unbiased on average".
It cannot answer the question an operational reader asks, which is whether the
product got *this storm* right at *this town*. That needs the comparison done
per event and per station.

Gauges are the only reference here independent of AORC's own construction, so
they are also the only check on whether apparent skill is partly an artefact of
scoring against the analysis the model was trained on. Two caveats carry
through from gauge_validation:

* GHCN-D daily totals end at a local observation hour, not 00 UTC, so a storm
  straddling that hour lands on a different day in the gauge than in the grid.
  Each station is tested at shifts of -1, 0 and +1 days and its best alignment
  is used; a station whose best shift is non-zero is reported, not hidden.
* A gauge is a point and a grid cell is an area, so some disagreement is
  representativeness rather than error. This is why the peak ratio matters more
  than the absolute difference at any one station.

    python -m src.gauge_events --config config.yaml --events 2019-06-05,2020-05-12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .gauge_validation import fetch_station, stations_in_domain
from .utils import evaluation_grids, grids_from_config, load_config, setup_logging, subset_box

SHIFTS = (-1, 0, 1)


def nearest_cell(da: xr.DataArray, lat: float, lon: float):
    """Index of the grid cell containing a station."""
    i = int(np.abs(np.asarray(da["lat"]) - lat).argmin())
    j = int(np.abs(np.asarray(da["lon"]) - lon).argmin())
    return i, j


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--product", action="append", default=[], metavar="NAME=PATH[,PATH]")
    ap.add_argument("--events", default="", help="comma-separated YYYY-MM-DD to score individually")
    ap.add_argument("--min-days", type=int, default=400)
    ap.add_argument("--out", default="results/review/gauge_events.json")
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    t0, t1 = cfg["time"]["val_start"], cfg["time"]["val_end"]
    sl = slice(t0, t1)
    _, box = evaluation_grids(cfg, grids)

    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc")["precip"].sel(time=sl)
    if box:
        obs = subset_box(obs, box)

    prods = {}
    for s in a.product:
        name, paths = s.split("=", 1)
        das = []
        for p in [x for x in paths.split(",") if x]:
            ds = xr.open_dataset(p)
            v = "precipitation" if "precipitation" in ds else list(ds.data_vars)[0]
            das.append(ds[v])
        da = xr.concat(das, dim="time").sortby("time") if len(das) > 1 else das[0]
        da = da.sel(time=sl)
        prods[name] = (subset_box(da, box) if box else da).load()
    obs = obs.load()

    cache = proc / "ghcnd_cache"
    # fetch_station writes straight into this directory, so it has to exist
    # before the first station: on a domain with no cache yet, every
    # download otherwise fails with FileNotFoundError on the write.
    cache.mkdir(parents=True, exist_ok=True)
    y0, y1 = int(t0[:4]), int(t1[:4])
    st = stations_in_domain(grids, y0, y1)
    # stations_in_domain works from the model grid, which can be larger than the
    # box we actually score: the POWER arm covers 840x720 fine cells while its
    # evaluation box is 420x360. A station outside the box still finds a
    # "nearest" cell -- the nearest edge cell -- and contributes a comparison
    # between a gauge and a grid point tens of kilometres away.
    la, lo = np.asarray(obs["lat"]), np.asarray(obs["lon"])
    lat0, lat1 = float(min(la)), float(max(la))
    lon0, lon1 = float(min(lo)), float(max(lo))
    inside = [(sid, y, x) for sid, y, x in st if lat0 <= y <= lat1 and lon0 <= x <= lon1]
    if len(inside) != len(st):
        print(f"dropped {len(st) - len(inside)} stations outside the evaluation box "
              f"[{lon0:.2f},{lat0:.2f},{lon1:.2f},{lat1:.2f}]", flush=True)
    st = inside
    print(f"{len(st)} stations in the scored box; products: {', '.join(prods)}", flush=True)

    times = pd.to_datetime(obs["time"].values)
    tindex = {t: k for k, t in enumerate(times)}
    events = [e for e in a.events.split(",") if e]

    rows, kept, shift_counts = [], 0, {s: 0 for s in SHIFTS}
    for sid, lat, lon in st:
        df = fetch_station(sid, y0, y1, cache)
        if df is None or len(df) < a.min_days:
            continue
        i, j = nearest_cell(obs, lat, lon)
        g_all = obs.values[:, i, j]
        # Choose the observation-hour alignment on AORC, then apply that same
        # shift to every product, so the alignment cannot favour one of them.
        best, best_shift = -2.0, 0
        for s in SHIFTS:
            ser = df["mm"].reindex(times + pd.Timedelta(days=s))
            m = np.isfinite(ser.values) & np.isfinite(g_all)
            if m.sum() < a.min_days:
                continue
            r = float(np.corrcoef(ser.values[m], g_all[m])[0, 1])
            if r > best:
                best, best_shift = r, s
        if best < 0:
            continue
        shift_counts[best_shift] += 1
        kept += 1
        gauge = df["mm"].reindex(times + pd.Timedelta(days=best_shift)).values
        rec = {"sid": sid, "lat": lat, "lon": lon, "shift": best_shift, "r_aorc": best,
               "n": int(np.isfinite(gauge).sum())}
        m = np.isfinite(gauge)
        rec["AORC"] = {"rmse": float(np.sqrt(((obs.values[m, i, j] - gauge[m]) ** 2).mean())),
                       "bias": float((obs.values[m, i, j] - gauge[m]).mean())}
        for name, da in prods.items():
            p = da.values[:, i, j]
            mm = m & np.isfinite(p)
            rec[name] = {"rmse": float(np.sqrt(((p[mm] - gauge[mm]) ** 2).mean())),
                         "bias": float((p[mm] - gauge[mm]).mean())}
        # Per-event values at this station, for the storm panels.
        ev = {}
        for e in events:
            ts = pd.Timestamp(e)
            k = tindex.get(ts)
            if k is None or not np.isfinite(gauge[k]):
                continue
            ev[e] = {"gauge": float(gauge[k]), "AORC": float(obs.values[k, i, j]),
                     **{n: float(d.values[k, i, j]) for n, d in prods.items()}}
        if ev:
            rec["events"] = ev
        rows.append(rec)

    print(f"kept {kept} stations; obs-time shift counts {shift_counts}", flush=True)

    names = ["AORC"] + list(prods)
    summary = {}
    for n in names:
        rm = np.array([r[n]["rmse"] for r in rows if n in r])
        bi = np.array([r[n]["bias"] for r in rows if n in r])
        summary[n] = {"median_rmse": float(np.median(rm)), "mean_rmse": float(rm.mean()),
                      "median_bias": float(np.median(bi)), "n_stations": int(rm.size)}

    # Event aggregates across stations that reported that day.
    ev_summary = {}
    for e in events:
        vals = {n: [] for n in names + ["gauge"]}
        for r in rows:
            d = r.get("events", {}).get(e)
            if d:
                for n in vals:
                    vals[n].append(d[n])
        if not vals["gauge"]:
            continue
        ev_summary[e] = {
            "n_stations": len(vals["gauge"]),
            "gauge_mean": float(np.mean(vals["gauge"])), "gauge_max": float(np.max(vals["gauge"])),
            **{n: {"mean": float(np.mean(vals[n])), "max": float(np.max(vals[n])),
                   "rmse_vs_gauge": float(np.sqrt(np.mean(
                       (np.array(vals[n]) - np.array(vals["gauge"])) ** 2))),
                   "peak_ratio": float(np.max(vals[n]) / max(np.max(vals["gauge"]), 1e-6))}
               for n in names},
        }

    out = {"period": [t0, t1], "n_stations": kept, "shift_counts": shift_counts,
           "summary": summary, "events": ev_summary,
           "caveats": ["GHCN-D day ends at a local observation hour; per-station shift applied",
                       "gauge is a point, grid cell is an area",
                       "GHCN-D is not fully independent of AORC, which assimilates gauges"],
           "stations": rows}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print("written", dest)


if __name__ == "__main__":
    main()
