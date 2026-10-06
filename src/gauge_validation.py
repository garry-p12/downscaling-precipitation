"""Independent validation of the downscaled products against rain gauges.

    python -m src.gauge_validation

Every other number in this project is scored against AORC, which is a gridded
*analysis*, not truth - and one with visible radar artifacts (see
`results/figures/24_fields_three_days.png`). Gauges are the only independent
check available, so this answers one question: does the product ranking survive
against instruments, or have the models partly learned AORC's own errors?

Three caveats are built into the output rather than left to the reader:

* **Partial circularity.** AORC assimilates gauge data, so many of these
  stations are not fully independent of the reference.
* **Point vs area.** A gauge orifice is ~20 cm; a grid cell is ~0.86 km². For
  convective rain the two can differ by a factor of two with both correct. Every
  product therefore scores far worse here than against AORC; only the *relative*
  ranking is interpretable.
* **Observation time.** GHCN-D daily totals end at a local observation hour
  (commonly 07-08 local), not at 00 UTC. The script tests day offsets of -1, 0
  and +1 and reports the best alignment per station rather than assuming one.
"""
from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils import LOG, grids_from_config, load_config, open_downscaled, save_json, setup_logging, upsample_bilinear

INVENTORY = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
STATION_CSV = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{sid}.csv.gz"


def stations_in_domain(grids, y0: int, y1: int) -> list[tuple[str, float, float]]:
    lon_min, lat_min, lon_max, lat_max = grids.fine.edges
    txt = urllib.request.urlopen(INVENTORY, timeout=180).read().decode()
    out = []
    for ln in txt.splitlines():
        if ln[31:35].strip() != "PRCP":
            continue
        try:
            la, lo, fy, ly = float(ln[12:20]), float(ln[21:30]), int(ln[36:40]), int(ln[41:45])
        except ValueError:
            continue
        if lon_min <= lo <= lon_max and lat_min <= la <= lat_max and fy <= y0 and ly >= y1:
            out.append((ln[0:11], la, lo))
    return out


def fetch_station(sid: str, y0: int, y1: int, cache: Path) -> pd.DataFrame | None:
    """Daily PRCP in mm for one station, QC-failed values dropped."""
    f = cache / f"{sid}.csv"
    if f.exists():
        raw = f.read_text()
    else:
        try:
            raw = gzip.decompress(urllib.request.urlopen(STATION_CSV.format(sid=sid), timeout=120).read()).decode()
        except Exception:
            return None
        f.write_text(raw)
    recs = []
    for ln in raw.splitlines():
        p = ln.split(",")
        if len(p) < 7 or p[2] != "PRCP":
            continue
        d = p[1]
        if not (str(y0) <= d[:4] <= str(y1)):
            continue
        if p[5].strip():          # non-blank quality flag = failed QC
            continue
        try:
            recs.append((pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}"), int(p[3]) / 10.0,
                         p[7].strip() if len(p) > 7 else ""))
        except ValueError:
            continue
    if not recs:
        return None
    df = pd.DataFrame(recs, columns=["time", "mm", "obstime"]).drop_duplicates("time")
    return df.set_index("time").sort_index()


def load_products(cfg, grids, tslice) -> dict[str, xr.DataArray]:
    r = Path(cfg["paths"]["results"])
    d = next((r / n for n in ("v3", "v2", "final") if (r / n).exists() and list((r / n).glob("*_1km_*.nc"))), r / "v3")
    out = {"xgboost": open_downscaled(r, chunks={"time": 64}).sel(time=tslice)}
    for n in ("cnn", "swin", "diffusion", "stacked"):
        f = sorted(d.glob(f"{n}_1km_*.nc"))
        if f:
            key = "precipitation"
            out[n] = xr.open_dataset(f[-1], chunks={"time": 64})[key].sel(time=tslice)
    im = xr.open_dataset(Path(cfg["paths"]["processed"]) / "imerg_aligned_10km.nc")["precip"].sel(time=tslice).load()
    out["bilinear"] = upsample_bilinear(im, grids)
    out["aorc"] = xr.open_dataset(Path(cfg["paths"]["processed"]) / "aorc_aligned_1km.nc",
                                  chunks={"time": 64})["precip"].sel(time=tslice)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--min-days", type=int, default=500, help="minimum valid gauge days to keep a station")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-stations", type=int, default=0)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    grids = grids_from_config(cfg)
    tcfg = cfg["time"]
    y0, y1 = int(tcfg["val_start"][:4]), int(tcfg["val_end"][:4])
    tslice = slice(tcfg["val_start"], tcfg["val_end"])

    cache = Path(cfg["paths"]["processed"]) / "ghcnd_cache"
    cache.mkdir(parents=True, exist_ok=True)
    st = stations_in_domain(grids, y0, y1)
    if args.max_stations:
        st = st[: args.max_stations]
    LOG.info("%d candidate PRCP stations in domain", len(st))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        frames = list(ex.map(lambda s: fetch_station(s[0], y0, y1, cache), st))
    kept = [(s, df) for s, df in zip(st, frames) if df is not None and len(df) >= args.min_days]
    LOG.info("%d stations with >= %d valid days", len(kept), args.min_days)
    if not kept:
        raise SystemExit("no usable stations")

    products = load_products(cfg, grids, tslice)
    times = pd.DatetimeIndex(products["aorc"]["time"].values)
    lat, lon = grids.fine.lat, grids.fine.lon

    # gather each product's series at every station cell in one pass
    cells = {}
    for (sid, la, lo), df in kept:
        cells[sid] = (int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin()))
    series: dict[str, dict[str, np.ndarray]] = {}
    for name, da in products.items():
        ii = np.array([cells[sid][0] for (sid, _, _), _ in kept])
        jj = np.array([cells[sid][1] for (sid, _, _), _ in kept])
        vals = da.isel(lat=xr.DataArray(ii, dims="st"), lon=xr.DataArray(jj, dims="st")).compute().values
        series[name] = {sid: vals[:, k] for k, ((sid, _, _), _) in enumerate(kept)}
        LOG.info("extracted %s at %d stations", name, len(kept))

    # observation-time offsets: GHCN day D ends at ~07-08 local, so it mostly
    # covers the previous UTC day. Test -1/0/+1 and keep the best per station.
    results = {n: defaultdict(float) for n in products}
    per_station = []
    obs_hours = []
    for (sid, la, lo), df in kept:
        g = df["mm"].reindex(times).values
        oh = df["obstime"].mode()
        obs_hours.append(oh.iloc[0] if len(oh) else "")
        best_shift, best_r = 0, -np.inf
        ref = series["aorc"][sid]
        for sh in (-1, 0, 1):
            gg = np.roll(g, sh).astype(float)
            ok = np.isfinite(gg) & np.isfinite(ref)
            if ok.sum() < 100:
                continue
            r = np.corrcoef(gg[ok], ref[ok])[0, 1]
            if r > best_r:
                best_shift, best_r = sh, r
        gg = np.roll(g, best_shift).astype(float)
        row = {"station": sid, "lat": la, "lon": lon[cells[sid][1]], "shift": best_shift,
               "n": int(np.isfinite(gg).sum())}
        for n in products:
            v = series[n][sid]
            ok = np.isfinite(gg) & np.isfinite(v)
            if ok.sum() < 100:
                continue
            d = v[ok] - gg[ok]
            results[n]["se"] += float((d**2).sum())
            results[n]["ae"] += float(np.abs(d).sum())
            results[n]["bias"] += float(d.sum())
            results[n]["n"] += float(ok.sum())
            row[n] = float(np.sqrt((d**2).mean()))
        per_station.append(row)

    summary = {}
    for n, acc in results.items():
        if acc["n"] == 0:
            continue
        summary[n] = {"rmse": float(np.sqrt(acc["se"] / acc["n"])), "mae": float(acc["ae"] / acc["n"]),
                      "bias": float(acc["bias"] / acc["n"]), "n_obs": int(acc["n"])}
    shifts = pd.Series([r["shift"] for r in per_station]).value_counts().to_dict()
    out = {"period": [tcfg["val_start"], tcfg["val_end"]], "n_stations": len(kept),
           "min_days": args.min_days, "best_shift_counts": {str(k): int(v) for k, v in shifts.items()},
           "obs_time_mode": pd.Series(obs_hours).mode().iloc[0] if obs_hours else "",
           "summary": summary,
           "caveats": ["AORC assimilates gauge data, so this is only partly independent",
                       "gauge is a point, grid cell is ~0.86 km2 - representativeness error is large",
                       "GHCN daily totals end at a local observation hour; per-station day offset applied"]}
    rdir = Path(cfg["paths"]["results"])
    save_json(out, rdir / "gauge_validation.json")
    pd.DataFrame(per_station).to_csv(rdir / "gauge_validation_per_station.csv", index=False)

    print(f"\n=== Gauge validation: {len(kept)} stations, {out['period'][0]}..{out['period'][1]}")
    print(f"most common observation hour: {out['obs_time_mode']}  |  day-shift used: {shifts}")
    print(f"{'product':>10} {'RMSE':>8} {'MAE':>8} {'bias':>8}")
    for n, v in sorted(summary.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{n:>10} {v['rmse']:8.3f} {v['mae']:8.3f} {v['bias']:+8.3f}")
    return out


if __name__ == "__main__":
    main()
