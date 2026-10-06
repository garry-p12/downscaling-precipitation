"""Is the reference analysis stationary across our train/test split?

AORC v1.1 does not use one precipitation methodology throughout. Fall et al.
(2023) give NLDAS-2 as the primary daily CONUS source to 2001, Stage IV radar
from January 2002, a strong Livneh monthly constraint to 2009 weakening to a
climatological one through 2015, and -- the break that matters here -- *no bias
adjustment at all* from January 2016.

Our training period is almost entirely inside the adjusted era while dev and
test are entirely outside it. If the unadjusted product carries different
fine-scale structure, then the sub-10 km variance a model is trained toward is
not the variance it is scored against, which would bias every texture result in
the paper independently of any model or loss.

This measures the reference against itself, year by year: how much rain, how
much of the map is wet, and what fraction of its own spatial variance sits
below 10 km.

    python -m src.reference_drift --config config_vista.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .deep.metrics import rapsd
from .utils import evaluation_grids, grids_from_config, load_config, setup_logging, subset_box


def texture_share(fields: np.ndarray, cut_km: float = 10.0) -> float:
    """Fraction of the field's own spatial variance below ``cut_km``.

    Normalised by the field's total power, so a wet year and a dry year are
    comparable: this asks about the *shape* of the spectrum, not its size.
    """
    num = den = 0.0
    for f in fields:
        if not np.isfinite(f).any() or np.nanmax(f) <= 0:
            continue
        wl, p = rapsd(f)
        ok = np.isfinite(wl) & np.isfinite(p)
        den += float(np.nansum(p[ok]))
        num += float(np.nansum(p[ok & (wl <= cut_km)]))
    return num / den if den > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2020)
    ap.add_argument("--days", type=int, default=40, help="wettest N days per year")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": 32})["precip"]
    _, box = evaluation_grids(cfg, grids)
    if box:
        obs = subset_box(obs, box)

    rows = []
    print(f"{'year':>5s} {'mean':>7s} {'wet>1mm':>8s} {'p99':>7s} {'<10km share':>12s}")
    for year in range(a.start, a.end + 1):
        y = obs.sel(time=str(year))
        if y.sizes.get("time", 0) == 0:
            continue
        dm = y.mean(dim=("lat", "lon")).compute().values
        pick = np.sort(np.argsort(dm)[::-1][:a.days])
        f = y.isel(time=pick).values.astype(np.float64)
        ok = np.isfinite(f)
        rec = {
            "year": year,
            "n_days_total": int(y.sizes["time"]),
            "mean_mm": float(np.nanmean(y.mean(dim=("lat", "lon")).compute().values)),
            "wet_area_frac": float((f[ok] >= 1.0).mean()),
            "p99_mm": float(np.nanpercentile(f[ok], 99)),
            "sub10km_share": texture_share(f),
        }
        rows.append(rec)
        print(f"{year:5d} {rec['mean_mm']:7.3f} {rec['wet_area_frac']:8.3f} "
              f"{rec['p99_mm']:7.2f} {rec['sub10km_share']:12.5f}", flush=True)

    # the documented methodology breaks
    def avg(key, lo, hi):
        v = [r[key] for r in rows if lo <= r["year"] <= hi]
        return float(np.mean(v)) if v else float("nan")

    eras = {"2000-2001 NLDAS-2, strong constraint": (2000, 2001),
            "2002-2009 StageIV, strong constraint": (2002, 2009),
            "2010-2015 StageIV, climatological": (2010, 2015),
            "2016-2020 StageIV, NO adjustment": (2016, 2020)}
    print("\nera means")
    summary = {}
    for label, (lo, hi) in eras.items():
        summary[label] = {k: avg(k, lo, hi) for k in
                          ("mean_mm", "wet_area_frac", "p99_mm", "sub10km_share")}
        s = summary[label]
        print(f"  {label:42s} mean {s['mean_mm']:6.3f}  wet {s['wet_area_frac']:.3f}  "
              f"p99 {s['p99_mm']:6.2f}  <10km {s['sub10km_share']:.5f}")

    dest = Path(a.out) if a.out else Path(cfg["paths"]["results"]) / "reference_drift.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"years": rows, "eras": summary}, open(dest, "w"), indent=2)
    print("written", dest)


if __name__ == "__main__":
    main()
