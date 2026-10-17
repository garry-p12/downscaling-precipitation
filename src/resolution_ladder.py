"""How much of Austin's rainfall survives each candidate grid spacing?

The proposal this supports asks the city a practical question: at what grid
spacing should NASA POWER and NEX-CMIP6 products be delivered for asset-risk
and power-grid work over central Texas? That question has a part which is
independent of every model, every retrieval and every downscaling method:
how much of the real field a grid of a given spacing can represent at all.

This module measures that part. It takes AORC, the 1 km reference, averages
it onto each candidate grid, puts it back on the 1 km grid by the two
ordinary routes (block replication and bilinear interpolation), and scores
the result against the original. Nothing is trained and no satellite is
involved, so what comes out is a property of the *grid*:

    the irreducible error at spacing s -- the floor that the best
    conceivable product on that grid cannot go below.

That number is the right denominator for a decision. A product at 10 km whose
error is 4.4 mm is not 4.4 mm away from the truth because the retrieval is
bad; roughly 1.3 mm of it is the grid, and no amount of better data or better
AI removes that part. Conversely, if a 25 km grid has a floor larger than the
accuracy an application needs, the application cannot be served at 25 km,
whatever is delivered on it.

The factors below are chosen to land on the spacings the proposal names:

    factor  6  -> 0.050 deg,  5.6 km   MERRA-2 HydroSCS
    factor 12  -> 0.100 deg, 11.1 km   IMERG
    factor 30  -> 0.250 deg, 27.8 km   NEX-GDDP
    factor 60  -> 0.500 deg, 55.6 km   NASA POWER

with 2, 5, 15 and 20 filling the curve in between. Every factor divides both
420 and 360, so each block average is exact and no cells are dropped.

    python -m src.resolution_ladder --config config_vista.yaml \
        --out results/resolution_ladder.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from .deep.metrics import rapsd, spectral_ratio
from .utils import (evaluation_grids, grids_from_config, load_config,
                    setup_logging, subset_box)

FACTORS = (2, 5, 6, 12, 15, 20, 30, 60)
BANDS = ((2, 4), (4, 8), (8, 16), (16, 32), (32, 64))
HEAVY = 30.0          # mm/day, the Austin heavy-rain threshold used throughout
WET = 0.1             # mm/day, the wet/dry threshold AORC's wet fraction uses


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Exact block average onto the coarse grid (time, lat, lon)."""
    n, ny, nx = a.shape
    return np.nanmean(a.reshape(n, ny // f, f, nx // f, f), axis=(2, 4))


def block_replicate(c: np.ndarray, f: int) -> np.ndarray:
    return np.repeat(np.repeat(c, f, axis=1), f, axis=2)


def bilinear_up(c: np.ndarray, f: int) -> np.ndarray:
    """Bilinear upsample, matching how a downscaler's base field is built."""
    out = np.empty((c.shape[0], c.shape[1] * f, c.shape[2] * f), dtype=np.float32)
    for i in range(c.shape[0]):
        out[i] = ndimage.zoom(c[i], f, order=1, mode="nearest", grid_mode=True)
    return out


def detection(p: np.ndarray, o: np.ndarray, thr: float) -> dict:
    hit = np.sum((p >= thr) & (o >= thr), dtype=np.float64)
    miss = np.sum((p < thr) & (o >= thr), dtype=np.float64)
    fa = np.sum((p >= thr) & (o < thr), dtype=np.float64)
    pod = hit / max(hit + miss, 1)
    far = fa / max(hit + fa, 1)
    return {"pod": float(pod), "far": float(far),
            "csi": float(hit / max(hit + miss + fa, 1))}


def spectrum(p: np.ndarray, o: np.ndarray, idx) -> dict:
    """Mean RAPSD ratio and per-band energy, on the scorer's accumulation path."""
    sp = so = None
    n = 0
    for i in idx:
        if np.nanmax(o[i]) <= 0:
            continue
        _, a = rapsd(p[i])
        wl, b = rapsd(o[i])
        sp = a if sp is None else sp + a
        so = b if so is None else so + b
        n += 1
    if not n:
        return {}
    sp, so = sp / n, so / n
    ok = np.isfinite(sp) & np.isfinite(so) & (so > 0)
    out = {"spectral_ratio_sub10km": float(spectral_ratio(wl, sp, so, 10.0))}
    sub = ok & (wl <= 10)
    if sub.any():
        out["spectral_energy_sub10km"] = float(sp[sub].sum() / so[sub].sum())
    out["bands"] = {}
    for lo, hi in BANDS:
        s = ok & (wl >= lo) & (wl < hi)
        if s.any():
            out["bands"][f"{lo}-{hi}km"] = float(sp[s].sum() / so[s].sum())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_vista.yaml")
    ap.add_argument("--out", default="results/resolution_ladder.json")
    ap.add_argument("--factors", default=",".join(str(f) for f in FACTORS))
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    _, box = evaluation_grids(cfg, grids)
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    src = Path(cfg["paths"]["processed"])

    o = xr.open_dataset(src / "aorc_aligned_1km.nc")["precip"].sel(time=sl)
    o = (subset_box(o, box) if box else o)
    O = o.values.astype(np.float32)
    ny, nx = O.shape[1], O.shape[2]
    fine_deg = float(cfg["domain"]["coarse_res_deg"]) / int(cfg["domain"]["fine_factor"])
    print(f"reference {O.shape}, fine cell {fine_deg:.5f} deg", flush=True)

    # A fixed, reasonably wet subset for the spectra: the transform is the
    # expensive part and a dry day's spectrum is numerical noise anyway.
    wet_days = np.argsort(np.nanmean(np.where(np.isfinite(O), O, 0), axis=(1, 2)))[-150:]

    m = np.isfinite(O)
    obs_wet = float(np.mean(O[m] >= WET))
    res = {"_meta": {"days": int(O.shape[0]), "grid": [ny, nx],
                     "fine_res_deg": fine_deg, "heavy_mm": HEAVY,
                     "observed_wet_fraction": obs_wet}}

    for f in [int(x) for x in a.factors.split(",")]:
        if ny % f or nx % f:
            print(f"  factor {f}: does not divide {ny}x{nx}, skipped", flush=True)
            continue
        deg = fine_deg * f
        km = deg * 111.3
        C = block_mean(O, f)
        rec = {"factor": f, "res_deg": deg, "res_km": km,
               "coarse_grid": [ny // f, nx // f]}
        for name, up in (("nearest", block_replicate(C, f)),
                         ("bilinear", bilinear_up(C, f))):
            P = up[:, :ny, :nx].astype(np.float32)
            k = np.isfinite(P) & m
            d = P[k].astype(np.float64) - O[k].astype(np.float64)
            r = {"rmse": float(np.sqrt(np.mean(d ** 2))),
                 "mae": float(np.mean(np.abs(d))),
                 "wet_ratio": float(np.mean(P[k] >= WET) / max(obs_wet, 1e-9))}
            r.update(detection(P[k], O[k], HEAVY))
            r.update(spectrum(P, O, wet_days))
            rec[name] = r
        print(f"  {km:5.1f} km ({deg:.4f} deg, {ny//f:3d}x{nx//f:3d})  "
              f"nearest {rec['nearest']['rmse']:6.3f}  bilinear {rec['bilinear']['rmse']:6.3f}  "
              f"POD {rec['bilinear']['pod']:.3f}  "
              f"energy<10km {rec['bilinear'].get('spectral_energy_sub10km', float('nan')):.3f}",
              flush=True)
        res[f"f{f}"] = rec

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("written", a.out)


if __name__ == "__main__":
    main()
