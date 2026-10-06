"""Where in scale does a downscaler actually reduce the error?

Section 4.7 claims the gain from a coarser input comes from reconstructing the
10-50 km band. That was an inference from reasoning plus a feature-importance
shift. This measures it.

By Parseval, the mean squared error of a field equals the integral of the power
spectrum of the *error* field, so decomposing that spectrum into wavelength
bands partitions MSE by scale exactly. Comparing the partition for a model
against the partition for bilinear interpolation shows which scales the model
actually fixed, rather than which scales we think it fixed.

    python -m src.error_spectrum --config config_vista.yaml --deep-dir results/v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .deep.metrics import rapsd
from .utils import (evaluation_grids, grids_from_config, load_config, open_downscaled,
                    setup_logging, subset_box, upsample_bilinear, upsample_nearest)

# Wavelength bands in km, coarse to fine. The 10 km edge is the native IMERG
# scale and the cut the spectral ratio already uses; 50 km brackets the band
# POWER cannot resolve but IMERG can.
BANDS = [("> 100 km", 100.0, np.inf), ("50-100 km", 50.0, 100.0),
         ("20-50 km", 20.0, 50.0), ("10-20 km", 10.0, 20.0), ("< 10 km", 0.0, 10.0)]


def error_spectrum(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean RAPSD of (pred - obs) over a stack of days."""
    acc, n = None, 0
    for p, o in zip(pred, obs):
        ok = np.isfinite(p) & np.isfinite(o)
        if not ok.any():
            continue
        d = np.where(ok, p - o, 0.0)
        wl, power = rapsd(d)
        acc = power if acc is None else acc + power
        n += 1
    return wl, (acc / max(n, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--deep-dir", default=None)
    ap.add_argument("--days", type=int, default=120, help="wettest N test days")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    rdir = Path(cfg["paths"]["results"])
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])

    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": 32})["precip"].sel(time=sl)
    eval_grids, box = evaluation_grids(cfg, grids)
    if box:
        obs = subset_box(obs, box)

    # Pick the days first. Upsampling the coarse field for the whole test period
    # before subsetting needs several GB on the POWER grid, for fields that are
    # then thrown away.
    dm = obs.mean(dim=("lat", "lon")).compute().values
    pick = np.argsort(dm)[::-1][:a.days]
    times = obs["time"].values[np.sort(pick)]
    o = obs.sel(time=times).values.astype(np.float64)
    print(f"{len(times)} days, grid {o.shape[1]} x {o.shape[2]}", flush=True)

    deep = Path(a.deep_dir) if a.deep_dir else rdir
    coarse = xr.open_dataset(proc / "imerg_aligned_10km.nc")["precip"].sel(time=times).load()
    products = {"nearest": upsample_nearest(coarse, grids),
                "bilinear": upsample_bilinear(coarse, grids)}
    try:
        products["xgboost"] = open_downscaled(rdir, chunks={"time": 32}).sel(time=times)
    except FileNotFoundError:
        print("  no tree product in", rdir)
    for name in ("cnn", "cnn_spectral", "swin", "diffusion"):
        hits = sorted(deep.glob(f"{name}_1km_*.nc"))
        if not hits:
            continue
        ds = xr.open_dataset(hits[-1], chunks={"time": 32})
        products[name] = ds["precipitation"].sel(time=times)
        if "precipitation_member0" in ds:
            products[f"{name}_member"] = ds["precipitation_member0"].sel(time=times)
    if box:
        products = {k: subset_box(v, box) for k, v in products.items()}

    out = {"config": a.config, "n_days": len(times), "bands": [b[0] for b in BANDS],
           "products": {}}
    for name, da in products.items():
        p = np.asarray(da.values, dtype=np.float64)
        wl, power = error_spectrum(p, o)
        ok = np.isfinite(wl) & np.isfinite(power)
        total = float(np.nansum(power[ok]))
        shares, absolute = {}, {}
        for label, lo, hi in BANDS:
            m = ok & (wl > lo) & (wl <= hi)
            band = float(np.nansum(power[m]))
            absolute[label] = band
            shares[label] = band / total if total > 0 else float("nan")
        err = p - o
        valid = np.isfinite(err)
        out["products"][name] = {
            "rmse": float(np.sqrt(np.nanmean(err[valid] ** 2))),
            "error_power_total": total,
            "error_power_by_band": absolute,
            "share_by_band": shares,
        }
        print(f"  {name:18s} rmse {out['products'][name]['rmse']:7.3f}  "
              + "  ".join(f"{k} {v*100:4.1f}%" for k, v in shares.items()), flush=True)

    # where did each model actually beat interpolation?
    if "bilinear" in out["products"]:
        base = out["products"]["bilinear"]["error_power_by_band"]
        for name, rec in out["products"].items():
            if name == "bilinear":
                continue
            rec["power_reduction_vs_bilinear"] = {
                k: base[k] - rec["error_power_by_band"][k] for k in base}
            tot = sum(rec["power_reduction_vs_bilinear"].values())
            rec["share_of_gain_by_band"] = {
                k: (v / tot if tot != 0 else float("nan"))
                for k, v in rec["power_reduction_vs_bilinear"].items()}

    dest = Path(a.out) if a.out else rdir / "error_spectrum.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print("written", dest)


if __name__ == "__main__":
    main()
