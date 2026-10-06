"""Post-hoc fixes for the shrinkage a per-cell loss induces. No retraining.

A field fitted to minimise squared error is the conditional mean, so its
intensity distribution is compressed: §4.1's XGBoost product forecasts rain
above 30 mm only 0.742 times as often as it occurs. Two standard corrections
address that without touching the model.

**Quantile mapping.** Fit an empirical map from the product's own distribution
to the reference's on the dev year, then apply it to the test years. The map is
monotone, so every cell keeps its rank and the spatial pattern is untouched;
only the values change. It restores both the frequency of heavy rain and some
of the missing spectral amplitude, and it is the baseline any trained
intervention has to beat.

**Probability-matched mean** (Ebert 2001). Take the *pattern* from the ensemble
mean, which is well placed but smeared, and the *intensity distribution* from
the pooled members, which is realistic but individually misplaced. Reassigning
the pooled values onto the mean's rank order gives a field with the mean's
placement and a member's histogram.

    python -m src.postprocess quantile-map --config config_vista.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import (evaluation_grids, grids_from_config, load_config, open_downscaled,
                    setup_logging, subset_box)

NQ = 2001          # quantile grid; finer than the data warrants, and cheap


def fit_quantile_map(pred: np.ndarray, obs: np.ndarray, nq: int = NQ):
    """Empirical CDF map from prediction values to reference values."""
    p = pred[np.isfinite(pred)]
    o = obs[np.isfinite(obs)]
    q = np.linspace(0.0, 1.0, nq)
    return np.quantile(p, q), np.quantile(o, q)


def apply_quantile_map(x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Monotone interpolation through the fitted map, extrapolating the tail.

    Values above the largest dev-year prediction are scaled by the slope of the
    final mapped segment rather than clipped, since the test years contain days
    wetter than anything in the single dev year.
    """
    out = np.interp(x, src, dst)
    hi = src[-1]
    if hi > 0:
        tail = x > hi
        if tail.any():
            k = (dst[-1] - dst[-2]) / max(src[-1] - src[-2], 1e-6)
            out[tail] = dst[-1] + (x[tail] - hi) * max(k, 1.0)
    return np.where(np.isfinite(x), out, np.nan)


def probability_matched_mean(members: np.ndarray, mean_field: np.ndarray) -> np.ndarray:
    """Ebert (2001): the ensemble mean's pattern, the pooled members' histogram."""
    out = np.full_like(mean_field, np.nan)
    ok = np.isfinite(mean_field)
    n = int(ok.sum())
    if n == 0:
        return out
    pool = members[:, ok].ravel()
    pool = pool[np.isfinite(pool)]
    # take n values spanning the pooled distribution, then place them by the
    # mean field's rank order
    vals = np.sort(pool)[np.linspace(0, pool.size - 1, n).round().astype(int)]
    order = np.argsort(np.argsort(mean_field[ok]))
    out[ok] = vals[order]
    return out


def _cmd_quantile_map(a):
    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc, rdir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    _, box = evaluation_grids(cfg, grids)
    tcfg = cfg["time"]

    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": 32})["precip"]
    dev = slice(tcfg["dev_start"], tcfg["dev_end"]) if "dev_start" in tcfg else \
        slice("2018-01-01", "2018-12-31")
    test = slice(tcfg["val_start"], tcfg["val_end"])

    src_dir = Path(a.deep_dir) if a.deep_dir else rdir
    if a.product == "xgboost":
        full = open_downscaled(rdir, chunks={"time": 32})
    else:
        hits = sorted(src_dir.glob(f"{a.product}_1km_*.nc"))
        if not hits:
            raise SystemExit(f"no {a.product}_1km_*.nc in {src_dir}")
        full = xr.open_dataset(hits[-1], chunks={"time": 32})["precipitation"]

    dev_src = full
    if a.dev_dir:
        hits = sorted(Path(a.dev_dir).glob(f"{a.product}_1km_*.nc"))
        if not hits:
            raise SystemExit(f"no {a.product}_1km_*.nc in {a.dev_dir}")
        dev_src = xr.open_dataset(hits[-1], chunks={"time": 32})["precipitation"]
    pd_, od_ = dev_src.sel(time=dev), obs.sel(time=dev)
    if box:
        pd_, od_ = subset_box(pd_, box), subset_box(od_, box)
    print(f"fitting on {pd_.sizes['time']} dev days", flush=True)
    src, dst = fit_quantile_map(pd_.values.astype(np.float64), od_.values.astype(np.float64))
    print(f"  map: p50 {src[NQ//2]:.3f} -> {dst[NQ//2]:.3f} | "
          f"p99 {src[int(.99*NQ)]:.2f} -> {dst[int(.99*NQ)]:.2f} | "
          f"max {src[-1]:.1f} -> {dst[-1]:.1f}", flush=True)

    pt = full.sel(time=test)
    out = np.empty(pt.shape, np.float32)
    for i in range(0, pt.sizes["time"], 32):
        blk = pt.isel(time=slice(i, i + 32)).values.astype(np.float64)
        out[i:i + 32] = apply_quantile_map(blk, src, dst).astype(np.float32)
    da = xr.DataArray(out, coords=pt.coords, dims=pt.dims, name="precipitation",
                      attrs={"units": "mm/day", "long_name": f"{a.product}, quantile-mapped"})
    dest = Path(a.out); dest.mkdir(parents=True, exist_ok=True)
    y0, y1 = str(pt.time.values[0])[:4], str(pt.time.values[-1])[:4]
    path = dest / f"qm_1km_{y0}_{y1}.nc"
    da.to_dataset().to_netcdf(path, engine="netcdf4",
                              encoding={"precipitation": {"zlib": True, "complevel": 4}})
    print("written", path)


def _cmd_pm_mean(a):
    src = sorted(Path(a.deep_dir).glob("diffusion_1km_*.nc"))
    if not src:
        raise SystemExit(f"no diffusion_1km_*.nc in {a.deep_dir}")
    ds = xr.open_dataset(src[-1])
    if "precipitation_members" not in ds:
        raise SystemExit("this file has no `precipitation_members`; re-run "
                         "src.deep.infer with --save-members")
    mean = ds["precipitation"]
    mem = ds["precipitation_members"]
    print(f"{mem.sizes['member']} members, {mean.sizes['time']} days", flush=True)
    out = np.empty(mean.shape, np.float32)
    for i in range(mean.sizes["time"]):
        out[i] = probability_matched_mean(mem.isel(time=i).values.astype(np.float64),
                                          mean.isel(time=i).values.astype(np.float64))
        if i % 100 == 0:
            print(f"  {i}/{mean.sizes['time']}", flush=True)
    da = xr.DataArray(out, coords=mean.coords, dims=mean.dims, name="precipitation",
                      attrs={"units": "mm/day",
                             "long_name": "probability-matched ensemble mean"})
    dest = Path(a.out); dest.mkdir(parents=True, exist_ok=True)
    y0, y1 = str(mean.time.values[0])[:4], str(mean.time.values[-1])[:4]
    path = dest / f"qm_1km_{y0}_{y1}.nc"      # scored through the same slot
    da.to_dataset().to_netcdf(path, engine="netcdf4",
                              encoding={"precipitation": {"zlib": True, "complevel": 4}})
    print("written", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantile-map")
    q.add_argument("--config", default="config.yaml")
    q.add_argument("--product", default="xgboost")
    q.add_argument("--deep-dir", default=None)
    q.add_argument("--dev-dir", default=None,
                   help="separate file holding the dev-year predictions; deep products are "
                        "usually inferred on the test split only")
    q.add_argument("--out", required=True)
    m = sub.add_parser("pm-mean")
    m.add_argument("--deep-dir", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    setup_logging()
    if a.cmd == "quantile-map":
        _cmd_quantile_map(a)
    elif a.cmd == "pm-mean":
        _cmd_pm_mean(a)


if __name__ == "__main__":
    main()
