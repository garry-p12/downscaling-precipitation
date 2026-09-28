"""Stack the downscaled products into one blended field.

    python -m src.stacking --fit          # fit weights on the dev year
    python -m src.stacking --apply        # write the blended test-period product

Every model sees the same inputs and makes different errors, so a weighted
combination is usually better than any member - the cheapest remaining gain
once single-model skill has plateaued. Weights are fitted by non-negative least
squares on the **dev year (2018)** only; the test years are never used, and the
non-negativity constraint keeps the blend interpretable (no member is
subtracted) and guards against the weights overfitting noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import LOG, grids_from_config, load_config, open_downscaled, save_json, setup_logging, upsample_bilinear

MEMBERS = ("xgboost", "cnn", "swin", "diffusion")


def deep_dir(cfg, period: str = "test") -> Path:
    """Directory holding the deep products for a period.

    Dev-year inference is written separately (``results/v2_dev``) so the test
    products are never overwritten by a fit-time run.
    """
    r = Path(cfg["paths"]["results"])
    order = ("v3_dev", "v2_dev", "final_dev") if period == "dev" else ("v3", "v2", "final", "deep")
    for n in order:
        if (r / n).exists() and list((r / n).glob("*_1km_*.nc")):
            return r / n
    return r / ("v3_dev" if period == "dev" else "v3")


def load_members(cfg, grids, tslice, include_bilinear=True, period: str = "test") -> dict[str, xr.DataArray]:
    r = Path(cfg["paths"]["results"])
    d = deep_dir(cfg, period)
    out: dict[str, xr.DataArray] = {}
    try:
        out["xgboost"] = open_downscaled(r, chunks={"time": 64}).sel(time=tslice)
    except FileNotFoundError:
        LOG.warning("no tree product found")
    for n in ("cnn", "swin", "diffusion"):
        f = sorted(d.glob(f"{n}_1km_*.nc"))
        if f:
            out[n] = xr.open_dataset(f[-1], chunks={"time": 64})["precipitation"].sel(time=tslice)
    if include_bilinear:
        im = xr.open_dataset(Path(cfg["paths"]["processed"]) / "imerg_aligned_10km.nc")["precip"].sel(time=tslice).load()
        out["bilinear"] = upsample_bilinear(im, grids)
    return out


def _nnls_weights(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls

    w, _ = nnls(A, b)
    return w


def check_no_leakage(cfg, dev_start: str, dev_end: str) -> None:
    """Refuse to fit weights on a period any member was trained on.

    A member trained through the fitting period looks artificially skilful
    there and absorbs all the weight, which is not a finding about the models
    but about the protocol. This check exists because that happened once.
    """
    meta = Path(cfg["paths"]["models"]) / "model_metadata.json"
    if not meta.exists():
        return
    with open(meta) as f:
        tr = json.load(f).get("train_period")
    if tr and not (tr[1] < dev_start or tr[0] > dev_end):
        raise SystemExit(
            f"tree model was trained on {tr[0]}..{tr[1]}, which overlaps the stacking fit window "
            f"{dev_start}..{dev_end}. Retrain with time.train_end before {dev_start}, or use "
            "out-of-fold predictions for the fit.")


def fit(cfg, sample_cells: int = 400_000, seed: int = 0, chunk: int = 32) -> dict:
    """Fit non-negative blend weights on the dev year."""
    grids = grids_from_config(cfg)
    dev = slice("2018-01-01", "2018-12-31")
    check_no_leakage(cfg, dev.start, dev.stop)
    aorc = xr.open_dataset(Path(cfg["paths"]["processed"]) / "aorc_aligned_1km.nc",
                           chunks={"time": chunk})["precip"].sel(time=dev)
    members = load_members(cfg, grids, dev, period="dev")
    # Weights must be fitted where every member has predictions, and that has to
    # be the dev year - fitting on the test period would leak, and fitting on the
    # training period would flatter whichever members were trained on it.
    missing = [n for n, v in members.items() if v.sizes.get("time", 0) == 0]
    for n in missing:
        members.pop(n)
    if missing:
        LOG.warning("no dev-year predictions for %s - run inference with --split dev for those models",
                    ", ".join(missing))
    names = [n for n in (*MEMBERS, "bilinear") if n in members]
    if len(names) < 2:
        raise SystemExit(
            f"need at least two products covering {dev.start}..{dev.stop}; have {names}. "
            "Run `python -m src.deep.infer --split dev` for the deep models first.")
    rng = np.random.default_rng(seed)
    nt = aorc.sizes["time"]
    per = max(1, sample_cells // max(nt, 1))
    X, y = [], []
    for t0 in range(0, nt, chunk):
        sl = slice(t0, min(nt, t0 + chunk))
        o = aorc.isel(time=sl).values.astype(np.float32)
        P = [np.asarray(members[n].isel(time=sl).values, dtype=np.float32) for n in names]
        ok = np.isfinite(o)
        for p in P:
            ok &= np.isfinite(p)
        idx = np.flatnonzero(ok.reshape(-1))
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(per * (sl.stop - sl.start), idx.size), replace=False)
        X.append(np.stack([p.reshape(-1)[take] for p in P], axis=1))
        y.append(o.reshape(-1)[take])
    A = np.concatenate(X)
    b = np.concatenate(y)
    w = _nnls_weights(A, b)
    pred = A @ w
    res = {
        "members": names,
        "weights": {n: float(v) for n, v in zip(names, w)},
        "weight_sum": float(w.sum()),
        "n_samples": int(len(b)),
        "dev_rmse_blend": float(np.sqrt(np.mean((pred - b) ** 2))),
        "dev_rmse_members": {n: float(np.sqrt(np.mean((A[:, i] - b) ** 2))) for i, n in enumerate(names)},
        "dev_rmse_equal_weight": float(np.sqrt(np.mean((A.mean(1) - b) ** 2))),
    }
    LOG.info("weights: %s", {k: round(v, 3) for k, v in res["weights"].items()})
    LOG.info("dev RMSE  blend %.4f | equal-weight %.4f | best member %.4f",
             res["dev_rmse_blend"], res["dev_rmse_equal_weight"], min(res["dev_rmse_members"].values()))
    save_json(res, Path(cfg["paths"]["results"]) / "stacking_weights.json")
    return res


def apply(cfg, chunk: int = 32) -> Path:
    """Write the blended product for the test period."""
    grids = grids_from_config(cfg)
    rdir = Path(cfg["paths"]["results"])
    wfile = rdir / "stacking_weights.json"
    if not wfile.exists():
        raise SystemExit("run --fit first")
    with open(wfile) as f:
        spec = json.load(f)
    tcfg = cfg["time"]
    test = slice(tcfg["val_start"], tcfg["val_end"])
    members = load_members(cfg, grids, test)
    names = [n for n in spec["members"] if n in members]
    w = np.array([spec["weights"][n] for n in names], dtype=np.float32)
    LOG.info("blending %s with weights %s", names, np.round(w, 3).tolist())
    ref = members[names[0]]
    nt = ref.sizes["time"]
    out = np.empty((nt,) + grids.fine.shape, np.float32)
    for t0 in range(0, nt, chunk):
        sl = slice(t0, min(nt, t0 + chunk))
        acc = np.zeros((sl.stop - sl.start,) + grids.fine.shape, np.float32)
        for n, wi in zip(names, w):
            acc += wi * np.nan_to_num(np.asarray(members[n].isel(time=sl).values, dtype=np.float32))
        out[sl] = np.clip(acc, 0, None)
    da = xr.DataArray(out, dims=("time", "lat", "lon"),
                      coords={"time": ref["time"].values, "lat": grids.fine.lat, "lon": grids.fine.lon},
                      attrs={"units": "mm/day", "long_name": "stacked downscaled precipitation"})
    ds = xr.Dataset({"precipitation": da})
    ds.attrs.update({"method": "stacked", "members": ",".join(names),
                     "weights": ",".join(f"{n}={spec['weights'][n]:.4f}" for n in names),
                     "fitted_on": "dev year 2018"})
    dst = deep_dir(cfg, "test") / f"stacked_1km_{str(ref['time'].values[0])[:4]}_{str(ref['time'].values[-1])[:4]}.nc"
    enc = {"precipitation": {"zlib": True, "complevel": 4, "dtype": "float32"},
           "time": {"units": "days since 1970-01-01", "dtype": "int32"}}
    ds.to_netcdf(dst, engine="netcdf4", encoding=enc)
    LOG.info("stacked product -> %s", dst)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--samples", type=int, default=400_000)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    if args.fit or not args.apply:
        fit(cfg, sample_cells=args.samples)
    if args.apply:
        apply(cfg)


if __name__ == "__main__":
    main()
