"""Metrics the RSIR proposal names that our scorer did not compute.

Three additions, each tied to a sentence in the proposal.

**SSIM** ("Performance metrics -- including Mean Absolute Error (MAE), Root Mean
Square Error (RMSE), Structural Similarity Index (SSIM), and extreme event
capture"). SSIM asks whether two fields have the same local structure --
luminance, contrast and correlation inside a moving window -- rather than
whether they agree cell by cell. It is the natural companion to RMSE for a
product whose selling point is realistic texture, because a field can be
smooth and score well on RMSE while scoring badly here.

**KL divergence** ("rigorous uncertainty quantification techniques, including
ensemble spread analysis, KL divergence, and power spectral density
comparisons"). Taken over the *intensity distribution*: how much information is
lost by using the product's distribution of daily rainfall in place of the
observed one. Unlike RMSE it does not care where the rain was put, only whether
the product rains with the right frequency at the right depths -- which is
exactly the failure mode (hedging) that the per-cell losses in this study
produce. Computed on a common bin grid with a small floor, since KL is infinite
wherever the product assigns zero probability to an observed bin.

**Scale of use.** The proposal asks for products at 25, 10, 5 and 1 km. We
deliver one field at 1 km; a user who wants 10 km averages it. That is not the
same as training a 10 km model, and the difference is worth stating, but it is
the question a city actually faces: *if we use this product at 5 km, how good
is it there?* Every metric is therefore also reported with both fields
block-averaged to each candidate spacing.

    python -m src.proposal_metrics --config config_vista.yaml \
        --product M1=results/specM1/s2/cnn_1km_2019_2020.nc \
        --product XGBoost=results/xgboost_1km_2019_2020.nc \
        --out results/proposal_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from .utils import (evaluation_grids, grids_from_config, load_config,
                    setup_logging, subset_box)

SCALES = (1, 5, 10, 25)        # km, the proposal's set (1 km = native)
KL_EDGES = np.concatenate([[0.0, 0.1], np.geomspace(0.5, 400.0, 40)])


def ssim(p: np.ndarray, o: np.ndarray, win: int = 11, L: float | None = None) -> float:
    """Mean SSIM over a stack of 2-D fields, uniform window.

    The dynamic range L matters for precipitation and there is no convention:
    an 8-bit image uses 255, but rainfall has no ceiling. We use the observed
    99.9th percentile of the reference over the whole record, which is stable
    and is reported alongside the score so it can be reproduced.
    """
    if L is None:
        L = float(np.quantile(np.nan_to_num(o), 0.999))
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    size = (1, win, win)
    # Chunked over time: the intermediates are six float64 copies of the stack,
    # which is 5 GB for a two-year 1 km record and will not allocate.
    total, n = 0.0, 0
    for i in range(0, p.shape[0], 32):
        pc = np.nan_to_num(p[i:i + 32].astype(np.float64))
        oc = np.nan_to_num(o[i:i + 32].astype(np.float64))
        mu_p = ndimage.uniform_filter(pc, size=size, mode="nearest")
        mu_o = ndimage.uniform_filter(oc, size=size, mode="nearest")
        pp = ndimage.uniform_filter(pc * pc, size=size, mode="nearest") - mu_p ** 2
        oo = ndimage.uniform_filter(oc * oc, size=size, mode="nearest") - mu_o ** 2
        po = ndimage.uniform_filter(pc * oc, size=size, mode="nearest") - mu_p * mu_o
        num = (2 * mu_p * mu_o + c1) * (2 * po + c2)
        den = (mu_p ** 2 + mu_o ** 2 + c1) * (pp + oo + c2)
        m = num / den
        total += float(m.sum()); n += m.size
    return total / max(n, 1)


def kl_divergence(p: np.ndarray, o: np.ndarray, edges=KL_EDGES, floor: float = 1e-9) -> dict:
    """KL(observed || predicted) over the intensity distribution, in nats.

    The observed distribution is the reference, so the number reads as "the
    information lost by using the product's distribution in place of reality's".
    A floor is applied because KL is infinite wherever the product gives zero
    probability to a bin reality uses, which a product with a short tail always
    does.
    """
    ok = np.isfinite(p) & np.isfinite(o)
    hp, _ = np.histogram(p[ok], bins=edges)
    ho, _ = np.histogram(o[ok], bins=edges)
    hp = np.maximum(hp / max(hp.sum(), 1), floor)
    ho = np.maximum(ho / max(ho.sum(), 1), floor)
    hp, ho = hp / hp.sum(), ho / ho.sum()
    kl = float(np.sum(ho * np.log(ho / hp)))
    # The reverse direction penalises the opposite failure (raining where
    # reality never does), so both are reported.
    kl_rev = float(np.sum(hp * np.log(hp / ho)))
    return {"kl_obs_given_pred": kl, "kl_pred_given_obs": kl_rev,
            "js": float(0.5 * (np.sum(ho * np.log(ho / ((ho + hp) / 2)))
                               + np.sum(hp * np.log(hp / ((ho + hp) / 2)))))}


def coarsen(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a
    n, ny, nx = a.shape
    ny, nx = (ny // f) * f, (nx // f) * f
    return np.nanmean(a[:, :ny, :nx].reshape(n, ny // f, f, nx // f, f), axis=(2, 4))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_vista.yaml")
    ap.add_argument("--product", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", default="results/proposal_metrics.json")
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    _, box = evaluation_grids(cfg, grids)
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    src = Path(cfg["paths"]["processed"])

    o = xr.open_dataset(src / "aorc_aligned_1km.nc")["precip"].sel(time=sl)
    O = (subset_box(o, box) if box else o).values.astype(np.float32)
    fine_km = float(cfg["domain"]["coarse_res_deg"]) / int(cfg["domain"]["fine_factor"]) * 111.3
    L = float(np.quantile(np.nan_to_num(O), 0.999))
    print(f"reference {O.shape}, fine cell {fine_km:.2f} km, SSIM dynamic range {L:.1f} mm",
          flush=True)

    res = {"_meta": {"ssim_L_mm": L, "fine_km": fine_km,
                     "scales_km": list(SCALES), "kl_bins": len(KL_EDGES) - 1}}
    for spec in a.product:
        name, path = spec.split("=", 1)
        ds = xr.open_dataset(path)
        v = "precipitation" if "precipitation" in ds else list(ds.data_vars)[0]
        p = ds[v].sel(time=sl)
        P = (subset_box(p, box) if box else p).values.astype(np.float32)
        if P.shape != O.shape:
            raise SystemExit(f"{name}: shape {P.shape} != reference {O.shape}")
        rec = {}
        for km in SCALES:
            f = max(int(round(km / fine_km)), 1)
            Pc, Oc = coarsen(P, f), coarsen(O, f)
            k = np.isfinite(Pc) & np.isfinite(Oc)
            d = Pc[k].astype(np.float64) - Oc[k].astype(np.float64)
            r = {"factor": f, "actual_km": f * fine_km,
                 "rmse": float(np.sqrt(np.mean(d ** 2))),
                 "mae": float(np.mean(np.abs(d))),
                 "ssim": ssim(Pc, Oc, L=L)}
            r.update(kl_divergence(Pc, Oc))
            rec[f"{km}km"] = r
            print(f"  {name:<12} at {r['actual_km']:5.1f} km  rmse {r['rmse']:6.3f}  "
                  f"ssim {r['ssim']:.4f}  KL {r['kl_obs_given_pred']:.4f}", flush=True)
        res[name] = rec

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("written", a.out)


if __name__ == "__main__":
    main()
