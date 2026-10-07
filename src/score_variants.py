"""Score many prediction files against one reference, with a day-block bootstrap.

``model_comparison`` recomputes the baselines for every call and reports point
estimates only. For a seeded comparison we need the opposite: the reference read
once, many variants scored against it, and an interval that respects how heavy
rain is distributed in time.

Heavy cells are not independent -- they cluster on a few dozen convective days
out of 731 -- so the sampling unit for a confidence interval is the **day**, not
the cell. Everything here is accumulated per day and resampled by day, which is
what decides whether a 0.05 difference in POD is solid or an artefact of which
storms happened to fall in the test period.

    python -m src.score_variants --config config_vista.yaml \\
        --control ctl=results/x/ctl.nc --product E=results/x/e.nc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from .deep.metrics import rapsd, spectral_ratio
from .utils import (evaluation_grids, grids_from_config, load_config, setup_logging,
                    subset_box)

HEAVY = 30.0
SCALES = (1, 5, 15, 41)


def _fractions(mask: np.ndarray, scale: int) -> np.ndarray:
    m = mask.astype(np.float64)
    return m if scale == 1 else uniform_filter(m, size=scale, mode="constant")


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """NaN-aware block mean of a 2-D field onto an f-times coarser grid."""
    h, w = a.shape
    return np.nanmean(a[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f), axis=(1, 3))


def accumulate(pred: np.ndarray, obs: np.ndarray, heavy: float = HEAVY,
               factor: int | None = None) -> dict:
    """Per-day quantities, so a bootstrap can resample days without re-reading.

    ``heavy`` is the detection threshold in mm. 30 is the Austin convention, but
    it is not portable: on the Colorado Front Range the same 30 mm is reached so
    rarely that a model scores a frequency bias of 0.04, and POD/CSI/FSS all
    collapse into a corner where they say nothing about the model. Pass a
    threshold matched on *exceedance rate* to compare detection across domains.

    ``factor`` turns on the 10 km diagnostic: both fields are block-averaged to
    the coarse grid and scored there as well. That splits the 1 km error into
    the part already present in the coarse field -- which is the retrieval being
    wrong, and which no downscaler can fix -- and the part added by the 10 km to
    1 km step itself. The perfect-input oracle shows the second term is small,
    so without this split an improvement cannot be attributed to the half of the
    problem it actually came from.
    """
    n_days = pred.shape[0]
    out = {k: np.zeros(n_days) for k in ("sse", "n", "hits", "misses", "fa",
                                         "sp", "so", "spp", "soo", "spo")}
    for s in SCALES:
        out[f"fnum{s}"] = np.zeros(n_days)
        out[f"fden{s}"] = np.zeros(n_days)
    if factor:
        out["sse_c"] = np.zeros(n_days)
        out["n_c"] = np.zeros(n_days)
    sp_acc = so_acc = None
    n_spec = 0
    for i in range(n_days):
        p, o = pred[i], obs[i]
        ok = np.isfinite(p) & np.isfinite(o)
        if not ok.any():
            continue
        d = p[ok] - o[ok]
        out["sse"][i] = float((d ** 2).sum()); out["n"][i] = ok.sum()
        out["sp"][i] = p[ok].sum(); out["so"][i] = o[ok].sum()
        out["spp"][i] = (p[ok] ** 2).sum(); out["soo"][i] = (o[ok] ** 2).sum()
        out["spo"][i] = (p[ok] * o[ok]).sum()
        pe, oe = (p >= heavy) & ok, (o >= heavy) & ok
        out["hits"][i] = (pe & oe).sum()
        out["misses"][i] = (~pe & oe).sum()
        out["fa"][i] = (pe & ~oe).sum()
        for s in SCALES:
            pf, of = _fractions(pe, s), _fractions(oe, s)
            out[f"fnum{s}"][i] = float(((pf - of) ** 2).sum())
            out[f"fden{s}"][i] = float((pf ** 2 + of ** 2).sum())
        if factor:
            # Mask first: a block mean over cells the model never saw would
            # otherwise credit it with the reference's values there.
            pc = _block_mean(np.where(ok, p, np.nan), factor)
            oc = _block_mean(np.where(ok, o, np.nan), factor)
            okc = np.isfinite(pc) & np.isfinite(oc)
            out["sse_c"][i] = float(((pc[okc] - oc[okc]) ** 2).sum())
            out["n_c"][i] = float(okc.sum())
        if np.nanmax(o) > 0:
            _, ps = rapsd(np.where(ok, p, np.nan))
            wl, os_ = rapsd(np.where(ok, o, np.nan))
            sp_acc = ps if sp_acc is None else sp_acc + ps
            so_acc = os_ if so_acc is None else so_acc + os_
            n_spec += 1
    out["_spec"] = (wl, sp_acc / max(n_spec, 1), so_acc / max(n_spec, 1)) if n_spec else None
    return out


def summarise(a: dict, idx: np.ndarray | None = None) -> dict:
    """Point scores from the per-day accumulators, over a (possibly resampled) day set."""
    sel = slice(None) if idx is None else idx
    g = lambda k: a[k][sel].sum()
    n = max(g("n"), 1)
    rmse = float(np.sqrt(g("sse") / n))
    mp, mo = g("sp") / n, g("so") / n
    vp = max(g("spp") / n - mp ** 2, 1e-12); vo = max(g("soo") / n - mo ** 2, 1e-12)
    r = (g("spo") / n - mp * mo) / np.sqrt(vp * vo)
    kge = float(1 - np.sqrt((r - 1) ** 2 + (np.sqrt(vp / vo) - 1) ** 2 + (mp / mo - 1) ** 2))
    h, m_, f = g("hits"), g("misses"), g("fa")
    out = {"rmse": rmse, "kge": kge}
    if "sse_c" in a:
        nc = max(g("n_c"), 1)
        r10 = float(np.sqrt(g("sse_c") / nc))
        out["rmse_10km"] = r10
        # Variances add when the two error sources are independent, so the
        # downscaling-only term is what is left after removing the coarse error.
        # Clipped at zero: a negative value means they are not independent, not
        # that the downscaler is better than perfect.
        out["rmse_downscale"] = float(np.sqrt(max(rmse ** 2 - r10 ** 2, 0.0)))
        out["coarse_error_share"] = float(min(r10 ** 2 / max(rmse ** 2, 1e-12), 1.0))
    out.update({
           "pod": float(h / max(h + m_, 1)), "far": float(f / max(h + f, 1)),
           "csi": float(h / max(h + m_ + f, 1)),
           "frequency_bias": float((h + f) / max(h + m_, 1))})
    for s in SCALES:
        out[f"fss{s}"] = float(1 - g(f"fnum{s}") / max(g(f"fden{s}"), 1e-12))
    return out


def bootstrap(a: dict, n_boot: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    nd = len(a["n"])
    keys = ["pod", "csi", "far", "fss5", "fss15", "rmse", "kge"]
    if "sse_c" in a:
        keys += ["rmse_10km", "rmse_downscale"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, nd, nd)
        s = summarise(a, idx)
        for k in keys:
            draws[k].append(s[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--product", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--control", default=None, metavar="NAME=PATH")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--heavy", type=float, default=HEAVY,
                    help=f"detection threshold in mm (default {HEAVY:g}); see accumulate()")
    ap.add_argument("--match-exceedance", type=float, default=None, metavar="RATE",
                    help="ignore --heavy and pick the threshold whose observed exceedance "
                         "rate equals RATE, so detection means the same thing across domains")
    ap.add_argument("--no-coarse", action="store_true",
                    help="skip the 10 km diagnostic (block-mean scoring on the coarse grid)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    setup_logging()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": 32})["precip"].sel(time=sl)
    _, box = evaluation_grids(cfg, grids)
    if box:
        obs = subset_box(obs, box)
    o = obs.values.astype(np.float32)
    print(f"reference: {o.shape[0]} days, {o.shape[1]} x {o.shape[2]}", flush=True)

    heavy = a.heavy
    fin = o[np.isfinite(o)]
    if a.match_exceedance is not None:
        # The quantile of the threshold, not the threshold itself, is what makes
        # "POD at heavy rain" comparable between a convective and an orographic
        # domain. Austin's 30 mm sits at some exceedance rate; we ask Colorado
        # for the depth at that same rate.
        heavy = float(np.quantile(fin, 1.0 - a.match_exceedance))
        print(f"matched exceedance {a.match_exceedance:.3e} -> threshold {heavy:.2f} mm", flush=True)
    rate = float((fin >= heavy).mean())
    print(f"detection threshold {heavy:.2f} mm; observed exceedance {rate:.3e} "
          f"({int((fin >= heavy).sum())} of {fin.size} cells)", flush=True)

    items = ([a.control] if a.control else []) + a.product
    res, accs = {}, {}
    for spec in items:
        name, path = spec.split("=", 1)
        da = xr.open_dataset(path)["precipitation"].sel(time=sl)
        if box:
            da = subset_box(da, box)
        if da.shape != o.shape:
            # Silently scoring a short product against the first N reference
            # days produces a plausible number for a different period, which is
            # worse than failing: a one-year file compared against a two-year
            # reference once sat in a table beside two-year products.
            raise SystemExit(
                f"{name}: shape {da.shape} does not match the reference {o.shape}. "
                f"Pass every year of the product (comma-separate paths), or slice "
                f"the reference to match.")
        acc = accumulate(da.values.astype(np.float32), o, heavy,
                         factor=None if a.no_coarse else grids.factor)
        accs[name] = acc
        rec = summarise(acc)
        if acc["_spec"]:
            wl, sp, so = acc["_spec"]
            rec["spectral_ratio_sub10km"] = float(spectral_ratio(wl, sp, so, 10.0))
        rec["ci"] = bootstrap(acc, a.n_boot)
        res[name] = rec
        extra = ""
        if "rmse_10km" in rec:
            extra = (f"  | 10km {rec['rmse_10km']:.3f}  dwn {rec['rmse_downscale']:.3f}"
                     f"  coarse {rec['coarse_error_share']:.1%}")
        print(f"  {name:22s} rmse {rec['rmse']:.3f}  spec {rec.get('spectral_ratio_sub10km', float('nan')):.3f}"
              f"  POD {rec['pod']:.3f}  CSI {rec['csi']:.3f}{extra}", flush=True)

    # paired-by-day differences against the control
    if a.control:
        cname = a.control.split("=", 1)[0]
        base = accs[cname]
        rng = np.random.default_rng(1)
        nd = len(base["n"])
        for name, acc in accs.items():
            if name == cname:
                continue
            d = {k: [] for k in ("pod", "csi", "rmse", "fss5")}
            for _ in range(a.n_boot):
                idx = rng.integers(0, nd, nd)
                s1, s0 = summarise(acc, idx), summarise(base, idx)
                for k in d:
                    d[k].append(s1[k] - s0[k])
            res[name]["vs_control"] = {
                k: {"mean": float(np.mean(v)),
                    "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                for k, v in d.items()}

    # A score file that does not carry its own threshold invites exactly the
    # mistake this flag exists to fix: comparing a 30 mm POD against a matched
    # one and reading the difference as a model effect.
    res["_meta"] = {"heavy_mm": heavy, "observed_exceedance": rate,
                    "config": a.config, "n_boot": a.n_boot}

    dest = Path(a.out) if a.out else Path("results/variant_scores.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(dest, "w"), indent=2)
    print("written", dest)


if __name__ == "__main__":
    main()
