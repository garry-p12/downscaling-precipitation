"""Score every downscaling product on the same test days and compare them.

Products
--------
nearest / bilinear   IMERG 10 km repeated / bilinearly interpolated to 1 km
xgboost              the two-stage tree product (``main.py predict``)
cnn / swin           deep deterministic downscalers
diffusion            ensemble mean of the residual diffusion model
diffusion_member     a single diffusion realisation

Beyond the deterministic scores this computes the diagnostics that separate a
*blurry* field from a *realistic* one — power spectra, fractions skill score,
wet-area ratio and (for the ensemble) CRPS — because minimising RMSE drives a
model towards the conditional mean, which is smoother than any real rain field.

    python -m src.model_comparison --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .deep.metrics import crps_ensemble, fss, mean_rapsd, spectral_ratio, wet_area_ratio  # noqa: E402
from .utils import (LOG, grids_from_config, load_config, open_downscaled, save_json,  # noqa: E402
                    setup_logging, upsample_bilinear, upsample_nearest)
from .validation import INK, Contingency, SEQ_CMAP, Sums, _scalar  # noqa: E402

PALETTE = {
    "aorc": "#0b0b0b", "nearest": "#898781", "bilinear": "#eb6834", "xgboost": "#eda100",
    "cnn": "#2a78d6", "swin": "#1baf7a", "diffusion": "#4a3aa7", "diffusion_member": "#e87ba4",
}
LABELS = {
    "aorc": "AORC (observed)", "nearest": "IMERG (nearest)", "bilinear": "Bilinear", "xgboost": "XGBoost 2-stage",
    "cnn": "CNN (U-Net)", "swin": "Swin transformer", "diffusion": "Diffusion (ens. mean)",
    "diffusion_member": "Diffusion (1 member)",
}
FSS_THRESHOLDS = (1.0, 10.0, 30.0)
FSS_SCALES = (1, 5, 15, 41)


# --------------------------------------------------------------------------- #
def load_products(cfg: dict, grids, deep_dir: Path, val_slice: slice) -> dict:
    """Return {name: DataArray on the test slice} for every available product."""
    proc = Path(cfg["paths"]["processed"])
    rdir = Path(cfg["paths"]["results"])
    imerg = xr.open_dataset(proc / "imerg_aligned_10km.nc")["precip"].sel(time=val_slice).load()
    out = {
        "nearest": upsample_nearest(imerg, grids),
        "bilinear": upsample_bilinear(imerg, grids),
    }
    try:
        out["xgboost"] = open_downscaled(rdir, chunks={"time": 64}).sel(time=val_slice)
    except FileNotFoundError:
        LOG.warning("No tree product found in %s", rdir)
    for name in ("cnn", "swin", "diffusion"):
        f = sorted(deep_dir.glob(f"{name}_1km_*.nc"))
        if not f:
            continue
        ds = xr.open_dataset(f[-1], chunks={"time": 64})
        out[name] = ds["precipitation"].sel(time=val_slice)
        if "precipitation_member0" in ds:
            out[f"{name}_member"] = ds["precipitation_member0"].sel(time=val_slice)
    return out


def score(cfg: dict, deep_dir: Path, chunk: int = 32, spectra_days: int = 120) -> dict:
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    tcfg = cfg["time"]
    val_slice = slice(tcfg["val_start"], tcfg["val_end"])
    heavy = float(cfg["validation"].get("heavy_threshold_mm", 30.0))

    aorc = xr.open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": chunk})["precip"].sel(time=val_slice)
    products = load_products(cfg, grids, deep_dir, val_slice)
    names = list(products)
    LOG.info("Scoring %d products on %d test days: %s", len(names), aorc.sizes["time"], ", ".join(names))

    ny, nx = grids.fine.shape
    glob = {n: Sums() for n in names}
    cell = {n: Sums((ny, nx)) for n in names}
    ct = {n: {t: Contingency() for t in FSS_THRESHOLDS} for n in names}
    fss_acc = {n: {(t, s): np.zeros(3) for t in FSS_THRESHOLDS for s in FSS_SCALES} for n in names}
    wet_n = {n: np.zeros(2) for n in names}
    sum_field = {n: np.zeros((ny, nx)) for n in names}
    sum_obs = np.zeros((ny, nx))
    daily_mean_obs = []
    times = aorc["time"].values
    n_days = len(times)

    for t0 in range(0, n_days, chunk):
        sl = slice(t0, min(n_days, t0 + chunk))
        obs = aorc.isel(time=sl).values.astype(np.float32)
        ok0 = np.isfinite(obs)
        sum_obs += np.where(ok0, obs, 0).sum(0)
        daily_mean_obs.extend(np.nanmean(np.where(ok0, obs, np.nan), axis=(1, 2)).tolist())
        for n in names:
            p = np.asarray(products[n].isel(time=sl).values, dtype=np.float32)
            ok = ok0 & np.isfinite(p)
            glob[n].update(p, obs, ok)
            cell[n].update(p, obs, ok, axis=0)
            sum_field[n] += np.where(ok, p, 0).sum(0)
            for thr in FSS_THRESHOLDS:
                ct[n][thr].update(p, obs, ok, thr)
            wet_n[n] += [np.sum((p >= 1.0) & ok), np.sum((obs >= 1.0) & ok)]
            for thr in FSS_THRESHOLDS:
                for sc in FSS_SCALES:
                    fss_acc[n][(thr, sc)] += _fss_sums(p, obs, ok, thr, sc)
        LOG.info("  scored %s .. %s", str(times[sl][0])[:10], str(times[sl][-1])[:10])

    # Spectra on the wettest days (dry days carry no spatial signal).
    wettest = np.argsort(daily_mean_obs)[::-1][:spectra_days]
    obs_w = aorc.isel(time=wettest).values.astype(np.float32)
    wl, p_obs = mean_rapsd(obs_w, d=1.0)
    spectra = {"wavelength_km": wl.tolist(), "aorc": p_obs.tolist()}
    sratio = {}
    for n in names:
        _, p_n = mean_rapsd(np.asarray(products[n].isel(time=wettest).values, dtype=np.float32), d=1.0)
        spectra[n] = p_n.tolist()
        sratio[n] = spectral_ratio(wl, p_n, p_obs, max_wavelength=10.0)

    res = {"n_days": int(n_days), "period": [str(times[0])[:10], str(times[-1])[:10]], "products": {},
           "spectra": spectra, "spectra_days": int(spectra_days)}
    for n in names:
        g = _scalar(glob[n].metrics())
        c = cell[n].metrics()
        res["products"][n] = {
            "overall": g,
            "median_cell_nse": float(np.nanmedian(c["nse"])),
            "frac_cells_nse_gt_0.6": float(np.nanmean(c["nse"] > 0.6)),
            "detection": {str(t): ct[n][t].metrics() for t in FSS_THRESHOLDS},
            "fss": {f"{t}mm_{s}cell": _fss_value(fss_acc[n][(t, s)]) for t in FSS_THRESHOLDS for s in FSS_SCALES},
            "spectral_ratio_sub10km": sratio[n],
            "wet_area_ratio_1mm": float(wet_n[n][0] / max(wet_n[n][1], 1)),
        }
    ens = deep_dir / "diffusion_ensemble_stats.json"
    if ens.exists():
        res["diffusion_ensemble"] = json.load(open(ens))
    res["_fields"] = {"sum_obs": sum_obs, "sum_pred": sum_field, "wettest": wettest, "times": times}
    return res, products, aorc, grids


def _fss_sums(p, o, ok, thr, scale) -> np.ndarray:
    from scipy import ndimage

    bp = np.where(ok & (p >= thr), 1.0, 0.0)
    bo = np.where(ok & (o >= thr), 1.0, 0.0)
    size = (1, scale, scale)
    fp = ndimage.uniform_filter(bp, size=size, mode="nearest")
    fo = ndimage.uniform_filter(bo, size=size, mode="nearest")
    return np.array([np.sum((fp - fo) ** 2), np.sum(fp**2), np.sum(fo**2)])


def _fss_value(s: np.ndarray) -> float:
    den = s[1] + s[2]
    return float(1 - s[0] / den) if den > 0 else np.nan


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.set_facecolor(INK["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK["axis"])
    ax.tick_params(colors=INK["muted"], labelsize=8)
    ax.grid(True, color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def plot_spectra(res: dict, out: Path) -> Path:
    wl = np.array(res["spectra"]["wavelength_km"])
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=INK["surface"])
    _style(ax)
    ax.loglog(wl, res["spectra"]["aorc"], color=PALETTE["aorc"], lw=2.4, label=LABELS["aorc"], zorder=5)
    for n in res["products"]:
        if n not in res["spectra"]:
            continue
        ax.loglog(wl, res["spectra"][n], color=PALETTE.get(n, "#888"), lw=1.8,
                  ls="--" if n.endswith("member") else "-", label=LABELS.get(n, n))
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power spectral density")
    ax.invert_xaxis()
    ax.axvspan(wl.min(), 10, color=INK["grid"], alpha=0.35, zorder=0)
    ax.text(9.5, ax.get_ylim()[1] * 0.5, " sub-10 km ", fontsize=8, color=INK["secondary"], ha="right")
    ax.set_title("Spatial power spectra, wettest test days\n(a product below the black line is too smooth)",
                 fontsize=10, loc="left", color=INK["primary"])
    ax.legend(frameon=False, fontsize=8)
    p = out / "spectra.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fss(res: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, len(FSS_THRESHOLDS), figsize=(13, 3.6), facecolor=INK["surface"], sharey=True)
    for ax, thr in zip(axes, FSS_THRESHOLDS):
        _style(ax)
        for n, d in res["products"].items():
            y = [d["fss"][f"{thr}mm_{s}cell"] for s in FSS_SCALES]
            ax.plot(FSS_SCALES, y, color=PALETTE.get(n, "#888"), lw=1.8, marker="o", ms=4,
                    ls="--" if n.endswith("member") else "-", label=LABELS.get(n, n))
        ax.set_title(f"threshold {thr:g} mm/day", fontsize=9, loc="left", color=INK["primary"])
        ax.set_xlabel("Neighbourhood (km)")
        ax.set_xscale("log")
        ax.set_xticks(FSS_SCALES)
        ax.set_xticklabels([str(s) for s in FSS_SCALES])
    axes[0].set_ylabel("Fractions Skill Score")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    p = out / "fss.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_example_day(res: dict, products: dict, aorc, grids, out: Path) -> Path:
    t = int(res["_fields"]["wettest"][0])
    date = str(res["_fields"]["times"][t])[:10]
    names = ["aorc"] + list(products)
    ncol = min(4, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.6 * nrow), facecolor=INK["surface"])
    axes = np.atleast_1d(axes).ravel()
    obs = aorc.isel(time=t).values
    vmax = float(np.nanpercentile(obs, 99.5))
    lo_min, la_min, lo_max, la_max = grids.fine.edges
    for ax, n in zip(axes, names):
        f = obs if n == "aorc" else np.asarray(products[n].isel(time=t).values)
        im = ax.imshow(f, origin="lower", extent=(lo_min, lo_max, la_min, la_max), cmap=SEQ_CMAP,
                       vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(LABELS.get(n, n), fontsize=9, loc="left", color=INK["primary"])
        ax.tick_params(colors=INK["muted"], labelsize=7)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01, label="mm/day")
    fig.suptitle(f"Wettest test day: {date}", fontsize=11, color=INK["primary"], x=0.01, ha="left")
    p = out / f"example_day_{date}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def _markdown_table(df: pd.DataFrame) -> list[str]:
    """Render a DataFrame as a GitHub markdown table (avoids the tabulate dependency)."""
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [f"{v:g}" if isinstance(v, float) else str(v) for v in row]) + " |"
            for i, row in zip(df.index, df.values)]
    return [head, sep, *body]


def write_table(res: dict, out: Path) -> Path:
    rows = []
    for n, d in res["products"].items():
        o = d["overall"]
        # For a deterministic (single-valued) forecast CRPS reduces exactly to MAE,
        # so every product can be placed on the same probabilistic scale.
        crps = o["mae"]
        if n == "diffusion" and "diffusion_ensemble" in res:
            crps = res["diffusion_ensemble"]["crps_mm"]
        rows.append({
            "product": LABELS.get(n, n), "RMSE": o["rmse"], "MAE": o["mae"], "bias": o["bias"],
            "r": o["pearson_r"], "NSE": o["nse"], "KGE": o["kge"], "median cell NSE": d["median_cell_nse"],
            "CRPS": crps, "POD>30mm": d["detection"]["30.0"]["pod"], "CSI>30mm": d["detection"]["30.0"]["csi"],
            "FSS 10mm@15km": d["fss"]["10.0mm_15cell"], "spectral ratio <10km": d["spectral_ratio_sub10km"],
            "wet-area ratio": d["wet_area_ratio_1mm"],
        })
        res["products"][n]["crps_mm"] = crps
    df = pd.DataFrame(rows).set_index("product")
    lines = [f"# Model comparison — {res['period'][0]} .. {res['period'][1]} ({res['n_days']} test days)", "",
             *_markdown_table(df.round(3)), "",
             "*spectral ratio <10 km*: predicted/observed power below a 10 km wavelength. 1.0 = realistic "
             "texture; «1 = over-smoothed (a conditional-mean field); »1 = spurious fine-scale power, either "
             "blocky 10 km artifacts (nearest) or noise. *wet-area ratio*: predicted/observed fraction of "
             "cells above 1 mm/day. *CRPS*: for the deterministic products this is identical to MAE (a "
             "point forecast's CRPS); for the diffusion model it is the ensemble CRPS, which is what a "
             "probabilistic product should be judged on.", ""]
    if "diffusion_ensemble" in res:
        e = res["diffusion_ensemble"]
        lines += [f"Diffusion ensemble ({e['members']} members, {e['ddim_steps']} DDIM steps): "
                  f"CRPS {e['crps_mm']:.3f} mm/day, mean spread {e['ensemble_spread_mm']:.3f} mm/day.", ""]
    p = out / "model_comparison.md"
    p.write_text("\n".join(lines))
    df.round(4).to_csv(out / "model_comparison.csv")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--deep-dir", default=None)
    ap.add_argument("--spectra-days", type=int, default=120)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["results"]) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    rdir_ = Path(cfg["paths"]["results"])
    default_deep = next((rdir_ / n for n in ("v3", "v2", "final", "deep")
                             if (rdir_ / n).exists() and list((rdir_ / n).glob("*_1km_*.nc"))), rdir_ / "deep")
    deep_dir = Path(args.deep_dir) if args.deep_dir else default_deep

    res, products, aorc, grids = score(cfg, deep_dir, spectra_days=args.spectra_days)
    fields = res.pop("_fields")
    save_json(res, out / "model_comparison.json")
    figs = [plot_spectra(res, out), plot_fss(res, out)]
    res["_fields"] = fields
    figs.append(plot_example_day(res, products, aorc, grids, out))
    table = write_table(res, out)
    print(open(table).read())
    LOG.info("Comparison written to %s", out)


if __name__ == "__main__":
    main()
