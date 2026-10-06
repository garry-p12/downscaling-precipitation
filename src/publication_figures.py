"""Publication-grade figures for the downscaling paper.

Separate from ``figures.py`` / ``paper_figures.py`` / ``regime_figures.py``,
which render for slides: in-figure bold titles, saturated fills, value labels
larger than the axis type. Journals want the opposite -- the caption carries
the title, the type is uniform and small, the ink is thin, and the palette is
colourblind-safe. Every figure here is emitted as vector PDF plus 600 dpi PNG
at a true column width, so nothing is rescaled in the manuscript.

All numbers are read from pipeline outputs. Where a value is stated inline it
is a logged run, and the comment says which.

    python -m src.publication_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = Path("results/figures/publication")

# Column widths in inches: 88 mm single, 180 mm double (AGU/Elsevier standard).
W1, W2 = 3.46, 7.09

# Okabe-Ito, safe under all common colour-vision deficiencies.
BLACK   = "#000000"
ORANGE  = "#E69F00"
SKY     = "#56B4E9"
GREEN   = "#009E73"
YELLOW  = "#F0E442"
BLUE    = "#0072B2"
VERM    = "#D55E00"
PURPLE  = "#CC79A7"
GREY    = "#5A5A5A"
GREY_L  = "#BFBFBF"

PRODUCT_STYLE = {
    "nearest":          ("IMERG, nearest",    GREY,   "o"),
    "bilinear":         ("Bilinear",          GREY,   "s"),
    "xgboost":          ("XGBoost",           BLUE,   "D"),
    "cnn":              ("CNN",               GREEN,  "^"),
    "swin":             ("Swin",              ORANGE, "v"),
    "diffusion":        ("Diffusion, mean",   VERM,   "P"),
    "diffusion_member": ("Diffusion, member", PURPLE, "X"),
    "cnn_spectral":     ("CNN + spectral",    SKY,    "*"),
}


def house_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#2B2B2B",
        "grid.linewidth": 0.4,
        "grid.color": GREY_L,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,          # embed as TrueType, not Type 3
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def ax_style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, lw=0.4, color=GREY_L, zorder=0)
        ax.set_axisbelow(True)


def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  {name:38s} {caption}")


def load(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


# ============================================================ F1: forcing
def f1_forcing_ladder():
    """Skill tracks how orographically forced the precipitation is."""
    # Logged runs: config_colorado_{cool,warm}.yaml, config_colorado.yaml, config.yaml
    rows = [
        ("Colorado, Oct–Apr", "orographic",        20.8, 0.255),
        ("Colorado, all year",     "mixed",             15.0, 0.255),
        ("Colorado, May–Sep", "convective",         8.0, 0.255),
        ("Austin, all year",       "convective, flat",   6.8, 0.000),
    ]
    fig, ax = plt.subplots(figsize=(W1, 1.95))
    y = np.arange(len(rows))[::-1]
    vals = [r[2] for r in rows]
    ax.hlines(y, 0, vals, color=GREY, lw=0.8, zorder=2)
    ax.plot(vals, y, "o", color=BLUE, ms=4.5, zorder=3, clip_on=False)
    ax.axvline(20, color=VERM, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(20, len(rows) - 0.35, "20 % target", color=VERM, fontsize=6.5,
            ha="right", va="bottom", rotation=0)
    for yi, (lab, sub, v, _) in zip(y, rows):
        ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\n{r[1]}" for r in rows], fontsize=7, linespacing=1.25)
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("RMSE reduction against bilinear (%)")
    ax_style(ax, grid_axis="x")
    # The decisive pair: same skill, 25.5 points of terrain apart.
    ax.annotate("", xy=(8.0, y[2]), xytext=(6.8, y[3]),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=VERM,
                                connectionstyle="bar,fraction=0.28"))
    ax.text(11.4, (y[2] + y[3]) / 2, "1.2 points apart,\n25.5 points of terrain",
            fontsize=6.3, color=VERM, va="center", linespacing=1.3)
    save(fig, "F1_forcing_ladder",
         "RMSE reduction by forcing regime; the 8.0/6.8 pair is the result")


# ======================================================= F3: accuracy-realism
def _plane_points(cmp_json, extra=None):
    pts = []
    for k, p in cmp_json["products"].items():
        if k not in PRODUCT_STYLE:
            continue
        pts.append((k, p["overall"]["rmse"], p["spectral_ratio_sub10km"]))
    if extra:
        pts += extra
    return pts


def f3_accuracy_realism():
    """The trade-off, and the one product that escapes it."""
    aus = load("results/comparison/model_comparison.json")
    pwr = load("results/power/comparison/model_comparison.json")
    spc = load("results/spectral/comparison-spec-w001/model_comparison.json")
    ctl = load("results/spectral/comparison-spec-ctl/model_comparison.json")
    pspc = load("results/power/comparison-spec-w0.01/model_comparison.json")
    pctl = load("results/power/comparison-spec-w0.0/model_comparison.json")
    if not (aus and pwr):
        print("  F3 skipped: comparison json missing")
        return

    # Hand-placed labels: these points cluster, and an automatic placer either
    # overlaps them or drags leader lines across the panel.
    OFF = {
        ("a", "cnn_spectral"):     (7, 0, "left", "center"),
        ("a", "cnn_control"):      (7, 6, "left", "bottom"),
        ("a", "cnn"):              (7, -4, "left", "top"),
        ("a", "swin"):             (-7, -4, "right", "top"),
        ("a", "diffusion"):        (7, 3, "left", "bottom"),
        ("a", "diffusion_member"): (-7, 0, "right", "center"),
        ("a", "bilinear"):         (7, 0, "left", "center"),
        ("a", "xgboost"):          (7, 0, "left", "center"),
        ("b", "cnn"):              (0, -9, "center", "top"),
        ("b", "cnn_spectral"):     (-7, 0, "right", "center"),
        ("b", "cnn_control"):      (7, -5, "left", "top"),
        ("b", "swin"):             (-7, -3, "right", "top"),
        ("b", "diffusion"):        (-7, -3, "right", "top"),
        ("b", "diffusion_member"): (-7, 0, "right", "center"),
        ("b", "bilinear"):         (-7, 0, "right", "center"),
        ("b", "xgboost"):          (7, 0, "left", "center"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))
    for ax, data, title, letter, extra in (
        (axes[0], aus, "Austin \u2014 IMERG, 10 km input", "a",
         [("cnn_spectral", spc["products"]["cnn"]["overall"]["rmse"],
           spc["products"]["cnn"]["spectral_ratio_sub10km"])] if spc else None),
        (axes[1], pwr, "NASA POWER \u2014 50 km input", "b",
         [("cnn_spectral", pspc["products"]["cnn"]["overall"]["rmse"],
           pspc["products"]["cnn"]["spectral_ratio_sub10km"])] if pspc else None),
    ):
        ax.axhline(1.0, color=GREY_L, lw=0.7, ls=(0, (4, 3)), zorder=1)
        for k, rmse, spec in _plane_points(data, extra):
            if k == "nearest":          # blocky artefacts put it far off-scale
                continue
            lab, col, mk = PRODUCT_STYLE[k]
            ms = 7 if mk == "*" else 5
            ax.plot(rmse, spec, mk, color=col, ms=ms, mec="white", mew=0.5, zorder=4)
            dx, dy, ha, va = OFF.get((letter, k), (6, 3, "left", "bottom"))
            ax.annotate(lab, (rmse, spec), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.2, color=col, ha=ha, va=va)
        matched = ctl if letter == "a" else pctl
        if matched:
            c = matched["products"]["cnn"]
            x, y = c["overall"]["rmse"], c["spectral_ratio_sub10km"]
            ax.plot(x, y, "^", color=GREEN, ms=5, mfc="white", mew=0.9, zorder=4)
            dx, dy, ha, va = OFF.get((letter, "cnn_control"), (7, 6, "left", "bottom"))
            ax.annotate("CNN, matched control", (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.2, color=GREEN, ha=ha, va=va)
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 6)
        ax.set_xlabel("RMSE (mm day$^{-1}$)")
        ax.set_title(title, fontsize=7.5, pad=5)
        ax_style(ax, grid_axis=None)
        ax.text(0.985, 1.0, "correct texture", transform=ax.get_yaxis_transform(),
                fontsize=6.2, color=GREY, ha="right", va="bottom")
    axes[0].set_ylabel("retained sub-10 km variance\n(predicted / observed)")
    axes[0].margins(x=0.16)
    axes[1].margins(x=0.16)
    for a, L in zip(axes, "ab"):
        panel(a, L, dx=-0.13)
    save(fig, "F3_accuracy_realism_plane",
         "RMSE against retained fine-scale variance; note log y and the 1.0 line")


# ============================================================ F4: spectra
def f4_spectra():
    """Where in scale each product loses its variance."""
    aus = load("results/comparison/model_comparison.json")
    pwr = load("results/power/comparison/model_comparison.json")
    spc = load("results/spectral/comparison-spec-w001/model_comparison.json")
    if not (aus and pwr):
        print("  F4 skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.75), sharey=True)
    for ax, d, title in ((axes[0], aus, "Austin (IMERG, 10 km input)"),
                         (axes[1], pwr, "NASA POWER (50 km input)")):
        sp = d["spectra"]
        wl = np.asarray(sp["wavelength_km"], float)
        obs = np.asarray(sp["aorc"], float)
        ok = np.isfinite(wl) & np.isfinite(obs) & (obs > 0) & (wl > 1.8)
        ax.plot(wl[ok], obs[ok], color=BLACK, lw=1.4, zorder=5, label="AORC (observed)")
        for k in ("bilinear", "xgboost", "cnn", "swin", "diffusion", "diffusion_member"):
            if k not in sp:
                continue
            p = np.asarray(sp[k], float)
            m = ok & np.isfinite(p) & (p > 0)
            lab, col, _ = PRODUCT_STYLE[k]
            ax.plot(wl[m], p[m], color=col, lw=1.0, label=lab)
        # the spectral-penalty run, overlaid where it was trained
        if ax is axes[0] and spc and "cnn" in spc.get("spectra", {}):
            q = np.asarray(spc["spectra"]["cnn"], float)
            wq = np.asarray(spc["spectra"]["wavelength_km"], float)
            m = np.isfinite(wq) & np.isfinite(q) & (q > 0) & (wq > 1.8)
            ax.plot(wq[m], q[m], color=SKY, lw=1.3, ls=(0, (3, 1.4)),
                    label="CNN + spectral", zorder=4)
        ax.axvspan(wl[ok].min(), 10, color=GREY_L, alpha=0.25, lw=0, zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("wavelength (km)")
        ax.set_title(title, fontsize=7.5, pad=4)
        ax_style(ax, grid_axis=None)
    axes[0].set_ylabel("radially averaged power")
    axes[0].text(9.4, axes[0].get_ylim()[0] * 60, "< 10 km", fontsize=6.2, color=GREY)
    axes[0].legend(loc="lower left", ncol=1, handlelength=1.4, borderpad=0.2,
                   labelspacing=0.22)
    for a, L in zip(axes, "ab"):
        panel(a, L, dx=-0.10)
    save(fig, "F4_power_spectra",
         "RAPSD against AORC; shaded band is the sub-10 km range scored")


# ================================================ F5: the spectral exchange rate
def _ablate_rows():
    rows = []
    for f in sorted(Path("results/spectral/ablate-spectral").glob("*train_log*.json")):
        d = json.load(open(f))
        h = [x for x in d["history"] if np.isfinite(x.get("rmse", np.nan))]
        if not h:
            continue
        b = min(h, key=lambda x: x["rmse"])
        rows.append((d["args"].get("spectral_weight"), b, h))
    rows.sort(key=lambda r: r[0])
    return rows


def f5_spectral_exchange():
    """What a unit of texture costs in RMSE, and how the pair trains."""
    rows = _ablate_rows()
    if not rows:
        print("  F5 skipped: no ablation logs")
        return
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))

    ax = axes[0]
    w = [r[0] for r in rows]
    rmse = [r[1]["rmse"] for r in rows]
    spec = [r[1]["spec"] for r in rows]
    ax.plot(spec, rmse, "-", color=GREY, lw=0.8, zorder=2)
    ax.scatter(spec, rmse, c=range(len(rows)), cmap="viridis", s=26, zorder=3,
               edgecolors="white", linewidths=0.5)
    # stagger, the top three weights sit almost on top of one another
    woff = {0.0: (8, -3, "left"), 0.01: (8, -3, "left"), 0.05: (8, 1, "left"),
            0.2: (-8, 5, "right"), 1.0: (8, 0, "left")}
    for wi, sp_, r in zip(w, spec, rmse):
        dx, dy, ha = woff.get(wi, (7, 0, "left"))
        ax.annotate(f"w = {wi:g}", (sp_, r), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.2, ha=ha)
    ax.axvline(1.0, color=GREY_L, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("retained sub-10 km variance")
    ax.set_ylabel("dev RMSE (mm day$^{-1}$)")
    ax_style(ax, grid_axis=None)

    ax = axes[1]
    for f, col, lab in (("cnn_long_w0.0_train_log.json", GREY, "w = 0"),
                        ("cnn_long_w0.01_train_log.json", BLUE, "w = 0.01")):
        p = Path("results/spectral/spec-long") / f
        if not p.exists():
            continue
        h = json.load(open(p))["history"]
        st = [x["step"] / 1000 for x in h]
        ax.plot(st, [x["spec"] for x in h], color=col, lw=1.1, label=lab)
    ax.axhline(1.0, color=GREY_L, lw=0.7, ls=(0, (4, 3)))
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("retained sub-10 km variance")
    ax.legend(loc="lower right", handlelength=1.4)
    ax_style(ax, grid_axis=None)

    fig.subplots_adjust(wspace=0.34)
    for a, L in zip(axes, "ab"):
        panel(a, L, dx=-0.17)
    save(fig, "F5_spectral_exchange",
         "(a) weight sweep, best checkpoint each; (b) texture during training")


# ======================================================== F6: seed repeats
def f6_seeds():
    """Which differences survive run-to-run noise."""
    # Published families, section 5.8 (test RMSE, seeds 1/2/published).
    fam = [
        ("XGBoost",  [4.578, 4.583, 4.585, 4.583], BLUE),
        ("Swin",     [4.614, 4.646, 4.653],        ORANGE),
        ("CNN",      [4.701, 4.669, 4.663],        GREEN),
    ]
    # keyed by seed, so the connecting lines are the real within-seed pairs
    pairs: dict[float, dict[int, float]] = {}
    for f in sorted(Path("results/spectral/spec-seeds").glob("*train_log*.json")):
        d = json.load(open(f)); a = d["args"]
        h = [x for x in d["history"] if np.isfinite(x.get("rmse", np.nan))]
        if h:
            pairs.setdefault(a.get("spectral_weight"), {})[a.get("seed")] = \
                min(h, key=lambda x: x["rmse"])["rmse"]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.35))

    ax = axes[0]
    for i, (name, vals, col) in enumerate(fam):
        y = len(fam) - 1 - i
        ax.plot([min(vals), max(vals)], [y, y], color=col, lw=2.6, alpha=0.35,
                solid_capstyle="round")
        ax.plot(vals, [y] * len(vals), "o", color=col, ms=3.4, mec="white", mew=0.4)
        ax.text(max(vals) + 0.006, y, f"{max(vals)-min(vals):.3f}", fontsize=6.3,
                va="center", color=GREY)
    ax.set_yticks(range(len(fam)))
    ax.set_yticklabels([f[0] for f in fam][::-1])
    ax.set_xlabel("test RMSE (mm day$^{-1}$)")
    ax.set_ylim(-0.6, len(fam) - 0.4)
    ax_style(ax, grid_axis="x")
    ax.text(0.98, 0.06, "range across seeds", transform=ax.transAxes,
            fontsize=6.3, color=GREY, ha="right")

    ax = axes[1]
    if pairs:
        ks = sorted(pairs)
        seeds = sorted(set.intersection(*(set(pairs[k]) for k in ks)))
        for sd in seeds:
            ax.plot(range(len(ks)), [pairs[k][sd] for k in ks],
                    color=GREY_L, lw=0.6, zorder=2)
        for i, k in enumerate(ks):
            v = [pairs[k][sd] for sd in seeds]
            col = GREY if k == 0 else BLUE
            ax.plot([i] * len(v), v, "o", color=col, ms=4, mec="white", mew=0.5, zorder=3)
            ax.plot([i - 0.14, i + 0.14], [np.mean(v)] * 2, color=col, lw=1.6, zorder=4)
        d_mean = np.mean([pairs[ks[1]][sd] - pairs[ks[0]][sd] for sd in seeds])
        ax.text(0.5, 0.04, f"paired mean difference {d_mean:+.4f} mm day$^{{-1}}$",
                transform=ax.transAxes, ha="center", fontsize=6.3, color=GREY)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"w = {k:g}" for k in ks])
        ax.set_xlim(-0.45, len(ks) - 0.55)
        ax.set_ylabel("dev RMSE (mm day$^{-1}$)")
        ax_style(ax, grid_axis="y")
    for a, L in zip(axes, "ab"):
        panel(a, L, dx=-0.15)
    save(fig, "F6_seed_repeats",
         "(a) family spread, section 5.8; (b) the spectral pair, three seeds")


# ========================================================= F7: FSS breakdown
def f7_fss():
    """Does the added variance land in the right places?"""
    ctl = load("results/spectral/comparison-spec-ctl/model_comparison.json")
    spc = load("results/spectral/comparison-spec-w001/model_comparison.json")
    if not (ctl and spc):
        print("  F7 skipped")
        return
    a, b = ctl["products"]["cnn"]["fss"], spc["products"]["cnn"]["fss"]
    # keys are flat, "<threshold>mm_<scale>cell"
    grid: dict[float, dict[int, str]] = {}
    for k in a:
        thr, cell = k.split("mm_")
        grid.setdefault(float(thr), {})[int(cell.replace("cell", ""))] = k
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    scales = None
    for thr, col in zip(sorted(grid), (BLUE, ORANGE, VERM)):
        scales = sorted(grid[thr])
        d = [(b[grid[thr][s]] - a[grid[thr][s]]) * 1000 for s in scales]
        ax.plot(scales, d, "-o", color=col, ms=3.2, label=f"{thr:g} mm")
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(scales); ax.set_xticklabels([str(s) for s in scales])
    ax.set_xlabel("neighbourhood width (cells)")
    ax.set_ylabel("$\\Delta$ FSS $\\times\\,10^{3}$\n(spectral $-$ control)")
    ax.legend(title="threshold", loc="upper right", handlelength=1.3)
    ax_style(ax, grid_axis="y")
    save(fig, "F7_fss_placement",
         "placement skill: light rain improves, heavy rain does not")


# =============================================== F2: two domains, F8: inputs
def _dot_panel(ax, labels, series, xlabel, invert=False):
    y = np.arange(len(labels))[::-1]
    for (name, vals, col, mk, filled) in series:
        ax.plot(vals, y, mk, color=col, ms=4.4, mec=col, mew=0.9,
                mfc=col if filled else "white", ls="none", label=name, zorder=3)
    for yi in y:
        ax.axhline(yi, color=GREY_L, lw=0.4, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    if invert:
        ax.invert_xaxis()
    ax_style(ax, grid_axis=None)


def f2_two_domains():
    """Identical boxes, different regime."""
    # Section 4.5, logged runs: config.yaml and config_colorado.yaml.
    labels = ["RMSE reduction (%)", "Pearson $r$", "NSE", "KGE", "median cell NSE"]
    aus_ml = [6.8, 0.839, 0.702, 0.739, 0.711]
    aus_bl = [0.0, 0.815, 0.657, 0.787, 0.661]
    col_ml = [15.0, 0.717, 0.493, 0.474, 0.490]
    col_bl = [0.0, 0.576, 0.299, 0.215, 0.303]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    for ax, ml, bl, title in ((axes[0], aus_ml, aus_bl, "Austin (flat)"),
                              (axes[1], col_ml, col_bl, "Colorado Front Range")):
        _dot_panel(ax, labels,
                   [("bilinear", bl, GREY, "o", False),
                    ("downscaled", ml, BLUE, "o", True)],
                   "score")
        ax.set_xlim(-0.05, 1.0)
        ax.set_title(title, fontsize=7.5, pad=4)
        # the % reduction row lives on a different scale; annotate rather than plot
        ax.plot([], [])
        ax.text(0.995, len(labels) - 1, f"{ml[0]:.1f} %", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7, color=BLUE, fontweight="bold")
    axes[1].set_yticklabels([])
    axes[0].legend(loc="lower left", handlelength=1.2, bbox_to_anchor=(0.0, -0.03))
    for a, L in zip(axes, "ab"):
        panel(a, L, dx=-0.42 if L == "a" else -0.06)
    save(fig, "F2_two_domains",
         "same box size, period, splits and predictors; only regime differs")


def f8_input_resolution():
    """What changes when the coarse input is five times coarser."""
    aus = load("results/comparison/model_comparison.json")
    pwr = load("results/power/comparison/model_comparison.json")
    if not (aus and pwr):
        print("  F8 skipped")
        return
    order = ["bilinear", "xgboost", "swin", "cnn", "diffusion", "diffusion_member"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4))
    # tuple paths, because the detection threshold key is itself "30.0"
    specs = [(("overall", "rmse"), "RMSE (mm day$^{-1}$)"),
             (("detection", "30.0", "pod"), "POD above 30 mm"),
             (("spectral_ratio_sub10km",), "retained sub-10 km variance")]
    for ax, (key, xlabel) in zip(axes, specs):
        def get(d, k, _key=None):
            p = d["products"][k]
            for part in (_key or key):
                p = p[part]
            return p
        labels = [PRODUCT_STYLE[k][0] for k in order]
        _dot_panel(ax, labels,
                   [("IMERG, 10 km", [get(aus, k) for k in order], BLUE, "o", True),
                    ("POWER, 50 km", [get(pwr, k) for k in order], VERM, "s", True)],
                   xlabel)
        if key[0] == "spectral_ratio_sub10km":
            ax.set_xscale("log")
    for a in axes[1:]:
        a.set_yticklabels([])
    axes[0].legend(loc="lower right", handlelength=1.2)
    for a, L in zip(axes, "abc"):
        panel(a, L, dx=-0.58 if L == "a" else -0.08)
    save(fig, "F8_input_resolution",
         "every product, both inputs, scored on the same Austin box and days")


def main():
    house_style()
    print("publication figures ->", OUT)
    f1_forcing_ladder()
    f2_two_domains()
    f3_accuracy_realism()
    f4_spectra()
    f5_spectral_exchange()
    f6_seeds()
    f7_fss()
    f8_input_resolution()
    print("done")


if __name__ == "__main__":
    main()
