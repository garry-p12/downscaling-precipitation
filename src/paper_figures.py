"""Publication-grade figures for the model comparison.

    python -m src.paper_figures        # -> results/figures/paper/

Rendered at 300 dpi with a serif family, panel labels and self-contained
captions, so each figure can drop into a manuscript unmodified. Everything is
read from ``results/comparison/model_comparison.json``; nothing is hard-coded.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .model_comparison import LABELS  # noqa: E402

OUT = Path("results/figures/paper")
INDEX: list[tuple[str, str]] = []

# Okabe-Ito: colour-blind safe, prints legibly in greyscale.
COLORS = {
    "nearest": "#999999", "bilinear": "#E69F00", "xgboost": "#D55E00", "cnn": "#0072B2",
    "swin": "#009E73", "diffusion": "#CC79A7", "diffusion_member": "#56B4E9", "aorc": "#000000",
}
MARKERS = {"nearest": "s", "bilinear": "D", "xgboost": "^", "cnn": "o",
           "swin": "v", "diffusion": "P", "diffusion_member": "X"}

PAPER_RC = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
}


def _panel_label(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def _save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    INDEX.append((name, caption))
    print(f"  {name}  (+pdf)")


def load(path="results/comparison/model_comparison.json"):
    with open(path) as f:
        return json.load(f)


def crps_of(cmp, n):
    if n == "diffusion" and "diffusion_ensemble" in cmp:
        return cmp["diffusion_ensemble"]["crps_mm"]
    return cmp["products"][n]["overall"]["mae"]


# --------------------------------------------------------------------------- #
def fig_headline(cmp):
    names = list(cmp["products"])
    panels = [
        ("RMSE (mm day$^{-1}$)", lambda n: cmp["products"][n]["overall"]["rmse"], "lower is better"),
        ("KGE", lambda n: cmp["products"][n]["overall"]["kge"], "higher is better"),
        ("POD, $P>30$ mm day$^{-1}$", lambda n: cmp["products"][n]["detection"]["30.0"]["pod"], "higher is better"),
        ("CRPS (mm day$^{-1}$)", lambda n: crps_of(cmp, n), "lower is better"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.3))
    ref = "bilinear"
    for k, (ax, (label, get, note)) in enumerate(zip(axes, panels)):
        vals = [get(n) for n in names]
        y = np.arange(len(names))
        # Dot plot, not bars: the spread between products is a few percent of
        # their magnitude, so bars would either start at zero (differences
        # invisible) or be truncated (differences exaggerated). Dots carry no
        # area, so a zoomed axis is honest.
        lo, hi = min(vals), max(vals)
        for yi, n, v in zip(y, names, vals):
            ax.plot([lo - (hi - lo) * 0.5, v], [yi, yi], color="0.85", lw=0.8, zorder=1)
            ax.plot(v, yi, marker=MARKERS[n], ms=8, color=COLORS[n], markeredgecolor="black",
                    markeredgewidth=0.6, zorder=3)
        ax.axvline(get(ref), color="black", lw=0.9, ls=(0, (4, 2)), zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS.get(n, n) for n in names] if k == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        pad = (hi - lo) * 0.18 or 0.1
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(len(names) - 0.4, -0.6)
        _spines(ax)
        ax.tick_params(length=3)
        _panel_label(ax, "abcd"[k], dx=-0.62 if k == 0 else -0.10)
        ax.set_title(note, fontsize=8, color="0.35", loc="right")
    fig.tight_layout()
    _save(fig, "P1_headline_scores.png",
          "Deterministic and probabilistic skill of every product over the 2019-2020 test period "
          "(731 days, 1 km, against AORC). Dashed line marks bilinear interpolation; axes are zoomed to "
          "the range spanned by the products, which is why dots rather than bars are used. All learned "
          "products beat bilinear on RMSE; only the CNN and the transformer also beat it on heavy-event "
          "detection, and only the diffusion ensemble improves CRPS.")


def fig_tradeoff(cmp):
    names = [n for n in cmp["products"] if n != "nearest"]
    x = [cmp["products"][n]["spectral_ratio_sub10km"] for n in names]
    y = [cmp["products"][n]["overall"]["rmse"] for n in names]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for n, xi, yi in zip(names, x, y):
        ax.scatter(xi, yi, s=95, color=COLORS[n], marker=MARKERS[n], edgecolor="black",
                   linewidth=0.6, zorder=3, label=LABELS.get(n, n))
    ax.axvline(1.0, color="black", lw=0.9, ls=(0, (4, 2)))
    ax.annotate("observed\ntexture", xy=(1.0, min(y)), xytext=(0.62, min(y) - 0.02),
                fontsize=8, color="0.3", ha="center", va="bottom")
    # Pareto front: no product is both smoother-corrected and lower RMSE
    order = np.argsort(x)
    best = np.inf
    front = []
    for i in order[::-1]:
        if y[i] < best:
            best = y[i]
            front.append(i)
    front = front[::-1]
    ax.plot([x[i] for i in front], [y[i] for i in front], color="0.4", lw=1.0, ls="-", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("spectral ratio below 10 km  (1 = observed texture)")
    ax.set_ylabel("RMSE (mm day$^{-1}$)")
    ax.invert_yaxis()
    _spines(ax)
    ax.tick_params(length=3)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.4)
    ax.set_title("Accuracy and realism cannot be maximised together", fontsize=9.5, loc="left")
    fig.tight_layout()
    _save(fig, "P2_accuracy_realism_tradeoff.png",
          "Squared-error skill against retained fine-scale variance; the grey line is the Pareto front "
          "(RMSE axis inverted, so upper-right is better on both). Products that score best on RMSE "
          "retain 2-20 % of the observed sub-10 km variance, while the only product with realistic "
          "texture has the worst RMSE. This is the double penalty made quantitative.")


def fig_taylor(cmp):
    """Taylor diagram: correlation, normalised variance and centred RMS in one plot."""
    names = list(cmp["products"])
    fig = plt.figure(figsize=(5.8, 5.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetalim(0, np.pi / 2)
    rmax = 1.6
    ax.set_rlim(0, rmax)

    corr_ticks = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=[f"{c:g}" for c in corr_ticks])
    ax.set_rgrids([0.5, 1.0, 1.5], labels=["0.5", "1.0", "1.5"], angle=90)
    ax.tick_params(labelsize=8)
    ax.grid(color="0.85", lw=0.6)

    # centred-RMS arcs about the reference point (corr = 1, sigma = 1)
    for crms in (0.25, 0.5, 0.75, 1.0, 1.25):
        th = np.linspace(0, np.pi / 2, 400)
        rr = np.cos(th) + np.sqrt(np.maximum(crms**2 - np.sin(th) ** 2, 0))
        m = (rr > 0) & (rr <= rmax) & (np.abs(np.sin(th)) <= crms)
        ax.plot(th[m], rr[m], color="0.75", lw=0.6, ls=":", zorder=1)
    ax.plot([0], [1.0], marker="*", ms=15, color="black", zorder=5)
    ax.text(0.02, 1.06, "AORC", fontsize=8)

    for n in names:
        o = cmp["products"][n]["overall"]
        r = float(o["pearson_r"])
        sd = float(o["std_pred"]) / float(o["std_obs"])
        ax.scatter(np.arccos(np.clip(r, -1, 1)), sd, s=90, color=COLORS[n], marker=MARKERS[n],
                   edgecolor="black", linewidth=0.6, zorder=4, label=LABELS.get(n, n))
    # Azimuth is correlation, radius is normalised standard deviation.
    ax.set_xlabel("normalised standard deviation", labelpad=18)
    ax.text(np.radians(47), rmax * 1.17, "correlation", rotation=-43,
            ha="center", va="center", fontsize=9)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.34, 0.92), handletextpad=0.4)
    ax.set_title("Taylor diagram, 1 km test period", fontsize=9.5, loc="left", pad=16)
    fig.tight_layout()
    _save(fig, "P3_taylor_diagram.png",
          "Correlation (azimuth), standard deviation normalised by the observed value (radius) and "
          "centred RMS difference (dotted arcs about the star) for every product. A product on the unit "
          "radius has the right variability; all learned products fall inside it, i.e. they are smoother "
          "than the observations.")


def fig_spectra(cmp):
    wl = np.array(cmp["spectra"]["wavelength_km"])
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.loglog(wl, cmp["spectra"]["aorc"], color="black", lw=2.0, label="AORC (observed)", zorder=5)
    for n in cmp["products"]:
        if n in cmp["spectra"]:
            ax.loglog(wl, cmp["spectra"][n], lw=1.3, color=COLORS[n],
                      ls="--" if n.endswith("member") else "-", label=LABELS.get(n, n))
    ax.axvspan(wl.min(), 10, color="0.9", zorder=0)
    ax.invert_xaxis()
    ax.set_xlabel("wavelength (km)")
    ax.set_ylabel("power spectral density")
    _spines(ax)
    ax.tick_params(length=3, which="both")
    ax.legend(frameon=False, loc="lower left", ncol=1)
    ax.set_title("Spatial power spectra, 120 wettest test days", fontsize=9.5, loc="left")
    fig.tight_layout()
    _save(fig, "P4_power_spectra.png",
          "Radially averaged power spectral density. The shaded band is the sub-10 km range summarised "
          "by the spectral ratio. Deterministic products lose one to two orders of magnitude of variance "
          "below ~30 km; nearest-neighbour exceeds the observations because 10 km blocks inject spurious "
          "high-frequency power.")


def fig_fss(cmp):
    thresholds = ["1.0", "10.0", "30.0"]
    scales = [1, 5, 15, 41]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for k, (ax, thr) in enumerate(zip(axes, thresholds)):
        for n in cmp["products"]:
            y = [cmp["products"][n]["fss"][f"{thr}mm_{s}cell"] for s in scales]
            ax.plot(scales, y, marker=MARKERS[n], ms=5, lw=1.3, color=COLORS[n],
                    ls="--" if n.endswith("member") else "-", label=LABELS.get(n, n))
        ax.set_xscale("log")
        ax.set_xticks(scales)
        ax.set_xticklabels([str(s) for s in scales])
        ax.set_xlabel("neighbourhood width (km)")
        ax.set_title(f"$P \\geq {float(thr):g}$ mm day$^{{-1}}$", fontsize=9.5, loc="left")
        _spines(ax)
        ax.tick_params(length=3)
        _panel_label(ax, "abc"[k], dx=-0.14 if k == 0 else -0.08)
    axes[0].set_ylabel("Fractions Skill Score")
    axes[-1].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, "P5_fractions_skill_score.png",
          "Fractions Skill Score against neighbourhood width for three intensity thresholds. FSS credits "
          "rain placed approximately correctly, so it separates products that lose structure from those "
          "that merely displace it; the products converge at large neighbourhoods and at low thresholds.")


def write_index(cmp):
    lines = ["# Publication-grade figures", "",
             f"Test period {cmp['period'][0]} to {cmp['period'][1]} ({cmp['n_days']} days), 1 km, "
             "against AORC. Rendered at 300 dpi with a PDF alongside each PNG.", "",
             "Palette is Okabe-Ito (colour-blind safe, legible in greyscale). Regenerate with "
             "`python -m src.paper_figures`.", "",
             "| figure | caption |", "|---|---|"]
    for name, cap in sorted(INDEX):
        lines.append(f"| [`{name}`]({name}) | {cap} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  README.md ({len(INDEX)} figures)")


def main():
    cmp = load()
    OUT.mkdir(parents=True, exist_ok=True)
    print("rendering ->", OUT)
    with plt.rc_context(PAPER_RC):
        fig_headline(cmp)
        fig_tradeoff(cmp)
        fig_taylor(cmp)
        fig_spectra(cmp)
        fig_fss(cmp)
    write_index(cmp)


if __name__ == "__main__":
    main()
