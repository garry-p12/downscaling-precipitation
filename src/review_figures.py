"""Figures for the review questions: model comparison, distributions, storms, seasons.

Reuses the publication house style so these sit alongside F1-F8 without
restyling. Writes to results/figures/review as PDF plus 600 dpi PNG.

    python -m src.review_figures --analysis results/review/review_analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .publication_figures import (BLUE, GREEN, GREY, GREY_L, ORANGE, PURPLE, VERM,
                                  W1, W2, house_style, panel)

OUT = Path("results/figures/review")

STYLE = {"XGBoost": (BLUE, "D"), "CNN": (GREEN, "^"), "Swin": (ORANGE, "v"),
         "Diffusion": (VERM, "P"), "Stacked": (PURPLE, "X"), "Bilinear": (GREY, "s")}


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  {name:34s} {caption}")


def R1_metric_comparison(A, scores, out_names):
    """Headline metrics per product, with seed spread where it exists.

    The point of the error bars is that several of these differences are inside
    the seed spread and should not be read as a ranking.
    """
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0))
    panels = [("rmse", "RMSE (mm day$^{-1}$)", False),
              ("pod", "POD, >30 mm", True),
              ("csi", "CSI, >30 mm", True),
              ("spectral_ratio_sub10km", "sub-10 km power ratio", True)]
    for ax, (key, label, higher) in zip(axes, panels):
        names = [n for n in out_names if n in scores and key in scores[n]]
        vals = [scores[n][key] for n in names]
        cols = [STYLE.get(n, (GREY, "o"))[0] for n in names]
        err = [scores[n].get(f"{key}_sd", 0.0) for n in names]
        ax.bar(range(len(names)), vals, color=cols, width=0.68,
               yerr=err if any(err) else None, capsize=2,
               error_kw=dict(lw=0.7, ecolor="#333333"))
        if key == "spectral_ratio_sub10km":
            # 1.0 is correct texture, not "more is better".
            ax.axhline(1.0, color=GREY, lw=0.8, ls="--", zorder=0)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.margins(x=0.08)
    for ax, L in zip(axes, "abcd"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "R1_metric_comparison", "headline metrics per product")


def R2_intensity_distribution(A):
    """PDF and exceedance curve against AORC.

    Panel (b) is the one that matters: on a log exceedance axis the upper tail
    is visibly short for every product, which is what a squared-error fit does
    and what no aggregate RMSE reveals.
    """
    D = A["distribution"]
    bins = np.array(D["AORC"]["bins"])
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))

    ax = axes[0]
    ax.step(ctr, np.array(D["AORC"]["pdf"][:len(ctr)]), where="mid",
            color="black", lw=1.4, label="AORC (truth)", zorder=5)
    for n in STYLE:
        if n in D:
            c = STYLE[n][0]
            ax.step(ctr, np.array(D[n]["pdf"][:len(ctr)]), where="mid", color=c, lw=0.9, label=n)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("daily rainfall (mm)"); ax.set_ylabel("fraction of cells")
    ax.set_xlim(0.5, 500); ax.legend(frameon=False, fontsize=5.6, loc="lower left")

    ax = axes[1]
    ax.plot(bins, D["AORC"]["exceedance"], color="black", lw=1.4, label="AORC", zorder=5)
    for n in STYLE:
        if n in D:
            ax.plot(bins, D[n]["exceedance"], color=STYLE[n][0], lw=0.9, label=n)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("threshold (mm)"); ax.set_ylabel("P(rain $\\geq$ threshold)")
    ax.set_xlim(1, 500); ax.set_ylim(1e-7, 1)
    ax.axvline(30, color=GREY_L, lw=0.8, ls=":", zorder=0)
    ax.text(31, 2e-7, "30 mm", fontsize=5.5, color=GREY)
    for ax, L in zip(axes, "ab"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "R2_intensity_distribution", "PDF and exceedance vs AORC")


def R3_storm_events(A):
    """Per-event peak capture and RMSE, areal versus localised.

    Split by event type because they fail differently: widespread days keep most
    of the peak, isolated convective days lose three quarters of it.
    """
    ev = A["event_scores"]
    days = list(ev)
    prods = [p for p in STYLE if p in ev[days[0]]["products"]]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))

    ax = axes[0]
    x = np.arange(len(days)); w = 0.8 / len(prods)
    for i, p in enumerate(prods):
        v = [ev[d]["products"][p]["peak_ratio"] for d in days]
        ax.bar(x + i * w - 0.4 + w / 2, v, width=w, color=STYLE[p][0], label=p)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{ev[d]['kind']}" for d in days], rotation=45, ha="right", fontsize=5.4)
    ax.set_ylabel("predicted peak / observed peak")
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), handlelength=1.1, columnspacing=0.9)

    ax = axes[1]
    for p in prods:
        xs = [ev[d]["obs_max"] for d in days]
        ys = [ev[d]["products"][p]["peak_ratio"] for d in days]
        ax.scatter(xs, ys, color=STYLE[p][0], marker=STYLE[p][1], s=22, label=p, zorder=3)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("observed event peak (mm)")
    ax.set_ylabel("predicted peak / observed peak")
    ax.set_ylim(0, 1.15)
    for ax, L in zip(axes, "ab"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "R3_storm_events", "peak capture by event")


def R4_seasonal(A):
    """RMSE and POD by season, with the observed mean as context."""
    S = A["seasonal"]
    seasons = [s for s in ("DJF", "MAM", "JJA", "SON") if s in S]
    prods = [p for p in STYLE if p in S[seasons[0]]["products"]]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.1))

    ax = axes[0]
    ax.bar(seasons, [S[s]["obs_mean"] for s in seasons], color=GREY_L, width=0.6)
    ax.set_ylabel("observed mean (mm day$^{-1}$)")

    for ax, key, label in ((axes[1], "rmse", "RMSE (mm day$^{-1}$)"),
                           (axes[2], "pod", "POD, >30 mm")):
        for p in prods:
            ax.plot(seasons, [S[s]["products"][p][key] for s in seasons],
                    color=STYLE[p][0], marker=STYLE[p][1], ms=3.2, lw=1.0, label=p)
        ax.set_ylabel(label)
    axes[2].legend(frameon=False, fontsize=5.6, ncol=2)
    for ax, L in zip(axes, "abc"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "R4_seasonal", "skill by season")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", default="results/review/review_analysis.json")
    ap.add_argument("--scores", default=None,
                    help="a score_variants JSON, for the R1 metric panel")
    a = ap.parse_args(argv)
    house_style()
    A = json.load(open(a.analysis))
    print(f"figures from {a.analysis} ({A['n_days']} test days, {A['period'][0]}..{A['period'][1]})")

    if a.scores:
        raw = json.load(open(a.scores))
        scores = {k: v for k, v in raw.items() if not k.startswith("_")}
        R1_metric_comparison(A, scores, list(scores))
    R2_intensity_distribution(A)
    R3_storm_events(A)
    R4_seasonal(A)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
