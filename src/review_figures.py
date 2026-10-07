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
         "Diffusion": (VERM, "P"), "Stacked": (PURPLE, "X"), "Bilinear": (GREY, "s"),
         # Colorado carries the spectral-penalty variant instead of Swin/diffusion.
         "CNN+E": (ORANGE, "*")}


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
    ap.add_argument("--gauge", default=None, help="a gauge_events JSON, for R5")
    ap.add_argument("--out-dir", default=None, help="override the figure directory")
    a = ap.parse_args(argv)
    global OUT
    if a.out_dir:
        OUT = Path(a.out_dir)
    house_style()
    A = json.load(open(a.analysis))
    print(f"figures from {a.analysis} ({A['n_days']} test days, {A['period'][0]}..{A['period'][1]})")

    if a.scores:
        raw = json.load(open(a.scores))
        scores = {k: v for k, v in raw.items() if not k.startswith("_")}
        names = list(scores)
        R1_metric_comparison(A, scores, names)
        R6_performance_diagram(scores, names)
        R7_fss_scale(scores, names)
        R8_error_budget(scores, names)
        R9_quantile_quantile(A, names)
        R10_metric_heatmap(scores, names)
    R2_intensity_distribution(A)
    R3_storm_events(A)
    R4_seasonal(A)
    if a.gauge:
        G = json.load(open(a.gauge))
        R5_gauge_events(G, ["AORC"] + [n for n in STYLE if n in G["summary"]])
    print("done ->", OUT)




def R5_gauge_events(G, out_names):
    """Independent gauge check: peak capture per storm, and overall error.

    AORC is plotted alongside the products, not as a reference line, because
    the comparison that matters is whether the gap between a product and the
    gauges is larger than the gap between AORC and the gauges. Where AORC sits
    near 1.0 and the products near 0.2, the shortfall is the model's and not an
    artefact of scoring against an analysis.
    """
    ev = G["events"]
    days = list(ev)
    names = [n for n in out_names if n in G["summary"]]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))

    ax = axes[0]
    x = np.arange(len(days)); w = 0.8 / len(names)
    for i, n in enumerate(names):
        c = "black" if n == "AORC" else STYLE.get(n, (GREY, "o"))[0]
        ax.bar(x + i * w - 0.4 + w / 2, [ev[d][n]["peak_ratio"] for d in days],
               width=w, color=c, label=n)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(days, rotation=45, ha="right", fontsize=5.4)
    ax.set_ylabel("peak / gauge peak")
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=5.6, ncol=3, loc="upper center", handlelength=1.1)

    ax = axes[1]
    vals = [G["summary"][n]["median_rmse"] for n in names]
    cols = ["black" if n == "AORC" else STYLE.get(n, (GREY, "o"))[0] for n in names]
    ax.bar(range(len(names)), vals, color=cols, width=0.65)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("median per-station RMSE (mm day$^{-1}$)")
    ax.set_ylim(0, max(vals) * 1.18)
    for ax, L in zip(axes, "ab"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "R5_gauge_events", f"gauge check, {G['n_stations']} stations")

def R6_performance_diagram(scores, names):
    """Roebber (2009) performance diagram: the standard categorical-verification plot.

    Success ratio against POD, with CSI as curved contours and frequency bias as
    rays from the origin. It earns its place over four separate bar charts
    because POD, FAR, CSI and bias are not independent -- a product can only move
    along the bias ray it sits on without changing how often it forecasts the
    event. Perfect is the top-right corner.

    Axes are cropped to the cluster. Over the full unit square these five points
    occupy a few percent of the panel and the CSI contours between them are
    unreadable, which defeats the purpose of the diagram; the inset shows where
    the crop sits.
    """
    sr = np.array([1.0 - scores[n]["far"] for n in names])
    pod = np.array([scores[n]["pod"] for n in names])
    pad = 0.12
    x0, x1 = max(0.0, sr.min() - pad), min(1.0, sr.max() + pad)
    y0, y1 = max(0.0, pod.min() - pad), min(1.0, pod.max() + pad)

    fig, ax = plt.subplots(figsize=(W1 * 1.5, W1 * 1.5))
    # Evaluate the CSI field over the cropped window, not the unit square:
    # clabel places labels along the contour it was given, so a full-range
    # contour set puts every label outside the view and the curves arrive bare.
    gx = np.linspace(max(x0, 1e-3), x1, 400)
    gy = np.linspace(max(y0, 1e-3), y1, 400)
    SR, POD = np.meshgrid(gx, gy)
    csi = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    lv = np.round(np.arange(np.floor(csi.min() * 20) / 20,
                            csi.max() + 0.025, 0.025), 3)
    cs = ax.contour(SR, POD, csi, levels=lv, colors=GREY_L, linewidths=0.6, zorder=1)
    ax.clabel(cs, inline=True, fontsize=5.0, fmt="CSI %.3g")
    for b in (0.6, 0.8, 1.0, 1.2, 1.5):
        ax.plot([0, 1], [0, b], color=GREY_L, lw=0.5, ls=":", zorder=1)
        # Label each ray where it leaves the cropped panel, not at the unit square.
        xe = min(x1, y1 / b)
        ax.annotate(f"bias {b:g}", (xe, b * xe), fontsize=5.2, color=GREY,
                    ha="right", va="bottom", rotation=np.degrees(np.arctan(b)) * 0.75)

    # Stagger labels around each marker so none sits on top of another.
    order = np.argsort(-pod)
    offs = [(7, 5), (7, -9), (-7, 6), (-7, -10), (7, 5), (-7, 6)]
    for rank, i in enumerate(order):
        n = names[i]
        c, m = STYLE.get(n, (GREY, "o"))
        ax.scatter(sr[i], pod[i], color=c, marker=m, s=54, zorder=5,
                   edgecolor="white", linewidth=0.7)
        dx, dy = offs[rank % len(offs)]
        ax.annotate(n, (sr[i], pod[i]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.2, color="#222222", zorder=6,
                    ha="left" if dx > 0 else "right")

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("success ratio (1 $-$ FAR)")
    ax.set_ylabel("probability of detection")
    ax.set_aspect("equal")

    # Locator inset: the full unit square with the cropped region marked.
    ins = ax.inset_axes([0.69, 0.045, 0.29, 0.29])
    gu = np.linspace(0.001, 1, 300)
    SRu, PODu = np.meshgrid(gu, gu)
    csiu = 1.0 / (1.0 / SRu + 1.0 / PODu - 1.0)
    ins.contour(SRu, PODu, csiu, levels=[0.2, 0.4, 0.6, 0.8], colors=GREY_L, linewidths=0.4)
    ins.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="#222222", lw=0.8))
    ins.set_xlim(0, 1); ins.set_ylim(0, 1); ins.set_aspect("equal")
    ins.set_xticks([0, 1]); ins.set_yticks([0, 1])
    ins.tick_params(labelsize=4.5, length=1.5, pad=1)
    fig.tight_layout()
    save(fig, "R6_performance_diagram", "POD / SR / CSI / bias in one plot")


def R7_fss_scale(scores, names):
    """Fractions skill score against neighbourhood width (Roberts & Lean, 2008).

    A point-wise score double-penalises a field that is right but displaced. FSS
    relaxes the match over a window, so the curve shows the scale at which each
    product becomes useful rather than asserting a single number.
    """
    scales = [1, 5, 15, 41]
    fig, ax = plt.subplots(figsize=(W1 * 1.4, 2.2))
    for n in names:
        v = scores[n]
        y = [v.get(f"fss{k}") for k in scales]
        if any(q is None for q in y):
            continue
        c, m = STYLE.get(n, (GREY, "o"))
        ax.plot(scales, y, color=c, marker=m, ms=4, lw=1.2, label=n)
    ax.set_xscale("log")
    ax.set_xticks(scales); ax.set_xticklabels([f"{k}" for k in scales])
    ax.set_xlabel("neighbourhood width (1 km cells)")
    ax.set_ylabel("FSS, >30 mm")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "R7_fss_scale", "skill against matching scale")


def R8_error_budget(scores, names):
    """Where the error lives: the coarse field, or the 10 km to 1 km step.

    Stacked in variance because that is how independent error sources add; the
    bars are the square roots, so their heights read in mm. The point is the
    near-identical blue block -- every product inherits essentially the same
    coarse error, and they differ almost entirely in a term that is small.
    """
    names = [n for n in names if "rmse_10km" in scores[n]]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))

    ax = axes[0]
    x = np.arange(len(names))
    c10 = [scores[n]["rmse_10km"] for n in names]
    dwn = [scores[n]["rmse_downscale"] for n in names]
    tot = [scores[n]["rmse"] for n in names]
    ax.bar(x, c10, width=0.6, color=BLUE, label="coarse field (10 km)")
    # Offset by a 2 px-equivalent gap so the two blocks read as separate marks.
    ax.bar(x, np.array(tot) - np.array(c10), bottom=np.array(c10) + 0.012,
           width=0.6, color=ORANGE, label="10 km $\\rightarrow$ 1 km step")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("RMSE (mm day$^{-1}$)")
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    ax = axes[1]
    for n in names:
        c, m = STYLE.get(n, (GREY, "o"))
        ax.scatter(scores[n]["rmse_10km"], scores[n]["rmse_downscale"],
                   color=c, marker=m, s=46, zorder=5, edgecolor="white", linewidth=0.6)
        ax.annotate(n, (scores[n]["rmse_10km"], scores[n]["rmse_downscale"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6, color="#222222")
    ax.set_xlabel("coarse-field error (mm day$^{-1}$)")
    ax.set_ylabel("downscaling error (mm day$^{-1}$)")
    # Equal spans on both axes, so the eye compares the two terms honestly
    # rather than through two different zooms.
    xs = [scores[n]["rmse_10km"] for n in names]
    ys = [scores[n]["rmse_downscale"] for n in names]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.9 + 1e-6
    ax.set_xlim(np.mean(xs) - span / 2, np.mean(xs) + span / 2)
    ax.set_ylim(np.mean(ys) - span / 2, np.mean(ys) + span / 2)
    for ax, L in zip(axes, "ab"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "R8_error_budget", "coarse vs downscaling error")


def R9_quantile_quantile(A, names):
    """Q-Q against AORC: predicted quantile versus observed, on log axes.

    R2 shows the distributions; this shows the *mapping* between them, where a
    systematic shortfall is a departure from the 1:1 line rather than a gap
    between two curves the eye has to subtract.
    """
    D = A["distribution"]
    qs = ["0.5", "0.9", "0.99", "0.999", "0.9999"]
    obs = [D["AORC"]["quantiles"][q] for q in qs]
    fig, ax = plt.subplots(figsize=(W1 * 1.35, W1 * 1.35))
    lo, hi = 0.6 * min(obs), 1.6 * max(obs)
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", zorder=1, label="1:1")
    for n in names:
        if n not in D:
            continue
        c, m = STYLE.get(n, (GREY, "o"))
        ax.plot(obs, [D[n]["quantiles"][q] for q in qs], color=c, marker=m,
                ms=4.5, lw=1.0, label=n, zorder=4)
    for o, q in zip(obs, qs):
        ax.annotate(f"q{float(q) * 100:g}".rstrip("0").rstrip("."), (o, lo * 1.08),
                    fontsize=5.2, color=GREY, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("AORC quantile (mm)"); ax.set_ylabel("predicted quantile (mm)")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, "R9_quantile_quantile", "predicted vs observed quantiles")


def R10_metric_heatmap(scores, names):
    """Every metric at once, each normalised to its own best performer.

    A table of raw numbers cannot be scanned; this keeps all of them visible and
    makes the pattern -- no product wins everywhere -- immediate. Cells carry the
    raw value, so the normalisation never hides the number it came from.
    """
    rows = [("rmse", "RMSE", False), ("kge", "KGE", True), ("pod", "POD", True),
            ("far", "FAR", False), ("csi", "CSI", True),
            ("frequency_bias", "freq. bias", None), ("fss5", "FSS@5", True),
            ("fss41", "FSS@41", True), ("spectral_ratio_sub10km", "texture", None)]
    M = np.zeros((len(rows), len(names)))
    raw = np.zeros_like(M)
    for i, (k, _, higher) in enumerate(rows):
        v = np.array([scores[n][k] for n in names])
        raw[i] = v
        if higher is None:          # 1.0 is the target, not more or less
            d = np.abs(v - 1.0)
            M[i] = 1.0 - (d - d.min()) / (np.ptp(d) + 1e-12)
        elif higher:
            M[i] = (v - v.min()) / (np.ptp(v) + 1e-12)
        else:
            M[i] = (v.max() - v) / (np.ptp(v) + 1e-12)
    fig, ax = plt.subplots(figsize=(W1 * 1.6, 2.9))
    # Sequential, single hue, light to dark: magnitude, not identity.
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows])
    for i in range(len(rows)):
        for j in range(len(names)):
            ax.text(j, i, f"{raw[i, j]:.3g}", ha="center", va="center", fontsize=5.6,
                    color="white" if M[i, j] > 0.55 else "#222222")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("normalised within row (1 = best)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    save(fig, "R10_metric_heatmap", "all metrics, normalised per row")


if __name__ == "__main__":
    main()
