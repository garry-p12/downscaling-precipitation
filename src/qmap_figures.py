"""Figures for the quantile-mapping experiments.

Quantile mapping replaces each predicted value with the reference value at the
same rank, using a map fitted on the dev year alone. It is monotone, so no cell
changes its position in the ordering -- only the numbers move. That makes it a
pure distribution correction and a clean test of a specific claim: that the
products' detection failure is partly a calibration problem rather than a
placement one.

The result is that it depends entirely on how miscalibrated the product was to
begin with, which is what these figures show.

    python -m src.qmap_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .publication_figures import (BLUE, GREEN, GREY, GREY_L, ORANGE, VERM,
                                  W1, W2, house_style, panel)

OUT = Path("results/figures/qmap")
STYLE = {"XGBoost": (BLUE, "D"), "CNN": (GREEN, "^"), "Swin": (ORANGE, "v")}


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  {name:28s} {caption}")


def Q1_transfer(curves):
    """The fitted maps. Distance above the 1:1 line is how much each value is inflated."""
    fig, ax = plt.subplots(figsize=(W1 * 1.5, W1 * 1.5))
    lo, hi = 0.3, 300
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", label="no change", zorder=1)
    for n, (c, m) in STYLE.items():
        if n not in curves:
            continue
        s = np.array(curves[n]["src"]); d = np.array(curves[n]["dst"])
        ax.plot(s, d, color=c, lw=1.4, label=n, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("original predicted value (mm)")
    ax.set_ylabel("after quantile mapping (mm)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, "Q1_transfer_function", "the fitted maps")


def Q2_gain_vs_bias(S):
    """What the correction buys, against how miscalibrated the product was.

    The x axis is the product's frequency bias before mapping -- how often it
    forecast heavy rain relative to how often heavy rain happened. 1.0 is
    correct. The further below 1.0 a product starts, the more quantile mapping
    has to fix and the more detection it recovers; a product that was already
    calibrated gains nothing and pays the RMSE cost regardless.
    """
    pairs = [(n, f"{n}_QM") for n in STYLE if n in S and f"{n}_QM" in S]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))

    ax = axes[0]
    for n, q in pairs:
        c, m = STYLE[n]
        x = S[n]["frequency_bias"]
        ax.scatter(x, S[q]["csi"] - S[n]["csi"], color=c, marker=m, s=52,
                   zorder=5, edgecolor="white", linewidth=0.6)
        ax.annotate(n, (x, S[q]["csi"] - S[n]["csi"]), textcoords="offset points",
                    xytext=(7, 3), fontsize=6, color="#222222")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(1.0, color=GREY_L, lw=0.8, ls=":")
    ax.text(1.005, ax.get_ylim()[1] * 0.92, "already\ncalibrated", fontsize=5.4, color=GREY)
    ax.set_xlabel("frequency bias before mapping")
    ax.set_ylabel("CSI gained")
    ax.margins(x=0.22)

    ax = axes[1]
    x = np.arange(len(pairs)); w = 0.36
    ax.bar(x - w / 2, [S[q]["csi"] - S[n]["csi"] for n, q in pairs], width=w,
           color=GREEN, label="CSI gained")
    ax.bar(x + w / 2, [S[q]["rmse"] - S[n]["rmse"] for n, q in pairs], width=w,
           color=VERM, label="RMSE cost (mm)")
    for i, (n, q) in enumerate(pairs):
        for dx, v in ((-w / 2, S[q]["csi"] - S[n]["csi"]), (w / 2, S[q]["rmse"] - S[n]["rmse"])):
            ax.annotate(f"{v:+.3f}", (i + dx, v), xytext=(0, 3 if v >= 0 else -9),
                        textcoords="offset points", ha="center", fontsize=5.6, color="#222222")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nbias {S[n]['frequency_bias']:.2f}" for n, _ in pairs], fontsize=6)
    ax.set_ylabel("change")
    ax.legend(frameon=False, fontsize=6)
    ax.margins(y=0.25)
    for ax, L in zip(axes, "ab"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "Q2_gain_vs_bias", "the gain tracks the miscalibration")


def Q3_where_it_acts(S):
    """Does mapping change the coarse field or the sub-block detail?

    It is a per-cell monotone transform, so it cannot move rain between cells
    and should act entirely on the distribution. Splitting its effect confirms
    that: the 10 km term moves and the sharpening term does not, which places
    quantile mapping on the same side of the budget as everything else that
    has worked.
    """
    pairs = [(n, f"{n}_QM") for n in STYLE if n in S and f"{n}_QM" in S]
    fig, ax = plt.subplots(figsize=(W1 * 1.7, 2.3))
    x = np.arange(len(pairs)); w = 0.36
    ax.bar(x - w / 2, [S[q]["rmse_10km"] - S[n]["rmse_10km"] for n, q in pairs],
           width=w, color=BLUE, label="change in the 10 km field")
    ax.bar(x + w / 2, [S[q]["rmse_downscale"] - S[n]["rmse_downscale"] for n, q in pairs],
           width=w, color=ORANGE, label="change in the 10 km to 1 km step")
    for i, (n, q) in enumerate(pairs):
        for dx, v in ((-w / 2, S[q]["rmse_10km"] - S[n]["rmse_10km"]),
                      (w / 2, S[q]["rmse_downscale"] - S[n]["rmse_downscale"])):
            ax.annotate(f"{v:+.3f}", (i + dx, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=5.8, color="#222222")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in pairs])
    ax.set_ylabel("change in RMSE (mm day$^{-1}$)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.margins(y=0.3)
    fig.tight_layout()
    save(fig, "Q3_where_it_acts", "coarse field vs sharpening")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/qmap/scores.json")
    ap.add_argument("--curves", default="results/qmap/curves.json")
    a = ap.parse_args(argv)
    house_style()
    S = {k: v for k, v in json.load(open(a.scores)).items() if not k.startswith("_")}
    C = {k: v for k, v in json.load(open(a.curves)).items() if not k.startswith("_")}
    print("quantile-mapping figures:")
    Q1_transfer(C); Q2_gain_vs_bias(S); Q3_where_it_acts(S)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
