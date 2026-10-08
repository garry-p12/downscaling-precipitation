"""Figures for the attribution result: where does the skill actually come from?

The finding these exist to show is awkward to state in a table, because the
sharpening step's effect is *statistically significant and practically
negligible* at the same time. A figure that only asks "is it non-zero" says
yes and misleads. Every panel here therefore plots the sharpening gain beside
something that sets its scale -- the coarse correction, or the headroom a
perfect sharpener would have.

    python -m src.attribution_figures
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

OUT = Path("results/figures/attribution")


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  {name:32s} {caption}")


def load():
    j = lambda p: json.load(open(p))
    A, AI = j("results/review/baseline_scores.json"), j("results/review/isit_useful.json")
    P, PB = j("results/power/profit_scores.json"), j("results/power/bilinear_score.json")
    O = j("results/oracle_austin_scores.json")
    return {
        "IMERG 0.1$\\degree$": dict(bilinear=A["bilinear"], blocky=AI["XGB_blocky"],
                                    model=A["XGBoost"], oracle=O["orc12_ctl"]["rmse"]),
        "POWER 0.5$\\degree$": dict(bilinear=PB["bilinear"], blocky=P["XGB_blocky"],
                                    model=P["XGBoost"], oracle=O["orc60_ctl"]["rmse"]),
    }


def D1_ladder(D):
    """Every rung between plain interpolation and a perfect input.

    The two bars that matter are the ones that nearly touch: holding the
    model's own coarse field flat in blocks, and the model's full 1 km output.
    The distance between them is everything the sharpening step is worth. The
    distance to the oracle is what a perfect sharpener would have reached.
    """
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=False)
    for ax, (lbl, d) in zip(axes, D.items()):
        rows = [("bilinear", d["bilinear"]["rmse"], GREY),
                ("model, blocky", d["blocky"]["rmse"], ORANGE),
                ("model, 1 km", d["model"]["rmse"], BLUE),
                ("perfect input", d["oracle"], GREEN)]
        y = np.arange(len(rows))[::-1]
        for yy, (n, v, c) in zip(y, rows):
            ax.barh(yy, v, color=c, height=0.62)
            ax.annotate(f"{v:.3f}", (v, yy), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=6, color="#222222")
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel("RMSE (mm day$^{-1}$)")
        ax.set_title(lbl, fontsize=7)
        ax.set_xlim(0, max(r[1] for r in rows) * 1.22)
        # Brace the gap the sharpening actually buys, against the gap to perfect.
        gap = d["blocky"]["rmse"] - d["model"]["rmse"]
        ax.annotate(f"sharpening buys {gap:.3f} mm",
                    xy=(d["model"]["rmse"], 1.5), xytext=(d["model"]["rmse"] * 0.30, 1.52),
                    fontsize=5.8, color=VERM, ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=VERM, lw=0.7))
        # Place this to the right of the perfect-input bar, in clear space: at
        # the previous position it was drawn across the 1 km bar and unreadable.
        ax.annotate(f"what a perfect input\nwould reach: {d['model']['rmse'] - d['oracle']:.2f} mm lower",
                    xy=(d["oracle"], 0.0), xytext=(d["oracle"] + 0.45, 0.30),
                    fontsize=5.8, color=GREEN, va="center",
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.7,
                                    shrinkA=2, shrinkB=2))
    for ax, L in zip(axes, "ab"):
        panel(ax, L, dx=-0.30)
    fig.tight_layout()
    save(fig, "D1_skill_ladder", "bilinear -> blocky -> sharpened -> perfect")


def D2_waterfall(D):
    """The gain over bilinear, split into its two sources.

    Squared error is the additive scale, so the split is taken there and the
    bars are drawn in mm^2. Reading it in mm would double-count: the two
    contributions do not sum linearly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    for ax, (lbl, d) in zip(axes, D.items()):
        bl, md = d["bilinear"], d["model"]
        cv = bl["rmse_10km"] ** 2 - md["rmse_10km"] ** 2
        dv = bl["rmse_downscale"] ** 2 - md["rmse_downscale"] ** 2
        tot = cv + dv
        ax.bar([0], [cv], color=BLUE, width=0.6, label="correcting the coarse field")
        ax.bar([1], [dv], color=ORANGE, width=0.6, label="the 10 km $\\rightarrow$ 1 km step")
        for i, v in enumerate((cv, dv)):
            ax.annotate(f"{v:.2f} mm$^2$\n{v / tot:.0%}", (i, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=6, color="#222222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["coarse\ncorrection", "sharpening"])
        ax.set_ylabel("error variance removed (mm$^2$)")
        ax.set_title(lbl, fontsize=7)
        ax.set_ylim(0, tot * 1.18)
    for ax, L in zip(axes, "ab"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "D2_gain_split", "where the gain over bilinear comes from")


def D3_headroom(D):
    """How much of the downscaling error each model actually removes.

    "Headroom" is the downscaling term that plain interpolation leaves: the
    error that is not already in the coarse field, and therefore the only part
    a sharpening step can reach. The orange sliver is what the model removes
    from it.

    The comparison is model against bilinear, both measured as the part of
    their own error that is not explained by their coarse field. It is a
    different number from D1's, which holds the coarse field fixed and asks
    what sharpening adds over blocks; both are reported because they bound the
    same thing from two sides and both are small.

    The pair is the result. A 0.5 degree cell holds 3,600 fine cells against
    144, so averaging destroys far more structure and the headroom is 2.4x
    larger -- and the model takes a tenth as much of it. More room to win did
    not make winning easier.
    """
    labels = list(D)
    head = [D[k]["bilinear"]["rmse_downscale"] for k in labels]
    took = [D[k]["bilinear"]["rmse_downscale"] - D[k]["model"]["rmse_downscale"] for k in labels]
    blocky = [D[k]["blocky"]["rmse"] - D[k]["model"]["rmse"] for k in labels]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))

    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, head, color=GREY_L, width=0.55,
           label="headroom: error a sharpener could reach")
    ax.bar(x, took, color=VERM, width=0.55, label="removed by the model")
    for i, (h, t) in enumerate(zip(head, took)):
        ax.annotate(f"{h:.2f}", (i, h), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6, color="#222222")
        ax.annotate(f"{t:.3f}", (i, t), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=6, color=VERM)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("downscaling error (mm day$^{-1}$)")
    ax.set_ylim(0, max(head) * 1.28)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left")

    ax = axes[1]
    w = 0.34
    p1 = [100 * t / h for t, h in zip(took, head)]
    p2 = [100 * bq / h for bq, h in zip(blocky, head)]
    ax.bar(x - w / 2, p1, width=w, color=VERM, label="vs bilinear's downscaling")
    ax.bar(x + w / 2, p2, width=w, color=ORANGE, label="vs its own blocky field")
    for xi, v in zip(np.r_[x - w / 2, x + w / 2], p1 + p2):
        ax.annotate(f"{v:.1f}%", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6, color="#222222")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("share of headroom captured (%)")
    ax.set_ylim(0, max(p1 + p2) * 1.45)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    for ax, L in zip(axes, "ab"):
        panel(ax, L)
    fig.tight_layout()
    save(fig, "D3_headroom_vs_captured", "available against taken, both definitions")


def D4_metric_gains(D, AI):
    """The sharpening gain on each metric, beside the total gain over bilinear.

    One panel per metric, not one shared axis: RMSE is in mm and POD, CSI and
    FSS are dimensionless, so a single axis would be a dual scale in disguise
    and would make the millimetre bar dwarf everything by unit choice alone.

    Significance is not the question. The paired day-block intervals (black)
    exclude zero on three of four metrics, so the sharpening effect is real.
    The question is size, and the blue bar is the reference: on every metric
    the orange is a small fraction of it. POD is the exception worth reading --
    the blue bar is negative, because the model detects heavy rain *worse*
    than plain interpolation does, a cost of shrinking toward the mean.
    """
    keys = [("rmse", "RMSE (mm day$^{-1}$)", -1), ("pod", "POD", 1),
            ("csi", "CSI", 1), ("fss5", "FSS @5", 1)]
    d = D["IMERG 0.1$\\degree$"]
    bl, md, bk = d["bilinear"], d["model"], d["blocky"]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0))
    for ax, (k, lab, sgn) in zip(axes, keys):
        coarse = sgn * (md[k] - bl[k])
        sharp = sgn * (md[k] - bk[k])
        ax.barh([1], [coarse], height=0.5, color=BLUE)
        ax.barh([0], [sharp], height=0.5, color=VERM)
        ci = AI["XGB_blocky"].get("vs_control", {}).get(k)
        if ci:
            lo, hi = sorted(-sgn * np.array(ci["ci"]))
            ax.plot([lo, hi], [0, 0], color="#222222", lw=0.9, zorder=5)
        for yy, v, c in ((1, coarse, "#222222"), (0, sharp, VERM)):
            ax.annotate(f"{v:+.4g}", (v, yy), xytext=(4 if v >= 0 else -4, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right", fontsize=5.8, color=c)
        ax.axvline(0, color=GREY, lw=0.7)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["vs bilinear", "sharpening"] if ax is axes[0] else ["", ""])
        ax.set_xlabel(lab, fontsize=6.5)
        ax.set_ylim(-0.6, 1.6)
        ax.margins(x=0.38)
        ax.tick_params(axis="x", labelsize=5.5)
    for ax, L in zip(axes, "abcd"):
        panel(ax, L, dx=-0.10)
    fig.tight_layout()
    save(fig, "D4_metric_gains", "per metric, own axis each")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    house_style()
    D = load()
    AI = json.load(open("results/review/isit_useful.json"))
    print("attribution figures:")
    D1_ladder(D); D2_waterfall(D); D3_headroom(D); D4_metric_gains(D, AI)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
