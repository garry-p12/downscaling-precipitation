"""Figures for the two-domain regime comparison (paper sections 4.5, 4.6, 5.8).

Every number here is read from a pipeline output file or stated inline from a
logged run; none are recomputed approximations.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("results/figures/paper")
INK, INK2, INK3 = "#1a1a18", "#6c6d66", "#9b9c93"
ACCENT, COOL, WARM, FLAT = "#cf5f2e", "#2d6ca8", "#c98a04", "#8296a6"
LINE = "#e6e7df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color=LINE, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, name, caption):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  {name}  --  {caption}")


# ---------------------------------------------------------------- P6: ladder
def fig_forcing_ladder():
    """The paper's central result: skill tracks forcing, not terrain."""
    rows = [
        ("Colorado\ncool season", "Oct–Apr · orographic", 20.8, 0.255, COOL),
        ("Colorado\nall year", "mixed", 15.0, 0.255, "#7d8f5f"),
        ("Colorado\nwarm season", "May–Sep · convective", 8.0, 0.255, WARM),
        ("Austin\nall year", "convective, flat", 6.8, 0.000, FLAT),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), width_ratios=[2.5, 1])

    y = np.arange(len(rows))[::-1]
    for (lbl, sub, val, _, c), yy in zip(rows, y):
        ax.barh(yy, val, height=0.56, color=c, zorder=3)
        ax.text(val + 0.4, yy, f"{val:.1f} %", va="center", ha="left",
                fontsize=11, color=c, fontweight="bold")
        ax.text(-0.6, yy + 0.17, lbl.replace("\n", " "), va="center", ha="right",
                fontsize=10.5, color=INK, fontweight="semibold")
        ax.text(-0.6, yy - 0.19, sub, va="center", ha="right", fontsize=8.5, color=INK3)
    ax.axvline(20, color=ACCENT, ls="--", lw=1.3, zorder=2)
    ax.text(20, 1.02, "20 % target", color=ACCENT, fontsize=9, ha="center",
            va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 24); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("RMSE reduction vs bilinear interpolation (%)", fontsize=10, color=INK2)
    _style(ax); ax.grid(axis="x", color=LINE, lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_title("Skill tracks orographic forcing", fontsize=12.5, color=INK,
                 loc="left", pad=12, fontweight="semibold")

    # the decisive pair: same terrain, different forcing; different terrain, same skill
    pairs = [("Colorado\nwarm", 8.0, 0.255, WARM), ("Austin\nall year", 6.8, 0.000, FLAT)]
    for i, (lbl, val, slope, c) in enumerate(pairs):
        ax2.bar(i, val, width=0.5, color=c, zorder=3)
        ax2.text(i, val + 0.3, f"{val:.1f} %", ha="center", fontsize=10.5,
                 color=c, fontweight="bold")
        ax2.text(i, -0.26, f"{slope:.3f}", ha="center", fontsize=11,
                 color=INK, fontweight="bold", transform=ax2.get_xaxis_transform())
    ax2.set_xticks([0, 1]); ax2.set_xticklabels([p[0] for p in pairs], fontsize=9.5, color=INK)
    ax2.text(0.5, -0.36, "fraction of cells above 5° slope", ha="center", fontsize=8.5,
             color=INK3, transform=ax2.get_xaxis_transform())
    ax2.set_ylim(0, 11); ax2.set_ylabel("reduction (%)", fontsize=9.5, color=INK2)
    _style(ax2)
    ax2.set_title("Same skill,\n25 points of terrain apart", fontsize=10.5,
                  color=INK, loc="left", pad=12)
    _save(fig, "P6_forcing_ladder",
          "20.8 / 15.0 / 8.0 / 6.8 % -- warm-season Colorado matches flat Austin")


# ------------------------------------------------------------- P7: 2 domains
def fig_two_domains():
    """Austin vs Colorado on identical axes, from each domain's metrics file."""
    au = json.loads(Path("results/validation_metrics.json").read_text())["comparison_table"]
    co = json.loads(Path("results/colorado/validation_metrics.json").read_text())["comparison_table"]
    metrics = [("rmse", "RMSE", False), ("pearson_r", "Pearson r", True),
               ("nse", "NSE", True), ("kge", "KGE", True),
               ("median_cell_nse", "median cell NSE", True)]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.5))
    for ax, (key, lbl, higher) in zip(axes, metrics):
        for i, (tbl, name, c) in enumerate(((au, "Austin", FLAT), (co, "Colorado", COOL))):
            ml, bl = tbl[key]["ml"], tbl[key]["bilinear"]
            ax.plot([i - 0.17, i + 0.17], [bl, ml], color=c, lw=1.2, zorder=2)
            ax.scatter([i - 0.17], [bl], s=42, facecolor="white", edgecolor=c,
                       lw=1.6, zorder=3)
            ax.scatter([i + 0.17], [ml], s=52, color=c, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Austin", "Colorado"], fontsize=9)
        ax.set_title(lbl, fontsize=10.5, color=INK, pad=8)
        ax.set_xlim(-0.5, 1.5)
        _style(ax)
    axes[0].set_ylabel("open = bilinear   filled = ML", fontsize=9, color=INK2)
    fig.suptitle("Two domains, identical size and predictors, different terrain regime",
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=1.04,
                 fontweight="semibold")
    _save(fig, "P7_two_domains",
          "ML beats bilinear on every metric in Colorado, including KGE which it loses in Austin")


# ---------------------------------------------------------------- P8: seeds
def fig_seed_spread():
    """Seed noise against the differences the single-run table appeared to show."""
    xgb = [4.578, 4.583, 4.585, 4.583]
    cnn = [4.7005, 4.6692, 4.663]
    swin = [4.6140, 4.6461, 4.653]
    fig, ax = plt.subplots(figsize=(9.5, 4.3))

    for i, (vals, name, c) in enumerate(((xgb, "XGBoost\n4 seeds", COOL),
                                         (swin, "Swin 7.9 M\n3 seeds", ACCENT),
                                         (cnn, "CNN 1.65 M\n3 seeds", WARM))):
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [i, i], color=c, lw=5, alpha=0.3, zorder=2,
                solid_capstyle="round")
        ax.scatter(vals, [i] * len(vals), s=55, color=c, zorder=4)
        ax.text(hi + 0.012, i, f"spread {hi - lo:.3f}  ({(hi - lo) / np.mean(vals) * 100:.2f} %)",
                va="center", fontsize=9.5, color=c)
        ax.text(lo - 0.012, i, name, va="center", ha="right", fontsize=10, color=INK)

    # the single published pairing, against the difference of means
    ax.annotate("", xy=(4.653, 2.46), xytext=(4.663, 2.46),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(4.658, 2.56, "published pairing\n0.010", ha="center", fontsize=8.5, color=INK)
    ax.annotate("", xy=(4.6377, 2.08), xytext=(4.6776, 2.08),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.8))
    ax.text(4.6577, 2.16, "difference of means  0.040", ha="center", fontsize=9.5,
            color=ACCENT, fontweight="bold")
    ax.axvline(4.6530, color=INK3, ls=":", lw=1)
    ax.axvline(4.6630, color=INK3, ls=":", lw=1)
    ax.text(4.658, -0.52, "disjoint", ha="center", fontsize=8.5, color=INK3)
    ax.set_ylim(-0.75, 2.85); ax.set_yticks([])
    ax.set_xlabel("test RMSE, Austin 2019–2020 (mm day⁻¹)", fontsize=10, color=INK2)
    _style(ax); ax.grid(axis="x", color=LINE, lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_title("One run understated the CNN–Swin gap fourfold",
                 fontsize=12, color=INK, loc="left", pad=12, fontweight="semibold")
    _save(fig, "P8_seed_spread",
          "trees 0.15 %, deep ~0.8 %; Swin leads CNN 9/9, ranges disjoint")


if __name__ == "__main__":
    print("writing regime figures:")
    fig_forcing_ladder()
    fig_two_domains()
    fig_seed_spread()
