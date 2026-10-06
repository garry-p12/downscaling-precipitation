"""Figures for the sub-daily structure experiment.

    python -m src.subdaily_figures     # -> results/figures/

Two panels: what the feature is (a daily total hides its own intensity), and
what it bought across every model family.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .figures import FIG_DIR, INDEX, _save, _style, _title  # noqa: E402
from .model_comparison import PALETTE  # noqa: E402
from .validation import INK  # noqa: E402

# Test-period 1 km RMSE before/after adding the four sub-daily channels.
GAINS = {
    "XGBoost": (4.655, 4.583),
    "CNN": (4.694, 4.663),
    "Swin": (4.729, 4.653),
    "Diffusion\n(ens. mean)": (4.840, 4.807),
}
COLORS = [PALETTE["xgboost"], PALETTE["cnn"], PALETTE["swin"], PALETTE["diffusion"]]


def fig_concept():
    """Same daily total, opposite sub-daily structure."""
    hours = np.arange(48) * 0.5
    conv = np.zeros(48)
    conv[20:26] = [4, 14, 26, 20, 8, 3]          # one afternoon storm
    strat = np.full(48, conv.sum() / 30.0)
    strat[:9] = strat[39:] = 0.0                  # long gentle band
    strat *= conv.sum() / strat.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), facecolor=INK["surface"], sharey=True)
    for ax, (name, y, col) in zip(axes, [("Convective", conv, "#eb6834"), ("Stratiform", strat, "#2a78d6")]):
        ax.bar(hours, y, width=0.45, color=col, linewidth=0)
        _style(ax)
        wet = int((y > 0).sum())
        total = y.sum() * 0.5
        ci = total / (wet * 0.5)
        ax.set_xlabel("hour of day (UTC)")
        _title(ax, f"{name}",
               f"daily total {total:.0f} mm   wet half-hours {wet}/48   conditional intensity {ci:.1f} mm/h")
        ax.set_xlim(0, 24)
    axes[0].set_ylabel("rate (mm/h)")
    fig.suptitle("Identical daily total, opposite sub-grid structure — and the daily field cannot tell them apart",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3.0)
    _save(fig, "27_subdaily_concept.png",
          "Two days with the same accumulation. The IMERG daily granule counts its own raining half-hours "
          "(`precipitation_cnt_cond`), so total / (wet half-hours x 0.5 h) recovers the conditional "
          "intensity and separates these cases — without downloading the half-hourly product.")


def fig_gain():
    names = list(GAINS)
    before = [GAINS[n][0] for n in names]
    after = [GAINS[n][1] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), facecolor=INK["surface"],
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    y = np.arange(len(names))
    for yi, b, a, c in zip(y, before, after, COLORS):
        ax.plot([a, b], [yi, yi], color=c, lw=2.2, zorder=2, alpha=0.55)
        ax.plot(b, yi, "o", ms=8, mfc=INK["surface"], mec=c, mew=2.0, zorder=3)
        ax.plot(a, yi, "o", ms=10, color=c, zorder=4)
        # label to the right of the "before" marker so it can never collide
        # with the y-axis tick labels
        ax.text(b + 0.008, yi, f"{b:.3f} \u2192 {a:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK["secondary"])
    _style(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=INK["secondary"])
    ax.invert_yaxis()
    ax.axvline(4.919, color=INK["primary"], lw=1.3, ls="--")
    ax.text(4.919, -0.62, " bilinear 4.919", fontsize=8, color=INK["primary"], ha="left", va="center")
    ax.set_xlim(min(after) - 0.02, 4.96)
    ax.set_ylim(len(names) - 0.4, -0.75)
    ax.set_xlabel("test-period RMSE (mm/day)")
    _title(ax, "Every model family improved",
           "hollow = 27 features, filled = 31 features (+ sub-daily structure)")

    ax = axes[1]
    ranks = {"imerg_cond_intensity": 4, "imerg_wet_frac": 5, "imerg_mw_frac": 7, "imerg_prob_liquid": 31}
    labs = list(ranks)
    vals = [32 - ranks[k] for k in labs]     # invert so higher bar = better rank
    ax.barh(np.arange(len(labs)), vals, color=["#2a78d6", "#2a78d6", "#6da7ec", "#c9c8c2"],
            height=0.6, linewidth=0)
    _style(ax, grid_axis="x")
    ax.set_yticks(np.arange(len(labs)))
    ax.set_yticklabels(labs, fontsize=9, color=INK["secondary"])
    ax.invert_yaxis()
    ax.set_xticks([])
    for i, k in enumerate(labs):
        ax.text(vals[i] + 0.4, i, f"rank {ranks[k]} of 31", va="center", fontsize=8.5, color=INK["secondary"])
    ax.set_xlim(0, 34)
    _title(ax, "Where they rank by importance", "three of four in the top seven")
    fig.suptitle("Sub-daily structure: what it bought", fontsize=12, color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3.0)
    _save(fig, "28_subdaily_gain.png",
          "Adding four channels derived from the daily granule's half-hour counters improved every model "
          "family, and moved the diffusion ensemble's spread ratio from 0.59 to 0.81. "
          "`imerg_cond_intensity` and `imerg_wet_frac` rank 4th and 5th of 31 features, above every ERA5 "
          "field; `imerg_prob_liquid` ranks last, because Texas rainfall is almost always liquid.")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("rendering ->", FIG_DIR)
    fig_concept()
    fig_gain()


if __name__ == "__main__":
    main()
