"""Conceptual figures explaining what the metrics measure and why RMSE misleads.

    python -m src.explainer_figures      # -> results/figures/explainer/

These use small synthetic fields rather than project data: the point is to make
the mechanism visible, and a controlled example shows it far more clearly than a
real rainfall day. Numbers quoted in the captions are computed live from the
synthetic example, so they cannot drift out of date.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .deep.metrics import fss, rapsd, spectral_ratio  # noqa: E402
from .validation import DIV_CMAP, INK, SEQ_CMAP  # noqa: E402

OUT = Path("results/figures/explainer")
INDEX: list[tuple[str, str]] = []


def _style(ax, grid=True):
    ax.set_facecolor(INK["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK["axis"])
    ax.tick_params(colors=INK["muted"], labelsize=8)
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])
    if grid:
        ax.grid(True, color=INK["grid"], linewidth=0.6)
        ax.set_axisbelow(True)


def _map(ax, f, title, vmax, cmap=SEQ_CMAP, vmin=0, sub=None):
    im = ax.imshow(f, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9.5, loc="left", color=INK["primary"], pad=16 if sub else 5)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK["muted"], va="bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK["axis"])
    return im


def _save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor=INK["surface"])
    plt.close(fig)
    INDEX.append((name, caption))
    print(f"  {name}")


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
def fig_problem():
    """What 10 km -> 1 km actually asks for."""
    rng = np.random.default_rng(3)
    fine = np.zeros((48, 48))
    for cy, cx, amp, sd in [(14, 12, 60, 3.0), (30, 33, 45, 2.2), (20, 30, 25, 4.0), (38, 14, 30, 2.0)]:
        y, x = np.ogrid[:48, :48]
        fine += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sd**2))
    fine += 3 * np.clip(ndimage.gaussian_filter(rng.random((48, 48)), 2) - 0.45, 0, None) * 20
    coarse = fine.reshape(4, 12, 4, 12).mean((1, 3))
    naive = np.repeat(np.repeat(coarse, 12, 0), 12, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=INK["surface"])
    vmax = float(fine.max())
    _map(axes[0], coarse, "(a) What the satellite sees", vmax, sub="IMERG, 0.1 deg: 4 x 4 cells")
    for ax in (axes[0],):
        for i in range(5):
            ax.axhline(i - 0.5, color=INK["primary"], lw=0.6, alpha=0.4)
            ax.axvline(i - 0.5, color=INK["primary"], lw=0.6, alpha=0.4)
    _map(axes[1], naive, "(b) The only information-preserving answer", vmax,
         sub="each coarse value copied to its 12 x 12 block")
    im = _map(axes[2], fine, "(c) What actually fell", vmax, sub="AORC, 1/120 deg: 48 x 48 cells")
    axes[2].add_patch(Rectangle((-0.5, -0.5), 12, 12, fill=False, ec=INK["primary"], lw=1.6))
    axes[1].add_patch(Rectangle((-0.5, -0.5), 12, 12, fill=False, ec=INK["primary"], lw=1.6))
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("mm/day", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    fig.suptitle("The downscaling problem: invent 143 numbers per cell, given only their average",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "E1_the_problem.png",
          "One 10 km cell covers a 12 x 12 block of 1 km cells (outlined). The model must produce 144 "
          "values whose mean is fixed by the satellite. Nothing in the inputs says which square kilometre "
          "got the storm core, which is the constraint every metric below is probing.")


def fig_double_penalty():
    """Why squared error prefers a blur to a realistic field."""
    y, x = np.ogrid[:64, :64]
    truth = 50 * np.exp(-((y - 32) ** 2 + (x - 26) ** 2) / (2 * 3.5**2))
    displaced = 50 * np.exp(-((y - 32) ** 2 + (x - 34) ** 2) / (2 * 3.5**2))   # 8 cells east
    blurred = ndimage.gaussian_filter(truth, 7)
    blurred *= truth.sum() / blurred.sum()                                      # same total water

    fields = [("Truth", truth, "reference"), ("Smooth (conditional mean)", blurred, None),
              ("Realistic but displaced 8 km", displaced, None)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=INK["surface"])
    vmax = float(truth.max())
    scores = {}
    for ax, (name, f, sub0) in zip(axes, fields):
        sub = sub0
        if name != "Truth":
            r = rmse(f, truth)
            pk = 100 * f.max() / truth.max()
            scores[name] = (r, pk)
            sub = f"RMSE {r:.2f}   peak {pk:.0f}% of truth"
        im = _map(ax, f, name, vmax, sub=sub)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("mm/day", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    fig.suptitle("The double penalty: squared error prefers the blur", fontsize=12,
                 color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    b = scores["Smooth (conditional mean)"]
    d = scores["Realistic but displaced 8 km"]
    _save(fig, "E2_double_penalty.png",
          f"Both candidates carry the same total water. The smooth field scores RMSE {b[0]:.2f} while "
          f"keeping only {b[1]:.0f}% of the true peak; the sharp field displaced by 8 km scores "
          f"{d[0]:.2f} - {d[0]/b[0]:.1f}x worse - despite having the right intensity. A displaced feature "
          "is punished twice (missed where it was, false alarm where it was not), so minimising RMSE "
          "drives models towards fields that cannot occur in nature.")


def fig_fss():
    """What FSS credits that RMSE does not."""
    rng = np.random.default_rng(1)
    truth = ndimage.gaussian_filter(rng.random((128, 128)), 6)
    truth = (truth - truth.mean()) / truth.std()
    thr = float(np.quantile(truth, 0.9))
    shifted = np.roll(truth, 6, axis=1)
    smooth = ndimage.gaussian_filter(truth, 6)
    scales = [1, 3, 9, 27, 61]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=INK["surface"],
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for name, f, c in (("displaced 6 cells", shifted, "#2a78d6"), ("over-smoothed", smooth, "#eb6834")):
        ax.plot(scales, [fss(f, truth, thr, s) for s in scales], marker="o", ms=6, lw=2, color=c, label=name)
    ax.axhline(1.0, color=INK["primary"], lw=1.2, ls="--")
    _style(ax)
    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([str(s) for s in scales])
    ax.set_xlabel("neighbourhood width (cells)")
    ax.set_ylabel("Fractions Skill Score")
    ax.set_title("(a) FSS forgives displacement as the scale grows", fontsize=9.5, loc="left",
                 color=INK["primary"])
    ax.legend(frameon=False, fontsize=8)
    _map(axes[1], np.where(truth >= thr, 1.0, 0.0) + 2 * np.where(shifted >= thr, 1.0, 0.0), "(b) the two events",
         3.0, cmap=SEQ_CMAP, sub="truth, displaced, and their overlap")
    fig.suptitle("Fractions Skill Score: credit for being approximately right", fontsize=12,
                 color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "E3_fss.png",
          "A field displaced by 6 cells scores near zero at grid-point level but recovers as the "
          "neighbourhood widens - it had the right structure in roughly the right place. An "
          "over-smoothed field does not recover, because the structure is gone at every scale. "
          "RMSE cannot make this distinction.")


def fig_spectra():
    """What the spectral ratio measures."""
    rng = np.random.default_rng(0)
    truth = ndimage.gaussian_filter(rng.random((160, 160)), 2.0)
    truth = (truth - truth.mean()) / truth.std()
    smooth = ndimage.gaussian_filter(truth, 5)
    noisy = truth + 0.9 * rng.standard_normal(truth.shape)
    wl, p_t = rapsd(truth)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=INK["surface"])
    ax = axes[0]
    ratios = {}
    for name, f, c in (("truth", truth, INK["primary"]), ("over-smoothed", smooth, "#eb6834"),
                       ("noisy", noisy, "#1baf7a")):
        _, p = rapsd(f)
        ax.loglog(wl, p, lw=2, color=c, label=name)
        ratios[name] = spectral_ratio(wl, p, p_t, max_wavelength=10.0)
    _style(ax)
    ax.invert_xaxis()
    ax.axvspan(wl.min(), 10, color=INK["grid"], alpha=0.4, zorder=0)
    ax.set_xlabel("wavelength (cells)")
    ax.set_ylabel("power spectral density")
    ax.set_title("(a) power spectra; shaded = the sub-10 band scored", fontsize=9.5, loc="left",
                 color=INK["primary"])
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    names = list(ratios)
    ax.barh(np.arange(len(names)), [ratios[n] for n in names],
            color=[INK["primary"], "#eb6834", "#1baf7a"], height=0.55, linewidth=0)
    ax.axvline(1.0, color=INK["primary"], lw=1.3, ls="--")
    _style(ax, grid=False)
    ax.grid(True, axis="x", color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("spectral ratio below 10 cells")
    for i, n in enumerate(names):
        ax.text(ratios[n], i, f"  {ratios[n]:.2f}", va="center", fontsize=8.5, color=INK["secondary"])
    ax.set_title("(b) one number: 1.0 = right amount of texture", fontsize=9.5, loc="left",
                 color=INK["primary"])
    fig.suptitle("Spectral ratio: is the fine-scale variance right?", fontsize=12,
                 color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "E4_spectral_ratio.png",
          f"Smoothing removes power at short wavelengths (ratio {ratios['over-smoothed']:.2f}); adding "
          f"noise adds too much (ratio {ratios['noisy']:.2f}). The diagnostic is two-sided, so a product "
          "far above 1.0 is as wrong as one far below - it is texture in the wrong places, not skill.")


def fig_two_levels():
    """Why 10 km and 1 km scores are not comparable."""
    rng = np.random.default_rng(7)
    fine_truth = np.clip(ndimage.gaussian_filter(rng.random((96, 96)), 2.5) * 60 - 10, 0, None)
    err = ndimage.gaussian_filter(rng.standard_normal((96, 96)), 1.2) * 6
    fine_pred = np.clip(fine_truth + err, 0, None)
    c_truth = fine_truth.reshape(8, 12, 8, 12).mean((1, 3))
    c_pred = fine_pred.reshape(8, 12, 8, 12).mean((1, 3))
    r_fine, r_coarse = rmse(fine_pred, fine_truth), rmse(c_pred, c_truth)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), facecolor=INK["surface"])
    vmax = float(fine_truth.max())
    _map(axes[0], fine_truth, "(a) truth, 1 km", vmax)
    _map(axes[1], fine_pred, "(b) prediction, 1 km", vmax, sub=f"RMSE {r_fine:.2f}")
    _map(axes[2], c_truth, "(c) truth, block-averaged to 10 km", vmax)
    im = _map(axes[3], c_pred, "(d) prediction at 10 km", vmax, sub=f"RMSE {r_coarse:.2f}")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("mm/day", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    fig.suptitle("The same product scores differently at the two resolutions", fontsize=12,
                 color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "E5_two_levels.png",
          f"Identical prediction, two scoring levels: RMSE {r_fine:.2f} at 1 km against {r_coarse:.2f} at "
          f"10 km ({100*(1-r_coarse/r_fine):.0f}% lower) purely because averaging 144 cells cancels "
          "independent error. A 10 km score always flatters a product, so the project's 10 km and 1 km "
          "numbers must never be compared with each other - only within a level.")


def write_index():
    lines = ["# Explainer figures", "",
             "Conceptual figures for [`docs/METRICS.md`](../../../docs/METRICS.md). They use small "
             "synthetic fields, because the mechanisms are far clearer in a controlled example than in a "
             "real rainfall day. All quoted numbers are computed when the figure is rendered.", "",
             "Regenerate with `python -m src.explainer_figures`.", "",
             "| figure | what it explains |", "|---|---|"]
    for name, cap in sorted(INDEX):
        lines.append(f"| [`{name}`]({name}) | {cap} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  README.md ({len(INDEX)} figures)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("rendering ->", OUT)
    fig_problem()
    fig_double_penalty()
    fig_fss()
    fig_spectra()
    fig_two_levels()
    write_index()


if __name__ == "__main__":
    main()
