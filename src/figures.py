"""Render every result in the project as a labelled figure set.

    python -m src.figures            # -> results/figures/

Reads only the persisted result files (JSON training logs, metric dumps) plus
the map figures produced by the validation and comparison steps, so it is cheap
to re-run and never depends on the large NetCDF products being present.

Figures are numbered in reading order and each carries its own title, units and
provenance, so a file is interpretable on its own when pulled into a slide or
an issue. The index is written to results/figures/README.md.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .model_comparison import LABELS, PALETTE  # noqa: E402
from .validation import INK  # noqa: E402

FIG_DIR = Path("results/figures")
INDEX: list[tuple[str, str]] = []

# Dev-year reference lines (bilinear interpolation, 2018) used across ablations.
DEV_BILINEAR = {"rmse": 5.1525, "pod30": 0.6329, "csi30": 0.4933, "bias": -0.163}


def _style(ax, grid_axis="both"):
    ax.set_facecolor(INK["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK["axis"])
    ax.tick_params(colors=INK["muted"], labelsize=8)
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])
    ax.grid(True, axis=grid_axis, color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def _title(ax, text, sub=None):
    ax.set_title(text, fontsize=10, loc="left", color=INK["primary"], pad=20 if sub else 6)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8, color=INK["muted"], va="bottom")


def _save(fig, name, caption):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=INK["surface"])
    plt.close(fig)
    INDEX.append((name, caption))
    print(f"  {name}")


def _load(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 1. Test-period comparison
# --------------------------------------------------------------------------- #
def _crps(cmp: dict, name: str) -> float:
    """CRPS of a point forecast is its MAE; the ensemble gets its own value."""
    if name == "diffusion" and "diffusion_ensemble" in cmp:
        return cmp["diffusion_ensemble"]["crps_mm"]
    return cmp["products"][name]["overall"]["mae"]


def _barh_panel(ax, names, vals, label, zero_line=False, fmt="{:.3f}", ref=None, ref_label=None,
                colors=None, ticklabels=None):
    y = np.arange(len(names))
    cols = colors or [PALETTE.get(n, "#888") for n in names]
    ax.barh(y, vals, color=cols, height=0.62, linewidth=0)
    _style(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(ticklabels or [LABELS.get(n, n) for n in names], fontsize=8, color=INK["secondary"])
    ax.invert_yaxis()
    if zero_line:
        ax.axvline(0, color=INK["axis"], lw=1)
    if ref is not None:
        ax.axvline(ref, color=INK["primary"], lw=1.3, ls="--")
    lo, hi = min(list(vals) + ([ref] if ref is not None else []) + [0.0]), max(list(vals) + [0.0])
    span = (hi - lo) or 1.0
    # headroom on whichever side the value labels land
    pad_lo = span * (0.24 if min(vals) < 0 else 0.04)
    ax.set_xlim(lo - pad_lo, hi + span * 0.24)
    for yi, v in zip(y, vals):
        # place the value outside the bar end, whichever side that is
        off = span * 0.02 if v >= 0 else -span * 0.02
        ax.text(v + off, yi, fmt.format(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=INK["secondary"])
    _title(ax, label, ref_label)


def _dot_panel(ax, names, vals, label, fmt="{:.3f}", ref=None, ref_label=None, colors=None,
               better="lower"):
    """Cleveland dot plot: used where the spread between values is a small
    fraction of their magnitude. Bars would have to start at zero (making the
    differences invisible) or be truncated (making them look bigger than they
    are); dots imply no area, so the axis can legitimately be zoomed."""
    y = np.arange(len(names))
    cols = colors or [PALETTE.get(n, "#888") for n in names]
    lo = min(list(vals) + ([ref] if ref is not None else []))
    hi = max(list(vals) + ([ref] if ref is not None else []))
    span = (hi - lo) or 1.0
    for yi, v, c in zip(y, vals, cols):
        ax.plot([lo - span * 0.4, v], [yi, yi], color=INK["grid"], lw=1.0, zorder=1)
        ax.plot([v], [yi], "o", ms=9, color=c, zorder=3,
                markeredgecolor=INK["surface"], markeredgewidth=1.5)
        ax.text(v + span * 0.05, yi, fmt.format(v), va="center", fontsize=8, color=INK["secondary"], zorder=4)
    _style(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=INK["secondary"])
    ax.invert_yaxis()
    if ref is not None:
        ax.axvline(ref, color=INK["primary"], lw=1.3, ls="--", zorder=2)
    ax.set_xlim(lo - span * 0.25, hi + span * 0.45)
    ax.set_ylim(len(names) - 0.4, -0.6)
    _title(ax, label, ref_label)


def fig_test_metrics(cmp: dict):
    names = list(cmp["products"])
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8), facecolor=INK["surface"])
    getters = [
        ("RMSE (mm/day)", lambda n: cmp["products"][n]["overall"]["rmse"], False, None, None),
        ("MAE (mm/day)", lambda n: cmp["products"][n]["overall"]["mae"], False, None, None),
        ("Bias (mm/day)", lambda n: cmp["products"][n]["overall"]["bias"], True, None, None),
        ("NSE", lambda n: cmp["products"][n]["overall"]["nse"], False, None, None),
        ("CRPS (mm/day)", lambda n: _crps(cmp, n), False, None, "point forecast: CRPS = MAE"),
        ("POD, days > 30 mm", lambda n: cmp["products"][n]["detection"]["30.0"]["pod"], False, None, None),
    ]
    for ax, (label, get, zero, ref, sub) in zip(axes.ravel(), getters):
        _barh_panel(ax, names, [get(n) for n in names], label, zero_line=zero, ref=ref, ref_label=sub)
    fig.suptitle(f"Test period {cmp['period'][0]} to {cmp['period'][1]} ({cmp['n_days']} days), 1 km vs AORC",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=3.0, h_pad=2.5)
    _save(fig, "01_test_metrics_comparison.png",
          "Headline deterministic and probabilistic scores for every product on the 2019-2020 test years. "
          "Lower is better except NSE and POD. RMSE spans only 4.86-5.15 across four model classes, while "
          "CRPS separates them clearly - the diffusion ensemble wins the probabilistic score.")


def fig_detection(cmp: dict):
    names = list(cmp["products"])
    thresholds = ["1.0", "10.0", "30.0"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=INK["surface"], sharey=True)
    for ax, metric, lab in zip(axes, ("pod", "far", "csi"),
                               ("POD (hit rate)", "FAR (false alarm ratio)", "CSI (critical success index)")):
        x = np.arange(len(thresholds))
        w = 0.8 / len(names)
        for k, n in enumerate(names):
            y = [cmp["products"][n]["detection"][t][metric] for t in thresholds]
            ax.bar(x + (k - len(names) / 2 + 0.5) * w, y, width=w * 0.9,
                   color=PALETTE.get(n, "#888"), label=LABELS.get(n, n), linewidth=0)
        _style(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels([f"> {t.rstrip('0').rstrip('.')} mm" for t in thresholds])
        _title(ax, lab)
    axes[0].set_ylabel("score")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Event detection by intensity threshold", fontsize=12, color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    _save(fig, "05_detection_by_threshold.png",
          "Hit rate, false alarms and CSI at 1/10/30 mm. Every learned product loses to plain bilinear "
          "interpolation at the 30 mm threshold - the core extremes failure.")


def fig_intensity(val: dict):
    methods = ["ml", "bilinear", "nearest"]
    classes = ["dry", "light", "moderate", "heavy"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=INK["surface"])
    for ax, metric, lab in zip(axes, ("bias", "rmse", "pearson_r"),
                               ("Bias (mm/day)", "RMSE (mm/day)", "Pearson r")):
        x = np.arange(len(classes))
        w = 0.8 / len(methods)
        for k, m in enumerate(methods):
            y = [val["methods"][m]["by_intensity"][c][metric] for c in classes]
            col = PALETTE["xgboost"] if m == "ml" else PALETTE[m]
            lb = "XGBoost 2-stage" if m == "ml" else LABELS[m]
            ax.bar(x + (k - 1) * w, y, width=w * 0.9, color=col, label=lb, linewidth=0)
        _style(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(["dry\n<1 mm", "light\n1-10", "moderate\n10-30", "heavy\n>30"])
        if metric == "bias":
            ax.axhline(0, color=INK["axis"], lw=1)
        _title(ax, lab)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Skill stratified by observed intensity class", fontsize=12, color=INK["primary"],
                 x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    _save(fig, "04_skill_by_intensity.png",
          "Scores split by observed rainfall class. The heavy class carries a large negative bias "
          "(-14 mm for the tree model): all products shrink extremes toward the mean.")


# --------------------------------------------------------------------------- #
# 2. Ablations
# --------------------------------------------------------------------------- #
def _curves(ax, logs: dict, metric: str, colors: dict):
    for name, log in logs.items():
        h = log["history"]
        ax.plot([p["step"] for p in h], [p[metric] for p in h], lw=1.8, marker="o", ms=3.5,
                color=colors[name], label=name)


def fig_loss_ablation():
    files = {"log-space MSE": "models/ablate/cnn_logmse_train_log.json",
             "hybrid": "models/ablate/cnn_hybrid_train_log.json",
             "mm-space MSE": "models/ablate/cnn_mse_train_log.json"}
    logs = {k: _load(v) for k, v in files.items() if Path(v).exists()}
    if not logs:
        return
    colors = {"log-space MSE": "#e34948", "hybrid": "#eda100", "mm-space MSE": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=INK["surface"])
    for ax, metric, lab in zip(axes, ("rmse", "bias"), ("Dev-2018 RMSE (mm/day)", "Dev-2018 bias (mm/day)")):
        _curves(ax, logs, metric, colors)
        _style(ax)
        ax.set_xlabel("training step")
        ax.set_ylabel(lab)
        if metric == "rmse":
            ax.axhline(DEV_BILINEAR["rmse"], color=INK["primary"], lw=1.4, ls="--")
            ax.text(ax.get_xlim()[1], DEV_BILINEAR["rmse"], " bilinear 5.152", fontsize=8,
                    color=INK["primary"], va="bottom", ha="right")
        else:
            ax.axhline(0, color=INK["axis"], lw=1)
        _title(ax, lab)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Which space should the loss be taken in?", fontsize=12, color=INK["primary"],
                 x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=3.0)
    _save(fig, "06_loss_space_ablation.png",
          "Training in log space but scoring in mm costs 0.43 mm RMSE and drives bias to -1.4 mm: "
          "log-space MSE fits the mean of log precipitation, which under-predicts the mean in mm.")


def fig_capacity_ablation():
    files = {"CNN base 24 (1.65 M)": "models/deep/cnn_b24_train_log.json",
             "CNN base 64 (11.7 M)": "models/deep/cnn_b64_train_log.json",
             "Swin transformer (7.9 M)": "models/deep/swin_train_log.json"}
    # historical: these runs used the 1096-day training set
    logs = {k: _load(v) for k, v in files.items() if Path(v).exists()}
    if not logs:
        return
    colors = {"CNN base 24 (1.65 M)": PALETTE["cnn"], "CNN base 64 (11.7 M)": "#6da7ec",
              "Swin transformer (7.9 M)": PALETTE["swin"]}
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=INK["surface"])
    _curves(ax, logs, "rmse", colors)
    _style(ax)
    ax.axhline(DEV_BILINEAR["rmse"], color=INK["primary"], lw=1.4, ls="--")
    ax.text(ax.get_xlim()[1], DEV_BILINEAR["rmse"], " bilinear 5.152 ", fontsize=8,
            color=INK["primary"], va="bottom", ha="right")
    ax.set_xlabel("training step")
    ax.set_ylabel("Dev-2018 RMSE (mm/day)")
    ax.legend(frameon=False, fontsize=8)
    _title(ax, "Capacity and architecture barely matter",
           "all three peak at step ~500 then overfit; 7x more parameters changes RMSE by 0.002")
    fig.tight_layout()
    _save(fig, "07_capacity_architecture_ablation.png",
          "A 1.65 M CNN, an 11.7 M CNN and a 7.9 M transformer converge to the same dev RMSE and all "
          "overfit after ~500 steps. The binding constraint is ~1100 independent weather days, not capacity.")


def fig_extremes_ablation():
    cfgs = {"base (mm MSE)": "base", "quantile t=0.90": "quantile_t90", "quantile t=0.95": "quantile_t95",
            "heavy x3 weighting": "heavy_x3", "importance weighting": "importance", "uniform crops": "uniform_crops"}
    rows = {}
    for lab, tag in cfgs.items():
        p = Path(f"models/ablate3/cnn_{tag}_train_log.json")
        if p.exists():
            h = _load(p)["history"]
            rows[lab] = min(h, key=lambda x: x["rmse"])
    if not rows:
        return
    names = list(rows)
    cols = ["#898781", "#eda100", "#eb6834", "#2a78d6", "#1baf7a", "#4a3aa7"][: len(names)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=INK["surface"])
    specs = [("rmse", "Dev-2018 RMSE (mm/day)", "{:.4f}", DEV_BILINEAR["rmse"]),
             ("pod30", "POD, days > 30 mm", "{:.3f}", DEV_BILINEAR["pod30"]),
             ("bias", "Dev-2018 bias (mm/day)", "{:+.3f}", DEV_BILINEAR["bias"])]
    for ax, (metric, label, fmt, ref) in zip(axes, specs):
        vals = [rows[n][metric] for n in names]
        _dot_panel(ax, names, vals, label, fmt=fmt, ref=ref, ref_label="dashed = bilinear", colors=cols)
    fig.suptitle("Remedies for extreme-event underprediction (dev year, best checkpoint)",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.5)
    _save(fig, "08_extremes_remedies_ablation.png",
          "Six objectives at their best checkpoint. Heavy-event sample weighting is the only change that "
          "moves POD (0.612 -> 0.633, matching bilinear) and cuts bias 3.4x, at an RMSE cost of 0.0014. "
          "The quantile losses do almost nothing because the best checkpoint arrives at step 500.")


def fig_overfitting():
    sources = {"base (mm MSE)": "models/ablate3/cnn_base_train_log.json",
               "heavy x3": "models/ablate3/cnn_heavy_x3_train_log.json",
               "importance": "models/ablate3/cnn_importance_train_log.json",
               "uniform crops": "models/ablate3/cnn_uniform_crops_train_log.json",
               "quantile t=0.90": "models/ablate3/cnn_quantile_t90_train_log.json"}
    logs = {k: _load(v) for k, v in sources.items() if Path(v).exists()}
    if not logs:
        return
    cols = ["#898781", "#2a78d6", "#1baf7a", "#4a3aa7", "#eda100"]
    colors = dict(zip(logs, cols))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=INK["surface"])
    for ax, metric, lab in zip(axes, ("rmse", "pod30"),
                               ("Dev-2018 RMSE (mm/day)", "POD, days > 30 mm")):
        _curves(ax, logs, metric, colors)
        _style(ax)
        ax.axhline(DEV_BILINEAR[metric], color=INK["primary"], lw=1.4, ls="--")
        ax.set_xlabel("training step")
        ax.set_ylabel(lab)
        _title(ax, lab, "dashed = bilinear")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Every configuration peaks early then degrades", fontsize=12, color=INK["primary"],
                 x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=3.0)
    _save(fig, "09_overfitting_curves.png",
          "Dev skill against training step. RMSE is best at step 500-2000 and worsens thereafter, which is "
          "why loss-shape changes (quantile) never take effect: training stops before they can act.")


def fig_selection_criterion():
    """What the POD-floor checkpoint rule does on the extended record."""
    p = Path("models/v2/cnn_train_log.json")
    if not p.exists():
        return
    log = _load(p)
    h = log["history"]
    step = [x["step"] for x in h]
    rmse = [x["rmse"] for x in h]
    pod = [x["pod30"] for x in h]
    floor = log.get("pod_floor", 0.633)
    elig = [i for i, v in enumerate(pod) if v >= floor]
    chosen = min(elig, key=lambda i: rmse[i]) if elig else int(np.argmin(rmse))
    rmse_only = int(np.argmin(rmse))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=INK["surface"])
    for ax, y, lab, ref in ((axes[0], rmse, "Dev RMSE (mm/day)", None),
                            (axes[1], pod, "POD, days > 30 mm", floor)):
        ax.plot(step, y, lw=1.8, color=PALETTE["cnn"], marker="o", ms=3)
        _style(ax)
        ax.set_xlabel("training step")
        ax.set_ylabel(lab)
        if ref is not None:
            ax.axhline(ref, color=INK["primary"], lw=1.3, ls="--")
            ax.text(step[-1], ref, " POD floor (bilinear) ", fontsize=8, color=INK["primary"],
                    va="bottom", ha="right")
        ax.plot(step[chosen], y[chosen], "o", ms=13, mfc="none", mec=PALETTE["diffusion"], mew=2.2,
                zorder=5, label="selected (POD floor)")
        ax.plot(step[rmse_only], y[rmse_only], "s", ms=15, mfc="none", mec="#eb6834", mew=2.0,
                zorder=4, label="RMSE-only choice")
    axes[0].legend(frameon=False, fontsize=8)
    _title(axes[0], "Dev RMSE", "drifts up as the model learns extremes")
    _title(axes[1], "Heavy-event detection", "climbs throughout training")
    fig.suptitle("Why checkpoint selection needs more than RMSE (CNN, 2000-2020 record)",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=3.0)
    agree = chosen == rmse_only
    verdict = (
        f"Here both rules land on step {step[chosen]} (POD {pod[chosen]:.3f}): with 6423 training days the "
        "model clears the POD floor well before its RMSE optimum, so the constraint is slack. On the "
        "1096-day record the same rule changed the outcome - the model only reached high POD long after "
        "RMSE had started degrading. More data relaxes the trade-off rather than removing it."
        if agree else
        f"Selecting on RMSE alone keeps step {step[rmse_only]} (POD {pod[rmse_only]:.3f}); requiring "
        f"POD >= {floor:g} first keeps step {step[chosen]} (POD {pod[chosen]:.3f}).")
    _save(fig, "26_checkpoint_selection.png",
          f"The CNN's heavy-event POD rises from {pod[0]:.3f} to {max(pod):.3f} over training while dev "
          f"RMSE passes through a minimum at step {step[rmse_only]} and then drifts up by "
          f"{100*(max(rmse)/min(rmse)-1):.0f}%. " + verdict)


def fig_diffusion_calibration():
    p = Path("models/v2/calib.json")
    if not p.exists():
        p = Path("models/deep/diffusion_calibration.json")
    if not p.exists():
        return
    cal = _load(p)
    keys = sorted(cal["etas"], key=lambda k: float(k.split("_")[-1]))
    labels = [f"eta = {k.split('_')[-1]}" for k in keys]
    ratio = [cal["etas"][k]["spread_ratio"] for k in keys]
    bias = [cal["etas"][k]["bias_mm"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=INK["surface"])
    for ax, vals, ylab, title, sub, ref in (
        (axes[0], ratio, "sampled / observed spread", "Ensemble spread ratio",
         "1.0 = calibrated; below = over-confident", 1.0),
        (axes[1], bias, "precipitation bias (mm/day)", "Resulting mean bias",
         "0 = unbiased; observed mean 5.63 mm/day", 0.0),
    ):
        x = np.arange(len(keys))
        ax.bar(x, vals, color=PALETTE["diffusion"], width=0.5, linewidth=0)
        _style(ax, grid_axis="y")
        ax.axhline(ref, color=INK["primary"], lw=1.4, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylab)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.5, color=INK["secondary"])
        lo, hi = min(vals + [ref]), max(vals + [ref])
        pad = (hi - lo) * 0.18 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        _title(ax, title, sub)
    fig.suptitle("Diffusion sampler calibration on the dev year", fontsize=12, color=INK["primary"],
                 x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3.0)
    _save(fig, "10_diffusion_calibration.png",
          "After the x0 fix the ensemble recovers only ~56% of the observed sub-grid spread at every noise "
          "level, leaving the field ~2 mm/day too dry. Under-dispersed, i.e. over-confident: an ensemble "
          "spanning half the true uncertainty would under-warn at a flood threshold.")


def fig_diffusion_parameterisation():
    """The epsilon vs x0 failure, recorded from the two inference runs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), facecolor=INK["surface"])
    panels = [
        ("Test RMSE (mm/day)", ["epsilon-prediction", "x0-prediction"], [23.212, 5.148],
         ["#e34948", PALETTE["diffusion"]], "{:.3f}", 4.919, "dashed = bilinear 4.919"),
        ("Test bias (mm/day)", ["epsilon-prediction", "x0-prediction"], [4.491, -0.674],
         ["#e34948", PALETTE["diffusion"]], "{:+.3f}", 0.0, "dashed = unbiased"),
        ("Residual spread (scaled)", ["epsilon-prediction", "x0-prediction", "AORC observed"],
         [3.06, 0.58, 1.64], ["#e34948", PALETTE["diffusion"], INK["primary"]], "{:.2f}", 1.64,
         "dashed = observed 1.64"),
    ]
    for ax, (label, names, vals, cols, fmt, ref, sub) in zip(axes, panels):
        _barh_panel(ax, names, vals, label, fmt=fmt, ref=ref, ref_label=sub, colors=cols,
                    ticklabels=names, zero_line=(min(vals) < 0))
    fig.suptitle("Diffusion: epsilon-prediction was numerically unstable", fontsize=12,
                 color=INK["primary"], x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.5)
    _save(fig, "11_diffusion_parameterisation_fix.png",
          "The cosine schedule drives abar to 1e-8 at t=T, so recovering the residual divides by ~1e-4 and "
          "amplifies error 10,000x: sampled spread ran 87% too wide and the field was +4.5 mm biased. "
          "Predicting x0 divides by sqrt(1-abar) ~ 1 instead, leaving the opposite, milder failure "
          "(spread 56% of observed).")


def fig_spectral_summary(cmp: dict):
    names = [n for n in cmp["products"] if n != "nearest"]
    vals = [cmp["products"][n]["spectral_ratio_sub10km"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=INK["surface"])
    _barh_panel(ax, names, vals, "Fine-scale variance retained", ref=1.0,
                ref_label="predicted / observed power below 10 km; dashed = realistic (1.0). "
                          "IMERG-nearest omitted (32.7, blocky artifacts not skill)")
    fig.tight_layout()
    _save(fig, "03_spectral_ratio.png",
          "Every product carries 2-13% of the observed sub-10 km variance. Minimising squared error returns "
          "the conditional mean, which for convective rain is genuinely smooth - no architecture fixes this.")


# --------------------------------------------------------------------------- #
def copy_existing():
    """Bring the map/plot figures produced by validate and compare into the set."""
    mapping = [
        ("results/comparison/spectra.png", "02_power_spectra.png",
         "Radially averaged power spectra on the 120 wettest test days. All deterministic products fall "
         "1-2 orders of magnitude below AORC below ~30 km; IMERG-nearest sits above it (blocky 10 km steps)."),
        (sorted(Path("results/comparison").glob("example_day_*.png"))[0].as_posix()
         if list(Path("results/comparison").glob("example_day_*.png")) else "",
         "19_example_wettest_day.png",
         "The wettest test day at 1 km for every product against AORC - the visual counterpart to the "
         "spectra: deterministic fields are smooth blobs, AORC has structure."),
        ("results/comparison/fss.png", "12_fractions_skill_score.png",
         "Fractions Skill Score against neighbourhood size at 1/10/30 mm - credit for placing rain "
         "approximately correctly rather than exactly."),
        ("results/spatial_plots/bias_rmse_maps.png", "13_spatial_bias_rmse_maps.png",
         "Mean field, per-cell bias and per-cell RMSE for the tree product and both baselines, 2019-2020."),
        ("results/spatial_plots/nse_maps.png", "14_per_cell_nse_maps.png",
         "Per-cell Nash-Sutcliffe efficiency. Median cell NSE is 0.68 for the tree product."),
        ("results/qq_plot.png", "15_quantile_quantile.png",
         "Quantile-quantile against AORC on 200k sampled test cells; departures at the top end are the "
         "extreme-value underprediction."),
        ("results/timeseries_comparisons/timeseries_locations.png", "16_city_timeseries.png",
         "Daily series at seven towns across the domain for the test years."),
        ("results/feature_importance_stage1_10km.png", "17_feature_importance_stage1.png",
         "Gain importance for the 10 km tree stage. IMERG itself and its neighbourhood mean dominate; "
         "the climatological percentile ranks third."),
        ("results/feature_importance_stage2_1km.png", "18_feature_importance_stage2.png",
         "Gain importance for the 1 km residual stage - the stage that was ultimately rejected on the hold-out."),
    ]
    for src, dst, cap in mapping:
        if not src:
            continue
        p = Path(src)
        if p.exists():
            shutil.copy2(p, FIG_DIR / dst)
            INDEX.append((dst, cap))
            print(f"  {dst}  (copied)")


def write_index(cmp: dict):
    lines = ["# Result figures", "",
             f"Domain: Austin, TX (`[-99.0, 28.5, -96.0, 32.0]`). Test period "
             f"{cmp['period'][0]} to {cmp['period'][1]} ({cmp['n_days']} days) against AORC at 1 km.",
             "", "Regenerate with `python -m src.figures`.", "", "| figure | what it shows |", "|---|---|"]
    for name, cap in sorted(INDEX):
        lines.append(f"| [`{name}`]({name}) | {cap} |")
    lines += ["", "## Companion figure sets", "",
              "* [`explainer/`](explainer/) - conceptual figures for `docs/METRICS.md`: what the metrics "
              "measure and why RMSE alone misleads on this task.",
              "* [`paper/`](paper/) - publication-grade versions (300 dpi + PDF, Okabe-Ito palette, "
              "panel labels), including a Taylor diagram and the accuracy-realism trade-off.",
              "", "## Reading order", "",
              "1. **01, 03, 04, 05** - what the finished products do on the test years.",
              "2. **02, 12** - whether the fields look like rain (spectra, FSS) rather than merely scoring well.",
              "3. **06-09** - the training decisions that got there, each judged on the 2018 dev year only.",
              "4. **10, 11** - the diffusion model's two failure modes and the fix between them.",
              "5. **13-19** - spatial, per-location and single-day detail.", "",
              "Figures 13-18 are the validation plots for the tree product (the `ml` method in",
              "`results/validation_metrics.json`); 01-12 and 19 score every product together.", ""]
    (FIG_DIR / "README.md").write_text("\n".join(lines))
    print(f"  README.md ({len(INDEX)} figures)")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cmp = _load("results/comparison/model_comparison.json")
    val = _load("results/validation_metrics.json")
    print("rendering ->", FIG_DIR)
    fig_test_metrics(cmp)
    fig_spectral_summary(cmp)
    fig_intensity(val)
    fig_detection(cmp)
    fig_loss_ablation()
    fig_capacity_ablation()
    fig_extremes_ablation()
    fig_overfitting()
    fig_selection_criterion()
    fig_diffusion_calibration()
    fig_diffusion_parameterisation()
    copy_existing()
    write_index(cmp)


if __name__ == "__main__":
    main()
