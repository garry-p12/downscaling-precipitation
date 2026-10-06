"""Phase 5: validate downscaled 1 km precipitation against AORC (2019-2020).

Metrics are accumulated in streaming fashion over time chunks so that a full
CONUS-scale domain never has to be held in memory. Three products are
compared on identical (day, cell) samples:

* ``ml``        the downscaled product produced by :mod:`predict`
* ``bilinear``  IMERG 10 km bilinearly interpolated to 1 km (no ML)
* ``nearest``   IMERG 10 km repeated over each 12x12 block (original IMERG)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .utils import (LOG, GridPair, ensure_dirs, evaluation_grids, open_dataset, open_downscaled,  # noqa: E402
                    save_json, subset_box, upsample_bilinear, upsample_nearest)

# --------------------------------------------------------------------------- #
# Palette (dataviz reference palette; validated categorical order)
# --------------------------------------------------------------------------- #
SERIES = {"aorc": "#0b0b0b", "ml": "#2a78d6", "bilinear": "#eb6834", "nearest": "#1baf7a"}
LABELS = {"aorc": "AORC (observed)", "ml": "ML downscaled", "bilinear": "Bilinear baseline",
          "nearest": "IMERG original (nearest)"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "axis": "#c3c2b7", "surface": "#fcfcfb"}
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_red_blue", ["#c0392b", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"])
METHODS = ("ml", "bilinear", "nearest")
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def _style_axes(ax):
    ax.set_facecolor(INK["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK["axis"])
    ax.tick_params(colors=INK["muted"], labelsize=8)
    ax.yaxis.label.set_color(INK["secondary"])
    ax.xaxis.label.set_color(INK["secondary"])
    ax.grid(True, color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Streaming accumulators
# --------------------------------------------------------------------------- #
class Sums:
    """Running sums for pairwise (pred, obs) statistics, scalar or per-cell."""

    FIELDS = ("n", "sp", "so", "spp", "soo", "spo", "se2", "sae")

    def __init__(self, shape=()):
        for f in self.FIELDS:
            setattr(self, f, np.zeros(shape, dtype=np.float64))

    def update(self, p: np.ndarray, o: np.ndarray, ok: np.ndarray, axis=None):
        p = np.where(ok, p, 0.0).astype(np.float64)
        o = np.where(ok, o, 0.0).astype(np.float64)
        e = p - o
        self.n += ok.sum(axis=axis)
        self.sp += p.sum(axis=axis)
        self.so += o.sum(axis=axis)
        self.spp += (p * p).sum(axis=axis)
        self.soo += (o * o).sum(axis=axis)
        self.spo += (p * o).sum(axis=axis)
        self.se2 += (e * e).sum(axis=axis)
        self.sae += np.abs(e).sum(axis=axis)

    def metrics(self) -> dict:
        with np.errstate(invalid="ignore", divide="ignore"):
            n = np.maximum(self.n, 1)
            mp, mo = self.sp / n, self.so / n
            vp = self.spp / n - mp**2
            vo = self.soo / n - mo**2
            cov = self.spo / n - mp * mo
            r = cov / np.sqrt(np.maximum(vp, 0) * np.maximum(vo, 0))
            nse = 1 - self.se2 / (n * vo)
            sp, so = np.sqrt(np.maximum(vp, 0)), np.sqrt(np.maximum(vo, 0))
            kge = 1 - np.sqrt((r - 1) ** 2 + (sp / so - 1) ** 2 + (mp / mo - 1) ** 2)
            out = {
                "n": self.n, "bias": mp - mo, "rmse": np.sqrt(self.se2 / n), "mae": self.sae / n,
                "pearson_r": r, "nse": nse, "kge": kge, "mean_pred": mp, "mean_obs": mo,
                "std_pred": sp, "std_obs": so,
            }
        empty = self.n == 0
        for k in ("bias", "rmse", "mae", "pearson_r", "nse", "kge", "mean_pred", "mean_obs"):
            out[k] = np.where(empty, np.nan, out[k])
        return out


class Contingency:
    def __init__(self):
        self.hits = self.misses = self.false_alarms = self.correct_neg = 0

    def update(self, p, o, ok, thr):
        pe, oe = (p >= thr) & ok, (o >= thr) & ok
        self.hits += int((pe & oe).sum())
        self.misses += int((~pe & oe).sum())
        self.false_alarms += int((pe & ~oe).sum())
        self.correct_neg += int((~pe & ~oe & ok).sum())

    def metrics(self) -> dict:
        h, m, f = self.hits, self.misses, self.false_alarms
        return {
            "hits": h, "misses": m, "false_alarms": f, "correct_negatives": self.correct_neg,
            "pod": h / (h + m) if h + m else np.nan,
            "far": f / (h + f) if h + f else np.nan,
            "csi": h / (h + m + f) if h + m + f else np.nan,
            "frequency_bias": (h + f) / (h + m) if h + m else np.nan,
        }


class Reservoir:
    """Uniform random subsample of (obs, pred_1, pred_2, ...) rows."""

    def __init__(self, k: int, n_cols: int, seed: int = 0):
        self.k, self.rng = k, np.random.default_rng(seed)
        self.buf = np.empty((k, n_cols), np.float32)
        self.seen = 0
        self.filled = 0

    def update(self, rows: np.ndarray):
        m = len(rows)
        if m == 0:
            return
        take = min(m, self.k - self.filled)
        if take > 0:
            self.buf[self.filled:self.filled + take] = rows[:take]
            self.filled += take
        rest = rows[take:]
        self.seen += take
        if len(rest):
            idx = self.rng.integers(0, self.seen + np.arange(1, len(rest) + 1))
            keep = idx < self.k
            self.buf[idx[keep]] = rest[keep]
            self.seen += len(rest)

    def data(self) -> np.ndarray:
        return self.buf[:self.filled]


def _scalar(d: dict) -> dict:
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in d.items()}


def _pattern_corr(a, b) -> float:
    """NaN-aware Pearson correlation between two equally shaped arrays/lists."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 10:
        return np.nan
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def blockiness(field: np.ndarray, factor: int) -> float:
    """Mean |jump| across coarse-block boundaries / mean |jump| inside blocks.

    ~1 for a natural field, >>1 for a field with visible 10 km grid artifacts.
    """
    dlat = np.abs(np.diff(field, axis=0))
    dlon = np.abs(np.diff(field, axis=1))
    bi = (np.arange(1, field.shape[0]) % factor) == 0
    bj = (np.arange(1, field.shape[1]) % factor) == 0
    boundary = np.concatenate([dlat[bi].ravel(), dlon[:, bj].ravel()])
    interior = np.concatenate([dlat[~bi].ravel(), dlon[:, ~bj].ravel()])
    b, i = np.nanmean(boundary), np.nanmean(interior)
    if not np.isfinite(b) or not np.isfinite(i):
        return np.nan
    return float(min(b / max(i, 1e-6), 999.0))


# --------------------------------------------------------------------------- #
# Core metric computation
# --------------------------------------------------------------------------- #
def compute_metrics(pred: np.ndarray, obs: np.ndarray, by_intensity: bool = False,
                    bins=(1.0, 10.0, 30.0), heavy_threshold: float = 30.0) -> dict:
    """Metrics for two equally shaped arrays (any shape); NaNs are ignored."""
    p, o = np.asarray(pred, np.float64).ravel(), np.asarray(obs, np.float64).ravel()
    ok = ~np.isnan(p) & ~np.isnan(o)
    s = Sums()
    s.update(p, o, ok)
    out = {"overall": _scalar(s.metrics())}
    if by_intensity:
        out["by_intensity"] = {}
        edges = [-np.inf, *bins, np.inf]
        names = ["dry", "light", "moderate", "heavy"][: len(edges) - 1]
        for name, lo, hi in zip(names, edges[:-1], edges[1:]):
            m = ok & (o >= lo) & (o < hi)
            si = Sums()
            si.update(p, o, m)
            out["by_intensity"][name] = _scalar(si.metrics())
        c = Contingency()
        c.update(p, o, ok, heavy_threshold)
        out["heavy_event_detection"] = c.metrics()
    return out


class Validator:
    """Streams the validation period and produces metrics + figures."""

    def __init__(self, cfg: dict, grids: GridPair):
        self.cfg, self.grids = cfg, grids
        vcfg = cfg["validation"]
        self.bins = list(vcfg.get("intensity_bins_mm", [1, 10, 30]))
        self.heavy = float(vcfg.get("heavy_threshold_mm", 30))
        self.class_names = ["dry", "light", "moderate", "heavy"]
        ny, nx = grids.fine.shape
        self.global_ = {m: Sums() for m in METHODS}
        self.cell = {m: Sums((ny, nx)) for m in METHODS}
        self.intensity = {m: {c: Sums() for c in self.class_names} for m in METHODS}
        self.month = {m: {k: Sums() for k in range(1, 13)} for m in METHODS}
        self.heavy_ct = {m: Contingency() for m in METHODS}
        self.daily_spatial_corr = {m: [] for m in METHODS}
        self.daily_index = []
        self.quarter_sum = {}  # (year, q) -> {name: (ny,nx) accumulation}
        self.reservoir = Reservoir(int(vcfg.get("qq_samples", 200_000)), 1 + len(METHODS))
        self.sum_obs = np.zeros((ny, nx))
        self.sum_pred = {m: np.zeros((ny, nx)) for m in METHODS}
        self.n_days = 0
        self.block = {m: [] for m in (*METHODS, "aorc")}
        self.wettest = (-1.0, None, None)  # (domain-mean obs, date, {name: field})
        self.locations = self._locate(vcfg.get("sample_locations", []))
        self.loc_series = {name: {k: [] for k in ("aorc", *METHODS)} for name in self.locations}

    def _locate(self, locs) -> dict:
        out = {}
        lat, lon = self.grids.fine.lat, self.grids.fine.lon
        lo_min, la_min, lo_max, la_max = self.grids.fine.edges
        for loc in locs:
            if not (la_min <= loc["lat"] <= la_max and lo_min <= loc["lon"] <= lo_max):
                LOG.warning("Sample location %s is outside the domain; skipped", loc["name"])
                continue
            out[loc["name"]] = (int(np.abs(lat - loc["lat"]).argmin()), int(np.abs(lon - loc["lon"]).argmin()))
        return out

    def update(self, times: np.ndarray, obs: np.ndarray, preds: dict[str, np.ndarray]):
        ok = ~np.isnan(obs)
        for m in METHODS:
            ok &= ~np.isnan(preds[m])
        months = pd.DatetimeIndex(times).month
        years = pd.DatetimeIndex(times).year
        for m in METHODS:
            p = preds[m]
            self.global_[m].update(p, obs, ok)
            self.cell[m].update(p, obs, ok, axis=0)
            edges = [-np.inf, *self.bins, np.inf]
            for cname, lo, hi in zip(self.class_names, edges[:-1], edges[1:]):
                self.intensity[m][cname].update(p, obs, ok & (obs >= lo) & (obs < hi))
            for k in np.unique(months):
                sel = months == k
                self.month[m][int(k)].update(p[sel], obs[sel], ok[sel])
            self.heavy_ct[m].update(p, obs, ok, self.heavy)
            self.sum_pred[m] += np.where(ok, p, 0).sum(0)
        self.sum_obs += np.where(ok, obs, 0).sum(0)
        self.n_days += len(times)
        for d, t in enumerate(times):
            self.daily_index.append(t)
            dom_mean = np.nanmean(np.where(ok[d], obs[d], np.nan))
            for m in METHODS:
                self.daily_spatial_corr[m].append(_pattern_corr(preds[m][d], obs[d]) if dom_mean >= 0.5 else np.nan)
            key = (int(years[d]), (int(months[d]) - 1) // 3 + 1)
            q = self.quarter_sum.setdefault(key, {k: np.zeros(obs.shape[1:]) for k in ("aorc", *METHODS)})
            q["aorc"] += np.where(ok[d], obs[d], 0)
            for m in METHODS:
                q[m] += np.where(ok[d], preds[m][d], 0)
            if dom_mean >= 1.0:
                self.block["aorc"].append(blockiness(obs[d], self.grids.factor))
                for m in METHODS:
                    self.block[m].append(blockiness(preds[m][d], self.grids.factor))
            if dom_mean > self.wettest[0]:
                self.wettest = (float(dom_mean), str(t)[:10], {"aorc": obs[d].copy(), **{m: preds[m][d].copy() for m in METHODS}})
            for name, (i, j) in self.locations.items():
                self.loc_series[name]["aorc"].append(float(obs[d, i, j]))
                for m in METHODS:
                    self.loc_series[name][m].append(float(preds[m][d, i, j]))
        rows = np.stack([obs[ok], *[preds[m][ok] for m in METHODS]], axis=1)
        self.reservoir.update(rows)

    # ------------------------------------------------------------------ #
    def finalize(self, slope: np.ndarray | None = None) -> dict:
        vcfg = self.cfg["validation"]
        res: dict = {"methods": {}, "comparison_table": {}, "success_criteria": {}}
        for m in METHODS:
            g = _scalar(self.global_[m].metrics())
            cell = self.cell[m].metrics()
            inten = {c: _scalar(self.intensity[m][c].metrics()) for c in self.class_names}
            months = {k: _scalar(self.month[m][k].metrics()) for k in range(1, 13)}
            seasons = {}
            for s, ms in SEASONS.items():
                acc = Sums()
                for k in ms:
                    for f in Sums.FIELDS:
                        setattr(acc, f, getattr(acc, f) + getattr(self.month[m][k], f))
                seasons[s] = _scalar(acc.metrics())
            nse_cells = cell["nse"]
            valid = ~np.isnan(nse_cells)
            spatial = {
                "median_cell_nse": float(np.nanmedian(nse_cells)),
                "frac_cells_nse_gt_0.6": float(np.mean(nse_cells[valid] > 0.6)) if valid.any() else np.nan,
                "median_cell_corr": float(np.nanmedian(cell["pearson_r"])),
                "mean_daily_pattern_corr": float(np.nanmean(self.daily_spatial_corr[m])),
                "blockiness_ratio": float(np.nanmean(self.block[m])) if np.isfinite(self.block[m]).any() else np.nan,
            }
            if slope is not None:
                thr = max(float(vcfg.get("complex_terrain_slope_deg", 5.0)), float(np.nanquantile(slope, 0.8)))
                ct = valid & (slope >= thr)
                spatial["complex_terrain_slope_threshold_deg"] = thr
                spatial["complex_terrain_n_cells"] = int(ct.sum())
                spatial["complex_terrain_median_nse"] = float(np.nanmedian(nse_cells[ct])) if ct.any() else np.nan
                spatial["complex_terrain_frac_nse_gt_0.6"] = float(np.mean(nse_cells[ct] > 0.6)) if ct.any() else np.nan
            quarters = {}
            for (y, q), acc in sorted(self.quarter_sum.items()):
                idx = [i for i, t in enumerate(self.daily_index) if pd.Timestamp(t).year == y and (pd.Timestamp(t).month - 1) // 3 + 1 == q]
                quarters[f"{y}Q{q}"] = {
                    "accumulation_pattern_corr": _pattern_corr(acc[m], acc["aorc"]),
                    "mean_daily_pattern_corr": float(np.nanmean([self.daily_spatial_corr[m][i] for i in idx])) if idx else np.nan,
                }
            res["methods"][m] = {
                "overall": g, "by_intensity": inten, "heavy_event_detection": self.heavy_ct[m].metrics(),
                "by_month": months, "by_season": seasons, "spatial": spatial, "by_quarter": quarters,
            }
        # Comparison table (plan 5.2)
        for metric in ("rmse", "bias", "pearson_r", "nse", "kge", "mae"):
            res["comparison_table"][metric] = {m: res["methods"][m]["overall"][metric] for m in METHODS}
        res["comparison_table"]["pod_heavy"] = {m: res["methods"][m]["heavy_event_detection"]["pod"] for m in METHODS}
        res["comparison_table"]["median_cell_nse"] = {m: res["methods"][m]["spatial"]["median_cell_nse"] for m in METHODS}
        res["comparison_table"]["blockiness_ratio"] = {m: res["methods"][m]["spatial"]["blockiness_ratio"] for m in METHODS}

        ml, bl = res["methods"]["ml"], res["methods"]["bilinear"]
        rmse_red = 1 - ml["overall"]["rmse"] / bl["overall"]["rmse"] if bl["overall"]["rmse"] else np.nan
        aorc_block = float(np.nanmean(self.block["aorc"])) if self.block["aorc"] else np.nan
        res["success_criteria"] = {
            "rmse_reduction_vs_bilinear": {"value": rmse_red, "target": ">= 0.20", "pass": bool(rmse_red >= 0.20)},
            "median_cell_nse": {"value": ml["spatial"]["median_cell_nse"], "target": "> 0.6",
                                 "pass": bool(ml["spatial"]["median_cell_nse"] > 0.6)},
            "frac_cells_nse_gt_0.6": {"value": ml["spatial"]["frac_cells_nse_gt_0.6"], "target": "report"},
            "heavy_event_pod": {"value": ml["heavy_event_detection"]["pod"], "target": ">= 0.80",
                                "pass": bool(ml["heavy_event_detection"]["pod"] >= 0.80)
                                if not np.isnan(ml["heavy_event_detection"]["pod"]) else None},
            "no_spatial_artifacts": {"value": ml["spatial"]["blockiness_ratio"], "aorc_reference": aorc_block,
                                     "nearest_reference": res["methods"]["nearest"]["spatial"]["blockiness_ratio"],
                                     "target": "< 1.5", "pass": bool(ml["spatial"]["blockiness_ratio"] < 1.5)
                                     if not np.isnan(ml["spatial"]["blockiness_ratio"]) else None},
        }
        res["n_days"] = self.n_days
        res["period"] = [str(self.daily_index[0])[:10], str(self.daily_index[-1])[:10]] if self.daily_index else None
        return res


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _imshow(ax, field, grid, cmap, vmin, vmax, title):
    lo_min, la_min, lo_max, la_max = grid.edges
    im = ax.imshow(field, origin="lower", extent=(lo_min, lo_max, la_min, la_max), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK["primary"], loc="left")
    ax.tick_params(colors=INK["muted"], labelsize=7)
    for s in ax.spines.values():
        s.set_color(INK["axis"])
    return im


def plot_spatial_comparison(v: Validator, rdir: Path) -> Path:
    """Mean precipitation, bias and RMSE maps for all products (plan 7 fig 1)."""
    grid = v.grids.fine
    n = max(v.n_days, 1)
    mean_obs = v.sum_obs / n
    cells = {m: v.cell[m].metrics() for m in METHODS}
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), facecolor=INK["surface"])
    vmax = np.nanpercentile(mean_obs, 99)
    _imshow(axes[0, 0], mean_obs, grid, SEQ_CMAP, 0, vmax, "Mean daily precip - AORC (mm/day)")
    for k, m in enumerate(METHODS, start=1):
        im = _imshow(axes[0, k], v.sum_pred[m] / n, grid, SEQ_CMAP, 0, vmax, f"Mean daily precip - {LABELS[m]}")
    fig.colorbar(im, ax=axes[0, :], fraction=0.02, pad=0.01)
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    bmax = max(np.nanpercentile(np.abs(cells[m]["bias"]), 98) for m in METHODS) or 0.1
    rmax = max(np.nanpercentile(cells[m]["rmse"], 98) for m in METHODS) or 0.1
    for k, m in enumerate(METHODS, start=1):
        im_b = _imshow(axes[1, k], cells[m]["bias"], grid, DIV_CMAP, -bmax, bmax, f"Bias - {LABELS[m]} (mm/day)")
        im_r = _imshow(axes[2, k], cells[m]["rmse"], grid, SEQ_CMAP, 0, rmax, f"RMSE - {LABELS[m]} (mm/day)")
    fig.colorbar(im_b, ax=axes[1, :], fraction=0.02, pad=0.01)
    fig.colorbar(im_r, ax=axes[2, :], fraction=0.02, pad=0.01)
    fig.suptitle(f"Validation period {v.daily_index[0].astype('datetime64[D]')} .. {v.daily_index[-1].astype('datetime64[D]')}",
                 fontsize=11, color=INK["primary"], x=0.01, ha="left")
    out = rdir / "spatial_plots" / "bias_rmse_maps.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_nse_maps(v: Validator, rdir: Path) -> Path:
    grid = v.grids.fine
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=INK["surface"])
    for ax, m in zip(axes, METHODS):
        im = _imshow(ax, v.cell[m].metrics()["nse"], grid, DIV_CMAP, -1, 1, f"Per-cell NSE - {LABELS[m]}")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    out = rdir / "spatial_plots" / "nse_maps.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_example_day(v: Validator, rdir: Path) -> Path | None:
    if v.wettest[1] is None:
        return None
    _, date, fields = v.wettest
    grid = v.grids.fine
    vmax = np.nanpercentile(fields["aorc"], 99.5)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), facecolor=INK["surface"])
    for ax, m in zip(axes, ("aorc", "nearest", "bilinear", "ml")):
        im = _imshow(ax, fields[m], grid, SEQ_CMAP, 0, vmax, f"{LABELS[m]} - {date} (mm/day)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    out = rdir / "spatial_plots" / f"example_day_{date}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spatial_corr_windows(res: dict, rdir: Path) -> Path:
    quarters = list(res["methods"]["ml"]["by_quarter"])
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=INK["surface"])
    _style_axes(ax)
    x = np.arange(len(quarters))
    for m in METHODS:
        y = [res["methods"][m]["by_quarter"][q]["accumulation_pattern_corr"] for q in quarters]
        ax.plot(x, y, color=SERIES[m], lw=2, marker="o", ms=5, label=LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=0)
    ax.set_ylabel("Spatial correlation of 3-month accumulation")
    ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    out = rdir / "spatial_plots" / "spatial_corr_quarterly.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def quantile_quantile_plot(v: Validator, rdir: Path) -> Path:
    data = v.reservoir.data()
    obs_q = np.sort(data[:, 0])
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=INK["surface"])
    _style_axes(ax)
    hi = max(1.0, float(obs_q[-1]))
    ax.plot([0, hi], [0, hi], color=INK["axis"], lw=1, ls="--", label="1:1")
    for k, m in enumerate(METHODS, start=1):
        pq = np.sort(data[:, k])
        idx = np.unique(np.round(np.geomspace(1, len(pq), 300)).astype(int) - 1)
        ax.plot(obs_q[idx], pq[idx], color=SERIES[m], lw=0, marker="o", ms=4, alpha=0.8, label=LABELS[m])
    ax.set_xlabel("AORC quantiles (mm/day)")
    ax.set_ylabel("Product quantiles (mm/day)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Quantile-quantile, 1 km validation samples", fontsize=10, loc="left", color=INK["primary"])
    out = rdir / "qq_plot.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_timeseries_sample(v: Validator, rdir: Path) -> list[Path]:
    """Daily series at the configured locations (plan 7 fig 2), plus CSVs."""
    outs = []
    times = pd.DatetimeIndex(v.daily_index)
    tdir = rdir / "timeseries_comparisons"
    names = list(v.locations)
    if not names:
        return outs
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.2 * len(names)), sharex=True, facecolor=INK["surface"])
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        _style_axes(ax)
        s = v.loc_series[name]
        df = pd.DataFrame({"time": times, **{k: s[k] for k in ("aorc", *METHODS)}})
        df.to_csv(tdir / f"timeseries_{name.replace(' ', '_')}.csv", index=False)
        ax.plot(times, s["aorc"], color=SERIES["aorc"], lw=1.6, label=LABELS["aorc"])
        ax.plot(times, s["ml"], color=SERIES["ml"], lw=1.2, label=LABELS["ml"])
        ax.plot(times, s["bilinear"], color=SERIES["bilinear"], lw=1.0, alpha=0.8, label=LABELS["bilinear"])
        r_ml = _pattern_corr(s["aorc"], s["ml"])
        r_bl = _pattern_corr(s["aorc"], s["bilinear"])
        ax.set_title(f"{name}   r(ML)={r_ml:.2f}  r(bilinear)={r_bl:.2f}", fontsize=9, loc="left", color=INK["primary"])
        ax.set_ylabel("mm/day")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    out = tdir / "timeseries_locations.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    outs.append(out)
    return outs


def plot_seasonal(res: dict, rdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=INK["surface"])
    seasons = list(SEASONS)
    x = np.arange(len(seasons))
    w = 0.26
    for ax, metric in zip(axes, ("rmse", "nse")):
        _style_axes(ax)
        for k, m in enumerate(METHODS):
            y = [res["methods"][m]["by_season"][s][metric] for s in seasons]
            ax.bar(x + (k - 1) * w, y, width=w - 0.03, color=SERIES[m], label=LABELS[m], linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.set_ylabel({"rmse": "RMSE (mm/day)", "nse": "NSE"}[metric])
        if metric == "nse":
            ax.axhline(0, color=INK["axis"], lw=1)
    axes[0].legend(frameon=False, fontsize=8)
    out = rdir / "seasonal_metrics.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_importance(models_dir: Path, rdir: Path, top: int = 15) -> list[Path]:
    outs = []
    for tag, fname in (("stage1_10km", "feature_importance_coarse.csv"), ("stage2_1km", "feature_importance_fine.csv")):
        p = models_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p).head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1), facecolor=INK["surface"])
        _style_axes(ax)
        ax.barh(df["feature"], df["importance"], color=SERIES["ml"], height=0.6, linewidth=0)
        for y, val in enumerate(df["importance"]):
            ax.text(val, y, f" {val:.3f}", va="center", fontsize=8, color=INK["secondary"])
        ax.set_xlabel("Normalised gain importance")
        ax.set_title(f"Feature importance - {tag}", fontsize=10, loc="left", color=INK["primary"])
        out = rdir / f"feature_importance_{tag}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        outs.append(out)
    return outs


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def write_report(res: dict, rdir: Path, figures: list) -> Path:
    lines = ["# Validation report", "", f"Period: {res['period'][0]} .. {res['period'][1]} ({res['n_days']} days)", "",
             "| Metric | ML downscaled | Bilinear baseline | IMERG orig (nearest) |", "|---|---|---|---|"]
    for metric, row in res["comparison_table"].items():
        lines.append(f"| {metric} | " + " | ".join(f"{row[m]:.3f}" if row[m] == row[m] else "n/a" for m in METHODS) + " |")
    lines += ["", "## Success criteria", ""]
    for k, c in res["success_criteria"].items():
        val = c.get("value")
        vs = f"{val:.3f}" if isinstance(val, float) and val == val else str(val)
        status = {True: "PASS", False: "FAIL", None: "n/a"}.get(c.get("pass"), "report")
        lines.append(f"- **{k}**: {vs} (target {c['target']}) -> {status}")
    lines += ["", "## By intensity class (ML)", "", "| class | n | bias | rmse | r | nse |", "|---|---|---|---|---|---|"]
    for c, d in res["methods"]["ml"]["by_intensity"].items():
        lines.append(f"| {c} | {int(d['n'])} | {d['bias']:.3f} | {d['rmse']:.3f} | {d['pearson_r']:.3f} | {d['nse']:.3f} |")
    lines += ["", "## Figures", ""] + [f"- {Path(f).relative_to(rdir.parent)}" for f in figures if f]
    out = rdir / "validation_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def validate(cfg: dict, grids: GridPair, downscaled_path: str | Path | None = None) -> dict:
    """Run the full validation: metrics JSON, comparison table, figures."""
    ensure_dirs(cfg)
    proc = Path(cfg["paths"]["processed"])
    rdir = Path(cfg["paths"]["results"])
    mdir = Path(cfg["paths"]["models"])
    tcfg = cfg["time"]
    chunk = int(cfg["upsampling"].get("time_chunk", 32))
    val_slice = slice(tcfg["val_start"], tcfg["val_end"])

    if downscaled_path is None:
        ml_all = open_downscaled(rdir, chunks={"time": chunk})
        downscaled_path = str(rdir)
    else:
        ml_all = open_dataset(downscaled_path, chunks={"time": chunk})["precipitation"]
    ml = ml_all.sel(time=val_slice)
    aorc = open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": chunk})["precip"].sel(time=val_slice)
    imerg = open_dataset(proc / "imerg_aligned_10km.nc")["precip"].sel(time=val_slice).load()
    nlcd = open_dataset(proc / "nlcd_aligned_1km.nc").load()
    times = ml["time"].values
    if len(times) != aorc.sizes["time"] or len(times) != imerg.sizes["time"]:
        raise ValueError("Validation inputs have inconsistent time axes")

    eval_grids, eval_bbox = evaluation_grids(cfg, grids)
    if eval_bbox:
        LOG.info("scoring restricted to evaluation box %s (%s)", eval_bbox,
                 (cfg.get("evaluation") or {}).get("name", "sub-domain"))
    v = Validator(cfg, eval_grids)
    for t0 in range(0, len(times), chunk):
        sl = slice(t0, min(len(times), t0 + chunk))
        obs_da = aorc.isel(time=sl)
        im_da = imerg.isel(time=sl)
        bil = upsample_bilinear(im_da, grids)
        nea = upsample_nearest(im_da, grids)
        ml_da = ml.isel(time=sl)
        if eval_bbox:
            obs_da, bil, nea, ml_da = (subset_box(x, eval_bbox) for x in (obs_da, bil, nea, ml_da))
        obs = obs_da.values.astype(np.float32)
        preds = {"ml": ml_da.values.astype(np.float32),
                 "bilinear": bil.values.astype(np.float32),
                 "nearest": nea.values.astype(np.float32)}
        v.update(times[sl], obs, preds)
        LOG.info("  validated %s .. %s", str(times[sl][0])[:10], str(times[sl][-1])[:10])

    slope = subset_box(nlcd["slope"], eval_bbox) if eval_bbox else nlcd["slope"]
    res = v.finalize(slope=slope.values)
    res["downscaled_file"] = str(downscaled_path)
    res["method"] = ml_all.attrs.get("method", cfg["upsampling"].get("method"))
    res["evaluation_bbox"] = eval_bbox
    save_json(res, rdir / "validation_metrics.json")

    figures = [plot_spatial_comparison(v, rdir), plot_nse_maps(v, rdir), plot_example_day(v, rdir),
               plot_spatial_corr_windows(res, rdir), quantile_quantile_plot(v, rdir), plot_seasonal(res, rdir)]
    figures += plot_timeseries_sample(v, rdir)
    figures += plot_feature_importance(mdir, rdir)
    report = write_report(res, rdir, figures)
    print_results(res)
    LOG.info("Validation metrics -> %s ; report -> %s", rdir / "validation_metrics.json", report)
    return res


def print_results(res: dict) -> None:
    print("\n=== Validation (1 km vs AORC) ===")
    print(f"{'metric':<20}{'ML':>12}{'bilinear':>12}{'nearest':>12}")
    for metric, row in res["comparison_table"].items():
        print(f"{metric:<20}" + "".join(f"{row[m]:>12.3f}" if row[m] == row[m] else f"{'n/a':>12}" for m in METHODS))
    print("\nSuccess criteria:")
    for k, c in res["success_criteria"].items():
        val = c.get("value")
        vs = f"{val:.3f}" if isinstance(val, float) and val == val else str(val)
        status = {True: "PASS", False: "FAIL", None: "n/a"}.get(c.get("pass"), "report")
        print(f"  {k:<32} {vs:>8}  target {c['target']:<8} {status}")
