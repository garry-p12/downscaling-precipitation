"""Map figures: observed vs downscaled precipitation fields.

    python -m src.field_figures        # -> results/figures/2x_*.png

The metric figures say which product scores best; these say what the products
actually look like. That matters here because the headline scores are nearly
identical across four model classes while the fields are not: a conditional-mean
field can win RMSE while looking nothing like rain.

Produces, for the 2019-2020 test period:
    20  a heavy day, full domain, every product beside AORC
    21  the same day zoomed to ~70 km, where the texture difference is visible
    22  error maps (product - AORC) for that day
    23  test-period mean field per product
    24  three contrasting days x every product
    25  diffusion ensemble mean vs a single member vs AORC
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .figures import FIG_DIR, INDEX, _save, _title, write_index, _load  # noqa: E402
from .model_comparison import LABELS  # noqa: E402
from .utils import grids_from_config, load_config, upsample_bilinear, upsample_nearest  # noqa: E402
from .validation import DIV_CMAP, INK, SEQ_CMAP  # noqa: E402

ORDER = ["aorc", "nearest", "bilinear", "xgboost", "cnn", "swin", "diffusion", "diffusion_member"]
ALL_LABELS = {**LABELS, "aorc": "AORC (observed)"}


def deep_dir(cfg) -> Path:
    """Newest available deep products: v2 (2000-2020) > final (2015-2020) > deep."""
    rdir = Path(cfg["paths"]["results"])
    for name in ("v3", "v2", "final", "deep"):
        if (rdir / name).exists() and list((rdir / name).glob("*_1km_*.nc")):
            return rdir / name
    return rdir / "deep"


def load_fields(cfg, grids, val_slice):
    """Lazy handles for every 1 km product plus AORC, on the test period."""
    proc = Path(cfg["paths"]["processed"])
    rdir = Path(cfg["paths"]["results"])
    out = {"aorc": xr.open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": 32})["precip"].sel(time=val_slice)}
    imerg = xr.open_dataset(proc / "imerg_aligned_10km.nc")["precip"].sel(time=val_slice).load()
    out["nearest"] = upsample_nearest(imerg, grids)
    out["bilinear"] = upsample_bilinear(imerg, grids)
    xgb = sorted(rdir.glob("imerg_downscaled_1km_*.nc"))
    if xgb:
        out["xgboost"] = xr.open_dataset(xgb[-1], chunks={"time": 32})["precipitation"].sel(time=val_slice)
    for name in ("cnn", "swin", "diffusion"):
        f = sorted(deep_dir(cfg).glob(f"{name}_1km_*.nc"))
        if not f:
            continue
        ds = xr.open_dataset(f[-1], chunks={"time": 32})
        out[name] = ds["precipitation"].sel(time=val_slice)
        if "precipitation_member0" in ds:
            out[f"{name}_member"] = ds["precipitation_member0"].sel(time=val_slice)
    return out


def _extent(grid, box=None):
    if box is None:
        lo_min, la_min, lo_max, la_max = grid.edges
        return (lo_min, lo_max, la_min, la_max)
    return box


def _panel(ax, field, extent, cmap, vmin, vmax, title, sub=None):
    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    _title(ax, title, sub)
    ax.tick_params(colors=INK["muted"], labelsize=7)
    for s in ax.spines.values():
        s.set_color(INK["axis"])
    return im


def _grid_of_products(fields, names, day, grids, sl=None, cmap=SEQ_CMAP, diff=False,
                      vmax=None, ncol=4, figsize_scale=3.7):
    """Render one day for a list of products; returns (fig, axes, im)."""
    obs = fields["aorc"].sel(time=day).values
    if sl:
        obs = obs[sl]
    if vmax is None:
        vmax = float(np.nanpercentile(obs, 99.5)) or 1.0
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize_scale * ncol, figsize_scale * nrow * 0.92),
                             facecolor=INK["surface"])
    axes = np.atleast_1d(axes).ravel()
    grid = grids.fine
    if sl:
        lat = grid.lat[sl[0]]
        lon = grid.lon[sl[1]]
        extent = (lon[0], lon[-1], lat[0], lat[-1])
    else:
        extent = _extent(grid)
    im = None
    for ax, n in zip(axes, names):
        f = fields[n].sel(time=day).values
        if sl:
            f = f[sl]
        if diff and n != "aorc":
            f = f - obs
            im = _panel(ax, f, extent, DIV_CMAP, -vmax, vmax, ALL_LABELS.get(n, n))
        else:
            im = _panel(ax, f, extent, cmap, 0, vmax, ALL_LABELS.get(n, n))
    for ax in axes[len(names):]:
        ax.axis("off")
    return fig, axes, im


# --------------------------------------------------------------------------- #
def fig_heavy_day(fields, grids, day, names):
    fig, axes, im = _grid_of_products(fields, names, day, grids)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("mm/day", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    fig.suptitle(f"Heaviest test day, {str(day)[:10]} - full domain", fontsize=12,
                 color=INK["primary"], x=0.005, ha="left")
    _save(fig, "20_fields_heavy_day_domain.png",
          f"Every 1 km product against AORC for {str(day)[:10]}, the wettest day of the test period. "
          "Same colour scale throughout. The learned products track the large-scale pattern but render it "
          "as smooth blobs where AORC has sharp cores.")


def pick_texture_window(obs: np.ndarray, grids, half_deg: float = 0.32):
    """Sub-window with the most small-scale structure in the observed field.

    Texture differences are invisible in a uniformly wet window, so the box is
    chosen where AORC's local standard deviation is highest rather than at a
    fixed city centre.
    """
    from scipy import ndimage

    n = int(round(half_deg * 2 / grids.fine.res))
    loc_mean = ndimage.uniform_filter(np.nan_to_num(obs), size=15, mode="nearest")
    loc_var = ndimage.uniform_filter(np.nan_to_num(obs) ** 2, size=15, mode="nearest") - loc_mean**2
    score = ndimage.uniform_filter(np.sqrt(np.clip(loc_var, 0, None)), size=n, mode="constant")
    h, w = obs.shape
    score[: n // 2, :] = score[-(n // 2):, :] = 0        # keep the window inside the domain
    score[:, : n // 2] = score[:, -(n // 2):] = 0
    ci, cj = np.unravel_index(int(np.argmax(score)), score.shape)
    return slice(max(ci - n // 2, 0), min(ci + n // 2, h)), slice(max(cj - n // 2, 0), min(cj + n // 2, w))


def fig_zoom(fields, grids, day, names, half_deg=0.32):
    obs = fields["aorc"].sel(time=day).values
    sl = pick_texture_window(obs, grids, half_deg)
    lat, lon = grids.fine.lat, grids.fine.lon
    fig, axes, im = _grid_of_products(fields, names, day, grids, sl=sl, figsize_scale=3.5)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("mm/day", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    km = (sl[0].stop - sl[0].start) * grids.fine.res * 111.0
    clat, clon = lat[(sl[0].start + sl[0].stop) // 2], lon[(sl[1].start + sl[1].stop) // 2]
    fig.suptitle(f"{str(day)[:10]} zoomed to ~{km:.0f} km at {clat:.2f} N, {abs(clon):.2f} W "
                 f"- the most structured part of the field", fontsize=12, color=INK["primary"],
                 x=0.005, ha="left")
    _save(fig, "21_fields_zoom_texture.png",
          f"A ~{km:.0f} km window on {str(day)[:10]}, centred where AORC has the most small-scale "
          "structure. At this scale the difference between the products is "
          "obvious: AORC resolves a sharp rain band, IMERG-nearest renders 10 km blocks, and the "
          "deterministic products (bilinear, XGBoost, CNN, Swin) smooth the band away entirely. Only the "
          "single diffusion member carries comparable fine-scale structure. This is the spectral-ratio "
          "figure made visible.")


def fig_errors(fields, grids, day, names):
    obs = fields["aorc"].sel(time=day).values
    vmax = float(np.nanpercentile(np.abs(obs), 98)) or 1.0
    pred_names = [n for n in names if n != "aorc"]
    fig, axes, im = _grid_of_products(fields, pred_names, day, grids, diff=True, vmax=vmax, ncol=4)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("predicted - observed (mm/day)", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    fig.suptitle(f"Error fields, {str(day)[:10]} (blue = too wet, red = too dry)",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    _save(fig, "22_field_errors_heavy_day.png",
          f"Product minus AORC on {str(day)[:10]}. Errors are dipoles around the observed cores - the "
          "classic double penalty: a smooth field misses the peak and spills rain around it.")


def fig_mean_fields(fields, names):
    means = {}
    for n in names:
        v = fields[n]
        means[n] = v.mean("time").compute().values if hasattr(v.data, "dask") else v.mean("time").values
    vmax = float(np.nanpercentile(means["aorc"], 99))
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.4 * nrow), facecolor=INK["surface"])
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, n in zip(axes, names):
        im = _panel(ax, means[n], None, SEQ_CMAP, 0, vmax, ALL_LABELS.get(n, n),
                    f"domain mean {np.nanmean(means[n]):.2f} mm/day")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(names):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("mean mm/day", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    fig.suptitle("Test-period mean field (2019-2020)", fontsize=12, color=INK["primary"], x=0.005, ha="left")
    _save(fig, "23_mean_fields.png",
          "Two-year mean per product. Long averages hide the smoothing problem - every product reproduces "
          "the climatological gradient, which is why daily fields and spectra are the honest diagnostic.")


def fig_multi_day(fields, grids, days, names):
    nrow, ncol = len(days), len(names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), facecolor=INK["surface"])
    axes = np.atleast_2d(axes)
    extent = _extent(grids.fine)
    for r, (day, tag) in enumerate(days):
        obs = fields["aorc"].sel(time=day).values
        vmax = float(np.nanpercentile(obs, 99.5)) or 1.0
        for c, n in enumerate(names):
            ax = axes[r, c]
            im = _panel(ax, fields[n].sel(time=day).values, extent, SEQ_CMAP, 0, vmax,
                        ALL_LABELS.get(n, n) if r == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{tag}\n{str(day)[:10]}", fontsize=8, color=INK["secondary"])
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.015, pad=0.008)
    fig.suptitle("Three contrasting test days (each row on its own colour scale)",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    _save(fig, "24_fields_three_days.png",
          "A heavy, a moderate and a light day, each row on its own colour scale so the light day is not "
          "washed out. On the two low-intensity rows AORC shows radial spokes and banding - these are "
          "radar artifacts in the reference dataset itself, not model error, and they are a reminder that "
          "AORC is an analysis rather than truth. Every product inherits the large-scale pattern; the "
          "sharp cores are lost in all of them.")


def fig_ensemble(fields, grids, day):
    if "diffusion_member" not in fields:
        return
    names = ["aorc", "diffusion", "diffusion_member"]
    fig, axes, im = _grid_of_products(fields, names, day, grids, ncol=3, figsize_scale=4.0)
    for ax, sub in zip(axes, ["truth", "average of 6 members", "one realisation"]):
        _title(ax, ax.get_title(), sub)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("mm/day", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(colors=INK["muted"], labelsize=7)
    fig.suptitle(f"Diffusion: ensemble mean vs a single member, {str(day)[:10]}",
                 fontsize=12, color=INK["primary"], x=0.005, ha="left")
    _save(fig, "25_diffusion_ensemble_vs_member.png",
          "Averaging members smooths the field back toward the conditional mean, so the ensemble mean "
          "scores better on RMSE while the single member is the one meant to look like rain. The member "
          "does carry visible fine-scale structure the deterministic products lack - but it places it "
          "imprecisely, and domain-wide the ensemble still spans only 56% of the observed spread.")


def main():
    cfg = load_config("config.yaml")
    grids = grids_from_config(cfg)
    val = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    fields = load_fields(cfg, grids, val)
    names = [n for n in ORDER if n in fields]
    print("products:", ", ".join(names))

    dm = fields["aorc"].mean(("lat", "lon")).compute()
    sd = fields["aorc"].std(("lat", "lon")).compute()
    order = np.argsort(dm.values)
    times = fields["aorc"]["time"].values
    heavy = times[order[-1]]
    moderate = times[order[int(0.80 * len(order))]]
    light = times[order[int(0.55 * len(order))]]
    # The wettest day is often a smooth widespread front; the most *spatially
    # variable* day is the convective one where smoothing is visible.
    convective = times[int(np.argmax(sd.values))]
    print("days: heavy", str(heavy)[:10], "| moderate", str(moderate)[:10],
          "| light", str(light)[:10], "| most structured", str(convective)[:10])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_heavy_day(fields, grids, heavy, names)
    fig_zoom(fields, grids, convective, names)
    fig_errors(fields, grids, heavy, names)
    fig_mean_fields(fields, names)
    fig_multi_day(fields, grids, [(heavy, "heavy"), (moderate, "moderate"), (light, "light")],
                  [n for n in names if not n.endswith("_member")])
    fig_ensemble(fields, grids, convective)

    # merge with the metric-figure index so README lists everything
    cmp = _load("results/comparison/model_comparison.json")
    for name, cap in sorted(_existing_metric_index()):
        if name not in [i[0] for i in INDEX]:
            INDEX.append((name, cap))
    write_index(cmp)


def _existing_metric_index():
    """Re-read captions already written for the metric figures."""
    out = []
    readme = FIG_DIR / "README.md"
    if readme.exists():
        for line in readme.read_text().splitlines():
            if line.startswith("| [`"):
                name = line.split("`")[1]
                cap = line.split("|")[2].strip()
                out.append((name, cap))
    return out


if __name__ == "__main__":
    main()
