"""Phase 4: apply the trained model(s) to the full IMERG timeline at 1 km."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .feature_engineering import (
    source_prefix,
    FineFeatureBuilder,
    coarse_feature_names,
    coarse_features_10km,
    features_to_matrix,
    fine_features_1km,
)
from .train_model import load_model
from .utils import (LOG, GridPair, coarsen_mean, ensure_dirs, open_dataset, save_json, to_netcdf,
                    upsample_bilinear, upsample_nearest)


def predict_coarse(model, feats: xr.Dataset, names: list[str]) -> xr.DataArray:
    """Stage-1 prediction on the coarse grid for every day in ``feats``."""
    X = features_to_matrix(feats, names)
    ok = ~np.isnan(X).any(axis=1)
    out = np.full(X.shape[0], np.nan, dtype=np.float32)
    if ok.any():
        out[ok] = np.clip(model.predict(X[ok]), 0, None)
    nt, ny, nx = feats.sizes["time"], feats.sizes["lat"], feats.sizes["lon"]
    da = xr.DataArray(out.reshape(nt, ny, nx), dims=("time", "lat", "lon"),
                      coords={"time": feats["time"], "lat": feats["lat"], "lon": feats["lon"]},
                      attrs={"units": "mm/day", "long_name": "stage-1 corrected precipitation (10 km)"})
    return da


def downscale_day(imerg_day_10km: xr.DataArray, nlcd_1km: xr.Dataset, trained_model, clim: xr.DataArray,
                  grids: GridPair, cfg: dict, history: xr.DataArray | None = None) -> xr.DataArray:
    """Convenience wrapper: stage-1 prediction for one day (10 km).

    ``history`` may hold the preceding days so the rolling mean is meaningful.
    """
    da = imerg_day_10km.expand_dims("time") if "time" not in imerg_day_10km.dims else imerg_day_10km
    if history is not None:
        da = xr.concat([history, da], "time")
    feats = coarse_features_10km(da, nlcd_1km, clim, grids, cfg)
    pred = predict_coarse(trained_model, feats, coarse_feature_names(cfg))
    return pred.isel(time=-1)


def upsample_to_1km(pred_10km: xr.DataArray, fine_static: xr.Dataset, grids: GridPair, method: str = "bilinear+lapse",
                    cfg: dict | None = None, fine_model=None, builder: FineFeatureBuilder | None = None,
                    tslice: slice | None = None) -> xr.DataArray:
    """Upsample a (time, lat, lon) coarse prediction to the fine grid.

    Methods: ``bilinear`` | ``bilinear+lapse`` (elevation lapse-rate correction)
    | ``residual`` (stage-2 model adds sub-grid residuals). With
    ``upsampling.conserve_mass`` the block means are rescaled to match the
    coarse prediction.
    """
    ucfg = (cfg or {}).get("upsampling", {})
    bl = upsample_bilinear(pred_10km, grids)
    if method == "bilinear":
        out = bl
    elif method == "bilinear+lapse":
        alpha = float(ucfg.get("lapse_alpha", -0.0002))
        factor = 1.0 + alpha * fine_static["dem_anom"]
        out = (bl * factor.clip(min=0.0)).clip(min=0.0)
    elif method == "residual":
        if fine_model is None or builder is None:
            raise ValueError("residual method needs fine_model and builder")
        fields = builder.dynamic_fields(tslice if tslice is not None else slice(0, pred_10km.sizes["time"]))
        arr = np.empty(bl.shape, dtype=np.float32)
        for d in range(bl.sizes["time"]):
            X = builder.matrix(fields, d)
            ok = ~np.isnan(X).any(axis=1)
            res = np.zeros(X.shape[0], dtype=np.float32)
            if ok.any():
                res[ok] = fine_model.predict(X[ok])
            arr[d] = np.clip(fields["pred_bl"][d].reshape(-1) + res, 0, None).reshape(bl.shape[1:])
        out = bl.copy(data=arr)
    else:
        raise ValueError(f"Unknown upsampling method {method}")
    if ucfg.get("conserve_mass", False):
        block = coarsen_mean(out, grids)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (pred_10km / block).where(block > 1e-3, 1.0).fillna(1.0).clip(0.0, 5.0)
        out = out * upsample_nearest(ratio, grids)
    out.attrs = {"units": "mm/day", "long_name": f"downscaled precipitation ({method})"}
    return out.astype(np.float32)


def downscale_timeseries(cfg: dict, grids: GridPair, method: str | None = None) -> Path:
    """Run the full pipeline over ``time.start``..``time.end`` and write NetCDF.

    Output: ``results/imerg_downscaled_1km_{y0}_{y1}.nc`` with ``precipitation``
    (time, lat, lon) and ``uncertainty`` (lat, lon: std of 1 km residuals vs
    AORC over the training period).
    """
    ensure_dirs(cfg)
    proc = Path(cfg["paths"]["processed"])
    mdir = Path(cfg["paths"]["models"])
    rdir = Path(cfg["paths"]["results"])
    method = method or cfg["upsampling"].get("method", "auto")
    chunk = int(cfg["upsampling"].get("time_chunk", 32))
    tcfg = cfg["time"]
    if method == "auto":
        # Use the stage-2 training-tail verdict; fall back to bilinear if no fine model exists.
        if (mdir / "downscaler_fine.pkl").exists():
            _, fm = load_model(mdir / "downscaler_fine.pkl")
            method = fm.get("recommended_method", "residual")
        else:
            method = "bilinear"
        LOG.info("Upsampling method 'auto' -> %s", method)

    coarse_model, coarse_meta = load_model(mdir / "downscaler_coarse.pkl")
    imerg = open_dataset(proc / "imerg_aligned_10km.nc")["precip"].load()
    nlcd = open_dataset(proc / "nlcd_aligned_1km.nc").load()
    clim = open_dataset(proc / "imerg_climatology_10km.nc")["imerg_sorted"].load()
    feats = coarse_features_10km(imerg, nlcd, clim, grids, cfg)
    pred_c = predict_coarse(coarse_model, feats, coarse_feature_names(cfg))
    win = int(cfg["features"].get("rolling_days", 7))
    sp = source_prefix(cfg)
    coarse = xr.Dataset({"pred": pred_c, sp: feats[sp], f"{sp}_roll{win}": feats[f"{sp}_roll{win}"],
                         f"{sp}_pct": feats[f"{sp}_pct"]})
    fine_static = fine_features_1km(nlcd, grids)
    fine_model, fine_meta = (None, {})
    builder = None
    if method == "residual":
        fine_model, fine_meta = load_model(mdir / "downscaler_fine.pkl")
        builder = FineFeatureBuilder(coarse, fine_static, grids, cfg)

    aorc_path = proc / "aorc_aligned_1km.nc"
    aorc = open_dataset(aorc_path, chunks={"time": chunk})["precip"] if aorc_path.exists() else None
    train_mask = (pred_c["time"] >= np.datetime64(tcfg["train_start"])) & (pred_c["time"] <= np.datetime64(tcfg["train_end"]))
    res_sum = np.zeros(grids.fine.shape, np.float64)
    res_sq = np.zeros(grids.fine.shape, np.float64)
    res_n = np.zeros(grids.fine.shape, np.int32)

    to_netcdf(xr.Dataset({"precip": pred_c}), proc / "stage1_pred_10km.nc")
    years = sorted(set(pd.DatetimeIndex(pred_c["time"].values).year))
    year_files = []
    nt = pred_c.sizes["time"]
    for year in years:
        yfile = rdir / f"downscaled_1km_{year}.nc"
        year_files.append(yfile)
        if yfile.exists():
            LOG.info("Downscaled %d exists, skipping", year)
            continue
        t_idx = np.where(pd.DatetimeIndex(pred_c["time"].values).year == year)[0]
        blocks = []
        for t0 in range(t_idx[0], t_idx[-1] + 1, chunk):
            t1 = min(t_idx[-1] + 1, t0 + chunk)
            sl = slice(t0, t1)
            out = upsample_to_1km(pred_c.isel(time=sl), fine_static, grids, method, cfg, fine_model, builder, sl)
            blocks.append(out)
            if aorc is not None:
                tm = train_mask.values[sl]
                if tm.any():
                    obs = aorc.isel(time=sl).values[tm]
                    r = obs - out.values[tm]
                    ok = ~np.isnan(r)
                    res_sum += np.where(ok, r, 0).sum(0)
                    res_sq += np.where(ok, r * r, 0).sum(0)
                    res_n += ok.sum(0)
            LOG.info("  downscaled %s .. %s", str(pred_c.time.values[t0])[:10], str(pred_c.time.values[t1 - 1])[:10])
        da = xr.concat(blocks, "time")
        to_netcdf(xr.Dataset({"precipitation": da}), yfile)
        LOG.info("Wrote %s", yfile)

    with np.errstate(invalid="ignore", divide="ignore"):
        var = res_sq / np.maximum(res_n, 1) - (res_sum / np.maximum(res_n, 1)) ** 2
    unc = np.sqrt(np.clip(var, 0, None)).astype(np.float32)
    unc[res_n == 0] = np.nan
    unc_da = xr.DataArray(unc, dims=("lat", "lon"), coords=grids.fine.coords(),
                          attrs={"units": "mm/day", "long_name": "std of 1 km residuals vs AORC (training period)"})

    # Merging 20+ years into a single dask-backed NetCDF write stalls (a 4.5 GB
    # compressed write from a lazy graph makes no forward progress), so the
    # per-year files are the product for long records and the readers below
    # stitch them. Short records still get a single convenient file.
    n_days = int(pred_c.sizes["time"])
    write_merged = bool(cfg["upsampling"].get("write_merged", n_days <= 3000))
    if res_n.sum() > 0:
        to_netcdf(xr.Dataset({"uncertainty": unc_da}), rdir / "downscaled_uncertainty_1km.nc")
    if not write_merged:
        meta = {"method": method, "years": [int(y) for y in years], "n_days": n_days,
                "files": [f.name for f in year_files],
                "training_period": f"{tcfg['train_start']}..{tcfg['train_end']}",
                "validation_period": f"{tcfg['val_start']}..{tcfg['val_end']}",
                "coarse_model": coarse_meta.get("model_type", ""),
                "fine_model": fine_meta.get("model_type", ""),
                "created": datetime.now(timezone.utc).isoformat()}
        save_json(meta, rdir / "downscaled_1km_index.json")
        LOG.info("Downscaled product: %d yearly files in %s (index written; merge skipped for %d days)",
                 len(year_files), rdir, n_days)
        return rdir / "downscaled_1km_index.json"

    final = rdir / f"imerg_downscaled_1km_{years[0]}_{years[-1]}.nc"
    ds = xr.open_mfdataset(year_files, combine="by_coords", engine="netcdf4", chunks={"time": chunk})
    if res_n.sum() > 0:
        ds["uncertainty"] = unc_da
    ds.attrs.update({
        "title": "IMERG precipitation downscaled to 1 km",
        "method": method,
        "coarse_model": coarse_meta.get("model_type", ""),
        "fine_model": fine_meta.get("model_type", ""),
        "training_period": f"{tcfg['train_start']}..{tcfg['train_end']}",
        "validation_period": f"{tcfg['val_start']}..{tcfg['val_end']}",
        "coarse_features": ",".join(coarse_meta.get("feature_names", [])),
        "fine_features": ",".join(fine_meta.get("feature_names", [])),
        "domain": cfg["domain"]["name"],
        "created": datetime.now(timezone.utc).isoformat(),
    })
    to_netcdf(ds, final)
    ds.close()
    LOG.info("Downscaled dataset -> %s", final)
    return final
