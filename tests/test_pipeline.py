"""Unit and end-to-end tests. Run: pytest -q tests"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from src import data_pipeline as dp
from src.feature_engineering import compute_derived_features, percentile_rank
from src.utils import coarsen_mean, load_config, make_grids, upsample_bilinear, upsample_nearest
from src.validation import Reservoir, blockiness, compute_metrics

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Grids & regridding
# --------------------------------------------------------------------------- #
def test_make_grids_alignment():
    g = make_grids([-98.0, 30.0, -97.0, 31.0])
    assert g.coarse.shape == (10, 10) and g.fine.shape == (120, 120)
    assert np.isclose(g.coarse.lat[0], 30.05) and np.isclose(g.fine.lat[0], 30.0)
    assert np.isclose(g.fine.lat[-1], 31.0 - 1 / 120)
    # Every fine cell's parent block matches the coarse cell it falls in.
    assert np.allclose(np.floor((g.fine.lat + 1e-9 - 30.0) / 0.1), np.arange(120) // 12)


def test_make_grids_rejects_misaligned_bbox():
    with pytest.raises(ValueError):
        make_grids([-98.05, 30.0, -97.0, 31.0])


def test_coarsen_and_upsample_roundtrip():
    g = make_grids([-98.0, 30.0, -97.5, 30.5])
    rng = np.random.default_rng(0)
    coarse = xr.DataArray(rng.random((3,) + g.coarse.shape).astype(np.float32),
                          dims=("time", "lat", "lon"), coords={"time": pd.date_range("2015-01-01", periods=3), **g.coarse.coords()})
    fine = upsample_nearest(coarse, g)
    assert fine.shape == (3,) + g.fine.shape
    back = coarsen_mean(fine, g)
    assert np.allclose(back.values, coarse.values, atol=1e-6)
    bl = upsample_bilinear(coarse, g)
    assert bl.shape == fine.shape and not np.isnan(bl.values).any()
    # Bilinear preserves the value at fine cells nearest to the coarse centres.
    i = np.abs(g.fine.lat - g.coarse.lat[2]).argmin()
    j = np.abs(g.fine.lon - g.coarse.lon[1]).argmin()
    assert abs(float(bl[0, i, j]) - float(coarse[0, 2, 1])) < 0.15
    # Missing coarse cell -> NaN over its block, but neighbours are unaffected.
    coarse2 = coarse.copy()
    coarse2[0, 1, 1] = np.nan
    bl2 = upsample_bilinear(coarse2, g)
    assert np.isnan(bl2[0, 12:24, 12:24]).all()
    assert not np.isnan(bl2[0, 36:, 36:]).any()


# --------------------------------------------------------------------------- #
# IMERG parsing
# --------------------------------------------------------------------------- #
def test_parse_imerg_file_subsets_and_masks(tmp_path):
    import h5py

    g = make_grids([-98.0, 30.0, -97.5, 30.5])
    lat = np.round(np.arange(-89.95, 90, 0.1), 6)
    lon = np.round(np.arange(-179.95, 180, 0.1), 6)
    field = np.zeros((len(lon), len(lat)), np.float32)
    lo_idx = np.searchsorted(lon, g.coarse.lon)
    la_idx = np.searchsorted(lat, g.coarse.lat)
    for a, lo in enumerate(lo_idx):
        for b, la in enumerate(la_idx):
            field[lo, la] = 10 * a + b
    field[lo_idx[0], la_idx[0]] = -9999.9
    path = tmp_path / "3B-DAY.MS.MRG.3IMERG.20150102-S000000-E235959.V07B.nc4"
    with h5py.File(path, "w") as h:
        grp = h.create_group("Grid")
        grp.create_dataset("lat", data=lat.astype(np.float32))
        grp.create_dataset("lon", data=lon.astype(np.float32))
        d = grp.create_dataset("precipitation", data=field[None])
        d.attrs["_FillValue"] = np.float32(-9999.9)
        d.attrs["DimensionNames"] = np.bytes_("time,lon,lat")
    arr = dp.parse_imerg_file(path, g.coarse)
    assert arr.shape == g.coarse.shape
    assert np.isnan(arr[0, 0]) and arr[1, 2] == 21.0 and arr[4, 3] == 34.0
    info = dp.imerg_date_from_filename(path)
    assert info[0] == "DAY" and info[1] == np.datetime64("2015-01-02")
    ds = dp.parse_imerg([path], g.coarse)
    assert ds.precip.shape == (1,) + g.coarse.shape and str(ds.time.values[0])[:10] == "2015-01-02"


def test_parse_imerg_half_hourly_integrates(tmp_path):
    import h5py

    g = make_grids([-98.0, 30.0, -97.8, 30.2])
    lat = np.round(np.arange(29.95, 30.35, 0.1), 6)
    lon = np.round(np.arange(-98.05, -97.65, 0.1), 6)
    files = []
    for k in range(48):
        hh, mm = divmod(k * 30, 60)
        p = tmp_path / f"3B-HHR.MS.MRG.3IMERG.20150102-S{hh:02d}{mm:02d}00-E{hh:02d}{mm + 29:02d}59.{k * 30:04d}.V07B.HDF5"
        with h5py.File(p, "w") as h:
            grp = h.create_group("Grid")
            grp.create_dataset("lat", data=lat)
            grp.create_dataset("lon", data=lon)
            d = grp.create_dataset("precipitation", data=np.full((1, len(lon), len(lat)), 2.0, np.float32))  # mm/hr
            d.attrs["DimensionNames"] = np.bytes_("time,lon,lat")
        files.append(p)
    ds = dp.parse_imerg(files, g.coarse)
    assert np.allclose(ds.precip.values, 48.0)  # 2 mm/hr * 24 h


def test_parse_imerg_opendap_subset(tmp_path):
    """OPeNDAP subsets have root-level variables, no Grid group, no DimensionNames."""
    import h5py

    g = make_grids([-98.0, 30.0, -97.7, 30.2])  # 2 x 3 coarse cells
    lat = g.coarse.lat.astype(np.float64)
    lon = g.coarse.lon.astype(np.float32)
    field = np.arange(6, dtype=np.float32).reshape(3, 2)  # (lon, lat)
    path = tmp_path / "3B-DAY.MS.MRG.3IMERG.20190704-S000000-E235959.V07B.nc4"
    with h5py.File(path, "w") as h:
        h.create_dataset("lat", data=lat)
        h.create_dataset("lon", data=lon)
        d = h.create_dataset("precipitation", data=field[None])
        d.attrs["_FillValue"] = np.float32(-9999.9)
    arr = dp.parse_imerg_file(path, g.coarse)
    assert arr.shape == (2, 3) and np.array_equal(arr, field.T)
    i0, i1, j0, j1 = dp.imerg_global_index(g.coarse)
    assert (i0, i1, j0, j1) == (1200, 1201, 820, 822)


# --------------------------------------------------------------------------- #
# AORC (local Zarr stand-in for S3)
# --------------------------------------------------------------------------- #
def test_download_aorc_from_local_zarr(tmp_path):
    g = make_grids([-98.0, 30.0, -97.8, 30.2])
    # AORC-like native grid: 1/120 deg spacing stored with 6-decimal rounding, padded.
    lat = np.round(29.9 + 0.008333 * np.arange(40), 6)
    lon = np.round(-98.1 + 0.008333 * np.arange(40), 6)
    time = pd.date_range("2015-01-01", periods=72, freq="h")
    data = np.ones((72, 40, 40), np.float32) * 0.5  # 0.5 mm per hour -> 12 mm/day
    ds = xr.Dataset({"APCP_surface": (("time", "latitude", "longitude"), data)},
                    coords={"time": time, "latitude": lat, "longitude": lon})
    (tmp_path / "aorc").mkdir()
    ds.to_zarr(tmp_path / "aorc" / "2015.zarr")
    cfg = {
        "paths": {"processed": str(tmp_path / "proc"), "raw": str(tmp_path / "raw")},
        "data": {"aorc": {"bucket": "unused", "variable": "APCP_surface"}},
        "time": {"start": "2015-01-01", "end": "2015-01-03"},
    }
    (tmp_path / "proc").mkdir()
    out = dp.download_aorc(cfg, g, source=str(tmp_path / "aorc"))
    daily = xr.open_dataset(out[0])
    assert daily.precip.shape == (3,) + g.fine.shape
    assert np.allclose(daily.precip.values, 12.0)
    assert np.allclose(daily.lat.values, g.fine.lat)


# --------------------------------------------------------------------------- #
# Raster reprojection
# --------------------------------------------------------------------------- #
def test_reproject_mode_and_average():
    from rasterio.transform import from_origin

    g = make_grids([-98.0, 30.0, -97.9, 30.1])  # 12x12 fine cells
    # 10x finer source raster in EPSG:4326 (north-up), left half class 41, right half 82.
    res = g.fine.res / 10
    lon_min, lat_min, lon_max, lat_max = g.fine.edges
    h = int(round((lat_max - lat_min) / res)); w = int(round((lon_max - lon_min) / res))
    src = np.full((h, w), 41, np.uint8)
    src[:, w // 2:] = 82
    tr = from_origin(lon_min, lat_max, res, res)
    mode = dp.reproject_to_grid(src, tr, "EPSG:4326", g.fine, "mode", src_nodata=0, dst_dtype=np.uint8)
    assert (mode[:, :5] == 41).all() and (mode[:, 7:] == 82).all()
    frac = dp.reproject_to_grid((src == 82).astype(np.uint8), tr, "EPSG:4326", g.fine, "average")
    assert np.allclose(frac[:, :5], 0) and np.allclose(frac[:, 7:], 1) and 0 < frac[0, 6] <= 1


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_slope_aspect_of_tilted_plane():
    g = make_grids([-98.0, 30.0, -97.9, 30.1])
    lat2d, _ = np.meshgrid(g.fine.lat, g.fine.lon, indexing="ij")
    dem = xr.DataArray(((lat2d - 30.0) * 111195.0 * 0.1).astype(np.float32), dims=("lat", "lon"), coords=g.fine.coords())
    d = compute_derived_features(dem)
    assert np.allclose(d.slope.values[2:-2], np.degrees(np.arctan(0.1)), atol=0.05)
    assert np.allclose(d.aspect.values[2:-2], 180.0, atol=1.0)  # rises to the north -> faces south


def test_percentile_rank_uses_training_climatology():
    clim = np.sort(np.array([0, 0, 0, 1, 2, 5, 10], np.float32))[:, None, None]
    vals = np.array([0, 3, 100], np.float32)[:, None, None]
    pct = percentile_rank(vals, clim)
    assert np.allclose(pct[:, 0, 0], [3 / 7, 5 / 7, 1.0])


# --------------------------------------------------------------------------- #
# Validation metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_known_values():
    obs = np.array([0, 2, 5, 12, 35, 0.5, 20, 40], np.float64)
    m = compute_metrics(obs, obs, by_intensity=True)
    assert abs(m["overall"]["rmse"]) < 1e-9 and abs(m["overall"]["nse"] - 1) < 1e-9 and abs(m["overall"]["kge"] - 1) < 1e-9
    assert m["heavy_event_detection"]["pod"] == 1.0
    m2 = compute_metrics(obs + 1.0, obs, by_intensity=True)
    assert abs(m2["overall"]["bias"] - 1.0) < 1e-9 and abs(m2["overall"]["rmse"] - 1.0) < 1e-9
    assert m2["by_intensity"]["dry"]["n"] == 2 and m2["by_intensity"]["heavy"]["n"] == 2
    m3 = compute_metrics(np.zeros_like(obs), obs, by_intensity=True)
    assert m3["heavy_event_detection"]["pod"] == 0.0 and m3["heavy_event_detection"]["misses"] == 2


def test_blockiness_detects_grid_artifacts():
    rng = np.random.default_rng(0)
    coarse = rng.random((5, 5))
    blocky = np.repeat(np.repeat(coarse, 12, 0), 12, 1)
    smooth = np.cumsum(np.cumsum(rng.random((60, 60)), 0), 1)
    assert blockiness(blocky, 12) > 50
    assert 0.8 < blockiness(smooth, 12) < 1.3


def test_reservoir_sampling_size():
    r = Reservoir(100, 2, seed=1)
    for _ in range(10):
        r.update(np.random.default_rng(0).random((57, 2)).astype(np.float32))
    assert r.data().shape == (100, 2) and r.seen == 570


# --------------------------------------------------------------------------- #
# End-to-end on a tiny synthetic domain
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_end_to_end_synthetic(tmp_path):
    from src import synthetic
    from src.feature_engineering import prepare_training_data
    from src.predict import downscale_timeseries
    from src.train_model import train_coarse_stage, train_fine_stage
    from src.utils import ensure_dirs, grids_from_config
    from src.validation import validate

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["project_root"] = str(tmp_path)
    cfg["domain"]["bbox"] = [-98.0, 30.0, -97.6, 30.4]
    cfg["time"] = dict(start="2015-01-01", end="2016-06-30", train_start="2015-01-01", train_end="2015-12-31",
                       val_start="2016-01-01", val_end="2016-06-30")
    cfg["model"]["coarse"]["params"]["n_estimators"] = 60
    cfg["model"]["fine"]["params"]["n_estimators"] = 60
    cfg["model"]["fine"]["max_samples"] = 40000
    cfg["model"]["fine"]["val_samples"] = 8000
    cfg["validation"]["sample_locations"] = [{"name": "X", "lat": 30.2, "lon": -97.8}]
    cfg_path = tmp_path / "config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    cfg = load_config(cfg_path)
    grids = grids_from_config(cfg)
    ensure_dirs(cfg)

    synthetic.generate(cfg, grids, seed=1)
    dp.parse_imerg_dir(cfg, grids)
    dp.build_imerg_raw(cfg, grids)
    outputs = dp.align_grids(cfg, grids)
    imerg = xr.open_dataset(outputs["imerg_10km"])
    aorc_c = xr.open_dataset(outputs["aorc_10km"])
    aorc_f = xr.open_dataset(outputs["aorc_1km"])
    assert imerg.precip.shape == (547,) + grids.coarse.shape
    assert aorc_c.precip.shape == imerg.precip.shape and aorc_f.precip.shape == (547,) + grids.fine.shape
    # Block-mean consistency: coarsened AORC equals the block mean of AORC 1 km.
    manual = aorc_f.precip.isel(time=5).values.reshape(4, 12, 4, 12).mean((1, 3))
    assert np.allclose(manual, aorc_c.precip.isel(time=5).values, atol=1e-4)

    meta = prepare_training_data(cfg, grids)
    assert meta["splits"]["train"]["n_days"] == 365 and meta["splits"]["val"]["n_days"] == 182
    m1 = train_coarse_stage(cfg, grids)
    assert m1["metrics"]["val_rmse"] < m1["metrics"]["val_rmse_imerg_raw"]
    train_fine_stage(cfg, grids)
    out = downscale_timeseries(cfg, grids)
    ds = xr.open_dataset(out)
    assert ds.precipitation.shape == (547,) + grids.fine.shape and "uncertainty" in ds
    assert float(ds.precipitation.min()) >= 0
    res = validate(cfg, grids, out)
    assert res["comparison_table"]["rmse"]["ml"] < res["comparison_table"]["rmse"]["nearest"]
    assert res["comparison_table"]["blockiness_ratio"]["ml"] < 1.5
    assert (Path(cfg["paths"]["results"]) / "validation_metrics.json").exists()
    assert (Path(cfg["paths"]["results"]) / "spatial_plots" / "bias_rmse_maps.png").exists()
