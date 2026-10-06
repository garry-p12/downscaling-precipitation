"""Shared helpers: configuration, logging, grid definitions and regridding.

Grid convention
---------------
* The *coarse* grid is the native IMERG 0.1 deg grid: cell centres sit at
  ``edge + 0.05``.
* The *fine* grid is the native AORC 1/120 deg grid: cell centres sit at exact
  multiples of 1/120 deg, i.e. fine cell ``j`` of a coarse cell has its centre
  at ``coarse_edge + j/120``. Keeping AORC on its native grid means the 1 km
  ground truth is never interpolated. The resulting 12x12 block is offset from
  the IMERG cell by 1/240 deg (~460 m, <5 % of a coarse cell), which is
  negligible for block aggregation.
* All arrays are stored with ascending ``lat`` and ``lon`` dimensions.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

LOG = logging.getLogger("downscaling")

# Sub-daily fill value used by IMERG; anything below this is treated as missing.
IMERG_FILL = -9000.0

# NLCD class grouping used for one-hot / fractional land-cover features.
LC_GROUPS: dict[str, tuple[int, ...]] = {
    "water": (11, 12),
    "developed": (21, 22, 23, 24),
    "barren": (31,),
    "forest": (41, 42, 43),
    "shrub": (51, 52),
    "grass": (71, 72, 73, 74),
    "crop": (81, 82),
    "wetland": (90, 95),
}


# --------------------------------------------------------------------------- #
# Config & logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("botocore", "s3fs", "fsspec", "urllib3", "rasterio", "aiobotocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)  # s3fs "unclosed session" noise at exit


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML config and resolve all paths relative to the config file."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("project_root", "."))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    cfg["project_root"] = str(root)
    cfg["paths"] = {k: str((root / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["_config_path"] = str(path)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("raw", "processed", "auxiliary", "models", "results"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    for sub in ("imerg", "aorc", "nlcd"):
        Path(cfg["paths"]["raw"], sub).mkdir(parents=True, exist_ok=True)
    for sub in ("spatial_plots", "timeseries_comparisons"):
        Path(cfg["paths"]["results"], sub).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.datetime64,)):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """Regular lat/lon grid described by its cell centres (ascending)."""

    lat: np.ndarray
    lon: np.ndarray
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lat), len(self.lon)

    @property
    def size(self) -> int:
        return len(self.lat) * len(self.lon)

    @property
    def edges(self) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max) outer cell edges."""
        h = self.res / 2
        return (
            float(self.lon[0] - h),
            float(self.lat[0] - h),
            float(self.lon[-1] + h),
            float(self.lat[-1] + h),
        )

    def coords(self) -> dict:
        return {
            "lat": ("lat", self.lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", self.lon, {"units": "degrees_east", "standard_name": "longitude"}),
        }

    def transform(self):
        """rasterio Affine for a north-up raster (row 0 = northernmost cell)."""
        from rasterio.transform import from_origin

        lon_min, _, _, lat_max = self.edges
        return from_origin(lon_min, lat_max, self.res, self.res)

    def cell_area_km2(self) -> np.ndarray:
        """Approximate cell area (km^2) as a function of latitude, shape (nlat, 1)."""
        km_per_deg = 111.195
        dy = self.res * km_per_deg
        dx = self.res * km_per_deg * np.cos(np.deg2rad(self.lat))
        return (dx * dy)[:, None]

    def to_dict(self) -> dict:
        return {
            "res_deg": self.res,
            "nlat": len(self.lat),
            "nlon": len(self.lon),
            "lat_min": float(self.lat[0]),
            "lat_max": float(self.lat[-1]),
            "lon_min": float(self.lon[0]),
            "lon_max": float(self.lon[-1]),
            "edges": self.edges,
        }


@dataclass(frozen=True)
class GridPair:
    coarse: Grid
    fine: Grid
    factor: int

    def to_dict(self) -> dict:
        return {"coarse": self.coarse.to_dict(), "fine": self.fine.to_dict(), "factor": self.factor}


def make_grids(bbox, coarse_res: float = 0.1, factor: int = 12) -> GridPair:
    """Build the aligned coarse (IMERG) and fine (AORC) grids for ``bbox``."""
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    n_lat_c = int(round((lat_max - lat_min) / coarse_res))
    n_lon_c = int(round((lon_max - lon_min) / coarse_res))
    for name, span, n in (("lat", lat_max - lat_min, n_lat_c), ("lon", lon_max - lon_min, n_lon_c)):
        if n < 1 or abs(n * coarse_res - span) > 1e-6:
            raise ValueError(f"bbox {name} span {span} is not a multiple of coarse_res {coarse_res}")
    for v in (lon_min, lat_min):
        if abs(round(v / coarse_res) * coarse_res - v) > 1e-6:
            raise ValueError(f"bbox edge {v} must be a multiple of coarse_res {coarse_res}")

    coarse = Grid(
        lat=np.round(lat_min + coarse_res * (np.arange(n_lat_c) + 0.5), 6),
        lon=np.round(lon_min + coarse_res * (np.arange(n_lon_c) + 0.5), 6),
        res=coarse_res,
    )
    fine_res = coarse_res / factor
    fine = Grid(
        lat=np.round(lat_min + fine_res * np.arange(n_lat_c * factor), 6),
        lon=np.round(lon_min + fine_res * np.arange(n_lon_c * factor), 6),
        res=fine_res,
    )
    return GridPair(coarse=coarse, fine=fine, factor=factor)


def grids_from_config(cfg: dict) -> GridPair:
    d = cfg["domain"]
    return make_grids(d["bbox"], d.get("coarse_res_deg", 0.1), d.get("fine_factor", 12))


def subset_box(da: xr.DataArray, bbox, eps: float = 1e-6) -> xr.DataArray:
    """Restrict a (..., lat, lon) array to a lon/lat bounding box.

    Cell *centres* lie in ``[min, max)`` on both grids, while ``.sel`` with a
    slice is inclusive at both ends, so the upper edge is nudged inward by
    ``eps``. Without this the subset is one row and one column too large and no
    longer matches the grid built from the same bbox.
    """
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    return da.sel(lat=slice(lat_min - eps, lat_max - eps),
                  lon=slice(lon_min - eps, lon_max - eps))


def evaluation_grids(cfg: dict, grids: "GridPair") -> tuple["GridPair", list | None]:
    """Grids for the scoring sub-box, plus the box itself.

    When training spans a wider domain than we want to score on (for example
    reaching west into terrain while the holdout stays over Austin),
    ``evaluation.bbox`` restricts every metric to that sub-box so results stay
    comparable with runs made on the smaller domain. Absent, scoring covers the
    whole domain and this is a no-op.
    """
    bb = (cfg.get("evaluation") or {}).get("bbox")
    if not bb:
        return grids, None
    d = cfg["domain"]
    return make_grids(bb, d.get("coarse_res_deg", 0.1), d.get("fine_factor", 12)), list(bb)


def assert_on_grid(da: xr.DataArray, grid: Grid, name: str = "array") -> None:
    """Raise if ``da``'s lat/lon do not match ``grid`` exactly (to 1e-5 deg)."""
    if da.sizes["lat"] != len(grid.lat) or da.sizes["lon"] != len(grid.lon):
        raise ValueError(f"{name}: shape {da.sizes['lat']}x{da.sizes['lon']} != grid {grid.shape}")
    if not (np.allclose(da.lat.values, grid.lat, atol=1e-5) and np.allclose(da.lon.values, grid.lon, atol=1e-5)):
        raise ValueError(f"{name}: coordinates do not match the target grid")


# --------------------------------------------------------------------------- #
# Regridding
# --------------------------------------------------------------------------- #
def coarsen_mean(da: xr.DataArray, grids: GridPair, min_frac: float = 0.5) -> xr.DataArray:
    """Block-average a fine-grid array to the coarse grid (NaN-aware).

    Blocks with fewer than ``min_frac`` valid fine cells are set to NaN.
    """
    assert_on_grid(da, grids.fine, "coarsen_mean input")
    f = grids.factor
    c = da.coarsen(lat=f, lon=f, boundary="exact")
    mean = c.mean(skipna=True)
    frac = da.notnull().coarsen(lat=f, lon=f, boundary="exact").mean()
    mean = mean.where(frac >= min_frac)
    return mean.assign_coords(lat=grids.coarse.lat, lon=grids.coarse.lon)


def coarsen_reduce(da: xr.DataArray, grids: GridPair, how: str = "mean") -> xr.DataArray:
    """Block-reduce (mean/std/max/min) a fine-grid array to the coarse grid."""
    assert_on_grid(da, grids.fine, "coarsen_reduce input")
    f = grids.factor
    c = da.coarsen(lat=f, lon=f, boundary="exact")
    out = getattr(c, how)(skipna=True) if how in ("mean", "std", "max", "min") else c.mean()
    return out.assign_coords(lat=grids.coarse.lat, lon=grids.coarse.lon)


def upsample_nearest(da: xr.DataArray, grids: GridPair) -> xr.DataArray:
    """Repeat each coarse cell over its 12x12 fine block (no interpolation)."""
    assert_on_grid(da, grids.coarse, "upsample_nearest input")
    f = grids.factor
    lat_ax = da.dims.index("lat")
    lon_ax = da.dims.index("lon")
    arr = np.repeat(np.repeat(da.values, f, axis=lat_ax), f, axis=lon_ax)
    coords = {k: v for k, v in da.coords.items() if k not in ("lat", "lon")}
    coords.update(grids.fine.coords())
    return xr.DataArray(arr, dims=da.dims, coords=coords, attrs=da.attrs)


def upsample_bilinear(da: xr.DataArray, grids: GridPair, clip_zero: bool = True) -> xr.DataArray:
    """Bilinearly interpolate a coarse-grid array onto the fine grid.

    Fine cells outside the outermost coarse centres are linearly extrapolated
    (at most half a coarse cell) so the output covers the full domain.
    NaNs in the input are filled by nearest-neighbour first so that a single
    missing coarse cell does not blank out a 2x2 neighbourhood.
    """
    assert_on_grid(da, grids.coarse, "upsample_bilinear input")
    filled = fill_nan_nearest(da)
    out = filled.interp(
        lat=grids.fine.lat,
        lon=grids.fine.lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    # Restore NaN where the *nearest* coarse cell was missing.
    mask = upsample_nearest(da.isnull().astype(np.int8), grids).astype(bool)
    out = out.where(~mask)
    if clip_zero:
        out = out.clip(min=0.0)
    out = out.assign_coords(lat=grids.fine.lat, lon=grids.fine.lon)
    out.attrs = dict(da.attrs)
    return out


def fill_nan_nearest(da: xr.DataArray) -> xr.DataArray:
    """Fill NaNs in the lat/lon plane with the nearest valid value (per time step)."""
    from scipy import ndimage

    vals = da.values
    if not np.isnan(vals).any():
        return da
    lat_ax = da.dims.index("lat")
    lon_ax = da.dims.index("lon")
    moved = np.moveaxis(vals, (lat_ax, lon_ax), (-2, -1))
    flat = moved.reshape(-1, moved.shape[-2], moved.shape[-1]).copy()
    for i in range(flat.shape[0]):
        plane = flat[i]
        bad = np.isnan(plane)
        if bad.any() and not bad.all():
            idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
            flat[i] = plane[tuple(idx)]
    filled = np.moveaxis(flat.reshape(moved.shape), (-2, -1), (lat_ax, lon_ax))
    return da.copy(data=filled)


def block_index(grids: GridPair) -> tuple[np.ndarray, np.ndarray]:
    """For every fine cell, the (row, col) index of its parent coarse cell."""
    f = grids.factor
    ci = np.arange(len(grids.fine.lat)) // f
    cj = np.arange(len(grids.fine.lon)) // f
    return ci, cj


# --------------------------------------------------------------------------- #
# Array helpers
# --------------------------------------------------------------------------- #
def flatten_to_array(ds: xr.Dataset, var_names: list[str], dtype=np.float32) -> np.ndarray:
    """Stack variables of a Dataset into a 2-D (n_samples, n_features) array.

    Static (lat, lon) variables are broadcast along ``time`` if present.
    """
    cols = []
    dims = None
    for v in var_names:
        da = ds[v]
        if "time" in ds.dims and "time" not in da.dims:
            da = da.broadcast_like(ds["time"]).transpose("time", "lat", "lon")
        elif "time" in da.dims:
            da = da.transpose("time", "lat", "lon")
        else:
            da = da.transpose("lat", "lon")
        if dims is None:
            dims = da.shape
        cols.append(np.asarray(da.values, dtype=dtype).reshape(-1))
    return np.stack(cols, axis=1)


def open_dataset(path: str | Path, chunks: dict | None = None) -> xr.Dataset:
    """Open a NetCDF file produced by this pipeline with sane defaults."""
    return xr.open_dataset(path, chunks=chunks, engine="netcdf4")


def to_netcdf(ds: xr.Dataset, path: str | Path, complevel: int = 4, time_chunk: int = 32) -> None:
    """Write a dataset as compressed float32 NetCDF4 with time-chunked storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for v in ds.data_vars:
        da = ds[v]
        enc: dict = {"zlib": True, "complevel": complevel}
        if np.issubdtype(da.dtype, np.floating):
            enc["dtype"] = "float32"
        if "time" in da.dims and da.sizes["time"] > 1:
            chunks = tuple(min(time_chunk, da.sizes[d]) if d == "time" else da.sizes[d] for d in da.dims)
            enc["chunksizes"] = chunks
        encoding[v] = enc
    if "time" in ds.coords:
        encoding["time"] = {"units": "days since 1970-01-01", "dtype": "int32"}
    tmp = path.with_suffix(path.suffix + ".tmp")
    ds.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    tmp.replace(path)


def open_downscaled(results_dir: str | Path, chunks: dict | None = None) -> xr.DataArray:
    """Open the downscaled product, whether it is one merged file or per-year files."""
    d = Path(results_dir)
    merged = sorted(d.glob("imerg_downscaled_1km_*.nc"))
    if merged:
        return xr.open_dataset(merged[-1], chunks=chunks, engine="netcdf4")["precipitation"]
    years = sorted(d.glob("downscaled_1km_[0-9][0-9][0-9][0-9].nc"))
    if not years:
        raise FileNotFoundError(f"No downscaled product in {d}; run `main.py predict` first")
    return xr.open_mfdataset(years, combine="by_coords", engine="netcdf4",
                             chunks=chunks or {"time": 32})["precipitation"]


def date_range_daily(start, end) -> np.ndarray:
    import pandas as pd

    return pd.date_range(start, end, freq="D").values.astype("datetime64[ns]")
