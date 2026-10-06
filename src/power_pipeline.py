"""NASA POWER (MERRA-2) as the coarse input, in place of IMERG.

POWER serves daily `PRECTOTCORR` ("Precipitation Corrected", mm/day) from 1981
onward with no authentication. Two facts drive this module, both established
against the live API rather than the documentation:

* The docs claim a 0.5 deg x 0.5 deg grid. The regional endpoint actually
  returns the **native MERRA-2 grid, 0.5 deg lat x 0.625 deg lon**, with lat
  centres on multiples of 0.5 and lon centres on multiples of 0.625. Neither
  axis lines up with our coarse grid, whose centres sit at
  ``edge + res/2``. So both axes need remapping.
* A request may not span more than 366 days, so the record is fetched a year
  at a time. This is cheap: 150 cells x 365 days returns in about a second.

Remapping is **area-weighted (conservative)**, not interpolation, because
precipitation is an extensive quantity and the block means have to survive the
regrid for the downscaling target to mean anything.

Outputs match the IMERG pipeline's filenames so every downstream module works
unchanged; POWER simply uses its own ``paths.processed`` directory.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils import LOG, Grid, GridPair, to_netcdf

POWER_REGIONAL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
POWER_PARAM = "PRECTOTCORR"
POWER_FILL = -999.0
#: POWER refuses a span longer than this, so the record is fetched per year.
MAX_DAYS = 366


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def _request(bbox, start: str, end: str, param: str, retries: int = 5) -> dict:
    """One regional request, retried: the service 500s intermittently."""
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    q = urllib.parse.urlencode({
        "parameters": param, "community": "AG",
        "latitude-min": lat_min, "latitude-max": lat_max,
        "longitude-min": lon_min, "longitude-max": lon_max,
        "start": start, "end": end, "format": "JSON",
    })
    url = f"{POWER_REGIONAL}?{q}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 422:  # a request error; retrying cannot fix it
                try:
                    msg = json.loads(e.read())["messages"]
                except Exception:  # noqa: BLE001
                    msg = [f"HTTP {e.code}"]
                raise RuntimeError(f"POWER rejected the request: {msg}") from e
            detail = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            detail = f"{type(e).__name__}: {e}"
        if attempt == retries - 1:
            raise RuntimeError(f"POWER request failed after {retries} attempts ({detail})")
        wait = 15 * (attempt + 1)
        LOG.warning("POWER request failed (%s), retrying in %ds", detail, wait)
        time.sleep(wait)
    raise RuntimeError("unreachable")


def _to_dataarray(payload: dict, param: str) -> xr.DataArray:
    """POWER's GeoJSON -> (time, lat, lon) on POWER's own native grid."""
    feats = payload["features"]
    lons = sorted({round(f["geometry"]["coordinates"][0], 4) for f in feats})
    lats = sorted({round(f["geometry"]["coordinates"][1], 4) for f in feats})
    dates = sorted(next(iter(feats))["properties"]["parameter"][param].keys())
    li = {v: i for i, v in enumerate(lons)}
    ai = {v: i for i, v in enumerate(lats)}
    arr = np.full((len(dates), len(lats), len(lons)), np.nan, dtype=np.float32)
    di = {d: i for i, d in enumerate(dates)}
    for f in feats:
        lon, lat = round(f["geometry"]["coordinates"][0], 4), round(f["geometry"]["coordinates"][1], 4)
        series = f["properties"]["parameter"][param]
        j, i = li[lon], ai[lat]
        for d, v in series.items():
            arr[di[d], i, j] = v
    arr = np.where(arr <= POWER_FILL + 1e-6, np.nan, arr)
    return xr.DataArray(
        arr, dims=("time", "lat", "lon"),
        coords={"time": pd.to_datetime(dates, format="%Y%m%d"),
                "lat": np.array(lats, dtype="float64"),
                "lon": np.array(lons, dtype="float64")},
        attrs={"units": "mm/day", "long_name": "NASA POWER PRECTOTCORR (MERRA-2)"},
    )


def fetch_power(cfg: dict, grids: GridPair, start: str | None = None,
                end: str | None = None) -> Path:
    """Fetch the record year by year onto POWER's native grid and cache it."""
    pcfg = cfg["data"].get("power", {})
    param = pcfg.get("parameter", POWER_PARAM)
    pad = float(pcfg.get("pad_deg", 1.0))  # a margin so edge cells are covered
    d = cfg["domain"]
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in d["bbox"])
    bbox = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)

    start = start or cfg["time"]["start"]
    end = end or cfg["time"]["end"]
    cache = Path(cfg["paths"]["raw"]) / "power"
    cache.mkdir(parents=True, exist_ok=True)

    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    parts = []
    for y in years:
        f = cache / f"power_{param}_{y}.nc"
        y0 = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01"))
        y1 = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        if f.exists():
            LOG.info("POWER %d already cached", y)
            parts.append(f)
            continue
        LOG.info("POWER %d: %s .. %s", y, y0.date(), y1.date())
        payload = _request(bbox, y0.strftime("%Y%m%d"), y1.strftime("%Y%m%d"), param)
        da = _to_dataarray(payload, param)
        to_netcdf(xr.Dataset({"precip": da}), f)
        parts.append(f)

    ds = xr.open_mfdataset(parts, combine="by_coords", engine="netcdf4").load()
    out = Path(cfg["paths"]["processed"]) / "power_native.nc"
    to_netcdf(ds, out)
    LOG.info("POWER native grid -> %s  %s", out, dict(ds.sizes))
    return out


# --------------------------------------------------------------------------- #
# Conservative regrid
# --------------------------------------------------------------------------- #
def _edges_from_centres(c: np.ndarray) -> np.ndarray:
    """Cell edges for an evenly spaced coordinate of cell centres."""
    step = float(np.median(np.diff(c)))
    return np.concatenate([c - step / 2.0, [c[-1] + step / 2.0]])


def _overlap_weights(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """(n_dst, n_src) fraction of each destination cell covered by each source cell.

    Rows sum to 1 wherever the destination cell is fully covered, so the result
    is an area-weighted mean: the right operator for an intensive daily rate
    whose block mean must be preserved.
    """
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    w = np.clip(hi - lo, 0.0, None)
    tot = w.sum(axis=1, keepdims=True)
    return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)


def regrid_to_coarse(da: xr.DataArray, grid: Grid) -> xr.DataArray:
    """Area-weighted remap from POWER's native grid onto our coarse grid."""
    wl = _overlap_weights(_edges_from_centres(da["lat"].values),
                          _edges_from_centres(grid.lat))
    wo = _overlap_weights(_edges_from_centres(da["lon"].values),
                          _edges_from_centres(grid.lon))
    x = da.transpose("time", "lat", "lon").values.astype("float64")
    # NaN-aware: weight by the valid mask so partly-missing cells stay unbiased
    ok = np.isfinite(x)
    filled = np.where(ok, x, 0.0)
    num = np.einsum("al,tlm->tam", wl, filled)
    den = np.einsum("al,tlm->tam", wl, ok.astype("float64"))
    num = np.einsum("om,tam->tao", wo, num)
    den = np.einsum("om,tam->tao", wo, den)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
    return xr.DataArray(
        out.astype(np.float32), dims=("time", "lat", "lon"),
        coords={"time": da["time"], "lat": grid.lat, "lon": grid.lon},
        attrs=dict(da.attrs, regrid="area-weighted from MERRA-2 0.5x0.625"),
    )


def build_power_coarse(cfg: dict, grids: GridPair) -> Path:
    """Fetch, regrid and write the coarse product the rest of the pipeline reads.

    Writes ``imerg_raw_10km.nc`` -- the same slot the IMERG pipeline fills --
    because twelve modules read that path. POWER uses a separate
    ``paths.processed`` directory, so nothing is overwritten.
    """
    native = fetch_power(cfg, grids)
    ds = xr.open_dataset(native).load()
    da = regrid_to_coarse(ds["precip"], grids.coarse)
    out = Path(cfg["paths"]["processed"]) / "imerg_raw_10km.nc"
    to_netcdf(xr.Dataset({"precip": da}), out)
    frac = float(np.isfinite(da.values).mean())
    LOG.info("POWER regridded -> %s  %s  (%.1f%% valid)", out, dict(da.sizes), frac * 100)
    return out
