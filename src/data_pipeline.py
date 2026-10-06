"""Phase 1: download, parse and grid-align IMERG, AORC and NLCD/DEM.

Data sources
------------
IMERG  NASA GES DISC via ``earthaccess`` (Earthdata login required).
       Daily Final Run ``GPM_3IMERGDF`` (V07: variable ``precipitation``,
       V06: ``precipitationCal``). Half-hourly ``GPM_3IMERGHH`` is also
       parsed (rates are integrated to daily totals).
AORC   NOAA AORC v1.1 hourly 1 km Zarr on AWS (anonymous):
       ``s3://noaa-nws-aorc-v1-1-1km/{year}.zarr``, variable ``APCP_surface``.
NLCD   MRLC WCS (EPSG:5070, 30 m) for land cover and percent impervious.
DEM    Copernicus GLO-90 COGs on AWS (anonymous), 3 arc-second.

Outputs (``data/processed``)
----------------------------
imerg_raw_10km.nc        IMERG subset on the coarse grid (native 0.1 deg)
aorc_1km_daily_{Y}.nc    AORC daily totals on the fine grid, one file per year
imerg_aligned_10km.nc    IMERG on the full daily time axis
aorc_aligned_1km.nc      AORC 1 km on the full daily time axis
aorc_aligned_10km.nc     AORC block-averaged to the coarse grid
nlcd_aligned_1km.nc      dem, slope, aspect, lulc, imperv, lc_frac_* at 1 km
"""
from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils import (
    LC_GROUPS,
    LOG,
    Grid,
    GridPair,
    assert_on_grid,
    coarsen_mean,
    date_range_daily,
    open_dataset,
    to_netcdf,
)

IMERG_FILE_RE = re.compile(r"3B-(DAY|HHR|MO)[.\w-]*?\.(\d{8})(?:-S(\d{6}))?")


# =========================================================================== #
# IMERG
# =========================================================================== #
def imerg_date_from_filename(path: str | Path) -> tuple[str, np.datetime64, float] | None:
    """Return (product, date, hours_per_file) parsed from an IMERG filename."""
    m = IMERG_FILE_RE.search(Path(path).name)
    if not m:
        return None
    product, ymd = m.group(1), m.group(2)
    date = np.datetime64(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
    hours = {"DAY": 24.0, "HHR": 0.5, "MO": None}[product]
    return product, date, hours


def _imerg_variable(grp, variable: str = "auto") -> str:
    if variable != "auto":
        return variable
    for cand in ("precipitation", "precipitationCal", "precipitationUncal"):
        if cand in grp:
            return cand
    raise KeyError(f"No precipitation variable found in IMERG group; have {list(grp.keys())}")


#: Extra fields carried by the *daily* IMERG granules. They describe the
#: sub-daily structure and the retrieval quality that a daily total discards:
#: how many half-hours were raining, how much of the estimate came from a
#: microwave overpass rather than IR morphing, and IMERG's own error estimate.
#: ``randomError`` is deliberately excluded: the V07 daily granules label it
#: mm/day but carry a median near 5000 mm/day, which is not physical, so it is
#: not interpretable as an uncertainty and is left out rather than fed to a model.
IMERG_EXTRA_VARS = ("precipitation_cnt", "precipitation_cnt_cond", "MWprecipitation",
                    "probabilityLiquidPrecipitation")


def parse_imerg_file(path: str | Path, grid: Grid, variable: str = "auto",
                     extra_vars: tuple[str, ...] = ()) -> np.ndarray | dict[str, np.ndarray]:
    """Read one IMERG granule and return precipitation on ``grid`` (lat, lon).

    Daily files hold mm/day; half-hourly files hold mm/hr (the caller scales).
    Missing values are returned as NaN. With ``extra_vars`` a dict is returned
    holding the main field under ``"precip"`` plus whichever extras exist.
    """
    import h5py

    with h5py.File(path, "r") as h:
        grp = h["Grid"] if "Grid" in h else h
        var = _imerg_variable(grp, variable)
        lat = grp["lat"][:].astype(np.float64)
        lon = grp["lon"][:].astype(np.float64)
        dset = grp[var]
        dimnames = dset.attrs.get("DimensionNames", b"time,lon,lat")
        if isinstance(dimnames, bytes):
            dimnames = dimnames.decode()
        dims = [d.strip() for d in str(dimnames).split(",")]
        # Index ranges covering the target grid (nearest-neighbour by centre).
        ilat = np.array([int(np.argmin(np.abs(lat - v))) for v in grid.lat])
        ilon = np.array([int(np.argmin(np.abs(lon - v))) for v in grid.lon])
        tol = grid.res * 0.5
        if np.abs(lat[ilat] - grid.lat).max() > tol or np.abs(lon[ilon] - grid.lon).max() > tol:
            raise ValueError(f"{path}: IMERG grid does not cover the target domain")
        sl = {"time": 0, "lat": slice(ilat.min(), ilat.max() + 1), "lon": slice(ilon.min(), ilon.max() + 1)}
        slab = dset[tuple(sl[d] for d in dims)]
        # Reorder to (lat, lon).
        rem = [d for d in dims if d != "time"]
        arr = np.asarray(slab, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        n_la = ilat.max() - ilat.min() + 1
        n_lo = ilon.max() - ilon.min() + 1
        if rem == ["lon", "lat"] or (arr.shape == (n_lo, n_la) and n_lo != n_la):
            arr = arr.T
        if arr.shape != (n_la, n_lo):
            raise ValueError(f"{path}: unexpected slab shape {arr.shape}, expected {(n_la, n_lo)}")
        arr = arr[ilat - ilat.min()][:, ilon - ilon.min()]
        fill = dset.attrs.get("_FillValue", None)
        if fill is not None:
            arr = np.where(np.isclose(arr, np.float32(fill)), np.nan, arr)
        arr = np.where(arr < -1000, np.nan, arr)
        if not extra_vars:
            return arr
        out = {"precip": arr}
        for name in extra_vars:
            if name not in grp:
                continue
            d2 = grp[name]
            slab2 = d2[tuple(sl[d] for d in dims)]
            a2 = np.asarray(slab2, dtype=np.float32)
            if a2.ndim == 3:
                a2 = a2[0]
            if rem == ["lon", "lat"] or (a2.shape == (n_lo, n_la) and n_lo != n_la):
                a2 = a2.T
            a2 = a2[ilat - ilat.min()][:, ilon - ilon.min()]
            f2 = d2.attrs.get("_FillValue", None)
            if f2 is not None:
                a2 = np.where(np.isclose(a2, np.float32(f2)), np.nan, a2)
            out[name] = np.where(a2 < -1000, np.nan, a2)
    return out


def parse_imerg(files, grid: Grid, variable: str = "auto", min_valid_frac: float = 0.9,
                extra_vars: tuple[str, ...] = ()) -> xr.Dataset:
    """Parse a collection of IMERG granules into a daily dataset on ``grid``.

    Half-hourly granules are integrated (rate x 0.5 h) and summed per day;
    days with fewer than ``min_valid_frac`` of their slots valid become NaN.
    """
    by_day: dict[np.datetime64, list] = defaultdict(list)
    for f in files:
        info = imerg_date_from_filename(f)
        if info is None:
            LOG.warning("Skipping unrecognised IMERG file %s", f)
            continue
        product, date, hours = info
        if hours is None:
            LOG.warning("Skipping monthly IMERG file %s", f)
            continue
        by_day[date].append((f, hours))

    dates = sorted(by_day)
    out = np.full((len(dates),) + grid.shape, np.nan, dtype=np.float32)
    # Extras only exist on the daily product; half-hourly granules skip them.
    daily_only = extra_vars if all(h == 24.0 for v in by_day.values() for _, h in v) else ()
    extras = {v: np.full((len(dates),) + grid.shape, np.nan, np.float32) for v in daily_only}
    for k, date in enumerate(dates):
        entries = by_day[date]
        expected = 1 if entries[0][1] == 24.0 else 48
        total = np.zeros(grid.shape, np.float32)
        count = np.zeros(grid.shape, np.int16)
        for f, hours in entries:
            arr = parse_imerg_file(f, grid, variable, daily_only)
            if isinstance(arr, dict):
                for v in daily_only:
                    if v in arr:
                        extras[v][k] = arr[v]
                arr = arr["precip"]
            if hours == 24.0:
                scale = 1.0  # daily file already in mm/day
            else:
                scale = hours  # mm/hr * hours
            valid = ~np.isnan(arr)
            total[valid] += arr[valid] * scale
            count += valid
        good = count >= max(1, int(np.ceil(min_valid_frac * expected)))
        day = np.where(good, total, np.nan)
        # For partially-missing half-hourly days scale up to a full day.
        if expected > 1:
            day = np.where(good, total * (expected / np.maximum(count, 1)), np.nan)
        out[k] = day
    data = {"precip": (("time", "lat", "lon"), out,
                       {"units": "mm/day", "long_name": "IMERG precipitation"})}
    for v, a in extras.items():
        data[v] = (("time", "lat", "lon"), a)
    return xr.Dataset(data, coords={"time": np.array(dates, dtype="datetime64[ns]"), **grid.coords()})


def find_imerg_files(raw_dir: str | Path) -> list[Path]:
    exts = (".nc4", ".nc", ".h5", ".he5", ".hdf5", ".HDF5")
    files = [p for p in Path(raw_dir).rglob("*") if p.suffix in exts and IMERG_FILE_RE.search(p.name)]
    return sorted(files)


def _imerg_cache_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed"]) / "imerg_cache"


def parse_imerg_dir(cfg: dict, grids: GridPair, delete_raw: bool | None = None) -> list[Path]:
    """Parse every raw IMERG granule under ``data/raw/imerg`` into monthly caches."""
    raw_dir = Path(cfg["paths"]["raw"]) / "imerg"
    cache_dir = _imerg_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    variable = cfg["data"]["imerg"].get("variable", "auto")
    if delete_raw is None:
        delete_raw = not cfg["data"]["imerg"].get("keep_raw", True)

    files = find_imerg_files(raw_dir)
    by_month: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        info = imerg_date_from_filename(f)
        if info:
            by_month[str(info[1])[:7]].append(f)
    written = []
    for month in sorted(by_month):
        out = cache_dir / f"imerg_10km_{month.replace('-', '')}.nc"
        if out.exists():
            LOG.info("IMERG cache exists for %s, skipping", month)
            continue
        ds = parse_imerg(by_month[month], grids.coarse, variable)
        to_netcdf(ds, out)
        written.append(out)
        LOG.info("Parsed %d IMERG granules for %s -> %s", len(by_month[month]), month, out.name)
        if delete_raw:
            for f in by_month[month]:
                f.unlink(missing_ok=True)
    return written


def build_imerg_raw(cfg: dict, grids: GridPair) -> Path:
    """Concatenate the monthly IMERG caches into ``imerg_raw_10km.nc``."""
    cache_dir = _imerg_cache_dir(cfg)
    files = sorted(cache_dir.glob("imerg_10km_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No IMERG caches in {cache_dir}. Run `main.py download --imerg` (needs Earthdata login), "
            "drop granules into data/raw/imerg and run `main.py preprocess`, or use `--synthetic`."
        )
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4").sortby("time")
    ds = ds.drop_duplicates("time")
    out = Path(cfg["paths"]["processed"]) / "imerg_raw_10km.nc"
    to_netcdf(ds.load(), out)
    LOG.info("IMERG raw subset: %d days -> %s", ds.sizes["time"], out)
    return out


def _earthdata_credentials() -> tuple[str, str]:
    """Earthdata username/password from env vars, then ~/.netrc, then a prompt."""
    import getpass
    import netrc
    import sys

    user, pw = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
    if user and pw:
        return user, pw
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth and auth[0] and auth[2]:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    if sys.stdin.isatty():
        user = input("Earthdata username: ")
        pw = getpass.getpass("Earthdata password: ")
        return user, pw
    raise RuntimeError(
        "No Earthdata credentials. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD or add\n"
        "  machine urs.earthdata.nasa.gov login <user> password <pass>\n"
        "to ~/.netrc (chmod 600), and approve 'NASA GESDISC DATA ARCHIVE' at "
        "https://urs.earthdata.nasa.gov/approved_applications"
    )


def earthdata_session():
    """requests.Session that only sends Basic auth to urs.earthdata.nasa.gov.

    This is the pattern documented by GES DISC: the data server redirects to
    URS, which authenticates and redirects back with a session cookie.
    """
    from urllib.parse import urlparse

    import requests

    class SessionWithHeaderRedirection(requests.Session):
        AUTH_HOST = "urs.earthdata.nasa.gov"

        def rebuild_auth(self, prepared_request, response):
            headers = prepared_request.headers
            url = prepared_request.url
            if "Authorization" in headers:
                orig = urlparse(response.request.url).hostname
                redir = urlparse(url).hostname
                if orig != redir and redir != self.AUTH_HOST and orig != self.AUTH_HOST:
                    del headers["Authorization"]

    user, pw = _earthdata_credentials()
    sess = SessionWithHeaderRedirection()
    sess.auth = (user, pw)
    return sess


IMERG_OPENDAP_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/{short_name}.{version}/{yyyy}/{mm}/{fname}"


def imerg_global_index(grid: Grid, res: float = 0.1) -> tuple[int, int, int, int]:
    """(i0, i1, j0, j1) inclusive lat/lon indices of ``grid`` on the global IMERG grid."""
    i = np.round((grid.lat + 90 - res / 2) / res).astype(int)
    j = np.round((grid.lon + 180 - res / 2) / res).astype(int)
    return int(i.min()), int(i.max()), int(j.min()), int(j.max())


def _opendap_variable(session, url_base: str, variable: str) -> str:
    """Pick the precipitation variable name from the granule's DDS (no auth needed)."""
    if variable != "auto":
        return variable
    import requests

    r = requests.get(url_base + ".dds", timeout=120)
    r.raise_for_status()
    for cand in ("precipitation", "precipitationCal"):
        if (" " + cand + "[") in r.text or ("} " + cand + ";") in r.text:
            return cand
    raise KeyError(f"No precipitation variable in DDS of {url_base}")


def _fetch_opendap_subset(session, url: str, out: Path, retries: int = 5) -> Path:
    import time as _time

    for attempt in range(retries):
        try:
            r = session.get(url, timeout=300)
            if r.status_code == 401:
                raise PermissionError("Earthdata rejected the credentials (401): check ~/.netrc or EARTHDATA_* vars")
            if "urs.earthdata.nasa.gov" in r.url:
                raise PermissionError(
                    "Stuck at Earthdata Login: approve the 'NASA GESDISC DATA ARCHIVE' application at "
                    "https://urs.earthdata.nasa.gov/approved_applications and retry"
                )
            r.raise_for_status()
            if not (r.content.startswith(b"\x89HDF") or r.content.startswith(b"CDF")):
                raise RuntimeError(f"Unexpected OPeNDAP response ({r.headers.get('content-type')}): {r.text[:200]}")
            out.write_bytes(r.content)
            return out
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            LOG.warning("OPeNDAP fetch failed (%s), retrying in %ds", e, 15 * (attempt + 1))
            _time.sleep(15 * (attempt + 1))
    return out


def _search_granules(icfg: dict, m_start: str, m_end: str, retries: int = 5):
    """CMR granule search with backoff.

    Compute nodes lose DNS for cmr.earthdata.nasa.gov often enough that an
    unretried search throws away every hour of fetching that preceded it.
    """
    import time as _time

    import earthaccess

    for attempt in range(retries):
        try:
            return earthaccess.search_data(short_name=icfg["short_name"], version=str(icfg["version"]),
                                           temporal=(m_start, m_end))
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = 15 * (attempt + 1)
            LOG.warning("CMR search failed (%s), retrying in %ds", e, wait)
            _time.sleep(wait)
    return []


def _download_imerg_opendap(cfg: dict, grids: GridPair, start: str, end: str) -> None:
    """Server-side subset of each daily granule via GES DISC OPeNDAP (a few KB/day)."""
    from concurrent.futures import ThreadPoolExecutor

    import earthaccess

    icfg = cfg["data"]["imerg"]
    raw_dir = Path(cfg["paths"]["raw"]) / "imerg"
    cache_dir = _imerg_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = earthdata_session()
    i0, i1, j0, j1 = imerg_global_index(grids.coarse)
    workers = int(icfg.get("workers", 4))
    extra_vars = tuple(icfg.get("extra_vars", IMERG_EXTRA_VARS))
    variable = None

    for per in pd.period_range(start, end, freq="M"):
        tag = per.strftime("%Y%m")
        out = cache_dir / f"imerg_10km_{tag}.nc"
        if out.exists():
            LOG.info("IMERG %s already cached", tag)
            continue
        m_start = max(pd.Timestamp(start), per.start_time).strftime("%Y-%m-%d")
        m_end = min(pd.Timestamp(end), per.end_time).strftime("%Y-%m-%d")
        results = _search_granules(icfg, m_start, m_end)
        names = sorted({g.data_links()[0].split("/")[-1] for g in results})
        if not names:
            LOG.warning("No IMERG granules found for %s", tag)
            continue
        local = raw_dir / str(per.year)
        local.mkdir(parents=True, exist_ok=True)
        jobs = []
        for fname in names:
            base = IMERG_OPENDAP_BASE.format(short_name=icfg["short_name"], version=str(icfg["version"]),
                                             yyyy=per.strftime("%Y"), mm=per.strftime("%m"), fname=fname)
            if variable is None:
                variable = _opendap_variable(session, base, icfg.get("variable", "auto"))
                LOG.info("IMERG OPeNDAP variable: %s ; lat[%d:%d] lon[%d:%d]", variable, i0, i1, j0, j1)
            sub = f"[0:0][{j0}:{j1}][{i0}:{i1}]"
            want = [variable] + [v for v in extra_vars]
            url = (f"{base}.nc4?" + ",".join(f"{v}{sub}" for v in want)
                   + f",lat[{i0}:{i1}],lon[{j0}:{j1}]")
            target = local / fname
            if target.exists() and target.stat().st_size > 0:
                jobs.append(target)
            else:
                jobs.append((url, target))
        LOG.info("IMERG %s: fetching %d subsets via OPeNDAP", tag, sum(isinstance(j, tuple) for j in jobs))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            files = list(ex.map(lambda j: j if isinstance(j, Path) else _fetch_opendap_subset(session, *j), jobs))
        ds = parse_imerg(files, grids.coarse, variable, extra_vars=extra_vars)
        to_netcdf(ds, out)
        LOG.info("IMERG %s: %d days cached (%d fields, domain mean %.2f mm/day)", tag,
                 ds.sizes["time"], len(ds.data_vars), float(ds["precip"].mean()))
        if not icfg.get("keep_raw", False):
            for f in files:
                Path(f).unlink(missing_ok=True)


def _download_imerg_granules(cfg: dict, grids: GridPair, start: str, end: str) -> None:
    """Download full daily granules with earthaccess (~28 MB each) and parse them."""
    import earthaccess

    icfg = cfg["data"]["imerg"]
    raw_dir = Path(cfg["paths"]["raw"]) / "imerg"
    cache_dir = _imerg_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    auth = earthaccess.login(strategy="all")
    if not getattr(auth, "authenticated", False):
        raise RuntimeError(
            "Earthdata login failed. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD or add\n"
            "  machine urs.earthdata.nasa.gov login <user> password <pass>\n"
            "to ~/.netrc (chmod 600). Also approve 'NASA GESDISC DATA ARCHIVE' at "
            "https://urs.earthdata.nasa.gov/approved_applications."
        )

    for per in pd.period_range(start, end, freq="M"):
        tag = per.strftime("%Y%m")
        out = cache_dir / f"imerg_10km_{tag}.nc"
        if out.exists():
            LOG.info("IMERG %s already cached", tag)
            continue
        m_start = max(pd.Timestamp(start), per.start_time).strftime("%Y-%m-%d")
        m_end = min(pd.Timestamp(end), per.end_time).strftime("%Y-%m-%d")
        LOG.info("Searching IMERG %s %s (%s .. %s)", icfg["short_name"], icfg["version"], m_start, m_end)
        results = _search_granules(icfg, m_start, m_end)
        if not results:
            LOG.warning("No IMERG granules found for %s", tag)
            continue
        local = raw_dir / str(per.year)
        local.mkdir(parents=True, exist_ok=True)
        files = earthaccess.download(results, str(local))
        files = [Path(f) for f in files if f]
        ds = parse_imerg(files, grids.coarse, icfg.get("variable", "auto"))
        to_netcdf(ds, out)
        LOG.info("IMERG %s: %d granules -> %d days", tag, len(files), ds.sizes["time"])
        if not icfg.get("keep_raw", False):
            for f in files:
                Path(f).unlink(missing_ok=True)


def download_imerg(cfg: dict, grids: GridPair, start: str | None = None, end: str | None = None) -> None:
    """Fetch IMERG for the domain month by month and build ``imerg_raw_10km.nc``.

    ``data.imerg.mode``: ``opendap`` (default; GES DISC server-side subsetting,
    a few KB per day) or ``granule`` (full ~28 MB granules via earthaccess).
    Both need an Earthdata login (see README) and are resumable; raw files are
    deleted after parsing unless ``data.imerg.keep_raw`` is true.
    """
    start = start or cfg["time"]["start"]
    end = end or cfg["time"]["end"]
    mode = cfg["data"]["imerg"].get("mode", "opendap")
    if mode == "opendap":
        _download_imerg_opendap(cfg, grids, start, end)
    elif mode == "granule":
        _download_imerg_granules(cfg, grids, start, end)
    else:
        raise ValueError(f"Unknown IMERG download mode {mode}")
    build_imerg_raw(cfg, grids)


# =========================================================================== #
# AORC
# =========================================================================== #
def open_aorc_year(source: str, year: int) -> xr.Dataset:
    """Open one AORC yearly Zarr store from S3 (bucket name) or a local dir."""
    if os.path.isdir(source):
        return xr.open_zarr(os.path.join(source, f"{year}.zarr"), consolidated=None)
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=f"{source}/{year}.zarr", s3=fs, check=False)
    return xr.open_zarr(store, consolidated=True)


def aggregate_daily(da: xr.DataArray, how: str = "sum", min_count: int | None = None) -> xr.DataArray:
    """Aggregate a sub-daily (time, ...) array to UTC-day totals or means."""
    r = da.resample(time="1D")
    if how == "sum":
        return r.sum(skipna=True, min_count=min_count)
    if how == "mean":
        return r.mean(skipna=True)
    raise ValueError(how)


def aorc_daily_path(cfg: dict, year: int) -> Path:
    return Path(cfg["paths"]["processed"]) / f"aorc_1km_daily_{year}.nc"


def download_aorc(cfg: dict, grids: GridPair, start: str | None = None, end: str | None = None,
                  source: str | None = None) -> list[Path]:
    """Stream AORC hourly precipitation for the domain and write daily totals.

    One NetCDF per year on the fine grid. Existing years are skipped so the
    download can be resumed. Note: the hourly value stamped ``t`` is the
    accumulation of the preceding hour; daily sums use the timestamp's UTC
    date, consistent with IMERG daily files.
    """
    acfg = cfg["data"]["aorc"]
    source = source or acfg["bucket"]
    var = acfg.get("variable", "APCP_surface")
    start = pd.Timestamp(start or cfg["time"]["start"])
    end = pd.Timestamp(end or cfg["time"]["end"])
    fine = grids.fine
    lon_min, lat_min, lon_max, lat_max = fine.edges
    pad = fine.res
    written = []
    for year in range(start.year, end.year + 1):
        out = aorc_daily_path(cfg, year)
        if out.exists():
            LOG.info("AORC %d exists, skipping", year)
            written.append(out)
            continue
        LOG.info("Opening AORC %d from %s", year, source)
        ds = open_aorc_year(source, year)
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        da = ds[var].sel(
            {lat_name: slice(lat_min - pad, lat_max + pad), lon_name: slice(lon_min - pad, lon_max + pad)}
        )
        y0 = max(start, pd.Timestamp(f"{year}-01-01"))
        y1 = min(end, pd.Timestamp(f"{year}-12-31"))
        pieces = []
        for per in pd.period_range(y0, y1, freq="M"):
            m0 = max(y0, per.start_time)
            m1 = min(y1 + pd.Timedelta(hours=23, minutes=59), per.end_time)
            sub = da.sel(time=slice(m0, m1)).astype("float32")
            if sub.sizes["time"] == 0:
                continue
            daily = aggregate_daily(sub, "sum", min_count=20).compute()
            pieces.append(daily)
            LOG.info("  AORC %s: %d days", per.strftime("%Y-%m"), daily.sizes["time"])
        if not pieces:
            LOG.warning("No AORC data for %d", year)
            continue
        daily = xr.concat(pieces, "time").rename({lat_name: "lat", lon_name: "lon"})
        daily = snap_to_grid(daily, fine)
        daily = daily.transpose("time", "lat", "lon").astype("float32")
        daily.attrs = {"units": "mm/day", "long_name": "AORC daily precipitation", "source": str(source)}
        to_netcdf(xr.Dataset({"precip": daily}), out)
        written.append(out)
        LOG.info("AORC %d -> %s", year, out)
    return written


def snap_to_grid(da: xr.DataArray, grid: Grid) -> xr.DataArray:
    """Select the nearest native cell for every target centre and relabel coords."""
    tol = grid.res * 0.45
    da = da.sortby("lat").sortby("lon")
    out = da.sel(lat=grid.lat, lon=grid.lon, method="nearest", tolerance=tol)
    if out.sizes["lat"] != len(grid.lat) or out.sizes["lon"] != len(grid.lon):
        raise ValueError("Native grid does not cover the target grid")
    if len(np.unique(out.lat.values)) != len(grid.lat) or len(np.unique(out.lon.values)) != len(grid.lon):
        raise ValueError("Ambiguous nearest-neighbour mapping onto the target grid")
    return out.assign_coords(lat=grid.lat, lon=grid.lon)


# =========================================================================== #
# NLCD land cover / impervious (MRLC WCS) and Copernicus DEM
# =========================================================================== #
NLCD_CRS = "EPSG:5070"
COP_DEM_URL = "https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"


def _nlcd_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["raw"]) / "nlcd"


def _domain_in_crs(grid: Grid, crs: str, pad_m: float = 3000.0) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    lon_min, lat_min, lon_max, lat_max = grid.edges
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lons = np.linspace(lon_min, lon_max, 25)
    lats = np.linspace(lat_min, lat_max, 25)
    pts = [(lo, lat_min) for lo in lons] + [(lo, lat_max) for lo in lons]
    pts += [(lon_min, la) for la in lats] + [(lon_max, la) for la in lats]
    xs, ys = zip(*[t.transform(lo, la) for lo, la in pts])
    return min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m


def _fetch_wcs_tile(url: str, out: Path, retries: int = 5) -> None:
    """GET one WCS tile, retrying transient server failures.

    MRLC's geoserver answers overload with HTTP 200 carrying an HTML error
    page rather than a GeoTIFF, so the content is what has to be checked, not
    the status code. An unretried tile throws away every tile fetched before
    it, which on a full domain is close to an hour.
    """
    import time as _time

    import requests

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=600)
            ok = r.status_code == 200 and (r.content.startswith(b"II") or r.content.startswith(b"MM"))
            if ok:
                out.write_bytes(r.content)
                return
            detail = f"HTTP {r.status_code}, {len(r.content)} bytes, not a GeoTIFF"
        except Exception as e:  # noqa: BLE001
            detail = f"{type(e).__name__}: {e}"
        if attempt == retries - 1:
            raise RuntimeError(f"WCS tile failed after {retries} attempts ({detail}): {url}")
        wait = 20 * (attempt + 1)
        LOG.warning("WCS tile failed (%s), retrying in %ds", detail, wait)
        _time.sleep(wait)


def fetch_wcs_tiles(cfg: dict, grid: Grid, coverage: str, tag: str) -> list[Path]:
    """Download 30 m NLCD tiles covering the domain from the MRLC WCS.

    Tiles are cached to disk, so an interrupted download resumes.
    """
    ncfg = cfg["data"]["nlcd"]
    tile_m = float(ncfg.get("tile_km", 60)) * 1000.0
    x0, y0, x1, y1 = _domain_in_crs(grid, NLCD_CRS)
    out_dir = _nlcd_dir(cfg) / "tiles" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(x0, x1, tile_m)
    ys = np.arange(y0, y1, tile_m)
    files = []
    for i, tx in enumerate(xs):
        for j, ty in enumerate(ys):
            out = out_dir / f"{tag}_{i:02d}_{j:02d}.tif"
            files.append(out)
            if out.exists() and out.stat().st_size > 0:
                continue
            url = (
                f"{ncfg['wcs_url']}?service=WCS&version=2.0.1&request=GetCoverage&coverageId={coverage}"
                f"&subset=X({tx:.0f},{min(tx + tile_m, x1):.0f})&subset=Y({ty:.0f},{min(ty + tile_m, y1):.0f})"
                f"&format=image/geotiff"
            )
            LOG.info("WCS %s tile %d/%d", tag, i * len(ys) + j + 1, len(xs) * len(ys))
            _fetch_wcs_tile(url, out)
    return files


def _mosaic(files: list[Path]):
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(f) for f in files]
    try:
        arr, transform = merge(srcs)
        crs = srcs[0].crs
        nodata = srcs[0].nodata
        dtype = srcs[0].dtypes[0]
    finally:
        for s in srcs:
            s.close()
    return arr[0], transform, crs, nodata, dtype


def reproject_to_grid(src: np.ndarray, src_transform, src_crs, grid: Grid, resampling: str,
                      src_nodata=None, dst_dtype=np.float32) -> np.ndarray:
    """Warp ``src`` onto ``grid`` and return an (lat ascending, lon) array."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    dst = np.full(grid.shape, np.nan if np.issubdtype(dst_dtype, np.floating) else 0, dtype=dst_dtype)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform(),
        dst_crs="EPSG:4326",
        dst_nodata=np.nan if np.issubdtype(dst_dtype, np.floating) else 0,
        resampling=getattr(Resampling, resampling),
    )
    return dst[::-1]  # north-up raster -> ascending latitude


def write_geotiff(arr: np.ndarray, grid: Grid, path: str | Path, nodata=None, descriptions=None) -> None:
    """Write an (lat ascending, lon) array or a (band, lat, lon) stack as GeoTIFF."""
    import rasterio

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    data = arr[:, ::-1, :]  # ascending lat -> north-up
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
        dtype=data.dtype, crs="EPSG:4326", transform=grid.transform(), nodata=nodata,
        compress="deflate", tiled=False,
    ) as dst:
        dst.write(data)
        if descriptions:
            for b, d in enumerate(descriptions, start=1):
                dst.set_band_description(b, d)


def read_geotiff(path: str | Path, grid: Grid | None = None) -> np.ndarray:
    """Read a north-up GeoTIFF into an (lat ascending, lon) array (or band stack)."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read()
        if grid is not None and (src.height, src.width) != grid.shape:
            raise ValueError(f"{path}: shape {(src.height, src.width)} != grid {grid.shape}")
    data = data[:, ::-1, :]
    return data[0] if data.shape[0] == 1 else data


def download_dem(cfg: dict, grids: GridPair) -> Path:
    """Build ``dem_1km.tif`` from Copernicus GLO-90 tiles."""
    import requests

    out = _nlcd_dir(cfg) / "dem_1km.tif"
    if out.exists():
        LOG.info("DEM exists: %s", out)
        return out
    fine = grids.fine
    lon_min, lat_min, lon_max, lat_max = fine.edges
    tiles_dir = _nlcd_dir(cfg) / "dem_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for la in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lo in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            ns = f"N{la:02d}" if la >= 0 else f"S{-la:02d}"
            ew = f"E{lo:03d}" if lo >= 0 else f"W{-lo:03d}"
            name = f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"
            local = tiles_dir / f"{name}.tif"
            if not local.exists():
                url = COP_DEM_URL.format(name=name)
                LOG.info("Downloading DEM tile %s", name)
                r = requests.get(url, timeout=600)
                if r.status_code == 404:
                    LOG.warning("DEM tile %s not found (ocean?)", name)
                    continue
                r.raise_for_status()
                local.write_bytes(r.content)
            files.append(local)
    if not files:
        raise RuntimeError("No DEM tiles downloaded")
    arr, transform, crs, nodata, _ = _mosaic(files)
    method = cfg["data"]["nlcd"].get("dem_resampling", "average")
    dem = reproject_to_grid(arr.astype(np.float32), transform, crs, fine, method, src_nodata=nodata)
    write_geotiff(dem.astype(np.float32), fine, out, nodata=np.nan, descriptions=["elevation_m"])
    LOG.info("DEM 1 km -> %s (min %.0f, max %.0f m)", out, np.nanmin(dem), np.nanmax(dem))
    return out


def download_nlcd(cfg: dict, grids: GridPair) -> dict[str, Path]:
    """Build 1 km land-cover, class-fraction and impervious rasters from MRLC WCS."""
    ncfg = cfg["data"]["nlcd"]
    fine = grids.fine
    out_dir = _nlcd_dir(cfg)
    paths = {
        "dem": download_dem(cfg, grids),
        "lulc": out_dir / "lulc_1km.tif",
        "lc_frac": out_dir / "lc_frac_1km.tif",
        "impervious": out_dir / "impervious_1km.tif",
    }
    if not (paths["lulc"].exists() and paths["lc_frac"].exists()):
        tiles = fetch_wcs_tiles(cfg, fine, ncfg["landcover_coverage"], f"lc{ncfg['year']}")
        arr, transform, crs, nodata, _ = _mosaic(tiles)
        lulc = reproject_to_grid(arr, transform, crs, fine, "mode", src_nodata=0, dst_dtype=np.uint8)
        write_geotiff(lulc, fine, paths["lulc"], nodata=0, descriptions=["nlcd_class"])
        fracs = []
        for name, classes in LC_GROUPS.items():
            mask = np.isin(arr, classes).astype(np.uint8)
            fr = reproject_to_grid(mask, transform, crs, fine, "average", src_nodata=None)
            fracs.append(fr)
            LOG.info("  lc fraction %-10s mean %.3f", name, np.nanmean(fr))
        write_geotiff(np.stack(fracs).astype(np.float32), fine, paths["lc_frac"], nodata=np.nan,
                      descriptions=list(LC_GROUPS))
        LOG.info("Land cover 1 km -> %s", paths["lulc"])
    if not paths["impervious"].exists():
        tiles = fetch_wcs_tiles(cfg, fine, ncfg["impervious_coverage"], f"imp{ncfg['year']}")
        arr, transform, crs, nodata, _ = _mosaic(tiles)
        arr = arr.astype(np.float32)
        arr[arr > 100] = np.nan  # legacy nodata (127)
        imp = reproject_to_grid(arr, transform, crs, fine, "average", src_nodata=np.nan) / 100.0
        write_geotiff(imp.astype(np.float32), fine, paths["impervious"], nodata=np.nan,
                      descriptions=["impervious_fraction"])
        LOG.info("Impervious 1 km -> %s", paths["impervious"])
    return paths


def load_nlcd_1km(cfg: dict, grids: GridPair) -> xr.Dataset:
    """Load the 1 km static rasters into a Dataset on the fine grid.

    ``lc_frac_*`` come from ``lc_frac_1km.tif`` when present (30 m fractions),
    otherwise from a one-hot encoding of the dominant class.
    """
    fine = grids.fine
    d = _nlcd_dir(cfg)
    for name in ("dem_1km.tif", "lulc_1km.tif", "impervious_1km.tif"):
        if not (d / name).exists():
            raise FileNotFoundError(f"Missing {d / name}; run `main.py download --nlcd` or `--synthetic`")
    dem = read_geotiff(d / "dem_1km.tif", fine).astype(np.float32)
    lulc = read_geotiff(d / "lulc_1km.tif", fine).astype(np.uint8)
    imp = read_geotiff(d / "impervious_1km.tif", fine).astype(np.float32)
    ds = xr.Dataset(
        {
            "dem": (("lat", "lon"), dem, {"units": "m", "long_name": "elevation"}),
            "lulc": (("lat", "lon"), lulc, {"long_name": "NLCD class (mode)"}),
            "imperv": (("lat", "lon"), np.clip(imp, 0, 1), {"long_name": "impervious fraction"}),
        },
        coords=fine.coords(),
    )
    if (d / "lc_frac_1km.tif").exists():
        fr = read_geotiff(d / "lc_frac_1km.tif", fine).astype(np.float32)
        for k, name in enumerate(LC_GROUPS):
            ds[f"lc_frac_{name}"] = (("lat", "lon"), fr[k])
    else:
        for name, classes in LC_GROUPS.items():
            ds[f"lc_frac_{name}"] = (("lat", "lon"), np.isin(lulc, classes).astype(np.float32))
    return ds


# =========================================================================== #
# Grid alignment (Step 4)
# =========================================================================== #
def align_grids(cfg: dict, grids: GridPair, force: bool = False) -> dict[str, Path]:
    """Produce the aligned IMERG / AORC / NLCD products on the master grids.

    Each product is skipped when its outputs already exist (unless ``force``);
    IMERG is skipped with a warning when no granules have been fetched yet so
    AORC/NLCD can be prepared before Earthdata credentials are available.
    """
    from .feature_engineering import compute_derived_features

    proc = Path(cfg["paths"]["processed"])
    aux = Path(cfg["paths"]["auxiliary"])
    time = date_range_daily(cfg["time"]["start"], cfg["time"]["end"])
    outputs = {
        "imerg_10km": proc / "imerg_aligned_10km.nc",
        "aorc_1km": proc / "aorc_aligned_1km.nc",
        "aorc_10km": proc / "aorc_aligned_10km.nc",
        "nlcd_1km": proc / "nlcd_aligned_1km.nc",
        "dem_slope_aspect": aux / "dem_slope_aspect.nc",
    }

    # --- IMERG ------------------------------------------------------------- #
    if outputs["imerg_10km"].exists() and not force:
        LOG.info("IMERG aligned exists: %s", outputs["imerg_10km"])
    else:
        raw = proc / "imerg_raw_10km.nc"
        if not raw.exists():
            try:
                build_imerg_raw(cfg, grids)
            except FileNotFoundError as e:
                LOG.warning("IMERG not aligned: %s", e)
                raw = None
        if raw is not None:
            imerg = open_dataset(raw).load()
            assert_on_grid(imerg["precip"], grids.coarse, "IMERG")
            imerg = imerg.reindex(time=time)
            n_missing = int(imerg["precip"].isnull().all(("lat", "lon")).sum())
            if n_missing:
                LOG.warning("IMERG: %d of %d days entirely missing", n_missing, len(time))
            to_netcdf(imerg, outputs["imerg_10km"])
            LOG.info("IMERG aligned -> %s", outputs["imerg_10km"])

    # --- AORC -------------------------------------------------------------- #
    if outputs["aorc_1km"].exists() and outputs["aorc_10km"].exists() and not force:
        LOG.info("AORC aligned exists: %s", outputs["aorc_1km"])
    else:
        years = range(pd.Timestamp(cfg["time"]["start"]).year, pd.Timestamp(cfg["time"]["end"]).year + 1)
        files = [aorc_daily_path(cfg, y) for y in years if aorc_daily_path(cfg, y).exists()]
        if not files:
            raise FileNotFoundError("No AORC daily files; run `main.py download --aorc` or `--synthetic`")
        import dask

        aorc = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4", chunks={"time": 64})
        assert_on_grid(aorc["precip"], grids.fine, "AORC")
        aorc = aorc.reindex(time=time)
        # netCDF4/HDF5 is not thread-safe. Writing the full multi-year cube
        # through dask's default threaded scheduler deadlocks: the .tmp file
        # freezes at a few tens of KB and the process sits at 0 % CPU
        # indefinitely. The synchronous scheduler is slower but finishes.
        with dask.config.set(scheduler="synchronous"):
            to_netcdf(aorc, outputs["aorc_1km"])
            LOG.info("AORC 1 km aligned -> %s", outputs["aorc_1km"])
            aorc_c = coarsen_mean(aorc["precip"], grids).astype("float32")
            aorc_c.attrs = {"units": "mm/day", "long_name": "AORC block-mean precipitation at 10 km"}
            to_netcdf(xr.Dataset({"precip": aorc_c}).load(), outputs["aorc_10km"])
        aorc.close()
        LOG.info("AORC 10 km coarsened -> %s", outputs["aorc_10km"])

    # --- NLCD / DEM -------------------------------------------------------- #
    if outputs["nlcd_1km"].exists() and not force:
        LOG.info("NLCD aligned exists: %s", outputs["nlcd_1km"])
    else:
        nlcd = load_nlcd_1km(cfg, grids)
        derived = compute_derived_features(nlcd["dem"], nlcd["lulc"])
        nlcd = nlcd.merge(derived)
        to_netcdf(nlcd, outputs["nlcd_1km"])
        to_netcdf(nlcd[["dem", "slope", "aspect"]], outputs["dem_slope_aspect"])
        LOG.info("NLCD aligned -> %s", outputs["nlcd_1km"])
    return {k: v for k, v in outputs.items() if v.exists()}


def cleanup_raw(cfg: dict, what: tuple[str, ...] = ("imerg",)) -> None:
    for sub in what:
        d = Path(cfg["paths"]["raw"]) / sub
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True)
