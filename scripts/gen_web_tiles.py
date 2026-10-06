"""Render per-day 1 km fields as PNG data tiles + exact per-day metrics.

Value encoding: 8-bit sqrt scale, p = VMAX*(v/255)**2, so the browser can
decode a real mm/day value from any pixel and compute differences client-side.
Alpha carries the validity mask.
"""
import json, sys, argparse
from pathlib import Path
import numpy as np, xarray as xr
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from src.utils import load_config, grids_from_config, upsample_bilinear, upsample_nearest

VMAX = 200.0
HEAVY = 30.0

FLOOR = 0.1  # below this is retrieval noise; zeroing it compresses far better

def enc(a):
    """float mm/day -> uint8 on a sqrt ramp, plus alpha mask."""
    ok = np.isfinite(a)
    z = np.where(ok, a, 0.0)
    z = np.where(z < FLOOR, 0.0, z)
    v = np.sqrt(np.clip(z, 0, VMAX) / VMAX) * 255.0
    return np.rint(v).astype(np.uint8), (ok.astype(np.uint8) * 255)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="first N days only (sizing test)")
    ap.add_argument("--wettest", type=int, default=0, help="keep the N wettest days")
    ap.add_argument("--stride", type=int, default=0, help="plus every Nth day for seasonal spread")
    a = ap.parse_args()

    cfg = load_config("config.yaml")
    grids = grids_from_config(cfg)
    out = Path(a.out); (out / "fields").mkdir(parents=True, exist_ok=True)

    sl = slice(a.start, a.end)
    aorc = xr.open_dataset("data/processed/aorc_aligned_1km.nc")["precip"].sel(time=sl)
    imerg = xr.open_dataset("data/processed/imerg_aligned_10km.nc")["precip"].sel(time=sl)
    srcs = {
        "xgboost": xr.open_mfdataset(["results/downscaled_1km_2019.nc",
                                      "results/downscaled_1km_2020.nc"])["precipitation"].sel(time=sl),
        "cnn":  xr.open_dataset("results/v3/cnn_1km_2019_2020.nc")["precipitation"].sel(time=sl),
        # same architecture and budget as `cnn`, trained with --spectral-weight 0.01
        "cnn_spectral": xr.open_dataset(
            "results/spectral/spec-w001/cnn_spectral_1km_2019_2020.nc")["precipitation"].sel(time=sl),
        "swin": xr.open_dataset("results/v3/swin_1km_2019_2020.nc")["precipitation"].sel(time=sl),
        "diffusion": xr.open_dataset("results/v3/diffusion_1km_2019_2020.nc")["precipitation"].sel(time=sl),
        "diffusion_member": xr.open_dataset(
            "results/v3/diffusion_1km_2019_2020.nc")["precipitation_member0"].sel(time=sl),
    }
    times = aorc["time"].values
    if a.wettest or a.stride:
        dm = aorc.mean(dim=("lat", "lon")).values
        keep = set()
        if a.wettest:
            keep |= set(np.argsort(dm)[::-1][:a.wettest].tolist())
        if a.stride:
            keep |= set(range(0, len(times), a.stride))
        sel = sorted(keep)
        print(f"selected {len(sel)} of {len(times)} days "
              f"(wettest {a.wettest}, stride {a.stride})")
        times = times[sel]
    if a.limit:
        times = times[:a.limit]
    n = len(times)
    ny, nx = aorc.sizes["lat"], aorc.sizes["lon"]
    print(f"{n} days, grid {ny} x {nx}")

    LAYERS = ["aorc", "nearest", "bilinear", "xgboost", "cnn", "cnn_spectral",
              "swin", "diffusion", "diffusion_member"]
    for L in LAYERS:
        (out / "fields" / L).mkdir(parents=True, exist_ok=True)

    days = []
    for t0 in range(0, n, a.chunk):
        ts = times[t0:t0 + a.chunk]
        obs = aorc.sel(time=ts).values.astype(np.float32)
        im = imerg.sel(time=ts)
        fields = {"aorc": obs,
                  "nearest": upsample_nearest(im, grids).values.astype(np.float32),
                  "bilinear": upsample_bilinear(im, grids).values.astype(np.float32)}
        for k, da in srcs.items():
            fields[k] = da.sel(time=ts).values.astype(np.float32)

        for i, t in enumerate(ts):
            date = str(t)[:10]
            o = obs[i]
            m_ok = np.isfinite(o)
            rec = {"date": date,
                   "obsMean": float(np.nanmean(o)), "obsMax": float(np.nanmax(o)),
                   "obsWet": float(np.mean(o[m_ok] >= 1.0)),
                   "obsHeavy": int(np.sum(o[m_ok] >= HEAVY)),
                   "m": {}}
            for L in LAYERS:
                f = fields[L][i]
                img = Image.fromarray(np.dstack(enc(f)), mode="LA")
                img.save(out / "fields" / L / f"{date}.png", optimize=True)
                if L == "aorc":
                    continue
                ok = m_ok & np.isfinite(f)
                e = f[ok] - o[ok]
                po, oo = f[ok] >= HEAVY, o[ok] >= HEAVY
                h, mi, fa = int((po & oo).sum()), int((~po & oo).sum()), int((po & ~oo).sum())
                sp, so = f[ok], o[ok]
                cc = np.corrcoef(sp, so)[0, 1] if sp.std() > 0 and so.std() > 0 else np.nan
                rec["m"][L] = {
                    "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
                    "bias": float(e.mean()), "r": None if not np.isfinite(cc) else float(cc),
                    "max": float(sp.max()), "mean": float(sp.mean()),
                    "wet": float(np.mean(sp >= 1.0)),
                    "pod": (h / (h + mi)) if (h + mi) else None,
                    "far": (fa / (h + fa)) if (h + fa) else None,
                    "csi": (h / (h + mi + fa)) if (h + mi + fa) else None,
                }
            days.append(rec)
        print(f"  {str(ts[0])[:10]} .. {str(ts[-1])[:10]}  ({t0 + len(ts)}/{n})", flush=True)

    lat = aorc["lat"].values; lon = aorc["lon"].values
    meta = {"vmax": VMAX, "heavy": HEAVY, "width": int(nx), "height": int(ny),
            "encoding": "p = vmax * (v/255)^2 ; alpha 0 = no data",
            "bbox": [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())],
            "layers": LAYERS, "nDays": len(days), "floor": FLOOR,
            "period": [days[0]["date"], days[-1]["date"]]}
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "days.json").write_text(json.dumps(days, separators=(",", ":")))
    print("meta:", json.dumps(meta))

main()
