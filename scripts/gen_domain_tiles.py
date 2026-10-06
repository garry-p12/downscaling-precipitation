"""Field tiles for any domain, for the web playground.

Generalises gen_web_tiles.py: takes a config, a set of named layers and an
optional evaluation box, so Austin (IMERG), Colorado (IMERG) and the POWER
domain all produce 420 x 360 tiles the same viewer can display.

Encoding is unchanged -- 8-bit sqrt ramp in luminance, validity in alpha --
so the browser decodes real mm/day and computes differences client-side.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from src.utils import (load_config, grids_from_config, subset_box,
                       upsample_bilinear, upsample_nearest, open_downscaled)

VMAX, HEAVY, FLOOR = 200.0, 30.0, 0.1


def enc(a):
    ok = np.isfinite(a)
    z = np.where(ok, a, 0.0)
    z = np.where(z < FLOOR, 0.0, z)
    return (np.rint(np.sqrt(np.clip(z, 0, VMAX) / VMAX) * 255).astype(np.uint8),
            (ok.astype(np.uint8) * 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None, help="dir of downscaled_1km_*.nc (default: cfg results)")
    ap.add_argument("--deep-dir", default=None, help="dir holding <name>_1km_*.nc for the deep products")
    ap.add_argument("--deep", default="", help="comma list of deep layers to include, e.g. cnn,swin")
    ap.add_argument("--wettest", type=int, default=90)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=24)
    a = ap.parse_args()

    cfg = load_config(a.config)
    grids = grids_from_config(cfg)
    proc = Path(cfg["paths"]["processed"])
    box = (cfg.get("evaluation") or {}).get("bbox")
    out = Path(a.out); (out / "fields").mkdir(parents=True, exist_ok=True)

    sl = slice(cfg["time"]["val_start"], cfg["time"]["val_end"])
    obs = xr.open_dataset(proc / "aorc_aligned_1km.nc")["precip"].sel(time=sl)
    coarse = xr.open_dataset(proc / "imerg_aligned_10km.nc")["precip"].sel(time=sl)
    ml = open_downscaled(Path(a.results or cfg["paths"]["results"])).sel(time=sl)

    # Deep products are named explicitly rather than globbed: a stale file from
    # a superseded run sits in the same directory as a good one, and a tile set
    # built from the wrong checkpoint looks exactly like a right one.
    deep = {}
    for name in [x.strip() for x in a.deep.split(",") if x.strip()]:
        ddir = Path(a.deep_dir or a.results or cfg["paths"]["results"])
        hits = sorted(ddir.glob(f"{name}_1km_*.nc"))
        if not hits:
            raise SystemExit(f"no {name}_1km_*.nc in {ddir}")
        print(f"deep layer {name}: {hits[-1]}", flush=True)
        ds = xr.open_dataset(hits[-1], chunks={"time": 24})
        deep[name] = ds["precipitation"].sel(time=sl)
        # the diffusion file carries one realisation alongside the ensemble mean;
        # it is the only product with realistic texture, so it gets its own layer
        if "precipitation_member0" in ds:
            deep[f"{name}_member"] = ds["precipitation_member0"].sel(time=sl)
            print(f"deep layer {name}_member: same file", flush=True)
    if box:
        obs = subset_box(obs, box)
    times = obs["time"].values

    dm = obs.mean(dim=("lat", "lon")).values
    keep = set(np.argsort(dm)[::-1][:a.wettest].tolist())
    if a.stride:
        keep |= set(range(0, len(times), a.stride))
    sel = sorted(keep)
    times = times[sel]
    print(f"{len(times)} of {len(dm)} days; grid {obs.sizes['lat']} x {obs.sizes['lon']}", flush=True)

    LAYERS = ["aorc", "nearest", "bilinear", "ml"] + list(deep)
    for L in LAYERS:
        (out / "fields" / L).mkdir(parents=True, exist_ok=True)

    days = []
    for t0 in range(0, len(times), a.chunk):
        ts = times[t0:t0 + a.chunk]
        o = obs.sel(time=ts).values.astype(np.float32)
        c = coarse.sel(time=ts)
        f = {"aorc": o,
             "nearest": upsample_nearest(c, grids),
             "bilinear": upsample_bilinear(c, grids),
             "ml": ml.sel(time=ts),
             **{k: v.sel(time=ts) for k, v in deep.items()}}
        for k in [L for L in LAYERS if L != "aorc"]:
            f[k] = (subset_box(f[k], box) if box else f[k]).values.astype(np.float32)
        for i, t in enumerate(ts):
            date = str(t)[:10]
            ok0 = np.isfinite(o[i])
            rec = {"date": date, "obsMean": float(np.nanmean(o[i])), "obsMax": float(np.nanmax(o[i])),
                   "obsHeavy": int(np.sum(o[i][ok0] >= HEAVY)), "m": {}}
            for L in LAYERS:
                Image.fromarray(np.dstack(enc(f[L][i])), mode="LA").save(
                    out / "fields" / L / f"{date}.png", optimize=True)
                if L == "aorc":
                    continue
                ok = ok0 & np.isfinite(f[L][i])
                e = f[L][i][ok] - o[i][ok]
                rec["m"][L] = {"rmse": float(np.sqrt((e ** 2).mean())),
                               "bias": float(e.mean()), "max": float(f[L][i][ok].max())}
            days.append(rec)
        print(f"  {str(ts[0])[:10]} .. {str(ts[-1])[:10]} ({t0 + len(ts)}/{len(times)})", flush=True)

    lat, lon = obs["lat"].values, obs["lon"].values
    (out / "meta.json").write_text(json.dumps({
        "vmax": VMAX, "heavy": HEAVY, "floor": FLOOR,
        "width": int(obs.sizes["lon"]), "height": int(obs.sizes["lat"]),
        "bbox": [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())],
        "layers": LAYERS, "nDays": len(days), "factor": grids.factor,
        "period": [days[0]["date"], days[-1]["date"]]}))
    (out / "days.json").write_text(json.dumps(days, separators=(",", ":")))
    print("done:", out)


main()
