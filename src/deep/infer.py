"""Full-domain inference for the deep downscalers.

Writes NetCDF in the same layout as the tree-based product so the existing
validation framework can score it:

    results/deep/{model}_1km_{y0}_{y1}.nc
        precipitation(time, lat, lon)     deterministic field, or the ensemble
                                          mean for the diffusion model
        precipitation_member0(...)        one diffusion realisation (kept so
                                          spectra / extremes can be scored on a
                                          single member rather than a mean)

For the diffusion model CRPS and ensemble spread are accumulated on the fly, so
the full ensemble never has to be written to disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from .data import DownscalingData, FullFieldDataset
from .diffusion import ResidualDiffusion
from .models import build_model
from .train import get_device


def load_checkpoint(path, n_cond, factor, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = build_model(ck["model"], n_cond, factor, base=a.get("base", 64), dim=a.get("dim", 180),
                    depths=(a.get("depth", 6),) * a.get("groups", 4), heads=a.get("heads", 6))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def crps_ensemble(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS estimate for an ensemble, shape (M, ...) vs (...)."""
    m = members.shape[0]
    mae = np.abs(members - obs[None]).mean(0)
    s = np.sort(members, axis=0)
    # E|X-X'| via the sorted-sample identity, O(M log M) instead of O(M^2)
    w = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (members.ndim - 1))
    spread = 2.0 * (w * s).sum(0) / (m * (m - 1))
    return mae - 0.5 * spread


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results/deep")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    ap.add_argument("--members", type=int, default=8, help="diffusion ensemble size")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0 = deterministic)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", choices=["bf16", "off"], default="bf16")
    args = ap.parse_args(argv)

    device = get_device()
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float32
    data = DownscalingData(args.data_dir)
    model, ck = load_checkpoint(args.ckpt, data.n_cond, data.factor, device)
    name = ck["model"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff = mean_model = None
    if name == "diffusion":
        mean_model, _ = load_checkpoint(ck["mean_ckpt"], data.n_cond, data.factor, device)
        diff = ResidualDiffusion(ck["args"]["timesteps"], device=device, res_scale=ck["res_scale"],
                                 parameterization=ck.get("parameterization", "eps"),
                             min_snr_gamma=ck.get("min_snr_gamma", 0.0))

    t_idx = None if args.split != "all" else np.arange(len(data.time))
    ds_eval = FullFieldDataset(data, None if t_idx is not None else args.split, t_idx)
    dl = DataLoader(ds_eval, batch_size=args.batch, num_workers=4, pin_memory=True)

    n = len(ds_eval)
    H, W = data.shape_f
    pred = np.empty((n, H, W), np.float32)
    member0 = np.empty((n, H, W), np.float32) if diff is not None else None
    times = np.empty(n, dtype="datetime64[ns]")
    crps_sum = sprd_sum = cnt = 0.0
    k = 0
    for cond, coarse, tgt, mask, t in dl:
        cond, coarse = cond.to(device), coarse.to(device)
        b = cond.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            if diff is None:
                u = model(cond, coarse).float()
                p = data.norm.to_mm(u)[:, 0].cpu().numpy()
            else:
                mu = mean_model(cond, coarse).float()
                mem = []
                for mi in range(args.members):
                    g = torch.Generator(device=device).manual_seed(args.seed * 1000 + k * 97 + mi)
                    r = diff.sample(model, cond, coarse, mu, data.factor, steps=args.steps, eta=args.eta,
                                    generator=g)
                    mem.append(data.norm.to_mm((mu + r).float())[:, 0].cpu().numpy())
                mem = np.stack(mem)  # (M, b, H, W)
                p = mem.mean(0)
                member0[k:k + b] = mem[0]
                o = data.norm.to_mm(tgt.numpy())
                o = o[:, 0] if o.ndim == 4 else o
                # tgt has had its NaNs zero-filled, so validity must come from
                # the dataset mask rather than isfinite(o).
                mk = mask.numpy()
                valid = (mk[:, 0] if mk.ndim == 4 else mk) > 0
                c = crps_ensemble(mem, o)
                crps_sum += float(np.where(valid, c, 0).sum())
                sprd_sum += float(np.where(valid, mem.std(0), 0).sum())
                cnt += float(valid.sum())
        pred[k:k + b] = p
        times[k:k + b] = data.time[t.numpy()]
        k += b
        if k % 100 < args.batch:
            print(f"  {k}/{n} days", flush=True)

    coords = {"time": times, "lat": data.lat, "lon": data.lon}
    out = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), pred,
                           {"units": "mm/day", "long_name": f"{name} downscaled precipitation"})},
        coords=coords,
    )
    if member0 is not None:
        out["precipitation_member0"] = (("time", "lat", "lon"), member0,
                                        {"units": "mm/day", "long_name": "single diffusion realisation"})
    out.attrs.update({"method": f"deep_{name}", "checkpoint": str(args.ckpt), "split": args.split, "eta": args.eta,
                      "dev_score": float(ck.get("dev_score", np.nan)), "members": args.members if diff else 1,
                      "ddim_steps": args.steps if diff else 0})
    y0, y1 = str(times[0])[:4], str(times[-1])[:4]
    path = out_dir / f"{name}_1km_{y0}_{y1}.nc"
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out.data_vars}
    enc["time"] = {"units": "days since 1970-01-01", "dtype": "int32"}
    out.to_netcdf(path, engine="netcdf4", encoding=enc)
    print(f"[written] {path}")
    if cnt:
        stats = {"crps_mm": crps_sum / cnt, "ensemble_spread_mm": sprd_sum / cnt, "members": args.members,
                 "ddim_steps": args.steps}
        json.dump(stats, open(out_dir / f"{name}_ensemble_stats.json", "w"), indent=2)
        print("[ensemble]", stats)


if __name__ == "__main__":
    main()
