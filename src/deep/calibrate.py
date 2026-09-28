"""Check that sampled residuals have the same spread as the observed ones.

A conditional generative downscaler is only useful if its samples are
*calibrated*: too little spread and every member is the blurry conditional
mean, too much and exponentiating back to mm inflates the field (log-space
over-dispersion becomes a large positive bias, because E[exp(x)] grows with
Var[x]). This compares, on the **dev** year only, the sampled residual
distribution against the observed one for several DDIM noise levels, and
prints the resulting mm-space bias so the sampler can be chosen before the
test years are touched.

    python -m src.deep.calibrate --ckpt models/deep/diffusion_best.pt --days 12
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .data import DownscalingData, FullFieldDataset
from .diffusion import ResidualDiffusion
from .infer import load_checkpoint
from .train import get_device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", default="40", help="comma-separated DDIM step counts to sweep")
    ap.add_argument("--etas", default="0.0,0.5,1.0")
    ap.add_argument("--out", default="models/deep/diffusion_calibration.json")
    args = ap.parse_args(argv)

    device = get_device()
    data = DownscalingData(args.data_dir)
    model, ck = load_checkpoint(args.ckpt, data.n_cond, data.factor, device)
    mean_model, _ = load_checkpoint(ck["mean_ckpt"], data.n_cond, data.factor, device)
    diff = ResidualDiffusion(ck["args"]["timesteps"], device=device, res_scale=ck["res_scale"],
                             parameterization=ck.get("parameterization", "eps"),
                             min_snr_gamma=ck.get("min_snr_gamma", 0.0))
    rs = ck["res_scale"]

    idx = data.split_index("dev")
    idx = idx[:: max(1, len(idx) // args.days)][: args.days]
    ds = FullFieldDataset(data, t_idx=idx)
    amp = torch.bfloat16 if device.type == "cuda" else torch.float32

    obs_r, obs_mm = [], []
    ctx = []
    with torch.no_grad():
        for i in range(len(ds)):
            cond, coarse, tgt, mask, _ = ds[i]
            cond, coarse = cond[None].to(device), coarse[None].to(device)
            tgt, mask = tgt[None].to(device), mask[None].to(device)
            with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
                mu = mean_model(cond, coarse).float()
            m = mask.bool()
            obs_r.append((((tgt - mu) / rs)[m]).float().cpu().numpy())
            obs_mm.append(data.norm.to_mm(tgt)[m].float().cpu().numpy())
            ctx.append((cond, coarse, mu, m))
    obs_r = np.concatenate(obs_r)
    obs_mm = np.concatenate(obs_mm)

    result = {"observed": {"residual_std": float(obs_r.std()), "residual_mean": float(obs_r.mean()),
                           "precip_mean_mm": float(obs_mm.mean())},
              "checkpoint": args.ckpt, "parameterization": diff.parameterization,
              "res_scale": rs, "days": len(idx), "etas": {}}
    print(f"observed: residual std {obs_r.std():.3f} mean {obs_r.mean():+.3f} | precip mean {obs_mm.mean():.3f} mm")

    best, best_score = None, np.inf
    grid = [(int(st), float(e)) for st in str(args.steps).split(",") for e in args.etas.split(",")]
    for steps, eta in grid:
        srs, smm = [], []
        with torch.no_grad():
            for k, (cond, coarse, mu, m) in enumerate(ctx):
                for j in range(args.members):
                    g = torch.Generator(device=device).manual_seed(1000 * k + j)
                    with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
                        r = diff.sample(model, cond, coarse, mu, data.factor, steps=steps, eta=eta, generator=g)
                    srs.append(((r / rs)[m]).float().cpu().numpy())
                    smm.append(data.norm.to_mm(mu + r.float())[m].float().cpu().numpy())
        srs = np.concatenate(srs)
        smm = np.concatenate(smm)
        ratio = float(srs.std() / max(obs_r.std(), 1e-6))
        bias = float(smm.mean() - obs_mm.mean())
        # rank by log-ratio of spread plus relative mm bias: both must be ~0
        score = abs(np.log(max(ratio, 1e-3))) + abs(bias) / max(obs_mm.mean(), 1e-6)
        result["etas"][f"{steps}_{eta}"] = {"steps": steps, "eta": eta, "residual_std": float(srs.std()),
                                            "spread_ratio": ratio, "precip_mean_mm": float(smm.mean()),
                                            "bias_mm": bias, "score": score}
        print(f"steps={steps:<4} eta={eta:<4}: residual std {srs.std():.3f} (ratio {ratio:5.2f})  "
              f"precip mean {smm.mean():6.3f} mm (bias {bias:+.3f})  score {score:.3f}", flush=True)
        if score < best_score:
            best, best_score = (steps, eta), score
    result["best_steps"], result["best_eta"] = best
    print(f"\nbest on the dev year: steps={best[0]} eta={best[1]}")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    return best


if __name__ == "__main__":
    main()
