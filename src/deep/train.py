"""Training entry point for the deep downscalers.

    python -m src.deep.train --model cnn       --data-dir data/processed --out models/deep
    python -m src.deep.train --model swin      ...
    python -m src.deep.train --model diffusion --mean-ckpt models/deep/cnn_best.pt ...

Model selection uses the 2018 dev year only; 2019-2020 is never seen.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DownscalingData, FullFieldDataset, PatchDataset
from .diffusion import ResidualDiffusion
from .models import Downscaler, build_model, count_params


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


# Networks always *output* standardised log space (stable, and what the
# diffusion model conditions on), but the loss can be taken in either space.
# Log-space MSE fits the conditional mean of log precipitation, which by
# Jensen's inequality under-estimates the conditional mean in mm — the metric
# everything is scored on. ``mse``/``hybrid`` therefore push the loss back into
# mm space.
U_CLAMP = (-5.0, 10.0)


def quantile_loss(pred, obs, tau: float):
    """Pinball loss. Minimised by the tau-quantile rather than the mean."""
    err = obs - pred
    return torch.maximum(tau * err, (tau - 1.0) * err)


def masked_loss(u_pred, u_tgt, mask, norm, kind: str, scale: float, w_log: float = 0.5,
                tau: float = 0.9, w_quantile: float = 0.3, heavy_mm: float = 0.0,
                heavy_weight: float = 1.0, sample_w=None):
    """Masked training loss.

    ``kind``
        ``logmse``   MSE in standardised log space.
        ``mse``      MSE in mm space (the metric everything is scored on).
        ``hybrid``   convex blend of the two.
        ``quantile`` ``(1-w_quantile)`` x mm MSE + ``w_quantile`` x pinball loss at
                     ``tau``. The pinball term targets an upper quantile, so it
                     pushes the field up where the distribution is right-skewed
                     -- it trades mean bias for heavy-event detection.

    ``heavy_mm``/``heavy_weight`` additionally upweight cells whose *observed*
    value exceeds ``heavy_mm``. ``sample_w`` is a per-sample importance weight
    (see :class:`PatchDataset`), used to undo wet-biased crop sampling.
    """
    w = mask
    if sample_w is not None:
        w = w * sample_w.view(-1, 1, 1, 1)
    if heavy_mm > 0 and heavy_weight != 1.0:
        o_mm = norm.to_mm(u_tgt)
        w = w * torch.where(o_mm >= heavy_mm, heavy_weight, 1.0)
    denom = w.sum().clamp(min=1.0)

    log_mse = (w * (u_pred - u_tgt) ** 2).sum() / denom
    if kind == "logmse":
        return log_mse
    p = norm.to_mm(u_pred.clamp(*U_CLAMP))
    o = norm.to_mm(u_tgt)
    mm_mse = (w * ((p - o) / scale) ** 2).sum() / denom
    if kind == "mse":
        return mm_mse
    if kind == "hybrid":
        return w_log * log_mse + (1 - w_log) * mm_mse
    if kind == "quantile":
        q = (w * quantile_loss(p / scale, o / scale, tau)).sum() / denom
        return (1 - w_quantile) * mm_mse + w_quantile * q
    raise ValueError(f"unknown loss {kind}")


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def eval_deterministic(model, data, loader, device, amp_dtype, heavy_mm: float = 30.0) -> dict:
    """Full-field scores in mm on a split.

    Reports heavy-event detection alongside RMSE because the losses that improve
    extremes (pinball, heavy upweighting) are expected to cost mean-square
    skill: without POD in view the ablation cannot see the trade it is making.
    """
    model.eval()
    se = n = bias = ae = 0.0
    hits = misses = false_alarms = 0
    for cond, coarse, tgt, mask, _ in loader:
        cond, coarse, tgt, mask = [x.to(device, non_blocking=True) for x in (cond, coarse, tgt, mask)]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            u = model(cond, coarse)
        p = data.norm.to_mm(u.float())
        o = data.norm.to_mm(tgt.float())
        d = (p - o) * mask
        se += float((d**2).sum())
        ae += float(d.abs().sum())
        bias += float(d.sum())
        n += float(mask.sum())
        m = mask.bool()
        pe, oe = (p >= heavy_mm) & m, (o >= heavy_mm) & m
        hits += int((pe & oe).sum())
        misses += int((~pe & oe).sum())
        false_alarms += int((pe & ~oe).sum())
    model.train()
    return {"rmse": math.sqrt(se / max(n, 1)), "bias": bias / max(n, 1), "mae": ae / max(n, 1),
            "pod30": hits / max(hits + misses, 1), "csi30": hits / max(hits + misses + false_alarms, 1)}


@torch.no_grad()
def eval_diffusion_loss(net, diff, mean_model, data, loader, device, amp_dtype, factor) -> dict:
    net.eval()
    tot = k = 0.0
    # Fork the RNG so the fixed evaluation noise does not perturb training.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        for cond, coarse, tgt, mask, _ in loader:
            cond, coarse, tgt, mask = [x.to(device, non_blocking=True) for x in (cond, coarse, tgt, mask)]
            tgt = tgt if tgt.dim() == 4 else tgt[:, None]
            mask = mask if mask.dim() == 4 else mask[:, None]
            torch.manual_seed(1234 + int(k))
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                mu = mean_model(cond, coarse)
                loss = diff.loss(net, cond, coarse, mu.float(), tgt, mask, factor)
            tot += float(loss)
            k += 1
    net.train()
    return {"loss": tot / max(k, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cnn", "swin", "diffusion"], required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="models/deep")
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--no-augment", action="store_true", help="disable flip augmentation")
    ap.add_argument("--wet-bias", type=float, default=0.8,
                    help="fraction of crops drawn towards rain; 0 = uniform, matching the evaluation distribution")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dev-days", type=int, default=0, help="evaluate on every k-th dev day (0 = all)")
    ap.add_argument("--base", type=int, default=64, help="U-Net base width")
    ap.add_argument("--dim", type=int, default=180, help="Swin embedding dim")
    ap.add_argument("--groups", type=int, default=4, help="Swin RSTB groups")
    ap.add_argument("--depth", type=int, default=6, help="Swin layers per group")
    ap.add_argument("--heads", type=int, default=6, help="Swin attention heads")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--param", choices=["x0", "eps", "v"], default="x0",
                    help="diffusion target; x0 is stable where the cosine schedule drives abar -> 0")
    ap.add_argument("--min-snr-gamma", type=float, default=0.0,
                    help="min-SNR-gamma loss weighting; rebalances timesteps away from the high-noise "
                         "end where the optimum is the conditional mean (0 = uniform)")
    ap.add_argument("--mean-ckpt", default=None, help="frozen deterministic model for residual diffusion")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop training after this wall-clock budget so inference still runs (0 = no limit)")
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--loss", choices=["logmse", "mse", "hybrid", "quantile"], default="mse",
                    help="space the deterministic loss is taken in (see masked_loss)")
    ap.add_argument("--w-log", type=float, default=0.5, help="log-space weight for --loss hybrid")
    ap.add_argument("--tau", type=float, default=0.9, help="quantile for --loss quantile")
    ap.add_argument("--w-quantile", type=float, default=0.3, help="pinball weight for --loss quantile")
    ap.add_argument("--heavy-mm", type=float, default=0.0, help="upweight observed cells above this (mm/day)")
    ap.add_argument("--heavy-weight", type=float, default=1.0, help="weight applied above --heavy-mm")
    ap.add_argument("--importance-weight", action="store_true",
                    help="reweight wet-biased crops back to the evaluation distribution")
    ap.add_argument("--tag", default="", help="suffix for checkpoint/log filenames")
    ap.add_argument("--select", choices=["rmse", "rmse_pod"], default="rmse_pod",
                    help="checkpoint criterion: plain RMSE, or best RMSE among checkpoints whose "
                         "heavy-event POD clears --pod-floor (falls back to RMSE if none do)")
    ap.add_argument("--pod-floor", type=float, default=0.633,
                    help="POD>30mm a checkpoint must reach to be eligible under --select rmse_pod "
                         "(default: bilinear's dev-2018 value)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": torch.float32}[args.amp]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name_tag = args.model + (f"_{args.tag}" if args.tag else "")

    t0 = time.time()
    data = DownscalingData(args.data_dir)
    print(f"[data] coarse {data.shape_c} fine {data.shape_f} | train {len(data.split_index('train'))} "
          f"dev {len(data.split_index('dev'))} test {len(data.split_index('test'))} days | "
          f"cond channels {data.n_cond} | loaded in {time.time()-t0:.1f}s", flush=True)

    train_ds = PatchDataset(data, "train", patch=args.patch, n_per_epoch=args.batch * 200,
                            wet_bias=args.wet_bias, seed=args.seed, augment=not args.no_augment)
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True)
    dev_idx = data.split_index("dev")
    if args.dev_days:
        dev_idx = dev_idx[:: max(1, len(dev_idx) // args.dev_days)]
    dev_dl = DataLoader(FullFieldDataset(data, t_idx=dev_idx), batch_size=4, num_workers=2, pin_memory=True)
    print(f"[data] dev evaluation on {len(dev_idx)} days", flush=True)

    factor = data.factor
    if args.model in ("cnn", "swin"):
        model = build_model(args.model, data.n_cond, factor, base=args.base, dim=args.dim,
                            depths=(args.depth,) * args.groups, heads=args.heads).to(device)
        net, diff, mean_model = model, None, None
    else:
        if not args.mean_ckpt:
            raise SystemExit("--mean-ckpt (a trained cnn/swin checkpoint) is required for diffusion")
        ck = torch.load(args.mean_ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        mean_model = build_model(ck["model"], data.n_cond, factor, base=a.get("base", 64), dim=a.get("dim", 180),
                                 depths=(a.get("depth", 6),) * a.get("groups", 4),
                                 heads=a.get("heads", 6)).to(device)
        mean_model.load_state_dict(ck["state_dict"])
        mean_model.eval().requires_grad_(False)
        net = build_model("diffusion", data.n_cond, factor, base=args.base).to(device)
        # Scale residuals to unit variance so the cosine schedule is well matched.
        # This must be measured on the same distribution the model trains on
        # (wet-biased crops), and over enough samples to be stable: estimating
        # it from a handful of full fields understated it by ~40 %, which left
        # the diffusion operating at the wrong signal-to-noise ratio.
        scale_ds = PatchDataset(data, "train", patch=args.patch, n_per_epoch=args.batch * 8,
                                wet_bias=args.wet_bias, seed=args.seed + 7, augment=False)
        rs = []
        with torch.no_grad():
            for c, co, tg, mk, _w in DataLoader(scale_ds, batch_size=args.batch, num_workers=2):
                c, co, tg, mk = [x.to(device) for x in (c, co, tg, mk)]
                r = (tg - mean_model(c, co))[mk.bool()]
                rs.append(r.float().cpu())
        res_scale = float(torch.cat(rs).std().clamp(min=1e-3))
        diff = ResidualDiffusion(args.timesteps, device=device, res_scale=res_scale,
                                 parameterization=args.param, min_snr_gamma=args.min_snr_gamma)
        print(f"[diffusion] residual scale {res_scale:.4f} from {len(torch.cat(rs))/1e6:.1f} M cells "
              f"| parameterization {args.param} | min-SNR gamma {args.min_snr_gamma}", flush=True)
        model = net
    print(f"[model] {args.model}: {count_params(model)/1e6:.2f} M parameters on {device}", flush=True)

    mm_scale = float(np.nanstd(data.fine[data.split_index("train")]))
    print(f"[loss] {args.loss} (mm scale {mm_scale:.2f})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))
    ema = EMA(model, args.ema) if args.ema else None

    best = float("inf")
    best_eligible_seen = False
    history = []
    step = 0
    epoch = 0
    stop = False
    t_start = time.time()
    while step < args.steps and not stop:
        train_ds.set_epoch(epoch)
        for cond, coarse, tgt, mask, sw in train_dl:
            if step >= args.steps or stop:
                break
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                print(f"[budget] {args.max_minutes:g} min reached at step {step}; finishing up", flush=True)
                stop = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            cond, coarse, tgt, mask, sw = [x.to(device, non_blocking=True) for x in (cond, coarse, tgt, mask, sw)]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda" and args.amp != "off"):
                if diff is None:
                    u = model(cond, coarse)
                    # The loss (in particular the expm1 in mm space) is taken in
                    # fp32: bf16 has ~3 decimal digits, which is not enough for
                    # an exponential and its gradient.
                    loss = masked_loss(u.float(), tgt, mask, data.norm, args.loss, mm_scale, args.w_log,
                                       args.tau, args.w_quantile, args.heavy_mm, args.heavy_weight,
                                       sw if args.importance_weight else None)
                else:
                    with torch.no_grad():
                        mu = mean_model(cond, coarse).float()
                    loss = diff.loss(net, cond, coarse, mu, tgt, mask, factor)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            step += 1

            if step % 200 == 0:
                el = time.time() - t_start
                print(f"[{step:>6}/{args.steps}] loss {float(loss):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{step/el:.1f} it/s", flush=True)
            if step % args.eval_every == 0 or step == args.steps or stop:
                tgt_model = ema.shadow if ema else model
                if diff is None:
                    m = eval_deterministic(tgt_model, data, dev_dl, device, amp_dtype)
                    score = m["rmse"]
                else:
                    m = eval_diffusion_loss(tgt_model, diff, mean_model, data, dev_dl, device, amp_dtype, factor)
                    score = m["loss"]
                history.append({"step": step, **m})
                print(f"[dev {step}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
                # Selecting on RMSE alone can never keep an extremes-favouring
                # model: heavy-event weighting improves POD at later steps while
                # RMSE drifts up, so the early checkpoint always wins. Under
                # 'rmse_pod' a checkpoint must first clear the POD floor.
                eligible = (diff is not None or args.select == "rmse"
                            or m.get("pod30", 0.0) >= args.pod_floor)
                if eligible and not best_eligible_seen and diff is None and args.select == "rmse_pod":
                    best, best_eligible_seen = float("inf"), True   # discard RMSE-only leaders
                if eligible and score < best:
                    best = score
                    ckpt = {
                        "model": args.model,
                        "state_dict": {k: v.cpu() for k, v in tgt_model.state_dict().items()},
                        "args": vars(args),
                        "norm": data.norm.to_dict(),
                        "n_cond": data.n_cond,
                        "factor": factor,
                        "dev_score": score,
                        "dev_metrics": m,
                        "step": step,
                    }
                    if diff is not None:
                        ckpt["res_scale"] = diff.res_scale
                        ckpt["parameterization"] = diff.parameterization
                        ckpt["min_snr_gamma"] = diff.min_snr_gamma
                        ckpt["mean_ckpt"] = args.mean_ckpt
                    torch.save(ckpt, out / f"{name_tag}_best.pt")
                    print(f"        saved {name_tag}_best.pt (score {score:.4f})", flush=True)
        epoch += 1

    if stop and (not history or history[-1]["step"] != step):
        tgt_model = ema.shadow if ema else model
        m = (eval_deterministic(tgt_model, data, dev_dl, device, amp_dtype) if diff is None
             else eval_diffusion_loss(tgt_model, diff, mean_model, data, dev_dl, device, amp_dtype, factor))
        score = m["rmse"] if diff is None else m["loss"]
        history.append({"step": step, **m})
        print(f"[dev {step}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        if score < best:
            best = score
            ckpt = {"model": args.model, "state_dict": {k: v.cpu() for k, v in tgt_model.state_dict().items()},
                    "args": vars(args), "norm": data.norm.to_dict(), "n_cond": data.n_cond, "factor": factor,
                    "dev_score": score, "dev_metrics": m, "step": step}
            if diff is not None:
                ckpt["res_scale"] = diff.res_scale
                ckpt["parameterization"] = diff.parameterization
                ckpt["min_snr_gamma"] = diff.min_snr_gamma
                ckpt["mean_ckpt"] = args.mean_ckpt
            torch.save(ckpt, out / f"{name_tag}_best.pt")
            print(f"        saved {name_tag}_best.pt (score {score:.4f})", flush=True)

    json.dump({"history": history, "best": best, "args": vars(args), "steps_done": step,
               "minutes": (time.time() - t_start) / 60, "params": count_params(model),
             "select": args.select, "pod_floor": args.pod_floor},
              open(out / f"{name_tag}_train_log.json", "w"), indent=2)
    print(f"[done] best dev score {best:.4f} in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
