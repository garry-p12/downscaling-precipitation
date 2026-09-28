#!/bin/bash
#SBATCH -J dsc-smoke
#SBATCH -o logs/smoke.%j.out
#SBATCH -e logs/smoke.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:25:00
#SBATCH -A ATM23014

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd $SCRATCH/downscaling
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 - <<'PY'
import sys, time, torch
sys.path.insert(0, ".")
from src.deep.data import DownscalingData, PatchDataset, FullFieldDataset
from src.deep.models import build_model, count_params
from src.deep.diffusion import ResidualDiffusion
from torch.utils.data import DataLoader
dev = torch.device("cuda")
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
d = DownscalingData("data")
ds = PatchDataset(d, "train", patch=96, n_per_epoch=256)
dl = DataLoader(ds, batch_size=32, num_workers=8, drop_last=True)
for name, kw in [("cnn", {}), ("swin", dict(dim=180, depths=(6,6,6,6), heads=6))]:
    m = build_model(name, d.n_cond, 12, **kw).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    t0 = time.time(); n = 0
    for cond, co, tg, mk in dl:
        cond, co, tg, mk = [x.to(dev, non_blocking=True) for x in (cond, co, tg, mk)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            u = m(cond, co); loss = (mk*(u-tg)**2).sum()/mk.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); n += 1
    torch.cuda.synchronize()
    ips = n/(time.time()-t0)
    ff = FullFieldDataset(d, "dev"); c2, co2, *_ = ff[0]
    t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(c2[None].to(dev), co2[None].to(dev))
    torch.cuda.synchronize()
    print(f"{name}: {count_params(m)/1e6:.2f}M params | {ips:.1f} it/s (bs32 96px) | "
          f"full-domain fwd {time.time()-t1:.2f}s | peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB | loss {float(loss):.4f}", flush=True)
    torch.cuda.reset_peak_memory_stats()
# diffusion step timing
mean = build_model("cnn", d.n_cond, 12).to(dev).eval()
net = build_model("diffusion", d.n_cond, 12).to(dev)
diff = ResidualDiffusion(1000, device=dev, res_scale=1.0)
opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
t0=time.time(); n=0
for cond, co, tg, mk in dl:
    cond, co, tg, mk = [x.to(dev) for x in (cond, co, tg, mk)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad(): mu = mean(cond, co).float()
        loss = diff.loss(net, cond, co, mu, tg, mk, 12)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); n+=1
torch.cuda.synchronize()
print(f"diffusion: {count_params(net)/1e6:.2f}M params | {n/(time.time()-t0):.1f} it/s | peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
ff = FullFieldDataset(d, "dev"); c2, co2, *_ = ff[0]
c2, co2 = c2[None].to(dev), co2[None].to(dev)
t0=time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    mu = mean(c2, co2).float()
    r = diff.sample(net, c2, co2, mu, 12, steps=50)
torch.cuda.synchronize()
print(f"diffusion sampling: 1 member, 50 DDIM steps, full domain: {time.time()-t0:.1f}s", flush=True)
PY
