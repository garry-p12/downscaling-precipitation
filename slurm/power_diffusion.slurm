#!/bin/bash
# Residual diffusion on the POWER domain -- the one product built to keep
# fine-scale texture rather than minimise RMSE. Mirrors slurm/final_diffusion.slurm
# (x0 vs v, both min-SNR weighted, picked on sampled-spread calibration) with
# the POWER geometry:
#   * patch 180 = 3x3 coarse cells at factor 60; Austin's 96 would be 1.6 cells
#   * batch 8, as the POWER deterministic runs use, so memory behaves the same
#   * mean model is swin, the better of the two deterministic POWER products
#     (test RMSE 5.886 against the CNN's 6.078), so the residual is smaller
#SBATCH -J pw-diff
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy here is built against gcc14; the gcc13 libstdc++ lacks CXXABI_1.3.15.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$SCRATCH/downscaling"
DATA=/work/11755/gurup12/dsc-power
MEAN=${MEAN:-models/power/swin_best.pt}
mkdir -p logs models/power results/power

echo "=== POWER diffusion on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f "$MEAN" || { echo "FATAL: mean checkpoint $MEAN missing"; exit 1; }

for PARAM in x0 v; do
  echo "----- param=$PARAM  min-SNR gamma=5"
  python3 -m src.deep.train --model diffusion --param "$PARAM" --min-snr-gamma 5.0 --tag "$PARAM" \
      --steps 30000 --lr 2e-4 --max-minutes 45 --mean-ckpt "$MEAN" \
      --data-dir "$DATA" --out models/power --batch 8 --patch 180 --eval-every 3000
  python3 -m src.deep.calibrate --ckpt "models/power/diffusion_${PARAM}_best.pt" --data-dir "$DATA" \
      --days 10 --members 4 --steps 40 --etas 0.0,1.0 \
      --out "models/power/calib_${PARAM}.json"
done

BEST=$(python3 - <<'PY'
import json
best, score = None, 1e9
for p in ("x0", "v"):
    c = json.load(open(f"models/power/calib_{p}.json"))
    s = min(v["score"] for v in c["etas"].values())
    print(f"{p}: best score {s:.3f}, spread ratios "
          + ", ".join(f"{k}={v['spread_ratio']:.2f}" for k, v in c["etas"].items()))
    if s < score:
        best, score = p, s
print(best)
PY
)
PARAM=$(echo "$BEST" | tail -1)
echo "----- chosen parameterisation: $PARAM"
ETA=$(python3 -c "
import json;c=json.load(open('models/power/calib_${PARAM}.json'));print(c['best_eta'])")
python3 -m src.deep.infer --ckpt "models/power/diffusion_${PARAM}_best.pt" --data-dir "$DATA" \
    --out-dir results/power --split test --members 6 --steps 40 --batch 2 --eta "$ETA"
echo "=== done $(date)"
