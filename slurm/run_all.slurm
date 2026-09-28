#!/bin/bash
# One allocation: capacity selection for the CNN, then final training and
# test-period inference for all three deep downscalers.
#
# Ordered cheapest-first with explicit wall-clock budgets (MAXMIN) so that a run
# which is still improving gets truncated rather than costing the later models
# their inference step.
#SBATCH -J dsc-all
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
mkdir -p models/deep results/deep
echo "=== run_all $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

COMMON=(--loss mse --data-dir data --out models/deep --batch 32 --patch 96 --eval-every 500)

# ---- 1. CNN capacity selection on the dev year -----------------------------
for B in 24 64; do
  echo "----- cnn base=$B"
  python3 -m src.deep.train --model cnn --tag "b$B" --base "$B" --steps 8000 --lr 2e-4 \
      --max-minutes 8 "${COMMON[@]}"
done
BEST=$(python3 - <<'PY'
import json, glob, os
best = min(glob.glob("models/deep/cnn_b*_train_log.json"), key=lambda f: json.load(open(f))["best"])
tag = os.path.basename(best).replace("cnn_", "").replace("_train_log.json", "")
print(tag)
PY
)
echo "----- best CNN config: $BEST"
cp "models/deep/cnn_${BEST}_best.pt" models/deep/cnn_best.pt
python3 -m src.deep.infer --ckpt models/deep/cnn_best.pt --data-dir data --out-dir results/deep --split test

# ---- 2. Swin transformer ---------------------------------------------------
echo "----- swin"
python3 -m src.deep.train --model swin --steps 8000 --lr 1.5e-4 --max-minutes 25 "${COMMON[@]}"
python3 -m src.deep.infer --ckpt models/deep/swin_best.pt --data-dir data --out-dir results/deep --split test

# ---- 3. Residual diffusion (conditioned on the CNN mean) -------------------
echo "----- diffusion"
python3 -m src.deep.train --model diffusion --steps 30000 --lr 2e-4 --max-minutes 25 \
    --mean-ckpt models/deep/cnn_best.pt --data-dir data --out models/deep --batch 32 --patch 96 --eval-every 2000
python3 -m src.deep.infer --ckpt models/deep/diffusion_best.pt --data-dir data --out-dir results/deep \
    --split test --members 6 --steps 40 --batch 4

echo "=== done $(date)"
ls -la results/deep
