#!/bin/bash
# One POWER deep model. Split from the combined job so each piece is short
# enough to backfill. Usage: sbatch slurm/power_deep_one.slurm <cnn|swin>
#
# patch 180 = 3x3 coarse cells (factor 60); 96 would be 1.6 cells and is invalid.
# batch 8 on BOTH clusters: the A100 has 40 GB against the GH200's 96 GB, and
# the POWER fine grid is 840x720 against Austin's 420x360. Holding batch fixed
# keeps the two POWER models comparable with each other.
#SBATCH -J pw-deep
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A ATM23014
set -euo pipefail
MODEL=${1:?model}
export PYTHONPATH=$SCRATCH/downscaling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"; export PYTHONUNBUFFERED=1
DATA=/work/11755/gurup12/dsc-power
mkdir -p logs models/power results/power
echo "=== POWER ${MODEL} on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY - <<'PYCHK'
import sys, torch
print("torch", torch.__version__, "| built for cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FATAL: torch cannot see a GPU on this node -- refusing to train on CPU")
print("device:", torch.cuda.get_device_name(0))
PYCHK
COMMON=(--loss mse --heavy-mm 30 --heavy-weight 3.0 --select rmse_pod
        --data-dir "$DATA" --out models/power --batch 8 --patch 180
        --eval-every 1000 --dev-days 120)
if [ "$MODEL" = "cnn" ]; then
  $PY -m src.deep.train --model cnn --base 24 --steps 40000 --lr 2e-4 --max-minutes 70 "${COMMON[@]}"
else
  $PY -m src.deep.train --model swin --steps 12500 --lr 1.5e-4 --max-minutes 160 "${COMMON[@]}"
fi
$PY -m src.deep.infer --ckpt "models/power/${MODEL}_best.pt" --data-dir "$DATA" \
    --out-dir results/power --split test
echo "=== done $(date)"
