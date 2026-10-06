#!/bin/bash
# Deep models on the POWER domain. The refinement factor is 60 here, not 12,
# and the loader reads it off the grids -- nothing is passed explicitly.
#SBATCH -J pw-deep
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A ATM23014
set -euo pipefail
export PYTHONPATH=$SCRATCH/downscaling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"; export PYTHONUNBUFFERED=1
mkdir -p logs models/power results/power
echo "=== POWER deep models $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY - <<'PYCHK'
import sys, torch
print("torch", torch.__version__, "| built for cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FATAL: torch cannot see a GPU on this node -- refusing to train on CPU")
print("device:", torch.cuda.get_device_name(0))
PYCHK
COMMON=(--loss mse --heavy-mm 30 --heavy-weight 3.0 --select rmse_pod
        --data-dir /work/11755/gurup12/dsc-power --out models/power --batch 16 --patch 96
        --eval-every 1000 --dev-days 120)
echo "----- cnn"
$PY -m src.deep.train --model cnn --base 24 --steps 40000 --lr 2e-4 --max-minutes 70 "${COMMON[@]}"
$PY -m src.deep.infer --ckpt models/power/cnn_best.pt --data-dir /work/11755/gurup12/dsc-power --out-dir results/power --split test
echo "----- swin"
$PY -m src.deep.train --model swin --steps 12500 --lr 1.5e-4 --max-minutes 150 "${COMMON[@]}"
$PY -m src.deep.infer --ckpt models/power/swin_best.pt --data-dir /work/11755/gurup12/dsc-power --out-dir results/power --split test
echo "=== done $(date)"
