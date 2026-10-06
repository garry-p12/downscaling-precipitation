#!/bin/bash
# One architecture, one seed -- Lonestar6 variant.
#
# LS6 is x86_64 with A100s, so it needs its own env (the Vista miniforge is
# aarch64 and will not execute here). Data comes from shared Stockyard, which
# both systems mount, so nothing is re-downloaded.
# Usage: sbatch slurm/seed_one_ls6.slurm <cnn|swin> <seed>
#SBATCH -J dsc-seed
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A ATM23014

set -euo pipefail
MODEL=${1:?model}
SEED=${2:?seed}
# Seed 0 (the published run) was wall-clock limited, not step limited: swin
# stopped at 12,494 steps and cnn reached its full 40,000. Repeats must match
# the STEP budget, not the minutes, or hardware speed is confounded with seed.
STEPS=${3:-}
export PYTHONPATH=
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
# A100-PCIE has 40 GB against the GH200's 96 GB, and swin at batch 32 /
# patch 96 sits right at the edge. Expandable segments avoid the
# fragmentation that pushed it over -- this changes allocation only,
# never the arithmetic, so seeds stay comparable to the published run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=/work/11755/gurup12/ls6/dsc-data
OUT=models/seed${SEED}; RES=results/seed${SEED}
mkdir -p "$OUT" "$RES" logs
echo "=== ${MODEL} seed ${SEED} on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Fail loudly on CPU fallback. A pip-installed torch whose CUDA build does not
# match the node driver reports cuda=False and trains on CPU at ~0.2 it/s --
# the job still "succeeds" and writes a checkpoint that is not comparable to a
# GPU-trained run. Assert instead.
$PY - <<'PYCHK'
import sys, torch
print("torch", torch.__version__, "| built for cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FATAL: torch cannot see a GPU on this node -- refusing to train on CPU")
print("device:", torch.cuda.get_device_name(0))
PYCHK

COMMON=(--loss mse --heavy-mm 30 --heavy-weight 3.0 --select rmse_pod --data-dir "$DATA"
        --batch 32 --patch 96 --eval-every 1000 --dev-days 120 --seed "$SEED" --out "$OUT")

if [ "$MODEL" = "cnn" ]; then
  $PY -m src.deep.train --model cnn --base 24 --steps "${STEPS:-40000}" --lr 2e-4 \
      --max-minutes 90 "${COMMON[@]}"
else
  $PY -m src.deep.train --model swin --steps "${STEPS:-12500}" --lr 1.5e-4 \
      --max-minutes 200 "${COMMON[@]}"
fi
$PY -m src.deep.infer --ckpt "$OUT/${MODEL}_best.pt" --data-dir "$DATA" --out-dir "$RES" --split test
echo "=== done $(date)"
