#!/bin/bash
# One architecture, one seed. Split out of the 12-hour monolith so each piece
# is short enough for Slurm's backfill scheduler to slot into a gap.
# Usage: sbatch slurm/seed_one.slurm <cnn|swin> <seed>
#SBATCH -J dsc-seed
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A ATM23014

set -euo pipefail
MODEL=${1:?model}
SEED=${2:?seed}
# Match the published run's STEP budget, not its minutes: seed 0's swin was
# wall-clock limited at 12,494 steps, so matching time across different GPUs
# would confound hardware with seed.
STEPS=${3:-}
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
OUT=models/seed${SEED}; RES=results/seed${SEED}
mkdir -p "$OUT" "$RES" logs
echo "=== ${MODEL} seed ${SEED} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

COMMON=(--loss mse --heavy-mm 30 --heavy-weight 3.0 --select rmse_pod --data-dir data
        --batch 32 --patch 96 --eval-every 1000 --dev-days 120 --seed "$SEED" --out "$OUT")

if [ "$MODEL" = "cnn" ]; then
  python3 -m src.deep.train --model cnn --base 24 --steps "${STEPS:-40000}" --lr 2e-4 \
      --max-minutes 90 "${COMMON[@]}"
else
  python3 -m src.deep.train --model swin --steps "${STEPS:-12500}" --lr 1.5e-4 \
      --max-minutes 220 "${COMMON[@]}"
fi
python3 -m src.deep.infer --ckpt "$OUT/${MODEL}_best.pt" --data-dir data --out-dir "$RES" --split test
echo "=== done $(date)"
