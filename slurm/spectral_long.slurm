#!/bin/bash
# Neither config had converged at 6k steps -- the control's dev RMSE was still
# falling monotonically, so the ablation compared two unfinished models. Same
# two settings, five times the budget, to see whether the gap closes, holds or
# reverses once both have run out of improvement.
#SBATCH -J spec-long
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
cd "$SCRATCH/downscaling"
mkdir -p logs models/spec-long
echo "=== spectral long run $(date)"
for W in 0.0 0.01; do
  echo "----- spectral-weight=$W"
  python3 -m src.deep.train --model cnn --base 24 --loss mse --select rmse \
      --spectral-weight "$W" --spectral-cut 10 --tag "long_w$W" \
      --steps 30000 --eval-every 1000 --max-minutes 30 \
      --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/spec-long
done
echo "=== done $(date)"
