#!/bin/bash
# Is the RMSE cost of the spectral term real, or is it seed noise?
#
# The ablation put w=0.01 at +0.016 mm dev RMSE against the control. Earlier
# seed work on this CNN measured a spread of 0.037 mm across three runs, so the
# "cost" is smaller than the noise it would have to clear. Three seeds each,
# everything else identical to slurm/ablate_spectral.slurm so the numbers drop
# straight into that table.
#SBATCH -J spec-seeds
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
mkdir -p logs models/spec-seeds
echo "=== spectral seed repeat $(date)"
for SEED in 0 1 2; do
  for W in 0.0 0.01; do
    echo "----- seed=$SEED spectral-weight=$W"
    python3 -m src.deep.train --model cnn --base 24 --loss mse --select rmse \
        --spectral-weight "$W" --spectral-cut 10 --seed "$SEED" --tag "s${SEED}_w${W}" \
        --steps 6000 --eval-every 500 --max-minutes 18 \
        --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/spec-seeds
  done
done
echo "=== done $(date)"
