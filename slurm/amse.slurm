#!/bin/bash
# AMSE against the additive spectral penalty, on identical settings.
#
# The additive penalty needs a weight and lands wherever its two terms balance:
# every weight we tried, in either space, overshoots (1.15-2.27 against a target
# of 1.0). AMSE (Subich et al. 2025) restructures MSE instead of adding to it,
# so its amplitude term is minimised exactly at the right spectrum and there is
# no weight to choose. Same seed, same budget, same --select rmse as
# slurm/ablate_spectral.slurm, so the rows drop into that table.
#SBATCH -J amse
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
mkdir -p logs models/amse
echo "=== AMSE $(date)"
for SEED in 0 1 2; do
  echo "----- amse seed=$SEED"
  python3 -m src.deep.train --model cnn --base 24 --loss amse --select rmse --seed "$SEED" \
      --tag "amse_s$SEED" --steps 6000 --eval-every 500 --max-minutes 18 \
      --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/amse
done
echo "=== done $(date)"
