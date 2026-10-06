#!/bin/bash
# Is mm the wrong space for the spectral penalty?
#
# Subich et al. (2025, ICML) report that total precipitation is the variable
# their spherical-harmonic loss handles worst, because "precipitation is often
# localized but always non-negative, and consequently its spectral decomposition
# does not really resemble the normally-distributed random values" the method
# assumes. Our network already predicts standardised log precipitation, which is
# much closer to Gaussian. This sweeps the penalty taken in that space against
# the mm-space runs of slurm/ablate_spectral.slurm -- same seed, same budget,
# same selection, so the two tables line up row for row.
#SBATCH -J spec-logspace
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
mkdir -p logs models/spec-logspace
echo "=== spectral log-space sweep $(date)"
for W in 0.003 0.01 0.03 0.1; do
  echo "----- log-space spectral-weight=$W"
  python3 -m src.deep.train --model cnn --base 24 --loss mse --select rmse \
      --spectral-weight "$W" --spectral-space log --spectral-cut 10 --tag "log$W" \
      --steps 6000 --eval-every 500 --max-minutes 18 \
      --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/spec-logspace
done
echo "=== done $(date)"
