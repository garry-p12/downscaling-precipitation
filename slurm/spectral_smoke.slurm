#!/bin/bash
# Three hundred steps on the Austin data, only to prove the spectral term runs
# on real fields, reports `spec` on the dev line and does not blow up the loss.
# Not an experiment -- see slurm/ablate_spectral.slurm for that.
#SBATCH -J spec-smoke
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:25:00
#SBATCH -A ATM23014
set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
cd "$SCRATCH/downscaling"
mkdir -p logs models/spec-smoke
for W in 0.0 0.05; do
  echo "===== spectral-weight $W ====="
  python3 -m src.deep.train --model cnn --base 24 --steps 300 --eval-every 150 \
      --max-minutes 6 --loss mse --select rmse --spectral-weight "$W" --tag "w$W" \
      --data-dir data --out models/spec-smoke --batch 16 --patch 96 --dev-days 40
done
echo "=== done $(date)"
