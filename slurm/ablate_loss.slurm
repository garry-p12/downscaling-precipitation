#!/bin/bash
# Pick the deterministic training objective on the 2018 dev year (never the test years).
#SBATCH -J dsc-ablate
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
echo "=== loss ablation $(date)"
for LOSS in logmse mse hybrid; do
  echo "----- loss=$LOSS"
  python3 -m src.deep.train --model cnn --loss "$LOSS" --tag "$LOSS" --steps 6000 --eval-every 1000 \
      --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/ablate
done
echo "=== done $(date)"
