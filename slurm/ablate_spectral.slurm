#!/bin/bash
# What does fine-scale structure cost in RMSE?
#
# Every --loss option is a per-cell loss, and a per-cell loss is minimised by
# the conditional mean -- a field far smoother than real rain. This sweeps the
# weight on the spectral term against a w=0 control, all else fixed, and reads
# off the exchange rate: how much dev RMSE buys how much of the observed
# sub-10km power back.
#
# Austin first because it is the cheap domain and the one with a diffusion
# product to compare against (spectral ratio 0.84 for a single member, 0.02 for
# XGBoost). Selection is plain --select rmse so the checkpoint criterion does
# not confound the loss comparison; what moves is the objective alone.
#SBATCH -J spec-ablate
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy here is built against gcc14; the gcc13 libstdc++ lacks CXXABI_1.3.15.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
cd "$SCRATCH/downscaling"
mkdir -p logs models/ablate-spectral
echo "=== spectral ablation $(date)"
for W in 0.0 0.01 0.05 0.2 1.0; do
  echo "----- spectral-weight=$W"
  python3 -m src.deep.train --model cnn --base 24 --loss mse --select rmse \
      --spectral-weight "$W" --spectral-cut 10 --tag "s$W" \
      --steps 6000 --eval-every 500 --max-minutes 18 \
      --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/ablate-spectral
done
echo "=== done $(date)"
