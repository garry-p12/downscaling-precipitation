#!/bin/bash
# Dev-year ablation of the remedies for extreme-event underprediction.
# Baseline on the dev year: RMSE 5.1365 (bilinear 5.152). Heavy-event POD is
# reported alongside because every one of these changes trades mean-square
# skill for tail skill -- the question is the exchange rate.
#SBATCH -J dsc-extreme
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
echo "=== extremes ablation $(date)"
COMMON=(--model cnn --base 24 --steps 6000 --eval-every 500 --batch 32 --patch 96 --lr 2e-4
        --data-dir data --out models/ablate3)

run () { local tag=$1; shift; echo "----- $tag"; python3 -m src.deep.train "${COMMON[@]}" --tag "$tag" "$@"; }

run base            --loss mse
run quantile_t90    --loss quantile --tau 0.9  --w-quantile 0.3
run quantile_t95    --loss quantile --tau 0.95 --w-quantile 0.3
run heavy_x3        --loss mse --heavy-mm 30 --heavy-weight 3.0
run importance      --loss mse --importance-weight
run uniform_crops   --loss mse --wet-bias 0.0
echo "=== done $(date)"
