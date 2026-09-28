#!/bin/bash
# Final deep models: ERA5 conditioning + heavy-event sample weighting.
#SBATCH -J dsc-final
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
echo "=== final deterministic models $(date)"
COMMON=(--loss mse --heavy-mm 30 --heavy-weight 3.0 --data-dir data --out models/final
        --batch 32 --patch 96 --eval-every 500)

echo "----- cnn (ERA5 + heavy x3)"
python3 -m src.deep.train --model cnn --base 24 --steps 8000 --lr 2e-4 --max-minutes 12 "${COMMON[@]}"
python3 -m src.deep.infer --ckpt models/final/cnn_best.pt --data-dir data --out-dir results/final --split test

echo "----- swin (ERA5 + heavy x3)"
python3 -m src.deep.train --model swin --steps 8000 --lr 1.5e-4 --max-minutes 30 "${COMMON[@]}"
python3 -m src.deep.infer --ckpt models/final/swin_best.pt --data-dir data --out-dir results/final --split test
echo "=== done $(date)"
