#!/bin/bash
# Retrain the residual diffusion model with x0-parameterisation and a properly
# estimated residual scale, calibrate the sampler on the dev year, then run
# test-period inference with the chosen noise level.
#SBATCH -J dsc-diff2
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
echo "=== diffusion rerun (x0) $(date)"

python3 -m src.deep.train --model diffusion --param x0 --steps 30000 --lr 2e-4 --max-minutes 35 \
    --mean-ckpt models/deep/cnn_best.pt --data-dir data --out models/deep \
    --batch 32 --patch 96 --eval-every 2000

echo "----- calibrating sampler on the dev year"
python3 -m src.deep.calibrate --ckpt models/deep/diffusion_best.pt --data-dir data \
    --days 10 --members 4 --steps 40 --etas 0.0,0.5,1.0 --out models/deep/diffusion_calibration.json
ETA=$(python3 -c "import json;print(json.load(open('models/deep/diffusion_calibration.json'))['best_eta'])")
echo "----- chosen eta: $ETA"

python3 -m src.deep.infer --ckpt models/deep/diffusion_best.pt --data-dir data --out-dir results/deep \
    --split test --members 6 --steps 40 --batch 4 --eta "$ETA"
echo "=== done $(date)"
