#!/bin/bash
# Second dev-year ablation: does flip augmentation + the climatological percentile
# channel + lower capacity let the CNN beat bilinear (dev-2018 RMSE 5.152)?
#SBATCH -J dsc-ablate2
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
echo "=== capacity/augmentation ablation $(date)"

run () {  # name, extra args...
  local tag=$1; shift
  echo "----- $tag"
  python3 -m src.deep.train --model cnn --loss mse --tag "$tag" --steps 8000 --eval-every 1000 \
      --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/ablate2 "$@"
}

run base64_aug            --base 64
run base24_aug            --base 24
run base24_aug_uniform    --base 24 --wet-bias 0.0
run base24_noaug          --base 24 --no-augment
echo "=== done $(date)"
