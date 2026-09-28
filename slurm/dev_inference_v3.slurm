#!/bin/bash
# Dev-year (2018) predictions for the deep models, needed to fit stacking
# weights without touching the test years.
#SBATCH -J dsc-devinf3
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
echo "=== dev-year inference $(date)"
for M in cnn swin; do
  python3 -m src.deep.infer --ckpt "models/v3/${M}_best.pt" --data-dir data \
      --out-dir results/v3_dev --split dev
done
ETA=$(python3 -c "import json;print(json.load(open('models/v3/calib.json'))['best_eta'])")
python3 -m src.deep.infer --ckpt models/v3/diffusion_best.pt --data-dir data \
    --out-dir results/v3_dev --split dev --members 6 --steps 40 --batch 4 --eta "$ETA"
echo "=== done $(date)"
