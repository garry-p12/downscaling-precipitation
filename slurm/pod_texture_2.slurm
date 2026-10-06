#!/bin/bash
# Pass 2: every diffusion member on the test split, so a probability-matched
# mean is possible. infer.py previously kept only member 0, which is enough to
# show texture but not to pool an intensity distribution.
#SBATCH -J pod-tex-2
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
mkdir -p logs results/diff-members
ETA=$(python3 -c "import json;print(json.load(open('models/v3/calib.json'))['best_eta'])" 2>/dev/null || echo 0.0)
echo "=== diffusion, all members, eta=$ETA  $(date)"
python3 -m src.deep.infer --ckpt models/v3/diffusion_best.pt --data-dir data \
    --out-dir results/diff-members --split test --members 6 --steps 40 \
    --batch 4 --eta "$ETA" --save-members
echo "=== done $(date)"
