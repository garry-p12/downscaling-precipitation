#!/bin/bash
# Deep models on the full 31-channel conditioning stack (ERA5 + sub-daily structure).
# Budgets are much larger than before: with ~6x the data the earlier models
# overfit by step 500, so the question is whether they now train to convergence.
#SBATCH -J dsc-deep3
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
echo "=== deep models on 2000-2020 $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON=(--loss mse --heavy-mm 30 --heavy-weight 3.0 --select rmse_pod --data-dir data
        --out models/v3 --batch 32 --patch 96 --eval-every 1000 --dev-days 120)

echo "----- cnn"
python3 -m src.deep.train --model cnn --base 24 --steps 40000 --lr 2e-4 --max-minutes 55 "${COMMON[@]}"
python3 -m src.deep.infer --ckpt models/v3/cnn_best.pt --data-dir data --out-dir results/v3 --split test

echo "----- swin (full budget this time)"
python3 -m src.deep.train --model swin --steps 40000 --lr 1.5e-4 --max-minutes 150 "${COMMON[@]}"
python3 -m src.deep.infer --ckpt models/v3/swin_best.pt --data-dir data --out-dir results/v3 --split test

echo "----- diffusion (v-parameterisation, min-SNR)"
python3 -m src.deep.train --model diffusion --param v --min-snr-gamma 5.0 --steps 60000 --lr 2e-4 \
    --max-minutes 90 --mean-ckpt models/v3/cnn_best.pt --data-dir data --out models/v3 \
    --batch 32 --patch 96 --eval-every 5000 --dev-days 120
python3 -m src.deep.calibrate --ckpt models/v3/diffusion_best.pt --data-dir data \
    --days 12 --members 4 --steps 40 --etas 0.0,1.0 --out models/v3/calib.json
ETA=$(python3 -c "import json;print(json.load(open('models/v3/calib.json'))['best_eta'])")
python3 -m src.deep.infer --ckpt models/v3/diffusion_best.pt --data-dir data --out-dir results/v3 \
    --split test --members 6 --steps 40 --batch 4 --eta "$ETA"
echo "=== done $(date)"
