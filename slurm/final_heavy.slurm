#!/bin/bash
# Final CNN + residual diffusion with the heavy-event loss chosen by the extremes
# ablation (slurm/ablate_extremes.slurm, job 1016491). On the dev year the
# heavy_x3 CNN (MSE, cells >= 30 mm/day weighted x3) matched the MSE baseline's
# RMSE (5.1379 vs 5.1365) while raising POD@30mm 0.612 -> 0.633 and cutting the
# dry bias -0.29 -> -0.08 mm/day. slurm/submit_heavy.sh promotes that dev-selected
# checkpoint to models/deep/cnn_best.pt before this job runs; the diffusion model
# must be retrained because its target is the residual from that mean.
#SBATCH -J dsc-heavy
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
echo "=== final heavy-loss CNN + diffusion $(date)"
python3 -c "import torch; ck = torch.load('models/deep/cnn_best.pt', weights_only=False); \
a = ck['args']; assert a.get('heavy_mm') == 30 and a.get('heavy_weight') == 3.0, 'cnn_best.pt is not the heavy_x3 model'; \
print('cnn_best.pt: step', ck['step'], 'dev rmse', round(ck['dev_score'], 4))"

echo "----- cnn inference"
python3 -m src.deep.infer --ckpt models/deep/cnn_best.pt --data-dir data --out-dir results/deep --split test

echo "----- diffusion on the heavy_x3 mean"
python3 -m src.deep.train --model diffusion --param x0 --steps 30000 --lr 2e-4 --max-minutes 35 \
    --mean-ckpt models/deep/cnn_best.pt --data-dir data --out models/deep \
    --batch 32 --patch 96 --eval-every 2000

echo "----- calibrating sampler on the dev year"
python3 -m src.deep.calibrate --ckpt models/deep/diffusion_best.pt --data-dir data \
    --days 10 --members 4 --steps 40 --etas 0.0,0.5,1.0 --out models/deep/diffusion_calibration.json
read -r DDIM ETA < <(python3 -c "import json; c = json.load(open('models/deep/diffusion_calibration.json')); \
print(c['best_steps'], c['best_eta'])")
echo "----- chosen sampler: $DDIM DDIM steps, eta $ETA"

python3 -m src.deep.infer --ckpt models/deep/diffusion_best.pt --data-dir data --out-dir results/deep \
    --split test --members 6 --steps "$DDIM" --batch 4 --eta "$ETA"
echo "=== done $(date)"
ls -la results/deep
