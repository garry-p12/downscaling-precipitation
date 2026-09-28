#!/usr/bin/env bash
# Switch the final deep models to the heavy-event loss (see slurm/final_heavy.slurm).
#   slurm/submit_heavy.sh [partition]
#
# 1. archives the MSE-trained models/results to models/deep_mse, results/deep_mse
#    (once; a rerun never overwrites the archive),
# 2. promotes the dev-selected heavy_x3 CNN from the ablation to models/deep/cnn_best.pt,
# 3. submits the Swin retrain (same loss) and the CNN-inference + diffusion job,
#    which are independent of each other.
set -euo pipefail
PART=${1:-gh-dev}
HEAVY=(--loss mse --heavy-mm 30 --heavy-weight 3.0)
cd "$SCRATCH/downscaling"

SRC=models/ablate3/cnn_heavy_x3_best.pt
[ -f "$SRC" ] || { echo "missing $SRC - run slurm/ablate_extremes.slurm first" >&2; exit 1; }

if [ ! -d models/deep_mse ]; then
  mkdir -p models/deep_mse results/deep_mse
  cp -p models/deep/*.pt models/deep/*.json models/deep_mse/
  cp -p results/deep/* results/deep_mse/
  echo "archived MSE models -> models/deep_mse, results/deep_mse"
fi
cp -p "$SRC" models/deep/cnn_best.pt
cp -p models/ablate3/cnn_heavy_x3_train_log.json models/deep/cnn_train_log.json
echo "promoted $SRC -> models/deep/cnn_best.pt"

sub() { sbatch --parsable "$@" | tail -1; }

# Swin reached only ~1900 steps in 25 min before, so the schedule is sized to the
# budget (the cosine decay then actually completes) and evaluated often enough to
# catch an early optimum like the CNN's.
SWIN=$(sub -J dsc-swin -p "$PART" -t 00:55:00 \
       --export=ALL,MODEL=swin,STEPS=2000,LR=1.5e-4,MAXMIN=35,EXTRA="${HEAVY[*]} --eval-every 250" \
       slurm/train_deep.slurm)
echo "swin              $SWIN"

HEAVYJOB=$(sub -p "$PART" slurm/final_heavy.slurm)
echo "cnn + diffusion   $HEAVYJOB"
squeue -u "$USER" -h -o "%.10i %.12j %.8P %.9T %R"
