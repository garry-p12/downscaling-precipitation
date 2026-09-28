#!/usr/bin/env bash
# Submit the three final training runs, chained so the diffusion model gets its
# deterministic mean from the finished CNN.
#   slurm/submit_final.sh <loss> [partition]
set -euo pipefail
LOSS=${1:-mse}
PART=${2:-gh-dev}
cd "$SCRATCH/downscaling"

sub() { sbatch --parsable "$@" 2>/dev/null | tail -1; }

CNN=$(sub -J dsc-cnn -p "$PART" -t 01:00:00 \
      --export=ALL,MODEL=cnn,STEPS=20000,MAXMIN=40,EXTRA="--loss $LOSS" slurm/train_deep.slurm)
echo "cnn        $CNN"

SWIN=$(sub -J dsc-swin -p "$PART" -t 02:00:00 \
       --export=ALL,MODEL=swin,STEPS=20000,LR=1.5e-4,MAXMIN=85,EXTRA="--loss $LOSS" slurm/train_deep.slurm)
echo "swin       $SWIN"

DIFF=$(sub -J dsc-diff -p "$PART" -t 02:00:00 --dependency=afterok:"$CNN" \
       --export=ALL,MODEL=diffusion,STEPS=40000,MEAN=models/deep/cnn_best.pt,MEMBERS=8,DDIM=50,MAXMIN=50,IBATCH=4 \
       slurm/train_deep.slurm)
echo "diffusion  $DIFF  (after $CNN)"
squeue -u "$USER" -h -o "%.10i %.10j %.8P %.9T %R"
