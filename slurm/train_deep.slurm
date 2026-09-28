#!/bin/bash
# Train one deep downscaler on a Vista GH200 node and run test-period inference.
#
#   sbatch -J dsc-cnn  --export=ALL,MODEL=cnn,STEPS=20000,MAXMIN=130  -t 03:00:00 slurm/train_deep.slurm
#   sbatch -J dsc-swin --export=ALL,MODEL=swin,STEPS=20000,MAXMIN=200 -t 04:00:00 slurm/train_deep.slurm
#   sbatch -J dsc-diff --dependency=afterok:<cnn-jobid> \
#          --export=ALL,MODEL=diffusion,STEPS=40000,MEAN=models/deep/cnn_best.pt,MAXMIN=110,IBATCH=4 \
#          -t 04:00:00 slurm/train_deep.slurm
#
# MAXMIN caps training wall-clock so inference always runs inside the allocation.
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"

MODEL=${MODEL:-cnn}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
PATCH=${PATCH:-96}
LR=${LR:-2e-4}
MAXMIN=${MAXMIN:-0}
MEMBERS=${MEMBERS:-8}
DDIM=${DDIM:-50}
IBATCH=${IBATCH:-2}
EXTRA=${EXTRA:-}

echo "=== $MODEL | steps $STEPS | budget ${MAXMIN}min | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TRAIN_ARGS=(--model "$MODEL" --steps "$STEPS" --batch "$BATCH" --patch "$PATCH" --lr "$LR"
            --data-dir data --out models/deep --max-minutes "$MAXMIN")
if [ "$MODEL" = "diffusion" ]; then
  TRAIN_ARGS+=(--eval-every 2000 --mean-ckpt "${MEAN:-models/deep/cnn_best.pt}")
else
  TRAIN_ARGS+=(--eval-every 1000)
fi
# shellcheck disable=SC2086
python3 -m src.deep.train "${TRAIN_ARGS[@]}" $EXTRA

INFER_ARGS=(--ckpt "models/deep/${MODEL}_best.pt" --data-dir data --out-dir results/deep --split test)
if [ "$MODEL" = "diffusion" ]; then
  INFER_ARGS+=(--members "$MEMBERS" --steps "$DDIM" --batch "$IBATCH")
fi
python3 -m src.deep.infer "${INFER_ARGS[@]}"
echo "=== done $(date)"
