#!/bin/bash
# Does the spectral term work where texture collapse is worst?
#
# On POWER the deterministic products carry essentially no sub-10km power
# (XGBoost 0.0006, bilinear 0.0003) and even diffusion only reaches 0.42,
# against 0.84 on Austin. A matched control/spectral pair, identical but for
# --spectral-weight, scored the published way.
#
# Both arms use --select rmse so the checkpoint criterion cannot confound the
# loss comparison, and --pod-floor is irrelevant under that criterion.
#SBATCH -J pw-spec
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A ATM23014
set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SCRATCH/downscaling"
DATA=/work/11755/gurup12/dsc-power
mkdir -p logs models/power-spec
# gh-dev caps wall time per job through its QOS, and the pair does not fit in one
# slot. Pass a single weight to run one arm; with no argument it runs both, which
# only suits a long partition.
WEIGHTS=${1:-"0.0 0.01"}
echo "=== POWER spectral, weights [$WEIGHTS], on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for W in $WEIGHTS; do
  echo "----- spectral-weight=$W"
  python3 -m src.deep.train --model cnn --base 24 --loss mse --select rmse \
      --spectral-weight "$W" --spectral-cut 10 --tag "w$W" \
      --steps 40000 --eval-every 1000 --max-minutes 60 --lr 2e-4 \
      --data-dir "$DATA" --out models/power-spec --batch 8 --patch 180 --dev-days 120
  OUT=results/power-spec-w$W
  rm -rf "$OUT"
  python3 -m src.deep.infer --ckpt "models/power-spec/cnn_w${W}_best.pt" \
      --data-dir "$DATA" --out-dir "$OUT" --split test
  python3 -m src.model_comparison --config config_power_vista.yaml --deep-dir "$OUT"
  rm -rf "results/power/comparison-spec-w$W"
  mv results/power/comparison "results/power/comparison-spec-w$W"
  echo "----- w=$W table"; cat "results/power/comparison-spec-w$W/model_comparison.csv"
done
echo "=== done $(date)"
