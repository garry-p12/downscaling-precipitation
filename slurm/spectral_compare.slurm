#!/bin/bash
# Put the spectral model in the published table.
#
# The ablation's `spec` is a dev-year number; the 0.02 / 0.13 / 0.84 figures
# everyone quotes are test-set, measured by src.model_comparison. Run inference
# for the control and the w=0.01 checkpoint and score each the published way,
# so the two are finally comparable.
#
# model_comparison always writes to <results>/comparison, so each run is moved
# aside immediately and anything that was there first is restored on exit.
#SBATCH -J spec-cmp
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
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
cd "$SCRATCH/downscaling"
mkdir -p logs

BAK=
if [ -d results/comparison ]; then
  BAK=results/comparison.before.$$
  mv results/comparison "$BAK"
fi
restore() {
  # the trap's exit status becomes the script's, and under `set -e` a bare
  # test that happens to be false fails a job whose work already succeeded
  rm -rf results/comparison
  if [ -n "$BAK" ] && [ -d "$BAK" ]; then mv "$BAK" results/comparison; fi
  return 0
}
trap restore EXIT

for V in ctl w001; do
  case "$V" in
    ctl)  CKPT=models/ablate-spectral/cnn_s0.0_best.pt ;;
    w001) CKPT=models/ablate-spectral/cnn_s0.01_best.pt ;;
  esac
  echo "===== $V from $CKPT"
  test -f "$CKPT" || { echo "FATAL: $CKPT missing"; exit 1; }
  rm -rf "results/spec-$V"
  python3 -m src.deep.infer --ckpt "$CKPT" --data-dir data \
      --out-dir "results/spec-$V" --split test
  python3 -m src.model_comparison --config config_vista.yaml --deep-dir "results/spec-$V"
  rm -rf "results/comparison-spec-$V"
  mv results/comparison "results/comparison-spec-$V"
  echo "----- $V table"
  cat "results/comparison-spec-$V/model_comparison.csv"
done
echo "=== done $(date)"
