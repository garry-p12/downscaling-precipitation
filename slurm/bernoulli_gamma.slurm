#!/bin/bash
# Bernoulli-gamma head: both domains, three seeds each.
#
# The claim being tested is not "better RMSE" -- the mean of a distribution fit
# has no reason to beat a model trained directly on squared error. It is that
# detection stops being a property of the weights. Two things decide it:
#
#   1. Does the mean stay competitive? If p*mu costs more than ~0.05 RMSE the
#      head is not a free addition to the menu.
#   2. Does calibration deliver frequency bias 1.0 on held-out data? It is 1.0
#      by construction on the data the cut is fitted to; the question is whether
#      a cut fitted on dev transfers to test.
#
# Colorado matters more than Austin here. Its control sits at a frequency bias
# of 0.15 even at a matched threshold -- a fivefold under-forecast that no loss
# weighting in this study has moved. If calibration fixes that, it is the first
# thing that has.
#SBATCH -J bg-head
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -uo pipefail
PY=/work/11755/gurup12/ls6/dsc-env/bin/python
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
cd /scratch/11755/gurup12/downscaling
mkdir -p logs models/bg results/bg
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# --select rmse, deliberately. The default rmse_pod floor is tuned to Austin's
# deterministic models and would reject this head's checkpoints on a criterion
# it is not trying to optimise.
COMMON=(--model cnn_bg --base 24 --steps 6000 --eval-every 500 --max-minutes 20
        --batch 32 --patch 96 --lr 2e-4 --select rmse --out models/bg)

for S in 0 1 2; do
  run $PY -m src.deep.train --seed $S --tag "au_bg_s$S" --data-dir data "${COMMON[@]}"
done
for S in 0 1 2; do
  run $PY -m src.deep.train --seed $S --tag "co_bg_s$S" --data-dir data_co "${COMMON[@]}"
done

for S in 0 1 2; do
  run $PY -m src.deep.infer --ckpt models/bg/cnn_bg_au_bg_s${S}_best.pt \
      --data-dir data --out-dir results/bg/au_s$S --split test
  run $PY -m src.deep.infer --ckpt models/bg/cnn_bg_co_bg_s${S}_best.pt \
      --data-dir data_co --out-dir results/bg/co_s$S --split test
done

AU=""; CO=""
for S in 0 1 2; do
  AU="$AU --product au_bg_s$S=results/bg/au_s$S/cnn_bg_1km_2019_2020.nc"
  CO="$CO --product co_bg_s$S=results/bg/co_s$S/cnn_bg_1km_2019_2020.nc"
done
run $PY -m src.score_variants --config config_vista.yaml $AU \
    --out results/bg_austin_scores.json
run $PY -m src.score_variants --config config_colorado_ls6.yaml $CO \
    --match-exceedance 2.023308e-02 --out results/bg_colorado_scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
