#!/bin/bash
# Perfect-input oracle, Colorado Front Range.
#
# Austin's oracle says 78 % of bilinear's error is the satellite being wrong
# rather than the grid being coarse. Terrain should move that number: orographic
# rain is organised by a fixed boundary condition, so more of the 1 km detail
# ought to be recoverable from a correct 10 km field, and the irreducible
# sub-grid variance ought to be smaller than Austin's 1.06.
#
# If instead the Colorado ceiling is no lower, the paper's central claim needs
# rewording: the terrain advantage would be about predictability given a *good*
# input, not about the satellite being less wrong over mountains.
#SBATCH -J oracle-colorado
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
mkdir -p logs models/oracle results/oracle
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }
COMMON=(--model cnn --base 24 --steps 6000 --eval-every 500 --max-minutes 20
        --batch 32 --patch 96 --lr 2e-4 --out models/oracle)

run $PY -m src.oracle_input --config config_colorado_ls6.yaml --out data_co_oracle_12 --factor 12

run $PY -m src.deep.train --loss mse --select rmse --seed 0 \
    --tag co_orc_ctl --data-dir data_co_oracle_12 "${COMMON[@]}"
run $PY -m src.deep.train --loss mse --select rmse --spectral-weight 0.01 \
    --heavy-mm 30 --heavy-weight 3.0 --seed 0 \
    --tag co_orc_E --data-dir data_co_oracle_12 "${COMMON[@]}"

for V in ctl E; do
  run $PY -m src.deep.infer --ckpt models/oracle/cnn_co_orc_${V}_best.pt \
      --data-dir data_co_oracle_12 --out-dir results/oracle/co_orc_$V --split test
done

# Detection at the matched threshold, not 30 mm -- over the Front Range 30 mm is
# 6.7x rarer than over Austin and the detection metrics stop describing the model.
run $PY -m src.score_variants --config config_colorado_ls6.yaml \
    --control co_orc_ctl=results/oracle/co_orc_ctl/cnn_1km_2019_2020.nc \
    --product co_orc_E=results/oracle/co_orc_E/cnn_1km_2019_2020.nc \
    --match-exceedance 2.023308e-02 --out results/oracle_colorado_scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
