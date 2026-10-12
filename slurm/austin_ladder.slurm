#!/bin/bash
# Rungs 2 and 3 of the structural ladder, Austin.
#
# The wet-fraction and short-tail failures are one failure: a per-cell loss is
# minimised by spreading a little water everywhere instead of committing a lot
# somewhere. So one operation addresses both -- decide which cells are dry,
# zero them, give their water to the cells that remain. Mass is preserved, the
# wet fraction becomes correct by construction, and the tail lengthens as a
# consequence.
#
# The decision needs a per-cell probability of rain. The Bernoulli-gamma head
# supplies one; a deterministic field can only offer its own value as a proxy,
# which cannot tell "a little rain" from "probably none". Both are run so the
# difference is measured rather than assumed.
#
# The cut is calibrated on 2018 and applied unchanged to 2019-2020.
#SBATCH -J austin-ladder
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs models/ladder results/ladder
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# --select rmse: the default rmse_pod floor is tuned to the deterministic
# models and would reject this head on a criterion it is not optimising.
COMMON=(--model cnn_bg --base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 32 --patch 96 --lr 2e-4 --select rmse --out models/ladder)

for S in 0 1 2; do
  run python3 -m src.deep.train --seed $S --tag "bg_s$S" --data-dir data "${COMMON[@]}"
done
for S in 0 1 2; do
  for SP in dev test; do
    run python3 -m src.deep.infer --ckpt models/ladder/cnn_bg_bg_s${S}_best.pt \
        --data-dir data --out-dir results/ladder/bg_s${S}_${SP} --split $SP --prob-thresh 0.1
  done
done

# Concentrate, calibrated on dev. The bg arms rank cells by P(rain); XGBoost
# can only rank by its own value, which is the comparison that prices the
# probability.
for S in 0 1 2; do
  run python3 -m src.occurrence --config config_vista.yaml \
      --pred results/ladder/bg_s${S}_test/cnn_bg_1km_2019_2020.nc \
      --dev-pred results/ladder/bg_s${S}_dev/cnn_bg_1km_2018_2018.nc \
      --out results/ladder/occ_bg_s$S
done
run python3 -m src.occurrence --config config_vista.yaml \
    --pred results/xgboost_1km_2019_2020.nc \
    --dev-pred results/downscaled_1km_2018.nc \
    --out results/ladder/occ_xgb

ARGS=""
for S in 0 1 2; do
  ARGS="$ARGS --product bg_s$S=results/ladder/bg_s${S}_test/cnn_bg_1km_2019_2020.nc"
  ARGS="$ARGS --product bg_conc_s$S=results/ladder/occ_bg_s$S/occ_1km_2019_2020.nc"
done
ARGS="$ARGS --product xgb_conc=results/ladder/occ_xgb/occ_1km_2019_2020.nc"
run python3 -m src.score_variants --config config_vista.yaml \
    --control XGBoost=results/xgboost_1km_2019_2020.nc $ARGS \
    --out results/ladder/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
