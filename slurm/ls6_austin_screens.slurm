#!/bin/bash
# Three Austin screens that close open loops, on shared Stockyard data.
#
#   seeds 3-4 for the combined configuration, taking it from n=3 to n=5
#   --select rmse_spec : a spectral floor in checkpoint selection. Selection has
#       moved outcomes more than architecture anywhere in this paper; on POWER
#       the RMSE optimum arrived before the texture had built, and AMSE's
#       fine-tune peaked at step 500 before its loss had done anything.
#   --loss multiscale  : MSE on average-pooled fields at 1/3/9/27 cells, the
#       neighbourhood family, as an alternative route to displacement tolerance
#SBATCH -J ls6-screens
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A ATM23014
set -uo pipefail
PY=/work/11755/gurup12/ls6/dsc-env/bin/python
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
cd "$SCRATCH/downscaling"
mkdir -p logs models/screens results/screens
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || echo "    FAILED"; }
COMMON=(--model cnn --base 24 --steps 6000 --eval-every 500 --max-minutes 20
        --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/screens)

for S in 3 4; do
  run $PY -m src.deep.train --loss mse --select rmse --spectral-weight 0.01 \
      --heavy-mm 30 --heavy-weight 3.0 --seed $S --tag "E_s$S" "${COMMON[@]}"
  run $PY -m src.deep.train --loss mse --select rmse --seed $S --tag "ctl_s$S" "${COMMON[@]}"
done
run $PY -m src.deep.train --loss mse --select rmse_spec --spec-floor 0.8 \
    --spectral-weight 0.01 --heavy-mm 30 --heavy-weight 3.0 --seed 0 --tag specfloor "${COMMON[@]}"
run $PY -m src.deep.train --loss multiscale --select rmse --seed 0 --tag multiscale "${COMMON[@]}"
run $PY -m src.deep.train --loss multiscale --select rmse --spectral-weight 0.01 \
    --heavy-mm 30 --heavy-weight 3.0 --seed 0 --tag ms_E "${COMMON[@]}"

for V in E_s3 E_s4 ctl_s3 ctl_s4 specfloor multiscale ms_E; do
  run $PY -m src.deep.infer --ckpt "models/screens/cnn_${V}_best.pt" --data-dir data \
      --out-dir "results/screens/$V" --split test
done
ARGS=(); for V in E_s3 E_s4 ctl_s4 specfloor multiscale ms_E; do
  P="results/screens/$V/cnn_1km_2019_2020.nc"; [ -f "$P" ] && ARGS+=(--product "$V=$P"); done
echo; echo "################ AUSTIN SCREENS, TEST ################"
$PY -m src.score_variants --config config_vista.yaml \
  --control "ctl_s3=results/screens/ctl_s3/cnn_1km_2019_2020.nc" \
  "${ARGS[@]}" --n-boot 1000 --out results/screens/scores.json 2>&1 | grep -viE warning | tail -12
echo "=== done $(date)"
