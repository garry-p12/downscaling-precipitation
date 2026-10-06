#!/bin/bash
# The combined configuration on POWER. Texture collapse is worst here --
# deterministic products retain 0.0003 to 0.284 of observed sub-10 km variance,
# and even a diffusion member only reaches 0.419 against Austin's 0.840.
# POWER data sits on shared Stockyard, so this runs on Lonestar6 while Vista
# takes Colorado.
#SBATCH -J ls6-pw-E
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
mkdir -p logs models/pw-E results/pw-E
DATA=/work/11755/gurup12/dsc-power
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || echo "    FAILED"; }
COMMON=(--model cnn --base 24 --select rmse --steps 40000 --eval-every 1000
        --max-minutes 55 --batch 8 --patch 180 --lr 2e-4 --dev-days 120
        --data-dir "$DATA" --out models/pw-E)

run $PY -m src.deep.train --loss mse --seed 0 --tag ctl "${COMMON[@]}"
run $PY -m src.deep.train --loss mse --spectral-weight 0.01 --heavy-mm 30 \
    --heavy-weight 3.0 --seed 0 --tag E "${COMMON[@]}"
for V in ctl E; do
  run $PY -m src.deep.infer --ckpt "models/pw-E/cnn_${V}_best.pt" --data-dir "$DATA" \
      --out-dir "results/pw-E/$V" --split test
done
echo; echo "################ POWER, TEST ################"
$PY -m src.score_variants --config config_power_vista.yaml \
  --control "ctl=results/pw-E/ctl/cnn_1km_2019_2020.nc" \
  --product "E=results/pw-E/E/cnn_1km_2019_2020.nc" \
  --n-boot 1000 --out results/pw-E/scores.json 2>&1 | grep -viE warning | tail -8
echo "=== done $(date)"
