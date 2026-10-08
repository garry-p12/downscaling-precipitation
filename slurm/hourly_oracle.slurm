#!/bin/bash
# Does sub-daily position tell the sharpener where inside the block it rained?
#
# Two runs differing in one thing: the hourly arm sees 24 coarse frames per day
# alongside the daily total, the daily arm sees only the total. Both have a
# perfect coarse field, so neither is doing retrieval correction and the whole
# difference is placement information.
#
# The daily arm is rerun rather than reusing orc12_ctl, which also carried ten
# ERA5 environmental channels. train.py has no flag to drop them, so the daily
# arm reads a directory with no ERA5 file at all and the hourly arm delivers
# its 24 frames through that same slot. The two therefore differ by exactly
# the sub-daily information and nothing else.
#SBATCH -J hourly-oracle
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
mkdir -p logs models/hourly results/hourly
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }
COMMON=(--model cnn --base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 32 --patch 96 --lr 2e-4 --select rmse --out models/hourly)

for S in 0 1; do
  run python3 -m src.deep.train --loss mse --seed $S --tag "daily_s$S" \
      --data-dir data_oracle_12_daily "${COMMON[@]}"
  run python3 -m src.deep.train --loss mse --seed $S --tag "hourly_s$S" \
      --data-dir data_oracle_12_hourly "${COMMON[@]}"
done

for S in 0 1; do
  for V in daily hourly; do
    DD=data_oracle_12_daily; [ "$V" = hourly ] && DD=data_oracle_12_hourly
    run python3 -m src.deep.infer --ckpt models/hourly/cnn_${V}_s${S}_best.pt \
        --data-dir $DD --out-dir results/hourly/${V}_s$S --split test
  done
done

ARGS=""
for S in 0 1; do for V in daily hourly; do
  ARGS="$ARGS --product ${V}_s$S=results/hourly/${V}_s$S/cnn_1km_2019_2020.nc"
done; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control oracle_bilinear=results/oracle/oracle_bilinear_1km_2019_2020.nc \
    $ARGS --out results/hourly/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed
# -- two of them finished in 46 seconds and were reported as successes.
exit $rc
