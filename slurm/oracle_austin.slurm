#!/bin/bash
# Perfect-input oracle, Austin, at two coarse resolutions.
#
# Bilinear from a perfect 0.1 deg input already scores 1.06 against the real
# satellite's 4.92, so the question a trained model answers here is narrow and
# worth answering: once the input is correct, does learning add anything at
# all, or is 1.06 simply the sub-grid variance and therefore irreducible?
#
# Factor 60 repeats it at 0.5 deg, the NASA POWER resolution, which turns the
# POWER arm's ceiling into something measured rather than inferred.
#SBATCH -J oracle-austin
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
mkdir -p logs models/oracle results/oracle
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }
COMMON=(--model cnn --base 24 --steps 6000 --eval-every 500 --max-minutes 20
        --batch 32 --patch 96 --lr 2e-4 --out models/oracle)

for F in 12 60; do
  run python3 -m src.oracle_input --config config_vista.yaml --out data_oracle_$F --factor $F
done

for F in 12 60; do
  run python3 -m src.deep.train --loss mse --select rmse --seed 0 \
      --tag "orc${F}_ctl" --data-dir data_oracle_$F "${COMMON[@]}"
  run python3 -m src.deep.train --loss mse --select rmse --spectral-weight 0.01 \
      --heavy-mm 30 --heavy-weight 3.0 --seed 0 \
      --tag "orc${F}_E" --data-dir data_oracle_$F "${COMMON[@]}"
  for V in ctl E; do
    run python3 -m src.deep.infer --ckpt models/oracle/cnn_orc${F}_${V}_best.pt \
        --data-dir data_oracle_$F --out-dir results/oracle/orc${F}_${V} --split test
  done
done

ARGS=""
for F in 12 60; do
  for V in ctl E; do
    ARGS="$ARGS --product orc${F}_${V}=results/oracle/orc${F}_${V}/cnn_1km_2019_2020.nc"
  done
done
run python3 -m src.score_variants --config config_vista.yaml $ARGS \
    --out results/oracle_austin_scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed
# -- two of them finished in 46 seconds and were reported as successes.
exit $rc
