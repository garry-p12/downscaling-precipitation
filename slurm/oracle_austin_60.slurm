#!/bin/bash
# The factor-60 half of the Austin oracle, which failed in job 1051848.
#
# Two bugs, both fixed: oracle_input read grids.fine.lat.values on a plain
# ndarray, and derived the coarse resolution by measuring the fine grid, whose
# spacing is stored truncated (0.008333), giving 0.49998 instead of 0.5.
#
# The patch size is the third trap and the reason this is a separate script:
# a fine patch must cover a whole number of coarse cells, so 96 is valid at
# factor 12 and invalid at 60. 180 is three coarse cells wide.
#
# This gives the 0.5 deg ceiling, which the POWER arm currently reasons about
# rather than measures.
#SBATCH -J oracle-au60
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
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
        --batch 32 --patch 180 --lr 2e-4 --out models/oracle)

run python3 -m src.oracle_input --config config_vista.yaml --out data_oracle_60 --factor 60

run python3 -m src.deep.train --loss mse --select rmse --seed 0 \
    --tag orc60_ctl --data-dir data_oracle_60 "${COMMON[@]}"
run python3 -m src.deep.train --loss mse --select rmse --spectral-weight 0.01 \
    --heavy-mm 30 --heavy-weight 3.0 --seed 0 \
    --tag orc60_E --data-dir data_oracle_60 "${COMMON[@]}"

for V in ctl E; do
  run python3 -m src.deep.infer --ckpt models/oracle/cnn_orc60_${V}_best.pt \
      --data-dir data_oracle_60 --out-dir results/oracle/orc60_$V --split test
done

# Scored together with the factor-12 products that already succeeded, so the
# input-resolution ladder comes out of one file.
run python3 -m src.score_variants --config config_vista.yaml \
    --product orc12_ctl=results/oracle/orc12_ctl/cnn_1km_2019_2020.nc \
    --product orc12_E=results/oracle/orc12_E/cnn_1km_2019_2020.nc \
    --product orc60_ctl=results/oracle/orc60_ctl/cnn_1km_2019_2020.nc \
    --product orc60_E=results/oracle/orc60_E/cnn_1km_2019_2020.nc \
    --out results/oracle_austin_scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
