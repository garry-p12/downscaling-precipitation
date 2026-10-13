#!/bin/bash
# Rescore the three E seeds with band energy, so the ladder compares 3 v 3.
#
# results/austin_struct/scores.json predates the band-energy metric, so the
# only E figure carrying per-band numbers is the shipped seed. Comparing three
# S1 seeds against one E seed would read a seed difference as an arm
# difference; E's own texture ranges 0.865 to 1.020 across seeds, which is
# wider than the effect being measured.
#SBATCH -J e-bands
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product E_s$S=results/austin_struct/E_s$S/cnn_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control XGBoost=results/xgboost_1km_2019_2020.nc \
    $ARGS --out results/austin_struct/scores_bands.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
