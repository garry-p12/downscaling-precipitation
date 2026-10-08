#!/bin/bash
# Can a sharpener that can ONLY place rain beat one that can also move mass?
#
# The reference is the oracle CNN at 0.723, which is free to adjust the block
# total as well as the detail. The redistribution head is pinned to the input's
# block mean, so its whole capacity goes to placement. Three outcomes and all
# are informative:
#
#   below 0.723  the constraint helps -- the unconstrained model was spending
#                capacity undoing a block total that was already correct
#   near 0.723   placement is what both are really doing, and the constraint
#                costs nothing while making the error budget exact
#   above 0.723  the unconstrained model was using mass adjustment to cover
#                placement it could not get right, which is worth knowing
#
# Scored against the oracle baselines already measured: nearest 1.333,
# bilinear 1.065, CNN 0.723. An untrained redistributor is exactly nearest,
# so the floor of its own learning curve is known.
#SBATCH -J redist-oracle
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
mkdir -p logs models/redist results/redist
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# patch 96 = 8 whole blocks at factor 12. The head rejects anything else, since
# a partial block would let edge cells escape the mass constraint.
COMMON=(--base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 32 --patch 96 --lr 2e-4 --select rmse --out models/redist)

for S in 0 1; do
  run python3 -m src.deep.train --model cnn_redist --loss mse --seed $S \
      --tag "redist_s$S" --data-dir data_oracle_12 "${COMMON[@]}"
done

for S in 0 1; do
  run python3 -m src.deep.infer --ckpt models/redist/cnn_redist_redist_s${S}_best.pt \
      --data-dir data_oracle_12 --out-dir results/redist/s$S --split test
done

ARGS=""
for S in 0 1; do ARGS="$ARGS --product redist_s$S=results/redist/s$S/cnn_redist_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control oracle_CNN=results/oracle/orc12_ctl/cnn_1km_2019_2020.nc \
    --product oracle_bilinear=results/oracle/oracle_bilinear_1km_2019_2020.nc \
    $ARGS --out results/redist/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed
# -- two of them finished in 46 seconds and were reported as successes.
exit $rc
