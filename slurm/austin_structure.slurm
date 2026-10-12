#!/bin/bash
# Two things: the structure-correct Austin deliverable, and the next step on it.
#
# (A) E -- spectral penalty + heavy weighting -- trained on the full Austin
#     record and inferred over the test period, so the product people download
#     has approximately correct fine-scale structure rather than 2 % of it.
#     Three seeds, because a deliverable should be the median of a replicated
#     run and not one draw.
#
# (B) The mass-conserving redistributor on the SAME real input. On a perfect
#     coarse field it reached texture 1.07 with no spectral term at all, which
#     is a better mechanism than penalising smoothness after the fact: the
#     block total is fixed, so flattening is not available and the network has
#     to place the mass. This asks whether that survives a coarse field that is
#     wrong, which is the case that matters for a product.
#SBATCH -J austin-structure
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
mkdir -p logs models/austin_struct results/austin_struct
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }
COMMON=(--base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 32 --patch 96 --lr 2e-4 --select rmse --out models/austin_struct)

for S in 0 1 2; do
  run python3 -m src.deep.train --model cnn --loss mse --spectral-weight 0.01 \
      --heavy-mm 30 --heavy-weight 3.0 --seed $S --tag "E_s$S" --data-dir data "${COMMON[@]}"
done
for S in 0 1; do
  run python3 -m src.deep.train --model cnn_redist --loss mse --seed $S \
      --tag "redist_s$S" --data-dir data "${COMMON[@]}"
done

for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt models/austin_struct/cnn_E_s${S}_best.pt \
      --data-dir data --out-dir results/austin_struct/E_s$S --split test
done
for S in 0 1; do
  run python3 -m src.deep.infer --ckpt models/austin_struct/cnn_redist_redist_s${S}_best.pt \
      --data-dir data --out-dir results/austin_struct/redist_s$S --split test
done

ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product E_s$S=results/austin_struct/E_s$S/cnn_1km_2019_2020.nc"; done
for S in 0 1; do ARGS="$ARGS --product redist_s$S=results/austin_struct/redist_s$S/cnn_redist_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control XGBoost=results/xgboost_1km_2019_2020.nc $ARGS \
    --out results/austin_struct/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
