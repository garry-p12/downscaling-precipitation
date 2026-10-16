#!/bin/bash
# The two-stage redistribution test, specified correctly this time.
#
# slurm/redistribute_oracle.slurm asked whether a sharpener that can only
# place mass beats one that can also move it, and the Austin run of it fed the
# head RAW IMERG. That tested nothing anyone wanted to know: the head is
# pinned to its input's block mean, so handing it an uncorrected coarse field
# guarantees it inherits every bit of the retrieval error. It scored 4.980
# against the control's 4.52 and the coarse share rose to 0.932, which is the
# arithmetic of the setup rather than a fact about redistribution.
#
# The question was always: given a coarse field that has ALREADY been
# corrected, is mass-conserving placement better than free sharpening? Stage 1
# of the XGBoost pipeline produces exactly such a field, and block-averaging
# its 1 km output recovers it -- that is the same quantity `rmse_10km`
# reports, 4.397 against the raw input's larger error.
#
# So: build a data directory whose coarse field is XGBoost's corrected 10 km
# field across the whole record, and train the redistribution head on it.
# Three outcomes, all informative:
#
#   beats 4.583   placement beats XGBoost's own sharpening, and the two-stage
#                 split is the better architecture
#   near 4.583    the sharpening stage is doing placement and nothing else,
#                 which makes the error budget exact and costs nothing
#   worse         free sharpening is using mass adjustment to cover placement
#                 it cannot do, which is worth knowing before we recommend
#                 mass conservation to anyone
#
# An untrained redistributor is exactly block replication, so the floor of its
# own learning curve is the blocky field and is known in advance.
#SBATCH -J redist-2stage
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
export OMP_NUM_THREADS=8 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs models/redist2 results/redist2
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# Stage-1 field for the whole record, not just the test years: the head is
# trained on 2000-2017 and must see the same kind of input there.
run python3 -m src.oracle_input --config config_vista.yaml \
    --source 'results/downscaled_1km_*.nc' --out data_xgb10

# patch 96 = 8 whole blocks at factor 12. The head rejects anything else,
# since a partial block would let edge cells escape the mass constraint.
COMMON=(--base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 32 --patch 96 --lr 2e-4 --select rmse --out models/redist2)

for S in 0 1 2; do
  run python3 -m src.deep.train --model cnn_redist --loss mse --seed $S \
      --tag "r2_s$S" --data-dir data_xgb10 "${COMMON[@]}"
done
for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt models/redist2/cnn_redist_r2_s${S}_best.pt \
      --data-dir data_xgb10 --out-dir results/redist2/s$S --split test
done

ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product r2_s$S=results/redist2/s$S/cnn_redist_1km_2019_2020.nc"; done
[ -f results/xgboost_blocky_1km_2019_2020.nc ] && \
  ARGS="$ARGS --product xgb_blocky=results/xgboost_blocky_1km_2019_2020.nc"
run python3 -m src.score_variants --config config_vista.yaml \
    --control XGBoost=results/xgboost_1km_2019_2020.nc \
    $ARGS --out results/redist2/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
