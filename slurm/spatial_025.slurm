#!/bin/bash
# The spatial stage for the projections arm: 0.25 deg -> 1 km.
#
# NEX-GDDP is published at 0.25 degrees, so the projections pipeline is
#
#     NEX-GDDP 0.25 deg  ->  bias correct (src.projections)  ->  downscale to 1 km
#
# and the downscaler is the piece that does not exist yet. Every model in this
# study so far reads either IMERG at 0.1 deg or POWER at 0.5 deg.
#
# It cannot be trained on CMIP6 directly, for the reason src.projections
# documents at length: a free-running historical simulation has no day-to-day
# correspondence with observations, so there are no input/target pairs. It is
# trained instead on AORC averaged to 0.25 degrees -- the perfect-input oracle
# at factor 30 -- which gives exactly the right grid pair with real paired
# days. At inference the bias-corrected CMIP6 field takes the place of the
# averaged AORC.
#
# That substitution is the pipeline's main assumption and it is worth stating:
# the downscaler sees a field whose statistics have been matched to
# observations but whose errors are not the errors it trained on. The oracle
# number it produces is therefore a ceiling, not a forecast of projection
# skill.
#
# patch 180 = 6 whole coarse cells at factor 30; 180^2 x 9 = 291,600 cells per
# step, matching the rest of the study.
#SBATCH -J spatial025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs models/spatial025 results/spatial025
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

COMMON=(--model cnn --loss mse --select rmse --base 24 --steps 6000
        --eval-every 500 --max-minutes 22 --batch 9 --patch 180 --lr 2e-4
        --data-dir data_oracle_30 --out models/spatial025)

# Plain, and with the objective the Austin ladder settled on, so the
# projections arm inherits that work rather than starting from the control.
run python3 -m src.deep.train --tag o30_ctl "${COMMON[@]}"
run python3 -m src.deep.train --tag o30_E --spectral-weight 0.01 --spectral-cut 96 \
    --heavy-mm 30 --heavy-weight 3.0 "${COMMON[@]}"

for V in ctl E; do
  run python3 -m src.deep.infer --ckpt "models/spatial025/cnn_o30_${V}_best.pt" \
      --data-dir data_oracle_30 --out-dir "results/spatial025/$V" --split test
done

python3 - <<'PY'
import yaml
c = yaml.safe_load(open("config_power_vista.yaml"))
c["paths"]["processed"] = "/scratch/11755/gurup12/downscaling/data_oracle_30"
c["domain"]["fine_factor"] = 30
c["domain"]["coarse_res_deg"] = 0.25
yaml.safe_dump(c, open("config_oracle30.yaml", "w"), sort_keys=False)
print("wrote config_oracle30.yaml")
PY

ARGS=""
for V in ctl E; do
  P="results/spatial025/$V/cnn_1km_2019_2020.nc"
  [ -f "$P" ] && ARGS="$ARGS --product o30_$V=$P"
done
run python3 -m src.score_variants --config config_oracle30.yaml $ARGS \
    --out results/spatial025/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
