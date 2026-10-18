#!/bin/bash
# Does architecture matter at the COARSE stage?
#
# 5.1 tested capacity and inductive bias at 1 km and found 0.05 % between a
# 1.65 M CNN, an 11.7 M CNN and a 7.9 M Swin. That result has been used
# throughout this study to justify not spending further on architecture. It
# tests the wrong stage.
#
# The error budget puts 91-92 % of every product's error in the 10 km field,
# so the coarse stage is where the headroom is -- and nobody has designed
# anything for it. Stage 1 of the tree pipeline is an XGBoost; the deep models
# correct the coarse field only implicitly, as a side effect of end-to-end
# training. Tellingly, the implicit correction is already the better one:
#
#     raw IMERG at 10 km            ~4.73
#     XGBoost stage 1                4.397
#     M1, implicit coarse field      4.251
#
# This runs the 5.1 comparison again at 10 km. The target is AORC averaged to
# the coarse grid and the input is IMERG on the same grid, so the refinement
# factor is 1 and the model is a pure residual bias corrector -- no sharpening
# anywhere in it. Patch 24 x batch 512 = 294,912 cells per step, matching the
# rest of the study.
#
# Three outcomes, all worth having:
#   families spread     architecture matters here and 5.1 does not generalise
#                       to the stage that carries the error
#   families converge   5.1's conclusion holds at both stages, and the
#                       coarse ceiling is set by information rather than model
#   all beat 4.251      a dedicated coarse model beats the end-to-end one,
#                       which would argue for splitting the pipeline again
#SBATCH -J coarse1
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
mkdir -p logs models/coarse1 results/coarse1
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

COMMON=(--loss mse --select rmse --steps 6000 --eval-every 500 --max-minutes 18
        --batch 512 --patch 24 --lr 2e-4 --data-dir data_coarse1 --out models/coarse1)

run python3 -m src.deep.train --model cnn  --base 24 --tag c_cnn24  "${COMMON[@]}"
run python3 -m src.deep.train --model cnn  --base 64 --tag c_cnn64  "${COMMON[@]}"
run python3 -m src.deep.train --model swin --dim 180 --tag c_swin   "${COMMON[@]}"
# Does the ladder's objective transfer to a stage with no sub-grid structure
# to recover? It should not: the spectral penalty acts below 10 km and this
# model's output IS 10 km. A null here is a check on the objective's scope.
run python3 -m src.deep.train --model cnn --base 24 --tag c_cnnE \
    --spectral-weight 0.01 --spectral-cut 96 --heavy-mm 30 --heavy-weight 3.0 "${COMMON[@]}"

for V in c_cnn24 c_cnn64 c_swin c_cnnE; do
  M=models/coarse1/cnn_${V}_best.pt
  [ -f "$M" ] || M=models/coarse1/swin_${V}_best.pt
  run python3 -m src.deep.infer --ckpt "$M" --data-dir data_coarse1 \
      --out-dir "results/coarse1/$V" --split test
done

python3 - <<'PY'
import yaml
c = yaml.safe_load(open("config_vista.yaml"))
c["paths"]["processed"] = "/scratch/11755/gurup12/downscaling/data_coarse1"
c["domain"]["fine_factor"] = 1
c["domain"]["coarse_res_deg"] = 0.1
yaml.safe_dump(c, open("config_coarse1.yaml", "w"), sort_keys=False)
print("wrote config_coarse1.yaml")
PY

ARGS=""
for V in c_cnn24 c_cnn64 c_swin c_cnnE; do
  for P in results/coarse1/$V/cnn_1km_2019_2020.nc results/coarse1/$V/swin_1km_2019_2020.nc; do
    [ -f "$P" ] && ARGS="$ARGS --product $V=$P"
  done
done
run python3 -m src.score_variants --config config_coarse1.yaml --no-coarse $ARGS \
    --out results/coarse1/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
