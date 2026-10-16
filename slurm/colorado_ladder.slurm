#!/bin/bash
# Does the Austin ladder replicate in terrain?
#
# Eleven arms were run on Austin and none on Colorado, so every claim the
# ladder makes is currently a one-domain claim. Colorado already has the
# anchor: three paired seeds of the control and of E (patch 96, cut 10,
# w = 0.01) in results/co-deep/scores.json, where E gave RMSE 2.938 against
# 2.978 and texture 0.351 -> 1.160.
#
# Two arms test the two things that actually moved Austin, against that
# anchor:
#
#   CO-M1   patch 192, cut 96, + unsmoothed-coarse channel, w = 0.01
#           On Austin this stack was worth -0.107 RMSE over E (4.550 ->
#           4.443) and took sub-10 km energy 0.399 -> 0.637.
#
#   CO-W10  the same at w = 0.10
#           On Austin raising the weight took 16-32 km from 0.60x bilinear
#           to 1.19x for +0.115 RMSE, the only lever that bought mesoscale
#           without inflating 2-4 km.
#
# Colorado is the harder case for both. Its precipitation is orographically
# forced in the cool season, so the coarse field already carries much of the
# spatial pattern and there may be less for a sharpener to add; against that,
# E's texture gain replicated here at a larger factor than on Austin (3.3x
# against 3.0x), so the objective does bite in terrain.
#
# Detection is scored at the matched exceedance depth, 14.50 mm, not 30 mm.
# At 30 mm the Colorado control sits at a frequency bias of 0.039 and POD,
# CSI and FSS all collapse into a corner that describes the threshold rather
# than the model. 2.023308e-02 is Austin's 30 mm exceedance rate.
#SBATCH -J co-ladder
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
mkdir -p logs models/co-ladder results/co-ladder
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }
CFG=config_colorado_vista.yaml

# Matched to the Austin arms: 192^2 x 8 = 294,912 fine cells per step.
COMMON=(--model cnn --loss mse --spectral-cut 96 --nearest-channel
        --heavy-mm 30 --heavy-weight 3.0 --data-dir data_co
        --base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 8 --patch 192 --lr 2e-4 --select rmse --out models/co-ladder)

for S in 0 1 2; do
  run python3 -m src.deep.train --spectral-weight 0.01 --seed $S --tag "coM1_s$S" "${COMMON[@]}"
done
for S in 0 1 2; do
  run python3 -m src.deep.train --spectral-weight 0.1 --seed $S --tag "coW10_s$S" "${COMMON[@]}"
done
for V in coM1 coW10; do
  for S in 0 1 2; do
    run python3 -m src.deep.infer --ckpt models/co-ladder/cnn_${V}_s${S}_best.pt \
        --data-dir data_co --out-dir results/co-ladder/${V}_s$S --split test
  done
done

ARGS=""
for V in coM1 coW10; do for S in 0 1 2; do
  P="results/co-ladder/${V}_s${S}/cnn_1km_2019_2020.nc"
  [ -f "$P" ] && ARGS="$ARGS --product ${V}_s$S=$P"
done; done
# Scored against Colorado's existing E anchor, paired by day.
run python3 -m src.score_variants --config $CFG \
    --control E_s0=results/co-deep/E_s0/cnn_1km_2019_2020.nc \
    --product co_E_s1=results/co-deep/E_s1/cnn_1km_2019_2020.nc \
    --product co_E_s2=results/co-deep/E_s2/cnn_1km_2019_2020.nc \
    $ARGS --match-exceedance 2.023308e-02 --out results/co-ladder/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
