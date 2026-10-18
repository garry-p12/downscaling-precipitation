#!/bin/bash
# Does a second, independent precipitation estimate beat a better model?
#
# The error budget leaves one lever. 91-92 % of the error is in the coarse
# field; the sharpener cannot reach it, and 5.1-at-10km showed architecture
# there is worth 0.1 mm/day at most. What is untested is giving the coarse
# stage better INFORMATION.
#
# IMERG and POWER are independent estimates of the same quantity, wrong in
# different ways: IMERG is a satellite retrieval, passive microwave where a
# sensor passed and infrared infill elsewhere; POWER's PRECTOTCORR is the
# MERRA-2 reanalysis bias-corrected against CPC gauges. src.fuse_power puts
# POWER on the IMERG grid as one more conditioning channel -- no extra
# resolution, since 0.5 deg interpolated to 0.1 is smooth across each 5x5
# block, only better values.
#
# Identical to the coarse1 arms in every other respect, so the channel is the
# only difference. The numbers to beat, on test at 10 km:
#
#     raw IMERG                4.730
#     XGBoost stage 1          4.397
#     c_cnn64 (best so far)    4.380
#     M1 implicit coarse       4.251   <- end-to-end, the real target
#
# Beating 4.251 would mean a dedicated coarse stage with two inputs finally
# overtakes what end-to-end training produces as a by-product.
#
# One caveat that cannot be designed away: POWER is bias-corrected against CPC
# gauges and AORC, the reference, also assimilates gauge data, so the two share
# some information. A gain here is not quite a gain from a fully independent
# source.
#SBATCH -J fusion
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
mkdir -p logs models/fusion results/fusion
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

COMMON=(--model cnn --loss mse --select rmse --steps 6000 --eval-every 500
        --max-minutes 18 --batch 512 --patch 24 --lr 2e-4 --out models/fusion)

# Three seeds, because the effect to detect may be of the same order as seed
# noise (5.8 puts that at 0.02-0.06) and a single run could not tell them apart.
for S in 0 1 2; do
  run python3 -m src.deep.train --base 64 --seed $S --tag "f_pow_s$S" \
      --data-dir data_coarse1_fused "${COMMON[@]}"
  run python3 -m src.deep.train --base 64 --seed $S --tag "f_ctl_s$S" \
      --data-dir data_coarse1 "${COMMON[@]}"
done

for V in f_pow f_ctl; do for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt "models/fusion/cnn_${V}_s${S}_best.pt" \
      --data-dir "$([ $V = f_pow ] && echo data_coarse1_fused || echo data_coarse1)" \
      --out-dir "results/fusion/${V}_s$S" --split test
done; done

ARGS=""
for V in f_pow f_ctl; do for S in 0 1 2; do
  P="results/fusion/${V}_s$S/cnn_1km_2019_2020.nc"
  [ -f "$P" ] && ARGS="$ARGS --product ${V}_s$S=$P"
done; done
run python3 -m src.score_variants --config config_coarse1.yaml --no-coarse $ARGS \
    --out results/fusion/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
