#!/bin/bash
# Settle the POWER channel: two more seeds, and the gauge check.
#
# At 1 km the channel gave -0.066 mm/day, matching the -0.067 measured at the
# coarse stage, with all three seeds improving and the 10 km field moving by
# -0.070 -- the right mechanism. But seed 2 improved by only 0.005, the ranges
# overlapped, and a paired t on three seeds gives |t| = 2.15 against the 4.30
# that two degrees of freedom demand. Five seeds should settle it.
#
# The gauge check is the one that matters more. POWER's PRECTOTCORR is
# bias-corrected against CPC gauges, and AORC -- our reference -- assimilates
# gauge data too. So part of the gain measured against AORC may be information
# the reference already carries rather than skill POWER adds. GHCN-D stations
# are measurements that do not come from either construction, so they can tell
# the two apart. If the gain vanishes against gauges, it is an artefact of the
# reference and the channel should not ship.
#SBATCH -J fuseconf
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:45:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs models/fuse1km results/fuse1km
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

M1=(--model cnn --loss mse --select rmse --spectral-weight 0.01 --spectral-cut 96
    --nearest-channel --heavy-mm 30 --heavy-weight 3.0 --base 24 --steps 6000
    --eval-every 500 --max-minutes 25 --batch 8 --patch 192 --lr 2e-4
    --out models/fuse1km)

for S in 3 4; do
  run python3 -m src.deep.train --seed $S --tag "k_pow_s$S" --data-dir data_fused "${M1[@]}"
  run python3 -m src.deep.train --seed $S --tag "k_ctl_s$S" --data-dir data       "${M1[@]}"
  for V in k_pow k_ctl; do
    D=$([ $V = k_pow ] && echo data_fused || echo data)
    run python3 -m src.deep.infer --ckpt "models/fuse1km/cnn_${V}_s${S}_best.pt" \
        --data-dir "$D" --out-dir "results/fuse1km/${V}_s$S" --split test
  done
done

ARGS=""
for V in k_pow k_ctl; do for S in 0 1 2 3 4; do
  P="results/fuse1km/${V}_s$S/cnn_1km_2019_2020.nc"
  [ -f "$P" ] && ARGS="$ARGS --product ${V}_s$S=$P"
done; done
run python3 -m src.score_variants --config config_vista.yaml $ARGS \
    --out results/fuse1km/scores5.json

# Against GHCN-D: measurements from neither AORC's construction nor POWER's
# bias correction. One seed per arm is enough to see whether the gain survives.
GA=""
for V in k_pow k_ctl; do
  P="results/fuse1km/${V}_s1/cnn_1km_2019_2020.nc"
  [ -f "$P" ] && GA="$GA --product ${V}=$P"
done
run python3 -m src.gauge_events --config config_vista.yaml $GA \
    --out results/fuse1km/gauge.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
