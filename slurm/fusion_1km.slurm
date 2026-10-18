#!/bin/bash
# Does the POWER channel still pay at 1 km, in the shipped configuration?
#
# At the coarse stage it was worth -0.067 mm/day with disjoint seed ranges, and
# beat the architecture lever on the same task by 1.9x. This asks whether that
# survives in the product M1 actually is.
#
# Two reasons to expect less here. End-to-end training on the 1 km target
# already extracts coarse-scale information more effectively than a dedicated
# coarse stage does -- M1's implicit 10 km field is 4.251 against the best
# dedicated model's 4.323 even with POWER. And whatever POWER adds has to
# survive passing through a sharpener that the error budget says contributes
# almost nothing.
#
# M1's exact configuration on both arms, three seeds each, paired by seed. The
# control reproduces runs we already have (4.443 / 4.432 / 4.455), which is a
# check that nothing else has drifted.
#SBATCH -J fuse1km
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

# M1 exactly: spectral penalty over the whole resolvable spectrum, heavy-cell
# weighting, the unsmoothed-coarse channel, 192 km patch.
M1=(--model cnn --loss mse --select rmse --spectral-weight 0.01 --spectral-cut 96
    --nearest-channel --heavy-mm 30 --heavy-weight 3.0 --base 24 --steps 6000
    --eval-every 500 --max-minutes 25 --batch 8 --patch 192 --lr 2e-4
    --out models/fuse1km)

for S in 0 1 2; do
  run python3 -m src.deep.train --seed $S --tag "k_pow_s$S" --data-dir data_fused "${M1[@]}"
  run python3 -m src.deep.train --seed $S --tag "k_ctl_s$S" --data-dir data       "${M1[@]}"
done

for V in k_pow k_ctl; do for S in 0 1 2; do
  D=$([ $V = k_pow ] && echo data_fused || echo data)
  run python3 -m src.deep.infer --ckpt "models/fuse1km/cnn_${V}_s${S}_best.pt" \
      --data-dir "$D" --out-dir "results/fuse1km/${V}_s$S" --split test
done; done

ARGS=""
for V in k_pow k_ctl; do for S in 0 1 2; do
  P="results/fuse1km/${V}_s$S/cnn_1km_2019_2020.nc"
  [ -f "$P" ] && ARGS="$ARGS --product ${V}_s$S=$P"
done; done
run python3 -m src.score_variants --config config_vista.yaml $ARGS \
    --out results/fuse1km/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
