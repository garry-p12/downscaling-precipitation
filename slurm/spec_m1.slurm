#!/bin/bash
# M1: show the network the coarse field it has never been allowed to see.
#
# The mesoscale is the one band five arms have failed to move, and it is not
# an information problem. Measured against AORC:
#
#     field                      16-32 km   32-64 km
#     IMERG nearest (the input)    0.382      0.259
#     oracle, AORC avgd to 10 km   0.365      0.665
#     bilinear                     0.154      0.211
#     E                            0.087      0.122
#     P2                           0.103      0.116
#
# At 16-32 km the real input already carries as much variance as a *perfect*
# 10 km input, and the trained models deliver under a third of it. So the
# information arrives and is then destroyed.
#
# Bilinear is where it goes. Upsampling the input bilinearly drops that band
# from 0.382 to 0.154 before the network sees anything, and every model here
# is bilinear plus a correction conditioned on [cond, bilinear]. The network
# is asked to re-invent variance that was discarded one line earlier, from a
# field that no longer contains it.
#
# --nearest-channel appends the block-replicated coarse field as an extra
# conditioning channel. The base stays bilinear and the output layer stays
# zero-initialised, so an untrained model is still exactly the bilinear
# baseline (tests/test_deep.py asserts this). Only what the network can see
# changes, which keeps this comparable with every arm before it.
#
# Otherwise identical to P2, the best arm so far, so the channel is the only
# difference.
#SBATCH -J spec-M1
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
mkdir -p logs models/specM1 results/specM1
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# Note for anyone reading the logs: `evaluate` reports its dev `spec=` at
# --spectral-cut, so with the cut at 96 that line is the ratio over the whole
# spectrum and is NOT comparable with the 0.8-1.0 printed by the E runs.
# Selection is --select rmse and never reads it; the numbers that matter come
# from score_variants at the end, which always cuts at 10 km and also prints
# the per-band energy.
#
# Identical to slurm/austin_structure.slurm, which produced E.
COMMON=(--base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 8 --patch 192 --lr 2e-4 --select rmse --out models/specM1)

for S in 0 1 2; do
  run python3 -m src.deep.train --model cnn --loss mse \
      --spectral-weight 0.01 --spectral-cut 96 --nearest-channel \
      --heavy-mm 30 --heavy-weight 3.0 --seed $S --tag "specM1_s$S" \
      --data-dir data "${COMMON[@]}"
done

for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt models/specM1/cnn_specM1_s${S}_best.pt \
      --data-dir data --out-dir results/specM1/s$S --split test
done

# Scored against the shipped model, paired by day. score_variants now reports
# band energy as well as the sub-10 km scalar; the two disagree by a factor of
# two on E and the bands are the point of this experiment.
ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product specM1_s$S=results/specM1/s$S/cnn_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control E_s1=results/austin_struct/E_s1/cnn_1km_2019_2020.nc \
    $ARGS --out results/specM1/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed.
exit $rc
