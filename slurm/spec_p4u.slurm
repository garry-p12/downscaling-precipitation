#!/bin/bash
# P4u: uniform weighting at a 288 km patch.
#
# Doubling the patch from 96 to 192 was worth more than either spectral knob:
# RMSE 4.534 -> 4.466, sub-10 km energy 0.460 -> 0.669, 4-8 km +48 %,
# 8-16 km +58 %, at matched cells per step. This asks whether that is a trend
# or a step, and it is the control for a second question as well.
#
# The mesoscale is the one band nothing has fixed. Measured against a perfect
# 10 km input, it is not an information problem:
#
#     field                      16-32 km   32-64 km
#     IMERG nearest (the input)    0.382      0.259
#     oracle, AORC avgd to 10 km   0.365      0.665
#     bilinear                     0.154      0.211
#     E                            0.087      0.122
#     P3 (best arm so far)         0.122      0.123
#
# At 16-32 km the real input already carries as much variance as a perfect one
# and the CNN delivers under a quarter of it, so the model is destroying
# mesoscale structure that arrives intact. P3 reached 0.122 only once the
# patch was large enough to give that octave real bins; this arm has 9 bins
# there against P3's 6 and E's 3.
#
# CAVEAT, stated because it cannot be removed: 288 does not divide the cell
# budget evenly. 288^2 x 4 is 331,776 cells per step against the 294,912 every
# earlier arm used, 12.5 % more. The next step down, batch 3, is 15.6 % under.
# The excess favours this arm, so a null result is the stronger reading and a
# positive one owes some unknown part of itself to the extra data.
#
# P4u continues the P2 line: uniform bins, cut 96, patch doubled again.
#SBATCH -J spec-P4u
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
mkdir -p logs models/specP4u results/specP4u
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
        --batch 4 --patch 288 --lr 2e-4 --select rmse --out models/specP4u)

for S in 0 1 2; do
  run python3 -m src.deep.train --model cnn --loss mse \
      --spectral-weight 0.01 --spectral-cut 96 \
      --heavy-mm 30 --heavy-weight 3.0 --seed $S --tag "specP4u_s$S" \
      --data-dir data "${COMMON[@]}"
done

for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt models/specP4u/cnn_specP4u_s${S}_best.pt \
      --data-dir data --out-dir results/specP4u/s$S --split test
done

# Scored against the shipped model, paired by day. score_variants now reports
# band energy as well as the sub-10 km scalar; the two disagree by a factor of
# two on E and the bands are the point of this experiment.
ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product specP4u_s$S=results/specP4u/s$S/cnn_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control E_s1=results/austin_struct/E_s1/cnn_1km_2019_2020.nc \
    $ARGS --out results/specP4u/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed.
exit $rc
