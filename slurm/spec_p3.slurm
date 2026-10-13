#!/bin/bash
# P3: S3 at a 192 km patch.
#
# S1, S2 and S3 each moved the band they were aimed at and none fixed the
# mesoscale. Best arm reached 0.106 at 16-32 km and 0.130 at 32-64 km, where
# plain bilinear sits at 0.154 and 0.211 -- every trained arm is still worse
# than interpolation at the scales catchment hydrology cares about.
#
# The reason is almost certainly the patch, not the objective. At a 96-cell
# patch those octaves barely exist in the loss:
#
#     octave        bins at patch 96    at 192
#     16-32 km             3               6
#     32-64 km             2               3
#
# and rapsd_torch removes the patch mean and applies a Hanning window, which
# attenuates exactly the longest in-patch wavelengths. So even when octave
# weighting hands 32-64 km a sixth of the penalty, that sixth lands on two
# noisy, heavily windowed bins. Doubling the patch triples the bins and moves
# the window further from the band of interest.
#
# --batch 8 is not a memory concession, it is the control: 192^2 x 8 is
# 294,912 fine cells per step, exactly what 96^2 x 32 gave the arms this is
# compared against. Steps, learning rate and everything else are unchanged, so
# the only difference from S3 is the patch.
#
# P3 is the arm octave weighting was always meant for. S3 starved the
# short end -- 4-8 km fell to 0.418, below E's 0.461 -- because six octaves
# split the penalty over bands that had 2 and 3 bins to spend it on. With
# the bins tripled, the same split may stop being a trade.
#SBATCH -J spec-P3
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
mkdir -p logs models/specP3 results/specP3
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
        --batch 8 --patch 192 --lr 2e-4 --select rmse --out models/specP3)

for S in 0 1 2; do
  run python3 -m src.deep.train --model cnn --loss mse \
      --spectral-weight 0.01 --spectral-cut 96 --spectral-bin-weight octave \
      --heavy-mm 30 --heavy-weight 3.0 --seed $S --tag "specP3_s$S" \
      --data-dir data "${COMMON[@]}"
done

for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt models/specP3/cnn_specP3_s${S}_best.pt \
      --data-dir data --out-dir results/specP3/s$S --split test
done

# Scored against the shipped model, paired by day. score_variants now reports
# band energy as well as the sub-10 km scalar; the two disagree by a factor of
# two on E and the bands are the point of this experiment.
ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product specP3_s$S=results/specP3/s$S/cnn_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control E_s1=results/austin_struct/E_s1/cnn_1km_2019_2020.nc \
    $ARGS --out results/specP3/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed.
exit $rc
