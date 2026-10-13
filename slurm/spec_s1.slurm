#!/bin/bash
# S1: give every octave of wavelength the same weight.
#
# E recovers fine-scale variance in proportion to how much of the penalty each
# scale receives, and nothing else. Measured on the shipped model, with a
# 96-cell patch and the penalty's radial bins uniform in wavenumber:
#
#     band        share of the loss   energy recovered
#     2-4 km            61.5 %             0.904
#     4-8 km            30.8 %             0.461
#     8-10 km            7.7 %             0.363
#     10-64 km           0   %             0.07-0.12
#
# The rank order is exact. Above the cut E is also *worse than bilinear*
# (16-32 km: 0.084 against 0.154; 32-64 km: 0.118 against 0.211), so the
# penalty has been buying 2-4 km texture partly by shrinking the mesoscale.
#
# Three arms separate the two possible causes. Everything else is held at the
# shipped configuration, so the only difference from E is this one knob.
#
# S1 keeps the 10 km cut and changes only how the bins inside it
# share the penalty. If the 4-10 km deficit is caused by the bin layout
# rather than by the cut, this is the arm that moves it.
#SBATCH -J spec-S1
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
mkdir -p logs models/specS1 results/specS1
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# Identical to slurm/austin_structure.slurm, which produced E.
COMMON=(--base 24 --steps 6000 --eval-every 500 --max-minutes 25
        --batch 32 --patch 96 --lr 2e-4 --select rmse --out models/specS1)

for S in 0 1 2; do
  run python3 -m src.deep.train --model cnn --loss mse \
      --spectral-weight 0.01 --spectral-bin-weight octave \
      --heavy-mm 30 --heavy-weight 3.0 --seed $S --tag "specS1_s$S" \
      --data-dir data "${COMMON[@]}"
done

for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt models/specS1/cnn_specS1_s${S}_best.pt \
      --data-dir data --out-dir results/specS1/s$S --split test
done

# Scored against the shipped model, paired by day. score_variants now reports
# band energy as well as the sub-10 km scalar; the two disagree by a factor of
# two on E and the bands are the point of this experiment.
ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product specS1_s$S=results/specS1/s$S/cnn_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control E_s1=results/austin_struct/E_s1/cnn_1km_2019_2020.nc \
    $ARGS --out results/specS1/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed.
exit $rc
