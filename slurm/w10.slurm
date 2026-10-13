#!/bin/bash
# W10: ten times the spectral weight.
#
# Where the ladder stands after eight arms (3 seeds each, test 2019-2020):
#
#   arm                 RMSE   energy   16-32 km   vs bilinear
#   E   p96  c10 unif  4.550    0.399     0.087       0.56x
#   P2  p192 c96 unif  4.466    0.669     0.103       0.67x
#   P4o p288 c96 oct   4.515    0.765     0.140       0.91x
#   M1  p192 c96 +NN   4.443    0.637     0.093       0.60x
#
# Accuracy and mesoscale sit at opposite ends: M1 has the best RMSE in the
# study and nearly the worst mesoscale, P4o the reverse. Nothing has both.
#
# M1 also refuted the reason it was run. The claim was that the CNN loses the
# mesoscale because its bilinear base discards it before the network looks;
# given the unsmoothed field as an extra channel, 16-32 km got *worse*
# (0.093 against P2's 0.103). The network could see the variance and did not
# use it, so the obstacle is the loss, not the information.
#
# The far end of the same lever. 4.2 % RMSE cost at w=1.0 and cut 10 was the
# old sweep's finding, so 0.1 should be near the edge of free. Paired with
# W3 this says whether the mesoscale responds to pressure at all.
#SBATCH -J spec-W10
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
mkdir -p logs models/specW10 results/specW10
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
        --batch 8 --patch 192 --lr 2e-4 --select rmse --out models/specW10)

for S in 0 1 2; do
  run python3 -m src.deep.train --model cnn --loss mse \
      --spectral-weight 0.1 --spectral-cut 96 --nearest-channel \
      --heavy-mm 30 --heavy-weight 3.0 --seed $S --tag "specW10_s$S" \
      --data-dir data "${COMMON[@]}"
done

for S in 0 1 2; do
  run python3 -m src.deep.infer --ckpt models/specW10/cnn_specW10_s${S}_best.pt \
      --data-dir data --out-dir results/specW10/s$S --split test
done

# Scored against the shipped model, paired by day. score_variants now reports
# band energy as well as the sub-10 km scalar; the two disagree by a factor of
# two on E and the bands are the point of this experiment.
ARGS=""
for S in 0 1 2; do ARGS="$ARGS --product specW10_s$S=results/specW10/s$S/cnn_1km_2019_2020.nc"; done
run python3 -m src.score_variants --config config_vista.yaml \
    --control E_s1=results/austin_struct/E_s1/cnn_1km_2019_2020.nc \
    $ARGS --out results/specW10/scores.json

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
# Exit with the real status. Ending on the `if` meant the script always
# returned 0, so SLURM logged COMPLETED for a job whose every step had failed.
exit $rc
