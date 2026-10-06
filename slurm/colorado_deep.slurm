#!/bin/bash
# The gap the paper's own title exposes: every texture and detection result is
# from the FLAT domain, while the headline claim is about orographic forcing.
# The tree pipeline alone ran on Colorado. Nobody has asked whether realistic
# texture is easier or harder when terrain forces the pattern.
#
# Control vs the combined configuration (spectral + heavy x3), three seeds,
# identical settings to the Austin runs so the two domains are comparable.
#SBATCH -J co-deep
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
cd "$SCRATCH/downscaling"
mkdir -p logs models/co-deep results/co-deep
CFG=config_colorado_vista.yaml
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || echo "    FAILED"; }
COMMON=(--model cnn --base 24 --select rmse --steps 6000 --eval-every 500
        --max-minutes 16 --batch 32 --patch 96 --lr 2e-4 --data-dir data_co --out models/co-deep)

for S in 0 1 2; do
  run python3 -m src.deep.train --loss mse --seed $S --tag "ctl_s$S" "${COMMON[@]}"
  run python3 -m src.deep.train --loss mse --spectral-weight 0.01 --heavy-mm 30 \
      --heavy-weight 3.0 --seed $S --tag "E_s$S" "${COMMON[@]}"
done
for S in 0 1 2; do
  for V in ctl E; do
    run python3 -m src.deep.infer --ckpt "models/co-deep/cnn_${V}_s${S}_best.pt" \
        --data-dir data_co --out-dir "results/co-deep/${V}_s${S}" --split test
  done
done
ARGS=(); for S in 0 1 2; do for V in ctl E; do
  P="results/co-deep/${V}_s${S}/cnn_1km_2019_2020.nc"; [ -f "$P" ] && ARGS+=(--product "${V}_s${S}=$P"); done; done
echo; echo "################ COLORADO, TEST, PER SEED ################"
python3 -m src.score_variants --config $CFG \
  --control "ctl_s0=results/co-deep/ctl_s0/cnn_1km_2019_2020.nc" \
  "${ARGS[@]}" --n-boot 1000 --out results/co-deep/scores.json 2>&1 | grep -viE warning | tail -16
echo "=== done $(date)"
