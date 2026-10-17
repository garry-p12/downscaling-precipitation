#!/bin/bash
# Does downscaling get easier as the target grid coarsens?
#
# The RSIR proposal asks for POWER downscaled to 25, 10, 5 and 1 km. We have
# only ever produced 1 km, and src.proposal_metrics answers a different
# question -- how good the 1 km product is when a user averages it. This asks
# the proposal's question directly: train a model to predict each grid.
#
# On the POWER arm (coarse 0.5 deg, fine 1/120 deg) the four spacings are
# exactly factors 2, 5, 10 and 60:
#
#     factor  2  -> 0.25 deg, 28 x 24    the NEX-GDDP grid
#     factor  5  -> 0.10 deg, 70 x 60    the IMERG grid
#     factor 10  -> 0.05 deg, 140 x 120  the MERRA-2 HydroSCS grid
#     factor 60  -> 1/120 deg, 840 x 720 what we already build
#
# Cells per step are held near 294,912 across all four so capacity per update
# is comparable and the factor is what differs:
#
#     f2   patch  24 x batch 512 = 294,912
#     f5   patch  60 x batch  82 = 295,200
#     f10  patch 120 x batch  20 = 288,000
#     f60  patch 180 x batch   9 = 291,600
#
# Each patch is a whole number of coarse cells, which the loader requires.
#
# The expected result is that skill over bilinear SHRINKS as the target
# coarsens, because the sharpening task disappears: at factor 2 there is only
# one halving to do. If instead it grows, the models are doing coarse-scale
# bias correction rather than downscaling, which is what the error budget in
# 4.10 already says and this would confirm from a new direction.
#SBATCH -J multires
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs models/multires results/multires
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# A config per output grid: same domain and splits, different fine_factor and
# processed path. Written here so the scorer finds the matching target.
python3 - <<'PY'
import yaml
for f, d in ((2, "dsc-power-f2"), (5, "dsc-power-f5"), (10, "dsc-power-f10"),
             (60, "dsc-power")):
    c = yaml.safe_load(open("config_power_vista.yaml"))
    c["paths"]["processed"] = f"/work/11755/gurup12/{d}"
    c["domain"]["fine_factor"] = f
    yaml.safe_dump(c, open(f"config_power_f{f}.yaml", "w"), sort_keys=False)
    print("wrote", f"config_power_f{f}.yaml", "factor", f, "->", d)
PY

COMMON=(--model cnn --loss mse --select rmse --base 24 --steps 6000
        --eval-every 500 --max-minutes 18 --lr 2e-4 --out models/multires)

train(){  # factor patch batch dir
  run python3 -m src.deep.train --tag "mr_f$1" --data-dir "/work/11755/gurup12/$4" \
      --patch "$2" --batch "$3" "${COMMON[@]}"
  run python3 -m src.deep.infer --ckpt "models/multires/cnn_mr_f$1_best.pt" \
      --data-dir "/work/11755/gurup12/$4" --out-dir "results/multires/f$1" --split test
}
train 2   24 512 dsc-power-f2
train 5   60  82 dsc-power-f5
train 10 120  20 dsc-power-f10
train 60 180   9 dsc-power

for F in 2 5 10 60; do
  P="results/multires/f$F/cnn_1km_2019_2020.nc"
  [ -f "$P" ] && run python3 -m src.score_variants --config "config_power_f$F.yaml" \
      --product "cnn_f$F=$P" --out "results/multires/scores_f$F.json"
done

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
