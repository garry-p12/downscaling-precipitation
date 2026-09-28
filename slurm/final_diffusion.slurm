#!/bin/bash
# Diffusion calibration attempt: x0 vs v parameterisation, both with min-SNR
# weighting, picked on the dev year by how well sampled spread matches observed.
#SBATCH -J dsc-diff3
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
echo "=== diffusion calibration $(date)"
MEAN=${MEAN:-models/final/cnn_best.pt}

for PARAM in x0 v; do
  echo "----- param=$PARAM  min-SNR gamma=5"
  python3 -m src.deep.train --model diffusion --param "$PARAM" --min-snr-gamma 5.0 --tag "$PARAM" \
      --steps 30000 --lr 2e-4 --max-minutes 22 --mean-ckpt "$MEAN" \
      --data-dir data --out models/final --batch 32 --patch 96 --eval-every 3000
  python3 -m src.deep.calibrate --ckpt "models/final/diffusion_${PARAM}_best.pt" --data-dir data \
      --days 10 --members 4 --steps 40 --etas 0.0,1.0 \
      --out "models/final/calib_${PARAM}.json"
done

BEST=$(python3 - <<'PY'
import json
best, score = None, 1e9
for p in ("x0", "v"):
    c = json.load(open(f"models/final/calib_{p}.json"))
    s = min(v["score"] for v in c["etas"].values())
    print(f"{p}: best score {s:.3f}, spread ratios "
          + ", ".join(f"{k}={v['spread_ratio']:.2f}" for k, v in c["etas"].items()))
    if s < score:
        best, score = p, s
print(best)
PY
)
PARAM=$(echo "$BEST" | tail -1)
echo "----- chosen parameterisation: $PARAM"
ETA=$(python3 -c "
import json;c=json.load(open('models/final/calib_${PARAM}.json'));print(c['best_eta'])")
python3 -m src.deep.infer --ckpt "models/final/diffusion_${PARAM}_best.pt" --data-dir data \
    --out-dir results/final --split test --members 6 --steps 40 --batch 4 --eta "$ETA"
echo "=== done $(date)"
