#!/bin/bash
# Seed repeats on the Austin domain.
#
# Every number in the results table comes from one training run, so
# "Swin 4.653 vs CNN 4.663" is currently indistinguishable from run-to-run
# noise -- and we have said so in the paper twice. Seed 0 already exists as
# models/v3; this adds seeds 1 and 2 for the two architectures whose RMSE
# ranking is actually contested, plus three tree seeds.
#
# Diffusion repeats are deliberately skipped: its claims are CRPS and spectral
# ratio, not the RMSE ranking, and it is the most expensive model to train.
#SBATCH -J dsc-seeds
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
echo "=== seed sweep $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

COMMON=(--loss mse --heavy-mm 30 --heavy-weight 3.0 --select rmse_pod --data-dir data
        --batch 32 --patch 96 --eval-every 1000 --dev-days 120)

for SEED in 1 2; do
  OUT=models/seed${SEED}
  RES=results/seed${SEED}
  mkdir -p "$OUT" "$RES"

  echo "----- cnn seed ${SEED}"
  python3 -m src.deep.train --model cnn --base 24 --steps 40000 --lr 2e-4 \
      --max-minutes 55 --seed "$SEED" --out "$OUT" "${COMMON[@]}"
  python3 -m src.deep.infer --ckpt "$OUT/cnn_best.pt" --data-dir data --out-dir "$RES" --split test

  echo "----- swin seed ${SEED}"
  python3 -m src.deep.train --model swin --steps 40000 --lr 1.5e-4 \
      --max-minutes 150 --seed "$SEED" --out "$OUT" "${COMMON[@]}"
  python3 -m src.deep.infer --ckpt "$OUT/swin_best.pt" --data-dir data --out-dir "$RES" --split test
done

echo "----- tree seeds (cheap, CPU; seed 42 is the published run)"
for SEED in 1 2 3; do
  sed -e "s|^  models: models$|  models: models/tree_seed${SEED}|" \
      -e "s|^  results: results$|  results: results/tree_seed${SEED}|" \
      config_vista.yaml > "config_tree_seed${SEED}.yaml"
  mkdir -p "models/tree_seed${SEED}" "results/tree_seed${SEED}"
  echo "  -- tree seed ${SEED}"
  python3 main.py --config "config_tree_seed${SEED}.yaml" train --stage both --seed "$SEED"
  python3 main.py --config "config_tree_seed${SEED}.yaml" predict
  python3 main.py --config "config_tree_seed${SEED}.yaml" validate
done

echo "----- gathering"
python3 - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("results/tree_seed*/validation_metrics.json")):
    d = json.load(open(f))["comparison_table"]
    rows.append((os.path.dirname(f), d["rmse"]["ml"], d["rmse"]["bilinear"]))
print("tree seeds (Austin test 2019-2020):")
for n, r, b in rows:
    print(f"  {n:26s} ML {r:.4f}   bilinear {b:.4f}")
PY
echo "=== done $(date)"
