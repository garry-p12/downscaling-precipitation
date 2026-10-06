#!/bin/bash
# Tree seed repeats. CPU-only, so this has no business occupying a GPU node.
#SBATCH -J dsc-treeseed
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
echo "=== tree seeds $(date)  (seed 42 is the published run)"
for SEED in 1 2 3; do
  sed -e "s|^  models: models$|  models: models/tree_seed${SEED}|" \
      -e "s|^  results: results$|  results: results/tree_seed${SEED}|" \
      config_vista.yaml > "config_tree_seed${SEED}.yaml"
  mkdir -p "models/tree_seed${SEED}" "results/tree_seed${SEED}"
  echo "----- tree seed ${SEED}"
  python3 main.py --config "config_tree_seed${SEED}.yaml" train --stage both --seed "$SEED"
  python3 main.py --config "config_tree_seed${SEED}.yaml" predict
  python3 main.py --config "config_tree_seed${SEED}.yaml" validate
done
python3 - <<'PY'
import json, glob, os
print("tree seeds (Austin test 2019-2020):")
for f in sorted(glob.glob("results/tree_seed*/validation_metrics.json")):
    d = json.load(open(f))["comparison_table"]
    print(f"  {os.path.dirname(f):26s} ML {d['rmse']['ml']:.4f}   bilinear {d['rmse']['bilinear']:.4f}")
PY
echo "=== done $(date)"
