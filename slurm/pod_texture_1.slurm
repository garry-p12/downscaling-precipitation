#!/bin/bash
# Pass 1 of the POD/texture push: the two cheap trained screens, plus the
# dev-year inference that unlocks quantile mapping for the deep products.
#
#   E  spectral penalty + heavy-day x3 upweighting. The two interventions that
#      each worked alone -- one for texture, one for detection (+0.030 POD at no
#      RMSE cost, section 5.4) -- act on different things and should compose.
#   F  heavy-conditioned spectral penalty. Section 4.8 found the added structure
#      was light-rain structure, because most rain-bearing crops are light.
#      Restricting the penalty to crops that actually contain heavy rain aims it
#      at the regime detection scores live in.
#   G  dev-split inference for cnn / cnn_spectral / swin, so the quantile map
#      can be fitted for them the same way it was for the tree product.
#SBATCH -J pod-tex-1
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
cd "$SCRATCH/downscaling"
mkdir -p logs models/podtex results/dev-preds
COMMON=(--model cnn --base 24 --loss mse --select rmse --seed 0 --steps 6000
        --eval-every 500 --max-minutes 18 --batch 32 --patch 96 --lr 2e-4
        --data-dir data --out models/podtex)

echo "=== E: spectral 0.01 + heavy x3  $(date)"
python3 -m src.deep.train --spectral-weight 0.01 --heavy-mm 30 --heavy-weight 3.0 \
    --tag specheavy "${COMMON[@]}"

echo "=== F: heavy-conditioned spectral (crops reaching 30 mm only)"
python3 -m src.deep.train --spectral-weight 0.01 --spectral-wet-mm 30 \
    --tag specheavycond "${COMMON[@]}"

echo "=== G: dev-split inference for the deep products"
python3 -m src.deep.infer --ckpt models/v3/cnn_best.pt  --data-dir data \
    --out-dir results/dev-preds --split dev
python3 -m src.deep.infer --ckpt models/v3/swin_best.pt --data-dir data \
    --out-dir results/dev-preds --split dev
python3 -m src.deep.infer --ckpt models/ablate-spectral/cnn_s0.01_best.pt --data-dir data \
    --out-dir results/dev-preds-spec --split dev
echo "=== done $(date)"
