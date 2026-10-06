#!/bin/bash
# Is the AMSE negative a finding about AMSE, or about the regime we ran it in?
#
# Per-sample AMSE on 96-cell crops cost +0.082 mm day-1 paired against the MSE
# control (t = 3.86, 3/3 seeds) and delivered LESS texture than doing nothing.
# Three alternative explanations, screened one seed each:
#   A  the per-bin max() switches on noise, because one crop gives a hopeless
#      amplitude estimate     -> pool the spectra across the batch first
#   B  too few modes in the inner annuli of a 96-cell crop  -> widen the bins
#   C  Subich fine-tune a trained model, where the coherence term is informative;
#      from a random start it is not  -> fine-tune the control checkpoint
# D is the control for C: the same fine-tune with plain MSE, so "AMSE helped"
# can be told apart from "more training helped".
#
# All arms are rain-masked to match --spectral-weight's mask, removing the other
# asymmetry between the two losses.
#SBATCH -J amse-screen
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
mkdir -p logs models/amse-screen
CTL=models/spec-seeds/cnn_s0_w0.0_best.pt
test -f "$CTL" || { echo "FATAL: control checkpoint $CTL missing"; exit 1; }
COMMON=(--model cnn --base 24 --select rmse --seed 0 --eval-every 500
        --batch 32 --patch 96 --data-dir data --out models/amse-screen)

echo "=== A: batch-pooled, rain-masked $(date)"
python3 -m src.deep.train --loss amse --amse-pool batch --amse-wet-mm 1.0 \
    --tag poolA --steps 6000 --max-minutes 18 --lr 2e-4 "${COMMON[@]}"

echo "=== B: batch-pooled, rain-masked, bins grouped x4"
python3 -m src.deep.train --loss amse --amse-pool batch --amse-wet-mm 1.0 --amse-bin-group 4 \
    --tag poolB --steps 6000 --max-minutes 18 --lr 2e-4 "${COMMON[@]}"

echo "=== C: fine-tune the MSE control with pooled AMSE"
python3 -m src.deep.train --loss amse --amse-pool batch --amse-wet-mm 1.0 \
    --init-ckpt "$CTL" --tag ftAMSE --steps 3000 --max-minutes 12 --lr 5e-5 "${COMMON[@]}"

echo "=== D: control for C -- same fine-tune, plain MSE"
python3 -m src.deep.train --loss mse \
    --init-ckpt "$CTL" --tag ftMSE --steps 3000 --max-minutes 12 --lr 5e-5 "${COMMON[@]}"
echo "=== done $(date)"
