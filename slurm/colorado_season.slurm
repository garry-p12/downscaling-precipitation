#!/bin/bash
# Does the 1 km residual stage pass its gate in the orographic season?
#
# Stage 2 has been auto-rejected five times -- Austin (x3), the wide domain, and
# Colorado all-year (2.583 vs 2.588, a real 0.19 % gain but under the 0.5 %
# gate). Colorado's skill is strongly seasonal: 20.8 % RMSE reduction Oct-Apr
# against 8.0 % May-Sep. If sub-grid placement is recoverable anywhere, it is in
# the cool half-year, when terrain rather than convection decides where precip
# lands. Warm season is run as the control.
#SBATCH -J dsc-co-season
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014

set -euo pipefail
export PYTHONPATH=$SCRATCH/downscaling
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
echo "=== seasonal stage-2 gate $(date)"
for S in cool warm; do
  mkdir -p "models/colorado_${S}" "results/colorado_${S}"
  cp -n models/colorado/downscaler_coarse.pkl "models/colorado_${S}/" 2>/dev/null || true
  cp -n models/colorado/model_metadata.json "models/colorado_${S}/" 2>/dev/null || true
  echo "----- ${S} season"
  $PY main.py --config "config_colorado_${S}.yaml" train --stage fine
done
echo "=== done $(date)"
