#!/bin/bash
# NLCD + DEM for the POWER domain. Split out because it is the long pole:
# ~130 tiles x 2 layers against MRLC's WCS, which is also the flakiest service
# in the pipeline (tiles cache to disk, so a restart resumes).
#SBATCH -J pw-nlcd
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -A ATM23014
set -euo pipefail
export PYTHONPATH=$SCRATCH/downscaling
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"; export PYTHONUNBUFFERED=1
mkdir -p logs data_power/raw
echo "=== POWER domain: NLCD + DEM $(date)"
$PY main.py --config config_power_ls6.yaml download --nlcd
echo "=== done $(date)"
