#!/bin/bash
# Score every POWER product on the same test days: nearest, bilinear, the
# XGBoost two-stage product and whatever deep .nc files sit in results/power.
# CPU only -- this is netCDF reduction, not training.
#SBATCH -J pw-compare
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gg
#SBATCH -t 04:00:00
#SBATCH -A ATM23014
set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling
export PYTHONUNBUFFERED=1
cd "$SCRATCH/downscaling"
mkdir -p logs
echo "=== POWER comparison on $(hostname) $(date)"
ls -la results/power/*_1km_*.nc
python3 -m src.model_comparison --config config_power_vista.yaml --deep-dir results/power
echo "=== done $(date)"
