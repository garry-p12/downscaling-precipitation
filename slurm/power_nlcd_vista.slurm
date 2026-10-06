#!/bin/bash
# NLCD + DEM for the POWER domain. Split out because it is the long pole:
# ~130 tiles x 2 layers against MRLC's WCS, which is also the flakiest service
# in the pipeline (tiles cache to disk, so a restart resumes).
#SBATCH -J pwv-nlcd
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -A ATM23014
set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling
PY=python3
cd "$SCRATCH/downscaling"; export PYTHONUNBUFFERED=1
mkdir -p logs /work/11755/gurup12/dsc-power/raw
echo "=== POWER domain: NLCD + DEM $(date)"
$PY main.py --config config_power_vista.yaml download --nlcd
echo "=== done $(date)"
