#!/bin/bash
# POWER + AORC + ERA5 for the expanded Texas domain.
# POWER itself is trivial (~21 requests, about a minute); AORC dominates.
#SBATCH -J pw-data
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ATM23014
set -euo pipefail
export PYTHONPATH=$SCRATCH/downscaling
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"; export PYTHONUNBUFFERED=1
CFG=config_power_ls6.yaml
mkdir -p logs data_power/raw
echo "=== POWER coarse input $(date)"
$PY - <<'PYPOWER'
import sys; sys.path.insert(0, ".")
from src.utils import load_config, grids_from_config, setup_logging
from src.power_pipeline import build_power_coarse
setup_logging()
cfg = load_config("config_power_ls6.yaml")
build_power_coarse(cfg, grids_from_config(cfg))
PYPOWER

echo "----- AORC (21 years, 604,800 fine cells)"
$PY main.py --config $CFG download --aorc

echo "----- ERA5"
$PY - <<'PYERA5'
import sys; sys.path.insert(0, ".")
import dask; dask.config.set(scheduler="threads", num_workers=48)
from src.utils import load_config, grids_from_config, setup_logging
from src.atmos import build_era5_features
setup_logging()
cfg = load_config("config_power_ls6.yaml")
build_era5_features(cfg, grids_from_config(cfg))
PYERA5
echo "=== done $(date)"
