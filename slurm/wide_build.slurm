#!/bin/bash
# Wide-domain experiment: train over 8 deg of longitude reaching real terrain
# (21% of cells >5 deg slope at W104 vs 0% at Austin), score only on the
# original Austin box so the result is directly comparable.
#SBATCH -J dsc-wide
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
CFG=config_wide_vista.yaml
echo "=== wide-domain build $(date)"
mkdir -p data_wide/raw results/wide models/wide

echo "----- NLCD + DEM"
python3 main.py --config $CFG download --nlcd

echo "----- AORC"
python3 main.py --config $CFG download --aorc

echo "----- IMERG (with sub-daily structure fields)"
python3 main.py --config $CFG download --imerg

echo "----- ERA5"
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
import dask; dask.config.set(scheduler="threads", num_workers=48)
from src.utils import load_config, grids_from_config, setup_logging
from src.atmos import build_era5_features
setup_logging()
cfg = load_config("config_wide_vista.yaml")
build_era5_features(cfg, grids_from_config(cfg))
PY

echo "----- preprocess + tree pipeline (scored on the Austin box only)"
python3 main.py --config $CFG preprocess --force
python3 main.py --config $CFG features
python3 main.py --config $CFG train
python3 main.py --config $CFG predict
python3 main.py --config $CFG validate
echo "=== done $(date)"
