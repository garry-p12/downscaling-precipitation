#!/bin/bash
# Rebuild everything on the extended 2000-2020 record: align grids, regrid ERA5,
# then the full tree pipeline (features -> train -> predict -> validate).
#SBATCH -J dsc-rebuild
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
echo "=== rebuild on 2000-2020 $(date)"

echo "----- ERA5 regrid"
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from src.utils import load_config, grids_from_config, setup_logging
from src.atmos import build_era5_features
setup_logging()
cfg = load_config("config_vista.yaml"); 
build_era5_features(cfg, grids_from_config(cfg))
PY

echo "----- preprocess (parse IMERG cache + align all grids)"
python3 main.py --config config_vista.yaml preprocess --force

echo "----- tree: features / train / predict / validate"
python3 main.py --config config_vista.yaml features
python3 main.py --config config_vista.yaml train
python3 main.py --config config_vista.yaml predict
python3 main.py --config config_vista.yaml validate
echo "=== done $(date)"
