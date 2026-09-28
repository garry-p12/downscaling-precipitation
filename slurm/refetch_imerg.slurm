#!/bin/bash
# Re-fetch IMERG 2000-2020 carrying the sub-daily structure and quality fields
# (precipitation_cnt, precipitation_cnt_cond, MWprecipitation,
# probabilityLiquidPrecipitation), then rebuild the tree pipeline on them.
#SBATCH -J dsc-imerg2
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
echo "=== IMERG re-fetch with structure fields $(date)"

# Stale artefacts must go or the pipeline silently reuses them: the monthly
# caches hold precip only, and `predict` skips any year whose file already
# exists, which would make `validate` score the previous product.
rm -rf data/imerg_cache data/imerg_raw_10km.nc
rm -f data/imerg_aligned_10km.nc data/features_*.nc data/X_*.npy data/y_*.npy data/idx_*.npz
rm -f data/imerg_climatology_10km.nc data/stage1_pred_10km.nc
rm -f results/downscaled_1km_*.nc results/imerg_downscaled_1km_*.nc results/downscaled_1km_index.json
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from src.utils import load_config, grids_from_config, setup_logging
from src.data_pipeline import download_imerg
setup_logging()
cfg = load_config("config_vista.yaml")
download_imerg(cfg, grids_from_config(cfg))
PY

echo "----- preprocess + tree pipeline"
python3 main.py --config config_vista.yaml preprocess --force
python3 main.py --config config_vista.yaml features
python3 main.py --config config_vista.yaml train
python3 main.py --config config_vista.yaml predict
python3 main.py --config config_vista.yaml validate
echo "=== done $(date)"
