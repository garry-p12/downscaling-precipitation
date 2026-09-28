#!/bin/bash
# Extend the record to 2000-2020: AORC, IMERG and ERA5 for the full period.
# I/O bound, so it runs on the CPU partition. Every stage is resumable - AORC
# skips existing years, IMERG skips cached months, ERA5 skips cached months -
# so the job can be resubmitted after a timeout and will pick up where it left off.
#SBATCH -J dsc-fetch
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
echo "=== fetch 2000-2020 $(date)"

echo "----- AORC"
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from src.utils import load_config, grids_from_config, setup_logging
from src.data_pipeline import download_aorc
setup_logging()
cfg = load_config("data/config.yaml"); cfg["paths"]["processed"] = "data"
download_aorc(cfg, grids_from_config(cfg))
PY

echo "----- IMERG"
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from src.utils import load_config, grids_from_config, setup_logging
from src.data_pipeline import download_imerg
setup_logging()
cfg = load_config("data/config.yaml"); cfg["paths"]["processed"] = "data"
cfg["paths"]["raw"] = "data/raw"
download_imerg(cfg, grids_from_config(cfg))
PY

echo "----- ERA5 (cache only; the regrid needs scipy, which is ABI-broken under this module set)"
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
import dask; dask.config.set(scheduler="threads", num_workers=48)
from src.utils import load_config, grids_from_config, setup_logging
from src.atmos import fetch_era5_daily
setup_logging()
cfg = load_config("data/config.yaml")
g = grids_from_config(cfg)
fetch_era5_daily(g, cfg["time"]["start"], cfg["time"]["end"], cache_dir="data/era5_cache")
PY
echo "=== done $(date)"
