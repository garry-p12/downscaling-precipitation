#!/bin/bash
# Download ERA5 environmental predictors (I/O bound - runs on the CPU partition).
#SBATCH -J dsc-era5
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A ATM23014

set -euo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
# scipy on this module set is built against gcc14; the gcc13 libstdc++ that
# comes with it lacks CXXABI_1.3.15, which breaks scipy.interpolate/optimize
# (and therefore xarray's linear interp). Prepend the newer runtime.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRATCH/downscaling"
echo "=== ERA5 fetch $(date)"
# The python3 module ships zarr/gcsfs but not dask; install it once on a login
# node with `python3 -m pip install --user dask`.
python3 -c "import dask, zarr, gcsfs" || { echo "missing ERA5 dependency (see above)" >&2; exit 1; }
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
import dask; dask.config.set(scheduler="threads", num_workers=48)
from src.utils import load_config, grids_from_config, setup_logging
from src.atmos import build_era5_features
setup_logging()
cfg = load_config("data/config.yaml")
cfg["paths"]["processed"] = "data"
build_era5_features(cfg, grids_from_config(cfg))
PY
echo "=== done $(date)"
