#!/bin/bash
# Colorado Front Range: the terrain-rich counterpart to Austin.
#
# Austin asked "can ML recover sub-grid rainfall with no orographic forcing?"
# and answered 6.8% over bilinear, with the 1 km residual stage rejected on
# every configuration. This domain is the same size (1050 coarse cells, same
# period, same 31 features) but spans the Continental Divide, so the sub-grid
# pattern should be partly deterministic. If the 20% target is reachable
# anywhere, it is reachable here.
#
# Caveat to read the results with: Colorado cool-season precipitation is
# largely snow, and IMERG is weak over snow-covered terrain. The seasonal
# breakdown in the validation report matters more here than it did in Texas.
#SBATCH -J dsc-co
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
CFG=config_colorado_vista.yaml
echo "=== Colorado Front Range build $(date)"
mkdir -p data_co/raw results/colorado models/colorado logs

echo "----- NLCD + DEM"
python3 main.py --config $CFG download --nlcd
python3 - <<'PY'
import rasterio, numpy as np
with rasterio.open("data_co/raw/nlcd/dem_1km.tif") as s:
    z = s.read(1).astype("float32"); z = z[np.isfinite(z)]
print(f"DEM: min {z.min():.0f} m  mean {z.mean():.0f} m  max {z.max():.0f} m")
PY

echo "----- AORC"
python3 main.py --config $CFG download --aorc

echo "----- IMERG (with sub-daily structure fields)"
python3 main.py --config $CFG download --imerg

echo "----- ERA5"
# `download` has no --era5 flag; ERA5 is built through src.atmos directly.
python3 - <<'PYERA5'
import sys; sys.path.insert(0, ".")
import dask; dask.config.set(scheduler="threads", num_workers=48)
from src.utils import load_config, grids_from_config, setup_logging
from src.atmos import build_era5_features
setup_logging()
cfg = load_config("config_colorado_vista.yaml")
build_era5_features(cfg, grids_from_config(cfg))
PYERA5

echo "----- preprocess + features"
python3 main.py --config $CFG preprocess --force
python3 main.py --config $CFG features

echo "----- terrain summary (the thing Austin did not have)"
python3 - <<'PY'
import xarray as xr, numpy as np
d = xr.open_dataset("data_co/nlcd_aligned_1km.nc")
s = d["slope"].values
print(f"slope: mean {np.nanmean(s):.2f} deg  max {np.nanmax(s):.2f} deg")
print(f"fraction of 1 km cells above 5 deg slope: {np.nanmean(s > 5):.3f}   (Austin: 0.000)")
print(f"elevation range on the 1 km grid: {np.nanmin(d['dem'].values):.0f} .. {np.nanmax(d['dem'].values):.0f} m")
PY

echo "----- tree: train / predict / validate"
python3 main.py --config $CFG train --stage both
python3 main.py --config $CFG predict
python3 main.py --config $CFG validate

echo "=== done $(date)"
