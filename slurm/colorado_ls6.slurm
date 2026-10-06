#!/bin/bash
# Colorado, remaining stages on Lonestar6.
#
# All downloads finished on Vista (NLCD/DEM, AORC 21 yr, IMERG 247 mo, ERA5)
# and the data now sits on shared Stockyard, which both systems mount. What is
# left -- AORC/NLCD alignment, features, tree -- is about an hour of CPU, so it
# fits the 2 h development queue instead of waiting 10 h for a Vista slot.
#SBATCH -J dsc-co-ls6
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014

set -euo pipefail
export PYTHONPATH=
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
CFG=config_colorado_ls6.yaml
mkdir -p logs models/colorado results/colorado
echo "=== Colorado, remaining stages on $(hostname) $(date)"
$PY -c "import xarray,dask;print('xarray',xarray.__version__,'dask',dask.__version__)"

echo "----- preprocess + features"
$PY main.py --config $CFG preprocess --force
$PY main.py --config $CFG features

echo "----- terrain summary (the thing Austin did not have)"
$PY - <<'PYTERR'
import xarray as xr, numpy as np
d = xr.open_dataset("/work/11755/gurup12/ls6/dsc-data-co/nlcd_aligned_1km.nc")
s = d["slope"].values
print(f"slope: mean {np.nanmean(s):.2f} deg  max {np.nanmax(s):.2f} deg")
print(f"fraction of 1 km cells above 5 deg slope: {np.nanmean(s > 5):.3f}   (Austin: 0.000)")
print(f"elevation on the 1 km grid: {np.nanmin(d['dem'].values):.0f} .. {np.nanmax(d['dem'].values):.0f} m")
PYTERR

echo "----- tree: train / predict / validate"
$PY main.py --config $CFG train --stage both
$PY main.py --config $CFG predict
$PY main.py --config $CFG validate
echo "=== done $(date)"
