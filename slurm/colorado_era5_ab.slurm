#!/bin/bash
# Does ERA5 precipitation help more in the orographic, cool-season regime?
#
# On Austin it cut stage-1 10 km RMSE by 0.172 mm, and era5_tp entered as the
# fourth most important feature. Austin is the weak test: flat, convective,
# and with era5_sf at a 99th percentile of 0.000 there is no frozen
# precipitation for the reanalysis to be better at.
#
# Colorado is the real one. The oracle puts 83.6 % of its error in the coarse
# field, its control forecasts heavy rain at 15 % of the observed rate even at
# a matched threshold, and cool-season orographic precipitation is exactly the
# regime where a microwave retrieval struggles and a reanalysis does not.
#
# A/B by rebuilding the feature matrices with and without era5_tp, which is the
# only difference between the two arms. Trees, because they take seconds.
#SBATCH -J co-era5-ab
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A ATM23014
set -uo pipefail
PY=/work/11755/gurup12/ls6/dsc-env/bin/python
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
cd /scratch/11755/gurup12/downscaling
mkdir -p logs
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

# The arm without era5_tp: same config with the ERA5 feature list truncated.
$PY - <<'EOF'
import yaml
c = yaml.safe_load(open("config_colorado_ls6.yaml"))
c.setdefault("features", {})["era5_exclude"] = ["era5_tp"]
yaml.safe_dump(c, open("config_colorado_noera5tp.yaml", "w"), sort_keys=False)
print("wrote config_colorado_noera5tp.yaml")
EOF

echo "################ WITHOUT era5_tp"
run $PY main.py --config config_colorado_noera5tp.yaml features
run $PY main.py --config config_colorado_noera5tp.yaml train

echo
echo "################ WITH era5_tp"
run $PY main.py --config config_colorado_ls6.yaml features
run $PY main.py --config config_colorado_ls6.yaml train

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
