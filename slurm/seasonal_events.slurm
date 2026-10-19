#!/bin/bash
# Seasonal and event-based comparison including the shipped product.
#
# Asked for in review: "how about the suggestion in this email thread to
# compare seasonal, event based etc". The existing review in results/review
# predates M1 and covers XGBoost, CNN, Swin, Diffusion and the stacked blend.
#
# It also answers the review's sharper point -- that an RMSE difference of
# 0.1-0.3 mm/day does not mean much for rainfall -- from the other side. A
# product can be a quarter of a millimetre better on the annual mean and still
# be the wrong choice for a summer convective event or a winter frontal one,
# and splitting by season and by storm is what shows that.
#
# Six events rather than three, so the comparison is not resting on a single
# storm.
#SBATCH -J seasonal
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
# PREPEND. Overwriting this is what killed job 1079718: python3 could not find
# libpython3.11.so.1.0 and the job exited having produced nothing.
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs results/review_m1
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

run python3 -c "import numpy, xarray; print('env ok', numpy.__version__)"

run python3 -m src.review_analysis --config config_vista.yaml \
    --product M1=results/specM1/s2/cnn_1km_2019_2020.nc \
    --product E=results/austin_struct/E_s1/cnn_1km_2019_2020.nc \
    --product XGBoost=results/xgboost_1km_2019_2020.nc \
    --n-events 6 --out results/review_m1

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
