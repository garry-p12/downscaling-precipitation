#!/bin/bash
# End-to-end smoke test of the projections chain.
#
#   NEX-GDDP 0.25 deg  ->  bias correct  ->  downscale to 1 km
#
# This tests the PLUMBING, not the numbers. src.projections is still marked
# not-for-use: its corrected wet-day change carries a ~5 point positive offset
# and two of four models come out the wrong side of zero. What this checks is
# that a corrected CMIP6 field can be put where the loader expects a coarse
# input and carried through the trained 0.25 deg -> 1 km model to a 1 km field,
# with the grids, calendars and shapes all lining up.
#
# Two alignment points that are easy to get wrong and are handled here:
#   * CMIP6 stamps daily values at 12:00 and AORC at 00:00, so an exact join
#     finds nothing. The timestamps are floored to midnight.
#   * The scenario starts in 2015 while the reference record starts in 2000,
#     so only the overlap can be filled; the rest is left as NaN and the test
#     split (2019-2020) sits inside the overlap.
#SBATCH -J projsmoke
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs results/projections/smoke
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

python3 - <<'PY'
import numpy as np, xarray as xr
from pathlib import Path
src, dst = Path("data_oracle_30"), Path("data_proj_smoke")
dst.mkdir(exist_ok=True)
for n in ("aorc_aligned_1km.nc", "nlcd_aligned_1km.nc", "era5_aligned_10km.nc"):
    d = dst / n
    if d.exists() or d.is_symlink():
        d.unlink()
    d.symlink_to((src / n).resolve())
t = xr.open_dataset(src / "aorc_aligned_1km.nc").time.values
c = xr.open_dataset("results/projections/ACCESS-CM2_ssp245.nc")["precip"]
c = c.assign_coords(time=c.time.dt.floor("D").values)   # 12:00 -> 00:00
c = c.reindex(time=t)
# A day is unmatched when EVERY cell is NaN. Testing `.any()` instead counts a
# matched day as missing because the domain has a couple of NaN cells at its
# corner, which flagged all 7,519 days and blocked a pipeline that was working.
miss = int(np.isnan(c.values).all(axis=(1, 2)).sum())
test = c.sel(time=slice("2019-01-01", "2020-12-31"))
miss_test = int(np.isnan(test.values).all(axis=(1, 2)).sum())
print(f"unmatched days: {miss} of {len(t)} overall, {miss_test} of "
      f"{test.sizes['time']} in the test split", flush=True)
if miss_test:
    raise SystemExit("test split is not fully covered by the scenario")
c = c.fillna(0.0)          # pre-2015, plus the two corner cells
c.to_dataset(name="precip").to_netcdf(dst / "imerg_aligned_10km.nc")
print("coarse field written", dict(c.sizes), flush=True)
PY
[ $? -ne 0 ] && { echo "=== FAILED (alignment) ==="; exit 1; }

run python3 -m src.deep.infer --ckpt models/spatial025/cnn_o30_E_best.pt \
    --data-dir data_proj_smoke --out-dir results/projections/smoke --split test

P=results/projections/smoke/cnn_1km_2019_2020.nc
if [ -f "$P" ]; then
  python3 - <<PY
import numpy as np, xarray as xr
d = xr.open_dataset("$P")
v = "precipitation" if "precipitation" in d else list(d.data_vars)[0]
a = d[v].values
print(f"\nprojected 1 km field: {dict(d.sizes)}")
print(f"  mean {np.nanmean(a):.3f} mm/day, max {np.nanmax(a):.1f}, "
      f"wet fraction {np.nanmean(a >= 0.1):.3f}, non-finite {int(np.sum(~np.isfinite(a)))}")
PY
else
  echo "no output field"; rc=1
fi

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
exit $rc
