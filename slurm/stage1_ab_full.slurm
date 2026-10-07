#!/bin/bash
# Stage-1 ablations on the FULL 2000-2020 record, both domains.
#
# Everything measured locally used 2015-2020, which is 1096 training days
# against the 6423 the clusters hold. The ERA5 result (-0.172 mm) and the
# learning curve both came from that subset, so neither transfers. This
# re-measures on the record the paper would actually report.
#
# Two ablations, one factor each:
#   era5_tp          reanalysis precipitation, the only predictor that has moved
#                    the 10 km error so far
#   imerg_structure  IMERG's own ancillary fields -- mw_frac is the share of the
#                    daily total from microwave rather than IR infill, i.e. the
#                    retrieval reporting where it was guessing
#SBATCH -J stage1-ab
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8 2>/dev/null
export PYTHONPATH=/scratch/11755/gurup12/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

python3 - <<'PYEOF'
import yaml, itertools
base = {"au": "config_vista.yaml", "co": "config_colorado_vista.yaml"}
for dom, src in base.items():
    for tp, st in itertools.product((True, False), (True, False)):
        c = yaml.safe_load(open(src))
        f = c.setdefault("features", {})
        f["era5_exclude"] = [] if tp else ["era5_tp"]
        f["use_imerg_structure"] = st
        name = f"config_{dom}_tp{int(tp)}_st{int(st)}.yaml"
        yaml.safe_dump(c, open(name, "w"), sort_keys=False)
        print("wrote", name)
PYEOF

for D in au co; do
  for TP in 0 1; do
    for ST in 0 1; do
      echo
      echo "################ ${D}  era5_tp=${TP}  structure=${ST}"
      run python3 main.py --config config_${D}_tp${TP}_st${ST}.yaml features
      run python3 main.py --config config_${D}_tp${TP}_st${ST}.yaml train
    done
  done
done

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
