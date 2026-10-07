#!/bin/bash
# What are IMERG's own ancillary fields worth to stage 1?
#
# STRUCTURE_FEATURES -- imerg_wet_frac, imerg_cond_intensity, imerg_mw_frac,
# imerg_prob_liquid -- are derived from precipitation_cnt, precipitation_cnt_cond,
# MWprecipitation and probabilityLiquidPrecipitation, all of which are already
# present in the aligned files on both clusters.
#
# They are the most targeted predictors available, because they are IMERG
# describing its own weakness rather than an external estimate of the truth:
# mw_frac is the share of the daily total that came from microwave rather than
# IR infill, and prob_liquid is phase. The oracle puts 92-97 % of the error in
# the coarse field, and these say where in that field the retrieval was guessing.
#
# Both domains, with and without, trees only -- stage 1 fits in seconds.
#SBATCH -J struct-ab
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
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/11755/gurup12/downscaling
mkdir -p logs
rc=0
run(){ echo; echo ">>> $*"; "$@" && echo "    ok" || { echo "    FAILED"; rc=1; }; }

python3 - <<'EOF'
import yaml
for src, dst, flag in (("config_vista.yaml",    "config_au_nostruct.yaml", False),
                       ("config_vista.yaml",    "config_au_struct.yaml",   True),
                       ("config_colorado_vista.yaml", "config_co_nostruct.yaml", False),
                       ("config_colorado_vista.yaml", "config_co_struct.yaml",   True)):
    c = yaml.safe_load(open(src))
    c.setdefault("features", {})["use_imerg_structure"] = flag
    yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
    print("wrote", dst, "use_imerg_structure =", flag)
EOF

for D in au co; do
  for V in nostruct struct; do
    echo
    echo "################ ${D} / ${V}"
    run python3 main.py --config config_${D}_${V}.yaml features
    run python3 main.py --config config_${D}_${V}.yaml train
  done
done

if [ $rc -eq 0 ]; then echo "=== ALL OK ==="; else echo "=== FAILED (rc=$rc) ==="; fi
