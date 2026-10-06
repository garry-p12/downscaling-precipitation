#!/bin/bash
# Cross-domain transfer: apply the COLORADO-trained stage-1 model to AUSTIN data.
#
# Fills the fourth cell of the design. Austin elevations (100-668 m) lie entirely
# below the Colorado training range (1,221-4,245 m), so the tree cannot
# extrapolate on any terrain feature -- a poor score may mean the relationships
# do not transfer, or merely that the features are out of distribution. The
# scale-free predictors (imerg_pct, the sub-daily ratios) are the ones that can
# transfer, so a result near the 6.8% Austin baseline would say terrain
# contributed little in Austin anyway, which is itself consistent with section 5.7.
#SBATCH -J dsc-cross
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A ATM23014

set -euo pipefail
export PYTHONPATH=$SCRATCH/downscaling
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"
export PYTHONUNBUFFERED=1
CFG=config_co_on_austin.yaml
mkdir -p logs results/co_on_austin
echo "=== Colorado model on Austin data $(date)"
$PY - <<'PYCHK'
import json
m = json.load(open("models/colorado/model_metadata.json"))
print("model trained on:", m.get("domain", "colorado"), "| features:", len(m["feature_names"]))
PYCHK
echo "----- predict (Colorado model, Austin features)"
$PY main.py --config $CFG predict
echo "----- validate against Austin AORC"
$PY main.py --config $CFG validate
echo "=== done $(date)"
