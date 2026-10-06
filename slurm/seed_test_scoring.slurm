#!/bin/bash
# Per-seed test scoring for the four 6k-step configurations.
#
# Everything so far compares them on the dev year. The product recommendation
# has to be made on test, against a matched control -- the 6k-step, --select
# rmse control, NOT the 40k-step CNN of section 5.8 -- and with every comparison
# inside one split.
#
# Twelve checkpoints, already selected on dev; nothing is re-selected here.
#   control  cnn_s{0,1,2}_w0.0      plain mm-space MSE
#   spectral cnn_s{0,1,2}_w0.01     additive spectral penalty
#   E        cnn_specheavy*         spectral + heavy x3
#   F        cnn_specheavycond*     heavy-conditioned spectral
#SBATCH -J seed-test
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -uo pipefail
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SCRATCH/downscaling PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 DASK_SCHEDULER=synchronous DASK_NUM_WORKERS=1
cd "$SCRATCH/downscaling"
mkdir -p logs results/seedtest

infer () {  # ckpt, outname
  local out="results/seedtest/$2"
  if [ -f "$out/cnn_1km_2019_2020.nc" ]; then echo "    have $2"; return 0; fi
  echo ">>> infer $2"
  python3 -m src.deep.infer --ckpt "$1" --data-dir data --out-dir "$out" --split test \
    >/dev/null 2>&1 && echo "    ok" || echo "    FAILED $2"
}

for S in 0 1 2; do
  infer "models/spec-seeds/cnn_s${S}_w0.0_best.pt"  "ctl_s$S"
  infer "models/spec-seeds/cnn_s${S}_w0.01_best.pt" "spec_s$S"
done
infer "models/podtex/cnn_specheavy_best.pt"        "E_s0"
infer "models/podtex/cnn_specheavy_s1_best.pt"     "E_s1"
infer "models/podtex/cnn_specheavy_s2_best.pt"     "E_s2"
infer "models/podtex/cnn_specheavycond_best.pt"    "F_s0"
infer "models/podtex/cnn_specheavycond_s1_best.pt" "F_s1"
infer "models/podtex/cnn_specheavycond_s2_best.pt" "F_s2"

echo; echo ">>> scoring, day-block bootstrap"
ARGS=()
for S in 0 1 2; do
  for V in ctl spec E F; do
    P="results/seedtest/${V}_s${S}/cnn_1km_2019_2020.nc"
    [ -f "$P" ] && ARGS+=(--product "${V}_s${S}=$P")
  done
done
# control arm is ctl_s0 so paired differences are reported against it
python3 -m src.score_variants --config config_vista.yaml \
    --control "ctl_s0=results/seedtest/ctl_s0/cnn_1km_2019_2020.nc" \
    "${ARGS[@]}" --n-boot 1000 --out results/seedtest/scores.json 2>&1 | tail -20

echo; echo "################ TEST, PER SEED ################"
python3 - <<'PY'
import json, numpy as np
d = json.load(open("results/seedtest/scores.json"))
fam = {"ctl": "control (MSE)", "spec": "spectral only", "E": "spectral+heavy", "F": "heavy-cond"}
print("%-16s %5s %7s %7s %6s %6s %6s %6s %7s" % ("family","seed","RMSE","spec","POD","FAR","CSI","fbias","FSS30@5"))
for f in ("ctl","spec","E","F"):
    vals = []
    for s in (0,1,2):
        k = f"{f}_s{s}"
        if k not in d: continue
        v = d[k]; vals.append([v["rmse"], v.get("spectral_ratio_sub10km",np.nan), v["pod"], v["far"], v["csi"], v["frequency_bias"], v["fss5"]])
        print("%-16s %5d %7.3f %7.3f %6.3f %6.3f %6.3f %6.3f %7.3f" % ((fam[f], s) + tuple(vals[-1])))
    if len(vals) > 1:
        a = np.array(vals)
        print("%-16s %5s %7.3f %7.3f %6.3f %6.3f %6.3f %6.3f %7.3f  mean" % ((fam[f],"")+tuple(a.mean(0))))
        print("%-16s %5s %7.3f %7.3f %6.3f %6.3f %6.3f %6.3f %7.3f  sd" % ((fam[f],"")+tuple(a.std(0,ddof=1))))
print()
print("paired vs ctl_s0, day-block bootstrap 95% CI")
for k, v in d.items():
    if "vs_control" not in v: continue
    c = v["vs_control"]
    print("  %-10s dPOD %+.4f [%+.4f,%+.4f]  dCSI %+.4f [%+.4f,%+.4f]  dRMSE %+.4f [%+.4f,%+.4f]" % (
        k, c["pod"]["mean"], *c["pod"]["ci"], c["csi"]["mean"], *c["csi"]["ci"],
        c["rmse"]["mean"], *c["rmse"]["ci"]))
PY
echo "=== done $(date)"
