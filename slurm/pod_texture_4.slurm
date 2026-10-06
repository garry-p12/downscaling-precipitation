#!/bin/bash
# Pass 4: close the two loose ends.
#
#   (i)  quantile map on top of E (spectral + heavy x3). Its dev inference needs
#        more memory than a login node allows, so it runs here.
#   (ii) seeds 1 and 2 for E and F. Seed 0 gave E a +0.059 POD gain over the
#        spectral penalty alone and F a clear failure; three seeds each decides
#        whether either is real. Dev metrics suffice -- train.py already reports
#        rmse, spec, pod30 and csi30 -- so no extra inference is needed.
#SBATCH -J pod-tex-4
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
mkdir -p logs models/podtex results/variants
CFG=config_vista.yaml
run() { echo; echo ">>> $*"; "$@" && echo "    ok" || echo "    FAILED: $*"; }

# --- (i) quantile map for E ----------------------------------------------
run python3 -m src.deep.infer --ckpt models/podtex/cnn_specheavy_best.pt \
    --data-dir data --out-dir results/dev-preds-specheavy --split dev
run python3 -m src.postprocess quantile-map --config $CFG --product cnn \
    --deep-dir results/variants/specheavy --dev-dir results/dev-preds-specheavy \
    --out results/variants/qm-specheavy

KEEP=results/comparison.keep.$$
[ -d results/comparison ] && mv results/comparison "$KEEP"
restore() { rm -rf results/comparison; if [ -d "$KEEP" ]; then mv "$KEEP" results/comparison; fi; return 0; }
trap restore EXIT
if [ -d results/variants/qm-specheavy ]; then
  python3 -m src.model_comparison --config $CFG --deep-dir results/variants/qm-specheavy >/dev/null 2>&1 \
    && { rm -rf results/variants/cmp-qm-specheavy; mv results/comparison results/variants/cmp-qm-specheavy; \
         echo "    scored qm-specheavy"; } || echo "    FAILED scoring qm-specheavy"
fi

# --- (ii) seeds for E and F ----------------------------------------------
COMMON=(--model cnn --base 24 --loss mse --select rmse --steps 6000 --eval-every 500
        --max-minutes 18 --batch 32 --patch 96 --lr 2e-4 --data-dir data --out models/podtex)
for SEED in 1 2; do
  run python3 -m src.deep.train --spectral-weight 0.01 --heavy-mm 30 --heavy-weight 3.0 \
      --seed "$SEED" --tag "specheavy_s$SEED" "${COMMON[@]}"
  run python3 -m src.deep.train --spectral-weight 0.01 --spectral-wet-mm 30 \
      --seed "$SEED" --tag "specheavycond_s$SEED" "${COMMON[@]}"
done

echo; echo "################ SEED SPREAD (dev, best checkpoint) ################"
python3 - <<'PY'
import json, glob, math, re, numpy as np
groups = {}
for f in sorted(glob.glob("models/podtex/*train_log*.json")):
    d = json.load(open(f)); a = d["args"]; tag = a.get("tag", "")
    fam = "E spectral+heavy" if tag.startswith("specheavy") and "cond" not in tag else \
          "F heavy-conditioned" if "cond" in tag else None
    if fam is None: continue
    h = [x for x in d["history"] if math.isfinite(x.get("rmse", float("nan")))]
    if not h: continue
    b = min(h, key=lambda x: x["rmse"])
    groups.setdefault(fam, []).append((a.get("seed"), b["rmse"], b["spec"], b["pod30"], b["csi30"]))
print("%-22s %5s %9s %8s %8s %8s" % ("family","seed","devRMSE","spec","POD30","CSI30"))
for fam, rows in groups.items():
    for r in sorted(rows): print("%-22s %5s %9.4f %8.4f %8.4f %8.4f" % ((fam,) + r))
    a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    if len(a) > 1:
        print("%-22s %5s %9.4f %8.4f %8.4f %8.4f   (mean, n=%d)" % (
            fam, "", *a.mean(0), len(a)))
        print("%-22s %5s %9.4f %8.4f %8.4f %8.4f   (sd)" % (fam, "", *a.std(0, ddof=1)))
PY
echo "=== done $(date)"
