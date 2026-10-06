#!/bin/bash
# Pass 3: everything downstream of passes 1 and 2, unattended.
#
#   * test-split inference for the two screens (spectral+heavy, heavy-conditioned)
#   * quantile maps for cnn, cnn+spectral and swin, fitted on the dev year
#   * probability-matched mean from the six diffusion members
#   * one scored comparison per variant, collated into a single table
#
# Deliberately not `set -e`: a failure in one variant must not throw away the
# others. Each step is checked and reported instead.
#SBATCH -J pod-tex-3
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
mkdir -p logs results/variants
CFG=config_vista.yaml

run() { echo; echo ">>> $*"; "$@" && echo "    ok" || echo "    FAILED: $*"; }

# --- test-split inference for the two trained screens ---------------------
run python3 -m src.deep.infer --ckpt models/podtex/cnn_specheavy_best.pt \
    --data-dir data --out-dir results/variants/specheavy --split test
run python3 -m src.deep.infer --ckpt models/podtex/cnn_specheavycond_best.pt \
    --data-dir data --out-dir results/variants/specheavycond --split test

# --- quantile maps, fitted on dev and applied to test ---------------------
run python3 -m src.postprocess quantile-map --config $CFG --product cnn \
    --deep-dir results/v3 --dev-dir results/dev-preds --out results/variants/qm-cnn
run python3 -m src.postprocess quantile-map --config $CFG --product swin \
    --deep-dir results/v3 --dev-dir results/dev-preds --out results/variants/qm-swin
run python3 -m src.postprocess quantile-map --config $CFG --product cnn \
    --deep-dir results/spectral/spec-w001 --dev-dir results/dev-preds-spec \
    --out results/variants/qm-cnnspec

# --- probability-matched mean --------------------------------------------
run python3 -m src.postprocess pm-mean --deep-dir results/diff-members \
    --out results/variants/pm-mean

# --- score each variant, keeping the main table intact --------------------
KEEP=results/comparison.keep.$$
[ -d results/comparison ] && mv results/comparison "$KEEP"
restore() { rm -rf results/comparison; if [ -d "$KEEP" ]; then mv "$KEEP" results/comparison; fi; return 0; }
trap restore EXIT

for V in specheavy specheavycond qm-cnn qm-swin qm-cnnspec pm-mean; do
  [ -d "results/variants/$V" ] || { echo "skip $V (no outputs)"; continue; }
  echo; echo ">>> scoring $V"
  python3 -m src.model_comparison --config $CFG --deep-dir "results/variants/$V" >/dev/null 2>&1 \
    && { rm -rf "results/variants/cmp-$V"; mv results/comparison "results/variants/cmp-$V"; echo "    ok"; } \
    || echo "    FAILED scoring $V"
done

echo; echo "################ COLLATED ################"
python3 - <<'PY'
import json, glob, os
rows = []
base = "results/comparison/model_comparison.json"
if os.path.exists(base):
    d = json.load(open(base))["products"]
    for k in ("bilinear", "xgboost", "cnn", "swin", "diffusion", "diffusion_member"):
        if k in d: rows.append((k, d[k]))
for p in sorted(glob.glob("results/variants/cmp-*/model_comparison.json")):
    tag = p.split("cmp-")[1].split("/")[0]
    d = json.load(open(p))["products"]
    for k in ("cnn", "swin", "qm"):
        if k in d and k not in ("bilinear", "xgboost", "nearest"):
            rows.append((f"{tag}:{k}", d[k])); break
hdr = ("variant", "RMSE", "spec", "POD", "FAR", "CSI", "fbias", "KGE", "FSS30@5", "wetarea")
print("%-24s %7s %7s %6s %6s %6s %6s %6s %8s %8s" % hdr)
for name, v in rows:
    det = v["detection"]["30.0"]; o = v["overall"]
    print("%-24s %7.3f %7.4f %6.3f %6.3f %6.3f %6.3f %6.3f %8.3f %8.3f" % (
        name, o["rmse"], v["spectral_ratio_sub10km"], det["pod"], det["far"], det["csi"],
        det["frequency_bias"], o["kge"], v["fss"]["30.0mm_5cell"], v["wet_area_ratio_1mm"]))
PY
echo "=== done $(date)"
