#!/bin/bash
# Align, build features, train/predict/validate the tree. Scored on the Austin
# box only, so the result is directly comparable with the IMERG study's 6.8%.
#SBATCH -J pw-tree
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ATM23014
set -euo pipefail
export PYTHONPATH=$SCRATCH/downscaling
PY=$WORK/dsc-env/bin/python
cd "$SCRATCH/downscaling"; export PYTHONUNBUFFERED=1
CFG=config_power_ls6.yaml
mkdir -p logs models/power results/power
echo "=== POWER preprocess + features $(date)"
$PY main.py --config $CFG preprocess --force
$PY main.py --config $CFG features
echo "----- tree"
$PY main.py --config $CFG train --stage both
$PY main.py --config $CFG predict
$PY main.py --config $CFG validate
echo "=== done $(date)"
