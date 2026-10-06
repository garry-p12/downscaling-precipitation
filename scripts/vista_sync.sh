#!/usr/bin/env bash
# Push code / pull results between this repo and TACC Vista.
#
#   scripts/vista_sync.sh push      # all code -> $SCRATCH/downscaling (then verifies)
#   scripts/vista_sync.sh verify    # per-file checksum comparison
#   scripts/vista_sync.sh data      # aligned NetCDFs -> cluster (one-off)
#   scripts/vista_sync.sh pull      # models + downscaled NetCDFs -> here
#
# push uses globs, never an explicit file list. An explicit list silently stops
# shipping new modules, and jobs then run stale code while appearing to succeed
# - that cost two wasted cluster runs. `verify` compares per file so a no-op
# push cannot pass unnoticed.
set -euo pipefail
HOST=${VISTA_HOST:-vista2}
REMOTE=${VISTA_DIR:-/scratch/11755/gurup12/downscaling}
cd "$(dirname "$0")/.."

code_files() {
  ls src/*.py src/deep/*.py scripts/*.py slurm/*.slurm slurm/*.sh main.py config*.yaml 2>/dev/null
}

do_verify() {
  local files local_n
  files=$(code_files | sort | tr '\n' ' ')
  # shellcheck disable=SC2086
  shasum $files | awk '{print $1, $2}' | sort > /tmp/_vs_local
  ssh "$HOST" "cd '$REMOTE' && shasum $files 2>/dev/null" | awk '{print $1, $2}' | sort > /tmp/_vs_remote
  local_n=$(wc -l < /tmp/_vs_local | tr -d ' ')
  if diff -q /tmp/_vs_local /tmp/_vs_remote >/dev/null 2>&1; then
    echo "verify: remote matches local ($local_n files)"
    return 0
  fi
  echo "verify: MISMATCH" >&2
  join -j 2 -o 0,1.1,2.1 <(sort -k2 /tmp/_vs_local) <(sort -k2 /tmp/_vs_remote) 2>/dev/null \
    | awk '$2 != $3 {print "  differs: " $1}' >&2
  comm -23 <(cut -d' ' -f2 /tmp/_vs_local) <(cut -d' ' -f2 /tmp/_vs_remote) \
    | sed 's/^/  missing on remote: /' >&2
  return 1
}

case "${1:-}" in
  push)
    code_files | tar --no-xattrs -czf - -T - \
      | ssh "$HOST" "mkdir -p '$REMOTE' && cd '$REMOTE' && tar xzf -"
    echo "pushed $(code_files | wc -l | tr -d ' ') files"
    do_verify
    ;;
  verify) do_verify ;;
  data)
    scp data/processed/{imerg_aligned_10km.nc,aorc_aligned_1km.nc,aorc_aligned_10km.nc,nlcd_aligned_1km.nc} \
        config.yaml "$HOST:$REMOTE/data/"
    ;;
  pull)
    mkdir -p results models logs/vista
    ssh "$HOST" "cd '$REMOTE' && tar czf - results/v2 results/v2_dev results/final \
        models/v2 models/final models/ablate*/*_train_log.json results/*.json results/*.md \
        2>/dev/null || true" | tar xzf -
    echo "pulled results and models"
    ;;
  *) echo "usage: $0 {push|verify|data|pull}" >&2; exit 1 ;;
esac
