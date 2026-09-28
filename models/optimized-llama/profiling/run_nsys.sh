#!/usr/bin/env bash
# Nsight Systems system/timeline profile.
# Usage: bash profiling/run_nsys.sh <python_cmd>
set -euo pipefail

CMD="${1:?usage: run_nsys.sh <command>}"
OUT="profiling/reports/nsys_$(date +%s)"

mkdir -p profiling/reports

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output "${OUT}" \
  --force-overwrite=true \
  bash -c "${CMD}"

echo "==> Report written to ${OUT}.nsys-rep"
