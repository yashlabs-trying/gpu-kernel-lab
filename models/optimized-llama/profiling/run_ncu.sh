#!/usr/bin/env bash
# Nsight Compute kernel-level profile.
# Usage: bash profiling/run_ncu.sh <python_cmd>
set -euo pipefail

CMD="${1:?usage: run_ncu.sh <command>}"
OUT="profiling/reports/ncu_$(date +%s)"

mkdir -p profiling/reports

ncu \
  --set full \
  --export "${OUT}" \
  --force-overwrite \
  --target-processes all \
  bash -c "${CMD}"

echo "==> Report written to ${OUT}.ncu-rep"
