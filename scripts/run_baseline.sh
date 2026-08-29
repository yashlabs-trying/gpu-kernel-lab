#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
environment_file="results/environment_${timestamp}.json"
benchmark_file="results/baseline_${timestamp}.json"

python scripts/check_environment.py --json-out "${environment_file}"
python scripts/download_model.py
python scripts/smoke_test_model.py
python baseline/benchmark.py --output "${benchmark_file}"

echo
echo "Results created:"
echo "  ${environment_file}"
echo "  ${benchmark_file}"
echo
echo "Review them, then commit and push before terminating the Pod."
