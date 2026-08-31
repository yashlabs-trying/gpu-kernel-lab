#!/usr/bin/env bash
# Failures are recorded, not hidden. Benchmarks run only for passing lesson tests.
set -u
cd /workspace/gpu-kernel-lab
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export TRITON_CACHE_DIR=/workspace/.cache/triton
mkdir -p results/profiling_20260831
for lesson in learning/0*; do
  name=$(basename "$lesson")
  python -m pytest "$lesson" -q > "results/profiling_20260831/${name}_tests.log" 2>&1
  status=$?
  printf '%s tests_exit=%s\n' "$name" "$status"
  if [ "$status" -eq 0 ]; then
    python "$lesson/benchmark.py" --output "results/profiling_20260831/${name}.csv" > "results/profiling_20260831/${name}_benchmark.log" 2>&1
    printf '%s benchmark_exit=%s\n' "$name" "$?"
  fi
done
