#!/bin/bash
for v in 0.11.2 0.13.0 0.15.0 0.16.0 0.17.0 0.18.0 0.20.0; do
  echo "=== vllm $v ==="
  pip download vllm==$v --no-deps -d /tmp/vchk -q 2>/dev/null
  unzip -p /tmp/vchk/vllm-$v*.whl '*/METADATA' 2>/dev/null | grep -iE '^Requires-Dist: torch' | head -2
  rm -f /tmp/vchk/vllm-$v*.whl
done
