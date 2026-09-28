#!/usr/bin/env bash
# Download Llama-3.2-3B onto the GPU workspace (NOT committed to git).
# Requires HF_TOKEN for the gated model.
set -euo pipefail

MODEL="meta-llama/Llama-3.2-3B"
DEST="${MODEL_DIR:-/workspace/models/llama-3.2-3b}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Export it before running." >&2
  exit 1
fi

echo "==> Downloading ${MODEL} to ${DEST}"
mkdir -p "${DEST}"

pip install -q "huggingface_hub[cli]" 2>/dev/null || true

huggingface-cli download "${MODEL}" \
  --local-dir "${DEST}" \
  --token "${HF_TOKEN}"

echo "==> Done: ${DEST}"
