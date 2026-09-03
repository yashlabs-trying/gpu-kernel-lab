#!/usr/bin/env bash
# Source this file after reconnecting to a prepared RunPod volume:
#   source /workspace/gpu-kernel-lab/scripts/activate_runpod.sh
set -e

export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRITON_CACHE_DIR=/workspace/.cache/triton
export PIP_CACHE_DIR=/workspace/.cache/pip
export TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

source /workspace/gpu-kernel-lab/.venv/bin/activate
cd /workspace/gpu-kernel-lab

echo "KernelLab ready: $(python --version), repo $(git rev-parse --short HEAD)"
