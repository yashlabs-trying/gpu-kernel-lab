#!/usr/bin/env bash
# Rebuild environment on fresh RunPod A40 pod (venv lives on overlay, lost on restart)
set -euo pipefail

echo "==> venv"
python3 -m venv /opt/venv
source /opt/venv/bin/activate

echo "==> pip upgrade"
pip install --upgrade pip wheel setuptools -q

echo "==> torch (cu128)"
pip install torch --index-url https://download.pytorch.org/whl/cu128 -q

echo "==> vllm + transformers + quant + gguf"
pip install vllm transformers accelerate sentencepiece datasets bitsandbytes gguf -q

echo "==> done"
/opt/venv/bin/python -c "import torch, vllm, transformers, gguf; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__)"
