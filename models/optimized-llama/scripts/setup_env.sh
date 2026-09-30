#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${LLAMA_DIR}/../.." && pwd)"
VENV_DIR="${KERNELLAB_VENV:-${REPO_DIR}/.venv-llama}"
CUDA_VARIANT="${KERNELLAB_CUDA_VARIANT:-cu128}"

case "${CUDA_VARIANT}" in
  cu128)
    TORCH_VERSION="2.9.0"
    LOCK_FILE="${LLAMA_DIR}/requirements-cu128.lock"
    ;;
  cu130)
    TORCH_VERSION="2.13.0"
    LOCK_FILE="${LLAMA_DIR}/requirements-cu130.lock"
    ;;
  *)
    echo "Unsupported KERNELLAB_CUDA_VARIANT=${CUDA_VARIANT}; choose cu128 or cu130" >&2
    exit 2
    ;;
esac

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel setuptools
"${VENV_DIR}/bin/python" -m pip install \
  "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_VARIANT}"
"${VENV_DIR}/bin/python" -m pip install -r "${LOCK_FILE}"
"${VENV_DIR}/bin/python" -m pip install --no-deps -e "${REPO_DIR}"
"${VENV_DIR}/bin/kernellab" doctor

echo "Environment ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
