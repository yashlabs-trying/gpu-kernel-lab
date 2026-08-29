#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script must run on the Linux GPU Pod." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Deploy an NVIDIA GPU Pod first." >&2
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

mkdir -p results

python scripts/check_environment.py

echo
echo "Setup complete. Activate it later with: source .venv/bin/activate"
