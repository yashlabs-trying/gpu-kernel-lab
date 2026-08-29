#!/usr/bin/env python3
"""Validate the remote GPU software stack and print reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def command_output(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        import torch
        import transformers
        import triton
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run scripts/setup_runpod.sh first.")

    cuda_available = torch.cuda.is_available()
    gpu = {}
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_vram_gib": round(props.total_memory / 2**30, 3),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessors": props.multi_processor_count,
        }

    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "git_revision": command_output("git", "rev-parse", "HEAD"),
        "nvidia_smi": command_output(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ),
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not cuda_available:
        raise SystemExit("PyTorch cannot access CUDA.")

    # Force an actual GPU operation instead of trusting availability flags alone.
    left = torch.arange(1024, device="cuda", dtype=torch.float32)
    right = left + 1
    torch.cuda.synchronize()
    if not torch.equal(right.cpu(), torch.arange(1, 1025, dtype=torch.float32)):
        raise SystemExit("CUDA tensor smoke test produced an incorrect result.")

    print("CUDA tensor smoke test: PASS")


if __name__ == "__main__":
    main()
