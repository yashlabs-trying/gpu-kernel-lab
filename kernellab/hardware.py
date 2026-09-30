"""Safe hardware detection with a useful CPU-only fallback."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from typing import Any

GPU_FIELDS = (
    "name",
    "uuid",
    "driver_version",
    "compute_cap",
    "memory.total",
    "pstate",
)


def detect_hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "gpus": [],
    }
    if not result["nvidia_smi_available"]:
        return result
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        result["gpus"].append(dict(zip(GPU_FIELDS, values)))
    return result


def print_hardware() -> None:
    print(json.dumps(detect_hardware(), indent=2))
