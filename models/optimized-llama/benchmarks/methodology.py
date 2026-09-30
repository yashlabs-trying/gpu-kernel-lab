"""Shared, auditable benchmark primitives for the Llama project.

Wall-clock request latency uses ``perf_counter_ns`` (monotonic). GPU-only
latency is measured by the caller with CUDA events. Raw timestamps remain in
the result so every aggregate can be recomputed.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TAIL_REPETITIONS = 100


def percentile(values: Iterable[float], q: float) -> float:
    """Return a linearly interpolated percentile (q is in [0, 1])."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(
    values_ms: Iterable[float], *, tail_repetitions: int
) -> dict[str, float | int | None | bool]:
    """Summarize samples without advertising a weak p99.

    ``tail_repetitions`` is the number of independently measured benchmark
    requests, not the number of token intervals pooled across those requests.
    """
    values = [float(value) for value in values_ms]
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "p99_eligible": False,
            "min_ms": None,
            "max_ms": None,
        }
    eligible = tail_repetitions >= TAIL_REPETITIONS
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99) if eligible else None,
        "p99_eligible": eligible,
        "min_ms": min(values),
        "max_ms": max(values),
    }


@dataclass
class RequestSample:
    request_id: str
    start_type: str
    input_tokens: int
    output_tokens: int
    arrival_ns: int
    first_token_ns: int | None
    token_timestamps_ns: list[int]
    completed_ns: int
    gpu_ttft_ms: float | None = None
    gpu_decode_ms: list[float] | None = None
    output_token_ids: list[int] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ttft_ms"] = self.ttft_ms
        result["itl_ms"] = self.itl_ms
        result["end_to_end_ms"] = self.end_to_end_ms
        return result

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_ns is None:
            return None
        return (self.first_token_ns - self.arrival_ns) / 1_000_000

    @property
    def itl_ms(self) -> list[float]:
        return [
            (current - previous) / 1_000_000
            for previous, current in zip(
                self.token_timestamps_ns, self.token_timestamps_ns[1:]
            )
        ]

    @property
    def end_to_end_ms(self) -> float:
        return (self.completed_ns - self.arrival_ns) / 1_000_000


def aggregate_requests(
    samples: list[RequestSample], *, benchmark_start_ns: int, benchmark_end_ns: int
) -> dict[str, Any]:
    """Aggregate warm measured requests while retaining distinct token rates."""
    successful = [sample for sample in samples if sample.error is None]
    repetitions = len(samples)
    successful_repetitions = len(successful)
    wall_s = (benchmark_end_ns - benchmark_start_ns) / 1_000_000_000
    ttft = [sample.ttft_ms for sample in successful if sample.ttft_ms is not None]
    itl = [value for sample in successful for value in sample.itl_ms]
    end_to_end = [sample.end_to_end_ms for sample in successful]
    gpu_ttft = [
        sample.gpu_ttft_ms for sample in successful if sample.gpu_ttft_ms is not None
    ]
    gpu_decode = [
        value
        for sample in successful
        for value in (sample.gpu_decode_ms or [])
    ]
    input_tokens = sum(sample.input_tokens for sample in successful)
    output_tokens = sum(sample.output_tokens for sample in successful)
    return {
        "requests": repetitions,
        "successful_requests": len(successful),
        "failed_requests": repetitions - len(successful),
        "error_rate": (repetitions - len(successful)) / repetitions if repetitions else 0.0,
        "wall_seconds": wall_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_per_second": input_tokens / wall_s if wall_s else 0.0,
        "output_tokens_per_second": output_tokens / wall_s if wall_s else 0.0,
        "requests_per_second": len(successful) / wall_s if wall_s else 0.0,
        "request_latency": {
            "ttft": latency_summary(ttft, tail_repetitions=successful_repetitions),
            "itl": latency_summary(itl, tail_repetitions=successful_repetitions),
            "end_to_end": latency_summary(
                end_to_end, tail_repetitions=successful_repetitions
            ),
        },
        "gpu_only_latency": {
            "ttft": latency_summary(
                gpu_ttft, tail_repetitions=successful_repetitions
            ),
            "decode_step": latency_summary(
                gpu_decode, tail_repetitions=successful_repetitions
            ),
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repository,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    fields = [
        "name",
        "driver_version",
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "power.limit",
        "memory.used",
        "memory.total",
    ]
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if not output:
        return None
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    return dict(zip(fields, values))


def _nvidia_topology() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_metadata(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "vllm", "numpy", "pynvml")
        },
        "nvidia_smi": _nvidia_smi_snapshot(),
        "nvidia_topology": _nvidia_topology(),
    }
    try:
        import torch

        metadata["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime_version": torch.version.cuda,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            metadata["cuda"].update(
                {
                    "device_name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": props.total_memory,
                    "allocated_memory_bytes": torch.cuda.memory_allocated(0),
                    "reserved_memory_bytes": torch.cuda.memory_reserved(0),
                }
            )
    except ImportError:
        metadata["cuda"] = {"available": False, "runtime_version": None}
    return metadata


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
