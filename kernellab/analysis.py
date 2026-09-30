"""Post-process schema-v2 serving results without changing raw measurements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    aggregate = result["aggregate"]
    latency = aggregate["request_latency"]
    return {
        "benchmark": result["benchmark"],
        "model": result["model"],
        "concurrency": result["concurrency"],
        "input_tokens": result.get("prompt", {}).get("input_tokens"),
        "output_tokens": result.get("sampling", {}).get("output_tokens"),
        "output_tokens_per_second": aggregate["output_tokens_per_second"],
        "requests_per_second": aggregate["requests_per_second"],
        "error_rate": aggregate["error_rate"],
        "tensor_parallel_size": result.get("tensor_parallel_size", 1),
        "ttft_p99_ms": latency["ttft"]["p99_ms"],
        "itl_p99_ms": latency["itl"]["p99_ms"],
        "end_to_end_p99_ms": latency["end_to_end"]["p99_ms"],
    }


def throughput_latency_frontier(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row.get("end_to_end_p99_ms") is not None and row.get("error_rate", 1.0) == 0
    ]
    frontier = []
    for candidate in usable:
        dominated = any(
            other["output_tokens_per_second"] >= candidate["output_tokens_per_second"]
            and other["end_to_end_p99_ms"] <= candidate["end_to_end_p99_ms"]
            and (
                other["output_tokens_per_second"] > candidate["output_tokens_per_second"]
                or other["end_to_end_p99_ms"] < candidate["end_to_end_p99_ms"]
            )
            for other in usable
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: row["end_to_end_p99_ms"])


def saturation_point(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [row for row in rows if row.get("error_rate", 1.0) == 0]
    if not successful:
        return None
    return max(successful, key=lambda row: row["output_tokens_per_second"])


def length_fairness(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("end_to_end_p99_ms") is not None:
            groups[(row["concurrency"], row["output_tokens"])].append(row)
    fairness = []
    for (concurrency, output_tokens), group in sorted(groups.items()):
        if len({row["input_tokens"] for row in group}) < 2:
            continue
        fastest = min(group, key=lambda row: row["end_to_end_p99_ms"])
        slowest = max(group, key=lambda row: row["end_to_end_p99_ms"])
        fairness.append(
            {
                "concurrency": concurrency,
                "output_tokens": output_tokens,
                "fastest_input_tokens": fastest["input_tokens"],
                "slowest_input_tokens": slowest["input_tokens"],
                "p99_latency_ratio": slowest["end_to_end_p99_ms"]
                / fastest["end_to_end_p99_ms"],
            }
        )
    return fairness


def tensor_parallel_efficiency(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    baseline = next((row for row in rows if row.get("tensor_parallel_size") == 1), None)
    if baseline is None or baseline["output_tokens_per_second"] <= 0:
        return []
    output = []
    for row in sorted(rows, key=lambda item: item.get("tensor_parallel_size", 0)):
        size = row.get("tensor_parallel_size")
        if not isinstance(size, int) or size < 1:
            continue
        speedup = row["output_tokens_per_second"] / baseline["output_tokens_per_second"]
        output.append(
            {
                "tensor_parallel_size": size,
                "speedup": speedup,
                "scaling_efficiency": speedup / size,
            }
        )
    return output
