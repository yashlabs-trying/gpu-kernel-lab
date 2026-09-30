"""Schema-v2 benchmark result validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultValidationError(ValueError):
    pass


def _require(mapping: dict[str, Any], key: str, expected_type, location: str) -> Any:
    if key not in mapping:
        raise ResultValidationError(f"{location}: missing {key!r}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise ResultValidationError(
            f"{location}.{key}: expected {expected_type}, got {type(value).__name__}"
        )
    return value


def validate_benchmark_result(result: dict[str, Any]) -> list[str]:
    if result.get("schema_version") != 2:
        raise ResultValidationError("schema_version must equal 2")
    _require(result, "benchmark", str, "result")
    _require(result, "model", str, "result")
    methodology = _require(result, "methodology", dict, "result")
    warnings: list[str] = []

    raw = result.get("raw_samples")
    aggregate = result.get("aggregate")
    measurements = result.get("measurements")
    if raw is None and measurements is None:
        raise ResultValidationError("result must contain raw_samples or measurements")
    if raw is not None:
        if not isinstance(raw, list):
            raise ResultValidationError("raw_samples must be a list")
        if aggregate is None or not isinstance(aggregate, dict):
            raise ResultValidationError("request results require an aggregate object")
        repetitions = methodology.get("repetitions")
        if isinstance(repetitions, int) and repetitions != len(raw):
            raise ResultValidationError(
                f"methodology.repetitions={repetitions} but raw_samples has {len(raw)} entries"
            )
        for index, sample in enumerate(raw):
            if not isinstance(sample, dict):
                raise ResultValidationError(f"raw_samples[{index}] must be an object")
            required_fields = (
                "request_id",
                "input_tokens",
                "output_tokens",
                "arrival_ns",
                "completed_ns",
            )
            for field in required_fields:
                if field not in sample:
                    raise ResultValidationError(f"raw_samples[{index}] missing {field!r}")
            arrival = sample["arrival_ns"]
            completed = sample["completed_ns"]
            invalid_request_time = (
                not isinstance(arrival, int)
                or not isinstance(completed, int)
                or completed < arrival
            )
            if invalid_request_time:
                raise ResultValidationError(f"raw_samples[{index}] has invalid request timestamps")
            timestamps = sample.get("token_timestamps_ns", [])
            if not isinstance(timestamps, list):
                raise ResultValidationError(
                    f"raw_samples[{index}].token_timestamps_ns must be a list"
                )
            if sample.get("error") is None and len(timestamps) != sample["output_tokens"]:
                raise ResultValidationError(
                    f"raw_samples[{index}] output token/timestamp counts differ"
                )
            if timestamps != sorted(timestamps) or any(
                timestamp < arrival or timestamp > completed for timestamp in timestamps
            ):
                raise ResultValidationError(f"raw_samples[{index}] has non-monotonic token times")
            first = sample.get("first_token_ns")
            if timestamps and first != timestamps[0]:
                raise ResultValidationError(
                    f"raw_samples[{index}] first_token_ns does not match its first token"
                )
        successful = aggregate.get("successful_requests", 0)
        actual_successful = sum(sample.get("error") is None for sample in raw)
        if successful != actual_successful:
            raise ResultValidationError(
                f"aggregate successful_requests={successful}, expected {actual_successful}"
            )
        latency = aggregate.get("request_latency", {})
        for metric in ("ttft", "itl", "end_to_end"):
            summary = latency.get(metric, {}) if isinstance(latency, dict) else {}
            if summary.get("p99_ms") is not None and successful < 100:
                raise ResultValidationError(
                    f"{metric} reports p99 with only {successful} successful requests"
                )
        for rate in ("input_tokens_per_second", "output_tokens_per_second"):
            value = aggregate.get(rate)
            if not isinstance(value, (int, float)) or value < 0:
                raise ResultValidationError(f"aggregate has invalid {rate}")

    for key in ("environment", "environment_before"):
        if key in result:
            break
    else:
        warnings.append("environment metadata is missing")
    if methodology.get("warmups", 0) == 0:
        warnings.append("benchmark recorded zero warmups")
    return warnings


def validate_result_file(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise ResultValidationError("result root must be an object")
    return validate_benchmark_result(result)
