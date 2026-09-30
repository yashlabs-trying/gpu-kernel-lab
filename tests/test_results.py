from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernellab.cli import main
from kernellab.results import ResultValidationError, validate_benchmark_result


def valid_result():
    summary = {
        "count": 1,
        "mean_ms": 1.0,
        "median_ms": 1.0,
        "p90_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": None,
        "p99_eligible": False,
    }
    return {
        "schema_version": 2,
        "benchmark": "test",
        "model": "llama-3.2-3b",
        "methodology": {"warmups": 1, "repetitions": 1},
        "environment_before": {},
        "aggregate": {
            "successful_requests": 1,
            "input_tokens_per_second": 10.0,
            "output_tokens_per_second": 2.0,
            "request_latency": {"ttft": summary, "itl": summary, "end_to_end": summary},
        },
        "raw_samples": [
            {
                "request_id": "one",
                "input_tokens": 10,
                "output_tokens": 2,
                "arrival_ns": 1,
                "first_token_ns": 1,
                "token_timestamps_ns": [1, 2],
                "completed_ns": 2,
            }
        ],
    }


def test_valid_result_has_no_warnings():
    assert validate_benchmark_result(valid_result()) == []


def test_validator_rejects_unearned_p99():
    result = valid_result()
    result["aggregate"]["request_latency"]["ttft"]["p99_ms"] = 5.0
    with pytest.raises(ResultValidationError, match="reports p99"):
        validate_benchmark_result(result)


def test_validator_rejects_missing_raw_fields():
    result = valid_result()
    del result["raw_samples"][0]["arrival_ns"]
    with pytest.raises(ResultValidationError, match="arrival_ns"):
        validate_benchmark_result(result)


def test_validator_rejects_inconsistent_token_timestamps():
    result = valid_result()
    result["raw_samples"][0]["token_timestamps_ns"] = [2]
    with pytest.raises(ResultValidationError, match="counts differ"):
        validate_benchmark_result(result)


def test_cli_validates_file(tmp_path, capsys):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(valid_result()), encoding="utf-8")
    assert main(["validate-result", str(path)]) == 0
    assert capsys.readouterr().out.startswith("VALID")


def test_published_json_schema_accepts_valid_result():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "benchmark-result-v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(valid_result(), schema)
