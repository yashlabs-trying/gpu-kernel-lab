from __future__ import annotations

import sys
from pathlib import Path


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from methodology import RequestSample, aggregate_requests, latency_summary, percentile


def sample(index: int, *, error: str | None = None) -> RequestSample:
    start = 1_000_000_000 + index * 100_000_000
    return RequestSample(
        request_id=str(index),
        start_type="warm",
        input_tokens=10,
        output_tokens=3 if error is None else 0,
        arrival_ns=start,
        first_token_ns=None if error else start + 10_000_000,
        token_timestamps_ns=[]
        if error
        else [start + 10_000_000, start + 30_000_000, start + 55_000_000],
        completed_ns=start + 60_000_000,
        gpu_ttft_ms=None if error else 8.0,
        gpu_decode_ms=[] if error else [18.0, 23.0],
        error=error,
    )


def test_percentile_uses_linear_interpolation():
    assert percentile([0.0, 10.0], 0.5) == 5.0


def test_p99_is_withheld_below_100_request_repetitions():
    summary = latency_summary(range(1000), tail_repetitions=99)
    assert summary["p99_ms"] is None
    assert summary["p99_eligible"] is False


def test_p99_is_reported_at_100_request_repetitions():
    summary = latency_summary(range(100), tail_repetitions=100)
    assert summary["p99_ms"] == 98.01
    assert summary["p99_eligible"] is True


def test_request_metrics_use_first_token_and_consecutive_token_timestamps():
    request = sample(0)
    assert request.ttft_ms == 10.0
    assert request.itl_ms == [20.0, 25.0]
    assert request.end_to_end_ms == 60.0


def test_aggregate_separates_input_and_output_throughput_and_errors():
    aggregate = aggregate_requests(
        [sample(0), sample(1), sample(2, error="failed")],
        benchmark_start_ns=0,
        benchmark_end_ns=1_000_000_000,
    )
    assert aggregate["input_tokens"] == 20
    assert aggregate["output_tokens"] == 6
    assert aggregate["input_tokens_per_second"] == 20.0
    assert aggregate["output_tokens_per_second"] == 6.0
    assert aggregate["successful_requests"] == 2
    assert aggregate["failed_requests"] == 1
    assert aggregate["error_rate"] == 1 / 3


def test_failed_attempts_do_not_make_p99_eligible():
    requests = [sample(index) for index in range(99)]
    requests.append(sample(99, error="failed"))
    aggregate = aggregate_requests(
        requests,
        benchmark_start_ns=0,
        benchmark_end_ns=1_000_000_000,
    )
    assert aggregate["request_latency"]["ttft"]["p99_ms"] is None
