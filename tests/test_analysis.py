from __future__ import annotations

from kernellab.analysis import (
    length_fairness,
    saturation_point,
    tensor_parallel_efficiency,
    throughput_latency_frontier,
)


def row(concurrency, input_tokens, throughput, latency, error_rate=0.0):
    return {
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "output_tokens": 32,
        "output_tokens_per_second": throughput,
        "end_to_end_p99_ms": latency,
        "error_rate": error_rate,
    }


def test_frontier_saturation_and_fairness():
    rows = [
        row(1, 8, 10, 20),
        row(2, 8, 18, 30),
        row(4, 8, 17, 50),
        row(1, 128, 8, 40),
        row(8, 8, 30, 80, error_rate=0.1),
    ]
    assert saturation_point(rows)["concurrency"] == 2
    frontier = throughput_latency_frontier(rows)
    assert [(item["concurrency"], item["input_tokens"]) for item in frontier] == [
        (1, 8),
        (2, 8),
    ]
    fairness = length_fairness(rows)
    assert fairness[0]["p99_latency_ratio"] == 2.0


def test_tensor_parallel_efficiency_uses_tp1_as_baseline():
    rows = [
        {"tensor_parallel_size": 1, "output_tokens_per_second": 100.0},
        {"tensor_parallel_size": 2, "output_tokens_per_second": 175.0},
    ]
    scaling = tensor_parallel_efficiency(rows)
    assert scaling[1] == {
        "tensor_parallel_size": 2,
        "speedup": 1.75,
        "scaling_efficiency": 0.875,
    }
