"""Matched Llama benchmark matrix generation."""

from __future__ import annotations

from itertools import product
from typing import Any

DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_INPUT_LENGTHS = (8, 128, 512, 2048, 8192)
DEFAULT_OUTPUT_LENGTHS = (32, 128, 256)


def build_matrix(
    *,
    concurrency: tuple[int, ...] = DEFAULT_CONCURRENCY,
    input_lengths: tuple[int, ...] = DEFAULT_INPUT_LENGTHS,
    output_lengths: tuple[int, ...] = DEFAULT_OUTPUT_LENGTHS,
    dtypes: tuple[str, ...] = ("float16",),
    quantizations: tuple[str, ...] = ("none", "int8_per_channel_weight_only"),
    graph_modes: tuple[str, ...] = ("enabled", "disabled"),
    repetitions: int = 100,
    warmups: int = 5,
    seed: int = 1234,
) -> dict[str, Any]:
    values = concurrency + input_lengths + output_lengths + (repetitions,)
    if min(values) < 1 or warmups < 0:
        raise ValueError("lengths, concurrency, and repetitions must be positive")
    cases = []
    combinations = product(
        dtypes,
        quantizations,
        graph_modes,
        concurrency,
        input_lengths,
        output_lengths,
    )
    for case_id, (
        dtype,
        quantization,
        graph_mode,
        requests,
        input_len,
        output_len,
    ) in enumerate(
        combinations, start=1
    ):
        cases.append(
            {
                "case_id": f"llama-{case_id:04d}",
                "model": "llama-3.2-3b",
                "dtype": dtype,
                "quantization": quantization,
                "cuda_graphs": graph_mode == "enabled",
                "concurrency": requests,
                "input_tokens": input_len,
                "output_tokens": output_len,
                "warmups": warmups,
                "repetitions": repetitions,
                "seed": seed,
                "temperature": 0.0,
                "ignore_eos": True,
            }
        )
    return {
        "schema_version": 1,
        "kind": "matched-benchmark-matrix",
        "case_count": len(cases),
        "cases": cases,
    }
