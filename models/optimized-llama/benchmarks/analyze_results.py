#!/usr/bin/env python3
"""Build throughput/latency frontier, saturation, and length-fairness data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kernellab.analysis import (  # noqa: E402
    length_fairness,
    result_row,
    saturation_point,
    tensor_parallel_efficiency,
    throughput_latency_frontier,
)
from kernellab.results import validate_benchmark_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_benchmark_result(result)
        rows.append(result_row(result))
    analysis = {
        "schema_version": 1,
        "rows": rows,
        "throughput_latency_frontier": throughput_latency_frontier(rows),
        "saturation_point": saturation_point(rows),
        "length_fairness": length_fairness(rows),
        "tensor_parallel_efficiency": tensor_parallel_efficiency(rows),
    }
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
