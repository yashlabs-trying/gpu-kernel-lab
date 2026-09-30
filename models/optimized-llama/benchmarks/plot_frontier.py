#!/usr/bin/env python3
"""Plot output throughput against p99 end-to-end latency from analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = [row for row in analysis["rows"] if row["end_to_end_p99_ms"] is not None]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        [row["end_to_end_p99_ms"] for row in rows],
        [row["output_tokens_per_second"] for row in rows],
    )
    for row in rows:
        axis.annotate(
            f"c={row['concurrency']}, in={row['input_tokens']}",
            (row["end_to_end_p99_ms"], row["output_tokens_per_second"]),
            fontsize=7,
        )
    axis.set_xlabel("End-to-end p99 latency (ms)")
    axis.set_ylabel("Output tokens/s")
    axis.set_title("Llama throughput versus tail latency")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
