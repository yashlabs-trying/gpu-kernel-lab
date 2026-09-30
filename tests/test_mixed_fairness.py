from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "optimized-llama"
    / "benchmarks"
    / "bench_mixed_fairness.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("bench_mixed_fairness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_workload_labels_are_balanced_and_deterministic() -> None:
    first = MODULE.workload_labels(100, 1234)
    second = MODULE.workload_labels(100, 1234)

    assert first == second
    assert first.count("short") == 100
    assert first.count("long") == 100
    assert first[:10] != ["short"] * 10
