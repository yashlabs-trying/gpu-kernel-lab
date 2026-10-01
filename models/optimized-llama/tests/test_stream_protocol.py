from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_serving.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("bench_serving", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def clock(values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_stream_parser_uses_delta_token_ids_and_receipt_times():
    lines = [
        b": keepalive\n",
        b'data: {"choices": [{"token_ids": [7]}]}\n',
        b'data: {"choices": [{"token_ids": [8]}]}\n',
        b"data: [DONE]\n",
    ]
    ids, timestamps, multi = module.consume_sse(lines, clock_ns=clock([100, 150]))
    assert ids == [7, 8]
    assert timestamps == [100, 150]
    assert multi is False


def test_stream_parser_flags_multi_token_chunks():
    lines = [
        b'data: {"choices": [{"token_ids": [7, 8]}]}\n',
        b"data: [DONE]\n",
    ]
    ids, timestamps, multi = module.consume_sse(lines, clock_ns=clock([100]))
    assert ids == [7, 8]
    assert timestamps == [100, 100]
    assert multi is True
