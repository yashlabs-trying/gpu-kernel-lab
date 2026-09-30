from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "optimized-llama"
    / "scripts"
    / "model_integrity.py"
)
SPEC = importlib.util.spec_from_file_location("model_integrity", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_inventory_is_stable_and_detects_changes(tmp_path):
    (tmp_path / "config.json").write_text('{"model": "llama"}', encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"not tracked")
    before = module.inventory(tmp_path)
    assert list(before) == ["config.json"]
    assert before == module.inventory(tmp_path)
    (tmp_path / "config.json").write_text('{"model": "changed"}', encoding="utf-8")
    assert module.inventory(tmp_path) != before
