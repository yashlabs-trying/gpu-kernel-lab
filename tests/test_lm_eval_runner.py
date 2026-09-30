from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "optimized-llama"
    / "quality"
    / "run_lm_eval.py"
)
SPEC = importlib.util.spec_from_file_location("run_lm_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_command_is_deterministic(tmp_path: Path) -> None:
    args = argparse.Namespace(
        executable="lm_eval",
        model=tmp_path / "model",
        output_dir=tmp_path / "results",
        tasks=["hellaswag", "wikitext"],
        device="cuda:0",
        dtype="float16",
        batch_size="8",
        seed=7,
        limit=10.0,
    )

    command = MODULE.build_command(args)

    assert command[:4] == ["lm_eval", "run", "--model", "hf"]
    tasks_index = command.index("--tasks")
    assert command[tasks_index + 1 : tasks_index + 3] == ["hellaswag", "wikitext"]
    assert command.count("7") == 1
    assert command[-2:] == ["--limit", "10.0"]


def test_dry_run_does_not_require_model_or_executable(tmp_path: Path, capsys) -> None:
    result = MODULE.main(
        [
            "--model",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "results"),
            "--tasks",
            "piqa",
            "--dry-run",
        ]
    )

    assert result == 0
    assert "--tasks piqa" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()
