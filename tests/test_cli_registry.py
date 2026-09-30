from __future__ import annotations

import json

import pytest

from kernellab.cli import _vllm_executable, main
from kernellab.matrix import build_matrix
from kernellab.registry import get_model


def test_registry_has_stable_llama_identity():
    model = get_model("llama-3.2-3b")
    assert model.source == "meta-llama/Llama-3.2-3B"
    assert model.default_dtype == "float16"
    with pytest.raises(ValueError, match="unknown model"):
        get_model("missing")


def test_matrix_is_complete_and_matched():
    matrix = build_matrix(
        concurrency=(1, 2),
        input_lengths=(8,),
        output_lengths=(32, 128),
        dtypes=("float16",),
        quantizations=("none", "int8_per_channel_weight_only"),
        graph_modes=("enabled", "disabled"),
    )
    assert matrix["case_count"] == 16
    assert len({case["case_id"] for case in matrix["cases"]}) == 16
    assert all(case["repetitions"] == 100 for case in matrix["cases"])


def test_cli_models_and_dry_run_commands(capsys, tmp_path):
    assert main(["models"]) == 0
    models = json.loads(capsys.readouterr().out)
    assert "llama-3.2-3b" in models

    assert main(["download", "llama-3.2-3b", "--dry-run", "--destination", str(tmp_path)]) == 0
    download = json.loads(capsys.readouterr().out)
    assert download["source"] == "meta-llama/Llama-3.2-3B"

    assert main(["serve", "llama-3.2-3b", "--model-path", str(tmp_path), "--dry-run"]) == 0
    command = capsys.readouterr().out
    assert "--no-enable-prefix-caching" in command
    assert "--served-model-name llama-3.2-3b" in command
    assert "--chat-template" in command

    result_path = tmp_path / "result.json"
    assert (
        main(
            [
                "benchmark",
                "llama-3.2-3b",
                "--model-path",
                str(tmp_path),
                "--concurrency",
                "4",
                "--output",
                str(result_path),
                "--dry-run",
            ]
        )
        == 0
    )
    benchmark_command = capsys.readouterr().out
    assert "--concurrency 4" in benchmark_command


def test_cli_writes_plan(tmp_path):
    output = tmp_path / "matrix.json"
    code = main(
        [
            "plan",
            "--concurrency",
            "1",
            "--input-lengths",
            "8",
            "--output-lengths",
            "32",
            "--dtypes",
            "float16",
            "--quantizations",
            "none",
            "--graph-modes",
            "enabled",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert json.loads(output.read_text())["case_count"] == 1


def test_vllm_executable_falls_back_to_active_venv(monkeypatch, tmp_path):
    python = tmp_path / "python"
    vllm = tmp_path / "vllm"
    vllm.write_text("", encoding="utf-8")
    monkeypatch.setattr("kernellab.cli.shutil.which", lambda _name: None)
    monkeypatch.setattr("kernellab.cli.sys.executable", str(python))
    assert _vllm_executable() == str(vllm)
