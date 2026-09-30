import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "models"
    / "optimized-llama"
    / "quality"
    / "evaluate.py"
)


def _capture(path: Path, logits: np.ndarray, tokens: np.ndarray) -> None:
    np.savez(
        path,
        logits=logits,
        generated_token_ids=tokens,
        layer_00=np.ones((len(tokens), 2), dtype=np.float32),
    )


def test_quality_cli_exit_code_enforces_release_gate(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    passing = tmp_path / "passing.npz"
    failing = tmp_path / "failing.npz"
    tokens = np.asarray([0, 1])
    logits = np.asarray([[2.0, 1.0], [1.0, 2.0]])
    _capture(reference, logits, tokens)
    _capture(passing, logits.copy(), tokens.copy())
    _capture(failing, logits[:, ::-1], tokens.copy())

    command = [sys.executable, str(SCRIPT), "--reference", str(reference)]
    passed = subprocess.run(
        command + ["--candidate", str(passing), "--output", str(tmp_path / "pass.json")],
        check=False,
    )
    failed = subprocess.run(
        command + ["--candidate", str(failing), "--output", str(tmp_path / "fail.json")],
        check=False,
    )
    assert passed.returncode == 0
    assert failed.returncode == 1


def test_quality_cli_supports_stacked_corpus_captures(tmp_path: Path) -> None:
    reference = tmp_path / "corpus-reference.npz"
    candidate = tmp_path / "corpus-candidate.npz"
    output = tmp_path / "corpus-result.json"
    tokens = np.asarray([[0, 1], [1, 0]])
    logits = np.asarray(
        [
            [[2.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 1.0]],
        ]
    )
    _capture(reference, logits, tokens)
    _capture(candidate, logits.copy(), tokens.copy())

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        check=False,
    )

    assert completed.returncode == 0
