from __future__ import annotations

import numpy as np
import pytest

from kernellab.quality import (
    compare_layers,
    compare_logits,
    negative_log_likelihood,
    release_gate,
    token_agreement,
)


def test_identical_logits_pass_release_gate():
    logits = np.asarray([[1.0, 2.0, 0.0], [3.0, 1.0, 2.0]], dtype=np.float32)
    metrics = compare_logits(logits, logits.copy())
    assert metrics["argmax_agreement"] == 1.0
    assert metrics["cosine_min"] == pytest.approx(1.0)
    assert metrics["max_abs_difference"] == 0.0
    assert release_gate(metrics)["passed"] is True


def test_argmax_regression_fails_gate():
    reference = np.asarray([[2.0, 1.0], [2.0, 1.0]])
    candidate = np.asarray([[1.0, 2.0], [2.0, 1.0]])
    metrics = compare_logits(reference, candidate)
    assert metrics["argmax_agreement"] == 0.5
    assert release_gate(metrics)["passed"] is False


def test_nll_regression_is_part_of_release_gate():
    metrics = {
        "argmax_agreement": 1.0,
        "cosine_min": 1.0,
    }
    gate = release_gate(metrics, reference_nll=1.0, candidate_nll=1.02)
    assert gate["passed"] is False
    assert gate["checks"]["nll_regression"] is False


def test_nll_matches_uniform_distribution():
    result = negative_log_likelihood(np.zeros((3, 4)), np.asarray([0, 1, 2]))
    assert result["nll"] == pytest.approx(np.log(4))
    assert result["perplexity"] == pytest.approx(4.0)


def test_token_and_layer_comparison_are_explicit_about_mismatch():
    tokens = token_agreement(np.asarray([1, 2, 3]), np.asarray([1, 9]))
    assert tokens == {
        "reference_tokens": 3,
        "candidate_tokens": 2,
        "compared_tokens": 2,
        "matching_tokens": 1,
        "agreement": 0.5,
        "exact_match": False,
    }
    layers = compare_layers(
        {"0": np.ones((1, 2)), "1": np.ones((1, 2))},
        {"0": np.ones((1, 2)), "2": np.ones((1, 2))},
    )
    assert layers["missing_layers"] == ["1"]
    assert layers["unexpected_layers"] == ["2"]


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_logits(np.zeros((2, 3)), np.zeros((2, 4)))
