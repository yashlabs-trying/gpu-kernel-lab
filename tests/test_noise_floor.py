import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profiling"))
from argmax_noise_floor import compare_logits  # noqa: E402


def test_identical_logits_agree_fully():
    logits = torch.randn(5, 11)
    targets = logits.argmax(-1)
    result = compare_logits(logits, logits.clone(), targets, tie_margin=0.1)
    assert result["argmax_agreement"] == 1.0
    assert result["disagreements"] == 0
    assert result["nll_delta"] == pytest.approx(0.0)


def test_flip_on_near_tie_is_counted_as_near_tie():
    reference = torch.tensor([[2.00, 1.99, 0.0], [5.0, 0.0, 0.0]])
    candidate = torch.tensor([[1.99, 2.00, 0.0], [5.0, 0.0, 0.0]])
    result = compare_logits(reference, candidate, torch.tensor([0, 0]), tie_margin=0.05)
    assert result["argmax_agreement"] == 0.5
    assert result["disagreements"] == 1
    assert result["disagreements_on_near_ties"] == 1
    assert result["near_tie_positions"] == 1


def test_flip_with_large_margin_is_not_a_near_tie():
    reference = torch.tensor([[3.0, 1.0]])
    candidate = torch.tensor([[1.0, 3.0]])
    result = compare_logits(reference, candidate, torch.tensor([0]), tie_margin=0.5)
    assert result["disagreements"] == 1
    assert result["disagreements_on_near_ties"] == 0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        compare_logits(torch.randn(2, 3), torch.randn(3, 3), torch.tensor([0, 1]), 0.1)
