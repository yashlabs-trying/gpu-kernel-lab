import math

import pytest
import torch

from accuracy_utils import StreamingTensorMetrics, deterministic_prompts, evaluate_gate


def test_deterministic_prompts_are_stable_and_varied():
    first = deterministic_prompts(100, seed=7)
    assert first == deterministic_prompts(100, seed=7)
    assert first != deterministic_prompts(100, seed=8)
    assert len(first) == len(set(first)) == 100


def test_prompt_count_validation_and_expansion():
    with pytest.raises(ValueError):
        deterministic_prompts(0)
    prompts = deterministic_prompts(1000)
    assert len(prompts) == 1000


def test_streaming_tensor_metrics():
    stats = StreamingTensorMetrics()
    stats.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
    stats.update(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0]))
    row = stats.as_dict()
    assert row["count"] == 2
    assert row["max_abs"] == 1.0
    assert row["mean_abs"] == pytest.approx(0.25)
    assert 0 < row["min_cosine"] < 1
    assert row["finite"]


def test_nonfinite_values_fail_metrics_and_gate():
    stats = StreamingTensorMetrics()
    stats.update(torch.tensor([math.inf]), torch.tensor([1.0]))
    assert not stats.finite
    gate = evaluate_gate(prefill_agreement=1, decode_agreement=1,
                         minimum_layer_cosine=1, nll_delta=0,
                         finite=False)
    assert not gate["pass"] and not gate["checks"]["finite"]


@pytest.mark.parametrize("field,value", [
    ("prefill_agreement", .989), ("decode_agreement", .989),
    ("minimum_layer_cosine", .9989), ("nll_delta", .0101),
])
def test_each_threshold_is_enforced(field, value):
    values = dict(prefill_agreement=1.0, decode_agreement=1.0,
                  minimum_layer_cosine=1.0, nll_delta=0.0, finite=True)
    values[field] = value
    assert not evaluate_gate(**values)["pass"]


def test_missing_checkpoint_fails_gate():
    gate = evaluate_gate(prefill_agreement=1, decode_agreement=1,
                         minimum_layer_cosine=1, nll_delta=0,
                         finite=True, missing_checkpoints=1)
    assert not gate["pass"]
