from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernellab.llama_shapes import shape_manifest


def config(**overrides):
    values = {
        "model_type": "llama",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_llama_32_3b_shapes_are_derived_for_every_layer():
    manifest = shape_manifest(config())
    assert manifest["head_dim"] == 128
    assert len(manifest["per_layer"]) == 28
    shapes = manifest["per_layer"][0]["shapes"]
    assert shapes["q_proj_weight"] == [3072, 3072]
    assert shapes["k_proj_weight"] == [1024, 3072]
    assert shapes["gate_proj_weight"] == [8192, 3072]
    assert shapes["down_proj_weight"] == [3072, 8192]


def test_invalid_gqa_configuration_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        shape_manifest(config(num_key_value_heads=7))
