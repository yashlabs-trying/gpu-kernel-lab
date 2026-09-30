"""Derive Llama projection and normalization shapes from model configuration."""

from __future__ import annotations

from typing import Any


def shape_manifest(config: Any) -> dict[str, Any]:
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    intermediate = int(config.intermediate_size)
    if hidden % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if kv_heads > heads or heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    per_layer = {
        "input_layernorm": [hidden],
        "q_proj_weight": [heads * head_dim, hidden],
        "k_proj_weight": [kv_heads * head_dim, hidden],
        "v_proj_weight": [kv_heads * head_dim, hidden],
        "o_proj_weight": [hidden, heads * head_dim],
        "post_attention_layernorm": [hidden],
        "gate_proj_weight": [intermediate, hidden],
        "up_proj_weight": [intermediate, hidden],
        "down_proj_weight": [hidden, intermediate],
    }
    return {
        "model_type": getattr(config, "model_type", "llama"),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocabulary_size": int(config.vocab_size),
        "per_layer": [
            {"layer": index, "shapes": per_layer.copy()} for index in range(layers)
        ],
        "decode_hot_m": 1,
    }
