"""Small, explicit decode-only runtime optimizations for Qwen3.

This module deliberately does not replace the already-integrated Q/K RMSNorm,
RoPE, and static-cache write. It owns the two missing orchestration pieces:
one residual/RMSNorm boundary per decoder layer and fixed-buffer CUDA Graph
replay for a complete greedy decode step.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "learning/03_rmsnorm"))
from qwen_rmsnorm import triton_qwen_residual_rmsnorm
sys.path.insert(0, str(ROOT / "learning/11_static_kv"))
from static_kv import advance_position


@torch.inference_mode()
def install_decode_residual_norm(model):
    """Fuse attention residual-add with post-attention RMSNorm during B=1 decode.

    Prefill and every unsupported shape use the original layer implementation.
    The final MLP residual remains separate because the next layer owns its input
    RMSNorm; crossing that module boundary would require changing the model ABI.
    """
    if getattr(model, "_kernellab_residual_norm", None) is not None:
        raise ValueError("decode residual/RMSNorm fusion already installed")
    state = {"decode": False, "pending": None}

    def detect(_module, positional, keywords):
        ids = keywords.get("input_ids", positional[0] if positional else None)
        state["decode"] = ids is not None and tuple(ids.shape) == (1, 1)
        state["pending"] = None

    hook = model.register_forward_pre_hook(detect, with_kwargs=True)
    layers = list(model.model.layers)
    for index, layer in enumerate(layers):
        original = layer.forward

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=None,
            original=original,
            index=index,
            **kwargs,
        ):
            if not state["decode"] or hidden_states.shape != (1, 1, hidden_states.shape[-1]):
                return original(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            if state["pending"] is None:
                residual = hidden_states
                normalized = self.input_layernorm(hidden_states)
            else:
                mlp_output, previous_residual = state["pending"]
                normalized, residual = triton_qwen_residual_rmsnorm(
                    mlp_output,
                    previous_residual,
                    self.input_layernorm.weight,
                    self.input_layernorm.variance_epsilon,
                )
                state["pending"] = None
            attention_output, _ = self.self_attn(
                hidden_states=normalized,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            normalized, residual = triton_qwen_residual_rmsnorm(
                attention_output,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
            mlp_output = self.mlp(normalized)
            if index + 1 == len(layers):
                return residual + mlp_output
            # The following layer consumes these tensors with its input norm.
            # Returning residual preserves the model loop ABI; that placeholder
            # is intentionally ignored by the next patched layer.
            state["pending"] = (mlp_output, residual)
            return residual

        layer.forward = types.MethodType(forward, layer)

    model._kernellab_residual_norm = {
        "layers": len(layers),
        "state": state,
        "hook": hook,
        "fused_boundary": "attention residual -> post norm; MLP residual -> next-layer input norm",
    }
    return len(layers)


class GreedyDecodeGraph:
    """Capture and replay one fixed-address, batch-1 greedy decode step.

    The input token is updated inside the graph from the previous logits. Cache,
    mask, position, token, and output addresses must therefore remain stable.
    Sampling, logits processors, variable batch sizes, and EOS stopping belong
    outside this deliberately narrow graph.
    """

    @torch.inference_mode()
    def __init__(self, model, token, cache, attention_mask, position):
        if token.shape != (1, 1) or token.dtype != torch.long or not token.is_cuda:
            raise ValueError("token must be a CUDA int64 tensor with shape [1,1]")
        if position.shape != (1,) or position.dtype != torch.long or not position.is_cuda:
            raise ValueError("position must be a CUDA int64 tensor with shape [1]")
        if not isinstance(attention_mask, dict) or set(attention_mask) != {"full_attention"}:
            raise ValueError("expected {'full_attention': persistent_mask}")
        self.model = model
        self.token = token.clone()
        self.cache = cache
        self.attention_mask = attention_mask
        self.position = position
        self.position_ids = position.view(1, 1)

        # Compile/autotune and populate allocator pools before capture.
        self._step()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

        self.addresses = self._addresses()

    def _step(self):
        output = self.model(
            input_ids=self.token,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        # copy_ makes the next-token destination stable and records the update
        # inside the graph instead of allocating a new Python-visible tensor.
        self.token.copy_(output.logits[:, -1:].argmax(-1))
        advance_position(self.position)

    def _addresses(self):
        mask = self.attention_mask["full_attention"]
        return (self.token.data_ptr(), self.position.data_ptr(), mask.data_ptr())

    @torch.inference_mode()
    def replay(self):
        if self._addresses() != self.addresses:
            raise RuntimeError("a CUDA Graph input address changed")
        self.graph.replay()
        return self.token
