"""Educational tiled FlashAttention-style forward pass."""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr, lse_ptr, query_lengths_ptr, key_lengths_ptr,
    sequence_q: tl.constexpr, sequence_k: tl.constexpr, head_dim: tl.constexpr,
    heads: tl.constexpr, scale: tl.constexpr, CAUSAL: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(axis=0)
    batch_head = tl.program_id(axis=1)
    batch = batch_head // heads
    query_length = tl.load(query_lengths_ptr + batch)
    key_length = tl.load(key_lengths_ptr + batch)

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    dimension_offsets = tl.arange(0, BLOCK_D)
    valid_query = (query_offsets < sequence_q) & (query_offsets < query_length)
    valid_dimension = dimension_offsets < head_dim

    q_base = batch_head * sequence_q * head_dim
    q_ptrs = q_ptr + q_base + query_offsets[:, None] * head_dim + dimension_offsets[None, :]
    q = tl.load(q_ptrs, mask=valid_query[:, None] & valid_dimension[None, :], other=0.0)

    # One running maximum, denominator, and output accumulator per query row.
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    kv_base = batch_head * sequence_k * head_dim
    for key_start in range(0, tl.cdiv(sequence_k, BLOCK_N)):
        current_keys = key_start * BLOCK_N + key_offsets
        valid_key = (current_keys < sequence_k) & (current_keys < key_length)
        k_ptrs = k_ptr + kv_base + current_keys[:, None] * head_dim + dimension_offsets[None, :]
        v_ptrs = v_ptr + kv_base + current_keys[:, None] * head_dim + dimension_offsets[None, :]
        k = tl.load(k_ptrs, mask=valid_key[:, None] & valid_dimension[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=valid_key[:, None] & valid_dimension[None, :], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        score_mask = valid_query[:, None] & valid_key[None, :]
        if CAUSAL:
            score_mask &= current_keys[None, :] <= query_offsets[:, None]
        scores = tl.where(score_mask, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        # Prevent invalid padded query rows from evaluating -inf - -inf.
        new_max = tl.where(valid_query, new_max, 0.0)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        tile_sum = tl.sum(probabilities, axis=1)

        accumulator *= correction[:, None]
        if INPUT_IS_BF16:
            accumulator += tl.dot(probabilities.to(tl.bfloat16), v)
        else:
            accumulator += tl.dot(probabilities.to(tl.float16), v)
        running_sum = running_sum * correction + tile_sum
        running_max = new_max

    output = accumulator / running_sum[:, None]
    output_ptrs = output_ptr + q_base + query_offsets[:, None] * head_dim + dimension_offsets[None, :]
    tl.store(output_ptrs, output, mask=valid_query[:, None] & valid_dimension[None, :])
    lse_offsets = batch_head * sequence_q + query_offsets
    tl.store(lse_ptr + lse_offsets, running_max + tl.log(running_sum), mask=valid_query)


def pytorch_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    query_lengths: torch.Tensor, key_lengths: torch.Tensor, causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Readable reference that intentionally materializes scores/probabilities."""
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    batch, _, sequence_q, _ = q.shape
    sequence_k = k.shape[2]
    q_index = torch.arange(sequence_q, device=q.device)[None, None, :, None]
    k_index = torch.arange(sequence_k, device=q.device)[None, None, None, :]
    mask = (q_index < query_lengths[:, None, None, None]) & (k_index < key_lengths[:, None, None, None])
    if causal:
        mask &= k_index <= q_index
    scores = scores.masked_fill(~mask, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v.float()).to(q.dtype)
    output = output.masked_fill(q_index >= query_lengths[:, None, None, None], 0)
    return output, lse


def triton_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    query_lengths: torch.Tensor | None = None, key_lengths: torch.Tensor | None = None,
    causal: bool = True, block_m: int = 32, block_n: int = 32,
    num_warps: int = 4, num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("expected Q[B,H,Sq,D] and matching K/V[B,H,Sk,D]")
    if not all(t.is_cuda and t.is_contiguous() for t in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q/K/V must share FP16 or BF16 dtype")
    batch, heads, sequence_q, head_dim = q.shape
    sequence_k = k.shape[2]
    if head_dim > 256:
        raise ValueError("this teaching kernel supports head dimensions up to 256")
    device = q.device
    if query_lengths is None:
        query_lengths = torch.full((batch,), sequence_q, device=device, dtype=torch.int32)
    if key_lengths is None:
        key_lengths = torch.full((batch,), sequence_k, device=device, dtype=torch.int32)
    output, lse = torch.zeros_like(q), torch.full((batch, heads, sequence_q), -torch.inf, device=device)
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(sequence_q, block_m), batch * heads)
    attention_forward_kernel[grid](
        q, k, v, output, lse, query_lengths, key_lengths,
        sequence_q=sequence_q, sequence_k=sequence_k, head_dim=head_dim,
        heads=heads, scale=1.0 / math.sqrt(head_dim), CAUSAL=causal,
        INPUT_IS_BF16=q.dtype == torch.bfloat16, BLOCK_M=block_m,
        BLOCK_N=block_n, BLOCK_D=block_d, num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, lse

