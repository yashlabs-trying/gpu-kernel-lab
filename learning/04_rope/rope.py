"""Qwen-style half-rotation RoPE for Q and K in one Triton launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_qk_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    q_rows, sequence_length, head_dim, PAIR_BLOCK: tl.constexpr,
):
    row = tl.program_id(axis=0)
    pair = tl.arange(0, PAIR_BLOCK)
    half_dim = head_dim // 2
    mask = pair < half_dim

    # Q rows are followed by K rows in one launch. Both tensors are [B,H,S,D].
    is_q = row < q_rows
    tensor_row = tl.where(is_q, row, row - q_rows)
    position = tensor_row % sequence_length
    row_start = tensor_row * head_dim

    first_ptr = tl.where(is_q, q_ptr + row_start + pair, k_ptr + row_start + pair)
    second_ptr = tl.where(
        is_q, q_ptr + row_start + half_dim + pair,
        k_ptr + row_start + half_dim + pair,
    )
    first = tl.load(first_ptr, mask=mask).to(tl.float32)
    second = tl.load(second_ptr, mask=mask).to(tl.float32)

    # Compact [S,D/2] tables: load each rotation pair's cos/sin only once.
    table_offset = position * half_dim + pair
    cosine = tl.load(cos_ptr + table_offset, mask=mask).to(tl.float32)
    sine = tl.load(sin_ptr + table_offset, mask=mask).to(tl.float32)
    rotated_first = first * cosine - second * sine
    rotated_second = second * cosine + first * sine

    first_out = tl.where(
        is_q, q_out_ptr + row_start + pair, k_out_ptr + row_start + pair
    )
    second_out = tl.where(
        is_q, q_out_ptr + row_start + half_dim + pair,
        k_out_ptr + row_start + half_dim + pair,
    )
    tl.store(first_out, rotated_first, mask=mask)
    tl.store(second_out, rotated_second, mask=mask)


def build_rope_tables(
    sequence_length: int, head_dim: int, device: torch.device,
    dtype: torch.dtype, base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    pair = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inverse_frequency = 1.0 / (base ** (pair / head_dim))
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)
    return angles.cos().to(dtype).contiguous(), angles.sin().to(dtype).contiguous()


def pytorch_rope(
    x: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    half = x.shape[-1] // 2
    first, second = x[..., :half].float(), x[..., half:].float()
    view_shape = (1, 1, x.shape[2], half)
    cosine, sine = cosine.float().view(view_shape), sine.float().view(view_shape)
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), dim=-1
    ).to(x.dtype)


def triton_rope_qk(
    q: torch.Tensor, k: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor,
    pair_block: int | None = None, num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("Q and K must have layout [B,H,S,D]")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("Q and K must share B, S, and D; head counts may differ")
    if q.dtype != k.dtype or not q.is_cuda or not k.is_cuda:
        raise ValueError("Q and K must be same-dtype CUDA tensors")
    if not all(t.is_contiguous() for t in (q, k, cosine, sine)):
        raise ValueError("all tensors must be contiguous")
    head_dim, sequence_length = q.shape[-1], q.shape[2]
    if head_dim % 2 or cosine.shape != (sequence_length, head_dim // 2):
        raise ValueError("cosine must be [S,D/2] and D must be even")
    if sine.shape != cosine.shape or cosine.dtype != q.dtype or sine.dtype != q.dtype:
        raise ValueError("sine/cosine shapes and dtypes must match Q/K")

    pair_block = pair_block or triton.next_power_of_2(head_dim // 2)
    if pair_block < head_dim // 2 or pair_block & (pair_block - 1):
        raise ValueError("pair_block must be a power of two covering D/2")
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    q_rows, k_rows = q.numel() // head_dim, k.numel() // head_dim
    rope_qk_kernel[(q_rows + k_rows,)](
        q, k, cosine, sine, q_out, k_out, q_rows, sequence_length, head_dim,
        PAIR_BLOCK=pair_block, num_warps=num_warps,
    )
    return q_out, k_out

