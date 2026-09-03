"""Experimental W4A16 decode GEMV and vocabulary-tile top-k epilogue."""
import torch
import triton
import triton.language as tl


@triton.jit
def _project(X, PACKED, SCALES, N: tl.constexpr, K: tl.constexpr,
             GROUP: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    pair = tl.arange(0, GROUP // 2)
    accumulator = tl.zeros((BN,), tl.float32)
    for group in range(K // GROUP):
        code = tl.load(PACKED + row[:, None] * (K // 2) + group * (GROUP // 2) + pair[None, :],
                       row[:, None] < N, other=0).to(tl.int32)
        low, high = code & 15, code >> 4
        low = tl.where(low >= 8, low - 16, low).to(tl.float32)
        high = tl.where(high >= 8, high - 16, high).to(tl.float32)
        scale = tl.load(SCALES + row * (K // GROUP) + group, row < N, other=0)
        # Each packed byte is loaded once and supplies two adjacent K elements.
        x0 = tl.load(X + group * GROUP + pair * 2).to(tl.float32)
        x1 = tl.load(X + group * GROUP + pair * 2 + 1).to(tl.float32)
        accumulator += tl.sum((low * scale[:, None]) * x0[None, :] +
                              (high * scale[:, None]) * x1[None, :], axis=1)
    # Match the output dtype boundary of the proposed quantized linear layer.
    return accumulator.to(X.dtype.element_ty).to(tl.float32)


@triton.jit
def _gemv(X, PACKED, SCALES, Y, N: tl.constexpr, K: tl.constexpr,
          GROUP: tl.constexpr, BN: tl.constexpr):
    value = _project(X, PACKED, SCALES, N, K, GROUP, BN)
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    tl.store(Y + row, value, row < N)


@triton.jit
def _head_candidates(X, PACKED, SCALES, VALUES, INDICES,
                     N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr,
                     BN: tl.constexpr, TOP: tl.constexpr):
    value = _project(X, PACKED, SCALES, N, K, GROUP, BN)
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    eligible = row < N
    for rank in range(TOP):
        best = tl.max(tl.where(eligible, value, -float('inf')), 0)
        index = tl.min(tl.where(eligible & (value == best), row, 2147483647), 0)
        offset = tl.program_id(0) * TOP + rank
        tl.store(VALUES + offset, best)
        tl.store(INDICES + offset, index)
        eligible = eligible & (row != index)


def _validate(x, packed, block_n):
    packed.validate()
    if not x.is_cuda or x.device != packed.data.device:
        raise ValueError('activation and packed weights must be on one CUDA device')
    if x.ndim != 1 or x.numel() != packed.shape[1] or not x.is_contiguous():
        raise ValueError('one contiguous decode token x[K] required')
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('activations must be FP16 or BF16')
    if block_n not in (8, 16, 32, 64):
        raise ValueError('block_n must be 8,16,32,64')
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError('inference only')


def int4_gemv(x, packed, block_n=32):
    _validate(x, packed, block_n)
    n, k = packed.shape
    output = torch.empty(n, device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _gemv[(triton.cdiv(n, block_n),)](x, packed.data, packed.scales, output,
            n, k, packed.group_size, block_n, num_warps=4, enable_fp_fusion=False)
    return output


def int4_lm_head_topk(x, packed, top_k=1, block_n=32):
    """Fused LM-head/local selection, then global candidate merge.

    Does NOT produce full logits or probabilities. Supports raw greedy/top-k
    selection only, not repetition penalties, arbitrary logits processors, or
    top-p. Top-k ties may select different tied IDs than torch.topk; top-1 ties
    select the lowest vocabulary index. Caller must supply finite inputs.
    """
    _validate(x, packed, block_n)
    n, k = packed.shape
    if not 1 <= top_k <= min(8, block_n, n):
        raise ValueError('top_k must be between 1 and min(8,block_n,vocabulary)')
    tiles = triton.cdiv(n, block_n)
    values = torch.empty(tiles * top_k, device=x.device, dtype=torch.float32)
    indices = torch.empty(tiles * top_k, device=x.device, dtype=torch.int32)
    with torch.cuda.device(x.device):
        _head_candidates[(tiles,)](x, packed.data, packed.scales, values, indices,
            n, k, packed.group_size, block_n, top_k, num_warps=4, enable_fp_fusion=False)
        if top_k == 1:
            best, slot = values.max(dim=0, keepdim=True)
        else:
            best, slot = values.topk(top_k)
        return best, indices[slot].long()
