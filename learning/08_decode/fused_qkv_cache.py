"""Decode Q/K RMSNorm + RoPE + static K/V cache write in one launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def qk_norm_rope_cache_kernel(Q, K, V, QW, KW, COS, SIN, POSITIONS,
                              Q_OUT, K_CACHE, V_CACHE, ATTENTION_MASK,
                              Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
                              D: tl.constexpr, CAPACITY: tl.constexpr,
                              BLOCK: tl.constexpr, EPS: tl.constexpr):
    head, batch = tl.program_id(0), tl.program_id(1)
    is_q = head < Q_HEADS
    local_head = tl.where(is_q, head, head - Q_HEADS)
    valid_head = is_q | (local_head < KV_HEADS)
    col = tl.arange(0, BLOCK)
    valid = valid_head & (col < D)
    source = tl.where(is_q, Q, K)
    source_head = batch * tl.where(is_q, Q_HEADS, KV_HEADS) + local_head
    values = tl.load(source + source_head * D + col, mask=valid, other=0.).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / D
    normalized = values * tl.rsqrt(variance + EPS)
    rounded = normalized.to(Q.dtype.element_ty).to(tl.float32)
    weights = tl.load(tl.where(is_q, QW, KW) + col, mask=valid, other=0.).to(tl.float32)
    normed = rounded * weights

    half = D // 2
    pair = col % half
    position = tl.load(POSITIONS + batch)
    cosine = tl.load(COS + position * half + pair, mask=valid, other=0.).to(tl.float32)
    sine = tl.load(SIN + position * half + pair, mask=valid, other=0.).to(tl.float32)
    partner_col = tl.where(col < half, col + half, col - half)
    partner_value = tl.load(source + source_head * D + partner_col,
                            mask=valid, other=0.).to(tl.float32)
    # The partner must undergo the identical normalization and rounding.
    partner_normed = (partner_value * tl.rsqrt(variance + EPS)).to(Q.dtype.element_ty).to(tl.float32)
    partner_normed *= tl.load(tl.where(is_q, QW, KW) + partner_col,
                              mask=valid, other=0.).to(tl.float32)
    rotated = tl.where(col < half,
                       normed * cosine - partner_normed * sine,
                       normed * cosine + partner_normed * sine)
    q_offset = (batch * Q_HEADS + local_head) * D + col
    cache_offset = ((batch * KV_HEADS + local_head) * CAPACITY + position) * D + col
    tl.store(Q_OUT + q_offset, rotated, mask=is_q & valid)
    tl.store(K_CACHE + cache_offset, rotated, mask=(~is_q) & valid)
    # The K-head program also performs the contiguous V cache write.
    v = tl.load(V + (batch * KV_HEADS + local_head) * D + col,
                mask=(~is_q) & valid, other=0.)
    tl.store(V_CACHE + cache_offset, v, mask=(~is_q) & valid)
    # Exactly one K-head program per batch opens the current position in the
    # fixed additive mask. This adds no launch and leaves its address unchanged.
    tl.store(ATTENTION_MASK + batch * CAPACITY + position, 0.0,
             mask=(~is_q) & (local_head == 0) & valid_head & (col == 0))


def fused_qk_norm_rope_cache(q, k, v, q_weight, k_weight, cos, sin,
                             positions, k_cache, v_cache, attention_mask=None,
                             epsilon=1e-6,
                             check_bounds=False):
    """Exact logical sizes: Q[B,16,128], K/V[B,8,128], cache[B,8,C,128]."""
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError('expected Q[B,Hq,D] and matching K/V[B,Hkv,D]')
    batch, q_heads, d = q.shape
    if k.shape[0] != batch or k.shape[2] != d or d % 2:
        raise ValueError('incompatible Q/K dimensions')
    kv_heads, capacity = k.shape[1], k_cache.shape[2]
    if k_cache.shape != (batch,kv_heads,capacity,d) or v_cache.shape != k_cache.shape:
        raise ValueError('cache must be [B,Hkv,capacity,D]')
    if attention_mask is None:
        # Compatibility allocation for standalone lessons/tests only. The hot
        # integrated path always supplies one persistent [B,1,1,C] mask.
        attention_mask=torch.zeros((batch,1,1,capacity),device=q.device,dtype=q.dtype)
    if attention_mask.shape != (batch,1,1,capacity) or attention_mask.dtype != q.dtype:
        raise ValueError('attention_mask must be matching-dtype [B,1,1,capacity]')
    tensors=(q,k,v,q_weight,k_weight,cos,sin,positions,k_cache,v_cache,attention_mask)
    if not all(t.is_cuda and t.device == q.device and t.is_contiguous() for t in tensors):
        raise ValueError('all tensors must be contiguous on one CUDA device')
    if q.dtype not in (torch.float16,torch.bfloat16) or any(t.dtype != q.dtype for t in (k,v,q_weight,k_weight,cos,sin,k_cache,v_cache)):
        raise ValueError('Q/K/V/weights/tables/cache must share FP16 or BF16 dtype')
    if q_weight.shape != (d,) or k_weight.shape != (d,) or cos.shape != (capacity,d//2) or sin.shape != cos.shape:
        raise ValueError('invalid weight or compact RoPE table shape')
    if positions.shape != (batch,) or positions.dtype not in (torch.int32,torch.int64):
        raise ValueError('positions must be int32/int64 [B]')
    # A device-to-host `.item()` would synchronize every decode token. Enable
    # this only at API/test boundaries; the hot model path owns valid positions.
    if check_bounds and torch.any((positions < 0) | (positions >= capacity)).item():
        raise ValueError('cache position out of bounds')
    q_out=torch.empty_like(q)
    block=triton.next_power_of_2(d)
    qk_norm_rope_cache_kernel[(q_heads+kv_heads,batch)](
        q,k,v,q_weight,k_weight,cos,sin,positions,q_out,k_cache,v_cache,attention_mask,
        q_heads,kv_heads,d,capacity,block,epsilon,num_warps=4,enable_fp_fusion=False)
    return q_out
