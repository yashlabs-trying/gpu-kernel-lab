"""PyTorch reference and fused Triton SwiGLU activation.

For already-computed gate and up projection values:

    output[i] = SiLU(gate[i]) * up[i]
    SiLU(x) = x * sigmoid(x)
"""

import torch
import torch.nn.functional as torch_functional
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Adjacent lanes access adjacent elements, so all accesses are coalesced.
    # Compute nonlinear math in FP32 even when storage uses FP16 or BF16.
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    output = (gate * sigmoid_gate) * up

    # This store converts back to the output tensor's storage dtype.
    tl.store(output_ptr + offsets, output, mask=mask)


def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Trusted framework reference: eager SiLU followed by multiplication."""
    return torch_functional.silu(gate) * up


def triton_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    block_size: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Validate inputs, allocate output, and launch fused SwiGLU."""
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise ValueError("inputs must have matching shapes and dtypes")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("this kernel requires contiguous tensors")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")

    output = torch.empty_like(gate)
    n_elements = gate.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    swiglu_kernel[grid](
        gate,
        up,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output

