#!/usr/bin/env python3
"""Compile, launch, and validate a minimal Triton vector-add kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(left_ptr, right_ptr, output_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    left = tl.load(left_ptr + offsets, mask=mask)
    right = tl.load(right_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, left + right, mask=mask)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton smoke test.")

    size = 100_003
    torch.manual_seed(0)
    left = torch.randn(size, device="cuda", dtype=torch.float32)
    right = torch.randn(size, device="cuda", dtype=torch.float32)
    output = torch.empty_like(left)

    block = 256
    grid = (triton.cdiv(size, block),)
    vector_add_kernel[grid](left, right, output, size=size, BLOCK=block)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, left + right, rtol=0, atol=0)
    print(f"Triton compile/launch smoke test: PASS ({size} elements)")


if __name__ == "__main__":
    main()
