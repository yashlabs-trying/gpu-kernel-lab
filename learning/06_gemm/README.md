# Lesson 6: Tiled GEMM

GEMM computes:

```text
C[M,N] = A[M,K] @ B[K,N]
C[m,n] = sum over k of A[m,k] * B[k,n]
```

Unlike vector add, GEMM can reuse each loaded number many times. Good tiling turns
expensive global-memory loads into many Tensor Core operations.

## Hardware mapping

One Triton program owns a `BLOCK_M × BLOCK_N` output tile. It walks through K in
`BLOCK_K` pieces:

```text
load A tile [BLOCK_M, BLOCK_K]
load B tile [BLOCK_K, BLOCK_N]
             ↓
       accumulator += tl.dot(A tile, B tile)
             ↓
advance to next K tile
             ↓
store final C tile once
```

The grid contains:

```text
ceil(M/BLOCK_M) × ceil(N/BLOCK_N) programs
```

Nearby M programs are grouped so they visit related B tiles close together,
improving cache reuse.

## 1. Global memory versus shared memory and registers

The worst design would load A and B from global memory separately for every
individual multiply. Tiling instead describes vector A/B tiles with `tl.load` and
reuses them across `tl.dot`. The FP32 output accumulator remains on-chip through
the complete K loop.

Triton decides the final placement of staged tiles—registers, shared memory, and
cache behavior—during compilation. We should not claim a specific shared-memory
instruction merely because the source says `tl.load`. Nsight Compute, generated
PTX/SASS, shared-memory counters, register count, and spill counters provide the
evidence.

Important trade-off:

```text
larger tiles -> more reuse and Tensor Core work
larger tiles -> more registers/shared memory per program
larger tiles -> possibly lower occupancy or spills
```

## 2. Double buffering and advanced pipelining

While `tl.dot` consumes K tile `i`, an optimized pipeline can prepare tile
`i+1`. Two buffers express the basic ping-pong idea:

```text
buffer A: compute tile i       buffer B: load tile i+1
buffer B: compute tile i+1     buffer A: load tile i+2
```

Triton's `num_stages` tunes software-pipeline depth:

```text
num_stages=2 -> basic double-buffering candidate
num_stages=3/4 -> deeper look-ahead candidates
```

More stages can hide memory latency but consume more shared memory/register
resources. The benchmark holds tile shape fixed while testing stages `2,3,4`.
Whether the NVIDIA compiler emits asynchronous copies such as `cp.async` depends
on GPU architecture, types, alignment, Triton version, and compiler lowering. We
will verify generated code rather than equating `num_stages` with guaranteed
instructions.

## 3. Memory coalescing

A is contiguous across K and B is contiguous across N in row-major storage:

```python
A address = row * stride_am + k * stride_ak
B address = k * stride_bk + column * stride_bn
```

The two-dimensional pointer grids let neighboring logical lanes load neighboring
values along these contiguous dimensions. Masks insert zeros for partial M/N/K
tiles, so irregular shapes remain correct without invalid memory accesses.

Coalescing alone does not guarantee peak performance: tile alignment, Tensor Core
fragment layout, cache reuse, and bank conflicts also matter.

## 4. Write the result only at the end

The accumulator begins as a FP32 zero tile:

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
```

Every K tile updates it on-chip. We never write partial sums to global memory.
Only after the K loop finishes do we execute one masked `tl.store`. The store
converts FP32 accumulated results to BF16/FP16 output storage.

Writing partial C after every K tile would add global writes, later reads, and
synchronization—destroying much of GEMM's reuse advantage.

## 5. Vectorized operations and asynchronous movement

The kernel expresses whole tiles rather than scalar Python loops:

```python
a = tl.load(A tile)
b = tl.load(B tile)
accumulator += tl.dot(a, b)
```

`tl.dot` maps suitable FP16/BF16 tile work toward Tensor Core operations. The
only source loop walks K tiles; every operation inside it is vectorized over a
tile. `num_warps` chooses cooperative warp count, while `num_stages` gives the
compiler room to overlap tile movement and computation.

Again, vectorized source is visible; actual Tensor Core and async-copy use is a
profiling/assembly question. We will check both performance counters and codegen.

## Accuracy

Inputs and output use FP16 or BF16, but partial dot products accumulate into an
FP32 tile. Tests compare against `torch.mm` across power-of-two and irregular
shapes. GEMM results need tolerances because floating-point reductions may sum in
different orders.

## Tuning matrix

The benchmark changes:

```text
BLOCK_M: 32, 64, 128
BLOCK_N: 32, 64, 128
BLOCK_K: 32, 64
num_warps: 4, 8
num_stages: 2, 3, 4
```

It reports latency and:

```text
TFLOP/s = (2 × M × N × K) / seconds / 1e12
```

The factor 2 counts one multiply and one addition. Different shapes can prefer
different configurations, so one globally fixed tile is only a starting point;
production kernels use shape-dependent autotuning.

## Run on the Pod

```bash
source /workspace/gpu-kernel-lab/.venv/bin/activate
cd /workspace/gpu-kernel-lab/learning/06_gemm

pytest -q
python benchmark.py --dtype bfloat16 --output ../../results/gemm_bf16_rtx3090.csv
python benchmark.py --dtype float16 --output ../../results/gemm_fp16_rtx3090.csv
```

## Evidence checklist

- Are A/B global loads coalesced?
- Does each tile reuse loaded operands through `tl.dot`?
- Is C accumulated in FP32 and stored only once?
- Which tile/warp/stage combination wins for each shape?
- Do deeper stages help latency or reduce occupancy?
- How many registers and shared-memory bytes does each program use?
- Are there local-memory register spills?
- Does generated code use Tensor Cores?
- Does it use asynchronous copies on this GPU/compiler combination?
- Does every irregular-shape accuracy test pass?

We will answer the last five with the real RTX 3090, not assumptions.

