# Lesson 1: Vector Addition

Vector addition is deliberately simple: it lets us study GPU execution without
the mathematics hiding what the hardware is doing.

## Piece 1 — Operation and shape

```text
x      = [1, 2, 3, 4]
y      = [5, 6, 7, 8]
output = [6, 8, 10, 12]

output[i] = x[i] + y[i], for i from 0 through N - 1
```

Every element is independent, so many GPU lanes can calculate simultaneously.

- **Shape**: the dimensions; 1,000 values have shape `(1000,)`.
- **Size / `numel`**: total element count, here `1000`.
- **dtype**: bytes and number format; FP32 uses 4 bytes, BF16/FP16 use 2.
- **stride**: memory steps between logical neighbors. A contiguous vector has
  stride `(1,)`.
- **layout**: the physical arrangement. This first kernel accepts contiguous 1D
  tensors, making logical neighbors also memory neighbors.

An FP32 vector with one million elements consumes 4,000,000 bytes (about 3.81
MiB). Shape and byte size are related but are not the same fact.

## Piece 2 — Program, block size, and grid

A Triton **program** is one instance of the kernel. One program handles
`BLOCK_SIZE` consecutive elements. For `N = 10`, `BLOCK_SIZE = 4`:

```text
program 0 -> offsets [0, 1, 2, 3]
program 1 -> offsets [4, 5, 6, 7]
program 2 -> offsets [8, 9, 10, 11]
```

The **grid** tells Triton how many programs to launch:

```text
grid = ceil(N / BLOCK_SIZE)
grid = ceil(10 / 4) = 3
```

In code, `triton.cdiv(N, BLOCK_SIZE)` is ceiling division. Floor division would
miss the final two elements.

Triton's `BLOCK_SIZE` is a compile-time tile of logical elements. It is related
to, but not exactly the same abstraction as, a CUDA thread block. Triton maps
these logical operations onto CUDA warps during compilation.

## Piece 3 — Indexing

Each program finds its elements with:

```python
program_id = tl.program_id(0)
offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

`tl.arange` creates all local lane offsets at once. Adding the program's starting
position converts them into global vector indices.

## Piece 4 — Boundary mask

The final program above contains invalid offsets 10 and 11. We create:

```python
mask = offsets < N
```

The mask is passed to both loads and the store. Inactive lanes perform no memory
access. Without it, irregular shapes could cause out-of-bounds reads/writes,
wrong output, or memory corruption. That is why tests include 127, 129, 1,000,
and 100,003—not just comfortable powers of two.

## Piece 5 — Coalescing

Our neighboring lanes access neighboring addresses:

```text
lane:    0  1  2  3  4 ...
offset:  0  1  2  3  4 ...
```

This is **coalesced memory access**. The GPU can combine requests into efficient
wide memory transactions. A pattern like `0, 16, 32, 48, ...` would waste more
of each transaction. We will measure contiguous versus strided access separately
after this experiment.

## Piece 6 — What block size changes

```text
N = 1,000
BLOCK_SIZE 128 -> grid 8
BLOCK_SIZE 256 -> grid 4
BLOCK_SIZE 512 -> grid 2
```

Small blocks create more programs and scheduling work. Large blocks may waste
masked lanes or increase resource pressure. There is no universal winner, so we
measure 128, 256, 512, and 1024 instead of guessing.

## Piece 7 — Correct CUDA timing

CUDA launches are asynchronous: Python may continue before the GPU finishes.
Plain CPU wall-clock timing can therefore measure only dispatch.

Our benchmark:

1. Warms up the operation, excluding compilation and one-time setup.
2. Records CUDA events on the GPU timeline.
3. Executes multiple inner launches to make tiny durations measurable.
4. Synchronizes the ending event.
5. Repeats and reports the median rather than trusting one sample.

PyTorch uses `torch.add(..., out=output)` so both implementations reuse an output
allocation; otherwise we would unfairly include allocation only in PyTorch.

## Piece 8 — Why it is memory-bound

Each FP32 result requires two 4-byte reads and one 4-byte write: at least 12
useful bytes for only one addition. Arithmetic intensity is therefore `1 / 12 =
0.0833` additions per byte—very low. For large vectors, memory bandwidth usually
limits performance before arithmetic throughput does.

```text
effective GB/s = (12 * N) / elapsed_seconds / 1e9
```

This measures useful bytes, not exact physical DRAM traffic (caches and write
behavior affect that). Small vectors are normally launch/under-utilization bound.
As `N` grows, GB/s should rise and then flatten toward a bandwidth ceiling.

## Piece 9 — Run it on the Pod

```bash
source /workspace/gpu-kernel-lab/.venv/bin/activate
cd /workspace/gpu-kernel-lab/learning/01_vector_add
pytest -q
python benchmark.py --output ../../results/vector_add_rtx3090.csv
```

Correctness must pass before performance counts. Then we will answer:

1. When does launch overhead stop dominating?
2. Which block wins for each shape, and why?
3. What peak effective bandwidth do we observe?
4. How close is Triton to PyTorch for large vectors?
5. Are the differences repeatable?

The first goal is understanding and honest measurement—not claiming a speedup.
