# The Ultimate GPU & Model Optimization Report

> From bits to billing: how GPUs work, why inference is slow, how to make it
> absurdly fast, how to ship it, and how to get paid for it.

**Author note:** This is written to be read cover-to-cover OR jumped around.
Every part stands alone. Real numbers throughout come from this repo
(`results/baseline.md`, `results/sweep.md`) — a Llama-3.2-3B model optimized
on an NVIDIA A40. When you read "we measured X", it happened on real hardware.

---

## Table of Contents

- [Part 1 — How a GPU Actually Works (the mental model)](#part-1--how-a-gpu-actually-works-the-mental-model)
- [Part 2 — The One Idea That Matters: Memory-Bound vs Compute-Bound](#part-2--the-one-idea-that-matters-memory-bound-vs-compute-bound)
- [Part 3 — What Breaks and Where to Focus](#part-3--what-breaks-and-where-to-focus)
- [Part 4 — LLM Inference, Decoded](#part-4--llm-inference-decoded)
- [Part 5 — The Optimization Stack + Trade-Offs](#part-5--the-optimization-stack--trade-offs)
- [Part 6 — Multi-Architecture Portability](#part-6--multi-architecture-portability)
- [Part 7 — Multi-GPU and Big/Small Models](#part-7--multi-gpu-and-bigsmall-models)
- [Part 8 — The Freelancing Playbook](#part-8--the-freelancing-playbook)

---

# Part 1 — How a GPU Actually Works (the mental model)

## 1.1 The core mental model: a GPU is a factory with a slow supply line

Forget everything you've heard about "GPU = fast math". Here is the mental
model that will get you 90% of the way:

> A GPU is thousands of tiny workers (threads) that can do math very fast,
> but they're all waiting on one slow conveyor belt that delivers raw
> materials from far away (the memory).

The workers are **never** the bottleneck. The conveyor belt is. This one
sentence explains almost every optimization you will ever do.

Let me make it concrete with the A40:

- It can do roughly **149 TFLOP/s** of fp16 math (tera = trillion).
- It can move memory at roughly **696 GB/s** (gigabytes per second).

Ratio: `149,000 / 696 ≈ 214`. The GPU can do ~214 math operations for every
single byte it fetches. That means: **if your kernel does fewer than ~214
operations per byte it loads, the memory is the bottleneck, not the math.**
(We'll formalize this in Part 2 — it's called the roofline model.)

## 1.2 The memory hierarchy (bits up)

Data lives at different "distances" from the workers. Closer = faster but
smaller. This is the single most important table in GPU computing:

| Level | Size (A40) | Bandwidth | Latency | Who controls it |
|---|---|---|---|---|
| **Registers** | ~256 KB per SM | insane | ~1 cycle | compiler (mostly automatic) |
| **Shared memory / SRAM** | up to 228 KB per SM | ~19 TB/s | ~20 cycles | **you** (CUDA: `__shared__`; Triton: `tl.load` into blocks) |
| **L1/L2 cache** | ~40 MB L2 | high | ~200 cycles | hardware (automatic) |
| **HBM (VRAM)** | 45 GB | 696 GB/s | ~400 cycles | **you** (when you read weights/activations) |
| **CPU RAM** | 503 GB | ~50 GB/s (PCIe) | ~1000s cycles | you (offloading) |
| **Disk/NVMe** | TBs | ~5 GB/s | milliseconds | you (checkpoint loading) |

The rule that falls out of this table:

> **The farther data travels, the more you pay.** Every optimization is about
> (a) moving data to a closer tier, or (b) moving *less* data total.

When people say "fused kernel", what they really mean is "do the whole
computation in the fast tiers (registers + shared memory) and write the
result to HBM exactly once, instead of five times."

## 1.3 How the workers are organized (SMs, warps, blocks)

- **SM (Streaming Multiprocessor)**: a group of workers sharing one shared
  memory and one L1 cache. The A40 has ~84 SMs. Think of an SM as one
  "factory floor".
- **Thread**: the smallest worker. Executes one instruction on one piece of
  data.
- **Warp**: a group of **32 threads that always move together**. A warp is
  the real unit of execution — the GPU executes a whole warp at once (SIMT:
  single instruction, multiple threads). You almost never think in threads;
  you think in warps.
- **Block**: a group of warps (e.g. 256 threads = 8 warps) that share one
  shared-memory region and can synchronize with each other. Blocks are the
  unit that gets scheduled onto SMs.
- **Grid**: the whole collection of blocks for one kernel launch.

Why this matters: if your block is 256 threads but the work doesn't divide
evenly into 32-thread chunks, some threads idle. If two threads in a warp
take different branches (an `if` statement), the warp runs *both* branches
serially — this is **warp divergence** and it halves (or quarters) your speed.

## 1.4 Occupancy and latency hiding — why more workers = faster (to a point)

Memory latency is ~400 cycles. If a warp asks for data and just *waits*, it
wastes 400 cycles doing nothing. GPUs survive this via **latency hiding**:

> While warp A is waiting for memory, the SM immediately switches to warp B,
> which has already received its data and is ready to run.

This only works if there are *enough warps resident on the SM*. That's
**occupancy** — the fraction of the SM's maximum warps you have live.

- Low occupancy → the SM sits idle waiting on memory → slow.
- High occupancy → the SM always has a ready warp → memory latency hidden.

The catch: more resident warps means each warp gets fewer registers and less
shared memory (both are fixed per SM). This is the **register-pressure
trade-off**: using tons of registers makes each thread smarter but lets fewer
warps fit, which can *lower* occupancy and make you slower. This is why
tuning kernel block size / register count is an art, not a lookup.

## 1.5 What CUDA controls vs what Triton controls

This is the "teach me the difference" part. Both ultimately compile to the
same GPU machine code (PTX/SASS), but they expose **different levels of
abstraction**, and that changes what you are responsible for.

### CUDA C++ (low-level, full control)

You explicitly manage:

| Concept | CUDA exposes | You must get right |
|---|---|---|
| Threads/blocks/grid | `blockIdx`, `threadIdx`, `blockDim`, `gridDim` | correct indexing math |
| Shared memory | `__shared__ float x[256]` | sizing, bank conflicts |
| Synchronization | `__syncthreads()` | ordering, deadlocks |
| Registers | implicit, hint via `__launch_bounds__` | occupancy tuning |
| Memory layout | raw pointers | coalescing (Part 3) |
| Streams/concurrency | `cudaStream_t` | overlap compute/memcpy |

**Upside:** you can extract every last % of performance. You control the exact
block size, the exact shared-memory layout, the exact memory access pattern.
**Downside:** you own every bug (race conditions, bank conflicts, occupancy).
A CUDA kernel that's wrong silently produces garbage or hangs the GPU.

### Triton (higher-level, compiler does the hard part)

Triton (OpenAI's Python DSL) gives you a **block-based** mental model:

- You write a **`@triton.jit` function** that describes what *one block* does.
- You write `tl.load(ptr)` / `tl.store(ptr)` and Triton figures out coalescing.
- You write a **grid = (num_blocks,)** — Triton maps blocks to SMs/warps for you.
- You mark loops with `tl.range(..., num_stages=N)` to get software pipelining
  (automatic prefetch overlap) — you never hand-manage streams.
- You can `@triton.autotune(...)` over block sizes and Triton tries them all
  at runtime and picks the fastest.

**What Triton hides from you:** thread/warp indexing, shared-memory allocation
and bank-conflict avoidance, register allocation, launch configuration.

**What Triton still lets you control:** block size (via autotune), memory
tiling (`tl.load` shapes), pipelining depth, and crucially the *algorithm*
(tiling strategy, which is where 90% of performance comes from).

### The practical rule

> Use **Triton** for ~95% of work — it gets you 80–95% of hand-tuned CUDA
> speed at 5x the development speed, and it's **portable across architectures
> automatically** (Part 6). Drop to **CUDA C++** only when you've measured a
> specific kernel that Triton can't optimize (rare) or you need to exploit a
> hardware feature Triton doesn't expose (e.g. exotic tensor-core modes).

---

# Part 2 — The One Idea That Matters: Memory-Bound vs Compute-Bound

## 2.1 Arithmetic intensity

Define it. For any kernel:

```
arithmetic_intensity = (FLOPs performed) / (bytes moved to/from HBM)
```

- **High arithmetic intensity** (> ~100 FLOP/byte on A40): the math is the
  bottleneck → **compute-bound**. The GPU's tensor cores are saturated.
- **Low arithmetic intensity** (< ~20 FLOP/byte): the memory is the bottleneck
  → **memory-bound**. The math cores sit mostly idle waiting for bytes.

Most people *assume* everything is compute-bound (because "GPUs are fast at
math"). In reality, **almost all LLM inference is memory-bound**, which is why
the naive approach leaves the GPU at 50% utilization.

## 2.2 The roofline model (one chart that explains everything)

Plot throughput (FLOP/s) vs arithmetic intensity (FLOP/byte):

```
            ▲
 throughput │          roof (compute ceiling, e.g. 149 TFLOP/s)
   (log)    │        ╱
            │      ╱   ← diagonal "bandwidth wall": y = BW × intensity
            │    ╱
            │  ╱
            │╱
            └──────────────────────────►  arithmetic intensity (log)
```

- Below the diagonal: you're **memory-bound** (moving bytes). Moving less data
  (quantize, fuse) is the only way up.
- Above the diagonal, hitting the flat roof: you're **compute-bound**. Better
  math kernels (tensor cores) is the only way up.

The trick: for a given kernel, compute its intensity, find where it sits, and
you instantly know the *only* lever that matters. This is the "read the
profile and know what to do" skill.

## 2.3 Why LLM decode is always memory-bound

During generation (decoding), the model produces **one token at a time**. To
produce that single token, it must:

1. Read **all the model weights** from HBM (6.4 GB for our fp16 3B model).
2. Read the KV cache for the current sequence.

How much math does it do? Roughly `2 × params` FLOPs = ~6.4 GFLOP for 3B.

```
intensity = 6.4 GFLOP / 6.4 GB = 1 FLOP/byte
```

**1 FLOP/byte.** That's *catastrophically* low — deep in the memory-bound
region, 200x below the ~214 FLOP/byte break-even. This is why:

> The single-token decode throughput is *entirely* set by memory bandwidth:
> `tokens/s ≈ bandwidth / (bytes per token) ≈ 696 GB/s / 6.4 GB ≈ 108 tok/s`.

We measured **44 tok/s** on naive PyTorch — about 40% of that ceiling, because
naive PyTorch also wastes bandwidth on unfused ops and misses caching. The
gap between 44 and 108 is what Part 5's optimizations claw back.

## 2.4 Why prefill is compute-bound (and thus very different)

**Prefill** = processing the input prompt (all tokens at once, in parallel).
Here the model does `batch × seq_len` tokens of work in one big matrix
multiply. Arithmetic intensity scales with batch/sequence, so prefill
becomes **compute-bound** — the tensor cores finally get to shine.

This is why our sweep showed:

- `input_len=1024, output_len=64, batch=64` → **48,608 tok/s**
- `input_len=8, output_len=256, batch=64` → **3,588 tok/s**

Same model, same GPU. The long-input case is compute-bound (huge parallel
GEMMs), the long-output case is memory-bound (token-by-token decode). Two
different bottlenecks, two different fixes — and you must know which is which
or you'll optimize the wrong thing.

---

# Part 3 — What Breaks and Where to Focus

This is the "what an engineer should be paranoid about" section. Each item:
what it is, why it kills you, how to spot it, how to fix it.

## 3.1 Kernel launch overhead

Every kernel launch has a fixed cost (a few microseconds, CPU→GPU handoff). A
single LLM decode step can be **hundreds of tiny kernels** (one per norm, per
residual add, per activation). For a small model like 3B, the launch overhead
can *exceed* the actual compute.

- **Spot it:** Nsight Systems timeline shows a "picket fence" of tiny gaps
  between kernels; GPU utilization dips between launches.
- **Fix it:** (a) fuse kernels (fewer launches), (b) **CUDA graphs** — capture
  the whole decode step into one graph and replay it in a single submission.
  We saw this give ~10% free throughput (2988 → 3271 tok/s).

## 3.2 Uncoalesced memory access

A warp of 32 threads reading memory is **fast only if** the 32 threads read
32 *consecutive* bytes/words (one cache line) — this is **coalescing**. If
thread 0 reads address 0, thread 1 reads address 1000, thread 2 reads 2000,
the GPU issues ~32 separate memory transactions instead of 1 → up to 32x
slower.

- **Fix:** layout data so adjacent threads touch adjacent memory. Triton's
  `tl.load` on a contiguous block does this automatically; in raw CUDA it's
  on you.

## 3.3 Shared-memory bank conflicts

Shared memory is split into ~32 "banks". If two threads in a warp read from
the *same bank but different addresses* simultaneously, the request is
serialized (2-way, 4-way, ... conflict). Worst case 32-way = 32x slowdown.

- **Fix:** pad shared-memory arrays (e.g. `float x[32+1]` instead of `[32]`),
  or use `tl.trans`/swizzled layouts. Triton handles much of this for you.

## 3.4 Register pressure / low occupancy

If your kernel uses too many registers per thread, the SM can't fit many
warps, occupancy drops, and the SM can't hide memory latency (Part 1.4). You
made the kernel "smarter" and accidentally made it slower.

- **Spot it:** Nsight Compute shows "achieved occupancy" low vs theoretical;
  "register count" per thread near the cap (255).
- **Fix:** reduce register usage (simpler expressions, fewer live values), or
  explicitly tune `__launch_bounds__` / autotune block sizes.

## 3.5 Warp divergence

An `if/else` where threads in the *same warp* take different paths forces the
warp to execute both branches serially. Halves throughput at best.

- **Fix:** restructure so branches follow warp boundaries (e.g. branch on
  block/warp id, not per-element conditions), or accept it when unavoidable.

## 3.6 Atomic contention

When many threads `atomicAdd` to the same memory location, they serialize.
This is how you'd naively accumulate a sum across a block. Thousands of
threads → thousands of serialized updates → terrible.

- **Fix:** hierarchical reduction — accumulate locally (registers), then
  reduce within warp (shuffle instructions), then within block (shared
  memory), then one atomic per block.

## 3.7 Synchronization stalls

`__syncthreads()` forces all warps in a block to wait for the slowest. Too
many barriers = everyone waiting = idle SMs.

- **Fix:** minimize barriers, use independent tiling, let warps proceed
  independently where correctness allows.

## 3.8 Precision traps (this one is sneaky)

- **fp16** has limited range: values > 65504 overflow to infinity. Raw fp16
  training/inference often breaks unless you scale (loss scaling) or use a
  scheme that moves the magnitude into a scale factor.
- **bf16** has the *same range as fp32* (8-bit exponent) but only 7 mantissa
  bits — less precise but rarely overflows. That's why bf16 "just works" where
  fp16 needs care.
- **int8** has tiny range (−128..127). Activations with outlier channels
  destroy accuracy unless you **scale per-channel/tensor** (SmoothQuant — see
  Part 5).
- **fp8** (e4m3/e5m2) — new, hardware-accelerated on Ada/Hopper, but *not* on
  Ampere (A40/A100/3090), which is a frequent trap: FP8 kernels silently need
  INT8 fallback on older cards.

## 3.9 The engineer's prime directive: MEASURE FIRST

You will guess wrong. Every time. So:

1. **Profile** before optimizing — `nvidia-smi dmon`, `torch.profiler`, Nsight
   Systems (`nsys`, timeline), Nsight Compute (`ncu`, per-kernel stats).
2. **Find the actual bottleneck** — is it memory-bound or compute-bound
   (Part 2)? Which kernel eats the time?
3. **Fix only that**, re-measure, repeat.

> Heads-up from our repo: on containerized cloud GPUs (RunPod etc.), Nsight
> Compute often fails with `ERR_NVGPUCTRPERM` because the container lacks
> `CAP_SYS_ADMIN` and the host sets `RmProfilingAdminOnly=1`. **Nsight Systems
> works fine**, and `torch.profiler` gives kernel-level timing as a fallback.
> Don't burn a day fighting it — use the fallback.

---

# Part 4 — LLM Inference, Decoded

## 4.1 The transformer's moving parts

A decoder-only LLM (Llama, GPT, etc.) is a stack of identical **layers**.
Each layer has:

1. **Attention** (the "read" part):
   - Project input into **Q** (query), **K** (key), **V** (value) via weight matrices.
   - Compute `softmax(Q·Kᵀ/√d)·V` — each token looks at all previous tokens.
2. **MLP** (the "think" part): usually **SwiGLU** — a gated feed-forward with
   three weight matrices (gate, up, down) and an activation.
3. **RMSNorm** (normalize) and **RoPE** (rotary position encoding) — small
   but numerous elementwise ops (cheap compute, but they create memory
   round-trips if not fused).

Vocabulary size is huge (128K tokens for Llama-3.2), so the **final output
projection** (hidden → vocab) is one of the largest matrices in the model.

## 4.2 The two phases

- **Prefill**: ingest the prompt. All prompt tokens processed in parallel.
  Compute-bound (Part 2.4). Latency matters (this is your **TTFT** —
  time-to-first-token).
- **Decode**: generate one token at a time, each conditioned on all previous.
  Memory-bound (Part 2.3). Throughput and **ITL** (inter-token latency) matter.

## 4.3 The KV cache (and why it grows)

In attention, each token needs the K and V of every *previous* token. Rather
than recompute them every step, you **cache** them. This KV cache grows
linearly with sequence length and is read back on every decode step.

```
KV cache size ≈ 2 (K and V) × num_layers × num_kv_heads × head_dim × seq_len × bytes
```

For long sequences this can exceed the model weights themselves. Managing it
(PagedAttention, KV quantization) is a first-class optimization (Part 5).

## 4.4 Why the whole game is the decode loop

Prefill is "easy" (parallel GEMMs, compute-bound, fast). Decode is the problem:
one token at a time, memory-bound, and every token re-reads all weights. Any
optimization that doesn't improve *decode* is mostly cosmetic for LLMs. Keep
saying this to yourself and you won't waste time optimizing prefill.

---

# Part 5 — The Optimization Stack + Trade-Offs

Here's the honest, measured order of impact for LLM inference, from biggest
to smallest. Our A40 / Llama-3.2-3B numbers prove it.

## 5.1 Lever #1: Continuous batching (the 50–1000x lever for throughput)

**The problem:** a single stream uses ~40% of bandwidth (44 tok/s of a ~108
ceiling). The GPU idles waiting on one request.

**The fix:** serve many requests at once, and — crucially — **continuously
refill** finished slots every step (not wait for the whole batch to finish).
Now the bandwidth that one stream couldn't use gets consumed by other streams.

- vLLM does this with **PagedAttention** (KV cache in fixed-size pages, like
  OS virtual memory — kills fragmentation, fits far bigger batches).

**Our measured result:**
| Batch | Throughput | vs 44 tok/s baseline |
|---|---|---|
| 1 | 75 tok/s | 1.7x |
| 8 | 564 tok/s | 12.8x |
| 64 | 3588 tok/s | 81x |

This single lever blows past any "5-7x" target. If a client says "make it
faster", this is 90% of the answer. Quantization and kernels are the other 10%.

## 5.2 Lever #2: Quantization (halve the bytes = ~2x the memory-bound speed)

Decode is memory-bound (1 FLOP/byte), so halving weight bytes (~2x) directly
doubles decode speed *and* halves VRAM (bigger batches → even more throughput).

The formats, simplest to trickiest:

| Scheme | Bits | Notes | Hardware |
|---|---|---|---|
| **W8A8 INT8** | 8 | weights + activations both int8; needs **SmoothQuant** to survive activation outliers | Ampere+ (int8 tensor cores) |
| **INT8 weight-only** | 8 | simplest; dequant on the fly | everywhere |
| **FP8 (e4m3)** | 8 | hardware-accelerated, low effort | Ada (40xx) / Hopper (H100) only |
| **GPTQ** | 4 | one-shot weight quant with error correction | needs calibration |
| **AWQ** | 4 | keeps ~1% of salient channels in fp16 | Ampere+ |

**SmoothQuant** (the W8A8 enabler): LLM activations have outlier *channels*
(huge values). Naively quantizing them to int8 wrecks accuracy. SmoothQuant
**moves the outlier magnitude from the activations into the weights** via
per-channel scales, making both quantizable. This is the key idea behind "8-bit
that still works".

**Trade-off table (the honest version):**
| Goal | What you pay |
|---|---|
| Max speed (4-bit) | some quality loss; needs calibration; harder to debug |
| 8-bit (near-lossless) | ~2x less speed than 4-bit, but ~2x faster than fp16 |
| fp16 (zero quality loss) | max memory/bandwidth; slowest |
| fp8 | only on newest GPUs — check the card first |

> We measured INT8 weight-only on A40 = **2964 tok/s vs 3271 fp16** at batch
> 64 — *slightly slower*. Reason: in batched (compute-bound-ish) mode, the
> dequant step cost more than the saved bandwidth. Quantization's speed win is
> biggest for **memory-bound single-stream decode**, not batched throughput.
> Know *which* regime you're in before promising a speedup.

## 5.3 Lever #3: FlashAttention + kernel fusion (kill wasted round-trips)

- **FlashAttention**: computes attention in SRAM tiles, never writing the full
  N×N attention matrix to HBM → O(N) instead of O(N²) memory. Huge for long
  contexts. (Our profile already showed `flash_fwd_kernel` — PyTorch ships it.)
- **Fusion**: naive PyTorch runs RMSNorm, RoPE, SiLU, residual adds as
  separate kernels, each a round-trip to HBM. Our profile showed **~15% of GPU
  time in these tiny elementwise/copy ops**. Fusing them into one kernel
  removes those round-trips.

## 5.4 Lever #4: CUDA graphs (kill launch overhead)

Capture the entire decode step (hundreds of kernels) into one graph, replay
with one launch. Especially impactful for small models where per-kernel launch
overhead dominates. We measured **+10% throughput** (2988 → 3271 tok/s).

## 5.5 Lever #5: Speculative decoding (latency, not throughput)

A tiny "draft" model proposes N tokens, the big model verifies them in one
parallel pass. Speeds up **single-stream latency** (fewer serial steps), but
does little for batched throughput. Use it when a client complains about
"first token feels slow" or "words trickle out", not when they want max
requests/sec.

## 5.6 The stack, summarized as a decision tree

```
Client wants faster inference?
│
├─ Many concurrent users / throughput?  → continuous batching (vLLM) FIRST
│     └─ still not enough? → quantize to 8-bit → int8/4-bit
├─ Single user, latency?               → speculative decoding + CUDA graphs
│     └─ still not enough? → quantize (memory-bound decode)
├─ Long documents / huge context?      → FlashAttention + KV-cache quant
└─ Fitting a big model on small GPU?   → quantize + offload (CPU/disk)
```

## 5.7 The grand trade-off triangle

You can't have all three. Pick two:

- **Speed** (lowest latency / highest throughput)
- **Quality** (no accuracy loss)
- **Simplicity** (easy to build, deploy, and maintain)

- Fast + Simple → quantized, off-the-shelf engine (vLLM + INT8). Small quality loss.
- Fast + Quality → bespoke kernels, aggressive fusion, careful calibration. Hard to build.
- Simple + Quality → fp16 + standard engine. Slowest.

Your job as the optimizer is to tell the client *which corner* they're paying
for. The good ones respect you more when you're honest about this.

---

# Part 6 — Multi-Architecture Portability

## 6.1 The problem: "works on my GPU" is a trap

You tune a kernel for an A100 (Ampere, 108 SMs, 164 KB shared mem) and ship it.
The client runs it on a 3060 (also Ampere, but 28 SMs, 96 KB shared mem) or a
Hopper H100 (different tensor-core features) — and it's slow, or crashes.

The causes:
1. **Hardcoded block sizes / grid sizes** tuned for one SM count.
2. **Tensor-core feature assumptions** (fp8 exists on Hopper, not Ampere).
3. **Shared-memory size assumptions** (a block needing 128 KB won't even *fit*
   on an SM with 96 KB → launch failure).
4. **Different `num_warps` optimal** per architecture.

## 6.2 Compute capability cheat sheet

`compute capability` (e.g. "8.6") = major.minor. Major = architecture, minor =
revision/features. What tensor cores support, by generation:

| Arch | Capability | Example | fp16 tc | bf16 tc | int8 tc | fp8 tc | tf32 |
|---|---|---|---|---|---|---|---|
| Volta | 7.0 | V100 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Turing | 7.5 | T4, 2080 | ✅ | ❌ | ✅ | ❌ | ❌ |
| Ampere | 8.0/8.6 | A100, A40, 3090 | ✅ | ✅ | ✅ | ❌ | ✅ |
| Ada | 8.9 | 4090, L4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hopper | 9.0 | H100 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Blackwell | 10.0 | B200 | ✅ | ✅ | ✅ | ✅ | ✅ |

The recurring gotcha: **fp8 requires Ada/Hopper+**. Ampere (A100/A40/3090) —
still extremely common in the wild — has **int8 tensor cores but no fp8**. Any
fp8 kernel must have an int8 (or fp16) fallback for Ampere.

## 6.3 How to write portable kernels

1. **Never hardcode block size or grid size.** Query at runtime:
   `cudaDeviceGetAttribute(...)`, `torch.cuda.get_device_capability()`,
   `num_warps`/`num_stages` via autotune.
2. **Use `@triton.autotune`** — it benchmarks a set of candidate configs
   (block sizes, warp counts, stages) on *the actual device* and picks the
   fastest. This is the single biggest portability win: the same source adapts
   to any GPU.
3. **Branch on capability for tensor-core mode**: `if capability >= (8,9): use
   fp8 else: use int8/fp16`. Gate with a runtime flag, not `#ifdef`-style
   hardcoding where possible.
4. **Respect shared-memory limits**: `tl.assume` / dynamic shared mem sizing
   based on `deviceProperties.sharedMemPerBlockOptin`. A block that needs more
   shared mem than the SM has will fail to launch.
5. **Test on a capability matrix** before shipping: at least one of {7.x, 8.x,
   8.6/8.9, 9.0}. Even a 3060 + A100 + H100 covers the important cases.

## 6.4 Triton vs CUDA for portability

- **Triton**: write once, `autotune` adapts per-arch, the Triton compiler
  emits per-arch code. Best default for portability.
- **CUDA C++**: fastest, but you must hand-tune per arch (or use CUTLASS,
  which does arch-specific dispatch for you — it's the industry-standard
  kernel library NVIDIA uses internally).

## 6.5 The pragmatic stack for portability

```
Your model/kernel
      │
      ▼
vLLM / TensorRT-LLM (already portable, tuned per-arch)
      │
      ▼
CUTLASS / cuBLAS / FlashAttention (arch-dispatch handled)
      │
      ▼
Triton (your custom kernels, autotuned)
      │
      ▼
CUDA C++ (only the 1 kernel that genuinely needs it)
```

The more you live higher in this stack, the more "multi-architecture" comes
free. Only go lower when a profile proves you need it.

---

# Part 7 — Multi-GPU and Big/Small Models

## 7.1 Why you'd use more than one GPU

1. **The model doesn't fit in one GPU's VRAM.** (The common case for 70B+.)
2. **You want more throughput** than one GPU's bandwidth gives.

## 7.2 The three parallelisms (and when to use each)

| Strategy | How it works | Best for |
|---|---|---|
| **Tensor parallelism (TP)** | Split each layer's weight matrices *across* GPUs; each does a shard of the matmul, then they sync (allreduce). | Inference, low latency, model doesn't fit one GPU |
| **Pipeline parallelism (PP)** | Split the model *by layers* across GPUs; a token flows through like an assembly line. | Very large models; overlaps with TP |
| **Data parallelism (DP)** | Replicate the whole model on each GPU; split the *batch* across them. | Training, or serving many independent requests |

For **inference**, **tensor parallelism** is the default answer when a model
doesn't fit one GPU. Pipeline parallelism adds latency (assembly line), so it's
used for training or extreme sizes.

## 7.3 NVLink vs PCIe — the invisible cost

When you split across GPUs, they must communicate every layer. The link speed
decides if you win or lose:

- **NVLink** (~600 GB/s per link, H100): fast enough that TP scales well.
- **PCIe** (~32–64 GB/s): slow — TP across PCIe GPUs can be *slower* than one
  GPU, because communication exceeds the savings.

> Rule: tensor parallelism only pays off with **NVLink**. If the client's box
> is two GPUs on PCIe, don't promise TP speedups — check the interlink first.
> (In the cloud, ask: "do these GPUs have NVLink, or are they PCIe?")

## 7.4 Sizing math (do this in your head — Part 8 has the full table)

**Step 1 — weights:**
```
VRAM_weights = params × bytes_per_param
  fp16/bf16: ×2   |   fp32: ×4   |   int8: ×1   |   4-bit: ×0.5
```
- 7B fp16 = 14 GB
- 70B fp16 = 140 GB (needs 2×80GB or 3×48GB)
- 405B fp16 = 810 GB (needs 8×H100/4×B200)

**Step 2 — KV cache (per request):**
```
KV_bytes = 2 × layers × kv_heads × head_dim × seq_len × bytes
```
This scales with *context length and batch*, often dwarfing weights at long
contexts.

**Step 3 — total VRAM = weights + KV cache × batch + overhead (~10–20%).**

## 7.5 Small models (1–8B): one GPU usually wins

Small models fit easily in one GPU. Adding a second GPU for TP almost never
helps (communication cost > savings). For small models the levers are
**quantization + batching**, not multi-GPU. Our whole A40/3B repo is this case.

## 7.6 The mental rule

> If `params × bytes` fits in one GPU's VRAM with room for KV cache → stay on
> one GPU, optimize with batching + quantization. If it doesn't fit → tensor
> parallelism, and verify NVLink exists or it'll disappoint.

---

# Part 8 — The Freelancing Playbook

## 8.1 What clients are actually buying

Clients don't say "optimize my model." They say one of these (translate each
to the *real* need):

| Client says | They actually mean |
|---|---|
| "It's too slow" | lower latency (TTFT/ITL) or higher throughput — find out which |
| "It costs too much to run" | fewer GPU-hours = smaller/cheaper GPU, or faster so they can downsize |
| "The model doesn't fit" | quantization or multi-GPU, or a smaller model |
| "It gives bad answers" | quality problem, not speed — do NOT quantize aggressively |
| "We need it in production" | reliability, containers, monitoring, not just raw speed |

**The single most important freelancing skill: ask which metric matters before
touching anything.** Speed ≠ quality ≠ cost. You'll waste days otherwise.

## 8.2 The GPU-sizing mental math (make this automatic)

Memorize these and you can answer "how many GPUs for X?" in your head in 10
seconds:

**Memory (VRAM), per param:**
| Precision | GB per 1B params | 7B | 13B | 70B | 405B |
|---|---|---|---|---|---|
| fp32 | 4 | 28 | 52 | 280 | 1620 |
| fp16/bf16 | 2 | 14 | 26 | 140 | 810 |
| int8 | 1 | 7 | 13 | 70 | 405 |
| 4-bit | 0.5 | 3.5 | 6.5 | 35 | 203 |

**Common GPU VRAM:** A40/A100/4090 = 48/80/24 GB, H100 = 80 GB, B200 = 192 GB,
L4 = 24 GB, T4 = 16 GB, 3090 = 24 GB, 3060 = 12 GB.

**Throughput ceiling (decode, memory-bound):**
```
tok/s ≈ bandwidth / (params × bytes)
```
- 3090 (936 GB/s), 7B fp16: 936/14 ≈ 67 tok/s (single stream).
- A40 (696 GB/s), 3B fp16: 696/6.4 ≈ 108 tok/s. (We measured 44 naive, 75 vLLM.)

**The "how many GPUs" answer:**
1. `VRAM = params × bytes + KV cache + overhead`. If ≤ one GPU → 1 GPU.
2. If not → `ceil(total / single_GPU_VRAM)` GPUs, tensor-parallel, and check
   NVLink.

## 8.3 The cost-savings pitch (what to tell the client)

Clients buy money saved or money made. Quantify it:

- "Your 7B runs on 2×A100 ($8/hr). With 8-bit quant it fits on 1×A100 ($4/hr)
  and runs 1.8x faster → **~60% cost cut** with <2% quality loss."
- "You pay $2000/mo for the big GPU. I'll get 4-bit inference on a $400/mo
  tier → **save $1600/mo, pay for itself in a week.**"
- "Your app takes 3s to answer → users churn. I'll cut TTFT to 0.5s with
  speculative decoding → higher retention → more revenue."

Always frame in **$/month saved** or **revenue gained**, not "I reduced
FLOPs by 40%." The client cares about money.

## 8.4 Common client problems and the one-line fix

| Problem | Diagnosis | Fix |
|---|---|---|
| Slow responses (single user) | memory-bound decode, no speculation | CUDA graphs + speculative decoding + INT8 |
| Can't handle many users | no continuous batching | vLLM/TensorRT-LLM + PagedAttention |
| OOM crash | model + KV cache > VRAM | quantize, or cap context, or multi-GPU |
| Huge GPU bill | over-provisioned | quantize to smaller/cheaper GPU |
| Long documents fail | KV cache blowup at long context | FlashAttention + KV-cache quant + chunked prefill |
| Bad answers after "optimizing" | over-aggressive 4-bit | back off to 8-bit / SmoothQuant, re-eval perplexity |
| Works locally, dies in prod | env mismatch, no container | Docker + pinned deps + a smoke test |

## 8.5 Project types, by price band

- **$500–1500**: quantize a model (8/4-bit), set up vLLM serving, basic
  benchmark + report. 1–3 days.
- **$1500–4000**: full serving stack (continuous batching, PagedAttention,
  autoscaling, containers, monitoring) + optimization report + docs. 1–2 weeks.
- **$4000–7000+**: custom kernel work (Triton/CUDA), speculative decoding
  integration, multi-GPU tensor-parallel setup, bespoke latency targets,
  ongoing retainer. 2–6 weeks.

## 8.6 How to scope, price, execute, deliver (the process)

1. **Discovery call** — nail the metric (latency vs throughput vs cost), the
   model size, the hardware, the budget. Get a baseline number *before*
   quoting ("send me a sample request and I'll profile it").
2. **Proposal** — scope + your measured baseline + target + deliverable +
   fixed price. Always include: "quality gate" (e.g. perplexity delta ≤ 2%) so
   the client can verify you didn't trade accuracy for speed.
3. **Execute** — profile → optimize biggest lever first (Part 5) → re-measure
   → iterate. Keep the client updated with before/after numbers.
4. **Deliver** — a report with the before/after table, the trade-offs you made,
   reproducible setup (container + one-command run), and a walkthrough. The
   report *is* your product; it justifies the invoice.
5. **Upsell** — "want me to keep it fast as your traffic grows? retainer for
   $X/mo."

## 8.7 Red flags (when to walk away)

- Client can't articulate the metric ("just make it faster" with no number).
- Client expects a 100x speedup with zero quality loss (violates the triangle).
- No access to the target hardware ("optimize it on your machine, we'll trust
  you" — you need their GPU to be sure).
- Scope creep without re-quote.

## 8.8 The mindset: become the "optimization person"

After this report you should be able to, on a call, *in your head*:

- Estimate VRAM for any (model, precision) → how many GPUs.
- Estimate decode tok/s from bandwidth → whether it's memory-bound.
- Know the lever order (batching > quant > fusion > graphs > speculative).
- State the trade-off (speed vs quality vs simplicity) and which corner the
  client wants.
- Convert all of it into $/month saved.

That's the whole game. Everything else is practice.

---

## Final word

GPU optimization is 80% *knowing what matters* (memory, not math) and 20%
*technique* (quantization, fusion, batching). Most people burn weeks on the
20% and never learn the 80%. Now you know both — and you know how to get paid
for it.

**Go make something fast.**
