# Post 29: Matching and Beating llama.cpp Prefill

Starting at 22,319 tok/s (0.86× llama.cpp), ending at 29,497 tok/s (1.12×). This post documents the sequence of optimizations, the dead ends, and what actually worked — including the uncomfortable discovery that my earlier profiling was wrong.

## Starting Point

The previous session left GWEN's prefill at 22,319 tok/s (pp565) versus llama.cpp's 25,973 tok/s (pp512). The kernel breakdown from nsys showed three targets:

| Kernel | Time | % |
|--------|------|---|
| DeltaNet prefill | 8.7ms | 31% |
| CUTLASS GEMMs | 9.0ms | 32% |
| Flash attention | 3.2ms | 11% |
| Everything else | 7.4ms | 26% |

I attacked all three.

## Optimization 1: Multi-Query Flash Attention

The existing `kernel_batch_causal_attn` used one 32-thread warp per (head, query_position). At pp512 with 8 heads, that's 4,096 blocks of 32 threads — each independently loading K and V from L2 cache for every key position.

First attempt: **shared memory tiling** (load K/V tiles into shared, reuse across queries). Result: 27% *slower*. Only 128 blocks of 128 threads — terrible occupancy on 70 SMs. The `__syncthreads()` barriers between tiles killed the latency hiding that the old independent-warp design had.

Second attempt: **GQA grouping** (4 Q heads sharing a KV head in one block for L1 reuse). Result: marginal 2% improvement. L2 cache is shared across all SMs, so co-locating warps doesn't reduce L2 traffic.

Third attempt, the one that worked: **multi-query per warp**. Each warp processes 4 consecutive query positions, loading K/V once per key position and reusing across all 4 queries. Combined with GQA grouping (4 Q heads per block). Grid: `(n_kv_heads, ceil(N/4), 1)` with 128 threads.

This reduces L2 traffic by ~4× because 4 queries share every K/V read. Attention went from 3.2ms → 2.05ms.

## Optimization 2: Pre-Normalized DeltaNet Q/K

Profiled with `ncu` (needed `sudo` for GPU performance counters) and found the DeltaNet prefill kernel had:
- **long_scoreboard: 6.83** — dominant stall, waiting for memory loads
- **short_scoreboard: 5.87** — shuffle reduction dependency chains

The kernel was doing 12 warp reductions per token (6 for Q L2-norm, 6 for K L2-norm), while llama.cpp's equivalent does only 2 (just kv_col and attn_col dot products). They pre-normalize Q and K before the recurrence kernel.

Solution: `kernel_l2_normalize_qkv_batch` — a separate kernel that normalizes Q and K within the interleaved QKV tensor. One warp per (token, head), two L2 norms per warp. The DeltaNet kernel then reads pre-normalized values.

Cost: 0.15ms for the normalization kernel. Savings: 2.3ms from the DeltaNet kernel (7.7ms → 5.4ms). Net: -2.15ms.

## The DeltaNet Gap That Wasn't

The HANDOFF from the previous session claimed llama.cpp's DeltaNet kernel was 2.5× faster than ours. I spent time trying to close this gap — shared memory tiling (regressed), pointer-increment optimization (no change), state layout transposition (not needed).

Then I actually profiled both kernels with ncu side-by-side:

| Metric | GWEN | llama.cpp |
|--------|------|-----------|
| Duration | 364 µs | 382 µs |
| Executed IPC | 1.46 | 1.39 |
| long_scoreboard | 6.83 | 7.66 |
| L2 Hit Rate | 95.62% | 94.89% |

**Our kernel was already 5% faster.** The "2.5× gap" was a measurement error from the previous session — comparing different prompt lengths or including overhead outside the kernel. The nsys totals confirmed it: GWEN 5.35ms vs llama.cpp 5.47ms for 18 DeltaNet layers at pp500.

Lesson: profile before optimizing. And compare apples to apples.

## Optimization 3: CUTLASS Tile Tuning

The GEMMs used a single 128×128×32 tile configuration. At pp512, the N dimension only produces 4 tiles (512/128), leaving small GEMMs underutilizing the 70 SMs.

Added a 128×64×32 tile variant. Auto-select based on CTA tile count: GEMMs with fewer than 140 tiles (most of them at N=512) use the narrower tile, doubling the N-dimension parallelism.

CUTLASS GEMMs: 7.94ms → 6.35ms (-20%). All of it came from better SM utilization on the smaller GEMMs (M=1024, M=2048).

## Optimization 4: MMA Flash Attention (from llama.cpp)

After the multi-query optimization, flash attention was still 2.05ms — profiled with ncu to find out why:

| Metric | GWEN (scalar) | llama.cpp (MMA) |
|--------|---------------|-----------------|
| Duration | 437 µs/call | 48 µs/call |
| Issued Instructions | 58.3M | 5.4M |
| Registers/thread | 108 | 230 |

**10.8× more instructions.** Our kernel used scalar FMA (2 FLOPs/instruction). Theirs uses `mma.sync.m16n8k16` tensor cores (2048 FLOPs/instruction). No amount of tiling or batching can close a 10× instruction count gap.

Ported llama.cpp's `flash_attn_ext_f16` kernel directly. The kernel code is header-only templates — compiled against their headers via include path, no linking. The 4,300 lines of llama.cpp flash attention code breaks down as:
- 6 GPU architectures × 7 head dimensions × 5 quant formats = generality we don't need
- The actual D=256 Turing path we use: ~600 lines

Challenges:
- **Config mismatch**: I initially assumed nthreads=128, nstages=2 (from the Ampere config). SM_120 with ncols2=1 actually uses nthreads=64, nstages=0. Getting the shared memory size wrong caused illegal memory access.
- **Buffer aliasing**: First version reused the same F32 buffer for Q input and attention output — race condition between blocks reading Q and writing output.
- **Q type conversion**: The kernel expects F32 Q. Added FP16→F32 and F32→FP16 conversion kernels around the call (54µs overhead total).

Result: flash attention 2,050µs → 694µs for 6 layers (2.95× faster).

For context, llama.cpp's flash attention takes 445µs for the same 6 layers — we're at 1.56× of their speed. The remaining gap:

| | Before | After MMA | Savings |
|---|--------|-----------|---------|
| Flash attention | 2,050µs | 694µs | **-1,356µs** |
| FP16↔F32 convert | 0 | 54µs | new overhead |
| **Net** | | | **-1,302µs** |

The 249µs gap to llama.cpp comes from the FP16→F32→FP16 conversion (54µs) and their stream-K grid scheduling (we use simple tiling). Both are fixable — output Q GEMM directly as F32 to eliminate the conversion, adopt stream-K dispatch for better SM utilization.

## Proper Benchmarking

All previous numbers were ad-hoc (different prompts, single runs). Added `scripts/bench_prefill.sh` — runs both GWEN and llama.cpp at configurable prompt lengths with multiple iterations, reporting mean ± stddev.

## Final Results

`./scripts/bench_prefill.sh --runs 5 --lengths "128 256 512"`:

| Prompt | GWEN (tok/s) | llama.cpp (tok/s) | Ratio |
|--------|-------------|-------------------|-------|
| pp128 | 16,834 ± 66 | 14,477 ± 2,295 | **1.16×** |
| pp256 | 23,854 ± 29 | 21,297 ± 2,725 | **1.12×** |
| pp512 | 29,497 ± 101 | 26,386 ± 2,314 | **1.12×** |

Decode: ~780 tok/s with MTP speculative decoding (unchanged).

Note the variance: GWEN ±0.3% vs llama.cpp ±10%. Our deterministic kernel dispatch is more predictable than their CUDA graph approach.

## Kernel Breakdown (pp500, nsys)

| Kernel | Time | % |
|--------|------|---|
| DeltaNet prefill | 5.33ms | 27% |
| CUTLASS GEMMs (128×64) | 5.04ms | 25% |
| CUTLASS GEMMs (128×128) | 1.31ms | 7% |
| Conv1d | 1.48ms | 7% |
| MMA flash attention | 0.69ms | 4% |
| LM head GEMV | 0.69ms | 4% |
| Gate/beta + norms + misc | 1.5ms | 8% |

## What Didn't Work

1. **Shared memory tiling for flash attention** — 27% regression from low occupancy and __syncthreads overhead
2. **GQA L1 cache grouping** — negligible because L2 is shared across all SMs
3. **Shared memory tiling for DeltaNet** — regressed because cooperative loading serialized warps that were previously independent
4. **Pointer-increment optimization** — no measurable change (compiler already optimized the address math)
5. **Hand-ported MMA kernel** — 818 lines of generated code crashed with shared memory bugs; using llama.cpp's tested headers directly was the right call

## What's Left

DeltaNet (27%) and GEMMs (32%) dominate. Both are close to their theoretical limits:
- DeltaNet is memory-latency-bound (long_scoreboard dominant), sequential by nature
- GEMMs run at ~30% of peak tensor core throughput (limited by problem size)

The next frontier is SM_120-specific features: FP8 attention (2× throughput), L2 cache pinning for hot weights, TMA for weight prefetch. But that's for another day — we're beating llama.cpp by 12% across all prompt lengths.
