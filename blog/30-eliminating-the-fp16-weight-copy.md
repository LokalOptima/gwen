# Post 30: Eliminating the FP16 Weight Copy

We were storing every weight twice on the GPU — 530 MB in Q4_K for decode, plus 924 MB in FP16 for prefill. 1,454 MB total for a 0.8B model. This post documents how we eliminated the FP16 copy by porting llama.cpp's fused quantized GEMM kernel, and the five failed approaches along the way.

## The Problem

GWEN's two inference paths need different weight formats:

- **Decode (GEMV, N=1)**: Bandwidth-bound. Reads Q4_K weights directly via dp4a. Needs the original quantized data (530 MB).
- **Prefill (GEMM, N=128-512)**: Compute-bound. Uses CUTLASS mma.sync tensor cores, which only accept FP16. Needs dequantized FP16 weights (924 MB).

We were pre-dequanting the entire model to FP16 at load time and keeping both copies on GPU. A 13 MB temp buffer (the size of one weight matrix) would have been enough — we were making a full copy of the model just to avoid dequanting 13 MB per GEMM call.

## Per-Call Dequant: The Quick Fix

The simplest fix: delete the pre-dequant, use a 13 MB temp buffer, dequant per GEMM call. Three changes:

1. Remove `upload_weight_with_fp16()` — just `upload_weight()` for everything
2. `gwen_gemm_auto()` falls through to dequant+CUTLASS path (fp16_data is always null)
3. Fix `gwen_gemm()` to use the same tile auto-selection as `gwen_gemm_fp16()` (it was using 128×128 tiles for all sizes — half the overhead was from this bug)

Results at pp512: 25,272 tok/s (was 29,497 with pre-dequant). **924 MB saved, 14% slower, still tied with llama.cpp** (26,386 tok/s).

The overhead: 1.7ms per forward pass for dequanting each weight to the temp buffer before CUTLASS reads it.

## Five Failed Fused Kernels

We tried to eliminate the overhead entirely by writing a GEMM kernel that reads Q4_K directly — no temp buffer, no FP16 copy:

**Attempt 1: Naive dp4a GEMM.** One thread per output element, flat loops, per-element dequant with branchy scale extraction. 15× slower than CUTLASS. The per-element dequant recomputed d/dmin/scales for every single element instead of amortizing across the Q4_K block.

**Attempt 2: Tiled dp4a with shared memory.** Loaded Q4_K blocks to shared, Q8_1 activations to shared, dp4a in the compute phase. 5× slower. The fundamental issue: dp4a does 8 FLOPs/instruction while mma.sync does 2,048. dp4a can't compete with tensor cores for GEMM.

**Attempt 3: Dequant to FP16 in shared + mma.sync.** Loaded Q4_K to shared, dequanted to a second shared memory region as FP16, then ldmatrix + mma.sync. 8× slower. The shared→shared dequant pass was the bottleneck — writing FP16 to shared and reading it back added an entire extra pass.

**Attempt 4: Register-level dequant + mma.sync.** Dequanted Q4_K directly into MMA register fragments, skipping the shared memory intermediate for weights. 5× slower. The dequant still dominated — 8 half values per thread per mma.sync call, each requiring byte load + nibble extract + FMA + half2 pack.

**Attempt 5: Wider tiles (64×64) with warp looping.** Each warp computed multiple N-tiles to amortize dequant across more mma.sync calls. 4× slower. Better ratio but still dequant-bound.

The lesson: writing a competitive GEMM kernel from scratch is not a weekend project. CUTLASS and llama.cpp's mmq have years of optimization behind them.

## The Solution: Port llama.cpp's MMQ Kernel

Same approach as our flash attention port (blog post 29): compile against their headers, replicate the launch logic, call their kernel directly.

llama.cpp's `mul_mat_q` kernel in `mmq.cuh` is a template parameterized on quantization type and tile size. On SM_120, it uses the **Turing MMA path** — not dp4a, but actual tensor core MMA instructions via `mma.sync`. This is why their mmq is competitive with FP16 CUTLASS despite reading quantized data.

The port required:
- A separate compilation unit (`gemm_mmq.cu`) with llama.cpp include paths
- Including their `quantize.cu` for the `quantize_mmq_q8_1_cuda` function
- `ggml_abort` stub (same as flash attention)
- Replicating the launch parameters: ncols_x, nrows_x, stride_row_x, ncols_y, stride_col_dst

**The critical parameter: `mmq_x = 128`.** I initially used `mmq_x = 64` (6.7× slower). SM_120 has `TURING_MMA_AVAILABLE`, so `mmq_x_max = 128`. Doubling the tile width halved the kernel time.

## Stream-K: The Key to Matching llama.cpp

With standard tiling, the grid for attn_qkv (M=6144, K=1024, N=512) at mmq_y=128, mmq_x=128 is (48, 4) = 192 blocks on 70 SMs = 2.7 waves. The last partial wave wastes 40% of SM capacity.

**Stream-K scheduling** launches exactly 70 blocks (one per SM) and distributes K-dimension work evenly across them. No partial waves, no wasted SMs.

| Config | attn_qkv (µs) | ffn_gate (µs) | Sum all (µs) |
|--------|--------------|---------------|--------------|
| mmq_x=64, tiling | 551 | 371 | 2,549 |
| mmq_x=128, tiling | 245 | 166 | 1,493 |
| **mmq_x=128, stream-K** | **91** | **68** | **576** |

Stream-K was a **2.6× improvement** over standard tiling at the same mmq_x. The kernel went from 6.7× slower to within 43% of CUTLASS.

## Current State

The mmq kernel is integrated into the prefill path for all quantized weight types (Q4_K, Q5_K, Q6_K, Q8_0). One copy of weights on GPU, no FP16 duplicate.

| Metric | Pre-dequant (before) | MMQ fused (after) |
|--------|---------------------|-------------------|
| VRAM | 2,129 MB | 1,341 MB |
| pp128 | 16,834 tok/s | 12,166 tok/s |
| pp256 | 23,854 tok/s | 18,312 tok/s |
| pp512 | 29,497 tok/s | 24,273 tok/s |
| Decode | 780 tok/s | 849 tok/s |

Prefill is 28% slower but VRAM is 37% lower. Decode is unaffected (still uses dp4a GEMV on Q4_K directly).

## Remaining Overhead

The mmq wrapper adds FP16→F32 and F32→FP16 conversions around every GEMM call because llama.cpp's quantizer expects F32 input and the kernel outputs F32, while our pipeline is FP16. A direct FP16→Q8_1_mmq quantizer would eliminate this.

The stream-K fixup buffer and F32 activation/output temporaries add ~29 MB to the scratch buffer. This could be reduced by processing activations in FP16 throughout.

## What Worked, What Didn't

**Worked:**
1. Per-call dequant to temp buffer (simple, 14% overhead, 924 MB saved)
2. Porting llama.cpp's mmq kernel directly (proven code, MMA tensor cores)
3. Stream-K scheduling (2.6× improvement from eliminating partial waves)
4. mmq_x=128 on SM_120 (Turing MMA path, 2× over mmq_x=64)

**Didn't work:**
1. Five custom fused GEMM kernels (dp4a, shared-memory dequant, register-level dequant)
2. Double-buffered dequant pipeline (resource contention made it slower than serial)
3. Writing a competitive GEMM from scratch (CUTLASS has years of optimization)

The recurring lesson from this project: when llama.cpp has a proven solution, port it. Don't reinvent it.
