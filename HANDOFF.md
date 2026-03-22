# Agent Handoff Notes

Context for the next agent session.

---

## Current State (2026-03-22)

### Performance (bench_prefill.sh, 5 runs each)
- **Prefill pp512**: 29,497 ± 101 tok/s — **1.12× llama.cpp** (26,386 ± 2,314)
- **Prefill pp256**: 23,854 ± 29 tok/s — **1.12× llama.cpp** (21,297 ± 2,725)
- **Prefill pp128**: 16,834 ± 66 tok/s — **1.16× llama.cpp** (14,477 ± 2,295)
- **Decode**: ~780 tok/s with MTP speculative decoding (greedy)
- Starting point: 22,319 tok/s at pp565 (0.86× llama.cpp)

### What Was Completed (this session)

1. **Auto-download weights** to `~/.cache/gwen/` — zero-flag usage: `./build/gwen "prompt"`
2. **Removed v7 OOV gate** — reverted to v6 IDK neuron (v7 was 2% slower)
3. **Uploaded v6 weights** to GitHub release, removed obsolete `mtp_lm_head_20k.bin`
4. **DeltaNet prefill kernel rewrite** (ba3914d): S matrix in registers, warp-per-column layout (adapted from llama.cpp). 12.6ms → 1.63ms per prefill (6.9x faster)
5. **Batched full attention** (1358807): Replaced per-token loop (N×7 kernel launches per layer) with batched deinterleave/RMSNorm/RoPE + parallelized flash attention. 5ms → 0.18ms for 6 layers
6. **Pre-dequanted FP16 weights** (f6fdb46): Dequant Q4_K→FP16 once at load, skip per-prefill dequant. +950 MB VRAM, eliminates 3ms/prefill
7. **Full FP16 prefill pipeline** (49c0600): Eliminated all F32↔FP16 conversions in residual stream
8. **Multi-query flash attention** (b8f965f): 4 queries/warp + GQA grouping, K/V loaded once per key pos. 3.2ms → 2.0ms
9. **Pre-normalized DeltaNet Q/K** (b8f965f): Move L2 norm out of recurrence. 10 fewer warp reductions/token. 7.7ms → 5.4ms
10. **CUTLASS tile auto-selection** (091f031): 128×64 tiles for small GEMMs. 7.9ms → 6.4ms
11. **Prefill benchmark script** (da7b69a): `scripts/bench_prefill.sh` — proper comparison with statistics
12. **MMA flash attention** (f6920f0): Ported llama.cpp's `flash_attn_ext_f16` kernel (D=256, mma.sync.m16n8k16). 2.05ms → 0.69ms for 6 layers (2.95× faster). llama.cpp gets 0.45ms — we're at 1.56× of theirs. The 0.25ms gap is from FP16→F32→FP16 conversion overhead (54µs) and different grid scheduling (we use simple tiling, they use stream-K)
13. **Blog post 29**: Documented all optimizations, dead ends, and measurements

### Prefill Kernel Breakdown (pp500, nsys profiled)

| Kernel | Time | % | Notes |
|--------|------|---|-------|
| DeltaNet prefill fast | 5.33ms | 27% | Pre-normalized Q/K, ncu-verified same speed as llama.cpp |
| CUTLASS GEMMs (128×64) | 5.04ms | 25% | Auto-selected for small M at N=512 |
| CUTLASS GEMMs (128×128) | 1.31ms | 7% | For large M (attn_qkv M=6144) |
| Conv1d | 1.48ms | 7% | Already efficient |
| MMA flash attention | 0.69ms | 4% | Ported from llama.cpp (was 2.05ms scalar) |
| LM head GEMV | 0.69ms | 4% | Single token, one-time |
| Gate/beta + norms + misc | 1.5ms | 8% | Small kernels |
| L2 normalize QKV | 0.15ms | 1% | Pre-normalize Q/K for DeltaNet |
| FP16↔F32 convert (fattn) | 0.05ms | <1% | Q→F32 for MMA kernel, output→FP16 |

### Key Files Modified
- `src/inference.cu` — DeltaNet fast kernel, multi-query attention, FP16 pipeline, pre-norm, MMA attention call
- `src/kernels/fattn_mma.cu` — MMA flash attention wrapper (compiles against llama.cpp headers)
- `src/kernels/gemm_cutlass.cu` — Tile auto-selection (128×128 and 128×64)
- `src/model.cu` — Pre-dequant FP16 weights at upload
- `include/gwen/kernels.h` — gwen_flash_attn_mma declaration
- `include/gwen/mma_types.cuh` — MMA tile types (D=256 Turing path, fallback)
- `include/gwen/flash_attn_mma.cuh` — Extracted kernel (D=256, fallback if no llama.cpp)
- `include/gwen/cp_async.cuh` — cp.async wrappers (fallback)
- `include/gwen/paths.h` — Auto-download helpers
- `CMakeLists.txt` — --extended-lambda, llama.cpp include paths for fattn_mma.cu
- `scripts/bench_prefill.sh` — Proper prefill benchmark
- `blog/29-matching-and-beating-llama-cpp-prefill.md` — Full writeup

### Dead Ends (don't repeat)
1. **Shared memory tiling for flash attention** — 27% regression, low occupancy + syncthreads overhead
2. **Shared memory tiling for DeltaNet** — regression, cooperative loading serialized independent warps
3. **GQA L1 cache grouping** — negligible, L2 is shared across all SMs
4. **Hand-ported MMA kernel** (818 lines) — shared memory bugs, use llama.cpp's tested headers instead
5. **"DeltaNet 2.5× gap"** — was a measurement error. ncu confirmed kernels are within 5% of each other

---

## Next Steps: Push Further Beyond llama.cpp

We beat llama.cpp by 12% at pp512. Remaining optimizations:

### 1. Close Remaining Flash Attention Gap (694µs → 445µs target)
Our MMA flash attention is 1.56× llama.cpp's (694µs vs 445µs for 6 layers). Two known causes:
- **FP16→F32→FP16 conversion**: 54µs overhead. Fix: output Q GEMM directly as F32 (use gwen_gemm_f32out_auto), eliminate the pre-conversion kernel.
- **Grid scheduling**: We use simple tiling, llama.cpp uses stream-K. Fix: adopt stream-K dispatch with fixup kernel (already in their codebase, just needs wiring).

### 2. Fused Dequant-Matmul (future, VRAM savings)
Pre-dequanting to FP16 wastes ~950 MB VRAM. Read Q4_K directly during GEMM like llama.cpp's `mul_mat_q`. Only matters for VRAM-constrained scenarios.

### 3. DeltaNet Chunkwise for Long Sequences (future)
Sequential DeltaNet is O(N) per layer. At pp2048+, chunkwise parallel scan becomes worthwhile.

### 4. SM_120-Specific Optimizations (future)
- **FP8 flash attention** — 2× throughput on the attention bottleneck (mma.sync.m16n8k32.e4m3)
- **L2 cache pinning** — keep hot weights resident with cudaAccessPolicyWindow
- **TMA weight prefetch** — overlap next-layer weight loads with current-layer compute

---

## SM_120 (Blackwell Consumer) Primitives

llama.cpp targets broad GPU compatibility (Turing+). SM_120 has hardware features they don't use:

### TMA (Tensor Memory Accelerator)
Async global→shared copies with hardware address generation. Eliminates address computation overhead and overlaps loads with compute. llama.cpp uses `cp.async` (Ampere) but not TMA. Use for weight prefetching in GEMMs and K/V tile loading in flash attention.

### FP8 (E4M3/E5M2) Tensor Cores
Native FP8 MMA on SM_120. Quantize activations to FP8 for flash attention QK^T and softmax(QK^T)@V — **2× throughput** over FP16 MMA. llama.cpp doesn't use FP8 for attention at all. Precision is sufficient for attention scores (softmax normalizes away small errors).

### FP4 MMA (Mixed FP4×FP8)
SM_120 supports FP4×FP8 tensor core ops. Could keep weights in FP4 (~half the size of Q4_K's ~4.5 bits), activations in FP8. Ultimate bandwidth optimization for weight-bound GEMMs. Requires careful quantization calibration.

### 48 MB L2 Cache Pinning
RTX 5070 Ti has 48 MB L2 — large enough to pin frequently-used data across layers. Use `cudaAccessPolicyWindow` to guarantee L2 residency for:
- Norm weights (24 layers × ~4 KB each = ~96 KB)
- ssm_alpha/beta projections (18 layers × ~32 KB each = ~576 KB)
- Reduced LM head for MTP (~8 MB)
llama.cpp doesn't do cache pinning.

### Direct PTX — No Library Needed
CUTLASS is optional. All tensor core and TMA instructions are PTX intrinsics callable from inline asm:
```cuda
// FP8 MMA example — m16n8k32, E4M3 inputs, F32 accumulate
asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
             : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
               "r"(b0), "r"(b1),
               "f"(c0), "f"(c1), "f"(c2), "f"(c3));

// TMA async bulk load — global → shared
asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
             "[%0], [%1, {%2, %3}], [%4];"
             :: "r"(smem_addr), "l"(tensor_map), "r"(x), "r"(y), "r"(mbar_addr));
```
