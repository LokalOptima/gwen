# 36. Beating llama.cpp on Both Fronts

*gwen-bench, dead-end kernel fusions, and a one-line fix that doubled prefill speed.*

## Starting Point

Post #35 left the 9B model at 118 tok/s decode, matching llama.cpp. But we had no proper benchmarking tool, and prefill was untested. This session added a rigorous benchmark, investigated kernel-level optimizations, and found a massive prefill bottleneck hiding in plain sight.

## gwen-bench: A Proper Benchmark

First problem: GWEN's timing included prefill amortized into the decode number, making comparisons with llama-bench imprecise. llama-bench separates prompt processing (pp) from text generation (tg), runs warmup, and reports mean ± stddev across multiple repetitions.

I wrote `gwen-bench` as an exact replica of llama-bench's methodology:

1. **Warmup**: full prefill + 1 decode token (discarded)
2. **Timed loop**: for each repetition, reset state → prefill (if pp>0) → N decode steps → wall clock
3. **Random tokens**: not real generation — pure throughput measurement (same as llama-bench)
4. **Synchronized**: each `forward()` call completes before the next is timed

```
$ gwen-bench -m Qwen3.5-9B-UD-Q4_K_XL.gguf -p 0 -n 128 -r 5

| model                          |       size |     params |    backend |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --------------: | -------------------: |
| qwen35 9B Q4_K_XL              |   5.55 GiB |     8.95 B |       CUDA |           tg128 |       127.07 ± 0.08 |
```

The pure decode number (127 tok/s) is higher than the 118 tok/s reported in post #35 because gwen-bench doesn't include prefill overhead. With prefill buffers also allocated, decode drops to ~124 tok/s due to increased VRAM pressure (same behavior as llama-bench).

## Dead End: Kernel Fusion Experiments

The nsys profile from post #35 showed RMSNorm (4.7%) and Q8_1 quantize (1.6%) as the next targets after GEMV. The pattern in `forward_body` is:

```
gwen_rmsnorm_f32_input(...)     // 32 threads, 1 block
gwen_quantize_q8_1(...)          // 256 threads, 16 blocks
```

65 such pairs per token — 130 kernel launches that could be fused into 65.

### Attempt 1: 256-Thread Fused Kernel

Combined both operations into a single 256-thread kernel with cross-warp reduction for the RMS computation:

```cuda
__global__ void __launch_bounds__(256)
kernel_rmsnorm_f32_input_quantize_q8_1(...) {
    // Phase 1: 256 threads compute sum-of-squares, cross-warp reduce
    // Phase 2: each warp processes Q8_1 blocks with norm+weight applied
}
```

**Result: 124.96 tok/s (-1.6%).** The cross-warp `__syncthreads()` barrier killed it. A single 256-thread block on one SM can't match 16 blocks spread across multiple SMs for the Q8_1 phase.

### Attempt 2: 32-Thread Fused Kernel

Keep the original single-warp RMSNorm structure, process Q8_1 blocks serially in the normalize pass:

```cuda
__global__ void __launch_bounds__(32)
kernel_rmsnorm_f32_input_quantize_q8_1(...) {
    // Phase 1: warp shuffle sum-of-squares (no __syncthreads__)
    // Phase 2: for each Q8_1 block, apply norm, quantize, write
}
```

**Result: 105.00 tok/s (-17.4%).** Serializing 128 Q8_1 blocks through one warp is catastrophically slow. Each block needs warp-level amax and sum reductions (128 × warp shuffles), while the separate kernel parallelizes this across 128 warps.

### Attempt 3: `__launch_bounds__` Occupancy Hints

Added `__launch_bounds__(NW * 32, 48 / NW)` to Q6_K (48 regs → target 42) and Q8_0 (52 regs → target 42) kernels, hinting the compiler to reduce register usage for higher occupancy.

**Result: 124.65 tok/s (-1.9%).** The compiler spilled registers to local memory. The spill overhead was worse than the occupancy gain. These kernels are already at 78-82% bandwidth efficiency — the register pressure isn't actually the bottleneck.

### Why Fusion Failed

With CUDA graphs, the overhead between back-to-back kernels is ~1-2µs (not the ~5µs of individual launches). The Q8_1 quantize kernel finishes in 1µs by distributing work across 16 blocks on different SMs. Any fusion that serializes this work onto fewer SMs makes it slower, even though it eliminates a kernel transition. The graph executor's scheduling is already nearly optimal for these tiny kernels.

**Lesson: don't fuse tiny kernels that are already fast in a CUDA graph.** The parallel execution across SMs matters more than eliminating launch transitions.

## The Real Win: IQ4_XS Prefill

With gwen-bench measuring prefill separately, the first pp512 result was alarming:

```
GWEN:      2,736 tok/s
llama.cpp: 5,705 tok/s  (2.1× faster!)
```

An nsys profile of the 512-token prefill revealed the culprit:

| Kernel | Time (ms) | % of Total |
|--------|-----------|-----------|
| **IQ4_XS dp4a GEMV (sequential)** | **86.9** | **42.6%** |
| MMQ GEMM Q4_K | 23.9 | 11.7% |
| MMQ GEMM Q5_K | 16.2 | 8.0% |
| DeltaNet prefill | 12.6 | 6.2% |
| Everything else | 64.5 | 31.5% |

The 10 IQ4_XS tensors (5 layers × ffn_gate + ffn_up) had no MMQ GEMM kernel. The `do_gemm` lambda fell back to **sequential per-token GEMV** — 512 individual GEMV calls per tensor, 5,120 total. Each GEMV launches 12,288 blocks for a single token when a GEMM could process all 512 tokens in one launch.

### First Attempt: FP16 Dequant + cuBLAS

The fast path: dequant IQ4_XS → FP16 in a GPU kernel, then call cuBLAS `hgemm`:

```cuda
kernel_dequant_iq4xs_to_f16<<<M, 256>>>(w.device_data, scratch, M, K);
cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, ...);
```

**Result: 5,913 tok/s — 2.16× faster!** But correctness testing showed regressions. Two prompts that previously matched llama.cpp exactly now diverged:

| Prompt | Baseline | FP16 Dequant |
|--------|----------|-------------|
| `The capital of France is` | PASS | **DIFF@32** |
| `def factorial(n):` | PASS | **DIFF@97** |

The FP16 dequant → FP16 GEMM path uses different floating-point arithmetic than the integer dp4a path. The logit differences are tiny but cascade into different token selections for ambiguous positions. llama.cpp uses integer MMQ for IQ4_XS prefill, which matches the decode path's dp4a precision.

### The Fix: MMQ GEMM for IQ4_XS

The llama.cpp MMQ headers (which GWEN compiles against) already had full IQ4_XS support. The fix was adding one case to the dispatch:

```cpp
case GGMLType::IQ4_XS: launch_mmq<GGML_TYPE_IQ4_XS>(...); break;
```

That's it. The template machinery, tile loading functions, `__byte_perm` lookup tables, and stream-K scheduling were all already implemented in the llama.cpp MMQ infrastructure. We just never instantiated the IQ4_XS template.

**Result: 5,997 tok/s.** Slightly faster than the FP16 path (no dequant overhead, no 288 MB scratch buffer). And perfect correctness — 7/9 prompts match llama.cpp exactly, identical to the baseline (the 2 that diverge are pre-existing precision differences on ambiguous completions).

## Final Results

### Decode (tg128)

| Engine | tok/s | vs llama.cpp |
|--------|-------|-------------|
| llama.cpp | 117.94 ± 0.16 | — |
| **GWEN** | **127.07 ± 0.08** | **+7.7%** |

### Prefill (pp512)

| Engine | tok/s | vs llama.cpp |
|--------|-------|-------------|
| llama.cpp | 5,739.65 ± 339.24 | — |
| **GWEN** | **5,996.64 ± 30.71** | **+4.5%** |

GWEN beats llama.cpp on both prefill and decode for the 9B model. The stddev is also notably tighter (±0.08 vs ±0.16 for decode, ±30 vs ±339 for prefill) — CUDA graphs give very consistent timing.

### Correctness

7/9 test prompts produce identical output to llama.cpp. The 2 that diverge (`Hello`, `import numpy as np`) are pre-existing FP16 precision differences (same as before these changes, same divergence positions). The integer dp4a arithmetic in both decode and prefill paths matches llama.cpp's MMQ implementation.

## Lessons Learned

1. **Measure before you celebrate.** The 118 tok/s from post #35 included prefill overhead. gwen-bench revealed the true decode speed (127 tok/s) and exposed the prefill disaster (2,736 vs 5,705 tok/s). Without separate measurements, I'd have thought we matched llama.cpp when we were actually 2× slower on prefill.

2. **Don't fuse what CUDA graphs already optimize.** Three attempts at kernel fusion all made things worse. CUDA graphs eliminate CPU-side launch overhead, and the GPU's ability to scatter tiny kernels across SMs is more valuable than eliminating inter-kernel transitions. Fusion only helps when the intermediate data is large enough that avoiding the write/read matters — not for 8 KB FP16 buffers.

3. **Profile the full pipeline, not just decode.** Decode gets all the attention because it's the steady-state bottleneck. But a 42.6%-of-prefill bottleneck in IQ4_XS was invisible until profiling the prefill path separately.

4. **Check if the infrastructure already supports what you need.** The MMQ GEMM template for IQ4_XS existed in the llama.cpp headers we compile against. The fix was literally one line of dispatch code. The FP16 dequant kernel I wrote first (40+ lines of CUDA) was both slower and less correct than the one-liner.

5. **Precision matters more than speed.** The FP16 dequant path was slightly faster (5,913 vs 5,997 tok/s) but caused correctness regressions. The MMQ path preserves integer arithmetic compatibility with both our decode path and llama.cpp's implementation, making it the right choice even though the performance difference is negligible.
