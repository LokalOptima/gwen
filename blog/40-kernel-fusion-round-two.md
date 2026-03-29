# 40. Kernel Fusion Round Two

*Revisiting a failed optimization from post #36, fixing the scaling bug, and cutting non-GEMV overhead by 26%.*

## Starting Point

Post #39 left GWEN at 128.11 tok/s decode on the 9B model, beating llama.cpp's 118.07 tok/s by 8.5%. Prefill was roughly even (GWEN slightly ahead at pp512, slightly behind at pp128). Good numbers — but the profiler told a clear story about where time was being wasted.

## Profiling: Where the Time Goes

```
Total:     7.785 ms  (128 tok/s)
GEMV:      6.913 ms  (88.8%)  — 96% of DRAM roofline
Non-GEMV:  0.872 ms  (11.2%)  — the target
```

The GEMV path (weight loads + dp4a dot products) was already at 96% of the DRAM bandwidth roofline. No room there. But 0.872 ms of non-GEMV overhead — kernel launches, element-wise ops, reductions — that's where the fat is.

Looking at the decode forward pass, the pattern repeats 24 times per token:

```
gwen_rmsnorm_f32w(...)        // 1 block × 32 threads
gwen_quantize_q8_1(...)       // 15 blocks × 256 threads
// ... GEMV ...
gwen_swiglu(...)              // 15 blocks × 256 threads
gwen_quantize_q8_1(...)       // 15 blocks × 256 threads
// ... GEMV ...
```

Even inside a CUDA graph (which eliminates host-side launch overhead), each kernel still has to execute sequentially on the GPU. The gap between kernels — drain the pipeline, refill it — adds up across ~243 kernel launches per token.

## What llama.cpp Does

I spent time digging through llama.cpp's CUDA backend looking for optimizations GWEN was missing. The most relevant finding: **MMVQ fusion**. llama.cpp's GEMV kernel accepts an optional `fusion` struct that lets it apply SwiGLU, bias addition, or gate activation directly in the GEMV epilogue. The gate values are read from L2 during the output writeback — no intermediate buffer, no separate activation kernel.

Other llama.cpp tricks (L2 cache hints in `cp.async`, device-specific warp count tuning, fused RoPE) were either already present in GWEN or offered negligible gains for single-token decode.

The key insight wasn't any single llama.cpp trick, but the general principle: GWEN already had fused kernels sitting unused in its codebase.

## The Fused Kernels Already Existed

Post #36 documented `kernel_rmsnorm_quantize_q8_1` — a 256-thread kernel that computes RMSNorm and Q8_1 quantization in a single launch. It was written for the 0.8B model (dim=1024), tested, and found to be **1.6% slower** than separate kernels. The conclusion at the time: "The cross-warp `__syncthreads()` barrier killed it."

Similarly, `kernel_swiglu_quantize_q8_1` existed — computing SwiGLU and writing Q8_1 directly, skipping the FP16 intermediate.

Neither kernel was wired into the decode path. Time to fix that.

## Wiring the Fused Kernels

The changes in `inference.cu` were mechanical. Five RMSNorm+quantize sites:

```cuda
// Before: 2 separate kernels
gwen_rmsnorm_f32w(buf_a, weight, x_norm, dim, eps, s);
if (use_dp4a) gwen_quantize_q8_1(x_norm, x_q8_a, dim, s);

// After: 1 fused kernel (dp4a path) or original (fallback)
if (use_dp4a)
    gwen_rmsnorm_quantize_q8_1(buf_a, weight, x_q8_a, x_norm, dim, eps, s);
else
    gwen_rmsnorm_f32w(buf_a, weight, x_norm, dim, eps, s);
```

And two SwiGLU+quantize sites:

```cuda
// Before: 2 separate kernels
gwen_swiglu(ffn_gate, ffn_up, ffn_out, n_ff, s);
if (use_dp4a) gwen_quantize_q8_1(ffn_out, x_q8_a, n_ff, s);

// After: 1 fused kernel — writes Q8_1 directly, skips FP16 intermediate
if (use_dp4a)
    gwen_swiglu_quantize_q8_1(ffn_gate, ffn_up, x_q8_a, n_ff, s);
else
    gwen_swiglu(ffn_gate, ffn_up, ffn_out, n_ff, s);
```

The SwiGLU fusion is especially clean: when `use_dp4a` is true, the downstream `ffn_down` GEMV reads from `x_q8_a` (the Q8_1 buffer), never touching `ffn_out`. So we can skip writing FP16 entirely.

Built, ran correctness checks — identical output to the pre-fusion build. Benchmarked:

```
126.48 tok/s  (-1.3%)
```

Slower. Again. The exact same failure mode as post #36.

## The Scaling Bug

The profiler showed the problem clearly:

```
Non-GEMV overhead: 0.973 ms (12.3%)  — UP from 0.872 ms
```

The fused RMSNorm+Q8_1 kernel uses 1 block × 256 threads (8 warps). For the 0.8B model with dim=1024: 32 Q8_1 blocks ÷ 8 warps = 4 iterations per warp. Fine.

For the 9B model with dim=3584: 112 Q8_1 blocks ÷ 8 warps = 14 iterations per warp. All serialized in a single thread block on a single SM.

The separate `quantize_q8_1` kernel uses 15 blocks × 256 threads. Those 15 blocks can execute across multiple SMs in parallel. The fused kernel forces all the work onto one SM.

The regression: 14 serial iterations × 49 calls per token ≈ +0.1 ms. Matches the observed +0.101 ms exactly.

## The Fix: Template on Warp Count

The solution: template the kernel on `NW` (number of warps) and dispatch based on dimension, just like the GEMV kernels:

```cuda
template<int NW>
__global__ void __launch_bounds__(NW * 32)
kernel_rmsnorm_quantize_q8_1(const half* x, const float* weight,
                              block_q8_1* y_q8, half* y_fp16,
                              int dim, float eps) {
    // Phase 1: sum-of-squares with NW warps
    // Cross-warp reduce via shared memory (NW entries)
    // Phase 2: each warp handles ceil(n_blocks / NW) Q8_1 blocks
}
```

Dispatch:

```cuda
void gwen_rmsnorm_quantize_q8_1(...) {
    if (dim <= 1024)
        kernel_rmsnorm_quantize_q8_1<8><<<1, 256, ...>>>();   // 4 iters/warp
    else
        kernel_rmsnorm_quantize_q8_1<32><<<1, 1024, ...>>>();  // 4 iters/warp
}
```

For dim=3584 with NW=32: 112 Q8_1 blocks ÷ 32 warps = 3.5 → 4 iterations per warp. Same as the dim=1024 case. The kernel stays in a single block (avoiding inter-block sync for the RMS reduction), but 1024 threads provide enough parallelism within that block.

## Results

```
Forward pass:   7.549 ms  (132.5 tok/s)
GEMV total:     6.908 ms  (unchanged)
Non-GEMV:       0.641 ms  (8.5%)
```

| Metric | Before | After | Delta |
|---|---|---|---|
| Decode tok/s | 128.11 | **132.06** | **+3.1%** |
| Forward pass | 7.785 ms | 7.549 ms | -0.236 ms |
| Non-GEMV overhead | 0.872 ms | **0.641 ms** | **-26.5%** |
| vs llama.cpp | +8.5% | **+11.9%** | +3.4 pp |

Prefill unchanged at ~6,050 tok/s (the fusions only affect the decode path).

## Correctness

The fused kernel changes the floating-point reduction order for sum-of-squares (32 warps reducing in shared memory vs 1 warp with shuffles). This changes `rms_inv` by a few ULP, but the effect is negligible — tested with:

1. **Bit-exact match** with pre-fusion GWEN output on multiple prompts
2. **Full test suite** (teacher forcing, free generation, determinism) — identical results before and after, including the same pre-existing GWEN↔llama.cpp divergences
3. **Deterministic**: 3 consecutive identical greedy runs confirmed

## What's Left on the Table

Non-GEMV overhead is now 0.641 ms. The remaining budget:

- **DeltaNet recurrence**: 0.200 ms (31%) — the S matrix update is fundamentally serial
- **RMSNorm (standalone)**: ~0.05 ms — the batched Q/K norms for full-attention layers
- **Gated RMSNorm + quantize**: ~0.08 ms — not yet fused (18 DeltaNet layers)
- **Sigmoid-mul + quantize**: ~0.03 ms — not yet fused (6 full-attention layers)
- **Everything else**: ~0.28 ms — conv1d, RoPE, KV cache, argmax, h2f conversion

The next fusion targets (gated_rmsnorm+quantize, sigmoid_mul+quantize) would save maybe 40-50 µs — diminishing returns. The real ceiling is the 6.908 ms of GEMV time, which is 96% of the DRAM bandwidth roofline. To go meaningfully faster on decode, we'd need to read fewer bytes (better quantization) or exploit L2 caching more aggressively.

## Lessons

1. **Failed optimizations have expiration dates.** The fused RMSNorm+Q8_1 kernel was correctly rejected for the 0.8B model in post #36. For the 9B model with 3.5× larger dimensions, the same kernel concept works — it just needed to scale the thread count.

2. **Profile before and after.** The first attempt made things worse. Without the profiler showing non-GEMV overhead went *up*, I would have assumed the fusion was just noise and moved on.

3. **The kernel that already exists is the fastest to write.** Both fused kernels were already in the codebase, tested and working. The entire optimization was wiring changes in `inference.cu` plus a template parameter in `rmsnorm.cu`.
