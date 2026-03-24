# Post 33 — FP4 GEMV: From 12 to 139 tok/s

Post 32 ended with GWEN running the 4B NVFP4 model at ~42 tok/s — correct output, but 4.3× slower than llama.cpp's 184 tok/s. This post is about finding and fixing the bottleneck.

## The Starting Point

```
GWEN NVFP4:     12.4 tok/s (no CUDA graph)
                 42.6 tok/s (with CUDA graph)
llama.cpp Q4_K:  184  tok/s
```

The initial FP4 code path skipped CUDA graph capture because I assumed the host-side `if/else` dispatches (choosing between FP4, FP8, and FP16 kernels) would prevent graph capture. They don't — the conditionals are evaluated once during capture and the same path is replayed. Enabling CUDA graph immediately brought decode from 12.4 to 42.6 tok/s.

But 42.6 vs 184 meant the FP4 GEMV kernel itself was slow.

## Profiling

`nsys profile --cuda-graph-trace=node` revealed the breakdown per token:

| Kernel | % GPU Time | Avg (μs) |
|--------|-----------|----------|
| `kernel_gemv_fp4` | 64.3% | 72 |
| `kernel_gemv_fp4_residual_f32` | 25.0% | 80 |
| `kernel_gemv_fp16` (lm_head) | 7.6% | 1,487 |
| Everything else | 3.1% | — |

FP4 GEMV was 89.3% of total GPU time. The lm_head FP16 GEMV achieved 95% of peak bandwidth (854 GB/s on the 5070 Ti's 896 GB/s). The FP4 kernels achieved ~13%.

## Why So Slow?

The original FP4 GEMV kernel had a 16-entry LUT in `__constant__` memory for FP4→FP16 conversion:

```cuda
static __device__ __constant__ uint16_t fp4_lut[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00,  // +0.0..+1.5
    0x4000, 0x4200, 0x4400, 0x4600,  // +2.0..+6.0
    0x8000, 0xB800, 0xBC00, 0xBE00,  // -0.0..-1.5
    0xC000, 0xC200, 0xC400, 0xC600,  // -2.0..-6.0
};
```

The inner loop for each FP4 byte:

```cuda
uint8_t b = bytes[k];
float w_lo = __half2float(__ushort_as_half(fp4_lut[b & 0xF])) * scale;
float w_hi = __half2float(__ushort_as_half(fp4_lut[b >> 4])) * scale;
sumf = fmaf(w_lo, x[j + k*2], sumf);
sumf = fmaf(w_hi, x[j + k*2 + 1], sumf);
```

Two problems:

### 1. Constant Memory Serialization

CUDA constant memory has a broadcast cache: when all threads in a warp access the **same** address, it broadcasts in one cycle. When threads access **different** addresses, accesses are serialized — up to 16 cycles for 16 unique indices.

FP4 weight values are essentially random across threads. Each thread decodes a different byte, producing different nibble indices. A warp of 32 threads accessing 16 possible LUT entries serializes badly. This turned every LUT lookup into a multi-cycle stall.

### 2. Thread Utilization

The kernel processed 32 FP4 elements per thread per iteration (16 packed bytes via `float4` load). For K=2560 (the most common dimension in the 4B model): only `2560 / 32 = 80` of 256 threads were active — 31% utilization.

Meanwhile, the fast FP16 lm_head kernel processed 2 elements per thread per iteration, keeping all 256 threads active for K=2560.

## The Fix: Shared Memory LUT

Moving the LUT from `__constant__` to `__shared__` memory eliminated the serialization:

```cuda
// Pre-computed as floats — eliminates per-element half→float conversion too
static __device__ __constant__ float fp4_lut_f32[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__global__ void kernel_gemv_fp4(...) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = fp4_lut_f32[threadIdx.x];
    __syncthreads();

    // Inner loop now uses shared memory:
    float w_lo = lut[b & 0xF] * scale;
    float w_hi = lut[b >> 4] * scale;
    ...
}
```

Shared memory has 32 banks. The 16-entry float LUT spans 16 banks. When 32 threads access random entries, worst case is 2-way bank conflicts (2 cycles) vs constant memory's 16 cycles. In practice, most accesses are conflict-free.

This also eliminates two per-element instructions: `__ushort_as_half()` and `__half2float()` — the LUT is pre-stored as `float`.

**This single change gave 3.1× improvement** (45 → 139 tok/s).

## Other Optimizations

These helped less individually but contributed together:

- **8 elements per thread** (down from 32): ensures all 256 threads are active for K=2560. The NVFP4 scale granularity is 16 elements, so 8 elements always falls within one scale block.
- **PTX E4M3 scale conversion**: `cvt.rn.f16x2.e4m3x2` hardware instruction instead of a branchy scalar function with `ldexpf`.
- **Vectorized `half2` input loads + `#pragma unroll`**: minor cleanup matching the FP8 kernel's style.

## A Formatting Bug

During the kernel rewrite, I reformatted the LUT array from one entry per line to four entries per line with inline comments:

```cuda
// BROKEN — // comments eat the rest of each line!
static __device__ __constant__ uint16_t fp4_lut[16] = {
    0x0000, // +0.0    0x3800, // +0.5    0x3C00, // +1.0    0x3E00, // +1.5
    ...
};
```

The `//` after the first value on each line turned everything else into a comment. The array had 4 values instead of 16, rest zero-initialized. Result: complete garbage output (Arabic/Chinese characters). Took a while to spot.

## Results

```
Before:   12.4 tok/s (no graph), 42.6 tok/s (with graph)
After:    139.1 tok/s (with graph + optimized kernel)
llama.cpp: 184 tok/s (Q4_K_M)
```

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Decode speed | 42.6 tok/s | 139.1 tok/s | **3.3×** |
| vs llama.cpp | 23% | 75.6% | — |
| FP4 GEMV bandwidth | ~13% of peak | ~42% of peak | **3.2×** |

The 0.8B FP8 model shows no regression: 317 tok/s.

## Why Not Faster?

llama.cpp's Q4_K kernel uses `dp4a` — a single instruction that computes 4 INT8 multiply-accumulates. They pre-quantize the input vector to INT8 and use integer arithmetic throughout the hot loop. This gives ~4× fewer instructions per weight byte compared to our FP4 LUT + FP32 FMA approach.

FP4 E2M1 values (0, ±0.5, ±1, ..., ±6) can be mapped to integers (multiply by 2 → all fit in INT8). A dp4a-based FP4 kernel is the natural next step to close the remaining 25% gap.
