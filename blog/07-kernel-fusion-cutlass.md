# Kernel Fusion + CUTLASS: 452 to 493 tok/s

*Blog post #7 in the GWEN series — death by a thousand launches*

## The Problem: 370 Kernel Launches

After the dp4a revolution in post #6, GWEN was at 452 tok/s with a 1.71ms forward pass. The profiler breakdown was:

| Component | Time | % Forward |
|-----------|------|-----------|
| GEMV (all 24 layers + LM head) | 1.17 ms | 68.5% |
| DeltaNet recurrence (18 layers) | 0.15 ms | 8.8% |
| **Other (norms, activations, quantize, residual)** | **0.39 ms** | **22.7%** |

That "other" category is ~370 small kernel launches per decode step. Each one is tiny — an RMSNorm over 1024 elements, a SwiGLU activation, a Q8_1 quantization, a residual add — but each launch incurs ~1-2 microseconds of overhead on top of the actual compute. At 370 launches, that's 370-740 microseconds of pure overhead.

The fix is obvious: fuse these small kernels together. The question is which fusions give the most bang for the buck.

## Fusion 1A: GEMV + Residual Add

Every output projection and FFN-down GEMV is followed by `gwen_add_inplace(y, residual, 1024)`. That's a separate kernel launch that reads 2KB, adds, and writes 2KB. Completely pointless when the GEMV kernel already has the result in registers.

The fix is trivial — add a `const half* residual` parameter to each dp4a kernel. The only change is in the final reduction write:

```cpp
if (threadIdx.x == 0) {
    if (residual)
        y[row] = __float2half(sumf + __half2float(residual[row]));
    else
        y[row] = __float2half(sumf);
}
```

This is a single branch, executed once per row, completely outside the hot loop. Zero overhead for the non-fused path. Eliminates 48 `add_inplace` kernel launches (2 per layer x 24 layers).

## Fusion 1B: SwiGLU + Q8_1 Quantize

Every FFN computes `SwiGLU(gate, up) → FP16 → quantize_q8_1 → dp4a GEMV`. The FP16 intermediate is 7KB (3584 x FP16) that gets written and immediately read back. Fusing SwiGLU with Q8_1 quantization skips this entirely.

Each warp handles one Q8_1 block (32 elements):

```cpp
// Compute SwiGLU in FP32
float g = __half2float(gate[base]);
float u = __half2float(up[base]);
float sig = 1.0f / (1.0f + expf(-g));
float val = g * sig * u;

// Quantize directly — warp shuffle for amax
float amax = fabsf(val);
for (int offset = 16; offset > 0; offset >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
float d = amax / 127.0f;
int8_t q = (int8_t)roundf(val * (d > 0 ? 127.0f / amax : 0));
```

Eliminates 48 kernel launches + 168KB of memory traffic per decode.

## Fusion 1C: RMSNorm + Q8_1 Quantize

This is the most impactful fusion structurally. The pattern `rmsnorm → quantize → GEMV` appears 5 times per layer iteration (pre-attention norm, post-attention norm, and the final LM head norm). Previously this was:

1. `kernel_rmsnorm_f32w<<<1, 32>>>` — 32 threads, computes norm
2. `kernel_quantize_q8_1<<<4, 256>>>` — 8 warps, quantizes 32 blocks

Fused into a single `<<<1, 256>>>` kernel (8 warps, 256 threads):

- **Phase 1**: All 256 threads compute partial sum-of-squares for 1024 elements (4 per thread). Cross-warp reduce via shared memory to get `rms_inv`.
- **Phase 2**: Each of 8 warps handles 4 Q8_1 blocks: apply norm + weight, find amax via warp shuffle, quantize.

The DeltaNet pre-attention norm is special — `kernel_compute_gate_beta` reads the FP16 normalized output directly. So that site passes a non-null `y_fp16` pointer; the other 4 sites pass `nullptr` and skip the FP16 write entirely.

## Fusion 1D: Gated RMSNorm + Q8_1 Quantize

DeltaNet's output path does per-head gated RMSNorm: `output = RMSNorm_per_head(x) * SiLU(gate)`, then quantizes for the output projection GEMV. Previously:

1. `kernel_gated_rmsnorm<<<16, 32>>>` — 1 warp per head
2. `kernel_quantize_q8_1<<<8, 256>>>` — quantizes 2048 elements (64 blocks)

Fused into `<<<16, 128>>>` (4 warps per head):

- 128 threads per head, each head has `dim_per_head=128` elements = exactly 4 Q8_1 blocks
- Phase 1: Sum-of-squares across 128 elements (1 per thread in the common case), 4-warp reduction
- Phase 2: Each warp handles one Q8_1 block — norm, weight, SiLU(gate), quantize

Perfect work distribution: 4 warps, 4 blocks, no waste.

## Dropping cuBLAS: CUTLASS 2.x GEMM

While the fusions target decode, I also replaced cuBLAS with CUTLASS for the prefill GEMM path. The motivation is dependency reduction — cuBLAS is a ~200MB shared library that we were linking just for a single HGEMM call.

CUTLASS 2.x `device::Gemm` targeting Sm80 generates `mma.sync` instructions that work perfectly on Sm120 (Blackwell consumer):

```cpp
using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,      // A (dequantized W)
    cutlass::half_t, cutlass::layout::ColumnMajor,    // B (input x)
    cutlass::half_t, cutlass::layout::ColumnMajor,    // C (output y)
    float,                                             // accumulator
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>>;
```

Same dequant-first pipeline (dequant weights to FP16 temp, then CUTLASS GEMM), just without the cuBLAS dependency. All 9 weight matrix tests pass with max diff < 1e-3 vs the legacy GEMV path.

## Results

Forward pass profile (100 runs):

| Component | Before | After |
|-----------|--------|-------|
| Forward pass | 1.71 ms | **1.70 ms** |
| GEMV total | 1.17 ms | 1.18 ms |
| Non-GEMV overhead | 0.39 ms (22.7%) | 0.38 ms (22.2%) |
| DeltaNet recurrence | 0.15 ms | 0.15 ms |

End-to-end decode (100 tokens):

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| GWEN tok/s | 452 | **493** | **+9%** |
| llama.cpp tok/s | 434 | 445 | — |
| vs llama.cpp | +4.2% | **+10.6%** | |
| Per-token latency | 2.21 ms | **2.03 ms** | -8% |

30/30 exact greedy token match maintained throughout.

The forward pass improvement is modest (~10 microseconds), but the end-to-end decode improvement is significant (+9%). The difference comes from CUDA graph efficiency — fewer kernel nodes in the graph means less replay overhead. The graph went from ~370 nodes to ~250 nodes, saving about 0.18ms per graph replay.

## What's Left

The forward pass is now 1.70ms against a theoretical minimum of 0.58ms — 34.2% bandwidth efficiency. The remaining gap:

- **LM head GEMV**: 0.25ms at 93% bandwidth — essentially optimal
- **Layer GEMVs**: 0.93ms at 41-55% bandwidth — room for improvement but diminishing returns
- **DeltaNet recurrence**: 0.15ms — inherently sequential
- **Everything else**: 0.38ms — further fusion opportunities are marginal

The biggest remaining lever would be persistent kernels that eliminate the CUDA graph overhead entirely (~0.33ms), but that's a major architectural change. For now, 493 tok/s on a consumer GPU running a hybrid DeltaNet+Transformer model feels pretty good.
