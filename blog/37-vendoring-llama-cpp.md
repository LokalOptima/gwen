# 37. Vendoring llama.cpp and Dead Code Cleanup

*Eliminating the external dependency, plus removing 300+ lines of dead experiments.*

## The Problem

GWEN had two kernel files that compiled against external llama.cpp headers:

- **`fattn_mma.cu`** — MMA flash attention, using llama.cpp's `fattn-mma-f16.cuh` (stream-K dispatch, cp.async pipeline, Turing m16n8k16 MMA)
- **`gemm_mmq.cu`** — Fused K-quant GEMM, using llama.cpp's `mmq.cuh` (tile loading, dp4a dot products, stream-K scheduling)

Both required `LLAMA_CPP_PATH` pointing at a llama.cpp source tree, pulling in 17 headers totaling ~19K lines. This meant:
1. Builds failed without llama.cpp cloned nearby
2. The 3.7 GB llama.cpp repo was a hidden dependency
3. Header updates could silently break GWEN

## The Naive Approach (Failed)

First attempt: rewrite both kernels from scratch using GWEN's own `flash_attn_mma.cuh` (an 820-line simplified Turing-only kernel extracted earlier). This kernel worked for decode (N=1) but crashed with illegal shared memory access for prefill (N>1). The bugs were:

1. **Q pointer not offset by token tile** — `blockIdx.z` (token tile index) was computed but never used to advance the Q pointer
2. **Output layout wrong** — kernel wrote [head, token, dim] but caller expected [token, head, dim] (identical for N=1, wrong for N>1)
3. **No boundary guard** — Q loading accessed jc >= N for the last tile

After debugging with compute-sanitizer and tracing the stride math through the MMA tile types, it became clear that "fixing" a broken custom kernel was the wrong approach when a working one already existed.

## The Right Approach: Vendor the Headers

Copied all 17 llama.cpp header files into `include/gwen/llama/`:

```
include/gwen/llama/
  common.cuh          (1427 lines)   fattn-mma-f16.cuh  (1828 lines)
  fattn-common.cuh    (1036 lines)   mmq.cuh            (4097 lines)
  mma.cuh             (1409 lines)   vecdotq.cuh        (1237 lines)
  cp-async.cuh        (57 lines)     convert.cuh        (56 lines)
  quantize.cuh        (41 lines)     vendors/cuda.h     (23 lines)
  ggml.h              (2771 lines)   ggml-impl.h        (778 lines)
  ggml-common.h       (1889 lines)   ggml-cuda.h        (47 lines)
  ggml-backend.h      (373 lines)    ggml-alloc.h       (85 lines)
  gguf.h              (202 lines)
```

Updated CMakeLists.txt to point include paths at `include/gwen/llama/` instead of the external llama.cpp tree. Removed the `LLAMA_CPP_PATH` variable entirely. Build now requires zero external dependencies beyond CUDA and CUTLASS.

## Dead Code Cleanup

With the vendoring done, I swept the codebase for dead code accumulated during optimization experiments:

**Removed from `inference.cu`:**
- `kernel_gated_rmsnorm_quantize_q8_1` + batch2 variant (~160 lines) — fused RMSNorm+Q8_1 kernels from a failed optimization experiment. CUDA graphs already minimize launch overhead, and parallelizing Q8_1 across 16 blocks on different SMs beats serializing into a single fused kernel.
- `kernel_dequant_iq4xs_to_f16` (~45 lines) — IQ4_XS dequant-to-FP16 kernel, replaced by the MMQ GEMM path which preserves dp4a integer precision.

**Removed from `rmsnorm.cu`:**
- `kernel_rmsnorm_f32_input_quantize_q8_1` + batch2 variant + wrappers (~120 lines) — F32-input fused RMSNorm+Q8_1, never called after the 32-thread version proved slower than separate kernels.

**Removed from `kernels.h`:**
- Declarations for all removed functions

**Deleted:**
- `include/gwen/flash_attn_mma.cuh` (820 lines) — the broken GWEN native flash attention kernel. Not included from any source file. If we want a custom kernel in the future, it needs to be rewritten with correct multi-token support.

**Total: ~1,145 lines of dead code removed.**

## Correctness Test

Added `scripts/test_correctness.sh` — a self-bootstrapping test suite that runs after every change:

1. **Generation correctness** — greedy decode vs golden reference tokens
2. **Prefill smoke test** — gwen_bench pp32 exits 0
3. **Decode smoke test** — gwen_bench tg32 exits 0
4. **Determinism** — 3 identical runs produce identical output

## Results

Performance unchanged (same kernels, just vendored):

| Test | GWEN | llama.cpp | Delta |
|------|------|-----------|-------|
| pp512 | 5,997 tok/s | 5,740 | **+4.5%** |
| tg128 | 124.8 tok/s | 115.4 | **+8.1%** |

All 4 correctness tests pass. Build no longer requires llama.cpp source tree.

## Next Steps

The vendored headers are 19K lines of multi-arch code (AMD, Volta, Ampere, Blackwell paths) of which only ~40% is SM_120-relevant. Stripping the dead architecture paths would cut this to ~8K lines.
