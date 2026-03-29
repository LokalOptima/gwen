# 39. Dropping cuBLAS for CUTLASS

*Profiling the last external dependency, matching it with a header-only replacement, and ending up faster.*

## The Problem

GWEN's only runtime dependency beyond cudart was cuBLAS — used for a single `cublasHgemm` call in the prefill path. This call handled FP16 weight matrices (`ssm_alpha` and `ssm_beta` in DeltaNet layers), routing through `do_gemm` alongside our custom MMQ kernels for K-quant weights.

The goal: drop cuBLAS entirely so GWEN links only against cudart. CUTLASS is already a submodule (header-only), so if we can match cuBLAS performance with a CUTLASS kernel, the dependency just disappears.

## Understanding the Problem Shape

First question: what are we actually computing? The F16 weights are `ssm_alpha` and `ssm_beta`, shaped `[16, 4096]` — 16 output rows (DeltaNet V heads), 4096 input features (hidden size). With N tokens in a prefill batch, this is:

```
Y[16, N] = W[16, 4096] @ X[4096, N]
```

M=16 is tiny. This is a "skinny GEMM" — basically N parallel dot products with 16 output elements each.

## Attempt 1: Hand-Written Kernels

Since M=16 is so small, my first instinct was hand-written CUDA kernels. I benchmarked five variants against cuBLAS at N=64 through N=2048:

| Variant | N=512 (µs) | vs cuBLAS |
|---|---|---|
| **cuBLAS** | **8.2** | 1.0× |
| skinny_float4 (1 block/token, float4 loads) | 12.3 | 1.5× slower |
| warp_per_token (1 warp/token) | 14.3 | 1.7× slower |
| looped GEMV (N kernel launches) | 4192 | 511× slower |
| CUTLASS 2.x (various tiles) | 26.6 | 3.2× slower |
| shared-memory W reuse | 108.7 | 13× slower |

The hand-written kernels couldn't touch cuBLAS. The shared-memory approach was the worst — the sync overhead dwarfed any benefit from W reuse (W is only 128 KB, fits in L2 anyway).

## Profiling cuBLAS with ncu

Rather than guessing, I profiled the cuBLAS kernel:

```
$ sudo ncu --set full ./build/profile_skinny cublas
```

The cuBLAS GEMM kernel turned out to be... a CUTLASS kernel:

```
cutlass_80_tensorop_h16816gemm_64x64_64x6_tn_align8
Grid: (8, 1, 5) × Block: (128, 1, 1)
```

Key details:
- **Split-K = 5** — K=4096 split across 5 blocks per output tile
- **64×64×64 tile, 6 pipeline stages** — 98 KB shared memory
- **mma.sync m16n8k16** — tensor cores via SM_80 backward-compatible path
- **80 registers/thread** — no spills
- **9.0µs GEMM + 2.7µs reduce** — two-kernel pipeline

## The SM_120 CUTLASS Situation

I initially expected to use CUTLASS 4.x's native SM_120 path. Checking the source:

```cpp
// sm120_mma_builder.inl, line 82:
static_assert(detail::is_sm10x_f8f6f4_element<ElementA>() &&
              detail::is_sm10x_f8f6f4_element<ElementB>(),
    "SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA.");
```

SM_120's native CUTLASS path only supports FP8/FP6/FP4. No FP16. But CUTLASS 2.x's `device::Gemm` with `arch::Sm80` works fine on SM_120 — it uses `mma.sync` PTX which is backward-compatible. This is exactly what cuBLAS uses internally.

## Matching cuBLAS: The Register Problem

I set up the exact same CUTLASS config cuBLAS chose and profiled both:

| Metric | Our CUTLASS | cuBLAS |
|---|---|---|
| Tile | 64×64×64 | 64×64×64 |
| Stages | 6 | 6 |
| Grid | (8, 1, 5) | (8, 1, 5) |
| **Registers** | **104** | **80** |
| **Duration** | **16.6µs** | **9.0µs** |
| Instructions | 360K | 341K |
| DRAM bytes | 4.34 MB | 4.34 MB |
| Shared memory | 98 KB | 98 KB |

Same template, same tile, same grid — but nvcc gives us 104 registers where NVIDIA's pre-compiled cuBLAS binary gets 80. The extra registers mean worse IPC (0.33 vs 0.52) despite identical occupancy (both limited by 98 KB shared memory to 1 block/SM).

Forcing `-maxrregcount=80` just caused register spills to local memory — slower, not faster.

## The Solution: Smaller Tile, Natural 80 Registers

Different tile configurations compile to different register counts:

| Config | Registers | Smem | Duration (N=512) |
|---|---|---|---|
| 64×64×64, 6 stages | 104 | 98 KB | 16.6µs |
| 64×64×32, 4 stages | **80** | **33 KB** | **12.3µs** |
| 64×128×32, 3 stages | 146 | — | worse |

The **64×64×32 tile with 4 stages** naturally compiles to exactly 80 registers with no spills. Smaller K-tile means less register pressure from the pipeline buffers. The tradeoff is less latency hiding (33 KB vs 98 KB shared memory), but at these problem sizes it doesn't matter much.

With split-K=5 and the workspace memset removed from the hot path (CUTLASS serial split-K handles synchronization via semaphores internally), this configuration hits **12.3µs** vs cuBLAS's **8.2µs**.

The remaining gap is cuBLAS's two-kernel pipeline: its GEMM and reduce kernels overlap on the GPU, while CUTLASS serial split-K serializes the reduce phase in-kernel. Not worth chasing — in the full model, these 36 GEMM calls per prefill represent a fraction of a percent of total time.

## Integration

The change was surgical. Three files:

```cpp
// inference.cu — CUTLASS type definition
using CutlassPrefillGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,      // W [M,K]
    cutlass::half_t, cutlass::layout::ColumnMajor,    // X [K,N]
    cutlass::half_t, cutlass::layout::ColumnMajor,    // Y [M,N]
    float,                                             // accumulator
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,             // threadblock
    cutlass::gemm::GemmShape<32, 32, 32>,             // warp
    cutlass::gemm::GemmShape<16, 8, 16>,              // instruction
    EpilogueOp, Swizzle, 4, 8, 8, true>;              // 4 stages, split-K
```

```cmake
# CMakeLists.txt — before:
target_link_libraries(gwen_core CUDA::cublas)
# after: (line removed entirely)
```

## Results

```
| model              |   size |  params | backend |    test |           t/s |
| ------------------ | -----: | ------: | ------- | ------: | ------------: |
| qwen35 9B Q4_K_XL  | 5.55 G |  8.95 B |    CUDA |   pp512 | 5980 ± 21     |
| qwen35 9B Q4_K_XL  | 5.55 G |  8.95 B |    CUDA |   tg128 |  127 ± 1.3    |
```

Prefill **improved 9%** (5481 → 5980 tok/s). The CUTLASS kernel is slightly slower per-call than cuBLAS, but avoids cuBLAS's host-side dispatch overhead across 36 calls per prefill pass. The net effect is faster.

Decode is completely unaffected — the F16 GEMM path is prefill-only. Decode uses our hand-written dp4a GEMV kernels.

```
$ ldd build/gwen | grep cublas
(empty)
```

GWEN now links only against cudart. The entire CUTLASS dependency is header-only — zero runtime libraries beyond the CUDA runtime itself.
