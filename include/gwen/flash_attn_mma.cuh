#pragma once
// Flash attention kernel — Turing mma.sync path, D=256 only.
//
// Extracted and simplified from llama.cpp:
//   ggml/src/ggml-cuda/fattn-mma-f16.cuh
//
// What was removed:
//   - Volta/AMD/Blackwell branches
//   - ALiBi (max_bias, slope, m0, m1, n_head_log2) — hardcoded slope=1.0f
//   - logit_softcap — removed (hardcode use_logit_softcap=false)
//   - attention sinks (sinks_f parameter)
//   - KV_max (dynamic sequence length clamping)
//   - Stream-K scheduling — replaced with simple 1-D grid over (head, sequence) tiles
//   - Quantized K/V paths — FP16 half2 only
//   - ncols2 (GQA batching) — hardcoded ncols2=1 for decode (1 Q token per KV head)
//   - needs_fixup / is_fixup — simple grid, each block owns exactly one output tile
//
// What remains:
//   - flash_attn_ext_f16_load_tile  — K/V tile loader (cp.async or sync)
//   - flash_attn_ext_f16_iter       — core KQ→softmax→VKQ loop
//   - flash_attn_ext_f16_process_tile — block-level orchestration
//   - gwen_flash_attn_mma_f16       — kernel entry point
//
// Configuration (Turing, D=256, ncols=8):
//   nthreads = 128 (4 warps), nbatch_fa = 64, nbatch_K2 = 128, nbatch_V2 = 128,
//   nbatch_combine = 128, Q_in_reg = true, nstages = 2 (cp.async pipeline)
//
// Kernel signature:
//   gwen_flash_attn_mma_f16<DKQ=256, DV=256, ncols=8><<<grid, 128, smem>>>( ... )
//
// Grid: (ne02 * ne03, 1, 1)  i.e. one block per (Q-head, sequence) pair.
// Each block processes the full KV sequence for one query position.

#include "mma_types.cuh"
#include "cp_async.cuh"

#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

using namespace gwen_mma;

// ---- Constants (from fattn-common.cuh) ---------------------------------------
#define GWEN_SOFTMAX_FTZ_THRESHOLD  -20.0f   // exp() below this is flushed to 0
#define GWEN_FATTN_KQ_MAX_OFFSET    (3.0f * 0.6931f)  // offset to reduce exp() overflow

// ---- Hardcoded configuration for DKQ=256, DV=256, ncols=8 (Turing) ---------
// From ggml_cuda_fattn_mma_get_config_turing(256, 256, 8):
//   nthreads=128, occupancy=2, nbatch_fa=64, nbatch_K2=128, nbatch_V2=128,
//   nbatch_combine=128, nstages_target=2, Q_in_reg=true

static constexpr int  GWEN_FA_DKQ            = 256;
static constexpr int  GWEN_FA_DV             = 256;
static constexpr int  GWEN_FA_NCOLS          = 8;    // Q tokens per block (decode = 1 or 8)
static constexpr int  GWEN_FA_NWARPS         = 4;    // nthreads / 32
static constexpr int  GWEN_FA_NBATCH_FA      = 64;   // KV rows per softmax step
static constexpr int  GWEN_FA_NBATCH_K2      = 128;  // K half2 values loaded per iter (= DKQ/2)
static constexpr int  GWEN_FA_NBATCH_V2      = 128;  // V half2 values loaded per iter (= DV/2)
static constexpr int  GWEN_FA_NBATCH_COMBINE = 128;  // VKQ half2 values combined per stride
static constexpr bool GWEN_FA_Q_IN_REG       = true;
// nstages=2 (double-buffer K tile with cp.async, V loaded synchronously per VKQ batch)
static constexpr int  GWEN_FA_NSTAGES        = 2;

// Tile strides in half2 units (include 4-element (8-byte) padding to avoid bank conflicts)
static constexpr int GWEN_STRIDE_TILE_Q = GWEN_FA_DKQ / 2 + 4;          // 132
static constexpr int GWEN_STRIDE_TILE_K = GWEN_FA_NBATCH_K2 + 4;        // 132
static constexpr int GWEN_STRIDE_TILE_V = GWEN_FA_NBATCH_V2 + 4;        // 132

// Tile types for ncols=8 (Turing narrow path):
//   cols_per_warp = T_B_KQ::I = 8
//   np = nwarps * cols_per_warp / ncols = 4 * 8 / 8 = 4  (4 warps per Q column)
// KQ:  K_A tile<16,8,half2>, Q_B tile<8,8,half2>, KQ_C tile<16,8,float>
// VKQ: V_A tile<16,8,half2>, B_VKQ tile<8,8,half2>, VKQ_C tile<16,4,half2>
using T_A_KQ  = tile_16_8_h2;  // K fragment (row-major)
using T_B_KQ  = tile_8_8_h2;   // Q fragment (col-major); T_B_KQ::I == 8 == cols_per_warp
using T_C_KQ  = tile_16_8_f32; // KQ accumulator (row-major)
using T_A_VKQ = tile_16_8_h2;  // V^T fragment
using T_B_VKQ = tile_8_8_h2;   // softmax(KQ) fragment
using T_C_VKQ = tile_16_4_h2;  // VKQ accumulator

static constexpr int GWEN_COLS_PER_WARP   = T_B_KQ::I;  // = 8
static constexpr int GWEN_COLS_PER_THREAD = 2;           // Turing: 2 KQ columns per thread
// np = number of warps working on a single Q column
static constexpr int GWEN_NP              =
    GWEN_FA_NWARPS * GWEN_COLS_PER_WARP / GWEN_FA_NCOLS;  // = 4

// ---- K/V tile loader ---------------------------------------------------------
//
// flash_attn_ext_f16_load_tile:
//   Loads a [nbatch_fa, nbatch_KV2] sub-tile of K or V from global memory
//   (provided as half2*) into shared memory.
//
//   stride_tile  — shared memory stride in half2 units (includes padding)
//   use_cp_async — use cp.async.cg (async, no OOB check) or synchronous load
//   oob_check    — when true, clamp rows >= i_sup to zero (last iteration guard)

template<int stride_tile, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV,
        half2 * const __restrict__ tile_KV,
        const int nbatch_KV2,
        const int stride_KV,
        const int i_sup) {

    constexpr int warp_size   = WARP_SIZE;
    constexpr int nwarps      = GWEN_FA_NWARPS;

    if constexpr (use_cp_async) {
        static_assert(!oob_check, "OOB check not compatible with cp.async");
        constexpr int preload       = 64;
        constexpr int h2_per_chunk  = 16 / sizeof(half2);  // 4 half2 per 16-byte chunk

        const unsigned int tile_KV_32 = gwen_cvta_generic_to_shared(tile_KV);

        // Load with decreasing stride (coalescing at different granularities).
        auto load = [&] __device__ (auto n) {
            const int stride_k = warp_size >> n;
            const int k0_start = stride_k == warp_size ? 0
                                 : nbatch_KV2 - nbatch_KV2 % (2 * stride_k);
            const int k0_stop  = nbatch_KV2 - nbatch_KV2 % (1 * stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) return;

            #pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps * stride_i) {
                const int i = i0 + threadIdx.y * stride_i
                    + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);
                if (i0 + nwarps * stride_i > nbatch_fa && i >= nbatch_fa) break;

                #pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size
                                        ? threadIdx.x : threadIdx.x % stride_k);
                    // Each cp.async copies 16 bytes = h2_per_chunk half2 values.
                    cp_async_cg_16<preload>(
                        tile_KV_32 + i * (stride_tile * sizeof(half2)) + k * 16,
                        KV + i * stride_KV + k * h2_per_chunk);
                }
            }
        };
        gwen_cuda_unroll<6>{}(load);

    } else {
        auto load = [&] __device__ (const int n) {
            const int stride_k = warp_size >> n;
            const int k0_start = stride_k == warp_size ? 0
                                 : nbatch_KV2 - nbatch_KV2 % (2 * stride_k);
            const int k0_stop  = nbatch_KV2 - nbatch_KV2 % (1 * stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) return;

            #pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps * stride_i) {
                const int i = i0 + threadIdx.y * stride_i
                    + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);
                if (i0 + nwarps * stride_i > nbatch_fa && i >= nbatch_fa) break;

                #pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size
                                        ? threadIdx.x : threadIdx.x % stride_k);
                    tile_KV[i * stride_tile + k] =
                        (!oob_check || i < i_sup)
                        ? KV[i * stride_KV + k]
                        : make_half2(0.0f, 0.0f);
                }
            }
        };
        gwen_cuda_unroll<4>{}(load);
    }
}

// ---- Helper: tile<16,8,float> -> tile<16,4,half2> (for narrow-path KQ->B conversion) -----
// Packs consecutive float pairs into half2, then applies movmatrix transpose.
// Result is a tile<16,4,half2> ready to be passed to get_transposed() -> tile<8,8,half2>.
static __device__ __forceinline__ tile_16_4_h2
gwen_half2_from_f32(const tile_16_8_f32 & tf) {
    tile_16_4_h2 ret;
    #pragma unroll
    for (int l0 = 0; l0 < tile_16_8_f32::ne; l0 += 2) {
        ret.x[l0 / 2] = make_half2(tf.x[l0 + 0], tf.x[l0 + 1]);
    }
    return ret;
}

// ---- Core iteration ----------------------------------------------------------
//
// flash_attn_ext_f16_iter:
//   Processes one batch of nbatch_fa KV rows:
//     1. Load K tile (or use pre-loaded tile for nstages>1)
//     2. Compute KQ = K @ Q^T  (MMA)
//     3. Online softmax update (max + rowsum)
//     4. Rescale previous VKQ accumulators
//     5. Convert KQ scores to B tiles
//     6. Load V tile
//     7. Compute VKQ += V^T @ softmax(KQ)  (MMA)
//
// Parameters:
//   Q_f2        — Q in float2 (global, read-only)
//   K_h2        — start of K for this sequence (global)
//   V_h2        — start of V for this sequence (global)
//   scale       — attention scale (1/sqrt(D))
//   stride_K    — K row stride in half2
//   stride_V    — V row stride in half2
//   tile_Q      — shared memory for Q (and later combine buffer)
//   tile_K      — shared memory for current K batch
//   tile_V      — shared memory for current V batch
//   Q_B         — per-warp Q register tile (only used when Q_in_reg)
//   VKQ_C       — VKQ accumulator (one tile<16,4,half2> per DV/T_C_VKQ::I step)
//   KQ_max      — running max per Q column (cols_per_thread values)
//   KQ_rowsum   — running row-sum per Q column
//   kb0         — KV batch index (row start = kb0 * nbatch_fa)
//   k_VKQ_sup   — number of valid rows in this batch (< nbatch_fa on last iter)

static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const int    stride_K,
        const int    stride_V,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        T_B_KQ       * const __restrict__ Q_B,          // DKQ/(2*T_B_KQ::J) = 16 tiles
        T_C_VKQ      * const __restrict__ VKQ_C,         // DV/T_C_VKQ::I tiles
        float        * const __restrict__ KQ_max,        // [cols_per_thread]
        float        * const __restrict__ KQ_rowsum,     // [cols_per_thread]
        const int    kb0,
        const int    k_VKQ_sup,
        const bool   last_iter) {

    constexpr int warp_size     = WARP_SIZE;
    constexpr int nwarps        = GWEN_FA_NWARPS;
    constexpr int nbatch_fa     = GWEN_FA_NBATCH_FA;
    constexpr int nbatch_K2     = GWEN_FA_NBATCH_K2;   // = DKQ/2 = 128
    constexpr int nbatch_V2     = GWEN_FA_NBATCH_V2;   // = DV/2  = 128
    constexpr int nstages       = GWEN_FA_NSTAGES;
    constexpr int np            = GWEN_NP;
    constexpr int cols_per_warp = GWEN_COLS_PER_WARP;
    constexpr int cols_per_thread = GWEN_COLS_PER_THREAD;

    const int k_VKQ_0 = kb0 * nbatch_fa;

    // With nstages==2: the K tile for this iteration was already loaded by the previous
    // iteration (or the pre-load before the loop).  We first issue the V load for this
    // iteration, wait for the K tile to arrive, then do KQ MMA, then wait for V.
    if constexpr (nstages > 1) {
        // Wait for K tile issued by the previous iteration's tail (or the pre-loop for iter 0).
        cp_async_wait_all();
        __syncthreads();
        // While K MMA is being prepared, issue V load for the current iteration asynchronously.
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        flash_attn_ext_f16_load_tile<GWEN_STRIDE_TILE_V, nbatch_fa, use_cp_async, oob_check>(
            V_h2 + (int64_t)k_VKQ_0 * stride_V, tile_V, nbatch_V2, stride_V, k_VKQ_sup);
    } else {
        // nstages==1: load K synchronously before KQ MMA.
        constexpr bool use_cp_async = false;
        constexpr bool oob_check    = true;
        flash_attn_ext_f16_load_tile<GWEN_STRIDE_TILE_K, nbatch_fa, use_cp_async, oob_check>(
            K_h2 + (int64_t)k_VKQ_0 * stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
        __syncthreads();
    }

    // ---- KQ MMA: iterate over D in chunks of nbatch_K2 (= DKQ/2 = 128) ----
    // KQ_C accumulates [nbatch_fa/(np*T_C_KQ::I)] tiles per warp.
    // T_C_KQ::I = 16, np = 4  => 64/(4*16) = 1 tile per warp.
    constexpr int n_KQ_C_tiles = nbatch_fa / (np * T_C_KQ::I);  // = 1
    T_C_KQ KQ_C[n_KQ_C_tiles];

    // For Q_in_reg=true, Q_B is already loaded. Iterate K over D dimension.
    #pragma unroll
    for (int k0_start = (nbatch_K2 - 1) - (nbatch_K2 - 1) % nbatch_K2;
         k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < (GWEN_FA_DKQ / 2)
                            ? k0_start + nbatch_K2 : GWEN_FA_DKQ / 2;
        // (Since nbatch_K2 == DKQ/2, this loop executes exactly once.)
        (void)k0_stop;

        // nstages==1: sync was done above; nstages==2: K tile was pre-loaded.
        #pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np * T_A_KQ::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np) * T_A_KQ::I;
            #pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                T_A_KQ K_A;
                load_ldmatrix(K_A,
                    tile_K + i_KQ_0 * GWEN_STRIDE_TILE_K + (k_KQ_0 - k0_start),
                    GWEN_STRIDE_TILE_K);
                // Narrow path (cols_per_warp==8): mma(KQ_C, K_A, Q_B)
                mma(KQ_C[i_KQ_00 / (np * T_A_KQ::I)], K_A,
                    Q_B[k_KQ_0 / T_A_KQ::J]);
            }
        }

        if constexpr (nstages <= 1) {
            __syncthreads();
        }
    }

    // ---- Online softmax ------------------------------------------------------
    // cols_per_warp==8: KQ values indexed by element index l % 2 == col.
    float KQ_max_new[cols_per_thread];
    #pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    // Find per-column maximum across all KQ_C tiles.
    #pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np * T_C_KQ::I) {
        #pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (k0 + (threadIdx.y % np) * T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                // Turing narrow: col index from l % 2
                const int KQ_idx = l % 2;
                KQ_max_new[KQ_idx] = fmaxf(
                    KQ_max_new[KQ_idx],
                    KQ_C[k0 / (np * T_C_KQ::I)].x[l] + GWEN_FATTN_KQ_MAX_OFFSET);
            }
        }
    }

    // Reduce max across 8 threads that own data for the same Q column.
    // For the narrow Turing path, values are spread across 8 threads (shfl with offsets 16..4).
    #pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        #pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col],
                __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
        }
    }

    // Compute exp() and accumulate row sums.
    #pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np * T_C_KQ::I) {
        #pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (k0 + (threadIdx.y % np) * T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                const int KQ_idx = l % 2;
                KQ_C[k0 / (np * T_C_KQ::I)].x[l] = expf(
                    KQ_C[k0 / (np * T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                KQ_rowsum_add[KQ_idx] += KQ_C[k0 / (np * T_C_KQ::I)].x[l];
            } else {
                KQ_C[k0 / (np * T_C_KQ::I)].x[l] = 0.0f;
            }
        }
    }

    // ---- Rescale VKQ accumulators and update running statistics -------------
    {
        float KQ_max_scale[cols_per_thread];
        #pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(diff);
            KQ_max[col]       = KQ_max_new[col];
            // Flush scale to zero if diff is below FTZ threshold (avoids NaN).
            *((uint32_t *) &KQ_max_scale[col]) *=
                (diff >= GWEN_SOFTMAX_FTZ_THRESHOLD);
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col]
                           + KQ_rowsum_add[col];
        }

        // Scale VKQ accumulator (narrow path: single half2 scale per col pair).
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
        constexpr int n_VKQ_C = GWEN_FA_DV / T_C_VKQ::I;
        #pragma unroll
        for (int i = 0; i < n_VKQ_C; ++i) {
            #pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
    }

    // ---- Convert KQ_C (float row-major) to B tiles (half2) for VKQ MMA -----
    // Narrow path (cols_per_warp==8): use get_half2 then get_transposed.
    constexpr int n_B = nbatch_fa / (np * 2 * T_B_VKQ::J);  // = 64/(4*2*8) = 1
    T_B_VKQ B[n_B];
    #pragma unroll
    for (int k = 0; k < n_B; ++k) {
        B[k] = get_transposed(
            gwen_half2_from_f32(KQ_C[k]));
    }

    // ---- Load V and compute VKQ MMA ----------------------------------------
    if constexpr (nstages > 1) {
        // Wait for V tile that was issued at the start of this iteration.
        cp_async_wait_all();
        __syncthreads();

        // Pre-load K tile for next iteration asynchronously (overlaps with VKQ MMA).
        if (!last_iter) {
            constexpr bool use_cp_async = true;
            constexpr bool oob_check    = false;
            constexpr int  k_sup_next   = nbatch_fa;
            flash_attn_ext_f16_load_tile<GWEN_STRIDE_TILE_K, nbatch_fa,
                                         use_cp_async, oob_check>(
                K_h2 + (int64_t)(k_VKQ_0 + nbatch_fa) * stride_K,
                tile_K, nbatch_K2, stride_K, k_sup_next);
        }
    }

    // Iterate over DV in chunks of 2*nbatch_V2.
    #pragma unroll
    for (int i0_start = 0; i0_start < GWEN_FA_DV; i0_start += 2 * nbatch_V2) {
        const int i0_stop = i0_start + 2 * nbatch_V2;

        if constexpr (nstages <= 1) {
            constexpr bool use_cp_async = false;
            constexpr bool oob_check    = true;
            flash_attn_ext_f16_load_tile<GWEN_STRIDE_TILE_V, nbatch_fa,
                                         use_cp_async, oob_check>(
                V_h2 + (int64_t)k_VKQ_0 * stride_V + i0_start / 2,
                tile_V, (i0_stop - i0_start) / 2, stride_V, k_VKQ_sup);
            __syncthreads();
        }

        // VKQ MMA: stride in DV == T_C_VKQ::I (narrow path, cols_per_warp==8)
        constexpr int i0_stride_vkq = T_C_VKQ::I;  // = 16
        #pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride_vkq) {
            #pragma unroll
            for (int k00 = 0; k00 < nbatch_fa / 2; k00 += np * T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np) * T_A_VKQ::J;
                T_A_VKQ A;
                // V is stored transposed in shared memory; use ldmatrix_trans.
                load_ldmatrix_trans(
                    A,
                    tile_V + 2 * k0 * GWEN_STRIDE_TILE_V + (i_VKQ_0 - i0_start) / 2,
                    GWEN_STRIDE_TILE_V);
                mma(VKQ_C[i_VKQ_0 / i0_stride_vkq],
                    A, B[k00 / (np * T_A_VKQ::J)]);
            }
        }

        if constexpr (nstages <= 1) {
            __syncthreads();
        }
    }
}

// ---- Block-level tile processor ----------------------------------------------
//
// flash_attn_ext_f16_process_tile:
//   Orchestrates the full attention computation for one Q tile (ncols Q tokens)
//   over the full KV sequence [kb0_start, kb0_stop).
//
//   On completion, the output (float) is written to dst[head_offset * DV/2 + k].
//   No fixup / partial-tile combining needed (simple grid, one block per output tile).

static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,       // Q for this head
        const half2  * const __restrict__ K_h2,       // K for this head
        const half2  * const __restrict__ V_h2,       // V for this head
        float2       * const __restrict__ dstk,        // output [ncols * DV/2]
        const float  scale,
        const int    stride_Q,   // Q row stride in float2
        const int    stride_K,   // K row stride in half2
        const int    stride_V,   // V row stride in half2
        const int    ne11,       // KV sequence length
        const int    kb0_start,
        const int    kb0_stop) {

    constexpr int warp_size     = WARP_SIZE;
    constexpr int nwarps        = GWEN_FA_NWARPS;
    constexpr int DKQ           = GWEN_FA_DKQ;
    constexpr int DV            = GWEN_FA_DV;
    constexpr int ncols         = GWEN_FA_NCOLS;
    constexpr int nbatch_fa     = GWEN_FA_NBATCH_FA;
    constexpr int nbatch_K2     = GWEN_FA_NBATCH_K2;
    constexpr int nbatch_V2     = GWEN_FA_NBATCH_V2;
    constexpr int nbatch_combine = GWEN_FA_NBATCH_COMBINE;
    constexpr int np            = GWEN_NP;
    constexpr int cols_per_warp = GWEN_COLS_PER_WARP;
    constexpr int cols_per_thread = GWEN_COLS_PER_THREAD;

    // ---- Shared memory layout ------------------------------------------------
    // With Q_in_reg=true and nstages=2:
    //   tile_Q  : [ncols * stride_tile_Q]        half2  (loaded once, then reused for combine)
    //   tile_K  : [nbatch_fa * stride_tile_K]    half2  (double-buffered)
    //   tile_V  : [nbatch_fa * stride_tile_V]    half2  (separate from tile_K)
    //   tile_Q is also reused as combine buffer at the end.
    //
    // Total shared memory (bytes):
    //   max(
    //     ncols*stride_tile_Q*sizeof(half2),              <- Q loading phase
    //     nbatch_fa*(stride_tile_K + stride_tile_V)*sizeof(half2)  <- KV phase (2-stage)
    //   ) + nwarps*cols_per_warp*(nbatch_combine+4)*sizeof(half2)  <- combine phase
    //
    // The kernel sets dynamic shared memory size before launch.
    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = tile_Q;   // Q_in_reg: K/V buffers overlap Q buffer
    half2 * tile_V    = tile_K + nbatch_fa * GWEN_STRIDE_TILE_K;

    // ---- Q tiles (loaded once into registers) --------------------------------
    constexpr int n_Q_B = DKQ / (2 * T_B_KQ::J);  // = 256/(2*8) = 16
    T_B_KQ Q_B[n_Q_B];

    constexpr int n_VKQ_C = DV / T_C_VKQ::I;   // = 256/16 = 16
    T_C_VKQ VKQ_C[n_VKQ_C];

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
    #pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX / 2.0f;
    }

    // ---- Load Q into tile_Q (shared) ----------------------------------------
    const half2 scale_h2 = make_half2(scale, scale);
    // Iterate over D with decreasing stride for better coalescing.
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0
                              : DKQ/2 - (DKQ/2) % (2 * stride_k);
        const int k0_stop_q = DKQ/2 - (DKQ/2) % (1 * stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop_q) continue;

        #pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps * stride_jc) {
            const int jc = jc0 + threadIdx.y * stride_jc
                + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps * stride_jc > ncols && jc >= ncols) break;

            // Simple case (ncols2=1): all jc are valid Q columns.
            #pragma unroll
            for (int k0 = k0_start; k0 < k0_stop_q; k0 += stride_k) {
                const int k = k0 + (stride_k == warp_size
                                    ? threadIdx.x : threadIdx.x % stride_k);
                const float2 tmp = Q_f2[jc * stride_Q + k];
                tile_Q[jc * GWEN_STRIDE_TILE_Q + k] =
                    scale_h2 * make_half2(tmp.x, tmp.y);
            }
        }
    }

    __syncthreads();

    // ---- Load Q from shared memory into registers ----------------------------
    const int j0 = (threadIdx.y / np) * cols_per_warp;
    #pragma unroll
    for (int k0 = 0; k0 < DKQ / 2; k0 += T_B_KQ::J) {
        load_ldmatrix(Q_B[k0 / T_B_KQ::J],
                      tile_Q + j0 * GWEN_STRIDE_TILE_Q + k0,
                      GWEN_STRIDE_TILE_Q);
    }

    __syncthreads();

    // ---- Pre-load K tile for first iteration (nstages=2) --------------------
    if constexpr (GWEN_FA_NSTAGES > 1) {
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_sup_first  = nbatch_fa;
        flash_attn_ext_f16_load_tile<GWEN_STRIDE_TILE_K, nbatch_fa,
                                     use_cp_async, oob_check>(
            K_h2 + (int64_t)kb0_start * nbatch_fa * stride_K,
            tile_K, nbatch_K2, stride_K, k_sup_first);
    }

    // ---- Main KV loop --------------------------------------------------------
    int kb0 = kb0_start;

    // All iterations except the last: k_VKQ_sup = nbatch_fa (no OOB check needed).
    for (; kb0 < kb0_stop - 1; ++kb0) {
        const int k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter(
            K_h2, V_h2, stride_K, stride_V,
            tile_Q, tile_K, tile_V,
            Q_B, VKQ_C, KQ_max, KQ_rowsum,
            kb0, k_VKQ_sup, /*last_iter=*/false);
    }
    // Last iteration: k_VKQ_sup = remainder rows.
    {
        const int k_VKQ_sup = ne11 - kb0 * nbatch_fa;
        flash_attn_ext_f16_iter(
            K_h2, V_h2, stride_K, stride_V,
            tile_Q, tile_K, tile_V,
            Q_B, VKQ_C, KQ_max, KQ_rowsum,
            kb0, k_VKQ_sup, /*last_iter=*/true);
    }

    // With 2-stage loading, the last iter may leave outstanding cp.async.
    // The combine phase uses tile_Q as shared memory, so we must ensure
    // all threads have finished the main loop before proceeding.
    if constexpr (GWEN_FA_NSTAGES > 1 && nwarps * cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // ---- Sum up partial KQ rowsums (spread across 8 threads in narrow path) -
    #pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        #pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col],
                                              offset, warp_size);
        }
    }

    // ---- Write VKQ accumulators to shared memory and then to VRAM -----------
    // Simple case (np==1, needs_fixup=false, is_fixup=false):
    //   No partial warp combination needed.
    //   Write VKQ data to tile_Q (reused as combine buffer), then copy to dstk.
    //
    // The combine buffer layout: tile_Q[jc][k] with stride = nbatch_combine+4
    // The last 8 bytes of each row store KQ_max + KQ_rowsum as float2 metadata.

    constexpr int tile_stride = nbatch_combine + 4;

    // cols_per_warp==8 path:
    //   jc_cwd = jc combine write data = warp-local Q column index.
    {
        const int jc_cwd = threadIdx.y * T_B_KQ::I + T_B_KQ::get_i(-1);
        // Write metadata (KQ_max, KQ_rowsum) into the padding area.
        // jc_cwmo = which of the 2 columns this thread owns.
        const int jc_cwmo = (threadIdx.x % (2 * T_C_VKQ::J)) / T_C_VKQ::J;
        const int jc_cwm  = threadIdx.y * (2 * T_C_VKQ::J)
                          + 2 * T_C_VKQ::get_j(-1) + jc_cwmo;
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]);
        if (threadIdx.x < 2 * T_C_VKQ::J) {
            ((float2 *) tile_Q)[jc_cwm * (tile_stride / 2) + nbatch_combine / 2] = KQ_cmr;
        }
    }

    __syncthreads();

    // Write VKQ accumulator to combine buffer.
    #pragma unroll
    for (int k00 = 0; k00 < DV / 2; k00 += nbatch_combine) {
        {
            const int jc_cwd = threadIdx.y * T_B_KQ::I + T_B_KQ::get_i(-1);
            #pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                // Convert VKQ_C (tile<16,4,half2>) to B matrix (tile<8,8,half2>)
                // for column-major storage in shared memory.
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1) / T_B_KQ::J]);
                #pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);
                    tile_Q[jc_cwd * tile_stride + k] = B.x[l];
                }
            }
        }

        __syncthreads();

        // Copy from shared memory to VRAM with normalisation.
        // np==1: no cross-warp combination; directly divide by KQ_rowsum.
        for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
            const int k0_start_c = stride_k == warp_size ? 0
                : nbatch_combine - nbatch_combine % (2 * stride_k);
            const int k0_stop_c  = nbatch_combine - nbatch_combine % (1 * stride_k);
            const int stride_jc  = warp_size / stride_k;

            if (k0_start_c == k0_stop_c) continue;

            #pragma unroll
            for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += nwarps * stride_jc) {
                const int jc_dst = jc0_dst + threadIdx.y * stride_jc
                    + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (jc0_dst + nwarps * stride_jc > ncols && jc_dst >= ncols) break;

                // For np==1, the KV-parallel warp offset is 0.
                const int jc_tile_K = jc_dst;
                const float * meta_j = (const float *) tile_Q
                    + jc_tile_K * tile_stride + nbatch_combine;
                const float KQ_rowsum_j = meta_j[1];  // meta: [KQ_cms, KQ_rowsum]

                #pragma unroll
                for (int k0 = k0_start_c; k0 < k0_stop_c; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size
                                        ? threadIdx.x : threadIdx.x % stride_k);
                    float2 dstk_val = __half22float2(
                        tile_Q[jc_tile_K * tile_stride + k]);
                    // Normalise by rowsum.
                    dstk_val.x /= KQ_rowsum_j;
                    dstk_val.y /= KQ_rowsum_j;
                    dstk[jc_dst * (DV / 2) + k00 + k] = dstk_val;
                }
            }
        }

        __syncthreads();
    }
}

// ---- Kernel entry point ------------------------------------------------------
//
// gwen_flash_attn_mma_f16:
//   Full-attention kernel, D=256, FP16, Turing mma.sync.
//
//   Grid:  (ne02 * ne03, ne01_z, 1)
//     blockIdx.x = Q-head index (0..ne02-1)
//     blockIdx.y = sequence index (0..ne03-1)
//     blockIdx.z = token index (0..ne01_z-1)  [always 0 for decode]
//
//   Block: (32, 4, 1)  i.e. 128 threads, 4 warps
//
//   Shared memory: computed by gwen_flash_attn_mma_smem_bytes() below.
//
// Tensor layout (all row-major in the sequence/token dimension):
//   Q:   [ne03, ne02, ne01_z, DKQ]  float   (nb01 bytes per row)
//   K:   [ne03, ne12, ne11,   DKQ]  half    (nb11 bytes per row)
//   V:   [ne03, ne12, ne11,   DV]   half    (nb21 bytes per row)
//   dst: [ne03, ne02, ne01_z, DV]   float
//
//   ne02 = Q heads, ne12 = KV heads, ne01_z = Q tokens (1 for decode),
//   ne11 = KV sequence length, ne03 = batch size.
//
// GQA (ne02 > ne12): each KV head serves gqa_ratio = ne02/ne12 Q heads.
// This kernel dispatches one block per Q head; the caller must loop or batch.
//
// For pure decode (ne01_z=1), ncols=8 means the kernel processes 8 logical
// Q-column slots but only 1 is populated (the rest are zeroed in tile_Q loading).

__launch_bounds__(GWEN_FA_NWARPS * WARP_SIZE, 2)
static __global__ void gwen_flash_attn_mma_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        float      * __restrict__ dst,
        const float scale,
        const int32_t ne00,           // DKQ (unused, hardcoded 256)
        const int32_t ne01_z,         // Q tokens per head
        const int32_t ne02,           // Q heads
        const int32_t ne03,           // batch size
        const int32_t nb01,           // Q row stride in bytes
        const int32_t nb02,           // Q head stride in bytes
        const int32_t nb03,           // Q batch stride in bytes
        const int32_t ne11,           // KV sequence length
        const int32_t ne12,           // KV heads
        const int32_t nb11,           // K row stride in bytes
        const int64_t nb12,           // K head stride in bytes
        const int64_t nb13,           // K batch stride in bytes
        const int32_t nb21,           // V row stride in bytes
        const int64_t nb22,           // V head stride in bytes
        const int64_t nb23) {         // V batch stride in bytes

    // Grid decomposition: one block per (Q-head, token, batch).
    // For decode (ne01_z=1): blockIdx.z=0, blockIdx.y=Q-head, blockIdx.x=batch.
    // Adjust as needed for your grid launch.
    const int z_Q      = blockIdx.x;   // Q head index
    const int sequence = blockIdx.y;   // batch index

    // GQA: map Q head to KV head.
    const int gqa_ratio = ne02 / ne12;
    const int z_KV      = z_Q / gqa_ratio;

    // Token index (for decode: always 0).
    const int jt = blockIdx.z;   // Q token block index

    const float2 * Q_f2 = (const float2 *) (Q + (int64_t)nb03 * sequence
                                               + (int64_t)nb02 * z_Q);
    const half2  * K_h2 = (const half2  *) (K + nb13 * sequence + nb12 * z_KV);
    const half2  * V_h2 = (const half2  *) (V + nb23 * sequence + nb22 * z_KV);

    float2 * dstk = ((float2 *) dst)
        + ((int64_t)sequence * ne01_z * ne02 + (int64_t)z_Q * ne01_z + jt)
          * (GWEN_FA_DV / 2);

    const int stride_Q = nb01 / sizeof(float2);    // Q row stride in float2
    const int stride_K = nb11 / sizeof(half2);     // K row stride in half2
    const int stride_V = nb21 / sizeof(half2);     // V row stride in half2

    // KV iteration range: full sequence.
    constexpr int nbatch_fa = GWEN_FA_NBATCH_FA;
    const int iter_k   = (ne11 + nbatch_fa - 1) / nbatch_fa;
    const int kb0_start = 0;
    const int kb0_stop  = iter_k;

    flash_attn_ext_f16_process_tile(
        Q_f2, K_h2, V_h2, dstk,
        scale,
        stride_Q, stride_K, stride_V,
        ne11, kb0_start, kb0_stop);
}

// ---- Shared memory size helper -----------------------------------------------
// Call this from the host to get the correct dynamic shared memory size.
inline size_t gwen_flash_attn_mma_smem_bytes() {
    constexpr int DKQ           = GWEN_FA_DKQ;
    constexpr int DV            = GWEN_FA_DV;
    constexpr int ncols         = GWEN_FA_NCOLS;
    constexpr int nwarps        = GWEN_FA_NWARPS;
    constexpr int nbatch_fa     = GWEN_FA_NBATCH_FA;
    constexpr int nbatch_K2     = GWEN_FA_NBATCH_K2;
    constexpr int nbatch_V2     = GWEN_FA_NBATCH_V2;
    constexpr int nbatch_combine = GWEN_FA_NBATCH_COMBINE;
    constexpr int cols_per_warp = GWEN_COLS_PER_WARP;
    constexpr int nstages       = GWEN_FA_NSTAGES;

    // Q region: ncols * stride_tile_Q half2 values
    const size_t nbytes_Q = ncols * GWEN_STRIDE_TILE_Q * sizeof(half2);

    // KV region (2-stage): both K and V buffers simultaneously
    const size_t nbytes_KV_2stage = (size_t)nbatch_fa
        * (GWEN_STRIDE_TILE_K + GWEN_STRIDE_TILE_V) * sizeof(half2);
    const size_t nbytes_KV_1stage = (size_t)nbatch_fa
        * (nbatch_K2 > nbatch_V2 ? nbatch_K2 : nbatch_V2) * sizeof(half2);
    const size_t nbytes_KV = nstages <= 1 ? nbytes_KV_1stage : nbytes_KV_2stage;

    // Combine region: nwarps*cols_per_warp rows, each nbatch_combine+4 half2 wide
    const size_t nbytes_combine = (size_t)nwarps * cols_per_warp
        * (nbatch_combine + 4) * sizeof(half2);

    // Q_in_reg=true: Q region and KV region overlap (Q loaded first, then reused)
    const size_t nbytes_main = nbytes_Q > nbytes_KV ? nbytes_Q : nbytes_KV;

    return nbytes_main > nbytes_combine ? nbytes_main : nbytes_combine;
}
