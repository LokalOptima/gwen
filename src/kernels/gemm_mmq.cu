// Fused quantized GEMM — ported from llama.cpp's mmq kernel.
// Uses vendored llama.cpp headers (include/gwen/llama/).
// Supports Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS weight types.
// MIT License applies to the llama.cpp kernel code.

extern "C" [[noreturn]] void ggml_abort(const char*, int, const char*, ...);

#include "gwen/llama_common.cuh"
#include "gwen/llama_mmq.cuh"
#include "gwen/llama_quantize.cuh"  // for CUDA_QUANTIZE_BLOCK_SIZE_MMQ

#include "gwen/kernels.h"

namespace gwen {

// ============================================================
// Direct FP16 → Q8_1_mmq quantizer (no F32 intermediate)
// ============================================================
// Adapted from llama.cpp's quantize_mmq_q8_1 but reads FP16 input directly.
// Each thread loads 4 consecutive half values, quantizes to int8.

template<mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1_fp16(
    const half* __restrict__ x,
    void* __restrict__ vy,
    const int64_t ne00,     // K (inner dim, may be < ne0 due to padding)
    const int64_t s01,      // stride between columns (in half elements)
    const int64_t ne0,      // K padded
    const int ne1)          // N (number of columns)
{
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;

    const int64_t i1 = blockIdx.x;  // column index

    // Load 4 half values, convert to float
    float4 xi;
    if (i0 + 3 < ne00) {
        const half* src = x + i1 * s01 + i0;
        xi.x = __half2float(src[0]);
        xi.y = __half2float(src[1]);
        xi.z = __half2float(src[2]);
        xi.w = __half2float(src[3]);
    } else {
        xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const half* src = x + i1 * s01;
        if (i0 + 0 < ne00) xi.x = __half2float(src[i0 + 0]);
        if (i0 + 1 < ne00) xi.y = __half2float(src[i0 + 1]);
        if (i0 + 2 < ne00) xi.z = __half2float(src[i0 + 2]);
    }

    block_q8_1_mmq* y = (block_q8_1_mmq*)vy;

    const int64_t ib0 = blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x / QK8_1);
    const int64_t ib  = ib0 + (i0 / (4 * QK8_1)) * ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4 * QK8_1);

    // Find max absolute value (warp reduction across vals_per_scale/4 threads)
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    #pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
        #pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);

    char4* yqs4 = (char4*)y[ib].qs;
    yqs4[iqs / 4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) return;
        y[ib].d2s6[2 + iqs / 16] = sum;
        if (iqs % 64 != 0) return;
        y[ib].d2s6[iqs / 64] = 1.0f / d_inv;
        return;
    }

    if (iqs % 32 != 0) return;
    const float d = 1.0f / d_inv;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs / 32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs / 32] = d;
    }
}

// Launch FP16 → Q8_1_mmq quantizer
static void quantize_mmq_q8_1_fp16_cuda(
    const half* x, void* vy, ggml_type type_src0,
    int64_t ne00, int64_t s01,
    int64_t ne0_padded, int ne1,
    cudaStream_t stream)
{
    const int64_t block_num_x = (ne0_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_x, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);

    // Select ds_layout based on weight type (same logic as llama.cpp)
    const mmq_q8_1_ds_layout ds_layout = mmq_get_q8_1_ds_layout(type_src0);

    switch (ds_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1_fp16<MMQ_Q8_1_DS_LAYOUT_D4><<<num_blocks, block_size, 0, stream>>>(
                x, vy, ne00, s01, ne0_padded, ne1);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1_fp16<MMQ_Q8_1_DS_LAYOUT_DS4><<<num_blocks, block_size, 0, stream>>>(
                x, vy, ne00, s01, ne0_padded, ne1);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1_fp16<MMQ_Q8_1_DS_LAYOUT_D2S6><<<num_blocks, block_size, 0, stream>>>(
                x, vy, ne00, s01, ne0_padded, ne1);
            break;
    }
}

// ============================================================
// FP16-output mmq kernel (eliminates F32 output buffer)
// ============================================================
// Writes half* directly instead of float*. The fixup buffer
// remains F32 for partial-sum accumulation; the fixup kernel
// reads existing FP16 from dst, adds the F32 correction, and
// writes FP16 back.

// Write-back identical to mmq_write_back_mma but stores FP16.
template<ggml_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_fp16(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, half * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int nwarps = mmq_get_nwarps_device();

    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;

    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

    static_assert(nwarps * tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                if (j > j_max) { continue; }
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (need_check && i > i_max) { continue; }
                dst[ids_dst[j] * stride + i] = __float2half(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
            }
        }
    }
}

// Process tile: same computation as mul_mat_q_process_tile,
// but non-fixup path writes FP16; fixup path writes F32 to tmp_fixup.
template <ggml_type type, int mmq_x, bool need_check, bool fixup>
static __device__ __forceinline__ void gwen_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, half * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int              nwarps     = mmq_get_nwarps_device();
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              mmq_y      = get_mmq_y_device();
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, need_check, type>::load_tiles;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y_sh = data_mul_mat_q + mmq_x;
    int * tile_x    = tile_y_sh + GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr vec_dot_mmq_t vec_dot = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;

    constexpr int ne_block = 4 * QK8_1;

    constexpr int ITER_K          = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y_sh[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y_sh, sum, 0);
        __syncthreads();
        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y_sh[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y_sh, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    if (fixup) {
        // Fixup buffer is always F32 (partial sums accumulated across blocks)
        mmq_write_back_mma<type, mmq_x, mmq_y, need_check>(
            sum, ids_dst, tmp_fixup + blockIdx.x * (mmq_x * mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        // Output directly as FP16
        mmq_write_back_fp16<type, mmq_x, mmq_y, need_check>(
            sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

// Simplified stream-K mmq kernel with FP16 output.
// No MoE (ids_dst/expert_bounds), single channel/sample.
template <ggml_type type, int mmq_x, bool need_check>
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __launch_bounds__(ggml_cuda_get_physical_warp_size() * mmq_get_nwarps_device(), 1)
#endif
static __global__ void gwen_mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y,
        half * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int ncols_max) {

    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();
    constexpr int ITER_K    = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;

    const int64_t blocks_per_ne00 = ncols_x / qk;
    const int ntx = (ncols_max + mmq_x - 1) / mmq_x;
    const int nty = (nrows_x   + mmq_y - 1) / mmq_y;

    // Initialize ids_dst with identity mapping (no MoE)
    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > mmq_x && j >= mmq_x) { break; }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    // Stream-K work partitioning
    const int64_t total_tiles_k = (int64_t)ntx * nty * blocks_per_ne00;
    int64_t kbc      = (int64_t) blockIdx.x      * total_tiles_k / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1) * total_tiles_k / gridDim.x;
    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);

    // Complete tiles (fixup=false)
    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        int64_t tmp = kbc;
        const int it = tmp / (ntx * blocks_per_ne00);
        tmp -= (int64_t)it * ntx * blocks_per_ne00;
        const int jt = tmp / blocks_per_ne00;

        constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
        const int offset_y   = jt * mmq_x * sz;
        const int offset_dst = jt * mmq_x * stride_col_dst + it * mmq_y;
        const int offset_x   = it * mmq_y * stride_row_x;

        constexpr bool fixup = false;
        gwen_process_tile<type, mmq_x, need_check, fixup>(
            x, offset_x, y + offset_y, ids_dst_shared,
            dst + offset_dst, tmp_fixup,
            stride_row_x, ncols_y, stride_col_dst,
            nrows_x - it * mmq_y - 1, ncols_dst - jt * mmq_x - 1,
            kb0_start, kb0_stop);

        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;
        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) { return; }

    // Last (partial) tile — writes to fixup buffer
    {
        int64_t tmp = kbc;
        const int it = tmp / (ntx * blocks_per_ne00);
        tmp -= (int64_t)it * ntx * blocks_per_ne00;
        const int jt = tmp / blocks_per_ne00;

        constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
        const int offset_y   = jt * mmq_x * sz;
        const int offset_dst = jt * mmq_x * stride_col_dst + it * mmq_y;
        const int offset_x   = it * mmq_y * stride_row_x;

        constexpr bool fixup = true;
        gwen_process_tile<type, mmq_x, need_check, fixup>(
            x, offset_x, y + offset_y, ids_dst_shared,
            dst + offset_dst, tmp_fixup,
            stride_row_x, ncols_y, stride_col_dst,
            nrows_x - it * mmq_y - 1, ncols_dst - jt * mmq_x - 1,
            kb0_start, kb0_stop);
    }
}

// Stream-K fixup kernel — reads F32 fixup buffer, writes FP16 to output.
// Each block checks if it wrote a partial tile that needs correction.
template <ggml_type type, int mmq_x, bool need_check>
static __global__ void gwen_stream_k_fixup(
        half * __restrict__ dst,
        const float * __restrict__ tmp_last_tile,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_col_dst, const int ncols_max) {

    constexpr int mmq_y     = get_mmq_y_device();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int ITER_K    = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t blocks_per_ne00 = ncols_x / qk;
    const int ntx = (ncols_max + mmq_x - 1) / mmq_x;
    const int nty = (nrows_x   + mmq_y - 1) / mmq_y;

    const int64_t total_tiles_k = (int64_t)ntx * nty * blocks_per_ne00;
    const int bidx0 = blockIdx.x;

    int64_t kbc0      = (int64_t) bidx0      * total_tiles_k / gridDim.x;
    int64_t kbc0_stop = (int64_t)(bidx0 + 1) * total_tiles_k / gridDim.x;
    kbc0      -= (kbc0      % blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= (kbc0_stop % blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % blocks_per_ne00 == 0;
    const bool did_not_write_last      = kbc0 / blocks_per_ne00 == kbc0_stop / blocks_per_ne00
                                      && kbc0_stop % blocks_per_ne00 != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) { return; }

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};
    bool any_fixup = false;

    int64_t bidx = bidx0 - 1;
    int64_t kbc_stop = kbc0;
    while (true) {
        int64_t kbc = bidx * total_tiles_k / gridDim.x;
        kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) {
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                sum[(j0 / nwarps) * (mmq_y / warp_size) + i0 / warp_size]
                    += tmp_last_tile[bidx * (mmq_x * mmq_y) + j * mmq_y + i];
            }
        }

        if (kbc % blocks_per_ne00 == 0 || kbc / blocks_per_ne00 < kbc0 / blocks_per_ne00) { break; }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) { return; }

    // Compute output tile coordinates
    int64_t tmp = kbc0;
    const int it = tmp / (ntx * blocks_per_ne00);
    tmp -= (int64_t)it * ntx * blocks_per_ne00;
    const int jt = tmp / blocks_per_ne00;

    const int offset_dst = jt * mmq_x * stride_col_dst + it * mmq_y;
    dst += offset_dst;

    const int i_max = nrows_x   - it * mmq_y - 1;
    const int j_max = ncols_dst - jt * mmq_x - 1;

    // Read existing FP16 from dst, add F32 fixup, write FP16 back
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) { return; }
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (need_check && i > i_max) { continue; }
            const float existing = __half2float(dst[j * stride_col_dst + i]);
            dst[j * stride_col_dst + i] = __float2half(
                existing + sum[(j0 / nwarps) * (mmq_y / warp_size) + i0 / warp_size]);
        }
    }
}

// SM_120 device properties
static constexpr int GWEN_CC = 1200;
static constexpr int GWEN_NSM = 70;
static constexpr int GWEN_WARP_SIZE = 32;
static constexpr int MMQ_X = 128;

template<ggml_type type>
static void ensure_smem_limit() {
    static bool done = false;
    if (done) return;
    const int nwarps = mmq_get_nwarps_host(GWEN_CC, GWEN_WARP_SIZE);
    const int mmq_y = get_mmq_y_host(GWEN_CC);
    const int nbytes = mmq_get_nbytes_shared<type>(MMQ_X, mmq_y, GWEN_CC, GWEN_WARP_SIZE, nwarps);
    cudaFuncSetAttribute((const void*)(gwen_mul_mat_q<type, MMQ_X, false>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes);
    cudaFuncSetAttribute((const void*)(gwen_mul_mat_q<type, MMQ_X, true>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes);
    done = true;
}

// Core mmq launch — FP16 in, FP16 out, no F32 intermediate anywhere
template<ggml_type type>
static void launch_mmq(
    const void* W, const half* X, half* Y, void* scratch,
    int M, int K, int N, cudaStream_t stream)
{
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    int K_padded = (K + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    int n_q8_blocks_per_col = K_padded / 128;
    size_t q8_size = (size_t)n_q8_blocks_per_col * N * sizeof(block_q8_1_mmq);

    // Scratch layout:
    //   [0 .. q8_size)     : Q8_1_mmq quantized activations
    //   [q8_size .. +fixup): stream-K fixup buffer (F32)
    void* X_q8_mmq = scratch;

    // FP16 → Q8_1_mmq directly
    quantize_mmq_q8_1_fp16_cuda(X, X_q8_mmq, type, K, K, K_padded, N, stream);

    // Launch mmq kernel — output goes directly to Y (FP16)
    ensure_smem_limit<type>();
    const int nwarps = mmq_get_nwarps_host(GWEN_CC, GWEN_WARP_SIZE);
    const int mmq_y = get_mmq_y_host(GWEN_CC);
    const int nbytes_shared = mmq_get_nbytes_shared<type>(MMQ_X, mmq_y, GWEN_CC, GWEN_WARP_SIZE, nwarps);
    int blocks_per_row = K / qk;

    const dim3 block_dims(GWEN_WARP_SIZE, nwarps, 1);
    const int nty = (M + mmq_y - 1) / mmq_y;
    const int ntx = (N + MMQ_X - 1) / MMQ_X;
    const dim3 grid_sk(GWEN_NSM, 1, 1);
    const bool fixup_needed = ntx * nty % GWEN_NSM != 0;
    float* tmp_fixup = fixup_needed
        ? reinterpret_cast<float*>(reinterpret_cast<char*>(scratch) + q8_size)
        : nullptr;

    if (M % mmq_y == 0) {
        gwen_mul_mat_q<type, MMQ_X, false><<<grid_sk, block_dims, nbytes_shared, stream>>>(
            reinterpret_cast<const char*>(W), reinterpret_cast<const int*>(X_q8_mmq),
            Y, tmp_fixup, K, M, N, blocks_per_row, N, M, N);
    } else {
        gwen_mul_mat_q<type, MMQ_X, true><<<grid_sk, block_dims, nbytes_shared, stream>>>(
            reinterpret_cast<const char*>(W), reinterpret_cast<const int*>(X_q8_mmq),
            Y, tmp_fixup, K, M, N, blocks_per_row, N, M, N);
    }

    if (fixup_needed) {
        gwen_stream_k_fixup<type, MMQ_X, true><<<grid_sk, block_dims, 0, stream>>>(
            Y, tmp_fixup, K, M, N, M, N);
    }
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemm_mmq(
    const void* W, GGMLType type, const half* X, half* Y, void* scratch,
    int M, int K, int N, cudaStream_t stream)
{
    switch (type) {
        case GGMLType::Q4_K: launch_mmq<GGML_TYPE_Q4_K>(W, X, Y, scratch, M, K, N, stream); break;
        case GGMLType::Q5_K: launch_mmq<GGML_TYPE_Q5_K>(W, X, Y, scratch, M, K, N, stream); break;
        case GGMLType::Q6_K: launch_mmq<GGML_TYPE_Q6_K>(W, X, Y, scratch, M, K, N, stream); break;
        case GGMLType::Q8_0:  launch_mmq<GGML_TYPE_Q8_0>(W, X, Y, scratch, M, K, N, stream); break;
        case GGMLType::IQ4_XS: launch_mmq<GGML_TYPE_IQ4_XS>(W, X, Y, scratch, M, K, N, stream); break;
        default:
            GWEN_CHECK(false, "Unsupported type for mmq GEMM");
    }
}

size_t gwen_gemm_mmq_scratch_size(int max_K, int max_N) {
    // Q8_1_mmq quantized activations: ceil(K_padded/128) * N * sizeof(block_q8_1_mmq)
    int K_padded = (max_K + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    int n_q8_blocks_per_col = K_padded / 128;
    size_t q8_size = (size_t)n_q8_blocks_per_col * max_N * sizeof(block_q8_1_mmq);

    // Stream-K fixup buffer: NSM * MMQ_X * mmq_y * sizeof(float)
    const int mmq_y = get_mmq_y_host(GWEN_CC);
    size_t fixup_size = (size_t)GWEN_NSM * MMQ_X * mmq_y * sizeof(float);

    return q8_size + fixup_size;
}

} // namespace gwen
