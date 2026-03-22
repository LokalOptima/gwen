// Fused quantized GEMM — ported from llama.cpp's mmq kernel.
// Compiled against llama.cpp headers (same approach as fattn_mma.cu).
// Supports Q4_K, Q5_K, Q6_K, Q8_0 weight types.
// MIT License applies to the llama.cpp code.

extern "C" [[noreturn]] void ggml_abort(const char*, int, const char*, ...);

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"  // for CUDA_QUANTIZE_BLOCK_SIZE_MMQ

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

// F32 → FP16 output conversion
static __global__ void kernel_f2h(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
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
    cudaFuncSetAttribute((const void*)(mul_mat_q<type, MMQ_X, false>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes);
    cudaFuncSetAttribute((const void*)(mul_mat_q<type, MMQ_X, true>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes);
    done = true;
}

// Core mmq launch — FP16 in, FP16 out, no F32 intermediate for activations
template<ggml_type type>
static void launch_mmq(
    const void* W, const half* X, half* Y, void* scratch,
    int M, int K, int N, cudaStream_t stream)
{
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    int K_padded = (K + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    int n_q8_blocks_per_col = K_padded / 128;
    size_t q8_size = (size_t)n_q8_blocks_per_col * N * sizeof(block_q8_1_mmq);

    // Scratch layout (no F32 activation buffer!):
    //   [0 .. q8_size)           : Q8_1_mmq quantized activations
    //   [q8_size .. +M*N*4)     : F32 GEMM output
    //   [+M*N*4 .. +fixup)      : stream-K fixup buffer
    void*  X_q8_mmq = scratch;
    float* dst_f32  = reinterpret_cast<float*>(reinterpret_cast<char*>(scratch) + q8_size);

    // FP16 → Q8_1_mmq directly (no F32 intermediate)
    quantize_mmq_q8_1_fp16_cuda(X, X_q8_mmq, type, K, K, K_padded, N, stream);

    // Launch mmq kernel
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
    float* tmp_fixup = fixup_needed ? (dst_f32 + M * N) : nullptr;

    if (M % mmq_y == 0) {
        mul_mat_q<type, MMQ_X, false><<<grid_sk, block_dims, nbytes_shared, stream>>>(
            reinterpret_cast<const char*>(W), reinterpret_cast<const int*>(X_q8_mmq),
            nullptr, nullptr, dst_f32, tmp_fixup,
            K, M, N, blocks_per_row, N, M,
            1, 1, 0, 0, 0, 1, 1, 0, 0, 0, N);
    } else {
        mul_mat_q<type, MMQ_X, true><<<grid_sk, block_dims, nbytes_shared, stream>>>(
            reinterpret_cast<const char*>(W), reinterpret_cast<const int*>(X_q8_mmq),
            nullptr, nullptr, dst_f32, tmp_fixup,
            K, M, N, blocks_per_row, N, M,
            1, 1, 0, 0, 0, 1, 1, 0, 0, 0, N);
    }

    if (fixup_needed) {
        mul_mat_q_stream_k_fixup<type, MMQ_X, true><<<grid_sk, block_dims, 0, stream>>>(
            nullptr, nullptr, dst_f32, tmp_fixup,
            K, M, N, (size_t)M, 1, (size_t)0, 1, (size_t)0, N);
    }
    GWEN_CHECK_CUDA(cudaGetLastError());

    // F32 → FP16 (mmq kernel outputs F32)
    int out_total = M * N;
    kernel_f2h<<<(out_total + 255) / 256, 256, 0, stream>>>(dst_f32, Y, out_total);
}

void gwen_gemm_q4k(
    const void* W, const half* X, half* Y, void* scratch,
    int M, int K, int N, cudaStream_t stream)
{
    launch_mmq<GGML_TYPE_Q4_K>(W, X, Y, scratch, M, K, N, stream);
}

void gwen_gemm_mmq(
    const void* W, GGMLType type, const half* X, half* Y, void* scratch,
    int M, int K, int N, cudaStream_t stream)
{
    switch (type) {
        case GGMLType::Q4_K: launch_mmq<GGML_TYPE_Q4_K>(W, X, Y, scratch, M, K, N, stream); break;
        case GGMLType::Q5_K: launch_mmq<GGML_TYPE_Q5_K>(W, X, Y, scratch, M, K, N, stream); break;
        case GGMLType::Q6_K: launch_mmq<GGML_TYPE_Q6_K>(W, X, Y, scratch, M, K, N, stream); break;
        case GGMLType::Q8_0: launch_mmq<GGML_TYPE_Q8_0>(W, X, Y, scratch, M, K, N, stream); break;
        default:
            GWEN_CHECK(false, "Unsupported type for mmq GEMM");
    }
}

} // namespace gwen
