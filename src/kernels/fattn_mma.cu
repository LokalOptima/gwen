// MMA-based flash attention for GWEN.
// Uses vendored llama.cpp headers (include/gwen/llama/).
// MIT License applies to the llama.cpp kernel code.

// Stub for ggml_abort (referenced by common.cuh error paths, never actually called)
extern "C" [[noreturn]] void ggml_abort(const char*, int, const char*, ...) { __builtin_trap(); }

#include "common.cuh"
#include "fattn-mma-f16.cuh"

#include "gwen/kernels.h"

namespace gwen {

static __global__ void kernel_h2f(const half* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

static __global__ void kernel_f2h(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

void gwen_flash_attn_mma(
    const half* Q_fp16,
    const half* K,
    const half* V,
    half* output_fp16,
    float* temp_f32,
    int N, int n_head, int n_kv_heads, int head_dim,
    float scale, cudaStream_t stream)
{
    constexpr int DKQ = 256, DV = 256, ncols1 = 8, ncols2 = 1;
    constexpr int ncols = ncols1 * ncols2;

    int total_q = N * n_head * head_dim;
    float* Q_f32  = temp_f32;
    float* dst_f32 = temp_f32 + total_q;

    // Convert Q FP16 → F32
    kernel_h2f<<<(total_q + 255) / 256, 256, 0, stream>>>(Q_fp16, Q_f32, total_q);

    // Query llama.cpp's config for SM_120 (cc=1200)
    const int cc = 1200;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, cc);
    const int nwarps = nthreads / WARP_SIZE;
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, ncols, cc);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc);
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));

    // Shared memory (matching llama.cpp's calculation exactly)
    const size_t nbytes_KV_1s = nbatch_fa * std::max(nbatch_K2+4, nbatch_V2+4) * sizeof(half2);
    const size_t nbytes_KV_2s = nbatch_fa * (nbatch_K2+4 + nbatch_V2+4) * sizeof(half2);
    const size_t nbytes_Q = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_combine_s = nwarps * cols_per_warp * (nbatch_combine+4) * sizeof(half2);
    const size_t nbytes_KV = nstages <= 1 ? nbytes_KV_1s : nbytes_KV_2s;
    const size_t smem = std::max(nbytes_combine_s, Q_in_reg ?
        std::max(nbytes_Q, nbytes_KV + nbytes_mask) :
        nbytes_Q + nbytes_KV + nbytes_mask);

    auto kernel_fn = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false>;

    static bool smem_set = false;
    if (!smem_set) {
        GWEN_CHECK_CUDA(cudaFuncSetAttribute(
            (const void*)kernel_fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        smem_set = true;
    }

    // ggml-style byte strides
    int32_t nb01 = n_head * head_dim * (int32_t)sizeof(float);
    int32_t nb02 = head_dim * (int32_t)sizeof(float);
    int32_t nb03 = N * nb01;

    int32_t nb11 = n_kv_heads * head_dim * (int32_t)sizeof(half);
    int32_t nb12 = head_dim * (int32_t)sizeof(half);
    int64_t nb13 = (int64_t)N * nb11;
    int32_t nb21 = nb11;
    int32_t nb22 = nb12;
    int64_t nb23 = nb13;

    uint3 ne01 = make_uint3(n_head, 0, N);
    int gqa_ratio = n_head / n_kv_heads;

    // Grid: stream-K style (one big 1D grid)
    int ntiles_x = (N + ncols1 - 1) / ncols1;
    int ntiles_total = ntiles_x * gqa_ratio * n_kv_heads;
    dim3 grid(ntiles_total, 1, 1);
    dim3 block(WARP_SIZE, nwarps);

    kernel_fn<<<grid, block, smem, stream>>>(
        (const char*)Q_f32,
        (const char*)K,
        (const char*)V,
        nullptr, nullptr, nullptr,  // mask, sinks, KV_max
        dst_f32,
        nullptr,  // dst_meta
        scale,
        0.0f, 0.0f, 0.0f, 0, 0.0f,
        head_dim, ne01, n_head, 1,
        nb01, nb02, nb03,
        head_dim, N, n_kv_heads, 1,
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        0, 0, 0,
        0, 0, (int64_t)0);
    GWEN_CHECK_CUDA(cudaGetLastError());

    // Convert output F32 → FP16
    kernel_f2h<<<(total_q + 255) / 256, 256, 0, stream>>>(dst_f32, output_fp16, total_q);
}

} // namespace gwen
