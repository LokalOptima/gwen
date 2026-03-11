#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

namespace gwen {

// RMSNorm for dim=1024 — one warp per vector
// output = x * rsqrt(mean(x^2) + eps) * weight
// Uses warp shuffle for reduction (no shared memory needed)
__global__ void __launch_bounds__(32)
kernel_rmsnorm_f32w(const half* __restrict__ x, const float* __restrict__ weight,
                    half* __restrict__ y, int dim, float eps) {
    int lane = threadIdx.x;  // 0..31

    // Accumulate sum of squares across the dimension
    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    float rms_inv = rsqrtf(sum_sq / dim + eps);

    // Apply normalization and weight
    for (int i = lane; i < dim; i += 32) {
        float val = __half2float(x[i]);
        float w = weight[i];
        y[i] = __float2half(val * rms_inv * w);
    }
}

// RMSNorm with FP16 weights
__global__ void __launch_bounds__(32)
kernel_rmsnorm(const half* __restrict__ x, const half* __restrict__ weight,
               half* __restrict__ y, int dim, float eps) {
    int lane = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    float rms_inv = rsqrtf(sum_sq / dim + eps);

    for (int i = lane; i < dim; i += 32) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        y[i] = __float2half(val * rms_inv * w);
    }
}

// Batched RMSNorm: one thread block per vector, shared weight across all vectors
// x: [n_vecs, dim], y: [n_vecs, dim] (may alias x for in-place)
__global__ void __launch_bounds__(32)
kernel_rmsnorm_batched_f32w(const half* __restrict__ x, const float* __restrict__ weight,
                            half* __restrict__ y, int n_vecs, int dim, float eps) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n_vecs) return;
    int lane = threadIdx.x;

    const half* xv = x + vec_idx * dim;
    half* yv = y + vec_idx * dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float val = __half2float(xv[i]);
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    float rms_inv = rsqrtf(sum_sq / dim + eps);

    for (int i = lane; i < dim; i += 32) {
        float val = __half2float(xv[i]);
        float w = weight[i];
        yv[i] = __float2half(val * rms_inv * w);
    }
}

// ============================================================
// Fused RMSNorm + Q8_1 Quantize (+ optional FP16 copy)
// ============================================================
// 256 threads (8 warps). dim=1024 → 4 elements per thread for RMS, 4 Q8_1 blocks per warp.
// Phase 1: All 256 threads compute sum-of-squares, cross-warp reduce via shared memory
// Phase 2: Each warp handles Q8_1 blocks: apply norm+weight, find amax, quantize
// Optionally writes FP16 output (needed by kernel_compute_gate_beta for DeltaNet pre-norm)
__global__ void __launch_bounds__(256)
kernel_rmsnorm_quantize_q8_1(const half* __restrict__ x,
                              const float* __restrict__ weight,
                              block_q8_1* __restrict__ y_q8,
                              half* __restrict__ y_fp16,  // may be nullptr
                              int dim, float eps) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Phase 1: sum of squares (each thread handles dim/256 elements)
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    // Warp 0 reduces
    if (warp_id == 0) {
        sum_sq = (lane < 8) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xFF, sum_sq, offset);
    }

    // Broadcast rms_inv to all threads
    __shared__ float s_rms_inv;
    if (tid == 0) s_rms_inv = rsqrtf(sum_sq / dim + eps);
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Phase 2: Each warp processes Q8_1 blocks
    // dim=1024 → 32 blocks, 8 warps → 4 blocks per warp
    int n_blocks = dim / 32;
    int blocks_per_warp = (n_blocks + 7) / 8;

    for (int bi = 0; bi < blocks_per_warp; bi++) {
        int blk_idx = warp_id * blocks_per_warp + bi;
        if (blk_idx >= n_blocks) break;

        int elem_idx = blk_idx * 32 + lane;
        float xv = __half2float(x[elem_idx]);
        float val = xv * rms_inv * weight[elem_idx];

        // Write FP16 if needed
        if (y_fp16) y_fp16[elem_idx] = __float2half(val);

        // Quantize: find amax
        float amax = fabsf(val);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        int8_t q = (int8_t)roundf(val * id);

        float sum = (float)q;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

        y_q8[blk_idx].qs[lane] = q;
        if (lane == 0)
            y_q8[blk_idx].ds = __halves2half2(__float2half(d), __float2half(sum));
    }
}

void gwen_rmsnorm_quantize_q8_1(const half* x, const float* weight, void* y_q8, half* y_fp16,
                                  int dim, float eps, cudaStream_t stream) {
    kernel_rmsnorm_quantize_q8_1<<<1, 256, 0, stream>>>(
        x, weight, static_cast<block_q8_1*>(y_q8), y_fp16, dim, eps);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_rmsnorm(const half* x, const half* weight, half* y,
                  int dim, float eps, cudaStream_t stream) {
    kernel_rmsnorm<<<1, 32, 0, stream>>>(x, weight, y, dim, eps);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_rmsnorm_f32w(const half* x, const float* weight, half* y,
                       int dim, float eps, cudaStream_t stream) {
    kernel_rmsnorm_f32w<<<1, 32, 0, stream>>>(x, weight, y, dim, eps);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_rmsnorm_batched_f32w(const half* x, const float* weight, half* y,
                               int n_vecs, int dim, float eps, cudaStream_t stream) {
    kernel_rmsnorm_batched_f32w<<<n_vecs, 32, 0, stream>>>(x, weight, y, n_vecs, dim, eps);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
