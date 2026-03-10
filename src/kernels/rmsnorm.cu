#include "gwen/kernels.h"

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

} // namespace gwen
