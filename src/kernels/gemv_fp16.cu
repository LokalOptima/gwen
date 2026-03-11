#include "gwen/kernels.h"

namespace gwen {

// FP16 GEMV: y[row] = dot(W[row, :], x[:])
// Each block computes one output row.
// W: [out_features, in_features] in FP16 (row-major)
// x: [in_features] in FP16
// y: [out_features] in FP16
template<bool ADD_RESIDUAL>
__global__ void __launch_bounds__(256)
kernel_gemv_fp16(const half* __restrict__ W,
                 const half* __restrict__ x,
                 half* __restrict__ y,
                 const half* __restrict__ residual,
                 int out_features, int in_features) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    const half* w_row = W + (size_t)row * in_features;
    int tid = threadIdx.x;

    // Use half2 vectorized loads when possible
    float sum = 0.0f;
    int in_half2 = in_features / 2;
    const half2* w_row2 = reinterpret_cast<const half2*>(w_row);
    const half2* x2 = reinterpret_cast<const half2*>(x);

    for (int j = tid; j < in_half2; j += blockDim.x) {
        half2 wv = w_row2[j];
        half2 xv = x2[j];
        sum += __half2float(wv.x) * __half2float(xv.x);
        sum += __half2float(wv.y) * __half2float(xv.y);
    }

    // Handle odd element if in_features is odd
    if (in_features & 1) {
        int last = in_features - 1;
        if (tid == 0) {
            sum += __half2float(w_row[last]) * __half2float(x[last]);
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[8];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0) {
            if constexpr (ADD_RESIDUAL) {
                sum += __half2float(residual[row]);
            }
            y[row] = __float2half(sum);
        }
    }
}

void gwen_gemv_fp16(const half* W, const half* x, half* y,
                    int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp16<false><<<out_features, 256, 0, stream>>>(
        W, x, y, nullptr, out_features, in_features);
}

void gwen_gemv_fp16_residual(const half* W, const half* x, half* y, const half* residual,
                              int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp16<true><<<out_features, 256, 0, stream>>>(
        W, x, y, residual, out_features, in_features);
}

} // namespace gwen
