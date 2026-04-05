#include "gwen/kernels.h"

namespace gwen {

// Online softmax: numerically stable, single pass
// One warp per row
__global__ void __launch_bounds__(32)
kernel_softmax(const half* __restrict__ x, half* __restrict__ y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int lane = threadIdx.x;

    const half* x_row = x + row * cols;
    half* y_row = y + row * cols;

    // Pass 1: find max
    float max_val = -1e30f;
    for (int i = lane; i < cols; i += 32) {
        float val = __half2float(x_row[i]);
        max_val = fmaxf(max_val, val);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // Pass 2: compute exp and sum
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32) {
        float val = expf(__half2float(x_row[i]) - max_val);
        sum += val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    float inv_sum = 1.0f / sum;

    // Pass 3: normalize
    for (int i = lane; i < cols; i += 32) {
        float val = expf(__half2float(x_row[i]) - max_val);
        y_row[i] = __float2half(val * inv_sum);
    }
}

// Causal softmax for attention scores
// Input: float scores [n_heads, seq_len] (scores for one query against all keys)
// Output: half probs [n_heads, seq_len]
// Masks out positions > current_pos
__global__ void __launch_bounds__(32)
kernel_causal_softmax(const float* __restrict__ x, half* __restrict__ y,
                      int n_heads, int seq_len) {
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int lane = threadIdx.x;

    const float* x_row = x + head * seq_len;
    half* y_row = y + head * seq_len;

    // Find max (only over valid positions, which is all seq_len for causal)
    float max_val = -1e30f;
    for (int i = lane; i < seq_len; i += 32) {
        max_val = fmaxf(max_val, x_row[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // Exp and sum
    float sum = 0.0f;
    for (int i = lane; i < seq_len; i += 32) {
        float val = expf(x_row[i] - max_val);
        sum += val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    float inv_sum = 1.0f / sum;

    // Normalize
    for (int i = lane; i < seq_len; i += 32) {
        float val = expf(x_row[i] - max_val);
        y_row[i] = __float2half(val * inv_sum);
    }
}

void gwen_softmax(const half* x, half* y, int rows, int cols, cudaStream_t stream) {
    kernel_softmax<<<rows, 32, 0, stream>>>(x, y, rows, cols);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_causal_softmax(const float* x, half* y, int n_heads, int seq_len,
                         cudaStream_t stream) {
    kernel_causal_softmax<<<n_heads, 32, 0, stream>>>(x, y, n_heads, seq_len);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
