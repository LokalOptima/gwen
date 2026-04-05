#include "gwen/kernels.h"

namespace gwen {

// Row-wise logsumexp for [n_rows, n_cols] FP16 → [n_rows] F32
// One block per row, 256 threads, two-pass: find max, then sum exp(x - max)
__global__ void __launch_bounds__(256)
kernel_logsumexp_rows(const half* __restrict__ x, float* __restrict__ log_Z,
                      int n_rows, int n_cols) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int tid = threadIdx.x;
    const half* x_row = x + (size_t)row * n_cols;

    float local_max = -1e30f;
    for (int i = tid; i < n_cols; i += 256)
        local_max = fmaxf(local_max, __half2float(x_row[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));

    __shared__ float smem[8];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : -1e30f;
        for (int offset = 4; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xFF, val, offset));
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += 256)
        local_sum += expf(__half2float(x_row[i]) - row_max);
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFF, val, offset);
        if (lane == 0) log_Z[row] = row_max + logf(val);
    }
}

// Compute p_idk from restricted logits + log_Z (partition function)
// p_idk = clamp(1 - sum(exp(restricted_logits - log_Z)), 0, 1)
// One warp per row
__global__ void __launch_bounds__(32)
kernel_p_idk_from_logits(const half* __restrict__ restricted_logits,
                         const float* __restrict__ log_Z,
                         float* __restrict__ p_idk,
                         int n_rows, int K) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int lane = threadIdx.x;
    float lz = log_Z[row];
    const half* r = restricted_logits + (size_t)row * K;

    float s = 0.0f;
    for (int i = lane; i < K; i += 32)
        s += expf(__half2float(r[i]) - lz);
    for (int offset = 16; offset > 0; offset >>= 1)
        s += __shfl_xor_sync(0xFFFFFFFF, s, offset);
    if (lane == 0)
        p_idk[row] = fminf(fmaxf(1.0f - s, 0.0f), 1.0f);
}

void gwen_logsumexp_rows(const half* x, float* log_Z,
                          int n_rows, int n_cols, cudaStream_t stream) {
    kernel_logsumexp_rows<<<n_rows, 256, 0, stream>>>(x, log_Z, n_rows, n_cols);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_p_idk_from_logits(const half* restricted_logits, const float* log_Z,
                              float* p_idk, int n_rows, int K, cudaStream_t stream) {
    kernel_p_idk_from_logits<<<n_rows, 32, 0, stream>>>(restricted_logits, log_Z,
                                                          p_idk, n_rows, K);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
