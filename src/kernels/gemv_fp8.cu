#include "gwen/kernels.h"
#include <cuda_fp8.h>

namespace gwen {

// ============================================================
// FP8 E4M3 GEMV — decode path
// ============================================================
// W: [out_features, in_features] in FP8 E4M3 (1 byte per element)
// scales: [out_features] F32 per-row scale factors
// x: [in_features] FP16 input vector
//
// Design based on:
// - FBGEMM fp8fp8bf16_fast_gemv.cu (Meta, production FP8 GEMV)
// - CUTLASS gemv_blockscaled.h (NVIDIA reference)
// - Blackwell NVFP4 hackathon findings (cache hints, register budget)
//
// Key decisions:
// - 128-bit vectorized loads (float4 = 16 FP8 elements)
// - PTX cvt.rn.f16x2.e4m3x2 for conversion (avoids known C++ intrinsic bugs)
// - F32 accumulation throughout (bandwidth-bound, compute is free)
// - 256 threads/block, 1 row per block (maximum parallelism for lm_head 248K rows)
// - Per-row F32 scale applied after reduction

// Convert 2 packed FP8 E4M3 bytes to half2 via PTX (safer than C intrinsic)
static __device__ __forceinline__ half2 fp8x2_to_half2(uint16_t packed_fp8) {
    uint32_t result;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(result) : "h"(packed_fp8));
    return reinterpret_cast<half2 const&>(result);
}

// Inner loop: load 16 FP8 weights, convert to FP16, dot product with FP16 input, accumulate F32
// Returns partial sum for this thread across all iterations
template<int BLOCK_DIM>
static __device__ __forceinline__ float fp8_dot_partial(
    const uint8_t* __restrict__ W_row,
    const half* __restrict__ x,
    int in_features, int tid)
{
    float sumf = 0.0f;

    // Each thread loads 16 FP8 bytes (128-bit) per iteration
    for (int j = tid * 16; j < in_features; j += BLOCK_DIM * 16) {
        // 128-bit coalesced load of 16 FP8 elements
        float4 w128 = *reinterpret_cast<const float4*>(W_row + j);

        // Extract as 8 pairs of uint16_t, convert each to half2
        const uint16_t* pairs = reinterpret_cast<const uint16_t*>(&w128);
        const half2* xp = reinterpret_cast<const half2*>(x + j);

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            half2 wh = fp8x2_to_half2(pairs[k]);
            half2 xh = xp[k];
            // F32 accumulation (bandwidth-bound, compute is free)
            float2 wf = __half22float2(wh);
            float2 xf = __half22float2(xh);
            sumf = fmaf(wf.x, xf.x, sumf);
            sumf = fmaf(wf.y, xf.y, sumf);
        }
    }
    return sumf;
}

// Reduction: warp shuffle + cross-warp shared memory
template<int BLOCK_DIM>
static __device__ __forceinline__ float reduce_sum(float sumf, int tid) {
    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumf += __shfl_xor_sync(0xFFFFFFFF, sumf, offset);

    constexpr int N_WARPS = BLOCK_DIM / 32;
    __shared__ float warp_sums[N_WARPS];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = sumf;
    __syncthreads();

    if (warp_id == 0) {
        sumf = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = N_WARPS / 2; offset > 0; offset >>= 1)
            sumf += __shfl_xor_sync(0xFFFFFFFF, sumf, offset);
    }
    return sumf;
}

// ============================================================
// Kernel: FP8 GEMV → FP16 output (no residual)
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp8(
    const uint8_t* __restrict__ W,
    const float* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int out_features, int in_features)
{
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;

    float sumf = fp8_dot_partial<256>(W + (size_t)row * in_features, x, in_features, tid);
    sumf = reduce_sum<256>(sumf, tid);

    if (tid == 0)
        y[row] = __float2half(sumf * scales[row]);
}

// ============================================================
// Kernel: FP8 GEMV → F32 output + F32 residual add
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp8_residual_f32(
    const uint8_t* __restrict__ W,
    const float* __restrict__ scales,
    const half* __restrict__ x,
    float* __restrict__ y_f32,
    const float* __restrict__ residual_f32,
    int out_features, int in_features)
{
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;

    float sumf = fp8_dot_partial<256>(W + (size_t)row * in_features, x, in_features, tid);
    sumf = reduce_sum<256>(sumf, tid);

    if (tid == 0)
        y_f32[row] = sumf * scales[row] + residual_f32[row];
}

// ============================================================
// Kernel: FP8 GEMV batch2 → FP16 output (read weights once, 2 dots)
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp8_batch2(
    const uint8_t* __restrict__ W,
    const float* __restrict__ scales,
    const half* __restrict__ x0, const half* __restrict__ x1,
    half* __restrict__ y0, half* __restrict__ y1,
    int out_features, int in_features)
{
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;

    const uint8_t* W_row = W + (size_t)row * in_features;
    float sf0 = 0.0f, sf1 = 0.0f;

    for (int j = tid * 16; j < in_features; j += 256 * 16) {
        float4 w128 = *reinterpret_cast<const float4*>(W_row + j);
        const uint16_t* pairs = reinterpret_cast<const uint16_t*>(&w128);
        const half2* xp0 = reinterpret_cast<const half2*>(x0 + j);
        const half2* xp1 = reinterpret_cast<const half2*>(x1 + j);

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            half2 wh = fp8x2_to_half2(pairs[k]);
            float2 wf = __half22float2(wh);
            float2 xf0 = __half22float2(xp0[k]);
            float2 xf1 = __half22float2(xp1[k]);
            sf0 = fmaf(wf.x, xf0.x, sf0); sf0 = fmaf(wf.y, xf0.y, sf0);
            sf1 = fmaf(wf.x, xf1.x, sf1); sf1 = fmaf(wf.y, xf1.y, sf1);
        }
    }

    // Dual reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
        sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
    }

    __shared__ float tmp[2][8];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) { tmp[0][warp_id] = sf0; tmp[1][warp_id] = sf1; }
    __syncthreads();

    if (warp_id == 0) {
        sf0 = (lane < 8) ? tmp[0][lane] : 0.0f;
        sf1 = (lane < 8) ? tmp[1][lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sf0 += __shfl_xor_sync(0xFF, sf0, offset);
            sf1 += __shfl_xor_sync(0xFF, sf1, offset);
        }
        if (lane == 0) {
            float s = scales[row];
            y0[row] = __float2half(sf0 * s);
            y1[row] = __float2half(sf1 * s);
        }
    }
}

// ============================================================
// Kernel: FP8 GEMV batch2 → F32 output + F32 residual
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp8_batch2_residual_f32(
    const uint8_t* __restrict__ W,
    const float* __restrict__ scales,
    const half* __restrict__ x0, const half* __restrict__ x1,
    float* __restrict__ y0_f32, float* __restrict__ y1_f32,
    const float* __restrict__ res0_f32, const float* __restrict__ res1_f32,
    int out_features, int in_features)
{
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;

    const uint8_t* W_row = W + (size_t)row * in_features;
    float sf0 = 0.0f, sf1 = 0.0f;

    for (int j = tid * 16; j < in_features; j += 256 * 16) {
        float4 w128 = *reinterpret_cast<const float4*>(W_row + j);
        const uint16_t* pairs = reinterpret_cast<const uint16_t*>(&w128);
        const half2* xp0 = reinterpret_cast<const half2*>(x0 + j);
        const half2* xp1 = reinterpret_cast<const half2*>(x1 + j);

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            half2 wh = fp8x2_to_half2(pairs[k]);
            float2 wf = __half22float2(wh);
            float2 xf0 = __half22float2(xp0[k]);
            float2 xf1 = __half22float2(xp1[k]);
            sf0 = fmaf(wf.x, xf0.x, sf0); sf0 = fmaf(wf.y, xf0.y, sf0);
            sf1 = fmaf(wf.x, xf1.x, sf1); sf1 = fmaf(wf.y, xf1.y, sf1);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
        sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
    }

    __shared__ float tmp[2][8];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) { tmp[0][warp_id] = sf0; tmp[1][warp_id] = sf1; }
    __syncthreads();

    if (warp_id == 0) {
        sf0 = (lane < 8) ? tmp[0][lane] : 0.0f;
        sf1 = (lane < 8) ? tmp[1][lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sf0 += __shfl_xor_sync(0xFF, sf0, offset);
            sf1 += __shfl_xor_sync(0xFF, sf1, offset);
        }
        if (lane == 0) {
            float s = scales[row];
            y0_f32[row] = sf0 * s + res0_f32[row];
            y1_f32[row] = sf1 * s + res1_f32[row];
        }
    }
}

// ============================================================
// F32-input RMSNorm (reads F32 residual stream, writes FP16)
// ============================================================

__global__ void __launch_bounds__(32)
kernel_rmsnorm_f32_input(
    const float* __restrict__ x_f32,
    const float* __restrict__ weight,
    half* __restrict__ y,
    int dim, float eps)
{
    int lane = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float val = x_f32[i];
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    float rms_inv = rsqrtf(sum_sq / dim + eps);
    for (int i = lane; i < dim; i += 32) {
        y[i] = __float2half(x_f32[i] * rms_inv * weight[i]);
    }
}

// Batch2: F32-input RMSNorm for two tokens in one launch (2 blocks × 32 threads)
__global__ void __launch_bounds__(32)
kernel_rmsnorm_f32_input_batch2(
    const float* __restrict__ x0_f32,
    const float* __restrict__ x1_f32,
    const float* __restrict__ weight,
    half* __restrict__ y0,
    half* __restrict__ y1,
    int dim, float eps)
{
    const float* x = (blockIdx.x == 0) ? x0_f32 : x1_f32;
    half* y = (blockIdx.x == 0) ? y0 : y1;
    int lane = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float val = x[i];
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    float rms_inv = rsqrtf(sum_sq / dim + eps);
    for (int i = lane; i < dim; i += 32) {
        y[i] = __float2half(x[i] * rms_inv * weight[i]);
    }
}

// FP16→F32 conversion (embedding → F32 residual init)
__global__ void __launch_bounds__(256)
kernel_fp16_to_f32(const half* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) y[idx] = __half2float(x[idx]);
}

// FP16→F32 conversion + F32 residual add: y[i] = half2float(x[i]) + residual[i]
__global__ void __launch_bounds__(256)
kernel_fp16_to_f32_add(const half* __restrict__ x, float* __restrict__ y,
                        const float* __restrict__ residual, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) y[idx] = __half2float(x[idx]) + residual[idx];
}

// F32→FP16 conversion (F32 residual → FP16 for MTP hidden copy)
__global__ void __launch_bounds__(256)
kernel_f32_to_fp16(const float* __restrict__ x, half* __restrict__ y, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) y[idx] = __float2half(x[idx]);
}

// ============================================================
// FP8→FP16 bulk dequant with per-row scaling (for prefill GEMM fallback)
// ============================================================
__global__ void __launch_bounds__(256)
kernel_dequant_fp8_to_fp16(
    const uint8_t* __restrict__ data,
    const float* __restrict__ scales,
    half* __restrict__ out,
    int rows, int cols)
{
    int idx = blockIdx.x * 256 + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    __nv_fp8_storage_t fp8_val = data[idx];
    __half_raw hr = __nv_cvt_fp8_to_halfraw(fp8_val, __NV_E4M3);
    half h = *reinterpret_cast<half*>(&hr);
    out[idx] = __float2half(__half2float(h) * scales[row]);
}

// ============================================================
// Launch wrappers
// ============================================================

void gwen_gemv_fp8(const void* W, const float* scales, const half* x, half* y,
                    int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp8<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W), scales, x, y, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_fp8_residual_f32(const void* W, const float* scales, const half* x,
                                  float* y_f32, const float* residual_f32,
                                  int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp8_residual_f32<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W), scales, x, y_f32, residual_f32,
        out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_fp8_batch2(const void* W, const float* scales,
                            const half* x0, const half* x1,
                            half* y0, half* y1,
                            int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp8_batch2<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W), scales, x0, x1, y0, y1,
        out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_fp8_batch2_residual_f32(const void* W, const float* scales,
                                         const half* x0, const half* x1,
                                         float* y0_f32, float* y1_f32,
                                         const float* res0_f32, const float* res1_f32,
                                         int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp8_batch2_residual_f32<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W), scales, x0, x1, y0_f32, y1_f32,
        res0_f32, res1_f32, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_rmsnorm_f32_input(const float* x_f32, const float* weight, half* y,
                              int dim, float eps, cudaStream_t stream) {
    kernel_rmsnorm_f32_input<<<1, 32, 0, stream>>>(x_f32, weight, y, dim, eps);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_rmsnorm_f32_input_batch2(const float* x0_f32, const float* x1_f32,
                                    const float* weight,
                                    half* y0, half* y1,
                                    int dim, float eps, cudaStream_t stream) {
    kernel_rmsnorm_f32_input_batch2<<<2, 32, 0, stream>>>(
        x0_f32, x1_f32, weight, y0, y1, dim, eps);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_fp16_to_f32(const half* x, float* y, int n, cudaStream_t stream) {
    kernel_fp16_to_f32<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_fp16_to_f32_add(const half* x, float* y, const float* residual, int n, cudaStream_t stream) {
    kernel_fp16_to_f32_add<<<(n + 255) / 256, 256, 0, stream>>>(x, y, residual, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_f32_to_fp16(const float* x, half* y, int n, cudaStream_t stream) {
    kernel_f32_to_fp16<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_dequant_fp8_to_fp16(const void* data, const float* scales, half* out,
                                int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    kernel_dequant_fp8_to_fp16<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint8_t*>(data), scales, out, rows, cols);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
