// FP4 E2M1 GEMV kernels for bandwidth-optimized decode
// Weight format: NVFP4 — packed FP4 E2M1 bytes with per-block E4M3 micro-scales
//
// Layout per weight tensor:
//   data:     [out_features, in_features/2] uint8 (2 FP4 values per byte)
//   scales:   [out_features, in_features/16] uint8 (E4M3 micro-scale per 16 elements)
//   scale2:   float32 per-tensor global scale
//
// Effective value: fp4_val * e4m3_block_scale * float32_global_scale
//
// Architecture: 256 threads/block, 1 row per block, 8 FP4 elements per thread per iteration.
// 8 elements = 4 packed bytes (uint32 load), 1 E4M3 scale (shared with adjacent 8 elements).
// For K=2560: all 256 threads active. For K=9216: 4-5 iterations per thread.

#include "gwen/common.h"
#include <cuda_fp16.h>

namespace gwen {

// FP4 E2M1 LUT: 16 entries as FP16 bit patterns
static __device__ __constant__ uint16_t fp4_lut[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00,  // +0.0, +0.5, +1.0, +1.5
    0x4000, 0x4200, 0x4400, 0x4600,  // +2.0, +3.0, +4.0, +6.0
    0x8000, 0xB800, 0xBC00, 0xBE00,  // -0.0, -0.5, -1.0, -1.5
    0xC000, 0xC200, 0xC400, 0xC600,  // -2.0, -3.0, -4.0, -6.0
};

// Single E4M3 byte → float via PTX (pack with zero byte, convert pair, take low half)
static __device__ __forceinline__ float e4m3_to_float_ptx(uint8_t val) {
    uint16_t packed = val;  // zero-extends: low=val, high=0
    uint32_t result;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(result) : "h"(packed));
    half2 h2 = reinterpret_cast<half2 const&>(result);
    return __half2float(__low2half(h2));
}

// Paired E4M3 conversion (for 32-element iterations)
static __device__ __forceinline__ half2 e4m3x2_to_half2(uint16_t packed) {
    uint32_t result;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(result) : "h"(packed));
    return reinterpret_cast<half2 const&>(result);
}

// Pre-computed FP4 LUT as floats (loaded into shared memory at block start)
static __device__ __constant__ float fp4_lut_f32[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// ============================================================
// Inner dot product: 8 FP4 elements per thread per iteration
// ============================================================
// Shared memory float LUT eliminates per-element half→float conversion
// and avoids constant memory serialization on divergent accesses.

template <int BLOCK_DIM>
__device__ __forceinline__ float fp4_dot_partial(
    const uint8_t* __restrict__ W_row,
    const uint8_t* __restrict__ scales_row,
    const half* __restrict__ x,
    int K,
    const float* __restrict__ lut)  // shared memory LUT
{
    int tid = threadIdx.x;
    float sumf = 0.0f;

    for (int j = tid * 8; j < K; j += BLOCK_DIM * 8) {
        uint32_t w32 = *reinterpret_cast<const uint32_t*>(W_row + j / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w32);

        float s = e4m3_to_float_ptx(scales_row[j / 16]);

        const half2* xp = reinterpret_cast<const half2*>(x + j);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint8_t b = bytes[k];
            float w_lo = lut[b & 0xF] * s;
            float w_hi = lut[b >> 4] * s;
            float2 xf = __half22float2(xp[k]);
            sumf = fmaf(w_lo, xf.x, sumf);
            sumf = fmaf(w_hi, xf.y, sumf);
        }
    }
    return sumf;
}

// Reduction: warp shuffle + cross-warp shared memory
template <int BLOCK_DIM>
__device__ __forceinline__ float reduce_sum(float sumf, int tid) {
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
// Kernel: FP4 GEMV — FP16 output
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp4(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ scales,
    float global_scale,
    const half* __restrict__ x,
    half* __restrict__ y,
    int M, int K)
{
    __shared__ float lut[16];
    int tid = threadIdx.x;
    if (tid < 16) lut[tid] = fp4_lut_f32[tid];
    __syncthreads();

    int row = blockIdx.x;
    if (row >= M) return;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sumf = fp4_dot_partial<256>(W_row, S_row, x, K, lut);
    sumf = reduce_sum<256>(sumf, tid);

    if (tid == 0)
        y[row] = __float2half(sumf * global_scale);
}

// ============================================================
// Kernel: FP4 GEMV — F32 output with F32 residual
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp4_residual_f32(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ scales,
    float global_scale,
    const half* __restrict__ x,
    float* __restrict__ y_f32,
    const float* __restrict__ residual_f32,
    int M, int K)
{
    __shared__ float lut[16];
    int tid = threadIdx.x;
    if (tid < 16) lut[tid] = fp4_lut_f32[tid];
    __syncthreads();

    int row = blockIdx.x;
    if (row >= M) return;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sumf = fp4_dot_partial<256>(W_row, S_row, x, K, lut);
    sumf = reduce_sum<256>(sumf, tid);

    if (tid == 0)
        y_f32[row] = sumf * global_scale + residual_f32[row];
}

// ============================================================
// Kernel: FP4 GEMV batch2 — read weights once, 2 dot products
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp4_batch2(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ scales,
    float global_scale,
    const half* __restrict__ x0, const half* __restrict__ x1,
    half* __restrict__ y0, half* __restrict__ y1,
    int M, int K)
{
    __shared__ float lut[16];
    int tid = threadIdx.x;
    if (tid < 16) lut[tid] = fp4_lut_f32[tid];
    __syncthreads();

    int row = blockIdx.x;
    if (row >= M) return;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sf0 = 0.0f, sf1 = 0.0f;

    for (int j = tid * 8; j < K; j += 256 * 8) {
        uint32_t w32 = *reinterpret_cast<const uint32_t*>(W_row + j / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w32);

        float s = e4m3_to_float_ptx(S_row[j / 16]);

        const half2* xp0 = reinterpret_cast<const half2*>(x0 + j);
        const half2* xp1 = reinterpret_cast<const half2*>(x1 + j);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint8_t b = bytes[k];
            float w_lo = lut[b & 0xF] * s;
            float w_hi = lut[b >> 4] * s;
            float2 xf0 = __half22float2(xp0[k]);
            float2 xf1 = __half22float2(xp1[k]);
            sf0 = fmaf(w_lo, xf0.x, sf0); sf0 = fmaf(w_hi, xf0.y, sf0);
            sf1 = fmaf(w_lo, xf1.x, sf1); sf1 = fmaf(w_hi, xf1.y, sf1);
        }
    }

    // Dual reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
        sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
    }

    __shared__ float tmp[2][8];
    int warp_id = tid / 32, lane = tid % 32;
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
            y0[row] = __float2half(sf0 * global_scale);
            y1[row] = __float2half(sf1 * global_scale);
        }
    }
}

// ============================================================
// Kernel: FP4 GEMV batch2 — F32 output with F32 residual
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp4_batch2_residual_f32(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ scales,
    float global_scale,
    const half* __restrict__ x0, const half* __restrict__ x1,
    float* __restrict__ y0_f32, float* __restrict__ y1_f32,
    const float* __restrict__ res0_f32, const float* __restrict__ res1_f32,
    int M, int K)
{
    __shared__ float lut[16];
    int tid = threadIdx.x;
    if (tid < 16) lut[tid] = fp4_lut_f32[tid];
    __syncthreads();

    int row = blockIdx.x;
    if (row >= M) return;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sf0 = 0.0f, sf1 = 0.0f;

    for (int j = tid * 8; j < K; j += 256 * 8) {
        uint32_t w32 = *reinterpret_cast<const uint32_t*>(W_row + j / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w32);

        float s = e4m3_to_float_ptx(S_row[j / 16]);

        const half2* xp0 = reinterpret_cast<const half2*>(x0 + j);
        const half2* xp1 = reinterpret_cast<const half2*>(x1 + j);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint8_t b = bytes[k];
            float w_lo = lut[b & 0xF] * s;
            float w_hi = lut[b >> 4] * s;
            float2 xf0 = __half22float2(xp0[k]);
            float2 xf1 = __half22float2(xp1[k]);
            sf0 = fmaf(w_lo, xf0.x, sf0); sf0 = fmaf(w_hi, xf0.y, sf0);
            sf1 = fmaf(w_lo, xf1.x, sf1); sf1 = fmaf(w_hi, xf1.y, sf1);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
        sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
    }

    __shared__ float tmp[2][8];
    int warp_id = tid / 32, lane = tid % 32;
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
            y0_f32[row] = sf0 * global_scale + res0_f32[row];
            y1_f32[row] = sf1 * global_scale + res1_f32[row];
        }
    }
}

// ============================================================
// Launch wrappers
// ============================================================

void gwen_gemv_fp4(const void* W, const void* scales, float global_scale,
                    const half* x, half* y,
                    int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp4<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W),
        static_cast<const uint8_t*>(scales),
        global_scale, x, y, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_fp4_residual_f32(const void* W, const void* scales, float global_scale,
                                  const half* x, float* y_f32, const float* residual_f32,
                                  int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp4_residual_f32<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W),
        static_cast<const uint8_t*>(scales),
        global_scale, x, y_f32, residual_f32, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_fp4_batch2(const void* W, const void* scales, float global_scale,
                            const half* x0, const half* x1,
                            half* y0, half* y1,
                            int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp4_batch2<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W),
        static_cast<const uint8_t*>(scales),
        global_scale, x0, x1, y0, y1, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_fp4_batch2_residual_f32(const void* W, const void* scales, float global_scale,
                                         const half* x0, const half* x1,
                                         float* y0_f32, float* y1_f32,
                                         const float* res0_f32, const float* res1_f32,
                                         int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_fp4_batch2_residual_f32<<<out_features, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W),
        static_cast<const uint8_t*>(scales),
        global_scale, x0, x1, y0_f32, y1_f32, res0_f32, res1_f32,
        out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
