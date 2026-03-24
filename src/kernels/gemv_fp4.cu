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
// Kernel architecture: 256 threads/block, 1 block per output row
// Inner loop: 128-bit vectorized weight loads → 32 FP4 elements per float4
// FP4→FP16 via 16-entry constant-memory LUT
// F32 accumulation throughout (bandwidth-bound, compute is free)

#include "gwen/common.h"
#include <cuda_fp16.h>

namespace gwen {

// ============================================================
// FP4 E2M1 value table (16 entries → FP16 bit patterns)
// ============================================================
// Encoding: [sign(1)][exp(2)][mantissa(1)]
// 0000=+0.0  0001=+0.5  0010=+1.0  0011=+1.5
// 0100=+2.0  0101=+3.0  0110=+4.0  0111=+6.0
// 1000=-0.0  1001=-0.5  1010=-1.0  1011=-1.5
// 1100=-2.0  1101=-3.0  1110=-4.0  1111=-6.0

static __device__ __constant__ uint16_t fp4_lut[16] = {
    0x0000, // +0.0
    0x3800, // +0.5
    0x3C00, // +1.0
    0x3E00, // +1.5
    0x4000, // +2.0
    0x4200, // +3.0
    0x4400, // +4.0
    0x4600, // +6.0
    0x8000, // -0.0
    0xB800, // -0.5
    0xBC00, // -1.0
    0xBE00, // -1.5
    0xC000, // -2.0
    0xC200, // -3.0
    0xC400, // -4.0
    0xC600, // -6.0
};

// ============================================================
// E4M3 FP8 → float conversion (for micro-scales)
// ============================================================
__device__ __forceinline__ float e4m3_to_float(uint8_t val) {
    // E4M3: [sign(1)][exp(4)][mantissa(3)], bias=7, no inf/nan
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 3) & 0xF;
    uint32_t mant = val & 0x7;

    float result;
    if (exp == 0) {
        // Subnormal: (-1)^s × 2^(-6) × (0.mantissa)
        result = ldexpf((float)mant / 8.0f, -6);
    } else {
        // Normal: (-1)^s × 2^(exp-7) × (1.mantissa)
        result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -result : result;
}

// ============================================================
// Inner dot product: FP4 weights × FP16 input, with block scales
// ============================================================
// Each thread processes 32 FP4 elements per iteration (16 bytes of packed data)
// Block scales: 1 E4M3 scale per 16 FP4 elements → 2 scales per 32-element chunk

template <int BLOCK_DIM>
__device__ float fp4_dot_partial(
    const uint8_t* __restrict__ W_row,        // [in_features/2] packed FP4
    const uint8_t* __restrict__ scales_row,   // [in_features/16] E4M3 scales
    const half* __restrict__ x,               // [in_features] FP16
    int in_features)
{
    int tid = threadIdx.x;
    float sumf = 0.0f;

    // Each iteration: 32 FP4 elements = 16 bytes of weight data
    // 2 micro-scale blocks of 16 elements each
    for (int j = tid * 32; j < in_features; j += BLOCK_DIM * 32) {
        // Load 16 bytes = 32 packed FP4 values via 128-bit vector load
        float4 w128 = *reinterpret_cast<const float4*>(W_row + j / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w128);

        // Load 2 block scales (each covers 16 elements)
        int scale_idx = j / 16;
        float s0 = e4m3_to_float(scales_row[scale_idx]);
        float s1 = e4m3_to_float(scales_row[scale_idx + 1]);

        // Process first 16 elements (8 bytes, scale s0)
        for (int k = 0; k < 8; k++) {
            uint8_t b = bytes[k];
            float w_lo = __half2float(__ushort_as_half(fp4_lut[b & 0xF])) * s0;
            float w_hi = __half2float(__ushort_as_half(fp4_lut[b >> 4])) * s0;
            float x0 = __half2float(x[j + k * 2]);
            float x1 = __half2float(x[j + k * 2 + 1]);
            sumf = fmaf(w_lo, x0, sumf);
            sumf = fmaf(w_hi, x1, sumf);
        }

        // Process next 16 elements (8 bytes, scale s1)
        for (int k = 8; k < 16; k++) {
            uint8_t b = bytes[k];
            float w_lo = __half2float(__ushort_as_half(fp4_lut[b & 0xF])) * s1;
            float w_hi = __half2float(__ushort_as_half(fp4_lut[b >> 4])) * s1;
            float x0 = __half2float(x[j + k * 2]);
            float x1 = __half2float(x[j + k * 2 + 1]);
            sumf = fmaf(w_lo, x0, sumf);
            sumf = fmaf(w_hi, x1, sumf);
        }
    }
    return sumf;
}

// ============================================================
// Warp + cross-warp reduction
// ============================================================
template <int BLOCK_DIM>
__device__ float reduce_sum(float val) {
    constexpr int N_WARPS = BLOCK_DIM / 32;

    // Intra-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

    // Cross-warp via shared memory
    __shared__ float warp_sums[N_WARPS];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        for (int offset = N_WARPS / 2; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================
// Kernel: FP4 GEMV — FP16 output
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_fp4(
    const uint8_t* __restrict__ W,         // [M, K/2] packed FP4
    const uint8_t* __restrict__ scales,    // [M, K/16] E4M3 micro-scales
    float global_scale,                     // per-tensor scale
    const half* __restrict__ x,            // [K] FP16 input
    half* __restrict__ y,                  // [M] FP16 output
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sumf = fp4_dot_partial<256>(W_row, S_row, x, K);
    sumf = reduce_sum<256>(sumf);

    if (threadIdx.x == 0)
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
    int row = blockIdx.x;
    if (row >= M) return;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sumf = fp4_dot_partial<256>(W_row, S_row, x, K);
    sumf = reduce_sum<256>(sumf);

    if (threadIdx.x == 0)
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
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sf0 = 0.0f, sf1 = 0.0f;

    for (int j = tid * 32; j < K; j += 256 * 32) {
        float4 w128 = *reinterpret_cast<const float4*>(W_row + j / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w128);

        int scale_idx = j / 16;
        float s0 = e4m3_to_float(S_row[scale_idx]);
        float s1 = e4m3_to_float(S_row[scale_idx + 1]);

        // First 16 elements (scale s0)
        for (int k = 0; k < 8; k++) {
            uint8_t b = bytes[k];
            float w_lo = __half2float(__ushort_as_half(fp4_lut[b & 0xF])) * s0;
            float w_hi = __half2float(__ushort_as_half(fp4_lut[b >> 4])) * s0;
            int idx = j + k * 2;
            sf0 = fmaf(w_lo, __half2float(x0[idx]), sf0);
            sf0 = fmaf(w_hi, __half2float(x0[idx + 1]), sf0);
            sf1 = fmaf(w_lo, __half2float(x1[idx]), sf1);
            sf1 = fmaf(w_hi, __half2float(x1[idx + 1]), sf1);
        }

        // Next 16 elements (scale s1)
        for (int k = 8; k < 16; k++) {
            uint8_t b = bytes[k];
            float w_lo = __half2float(__ushort_as_half(fp4_lut[b & 0xF])) * s1;
            float w_hi = __half2float(__ushort_as_half(fp4_lut[b >> 4])) * s1;
            int idx = j + k * 2;
            sf0 = fmaf(w_lo, __half2float(x0[idx]), sf0);
            sf0 = fmaf(w_hi, __half2float(x0[idx + 1]), sf0);
            sf1 = fmaf(w_lo, __half2float(x1[idx]), sf1);
            sf1 = fmaf(w_hi, __half2float(x1[idx + 1]), sf1);
        }
    }

    // Reduce both sums
    constexpr int N_WARPS = 8;
    __shared__ float warp_s0[N_WARPS], warp_s1[N_WARPS];
    int warp_id = tid / 32, lane = tid % 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
        sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
    }
    if (lane == 0) { warp_s0[warp_id] = sf0; warp_s1[warp_id] = sf1; }
    __syncthreads();

    if (warp_id == 0) {
        sf0 = (lane < N_WARPS) ? warp_s0[lane] : 0.0f;
        sf1 = (lane < N_WARPS) ? warp_s1[lane] : 0.0f;
        for (int offset = N_WARPS / 2; offset > 0; offset >>= 1) {
            sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
            sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
        }
    }

    if (tid == 0) {
        y0[row] = __float2half(sf0 * global_scale);
        y1[row] = __float2half(sf1 * global_scale);
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
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    const uint8_t* W_row = W + (size_t)row * (K / 2);
    const uint8_t* S_row = scales + (size_t)row * (K / 16);

    float sf0 = 0.0f, sf1 = 0.0f;

    for (int j = tid * 32; j < K; j += 256 * 32) {
        float4 w128 = *reinterpret_cast<const float4*>(W_row + j / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w128);

        int scale_idx = j / 16;
        float s0 = e4m3_to_float(S_row[scale_idx]);
        float s1 = e4m3_to_float(S_row[scale_idx + 1]);

        for (int k = 0; k < 8; k++) {
            uint8_t b = bytes[k];
            float w_lo = __half2float(__ushort_as_half(fp4_lut[b & 0xF])) * s0;
            float w_hi = __half2float(__ushort_as_half(fp4_lut[b >> 4])) * s0;
            int idx = j + k * 2;
            sf0 = fmaf(w_lo, __half2float(x0[idx]), sf0);
            sf0 = fmaf(w_hi, __half2float(x0[idx + 1]), sf0);
            sf1 = fmaf(w_lo, __half2float(x1[idx]), sf1);
            sf1 = fmaf(w_hi, __half2float(x1[idx + 1]), sf1);
        }

        for (int k = 8; k < 16; k++) {
            uint8_t b = bytes[k];
            float w_lo = __half2float(__ushort_as_half(fp4_lut[b & 0xF])) * s1;
            float w_hi = __half2float(__ushort_as_half(fp4_lut[b >> 4])) * s1;
            int idx = j + k * 2;
            sf0 = fmaf(w_lo, __half2float(x0[idx]), sf0);
            sf0 = fmaf(w_hi, __half2float(x0[idx + 1]), sf0);
            sf1 = fmaf(w_lo, __half2float(x1[idx]), sf1);
            sf1 = fmaf(w_hi, __half2float(x1[idx + 1]), sf1);
        }
    }

    constexpr int N_WARPS = 8;
    __shared__ float warp_s0[N_WARPS], warp_s1[N_WARPS];
    int warp_id = tid / 32, lane = tid % 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
        sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
    }
    if (lane == 0) { warp_s0[warp_id] = sf0; warp_s1[warp_id] = sf1; }
    __syncthreads();

    if (warp_id == 0) {
        sf0 = (lane < N_WARPS) ? warp_s0[lane] : 0.0f;
        sf1 = (lane < N_WARPS) ? warp_s1[lane] : 0.0f;
        for (int offset = N_WARPS / 2; offset > 0; offset >>= 1) {
            sf0 += __shfl_xor_sync(0xFFFFFFFF, sf0, offset);
            sf1 += __shfl_xor_sync(0xFFFFFFFF, sf1, offset);
        }
    }

    if (tid == 0) {
        y0_f32[row] = sf0 * global_scale + res0_f32[row];
        y1_f32[row] = sf1 * global_scale + res1_f32[row];
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
