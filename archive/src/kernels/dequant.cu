#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

namespace gwen {

// ============================================================
// Q8_0 dequantization kernel
// ============================================================
// 32 elements per block, 1 warp per block
__global__ void __launch_bounds__(256)
kernel_dequant_q8_0(const block_q8_0* __restrict__ src, half* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (block_idx >= n_blocks) return;

    const auto& blk = src[block_idx];
    float d = __half2float(blk.d);
    float val = d * blk.qs[lane];
    dst[block_idx * 32 + lane] = __float2half(val);
}

// ============================================================
// Q4_K dequantization kernel
// ============================================================
// 256 elements per block (8 sub-blocks of 32)
// Scale reconstruction: 6-bit scales packed into 12 bytes
__global__ void __launch_bounds__(256)
kernel_dequant_q4_k(const block_q4_k* __restrict__ src, half* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    const auto& blk = src[block_idx];
    int tid = threadIdx.x;  // 0..255, one thread per output element

    float d = __half2float(blk.d);
    float dmin = __half2float(blk.dmin);

    // Determine which sub-block (0..7) and position within (0..31)
    int sub_block = tid / 32;
    (void)(tid % 32);

    // Reconstruct 6-bit scale and min for this sub-block
    // scales are packed: first 8 sub-blocks have their scales in 12 bytes
    // Layout: scales[0..3] have low 4 bits for sub-blocks 0..7 (2 per byte)
    //         scales[4..7] have low 4 bits for sub-blocks 0..7 for mins
    //         scales[8..11] have high 2 bits interleaved
    uint8_t sc_lo, m_lo;

    if (sub_block < 4) {
        sc_lo = blk.scales[sub_block] & 0x3F;
        m_lo  = blk.scales[sub_block + 4] & 0x3F;
    } else {
        sc_lo = (blk.scales[sub_block + 4] & 0xF) | ((blk.scales[sub_block - 4] >> 6) << 4);
        m_lo  = (blk.scales[sub_block + 4] >> 4) | ((blk.scales[sub_block] >> 6) << 4);
    }

    float scale = d * sc_lo;
    float min = dmin * m_lo;

    // Extract 4-bit quantized value (ggml interleaved layout)
    // Layout: groups of 64 elements, each group = 32 low nibbles + 32 high nibbles
    // from 32 consecutive bytes
    int group = tid / 64;         // 0..3
    int within = tid % 64;
    int is_high = within / 32;    // 0 or 1
    int pos = within % 32;        // 0..31
    int qs_byte_idx = group * 32 + pos;
    int q_val = is_high ? (blk.qs[qs_byte_idx] >> 4) : (blk.qs[qs_byte_idx] & 0xF);

    float result = scale * q_val - min;
    dst[block_idx * 256 + tid] = __float2half(result);
}

// ============================================================
// Q5_K dequantization kernel
// ============================================================
__global__ void __launch_bounds__(256)
kernel_dequant_q5_k(const block_q5_k* __restrict__ src, half* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    const auto& blk = src[block_idx];
    int tid = threadIdx.x;  // 0..255

    float d = __half2float(blk.d);
    float dmin = __half2float(blk.dmin);

    int sub_block = tid / 32;
    (void)(tid % 32);

    // Reconstruct scales (same packing as Q4_K)
    uint8_t sc_lo, m_lo;
    if (sub_block < 4) {
        sc_lo = blk.scales[sub_block] & 0x3F;
        m_lo  = blk.scales[sub_block + 4] & 0x3F;
    } else {
        sc_lo = (blk.scales[sub_block + 4] & 0xF) | ((blk.scales[sub_block - 4] >> 6) << 4);
        m_lo  = (blk.scales[sub_block + 4] >> 4) | ((blk.scales[sub_block] >> 6) << 4);
    }

    float scale = d * sc_lo;
    float min = dmin * m_lo;

    // Extract 4 low bits (ggml interleaved layout, same as Q4_K)
    int group = tid / 64;
    int within = tid % 64;
    int is_high = within / 32;
    int pos = within % 32;
    int qs_byte_idx = group * 32 + pos;
    int q_lo = is_high ? (blk.qs[qs_byte_idx] >> 4) : (blk.qs[qs_byte_idx] & 0xF);

    // Extract 5th bit from qh (ggml interleaved layout)
    // qh[pos] has 8 bits: bit (group*2 + is_high) corresponds to this element
    int qh_bit = group * 2 + is_high;
    int q_hi = (blk.qh[pos] >> qh_bit) & 1;

    int q_val = q_lo | (q_hi << 4);
    float result = scale * q_val - min;
    dst[block_idx * 256 + tid] = __float2half(result);
}

// ============================================================
// Q6_K dequantization kernel
// ============================================================
__global__ void __launch_bounds__(256)
kernel_dequant_q6_k(const block_q6_k* __restrict__ src, half* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    const auto& blk = src[block_idx];
    int tid = threadIdx.x;  // 0..255

    float d = __half2float(blk.d);

    // Q6_K interleaved layout (ggml):
    // 256 elements = 2 halves of 128. Each half = 4 quarters of 32.
    int half_idx = tid / 128;        // 0 or 1
    int j = tid % 128;
    int quarter = j / 32;            // 0..3
    int pos = j % 32;                // 0..31

    int ql_byte = half_idx * 64 + (quarter & 1) * 32 + pos;
    int ql_nibble = (quarter >= 2) ? (blk.ql[ql_byte] >> 4) : (blk.ql[ql_byte] & 0xF);

    int qh_byte = half_idx * 32 + pos;
    int qh_shift = quarter * 2;
    int qh_bits = (blk.qh[qh_byte] >> qh_shift) & 0x3;

    int q_val = ql_nibble | (qh_bits << 4);
    int scale_idx = half_idx * 8 + quarter * 2 + pos / 16;
    int8_t scale = blk.scales[scale_idx];
    float result = d * scale * (q_val - 32);
    dst[block_idx * 256 + tid] = __float2half(result);
}

// ============================================================
// F32 → FP16 conversion kernel
// ============================================================
__global__ void __launch_bounds__(256)
kernel_dequant_f32(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// ============================================================
// Launch wrappers
// ============================================================

void gwen_dequant_q8_0(const void* src, half* dst, int n, cudaStream_t stream) {
    int n_blocks = n / 32;
    int threads = 256;
    int blocks = (n_blocks * 32 + threads - 1) / threads;
    kernel_dequant_q8_0<<<blocks, threads, 0, stream>>>(
        static_cast<const block_q8_0*>(src), dst, n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_dequant_q4_k(const void* src, half* dst, int n, cudaStream_t stream) {
    int n_blocks = n / 256;
    kernel_dequant_q4_k<<<n_blocks, 256, 0, stream>>>(
        static_cast<const block_q4_k*>(src), dst, n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_dequant_q5_k(const void* src, half* dst, int n, cudaStream_t stream) {
    int n_blocks = n / 256;
    kernel_dequant_q5_k<<<n_blocks, 256, 0, stream>>>(
        static_cast<const block_q5_k*>(src), dst, n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_dequant_q6_k(const void* src, half* dst, int n, cudaStream_t stream) {
    int n_blocks = n / 256;
    kernel_dequant_q6_k<<<n_blocks, 256, 0, stream>>>(
        static_cast<const block_q6_k*>(src), dst, n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_dequant_f32(const float* src, half* dst, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_dequant_f32<<<blocks, threads, 0, stream>>>(src, dst, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// F32 output dequantization kernels (for verified reference path)
// Same math as FP16 versions, but stores float instead of half.
// ============================================================

__global__ void __launch_bounds__(256)
kernel_dequant_q8_0_f32(const block_q8_0* __restrict__ src, float* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (block_idx >= n_blocks) return;
    const auto& blk = src[block_idx];
    float d = __half2float(blk.d);
    dst[block_idx * 32 + lane] = d * blk.qs[lane];
}

__global__ void __launch_bounds__(256)
kernel_dequant_q4_k_f32(const block_q4_k* __restrict__ src, float* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;
    const auto& blk = src[block_idx];
    int tid = threadIdx.x;
    float d = __half2float(blk.d);
    float dmin = __half2float(blk.dmin);
    int sub_block = tid / 32;
    uint8_t sc_lo, m_lo;
    if (sub_block < 4) {
        sc_lo = blk.scales[sub_block] & 0x3F;
        m_lo  = blk.scales[sub_block + 4] & 0x3F;
    } else {
        sc_lo = (blk.scales[sub_block + 4] & 0xF) | ((blk.scales[sub_block - 4] >> 6) << 4);
        m_lo  = (blk.scales[sub_block + 4] >> 4) | ((blk.scales[sub_block] >> 6) << 4);
    }
    float scale = d * sc_lo;
    float min = dmin * m_lo;
    int group = tid / 64;
    int within = tid % 64;
    int is_high = within / 32;
    int pos = within % 32;
    int qs_byte_idx = group * 32 + pos;
    int q_val = is_high ? (blk.qs[qs_byte_idx] >> 4) : (blk.qs[qs_byte_idx] & 0xF);
    dst[block_idx * 256 + tid] = scale * q_val - min;
}

__global__ void __launch_bounds__(256)
kernel_dequant_q5_k_f32(const block_q5_k* __restrict__ src, float* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;
    const auto& blk = src[block_idx];
    int tid = threadIdx.x;
    float d = __half2float(blk.d);
    float dmin = __half2float(blk.dmin);
    int sub_block = tid / 32;
    uint8_t sc_lo, m_lo;
    if (sub_block < 4) {
        sc_lo = blk.scales[sub_block] & 0x3F;
        m_lo  = blk.scales[sub_block + 4] & 0x3F;
    } else {
        sc_lo = (blk.scales[sub_block + 4] & 0xF) | ((blk.scales[sub_block - 4] >> 6) << 4);
        m_lo  = (blk.scales[sub_block + 4] >> 4) | ((blk.scales[sub_block] >> 6) << 4);
    }
    float scale = d * sc_lo;
    float min = dmin * m_lo;
    int group = tid / 64;
    int within = tid % 64;
    int is_high = within / 32;
    int pos = within % 32;
    int qs_byte_idx = group * 32 + pos;
    int q_lo = is_high ? (blk.qs[qs_byte_idx] >> 4) : (blk.qs[qs_byte_idx] & 0xF);
    int qh_bit = group * 2 + is_high;
    int q_hi = (blk.qh[pos] >> qh_bit) & 1;
    int q_val = q_lo | (q_hi << 4);
    dst[block_idx * 256 + tid] = scale * q_val - min;
}

__global__ void __launch_bounds__(256)
kernel_dequant_q6_k_f32(const block_q6_k* __restrict__ src, float* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;
    const auto& blk = src[block_idx];
    int tid = threadIdx.x;
    float d = __half2float(blk.d);
    int half_idx = tid / 128;
    int j = tid % 128;
    int quarter = j / 32;
    int pos = j % 32;
    int ql_byte = half_idx * 64 + (quarter & 1) * 32 + pos;
    int ql_nibble = (quarter >= 2) ? (blk.ql[ql_byte] >> 4) : (blk.ql[ql_byte] & 0xF);
    int qh_byte = half_idx * 32 + pos;
    int qh_shift = quarter * 2;
    int qh_bits = (blk.qh[qh_byte] >> qh_shift) & 0x3;
    int q_val = ql_nibble | (qh_bits << 4);
    int scale_idx = half_idx * 8 + quarter * 2 + pos / 16;
    int8_t scale = blk.scales[scale_idx];
    dst[block_idx * 256 + tid] = d * scale * (q_val - 32);
}

void gwen_dequant_to_f32(const void* src, float* dst, int n, GGMLType type, cudaStream_t stream) {
    switch (type) {
        case GGMLType::Q4_K: {
            int n_blocks = n / 256;
            kernel_dequant_q4_k_f32<<<n_blocks, 256, 0, stream>>>(
                static_cast<const block_q4_k*>(src), dst, n_blocks);
            break;
        }
        case GGMLType::Q5_K: {
            int n_blocks = n / 256;
            kernel_dequant_q5_k_f32<<<n_blocks, 256, 0, stream>>>(
                static_cast<const block_q5_k*>(src), dst, n_blocks);
            break;
        }
        case GGMLType::Q6_K: {
            int n_blocks = n / 256;
            kernel_dequant_q6_k_f32<<<n_blocks, 256, 0, stream>>>(
                static_cast<const block_q6_k*>(src), dst, n_blocks);
            break;
        }
        case GGMLType::Q8_0: {
            int n_blocks = n / 32;
            int threads = 256;
            int blocks = (n_blocks * 32 + threads - 1) / threads;
            kernel_dequant_q8_0_f32<<<blocks, threads, 0, stream>>>(
                static_cast<const block_q8_0*>(src), dst, n_blocks);
            break;
        }
        case GGMLType::F32: {
            // Already F32, just copy
            GWEN_CHECK_CUDA(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            break;
        }
        default:
            GWEN_CHECK(false, "Unsupported dequant type for F32 output");
    }
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_dequant(const void* src, half* dst, int n, GGMLType type, cudaStream_t stream) {
    switch (type) {
        case GGMLType::Q4_K: gwen_dequant_q4_k(src, dst, n, stream); break;
        case GGMLType::Q5_K: gwen_dequant_q5_k(src, dst, n, stream); break;
        case GGMLType::Q6_K: gwen_dequant_q6_k(src, dst, n, stream); break;
        case GGMLType::Q8_0: gwen_dequant_q8_0(src, dst, n, stream); break;
        case GGMLType::F32:  gwen_dequant_f32(static_cast<const float*>(src), dst, n, stream); break;
        default:
            GWEN_CHECK(false, "Unsupported dequant type");
    }
}

} // namespace gwen
