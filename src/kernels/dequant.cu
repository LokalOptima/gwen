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

    // Extract 4-bit quantized value
    // qs has 128 bytes = 256 nibbles
    // First 128 elements in low nibbles of qs[0..63], next 128 in high nibbles
    uint8_t q_byte;
    int q_val;
    if (tid < 128) {
        q_byte = blk.qs[tid / 2];
        q_val = (tid % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
    } else {
        q_byte = blk.qs[(tid - 128) / 2 + 64];
        q_val = ((tid - 128) % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
    }

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

    // Extract 4 low bits (same layout as Q4_K)
    uint8_t q_byte;
    int q_lo;
    if (tid < 128) {
        q_byte = blk.qs[tid / 2];
        q_lo = (tid % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
    } else {
        q_byte = blk.qs[(tid - 128) / 2 + 64];
        q_lo = ((tid - 128) % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
    }

    // Extract 5th bit from qh
    // qh has 32 bytes = 256 bits, one bit per element
    int qh_byte_idx = tid / 8;
    int qh_bit_idx = tid % 8;
    int q_hi = (blk.qh[qh_byte_idx] >> qh_bit_idx) & 1;

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

    // Q6_K has 16 sub-groups of 16 elements
    int sub_group = tid / 16;   // 0..15
    (void)(tid % 16);

    int8_t scale = blk.scales[sub_group];

    // Extract 6-bit quantized value
    // Lower 4 bits from ql[128]: 2 nibbles per byte
    int ql_idx = tid / 2;
    int ql_nibble;
    if (tid % 2 == 0) {
        ql_nibble = blk.ql[ql_idx] & 0xF;
    } else {
        ql_nibble = blk.ql[ql_idx] >> 4;
    }

    // Upper 2 bits from qh[64]: 4 values per byte (2 bits each)
    int qh_idx = tid / 4;
    int qh_shift = (tid % 4) * 2;
    int qh_bits = (blk.qh[qh_idx] >> qh_shift) & 0x3;

    int q_val = ql_nibble | (qh_bits << 4);
    // Q6_K values are signed: subtract 32 to center around 0
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
