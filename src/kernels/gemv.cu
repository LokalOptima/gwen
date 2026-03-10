#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

namespace gwen {

// ============================================================
// Fused Q4_K GEMV: y[row] = dot(W[row,:], x[:])
// ============================================================
// Each thread block handles one output row.
// The row has in_features elements stored as Q4_K blocks (256 elements each).
// Threads within the block cooperate to compute the dot product.
// Input vector x is loaded into shared memory.

__global__ void __launch_bounds__(256)
kernel_gemv_q4_k(const block_q4_k* __restrict__ W,
                 const half* __restrict__ x,
                 half* __restrict__ y,
                 int out_features, int in_features) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    int blocks_per_row = in_features / 256;
    int tid = threadIdx.x;  // 0..255

    // Each thread accumulates partial dot product
    float acc = 0.0f;

    // Process Q4_K blocks along the row
    // One block = 256 input elements, one thread per element within a block
    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        const auto& blk = W[row * blocks_per_row + blk_idx];

        float d = __half2float(blk.d);
        float dmin = __half2float(blk.dmin);

        int sub_block = tid / 32;

        // Reconstruct scale and min
        uint8_t sc_lo, m_lo;
        if (sub_block < 4) {
            sc_lo = blk.scales[sub_block] & 0x3F;
            m_lo  = blk.scales[sub_block + 4] & 0x3F;
        } else {
            sc_lo = (blk.scales[sub_block + 4] & 0xF) | ((blk.scales[sub_block - 4] >> 6) << 4);
            m_lo  = (blk.scales[sub_block + 4] >> 4) | ((blk.scales[sub_block] >> 6) << 4);
        }

        float scale = d * sc_lo;
        float min_val = dmin * m_lo;

        // Extract 4-bit quant
        uint8_t q_byte;
        int q_val;
        if (tid < 128) {
            q_byte = blk.qs[tid / 2];
            q_val = (tid % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
        } else {
            q_byte = blk.qs[(tid - 128) / 2 + 64];
            q_val = ((tid - 128) % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
        }

        float w_val = scale * q_val - min_val;
        float x_val = __half2float(x[blk_idx * 256 + tid]);
        acc += w_val * x_val;
    }

    // Warp reduction within each warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    // Now lane 0 of each warp has the partial sum for its warp
    // We have 8 warps (256 threads / 32), need to reduce across warps
    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        // Reduce 8 values
        for (int offset = 4; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xFF, acc, offset);
        }
        if (lane == 0) {
            y[row] = __float2half(acc);
        }
    }
}

// ============================================================
// Fused Q5_K GEMV
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_q5_k(const block_q5_k* __restrict__ W,
                 const half* __restrict__ x,
                 half* __restrict__ y,
                 int out_features, int in_features) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    int blocks_per_row = in_features / 256;
    int tid = threadIdx.x;

    float acc = 0.0f;

    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        const auto& blk = W[row * blocks_per_row + blk_idx];

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
        float min_val = dmin * m_lo;

        // 4 low bits
        uint8_t q_byte;
        int q_lo;
        if (tid < 128) {
            q_byte = blk.qs[tid / 2];
            q_lo = (tid % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
        } else {
            q_byte = blk.qs[(tid - 128) / 2 + 64];
            q_lo = ((tid - 128) % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
        }

        // 5th bit
        int qh_byte_idx = tid / 8;
        int qh_bit_idx = tid % 8;
        int q_hi = (blk.qh[qh_byte_idx] >> qh_bit_idx) & 1;

        int q_val = q_lo | (q_hi << 4);
        float w_val = scale * q_val - min_val;
        float x_val = __half2float(x[blk_idx * 256 + tid]);
        acc += w_val * x_val;
    }

    // Warp + cross-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xFF, acc, offset);
        }
        if (lane == 0) y[row] = __float2half(acc);
    }
}

// ============================================================
// Fused Q6_K GEMV
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_q6_k(const block_q6_k* __restrict__ W,
                 const half* __restrict__ x,
                 half* __restrict__ y,
                 int out_features, int in_features) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    int blocks_per_row = in_features / 256;
    int tid = threadIdx.x;

    float acc = 0.0f;

    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        const auto& blk = W[row * blocks_per_row + blk_idx];

        float d = __half2float(blk.d);
        int sub_group = tid / 16;
        int8_t scale = blk.scales[sub_group];

        // Lower 4 bits
        int ql_idx = tid / 2;
        int ql_nibble;
        if (tid % 2 == 0) {
            ql_nibble = blk.ql[ql_idx] & 0xF;
        } else {
            ql_nibble = blk.ql[ql_idx] >> 4;
        }

        // Upper 2 bits
        int qh_idx = tid / 4;
        int qh_shift = (tid % 4) * 2;
        int qh_bits = (blk.qh[qh_idx] >> qh_shift) & 0x3;

        int q_val = ql_nibble | (qh_bits << 4);
        float w_val = d * scale * (q_val - 32);
        float x_val = __half2float(x[blk_idx * 256 + tid]);
        acc += w_val * x_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xFF, acc, offset);
        }
        if (lane == 0) y[row] = __float2half(acc);
    }
}

// ============================================================
// Fused Q8_0 GEMV
// ============================================================
__global__ void __launch_bounds__(256)
kernel_gemv_q8_0(const block_q8_0* __restrict__ W,
                 const half* __restrict__ x,
                 half* __restrict__ y,
                 int out_features, int in_features) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    int blocks_per_row = in_features / 32;
    int tid = threadIdx.x;

    float acc = 0.0f;

    // Each thread processes multiple Q8_0 blocks
    for (int blk_idx = tid; blk_idx < blocks_per_row; blk_idx += blockDim.x) {
        const auto& blk = W[row * blocks_per_row + blk_idx];
        float d = __half2float(blk.d);

        // Process all 32 elements in this block
        float local_sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            float w_val = d * blk.qs[j];
            float x_val = __half2float(x[blk_idx * 32 + j]);
            local_sum += w_val * x_val;
        }
        acc += local_sum;
    }

    // Warp + cross-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xFF, acc, offset);
        }
        if (lane == 0) y[row] = __float2half(acc);
    }
}

// ============================================================
// Launch wrappers
// ============================================================

void gwen_gemv_q4_k(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_q4_k<<<out_features, 256, 0, stream>>>(
        static_cast<const block_q4_k*>(W), x, y, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_q5_k(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_q5_k<<<out_features, 256, 0, stream>>>(
        static_cast<const block_q5_k*>(W), x, y, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_q6_k(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_q6_k<<<out_features, 256, 0, stream>>>(
        static_cast<const block_q6_k*>(W), x, y, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_q8_0(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream) {
    kernel_gemv_q8_0<<<out_features, 256, 0, stream>>>(
        static_cast<const block_q8_0*>(W), x, y, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv(const void* W, const half* x, half* y,
               int out_features, int in_features, GGMLType type, cudaStream_t stream) {
    switch (type) {
        case GGMLType::Q4_K: gwen_gemv_q4_k(W, x, y, out_features, in_features, stream); break;
        case GGMLType::Q5_K: gwen_gemv_q5_k(W, x, y, out_features, in_features, stream); break;
        case GGMLType::Q6_K: gwen_gemv_q6_k(W, x, y, out_features, in_features, stream); break;
        case GGMLType::Q8_0: gwen_gemv_q8_0(W, x, y, out_features, in_features, stream); break;
        default: GWEN_CHECK(false, "Unsupported GEMV type");
    }
}

} // namespace gwen
