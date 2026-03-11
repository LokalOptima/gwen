#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

namespace gwen {

// ============================================================
// Q8_1 Quantization kernel — quantize FP16 input for dp4a GEMV
// ============================================================
// Each warp (32 threads) quantizes one Q8_1 block (32 elements).
// Multiple warps per thread block for efficiency.

__global__ void __launch_bounds__(256)
kernel_quantize_q8_1(const half* __restrict__ x,
                     block_q8_1* __restrict__ y,
                     int n_blocks) {
    // Each warp handles one Q8_1 block
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_global >= n_blocks) return;

    float val = __half2float(x[warp_global * 32 + lane]);

    // Find max absolute value (warp reduction)
    float amax = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t q = (int8_t)roundf(val * id);

    // Sum of quantized values (needed for Q4_K/Q5_K min value computation)
    float sum = (float)q;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    y[warp_global].qs[lane] = q;
    if (lane == 0)
        y[warp_global].ds = __halves2half2(__float2half(d), __float2half(sum));
}

void gwen_quantize_q8_1(const half* x, void* x_q8, int n, cudaStream_t stream) {
    int n_blocks = n / 32;
    // 256 threads = 8 warps per block, each warp handles one Q8_1 block
    int grid = (n_blocks + 7) / 8;
    kernel_quantize_q8_1<<<grid, 256, 0, stream>>>(
        x, static_cast<block_q8_1*>(x_q8), n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// dp4a-accelerated Q4_K GEMV
// ============================================================
// Following llama.cpp's approach: 128 threads (4 warps), 1 row per block.
// 16 threads per Q4_K block, each thread uses dp4a for 4-element SIMD dot products.
// Input x is pre-quantized to Q8_1 format.

static constexpr int NWARPS = 4;
static constexpr int QK_K = 256;  // elements per quantization super-block

__global__ void __launch_bounds__(NWARPS * 32)
kernel_gemv_q4_k_dp4a(const block_q4_k* __restrict__ W,
                       const block_q8_1* __restrict__ x_q8,
                       half* __restrict__ y,
                       int out_features, int blocks_per_row) {
    const int row = blockIdx.x;
    if (row >= out_features) return;

    const int tid = threadIdx.x + threadIdx.y * 32;  // 0..127

    // Q4_K: QI=32 int positions per block, VDR=2 positions per thread
    // 16 threads per Q4_K block, 8 Q4_K blocks per iteration
    constexpr int QI = 32;
    constexpr int VDR = 2;
    constexpr int BLOCKS_PER_ITER = VDR * NWARPS * 32 / QI;  // = 8

    float sumf = 0.0f;

    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int iqs = (tid * VDR) % QI;  // 0, 2, 4, ..., 30

        const block_q4_k& blk = W[row * blocks_per_row + kbx];

        // Which pair of Q8_1 sub-blocks (each Q4_K block = 8 Q8_1 blocks)
        // bq8_offset: 0, 2, 4, or 6
        const int bq8_offset = 2 * ((iqs / 2) / 4);

        // Vectorized load: 2 x int32 = 8 bytes = 16 nibbles from Q4_K qs
        const int* q4 = reinterpret_cast<const int*>(blk.qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        int v0 = q4[0];   // 8 nibbles from first 16-byte half
        int v1 = q4[4];   // 8 nibbles from second 16-byte half

        // Extract scales and mins for the two sub-blocks
        float d = __half2float(blk.d);
        float dmin = __half2float(blk.dmin);

        float sumf_d = 0.0f;
        float sumf_m = 0.0f;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            // Extract low (i=0) or high (i=1) nibbles
            int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;

            // Get Q8_1 block for this sub-block
            const block_q8_1& bq8 = x_q8[kbx * (QK_K / 32) + bq8_offset + i];
            const int* u = reinterpret_cast<const int*>(bq8.qs) + ((iqs / 2) % 4);
            float d8 = __low2float(bq8.ds);

            // dp4a: 4 x (int8 * int8) dot products per instruction
            int dot1 = __dp4a(v1i, u[4], __dp4a(v0i, u[0], 0));
            // Sum trick: dot of {1,1,1,1} with Q8_1 values for min subtraction
            int dot2 = __dp4a(0x01010101, u[4], __dp4a(0x01010101, u[0], 0));

            // Get scale and min for sub-block (bq8_offset + i)
            int sb = bq8_offset + i;
            int sc, m;
            if (sb < 4) {
                sc = blk.scales[sb] & 0x3F;
                m  = blk.scales[sb + 4] & 0x3F;
            } else {
                sc = (blk.scales[sb + 4] & 0xF) | ((blk.scales[sb - 4] >> 6) << 4);
                m  = (blk.scales[sb + 4] >> 4) | ((blk.scales[sb] >> 6) << 4);
            }

            sumf_d += d8 * (dot1 * sc);
            sumf_m += d8 * (dot2 * m);
        }

        sumf += d * sumf_d - dmin * sumf_m;
    }

    // Reduction: warps 1-3 write to shared memory, warp 0 accumulates and reduces
    __shared__ float tmp_shared[NWARPS - 1][32];

    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = sumf;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; w++)
            sumf += tmp_shared[w][threadIdx.x];

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sumf += __shfl_xor_sync(0xFFFFFFFF, sumf, offset);

        if (threadIdx.x == 0)
            y[row] = __float2half(sumf);
    }
}

// ============================================================
// dp4a-accelerated Q5_K GEMV
// ============================================================

__global__ void __launch_bounds__(NWARPS * 32)
kernel_gemv_q5_k_dp4a(const block_q5_k* __restrict__ W,
                       const block_q8_1* __restrict__ x_q8,
                       half* __restrict__ y,
                       int out_features, int blocks_per_row) {
    const int row = blockIdx.x;
    if (row >= out_features) return;

    const int tid = threadIdx.x + threadIdx.y * 32;

    constexpr int QI = 32;
    constexpr int VDR = 2;
    constexpr int BLOCKS_PER_ITER = VDR * NWARPS * 32 / QI;

    float sumf = 0.0f;

    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int iqs = (tid * VDR) % QI;
        const block_q5_k& blk = W[row * blocks_per_row + kbx];
        const int bq8_offset = 2 * ((iqs / 2) / 4);

        // Load low 4 bits (same layout as Q4_K)
        const int* ql = reinterpret_cast<const int*>(blk.qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        int vl0 = ql[0];
        int vl1 = ql[4];

        // Load high bits from qh
        const int* qh_ptr = reinterpret_cast<const int*>(blk.qh + 4 * ((iqs / 2) % 4));
        int vh0 = qh_ptr[0] >> bq8_offset;
        int vh1 = qh_ptr[4] >> bq8_offset;

        float d = __half2float(blk.d);
        float dmin = __half2float(blk.dmin);

        float sumf_d = 0.0f;
        float sumf_m = 0.0f;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            // Low 4 bits
            int vl0i = (vl0 >> (4 * i)) & 0x0F0F0F0F;
            int vl1i = (vl1 >> (4 * i)) & 0x0F0F0F0F;
            // High bit → bit 4
            int vh0i = ((vh0 >> i) << 4) & 0x10101010;
            int vh1i = ((vh1 >> i) << 4) & 0x10101010;
            // Merge: 5-bit value
            int v0i = vl0i | vh0i;
            int v1i = vl1i | vh1i;

            const block_q8_1& bq8 = x_q8[kbx * (QK_K / 32) + bq8_offset + i];
            const int* u = reinterpret_cast<const int*>(bq8.qs) + ((iqs / 2) % 4);
            float d8 = __low2float(bq8.ds);

            int dot1 = __dp4a(v0i, u[0], __dp4a(v1i, u[4], 0));
            int dot2 = __dp4a(0x01010101, u[0], __dp4a(0x01010101, u[4], 0));

            int sb = bq8_offset + i;
            int sc, m;
            if (sb < 4) {
                sc = blk.scales[sb] & 0x3F;
                m  = blk.scales[sb + 4] & 0x3F;
            } else {
                sc = (blk.scales[sb + 4] & 0xF) | ((blk.scales[sb - 4] >> 6) << 4);
                m  = (blk.scales[sb + 4] >> 4) | ((blk.scales[sb] >> 6) << 4);
            }

            sumf_d += d8 * (dot1 * sc);
            sumf_m += d8 * (dot2 * m);
        }

        sumf += d * sumf_d - dmin * sumf_m;
    }

    __shared__ float tmp_shared[NWARPS - 1][32];
    if (threadIdx.y > 0)
        tmp_shared[threadIdx.y - 1][threadIdx.x] = sumf;
    __syncthreads();

    if (threadIdx.y == 0) {
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; w++)
            sumf += tmp_shared[w][threadIdx.x];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sumf += __shfl_xor_sync(0xFFFFFFFF, sumf, offset);
        if (threadIdx.x == 0)
            y[row] = __float2half(sumf);
    }
}

// ============================================================
// dp4a-accelerated Q6_K GEMV
// ============================================================
// Q6_K: QI=32, VDR=1, QR=2, 32 threads per Q6_K block
// Uses __vsubss4 for center-shift (val - 32)

// Helper: load 4 bytes from 2-byte-aligned address (for ql/qh arrays)
static __device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    return x16[2 * i32] | (x16[2 * i32 + 1] << 16);
}

__global__ void __launch_bounds__(NWARPS * 32)
kernel_gemv_q6_k_dp4a(const block_q6_k* __restrict__ W,
                       const block_q8_1* __restrict__ x_q8,
                       half* __restrict__ y,
                       int out_features, int blocks_per_row) {
    const int row = blockIdx.x;
    if (row >= out_features) return;

    const int tid = threadIdx.x + threadIdx.y * 32;

    // Q6_K: QI=32 int positions, VDR=1, QR=2
    // 32 threads per Q6_K block, 4 Q6_K blocks per iteration
    constexpr int QI = 32;
    constexpr int VDR = 1;
    constexpr int QI8_1 = 8;  // int32s per Q8_1 block
    constexpr int BLOCKS_PER_ITER = VDR * NWARPS * 32 / QI;  // = 4

    float sumf = 0.0f;

    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int iqs = (tid * VDR) % QI;  // 0..31
        const block_q6_k& blk = W[row * blocks_per_row + kbx];

        // Which pair of Q8_1 blocks (8 total per Q6_K block)
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        // Which scale entry (16 total)
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        // Which 2-bit pair in qh bytes
        const int vh_shift = 2 * ((iqs % 16) / 8);

        // Load low 4 bits: 4 bytes from ql at byte offset 4*iqs
        const int vl = get_int_b2(blk.ql, iqs);
        // Load high 2 bits: 4 bytes from qh, shifted
        const int qh_idx = 8 * (iqs / 16) + iqs % 8;
        const int vh = get_int_b2(blk.qh, qh_idx) >> vh_shift;

        const int8_t* scales = blk.scales + scale_offset;
        float d = __half2float(blk.d);
        float local_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int8_t sc = scales[4 * i];

            const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = __vsubss4(vil | vih, 0x20202020);

            const block_q8_1& bq8 = x_q8[kbx * 8 + bq8_offset + 2 * i];
            const int u = reinterpret_cast<const int*>(bq8.qs)[iqs % QI8_1];
            const float d8 = __low2float(bq8.ds);

            local_sum += d8 * (__dp4a(vi, u, 0) * sc);
        }

        sumf += d * local_sum;
    }

    // Reduction
    __shared__ float tmp_shared[NWARPS - 1][32];
    if (threadIdx.y > 0)
        tmp_shared[threadIdx.y - 1][threadIdx.x] = sumf;
    __syncthreads();

    if (threadIdx.y == 0) {
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; w++)
            sumf += tmp_shared[w][threadIdx.x];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sumf += __shfl_xor_sync(0xFFFFFFFF, sumf, offset);
        if (threadIdx.x == 0)
            y[row] = __float2half(sumf);
    }
}

// ============================================================
// Legacy FP16 GEMV kernels (kept for fallback and Q8_0)
// ============================================================

__global__ void __launch_bounds__(256)
kernel_gemv_q4_k(const block_q4_k* __restrict__ W,
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

        int group = tid / 64;
        int within = tid % 64;
        int is_high = within / 32;
        int pos = within % 32;
        int qs_byte_idx = group * 32 + pos;
        int q_val = is_high ? (blk.qs[qs_byte_idx] >> 4) : (blk.qs[qs_byte_idx] & 0xF);

        float w_val = scale * q_val - min_val;
        float x_val = __half2float(x[blk_idx * 256 + tid]);
        acc += w_val * x_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1)
            acc += __shfl_xor_sync(0xFF, acc, offset);
        if (lane == 0) y[row] = __float2half(acc);
    }
}

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

        int group = tid / 64;
        int within = tid % 64;
        int is_high = within / 32;
        int pos = within % 32;
        int qs_byte_idx = group * 32 + pos;
        int q_lo = is_high ? (blk.qs[qs_byte_idx] >> 4) : (blk.qs[qs_byte_idx] & 0xF);
        int qh_bit = group * 2 + is_high;
        int q_hi = (blk.qh[pos] >> qh_bit) & 1;
        int q_val = q_lo | (q_hi << 4);

        float w_val = scale * q_val - min_val;
        float x_val = __half2float(x[blk_idx * 256 + tid]);
        acc += w_val * x_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1)
            acc += __shfl_xor_sync(0xFF, acc, offset);
        if (lane == 0) y[row] = __float2half(acc);
    }
}

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
        float w_val = d * scale * (q_val - 32);
        float x_val = __half2float(x[blk_idx * 256 + tid]);
        acc += w_val * x_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1)
            acc += __shfl_xor_sync(0xFF, acc, offset);
        if (lane == 0) y[row] = __float2half(acc);
    }
}

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

    for (int blk_idx = tid; blk_idx < blocks_per_row; blk_idx += blockDim.x) {
        const auto& blk = W[row * blocks_per_row + blk_idx];
        float d = __half2float(blk.d);
        float local_sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            float w_val = d * blk.qs[j];
            float x_val = __half2float(x[blk_idx * 32 + j]);
            local_sum += w_val * x_val;
        }
        acc += local_sum;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane < 8) {
        acc = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1)
            acc += __shfl_xor_sync(0xFF, acc, offset);
        if (lane == 0) y[row] = __float2half(acc);
    }
}

// ============================================================
// Launch wrappers — legacy FP16 input
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

// ============================================================
// Launch wrappers — dp4a with Q8_1 input
// ============================================================

void gwen_gemv_dp4a(const void* W, const void* x_q8, half* y,
                    int out_features, int in_features, GGMLType type, cudaStream_t stream) {
    int blocks_per_row = in_features / QK_K;
    dim3 block(32, NWARPS);  // 128 threads: 32 lanes × 4 warps

    switch (type) {
        case GGMLType::Q4_K:
            kernel_gemv_q4_k_dp4a<<<out_features, block, 0, stream>>>(
                static_cast<const block_q4_k*>(W),
                static_cast<const block_q8_1*>(x_q8),
                y, out_features, blocks_per_row);
            break;
        case GGMLType::Q5_K:
            kernel_gemv_q5_k_dp4a<<<out_features, block, 0, stream>>>(
                static_cast<const block_q5_k*>(W),
                static_cast<const block_q8_1*>(x_q8),
                y, out_features, blocks_per_row);
            break;
        case GGMLType::Q6_K:
            kernel_gemv_q6_k_dp4a<<<out_features, block, 0, stream>>>(
                static_cast<const block_q6_k*>(W),
                static_cast<const block_q8_1*>(x_q8),
                y, out_features, blocks_per_row);
            break;
        default:
            GWEN_CHECK(false, "Unsupported dp4a GEMV type (Q4_K/Q5_K/Q6_K only)");
    }
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
