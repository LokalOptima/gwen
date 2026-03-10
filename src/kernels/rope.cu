#include "gwen/kernels.h"

namespace gwen {

// imRoPE: interleaved multi-axis Rotary Position Encoding for Qwen3.5
//
// Only the first rope_dim (64) of head_dim (256) get rotary encoding.
// The 64 dims are split into sections: [11, 11, 10, 0] pairs = 32 pairs.
// Each pair (2 dims) uses cos/sin rotation.
// Sections 0,1,2 correspond to 3 axes (for text, all use same position).
//
// Interleaved layout: dims are rotated as adjacent pairs (0,1), (2,3), etc.
//
__global__ void __launch_bounds__(256)
kernel_rope(half* __restrict__ q, half* __restrict__ k,
            int n_heads, int n_kv_heads, int head_dim,
            int pos, float theta,
            int s0, int s1, int s2, int s3, int rope_dim) {
    int head = blockIdx.x;
    int is_k = (head >= n_heads);  // first n_heads blocks do Q, rest do K
    int actual_head;
    half* vec;

    if (!is_k) {
        actual_head = head;
        vec = q + actual_head * head_dim;
    } else {
        actual_head = head - n_heads;
        if (actual_head >= n_kv_heads) return;
        vec = k + actual_head * head_dim;
    }

    int tid = threadIdx.x;
    // Each thread handles one pair of dimensions
    int n_pairs = rope_dim / 2;
    if (tid >= n_pairs) return;

    int pair_idx = tid;  // which pair (0..31 for rope_dim=64)

    // Determine which axis this pair belongs to based on sections
    // Pairs 0..s0-1 → axis 0, s0..s0+s1-1 → axis 1, s0+s1..s0+s1+s2-1 → axis 2
    int axis_pos = pos;  // for text, all axes use same position

    // Compute frequency for this pair
    // freq = pos * theta^(-2*pair_idx / rope_dim)
    // But the pair_idx here is the global pair index within the section
    float freq_exp = -2.0f * pair_idx / (float)rope_dim;
    float freq = (float)axis_pos * powf(theta, freq_exp);

    float cos_val = cosf(freq);
    float sin_val = sinf(freq);

    // Interleaved: pair (2*tid, 2*tid+1)
    int d0 = 2 * tid;
    int d1 = 2 * tid + 1;

    float x0 = __half2float(vec[d0]);
    float x1 = __half2float(vec[d1]);

    // Apply rotation
    vec[d0] = __float2half(x0 * cos_val - x1 * sin_val);
    vec[d1] = __float2half(x0 * sin_val + x1 * cos_val);
}

void gwen_rope(half* q, half* k,
               int n_heads, int n_kv_heads, int head_dim,
               int pos, float theta,
               const int* sections, int rope_dim,
               cudaStream_t stream) {
    int n_pairs = rope_dim / 2;
    // Launch one block per head (Q heads + KV heads)
    int total_blocks = n_heads + n_kv_heads;
    // Thread count = max(n_pairs, 32) to cover all pairs
    int threads = ((n_pairs + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;

    kernel_rope<<<total_blocks, threads, 0, stream>>>(
        q, k, n_heads, n_kv_heads, head_dim, pos, theta,
        sections[0], sections[1], sections[2], sections[3], rope_dim);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// Embedding lookup
// ============================================================

// Simple embedding lookup — dequantize one row from the embedding table
__global__ void __launch_bounds__(256)
kernel_embed_lookup_q6k(const void* __restrict__ table, int token_id,
                        half* __restrict__ y, int dim) {
    // Q6_K: 256 elements per block, dim must be multiple of 256
    // For dim=1024, we have 4 blocks per row
    int blocks_per_row = dim / 256;
    int tid = threadIdx.x;  // 0..255

    // Process one block at a time
    for (int blk_local = 0; blk_local < blocks_per_row; blk_local++) {
        int blk_idx = token_id * blocks_per_row + blk_local;
        const uint8_t* base = static_cast<const uint8_t*>(table) + (size_t)blk_idx * 210;

        // Q6_K block layout: ql[128] + qh[64] + scales[16] + d(half)
        const uint8_t* ql = base;
        const uint8_t* qh = base + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(base + 192);
        half d_half;
        memcpy(&d_half, base + 208, sizeof(half));
        float d = __half2float(d_half);

        int sub_group = tid / 16;
        int8_t scale = scales[sub_group];

        int ql_idx = tid / 2;
        int ql_nibble;
        if (tid % 2 == 0) {
            ql_nibble = ql[ql_idx] & 0xF;
        } else {
            ql_nibble = ql[ql_idx] >> 4;
        }

        int qh_idx = tid / 4;
        int qh_shift = (tid % 4) * 2;
        int qh_bits = (qh[qh_idx] >> qh_shift) & 0x3;

        int q_val = ql_nibble | (qh_bits << 4);
        float result = d * scale * (q_val - 32);
        y[blk_local * 256 + tid] = __float2half(result);
    }
}

void gwen_embed_lookup(const void* table, GGMLType table_type,
                       int token_id, half* y, int dim,
                       cudaStream_t stream) {
    if (table_type == GGMLType::Q6_K) {
        kernel_embed_lookup_q6k<<<1, 256, 0, stream>>>(table, token_id, y, dim);
    } else {
        GWEN_CHECK(false, "Unsupported embedding type");
    }
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// L2 Normalization
// ============================================================
__global__ void __launch_bounds__(32)
kernel_l2_normalize(const half* __restrict__ x, half* __restrict__ y,
                    int n_vecs, int dim) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n_vecs) return;
    int lane = threadIdx.x;

    const half* xv = x + vec_idx * dim;
    half* yv = y + vec_idx * dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float val = __half2float(xv[i]);
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    float inv_norm = rsqrtf(sum_sq + 1e-12f);

    for (int i = lane; i < dim; i += 32) {
        yv[i] = __float2half(__half2float(xv[i]) * inv_norm);
    }
}

void gwen_l2_normalize(const half* x, half* y, int n_vecs, int dim,
                       cudaStream_t stream) {
    kernel_l2_normalize<<<n_vecs, 32, 0, stream>>>(x, y, n_vecs, dim);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
