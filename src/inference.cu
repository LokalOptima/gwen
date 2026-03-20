#include "gwen/inference.h"
#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

#include <cfloat>
#include <cstdio>
#include <algorithm>
#include <chrono>

namespace gwen {

// ============================================================
// DeltaNet decode kernel: state update + query (Delta Rule)
// ============================================================
// Per head:
//   1. S = S * exp(gate)              — exponential decay
//   2. sk = S^T @ k                   — project key through state
//   3. d = (v - sk) * beta            — compute delta
//   4. S += outer(k, d)               — rank-1 state update
//   5. o = S^T @ q                    — output
//
// S is [d_k, d_v] = [128, 128], stored in FP32 for numerical stability
// gate = ssm_a * softplus(alpha_proj + dt_bias), already computed
// beta = sigmoid(beta_proj), already computed
// Optimized 2-pass DeltaNet recurrence:
// Pass 1: Fused decay + S^T@k (read S once, write decayed, accumulate sk)
// Pass 2: Fused update + S^T@q (read/write S once, accumulate output)
// Only 1 sync needed (after loading k,q to shared memory)
// Each thread handles one column j through all dk rows — columns are independent.
__global__ void __launch_bounds__(128)
kernel_deltanet_decode(
    float* __restrict__ S,           // [n_heads, d_k, d_v] recurrent state
    const half* __restrict__ q_in,   // [n_heads * d_k] query vectors (L2-normalized)
    const half* __restrict__ k_in,   // [n_heads * d_k] key vectors (L2-normalized)
    const half* __restrict__ v_in,   // [n_heads * d_v] value vectors
    const float* __restrict__ gate,  // [n_heads] gate = ssm_a * softplus(alpha_proj + dt_bias)
    const float* __restrict__ beta_in,  // [n_heads] sigmoid(beta_proj)
    half* __restrict__ output,       // [n_heads * d_v] output
    float* __restrict__ d_out,       // [n_heads * d_v] delta (optional, for activation replay)
    int n_heads, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int j = threadIdx.x;  // 0..127, one thread per column

    float decay = expf(gate[head]);
    float b = beta_in[head];

    float* S_head = S + head * dk * dv;

    // Load k and q into shared memory (128 floats each = 512 bytes)
    __shared__ float sh_k[128];
    __shared__ float sh_q[128];
    sh_k[j] = __half2float(k_in[head * dk + j]);
    sh_q[j] = __half2float(q_in[head * dk + j]);
    __syncthreads();

    // Pass 1: Decay S and compute sk[j] = sum_i (decay * S[i][j]) * k[i]
    float sk_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; i++) {
        float val = S_head[i * 128 + j] * decay;
        S_head[i * 128 + j] = val;
        sk_j += val * sh_k[i];
    }

    // No sync needed — each thread works on its own column j

    // Pass 2: Update S and compute o[j] = sum_i S_updated[i][j] * q[i]
    float d_j = (__half2float(v_in[head * dv + j]) - sk_j) * b;
    float o_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; i++) {
        float updated = S_head[i * 128 + j] + sh_k[i] * d_j;
        S_head[i * 128 + j] = updated;
        o_j += updated * sh_q[i];
    }

    output[head * dv + j] = __float2half(o_j);

    // Save exact delta for activation replay undo (avoids precision loss from algebraic inversion)
    if (d_out) {
        d_out[head * dv + j] = d_j;
    }
}

void gwen_deltanet_decode(float* S, const half* q, const half* k, const half* v,
                          const float* alpha, const float* beta, half* output,
                          int n_heads, int dk, int dv, cudaStream_t stream) {
    kernel_deltanet_decode<<<n_heads, 128, 0, stream>>>(S, q, k, v, alpha, beta, output, nullptr, n_heads, dk, dv);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// Fused Conv1D + SiLU with rolling state
// ============================================================
// Applies 1D convolution with kernel_size=4, then SiLU activation
__global__ void __launch_bounds__(256)
kernel_conv1d_silu(
    half* __restrict__ output,           // [dim] output (may alias input)
    const half* __restrict__ input,      // [dim] current input
    float* __restrict__ conv_state,      // [kernel_size-1, dim] rolling state
    const float* __restrict__ weight,    // [kernel_size, dim] conv weights
    int dim, int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    float x_val = __half2float(input[idx]);

    float acc = 0.0f;
    for (int k = 0; k < kernel_size - 1; k++) {
        acc += weight[idx * kernel_size + k] * conv_state[k * dim + idx];
    }
    acc += weight[idx * kernel_size + (kernel_size - 1)] * x_val;

    // Fused SiLU: x * sigmoid(x)
    float silu = acc / (1.0f + expf(-acc));
    output[idx] = __float2half(silu);

    // Shift state
    for (int k = 0; k < kernel_size - 2; k++) {
        conv_state[k * dim + idx] = conv_state[(k + 1) * dim + idx];
    }
    conv_state[(kernel_size - 2) * dim + idx] = x_val;
}

// ============================================================
// Gated RMSNorm: output = RMSNorm_per_head(x) * SiLU(gate)
// ============================================================
// Each block handles one head. RMSNorm is computed per-head (dim_per_head elements).
__global__ void __launch_bounds__(32)
kernel_gated_rmsnorm(
    const half* __restrict__ x,        // [n_heads * dim_per_head] input
    const float* __restrict__ weight,  // [dim_per_head] norm weight (shared across heads)
    const half* __restrict__ gate,     // [n_heads * dim_per_head] gate (Z)
    half* __restrict__ output,         // [n_heads * dim_per_head] output
    int n_heads, int dim_per_head, float eps)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int lane = threadIdx.x;
    int offset_base = head * dim_per_head;

    // Per-head RMSNorm
    float sum_sq = 0.0f;
    for (int i = lane; i < dim_per_head; i += 32) {
        float val = __half2float(x[offset_base + i]);
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }
    float rms_inv = rsqrtf(sum_sq / dim_per_head + eps);

    // Apply: output = (x * rms_inv * weight) * silu(gate)
    for (int i = lane; i < dim_per_head; i += 32) {
        float x_val = __half2float(x[offset_base + i]) * rms_inv;
        float w = weight[i];
        float g = __half2float(gate[offset_base + i]);
        float silu_g = g / (1.0f + expf(-g));
        output[offset_base + i] = __float2half(x_val * w * silu_g);
    }
}

// ============================================================
// Fused Gated RMSNorm + Q8_1 Quantize
// ============================================================
// 128 threads (4 warps) per head. dim_per_head=128 → 4 Q8_1 blocks per head.
// Phase 1: All 128 threads compute per-head sum-of-squares (1 element each), cross-warp reduce
// Phase 2: Each warp handles one Q8_1 block: apply norm*weight*SiLU(gate), quantize
__global__ void __launch_bounds__(128)
kernel_gated_rmsnorm_quantize_q8_1(
    const half* __restrict__ x,        // [n_heads * dim_per_head]
    const float* __restrict__ weight,  // [dim_per_head] (shared)
    const half* __restrict__ gate,     // [n_heads * dim_per_head]
    block_q8_1* __restrict__ y_q8,     // [n_heads * dim_per_head / 32] Q8_1 blocks
    int n_heads, int dim_per_head, float eps)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;  // 0..127
    int warp_id = tid / 32;
    int lane = tid % 32;
    int offset_base = head * dim_per_head;

    // Phase 1: sum of squares (each thread handles dim_per_head/128 elements)
    float sum_sq = 0.0f;
    for (int i = tid; i < dim_per_head; i += 128) {
        float val = __half2float(x[offset_base + i]);
        sum_sq += val * val;
    }

    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[4];
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    __shared__ float s_rms_inv;
    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        s_rms_inv = rsqrtf(total / dim_per_head + eps);
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Phase 2: Each warp handles one Q8_1 block
    // dim_per_head=128 → 4 blocks, one per warp
    int blocks_per_head = dim_per_head / 32;
    int blk_idx = warp_id;
    if (blk_idx >= blocks_per_head) return;

    int elem_idx = offset_base + blk_idx * 32 + lane;
    int local_idx = blk_idx * 32 + lane;

    float x_val = __half2float(x[elem_idx]) * rms_inv;
    float w = weight[local_idx];
    float g = __half2float(gate[elem_idx]);
    float silu_g = g / (1.0f + expf(-g));
    float val = x_val * w * silu_g;

    // Quantize
    float amax = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t q = (int8_t)roundf(val * id);

    float sum = (float)q;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    int out_blk = head * blocks_per_head + blk_idx;
    y_q8[out_blk].qs[lane] = q;
    if (lane == 0)
        y_q8[out_blk].ds = __halves2half2(__float2half(d), __float2half(sum));
}

// Batch2: process two independent gated RMSNorm + Q8_1 in one launch
// blockIdx.x = head index, blockIdx.y = token (0=A, 1=B)
__global__ void __launch_bounds__(128)
kernel_gated_rmsnorm_quantize_q8_1_batch2(
    const half* __restrict__ x_a, const half* __restrict__ x_b,
    const float* __restrict__ weight,
    const half* __restrict__ gate_a, const half* __restrict__ gate_b,
    block_q8_1* __restrict__ y_q8_a, block_q8_1* __restrict__ y_q8_b,
    int n_heads, int dim_per_head, float eps)
{
    const half* x = (blockIdx.y == 0) ? x_a : x_b;
    const half* gate = (blockIdx.y == 0) ? gate_a : gate_b;
    block_q8_1* y_q8 = (blockIdx.y == 0) ? y_q8_a : y_q8_b;

    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int offset_base = head * dim_per_head;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim_per_head; i += 128) {
        float val = __half2float(x[offset_base + i]);
        sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float warp_sums[4];
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    __shared__ float s_rms_inv;
    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        s_rms_inv = rsqrtf(total / dim_per_head + eps);
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    int blocks_per_head = dim_per_head / 32;
    int blk_idx = warp_id;
    if (blk_idx >= blocks_per_head) return;

    int elem_idx = offset_base + blk_idx * 32 + lane;
    int local_idx = blk_idx * 32 + lane;

    float x_val = __half2float(x[elem_idx]) * rms_inv;
    float w = weight[local_idx];
    float g = __half2float(gate[elem_idx]);
    float silu_g = g / (1.0f + expf(-g));
    float val = x_val * w * silu_g;

    float amax = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t q = (int8_t)roundf(val * id);

    float sum = (float)q;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    int out_blk = head * blocks_per_head + blk_idx;
    y_q8[out_blk].qs[lane] = q;
    if (lane == 0)
        y_q8[out_blk].ds = __halves2half2(__float2half(d), __float2half(sum));
}

// ============================================================
// DeltaNet state update — split into prep + decode for occupancy
// ============================================================

// Prep kernel: L2-normalize Q,K + compute gate/beta → write to global
// 16 blocks × 128 threads (one per head). Fast (<1 μs).
__global__ void __launch_bounds__(128)
kernel_deltanet_prep(
    half* __restrict__ q_in,
    half* __restrict__ k_in,
    const half* __restrict__ x_norm,
    const void* __restrict__ alpha_w,
    const void* __restrict__ beta_w,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ decay_out,
    float* __restrict__ beta_out,
    float q_scale,
    int n_heads, int dk, int n_embed)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    half* q = q_in + head * dk;
    half* k = k_in + head * dk;

    float q_val = __half2float(q[tid]);
    float k_val = __half2float(k[tid]);

    // Q L2 norm
    float q_sq = q_val * q_val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, o);
    __shared__ float warp_buf[4];
    if (lane == 0) warp_buf[warp_id] = q_sq;
    __syncthreads();
    __shared__ float s_q_inv, s_k_inv;
    if (tid == 0) {
        float total = warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3];
        s_q_inv = rsqrtf(fmaxf(total, 1e-12f)) * q_scale;
    }
    __syncthreads();

    // K L2 norm
    float k_sq = k_val * k_val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, o);
    if (lane == 0) warp_buf[warp_id] = k_sq;
    __syncthreads();
    if (tid == 0) {
        float total = warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3];
        s_k_inv = rsqrtf(fmaxf(total, 1e-12f));
    }
    __syncthreads();

    q[tid] = __float2half(q_val * s_q_inv);
    k[tid] = __float2half(k_val * s_k_inv);

    // Gate and beta dot products
    const block_q8_0* ab = static_cast<const block_q8_0*>(alpha_w);
    const block_q8_0* bb = static_cast<const block_q8_0*>(beta_w);
    int bph = n_embed / 32;
    float aa = 0.0f, ba = 0.0f;
    for (int i = tid; i < n_embed; i += 128) {
        int blk = i / 32, elem = i % 32, idx = head * bph + blk;
        float x_val = __half2float(x_norm[i]);
        aa += (__half2float(ab[idx].d) * (float)ab[idx].qs[elem]) * x_val;
        ba += (__half2float(bb[idx].d) * (float)bb[idx].qs[elem]) * x_val;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        aa += __shfl_xor_sync(0xFFFFFFFF, aa, o);
        ba += __shfl_xor_sync(0xFFFFFFFF, ba, o);
    }
    __shared__ float sh_a[4], sh_b[4];
    if (lane == 0) { sh_a[warp_id] = aa; sh_b[warp_id] = ba; }
    __syncthreads();
    if (tid == 0) {
        float ta = sh_a[0]+sh_a[1]+sh_a[2]+sh_a[3];
        float tb = sh_b[0]+sh_b[1]+sh_b[2]+sh_b[3];
        float biased = ta + dt_bias[head];
        float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
        decay_out[head] = expf(ssm_a[head] * sp);
        beta_out[head] = 1.0f / (1.0f + expf(-tb));
    }
}

// Decode kernel: S matrix update, 32 blocks × 64 threads for 2× occupancy.
// blockIdx.x = head * 2 + col_group. Each block handles 64 columns.
__global__ void __launch_bounds__(64, 8)
kernel_deltanet_decode_v2(
    float* __restrict__ S,
    const half* __restrict__ q_in,
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ beta_in,
    half* __restrict__ output,
    int n_heads, int dk)
{
    int head = blockIdx.x / 2;
    int col_group = blockIdx.x % 2;
    if (head >= n_heads) return;
    int tid = threadIdx.x;  // 0..63
    int col = col_group * 64 + tid;

    float decay = decay_in[head];
    float b = beta_in[head];
    float* S_head = S + head * dk * dk;

    // Load all 128 q,k values into shared memory
    __shared__ float sh_k[128], sh_q[128];
    sh_k[col] = __half2float(k_in[head * dk + col]);
    sh_q[col] = __half2float(q_in[head * dk + col]);
    int other = (1 - col_group) * 64 + tid;
    sh_k[other] = __half2float(k_in[head * dk + other]);
    sh_q[other] = __half2float(q_in[head * dk + other]);
    __syncthreads();

    // Pass 1: Decay S and compute sk[col]
    float sk_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; i++) {
        float val = S_head[i * 128 + col] * decay;
        S_head[i * 128 + col] = val;
        sk_j += val * sh_k[i];
    }

    // Pass 2: Update S and compute output[col]
    float d_j = (__half2float(v_in[head * dk + col]) - sk_j) * b;
    float o_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; i++) {
        float updated = S_head[i * 128 + col] + sh_k[i] * d_j;
        S_head[i * 128 + col] = updated;
        o_j += updated * sh_q[i];
    }

    output[head * dk + col] = __float2half(o_j);
}

// Fused kernel with S matrix in shared memory.
// Loads 64 KB S into shared, operates at ~4 cycle latency instead of ~200 (L2).
// 16 blocks × 128 threads, 66 KB shared memory per block.
__global__ void __launch_bounds__(128)
kernel_deltanet_fused(
    float* __restrict__ S,              // [n_heads, dk, dk] recurrent state
    half* __restrict__ q_in,            // [n_heads * dk] Q from conv1d
    half* __restrict__ k_in,            // [n_heads * dk] K from conv1d
    const half* __restrict__ v_in,      // [n_heads * dk] V from conv1d
    const half* __restrict__ x_norm,    // [n_embed] pre-attention norm (for gate/beta)
    const void* __restrict__ alpha_w,   // Q8_0 [n_heads * n_embed/32 blocks]
    const void* __restrict__ beta_w,    // Q8_0 [n_heads * n_embed/32 blocks]
    const float* __restrict__ ssm_a,    // [n_heads]
    const float* __restrict__ dt_bias,  // [n_heads]
    half* __restrict__ output,          // [n_heads * dk]
    float q_scale,                      // 1/sqrt(dk) for Q normalization
    int n_heads, int dk, int n_embed)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;  // 0..127
    int warp_id = tid / 32;
    int lane = tid % 32;
    half* q = q_in + head * dk;
    half* k = k_in + head * dk;

    // Dynamic shared memory: S matrix [128][128] = 64 KB
    extern __shared__ float sh_S[];
    // Static shared for small buffers
    __shared__ float sh_q[128], sh_k[128];
    __shared__ float warp_buf[4];
    __shared__ float s_q_inv, s_k_inv, s_decay, s_beta;

    // ---- Async load S from global → shared (overlapped with Phase 1-2) ----
    float* S_head = S + head * dk * dk;
    // cp.async copies 4 bytes per thread, directly global→shared, non-blocking
    #pragma unroll 4
    for (int i = 0; i < 128 * 128; i += 128) {
        uint32_t smem_addr_val = __cvta_generic_to_shared(&sh_S[i + tid]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                     :: "r"(smem_addr_val), "l"(&S_head[i + tid]));
    }
    asm volatile("cp.async.commit_group;\n");

    // ---- Phase 1: L2-normalize Q and K (overlapped with S load) ----
    float q_val = __half2float(q[tid]);
    float k_val = __half2float(k[tid]);

    float q_sq = q_val * q_val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, o);
    if (lane == 0) warp_buf[warp_id] = q_sq;
    __syncthreads();

    if (tid == 0) {
        float total = warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3];
        s_q_inv = rsqrtf(fmaxf(total, 1e-12f)) * q_scale;
    }
    __syncthreads();

    float k_sq = k_val * k_val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, o);
    if (lane == 0) warp_buf[warp_id] = k_sq;
    __syncthreads();

    if (tid == 0) {
        float total = warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3];
        s_k_inv = rsqrtf(fmaxf(total, 1e-12f));
    }
    __syncthreads();

    sh_q[tid] = q_val * s_q_inv;
    sh_k[tid] = k_val * s_k_inv;

    // ---- Phase 2: Compute gate and beta (still overlapped with S load) ----
    const block_q8_0* alpha_blocks = static_cast<const block_q8_0*>(alpha_w);
    const block_q8_0* beta_blocks = static_cast<const block_q8_0*>(beta_w);
    int blocks_per_head = n_embed / 32;

    float alpha_acc = 0.0f, beta_acc = 0.0f;
    for (int i = tid; i < n_embed; i += 128) {
        int blk = i / 32, elem = i % 32;
        int blk_idx = head * blocks_per_head + blk;
        float ad = __half2float(alpha_blocks[blk_idx].d);
        float bd = __half2float(beta_blocks[blk_idx].d);
        float x_val = __half2float(x_norm[i]);
        alpha_acc += (ad * (float)alpha_blocks[blk_idx].qs[elem]) * x_val;
        beta_acc += (bd * (float)beta_blocks[blk_idx].qs[elem]) * x_val;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        alpha_acc += __shfl_xor_sync(0xFFFFFFFF, alpha_acc, o);
        beta_acc += __shfl_xor_sync(0xFFFFFFFF, beta_acc, o);
    }

    __shared__ float sh_alpha[4], sh_beta_arr[4];
    if (lane == 0) { sh_alpha[warp_id] = alpha_acc; sh_beta_arr[warp_id] = beta_acc; }
    __syncthreads();

    if (tid == 0) {
        float total_alpha = sh_alpha[0] + sh_alpha[1] + sh_alpha[2] + sh_alpha[3];
        float total_beta = sh_beta_arr[0] + sh_beta_arr[1] + sh_beta_arr[2] + sh_beta_arr[3];
        float alpha_biased = total_alpha + dt_bias[head];
        float sp = (alpha_biased > 20.0f) ? alpha_biased : logf(1.0f + expf(alpha_biased));
        s_decay = expf(ssm_a[head] * sp);
        s_beta = 1.0f / (1.0f + expf(-total_beta));
    }

    // ---- Wait for async S load to complete before Phase 3 ----
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // ---- Phase 3: S update in shared memory (~4 cycle access vs ~200 from L2) ----
    float decay = s_decay;
    float b = s_beta;

    // Pass 1: Decay S and compute sk[j]
    float sk_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; i++) {
        float val = sh_S[i * 128 + tid] * decay;
        sh_S[i * 128 + tid] = val;
        sk_j += val * sh_k[i];
    }

    // Pass 2: Update S and compute output
    float d_j = (__half2float(v_in[head * dk + tid]) - sk_j) * b;
    float o_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; i++) {
        float updated = sh_S[i * 128 + tid] + sh_k[i] * d_j;
        sh_S[i * 128 + tid] = updated;
        o_j += updated * sh_q[i];
    }

    output[head * dk + tid] = __float2half(o_j);

    // ---- Write S back to global ----
    __syncthreads();
    for (int i = 0; i < 128 * 128; i += 128) {
        S_head[i + tid] = sh_S[i + tid];
    }
}

// ============================================================
// Fused 2-token DeltaNet decode kernel
// ============================================================
// Processes token A then token B in a single kernel launch.
// S stays in shared memory across both tokens — no global round-trip.
// S snapshot is written from shared → global between A and B, overlapped
// with B's Phase 1-2 (L2-norm + gate/beta which don't touch S).
//
// Saves per DeltaNet layer vs 2 separate kernel_deltanet_fused calls:
//   - 1 kernel launch eliminated (~5 μs)
//   - S reload from global eliminated (64 KB L2 read)
//   - S snapshot overlapped with B's Phase 1-2

__global__ void __launch_bounds__(128)
kernel_deltanet_fused_2tok(
    float* __restrict__ S,              // [n_heads, dk, dk] recurrent state
    // Token A inputs
    half* __restrict__ q_in_a,          // [n_heads * dk]
    half* __restrict__ k_in_a,          // [n_heads * dk]
    const half* __restrict__ v_in_a,    // [n_heads * dk]
    const half* __restrict__ x_norm_a,  // [n_embed]
    // Token B inputs
    half* __restrict__ q_in_b,          // [n_heads * dk]
    half* __restrict__ k_in_b,          // [n_heads * dk]
    const half* __restrict__ v_in_b,    // [n_heads * dk]
    const half* __restrict__ x_norm_b,  // [n_embed]
    // Shared weights
    const void* __restrict__ alpha_w,   // Q8_0 [n_heads * n_embed/32 blocks]
    const void* __restrict__ beta_w,    // Q8_0 [n_heads * n_embed/32 blocks]
    const float* __restrict__ ssm_a,    // [n_heads]
    const float* __restrict__ dt_bias,  // [n_heads]
    // Outputs
    half* __restrict__ output_a,        // [n_heads * dk]
    half* __restrict__ output_b,        // [n_heads * dk]
    // S snapshot buffer (written between A and B)
    float* __restrict__ S_snapshot,     // [n_heads * dk * dk] or nullptr
    float q_scale,
    int n_heads, int dk, int n_embed)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Dynamic shared memory: S matrix [128][128] = 64 KB
    extern __shared__ float sh_S[];
    __shared__ float sh_q[128], sh_k[128];
    __shared__ float warp_buf[4];
    __shared__ float s_q_inv, s_k_inv, s_decay, s_beta;

    float* S_head = S + head * dk * dk;
    const block_q8_0* alpha_blocks = static_cast<const block_q8_0*>(alpha_w);
    const block_q8_0* beta_blocks = static_cast<const block_q8_0*>(beta_w);
    int blocks_per_head = n_embed / 32;

    // ================================================================
    // TOKEN A
    // ================================================================

    // Async load S from global → shared (overlapped with A's Phase 1-2)
    #pragma unroll 4
    for (int i = 0; i < 128 * 128; i += 128) {
        uint32_t smem_addr_val = __cvta_generic_to_shared(&sh_S[i + tid]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                     :: "r"(smem_addr_val), "l"(&S_head[i + tid]));
    }
    asm volatile("cp.async.commit_group;\n");

    // Phase 1: L2-normalize Q_a and K_a
    {
        half* q = q_in_a + head * dk;
        half* k = k_in_a + head * dk;
        float q_val = __half2float(q[tid]);
        float k_val = __half2float(k[tid]);

        float q_sq = q_val * q_val;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, o);
        if (lane == 0) warp_buf[warp_id] = q_sq;
        __syncthreads();
        if (tid == 0) s_q_inv = rsqrtf(fmaxf(warp_buf[0]+warp_buf[1]+warp_buf[2]+warp_buf[3], 1e-12f)) * q_scale;
        __syncthreads();

        float k_sq = k_val * k_val;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, o);
        if (lane == 0) warp_buf[warp_id] = k_sq;
        __syncthreads();
        if (tid == 0) s_k_inv = rsqrtf(fmaxf(warp_buf[0]+warp_buf[1]+warp_buf[2]+warp_buf[3], 1e-12f));
        __syncthreads();

        sh_q[tid] = q_val * s_q_inv;
        sh_k[tid] = k_val * s_k_inv;
    }

    // Phase 2: Gate and beta for token A
    {
        float alpha_acc = 0.0f, beta_acc = 0.0f;
        for (int i = tid; i < n_embed; i += 128) {
            int blk = i / 32, elem = i % 32;
            int blk_idx = head * blocks_per_head + blk;
            float ad = __half2float(alpha_blocks[blk_idx].d);
            float bd = __half2float(beta_blocks[blk_idx].d);
            float x_val = __half2float(x_norm_a[i]);
            alpha_acc += (ad * (float)alpha_blocks[blk_idx].qs[elem]) * x_val;
            beta_acc += (bd * (float)beta_blocks[blk_idx].qs[elem]) * x_val;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            alpha_acc += __shfl_xor_sync(0xFFFFFFFF, alpha_acc, o);
            beta_acc += __shfl_xor_sync(0xFFFFFFFF, beta_acc, o);
        }
        __shared__ float sh_alpha[4], sh_beta_arr[4];
        if (lane == 0) { sh_alpha[warp_id] = alpha_acc; sh_beta_arr[warp_id] = beta_acc; }
        __syncthreads();
        if (tid == 0) {
            float total_alpha = sh_alpha[0]+sh_alpha[1]+sh_alpha[2]+sh_alpha[3];
            float total_beta = sh_beta_arr[0]+sh_beta_arr[1]+sh_beta_arr[2]+sh_beta_arr[3];
            float alpha_biased = total_alpha + dt_bias[head];
            float sp = (alpha_biased > 20.0f) ? alpha_biased : logf(1.0f + expf(alpha_biased));
            s_decay = expf(ssm_a[head] * sp);
            s_beta = 1.0f / (1.0f + expf(-total_beta));
        }
    }

    // Wait for async S load
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // Phase 3A: S update for token A (2 passes over S in shared memory)
    {
        float decay = s_decay;
        float b = s_beta;
        float sk_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float val = sh_S[i * 128 + tid] * decay;
            sh_S[i * 128 + tid] = val;
            sk_j += val * sh_k[i];
        }
        float d_j = (__half2float(v_in_a[head * dk + tid]) - sk_j) * b;
        float o_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float updated = sh_S[i * 128 + tid] + sh_k[i] * d_j;
            sh_S[i * 128 + tid] = updated;
            o_j += updated * sh_q[i];
        }
        output_a[head * dk + tid] = __float2half(o_j);
    }

    // ================================================================
    // S SNAPSHOT (overlapped with B's Phase 1-2)
    // ================================================================
    // Write S from shared → snapshot buffer (global stores, non-blocking)
    __syncthreads();
    if (S_snapshot) {
        float* snap_head = S_snapshot + head * dk * dk;
        for (int i = 0; i < 128 * 128; i += 128)
            snap_head[i + tid] = sh_S[i + tid];
    }

    // ================================================================
    // TOKEN B (S is still in shared memory — no reload needed)
    // ================================================================

    // Phase 1: L2-normalize Q_b and K_b (runs while snapshot writes are in flight)
    {
        half* q = q_in_b + head * dk;
        half* k = k_in_b + head * dk;
        float q_val = __half2float(q[tid]);
        float k_val = __half2float(k[tid]);

        float q_sq = q_val * q_val;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, o);
        if (lane == 0) warp_buf[warp_id] = q_sq;
        __syncthreads();
        if (tid == 0) s_q_inv = rsqrtf(fmaxf(warp_buf[0]+warp_buf[1]+warp_buf[2]+warp_buf[3], 1e-12f)) * q_scale;
        __syncthreads();

        float k_sq = k_val * k_val;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, o);
        if (lane == 0) warp_buf[warp_id] = k_sq;
        __syncthreads();
        if (tid == 0) s_k_inv = rsqrtf(fmaxf(warp_buf[0]+warp_buf[1]+warp_buf[2]+warp_buf[3], 1e-12f));
        __syncthreads();

        sh_q[tid] = q_val * s_q_inv;
        sh_k[tid] = k_val * s_k_inv;
    }

    // Phase 2: Gate and beta for token B
    {
        float alpha_acc = 0.0f, beta_acc = 0.0f;
        for (int i = tid; i < n_embed; i += 128) {
            int blk = i / 32, elem = i % 32;
            int blk_idx = head * blocks_per_head + blk;
            float ad = __half2float(alpha_blocks[blk_idx].d);
            float bd = __half2float(beta_blocks[blk_idx].d);
            float x_val = __half2float(x_norm_b[i]);
            alpha_acc += (ad * (float)alpha_blocks[blk_idx].qs[elem]) * x_val;
            beta_acc += (bd * (float)beta_blocks[blk_idx].qs[elem]) * x_val;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            alpha_acc += __shfl_xor_sync(0xFFFFFFFF, alpha_acc, o);
            beta_acc += __shfl_xor_sync(0xFFFFFFFF, beta_acc, o);
        }
        __shared__ float sh_alpha_b[4], sh_beta_arr_b[4];
        if (lane == 0) { sh_alpha_b[warp_id] = alpha_acc; sh_beta_arr_b[warp_id] = beta_acc; }
        __syncthreads();
        if (tid == 0) {
            float total_alpha = sh_alpha_b[0]+sh_alpha_b[1]+sh_alpha_b[2]+sh_alpha_b[3];
            float total_beta = sh_beta_arr_b[0]+sh_beta_arr_b[1]+sh_beta_arr_b[2]+sh_beta_arr_b[3];
            float alpha_biased = total_alpha + dt_bias[head];
            float sp = (alpha_biased > 20.0f) ? alpha_biased : logf(1.0f + expf(alpha_biased));
            s_decay = expf(ssm_a[head] * sp);
            s_beta = 1.0f / (1.0f + expf(-total_beta));
        }
    }
    __syncthreads();

    // Phase 3B: S update for token B
    {
        float decay = s_decay;
        float b = s_beta;
        float sk_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float val = sh_S[i * 128 + tid] * decay;
            sh_S[i * 128 + tid] = val;
            sk_j += val * sh_k[i];
        }
        float d_j = (__half2float(v_in_b[head * dk + tid]) - sk_j) * b;
        float o_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float updated = sh_S[i * 128 + tid] + sh_k[i] * d_j;
            sh_S[i * 128 + tid] = updated;
            o_j += updated * sh_q[i];
        }
        output_b[head * dk + tid] = __float2half(o_j);
    }

    // Write final S back to global
    __syncthreads();
    for (int i = 0; i < 128 * 128; i += 128)
        S_head[i + tid] = sh_S[i + tid];
}

// ============================================================
// Gate/Beta computation for DeltaNet (standalone, for prefill path)
// ============================================================
// gate_i = ssm_a_i * softplus(alpha_proj_i + dt_bias_i)
//   where alpha_proj_i = dot(ssm_alpha_weight[i, :], x)
//   and softplus(x) = log(1 + exp(x))
// beta_i = sigmoid(dot(ssm_beta_weight[i, :], x))
__global__ void __launch_bounds__(32)
kernel_compute_gate_beta(
    const half* __restrict__ x,          // [n_embed] input
    const void* __restrict__ alpha_w,    // [n_embed, n_heads] Q8_0
    const void* __restrict__ beta_w,     // [n_embed, n_heads] Q8_0
    const float* __restrict__ ssm_a,     // [n_heads] A parameter (negative)
    const float* __restrict__ dt_bias,   // [n_heads]
    float* __restrict__ gate_out,        // [n_heads] gate = ssm_a * softplus(alpha_proj + dt_bias)
    float* __restrict__ beta_out,        // [n_heads] sigmoid(beta_proj)
    int n_embed, int n_heads)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int lane = threadIdx.x;

    const block_q8_0* alpha_blocks = static_cast<const block_q8_0*>(alpha_w);
    const block_q8_0* beta_blocks = static_cast<const block_q8_0*>(beta_w);

    int blocks_per_head = n_embed / 32;

    float alpha_acc = 0.0f;
    float beta_acc = 0.0f;

    for (int b = lane; b < blocks_per_head; b += 32) {
        const auto& ab = alpha_blocks[head * blocks_per_head + b];
        const auto& bb = beta_blocks[head * blocks_per_head + b];
        float ad = __half2float(ab.d);
        float bd = __half2float(bb.d);

        for (int j = 0; j < 32; j++) {
            float x_val = __half2float(x[b * 32 + j]);
            alpha_acc += (ad * ab.qs[j]) * x_val;
            beta_acc += (bd * bb.qs[j]) * x_val;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        alpha_acc += __shfl_xor_sync(0xFFFFFFFF, alpha_acc, offset);
        beta_acc += __shfl_xor_sync(0xFFFFFFFF, beta_acc, offset);
    }

    if (lane == 0) {
        // gate = ssm_a * softplus(alpha_proj + dt_bias)
        float alpha_biased = alpha_acc + dt_bias[head];
        float sp = (alpha_biased > 20.0f) ? alpha_biased : logf(1.0f + expf(alpha_biased));
        gate_out[head] = ssm_a[head] * sp;

        // beta = sigmoid(beta_proj)
        beta_out[head] = 1.0f / (1.0f + expf(-beta_acc));
    }
}

// ============================================================
// Batch DeltaNet prefill kernels
// ============================================================
// These replace the per-token sequential loop in forward_prefill,
// reducing ~73K kernel launches per sequence to ~5 per layer.

// Batch Conv1D + SiLU: processes all N tokens in one launch.
// Each thread handles one dimension, loops over N tokens sequentially
// maintaining the causal conv state in registers.
__global__ void __launch_bounds__(256)
kernel_batch_conv1d_silu(
    half* __restrict__ data,           // [N, dim] in-place
    float* __restrict__ conv_state,    // [kernel_size-1, dim] rolling state
    const float* __restrict__ weight,  // [dim, kernel_size] conv weights
    int N, int dim, int kernel_size)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Load initial conv state into registers
    float st0 = conv_state[0 * dim + d];
    float st1 = conv_state[1 * dim + d];
    float st2 = conv_state[2 * dim + d];

    float w0 = weight[d * kernel_size + 0];
    float w1 = weight[d * kernel_size + 1];
    float w2 = weight[d * kernel_size + 2];
    float w3 = weight[d * kernel_size + 3];

    for (int t = 0; t < N; t++) {
        float x_val = __half2float(data[(size_t)t * dim + d]);
        float acc = w3 * x_val + w2 * st2 + w1 * st1 + w0 * st0;
        float silu = acc / (1.0f + expf(-acc));
        data[(size_t)t * dim + d] = __float2half(silu);

        st0 = st1;
        st1 = st2;
        st2 = x_val;
    }

    // Save final conv state
    conv_state[0 * dim + d] = st0;
    conv_state[1 * dim + d] = st1;
    conv_state[2 * dim + d] = st2;
}

// Batch gate/beta computation for all N tokens.
// 2D grid: (n_heads, N), 32 threads per block (one warp).
__global__ void __launch_bounds__(32)
kernel_batch_compute_gate_beta(
    const half* __restrict__ x_batch,     // [N, n_embed]
    const void* __restrict__ alpha_w,     // Q8_0 [n_heads, n_embed/32] blocks
    const void* __restrict__ beta_w,      // Q8_0 [n_heads, n_embed/32] blocks
    const float* __restrict__ ssm_a,      // [n_heads]
    const float* __restrict__ dt_bias,    // [n_heads]
    float* __restrict__ gate_out,         // [N, n_heads]
    float* __restrict__ beta_out,         // [N, n_heads]
    int N, int n_embed, int n_heads)
{
    int head = blockIdx.x;
    int t = blockIdx.y;
    if (head >= n_heads || t >= N) return;
    int lane = threadIdx.x;

    const half* x = x_batch + (size_t)t * n_embed;
    const block_q8_0* alpha_blocks = static_cast<const block_q8_0*>(alpha_w);
    const block_q8_0* beta_blocks = static_cast<const block_q8_0*>(beta_w);
    int blocks_per_head = n_embed / 32;

    float alpha_acc = 0.0f;
    float beta_acc = 0.0f;

    for (int b = lane; b < blocks_per_head; b += 32) {
        const auto& ab = alpha_blocks[head * blocks_per_head + b];
        const auto& bb = beta_blocks[head * blocks_per_head + b];
        float ad = __half2float(ab.d);
        float bd = __half2float(bb.d);
        for (int j2 = 0; j2 < 32; j2++) {
            float x_val = __half2float(x[b * 32 + j2]);
            alpha_acc += (ad * ab.qs[j2]) * x_val;
            beta_acc += (bd * bb.qs[j2]) * x_val;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        alpha_acc += __shfl_xor_sync(0xFFFFFFFF, alpha_acc, offset);
        beta_acc += __shfl_xor_sync(0xFFFFFFFF, beta_acc, offset);
    }

    if (lane == 0) {
        float alpha_biased = alpha_acc + dt_bias[head];
        float sp = (alpha_biased > 20.0f) ? alpha_biased : logf(1.0f + expf(alpha_biased));
        gate_out[t * n_heads + head] = ssm_a[head] * sp;
        beta_out[t * n_heads + head] = 1.0f / (1.0f + expf(-beta_acc));
    }
}

// Persistent DeltaNet state update for all N tokens.
// One block per head (16 blocks), 128 threads = one column of S each.
// Includes L2 normalization of Q and K (avoids separate kernel launch).
// S stays in global memory with coalesced access (row-major, threads span columns).
__global__ void __launch_bounds__(128)
kernel_deltanet_prefill(
    float* __restrict__ S,              // [n_heads, dk, dv] recurrent state
    const half* __restrict__ qkv_batch, // [N, 3*ssm_inner] (after conv1d)
    const float* __restrict__ gate_batch, // [N, n_heads]
    const float* __restrict__ beta_batch, // [N, n_heads]
    half* __restrict__ output_batch,    // [N, ssm_inner]
    int N, int n_heads, int dk, int dv, int ssm_inner,
    float q_scale)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int j = threadIdx.x;  // 0..127, one column of S
    int warp_id = j / 32;
    int lane = j % 32;

    float* S_head = S + (size_t)head * dk * dv;

    __shared__ float sh_k[128];
    __shared__ float sh_q[128];
    __shared__ float sh_reduce[4];

    for (int t = 0; t < N; t++) {
        const half* q_ptr = qkv_batch + (size_t)t * ssm_inner * 3 + head * dk;
        const half* k_ptr = q_ptr + ssm_inner;
        const half* v_ptr = k_ptr + ssm_inner;

        // Load and L2-normalize Q
        float q_raw = __half2float(q_ptr[j]);
        float q_sq = q_raw * q_raw;
        for (int off = 16; off > 0; off >>= 1)
            q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, off);
        if (lane == 0) sh_reduce[warp_id] = q_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        // Match llama.cpp: fmaxf(sum, eps²) with eps=1e-6
        float q_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f)) * q_scale;

        // Load and L2-normalize K
        float k_raw = __half2float(k_ptr[j]);
        float k_sq = k_raw * k_raw;
        for (int off = 16; off > 0; off >>= 1)
            k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, off);
        if (lane == 0) sh_reduce[warp_id] = k_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        float k_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f));

        sh_q[j] = q_raw * q_inv;
        sh_k[j] = k_raw * k_inv;
        __syncthreads();

        float decay = expf(gate_batch[t * n_heads + head]);
        float b = beta_batch[t * n_heads + head];

        // Pass 1: Decay S and compute sk
        float sk_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float val = S_head[i * 128 + j] * decay;
            S_head[i * 128 + j] = val;
            sk_j += val * sh_k[i];
        }

        // Pass 2: Update S and compute output
        float d_j = (__half2float(v_ptr[j]) - sk_j) * b;
        float o_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float updated = S_head[i * 128 + j] + sh_k[i] * d_j;
            S_head[i * 128 + j] = updated;
            o_j += updated * sh_q[i];
        }

        output_batch[(size_t)t * ssm_inner + head * dv + j] = __float2half(o_j);
        __syncthreads();  // sync before next iteration's shared memory loads
    }
}

// Batch gated RMSNorm for all N tokens.
// Grid: N * n_heads blocks, 32 threads each.
__global__ void __launch_bounds__(32)
kernel_batch_gated_rmsnorm(
    const half* __restrict__ x,        // [N, n_heads * dim_per_head]
    const float* __restrict__ weight,  // [dim_per_head] (shared across heads)
    const half* __restrict__ gate,     // [N, n_heads * dim_per_head]
    half* __restrict__ output,         // [N, n_heads * dim_per_head]
    int N, int n_heads, int dim_per_head, float eps)
{
    int block_id = blockIdx.x;
    int t = block_id / n_heads;
    int head = block_id % n_heads;
    if (t >= N) return;

    int lane = threadIdx.x;
    int stride = n_heads * dim_per_head;
    int base = t * stride + head * dim_per_head;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim_per_head; i += 32) {
        float val = __half2float(x[base + i]);
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }
    float rms_inv = rsqrtf(sum_sq / dim_per_head + eps);

    for (int i = lane; i < dim_per_head; i += 32) {
        float x_val = __half2float(x[base + i]) * rms_inv;
        float w = weight[i];
        float g = __half2float(gate[base + i]);
        float silu_g = g / (1.0f + expf(-g));
        output[base + i] = __float2half(x_val * w * silu_g);
    }
}

// ============================================================
// Batch prefill kernels — multi-sequence batching
// ============================================================

// Batch conv1d + SiLU with B independent conv states.
// Grid: (ceil(dim/256), B). Each thread handles one dimension of one sequence.
__global__ void __launch_bounds__(256)
kernel_batch_conv1d_silu_multi(
    half* __restrict__ data,           // [B*L, dim] in-place (sequences contiguous in B)
    float* __restrict__ conv_states,   // [B * (kernel_size-1) * dim] all conv states contiguous
    const float* __restrict__ weight,  // [dim, kernel_size] conv weights (shared)
    int B, int L, int dim, int kernel_size)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= dim || b >= B) return;

    int conv_stride = (kernel_size - 1) * dim;
    float* conv = conv_states + (size_t)b * conv_stride;
    half* seq_data = data + (size_t)b * L * dim;

    float st0 = conv[0 * dim + d];
    float st1 = conv[1 * dim + d];
    float st2 = conv[2 * dim + d];

    float w0 = weight[d * kernel_size + 0];
    float w1 = weight[d * kernel_size + 1];
    float w2 = weight[d * kernel_size + 2];
    float w3 = weight[d * kernel_size + 3];

    for (int t = 0; t < L; t++) {
        float x_val = __half2float(seq_data[(size_t)t * dim + d]);
        float acc = w3 * x_val + w2 * st2 + w1 * st1 + w0 * st0;
        float silu = acc / (1.0f + expf(-acc));
        seq_data[(size_t)t * dim + d] = __float2half(silu);
        st0 = st1; st1 = st2; st2 = x_val;
    }

    conv[0 * dim + d] = st0;
    conv[1 * dim + d] = st1;
    conv[2 * dim + d] = st2;
}

// Persistent DeltaNet prefill for B sequences in parallel.
// Grid: n_heads * B blocks, 128 threads each.
// Each block processes L tokens of one (batch, head) pair.
__global__ void __launch_bounds__(128)
kernel_deltanet_prefill_batch(
    float* __restrict__ S,              // [B * n_heads * dk * dv] all S states
    const half* __restrict__ qkv_batch, // [B*L, 3*ssm_inner] (sequences contiguous in B)
    const float* __restrict__ gate_batch, // [B*L, n_heads]
    const float* __restrict__ beta_batch, // [B*L, n_heads]
    half* __restrict__ output_batch,    // [B*L, ssm_inner]
    int B, int L, int n_heads, int dk, int dv, int ssm_inner,
    float q_scale)
{
    int block_id = blockIdx.x;
    int b = block_id / n_heads;
    int head = block_id % n_heads;
    if (b >= B) return;
    int j = threadIdx.x;  // 0..127, one column of S
    int warp_id = j / 32;
    int lane = j % 32;

    float* S_head = S + ((size_t)b * n_heads + head) * dk * dv;

    __shared__ float sh_k[128];
    __shared__ float sh_q[128];
    __shared__ float sh_reduce[4];

    for (int t = 0; t < L; t++) {
        int global_t = b * L + t;
        const half* q_ptr = qkv_batch + (size_t)global_t * ssm_inner * 3 + head * dk;
        const half* k_ptr = q_ptr + ssm_inner;
        const half* v_ptr = k_ptr + ssm_inner;

        // Load and L2-normalize Q
        float q_raw = __half2float(q_ptr[j]);
        float q_sq = q_raw * q_raw;
        for (int off = 16; off > 0; off >>= 1)
            q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, off);
        if (lane == 0) sh_reduce[warp_id] = q_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        // Match llama.cpp: fmaxf(sum, eps²) with eps=1e-6
        float q_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f)) * q_scale;

        // Load and L2-normalize K
        float k_raw = __half2float(k_ptr[j]);
        float k_sq = k_raw * k_raw;
        for (int off = 16; off > 0; off >>= 1)
            k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, off);
        if (lane == 0) sh_reduce[warp_id] = k_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        float k_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f));

        sh_q[j] = q_raw * q_inv;
        sh_k[j] = k_raw * k_inv;
        __syncthreads();

        float decay = expf(gate_batch[global_t * n_heads + head]);
        float beta = beta_batch[global_t * n_heads + head];

        // Pass 1: Decay S and compute sk
        float sk_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float val = S_head[i * 128 + j] * decay;
            S_head[i * 128 + j] = val;
            sk_j += val * sh_k[i];
        }

        // Pass 2: Update S and compute output
        float d_j = (__half2float(v_ptr[j]) - sk_j) * beta;
        float o_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float updated = S_head[i * 128 + j] + sh_k[i] * d_j;
            S_head[i * 128 + j] = updated;
            o_j += updated * sh_q[i];
        }

        output_batch[(size_t)global_t * ssm_inner + head * dv + j] = __float2half(o_j);
        __syncthreads();
    }
}

// ============================================================
// Chunkwise DeltaNet kernels (parallel batch prefill)
// ============================================================

// Kernel 0: Cumulative gate prefix sum + L2-normalize Q and K in-place.
// Grid: (n_heads, B), 128 threads per block.
// Each block processes L tokens of one (batch, head) pair sequentially.
__global__ void __launch_bounds__(128)
kernel_cumgate_l2norm_batch(
    half* __restrict__ qkv,            // [B*L, 3*ssm_inner] Q/K/V packed, Q and K normalized in-place
    const float* __restrict__ gate,    // [B*L, n_heads] gate log-decay values
    float* __restrict__ gate_cumul,    // [B*L, n_heads] output: cumulative gate prefix sum
    int B, int L, int n_heads, int dk, int ssm_inner, float q_scale)
{
    int head = blockIdx.x;
    int b = blockIdx.y;
    if (b >= B || head >= n_heads) return;
    int j = threadIdx.x;  // 0..127 = dimension index within head
    int warp_id = j / 32;
    int lane = j % 32;

    __shared__ float sh_reduce[4];
    float cum_gate = 0.0f;

    for (int t = 0; t < L; t++) {
        int global_t = b * L + t;

        // Accumulate gate prefix sum (only one thread writes)
        float g = gate[global_t * n_heads + head];
        cum_gate += g;
        if (j == 0) {
            gate_cumul[global_t * n_heads + head] = cum_gate;
        }

        // L2-normalize Q (same logic as original kernel)
        half* q_ptr = qkv + (size_t)global_t * ssm_inner * 3 + head * dk;
        float q_raw = __half2float(q_ptr[j]);
        float q_sq = q_raw * q_raw;
        for (int off = 16; off > 0; off >>= 1)
            q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, off);
        if (lane == 0) sh_reduce[warp_id] = q_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        float q_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f)) * q_scale;
        q_ptr[j] = __float2half(q_raw * q_inv);

        // L2-normalize K
        half* k_ptr = qkv + (size_t)global_t * ssm_inner * 3 + ssm_inner + head * dk;
        float k_raw = __half2float(k_ptr[j]);
        float k_sq = k_raw * k_raw;
        for (int off = 16; off > 0; off >>= 1)
            k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, off);
        if (lane == 0) sh_reduce[warp_id] = k_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        float k_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f));
        k_ptr[j] = __float2half(k_raw * k_inv);
        __syncthreads();
    }
}

// Kernel 1: WY representation — compute A = beta*exp(g)*K@K^T, solve T=(I+A)^{-1}, then W=T@(beta*K), U=T@(beta*V).
// Grid: (NT*B, n_heads), 128 threads per block.
// One block per (chunk, batch, head). 128 threads = one per K/V dimension.
// Shared memory: T[64][64] FP32 = 16KB.
__global__ void __launch_bounds__(128, 4)
kernel_chunkwise_wy_repr(
    const half* __restrict__ qkv,        // [B*L, 3*ssm_inner] Q/K/V (Q,K already L2-normed)
    const float* __restrict__ beta,      // [B*L, n_heads]
    const float* __restrict__ gate_cumul,// [B*L, n_heads] cumulative gate
    half* __restrict__ W_out,            // [B*L, ssm_inner] output W
    half* __restrict__ U_out,            // [B*L, ssm_inner] output U
    int B, int L, int NT, int n_heads, int dk, int ssm_inner)
{
    int cb = blockIdx.x;  // chunk_batch index: 0 .. NT*B - 1
    int head = blockIdx.y;
    int b = cb / NT;
    int chunk = cb % NT;
    if (b >= B || head >= n_heads) return;

    int j = threadIdx.x;  // 0..127 = dimension index
    int C = 64;  // chunk size
    int chunk_start = b * L + chunk * C;
    int chunk_len = min(C, L - chunk * C);
    if (chunk_len <= 0) return;

    // Shared memory: T matrix [64][64] FP32 = 16KB
    __shared__ float sh_T[64][64];

    // --- Phase 1: Compute A = beta*exp(g[i]-g[j]) * K[i]@K[j] (strictly lower tri) ---
    // Store in sh_T. Each thread handles one dimension j of K, contributes to all (i,row) elements.
    // We build A column by column in a loop over rows.

    // First load K and beta and gate_cumul for this chunk into registers/shared
    // K is at qkv + token * 3*ssm_inner + ssm_inner + head*dk + j
    // With 128 threads (one per dim), we load K[t][j] for each t in 0..chunk_len-1

    // Zero T
    for (int i = j; i < 64 * 64; i += 128)
        sh_T[i / 64][i % 64] = 0.0f;
    __syncthreads();

    // Compute A[i][col] for i > col, using outer-product accumulation
    // A[i][col] = beta[i] * exp(g[i]-g[col]) * sum_d K[i][d]*K[col][d]
    // Iterate over columns (col), for each column load K[col][j], then for each row i>col
    // atomicAdd K[i][j]*K[col][j] contribution. But atomicAdd on shared is slow.

    // Better approach: for each pair of rows (i, col) compute dot product.
    // With 128 threads over the d dimension, use warp reductions.
    // But 64*63/2 = 2016 pairs × 128-dim dot product is a lot.

    // Most efficient for 128 threads: iterate over row pairs, each pair gets a full
    // 128-thread reduction for the dot product.
    // 2016 pairs → 2016 sequential reductions. Too slow.

    // Alternative: accumulate A via rank-1 updates. For each d:
    //   A[i][col] += K[i][d] * K[col][d] * beta[i] * exp(g[i]-g[col])
    // With one thread per d, each thread contributes to all (i,col) pairs.
    // But writing to shared A[64][64] from 128 threads needs atomics.

    // Practical approach: compute A one row at a time.
    // For row i: A[i][0..i-1] = beta[i] * exp(g[i]-g[col]) * K[i] @ K[col]^T
    // K[i] @ K[col]^T is a 128-dim dot product, reduced across 128 threads.

    // We process pairs (i, col) in batches. For each col, we can process all i > col.
    // But each dot product needs a full 128-thread reduction.

    // Let's use the following approach:
    // For each col in 0..chunk_len-1:
    //   Load K[col][j] into register
    //   For each i in col+1..chunk_len-1:
    //     Compute dot = K[i][j] * K[col][j] (per thread)
    //     Warp reduce + cross-warp reduce → A[i][col]
    // This is chunk_len^2/2 reductions, each taking ~10 instructions.
    // For chunk_len=64: 2016 reductions. With 128 threads: fast in shared mem.

    // Actually let's just do it more efficiently with a shared-mem K buffer.
    // Load all K[0..63][j] values, then compute row by row.

    // Register buffer for K values of this chunk
    float K_reg[64];  // K[t][j] for t=0..63 for this thread's dimension j
    float beta_vals[64];
    float g_vals[64];

    for (int t = 0; t < chunk_len; t++) {
        int gt = chunk_start + t;
        K_reg[t] = __half2float(qkv[(size_t)gt * ssm_inner * 3 + ssm_inner + head * dk + j]);
        if (j == 0) {
            beta_vals[t] = beta[gt * n_heads + head];
            g_vals[t] = gate_cumul[gt * n_heads + head];
        }
    }
    for (int t = chunk_len; t < 64; t++) {
        K_reg[t] = 0.0f;
        if (j == 0) { beta_vals[t] = 0.0f; g_vals[t] = 0.0f; }
    }

    // All threads load beta and g (they're small, global reads are fine)
    for (int t = 0; t < chunk_len; t++) {
        int gt = chunk_start + t;
        beta_vals[t] = beta[gt * n_heads + head];
        g_vals[t] = gate_cumul[gt * n_heads + head];
    }
    for (int t = chunk_len; t < 64; t++) {
        beta_vals[t] = 0.0f;
        g_vals[t] = 0.0f;
    }

    // Compute A row by row: A[i][col] for each i, all col < i
    __shared__ float sh_dot_reduce[4];  // for cross-warp reduction
    int warp_id = j / 32;
    int lane = j % 32;

    for (int i = 1; i < chunk_len; i++) {
        float ki_j = K_reg[i];
        float beta_i = beta_vals[i];
        float g_i = g_vals[i];

        for (int col = 0; col < i; col++) {
            // Dot product: K[i] @ K[col] across 128 dimensions
            float dot = ki_j * K_reg[col];
            // Warp reduce
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            // Cross-warp reduce (4 warps)
            if (lane == 0) sh_dot_reduce[warp_id] = dot;
            __syncthreads();
            if (j == 0) {
                float total = sh_dot_reduce[0] + sh_dot_reduce[1] +
                              sh_dot_reduce[2] + sh_dot_reduce[3];
                // A[i][col] = beta_i * exp(g_i - g_col) * total
                sh_T[i][col] = beta_i * expf(g_i - g_vals[col]) * total;
            }
            __syncthreads();
        }
    }

    // --- Phase 2: Triangular solve T = (I + A)^{-1} via row-by-row forward substitution ---
    // After this, sh_T contains T (lower triangular + identity diagonal).
    // T[i][col] = -A[i][col] - sum_{k=col+1}^{i-1} T[i][k] * A_orig... no.
    // Following fla: process row by row. For row i:
    //   b_a[col] = -A[i][col] for col < i
    //   b_a[col] += sum_{k=0}^{i-1} b_a[k] * T[k][col]  (T already computed for k < i)
    //   T[i][col] = b_a[col]
    // T[i][i] = 1

    // The current sh_T contains A (positive strictly lower triangular).
    // We'll transform it in-place to T.

    // First negate all entries (A → -A)
    for (int idx = j; idx < 64 * 64; idx += 128) {
        int r = idx / 64, c = idx % 64;
        if (r > c && r < chunk_len)
            sh_T[r][c] = -sh_T[r][c];
    }
    __syncthreads();

    // Forward substitution: row by row
    // For row i (starting from 2 since row 0 is zero and row 1 only has T[1][0] = -A[1][0]):
    for (int i = 2; i < chunk_len; i++) {
        // Each thread handles one column col of this row
        // We need all columns 0..i-1 to be updated
        // With 128 threads, process columns in batches of 128
        // (only 64 columns exist, so one pass is enough)
        if (j < i) {
            int col = j;
            // b_a[col] is currently sh_T[i][col] = -A[i][col]
            float b_a_col = sh_T[i][col];
            // Accumulate: sum_{k=0}^{i-1} sh_T[i][k] * sh_T[k][col]
            // But sh_T[i][k] is -A[i][k] (not yet corrected for rows > i).
            // Wait — we need the ORIGINAL -A[i][k] values times the ALREADY COMPUTED T[k][col].
            // The issue is that sh_T[k][col] for k < i has already been overwritten to T[k][col].
            // And sh_T[i][k] for k < i is still -A[i][k] since we haven't processed row i yet.
            // So: correction = sum_{k=0}^{i-1} (-A[i][k]) * T[k][col]

            float corr = 0.0f;
            for (int k = 0; k < i; k++) {
                corr += sh_T[i][k] * sh_T[k][col];
            }
            // But sh_T[i][k] = -A[i][k] and sh_T[k][col] = T[k][col] for k < i.
            // The formula: new_row[col] = -A[i][col] + sum_k (-A[i][k]) * T[k][col]
            // This is: T[i][col] = -A[i][col] - sum_k A[i][k] * T[k][col]
            // = -(A[i][col] + sum_k A[i][k] * T[k][col])
            // = -sum_{k=col}^{i-1} (I+A)[i][k] * T[k][col] + T[i][col]*1 ... hmm
            // Actually this is exactly (I+A)^{-1} row computation.
            // The corr includes sh_T[i][col] * sh_T[col][col].
            // sh_T[col][col] is 0 (we haven't set the diagonal yet).
            // So corr = sum_{k=0, k!=col}^{i-1} sh_T[i][k] * sh_T[k][col]
            // But sh_T[i][col] = -A[i][col], and sh_T[col][col] = 0.
            // The sum effectively skips diagonal terms.

            // Hmm, the T diagonal should be 1. Let me reconsider.
            // In fla's approach, T starts as -A (negated) and the identity is added at the END.
            // So during forward sub, T[k][k] = 0 (not 1 yet). The identity is only added post-loop.
            // The formula becomes:
            // T_final[i][col] = -A[i][col] + sum_{k=0}^{i-1} (-A[i][k]) * T_final[k][col]
            // where T_final[k][col] = T_no_identity[k][col] for k != col, and
            // T_final[k][k] = 1 (from identity added at end).
            // But during the loop, T[k][k] = 0, so the sum misses the k=col term.
            // We need to add it back: + (-A[i][col]) * 1 = -A[i][col].
            // Wait, that's already in b_a_col...

            // Let me re-derive from fla's code:
            // b_A initially = -A (strictly lower, zero diagonal)
            // For row i:
            //   b_a = -A[i,:] (restricted to cols < i)
            //   b_a += sum(b_a[:, None] * b_A, axis=rows)
            //     = b_a[j] += sum_k b_a[k] * b_A[k][j]
            //   b_A[i,:] = b_a
            // Then at end: b_A += I

            // So during forward sub, b_A[k][col] for k < i = T_no_identity[k][col]
            // b_a[col] = -A[i][col] + sum_{k=0}^{i-1} (-A[i][k]) * T_no_identity[k][col]
            // The self-term k=col: (-A[i][col]) * T_no_identity[col][col] = (-A[i][col]) * 0 = 0
            // So no self-term issue. Just accumulate straightforwardly.

            sh_T[i][col] = b_a_col + corr;
        }
        __syncthreads();
    }

    // Add identity to diagonal
    if (j < chunk_len) {
        sh_T[j][j] += 1.0f;
    }
    // Zero out rows/cols beyond chunk_len
    if (j < 64) {
        for (int c = 0; c < 64; c++) {
            if (j >= chunk_len || c >= chunk_len)
                sh_T[j][c] = 0.0f;
        }
    }
    __syncthreads();

    // --- Phase 3: Compute W = T @ (beta*exp(g_local)*K) and U = T @ (beta*V) ---
    // GATED W: W[t][j] = sum_{s=0}^{t} T[t][s] * beta[s] * exp(g_local[s]) * K[s][j]
    // where g_local[s] = g_cumul[s] - g_offset, g_offset = g_cumul[cs-1] (0 for first chunk).
    // The exp(g_local) factor accounts for how much h decays before token s sees it.
    // U is unchanged: U[t][j] = sum_{s=0}^{t} T[t][s] * beta[s] * V[s][j]
    float g_offset = (chunk > 0) ? gate_cumul[(chunk_start - 1) * n_heads + head] : 0.0f;

    for (int t = 0; t < chunk_len; t++) {
        float w_val = 0.0f;
        for (int s = 0; s <= t; s++) {  // T is lower triangular
            float g_local_s = g_vals[s] - g_offset;
            w_val += sh_T[t][s] * beta_vals[s] * expf(g_local_s) * K_reg[s];
        }
        int gt = chunk_start + t;
        W_out[(size_t)gt * ssm_inner + head * dk + j] = __float2half(w_val);
    }

    // For U, need V values. Load V and compute U.
    float V_reg[64];
    for (int t = 0; t < chunk_len; t++) {
        int gt = chunk_start + t;
        V_reg[t] = __half2float(qkv[(size_t)gt * ssm_inner * 3 + 2 * ssm_inner + head * dk + j]);
    }
    for (int t = chunk_len; t < 64; t++) V_reg[t] = 0.0f;

    for (int t = 0; t < chunk_len; t++) {
        float u_val = 0.0f;
        for (int s = 0; s <= t; s++) {
            u_val += sh_T[t][s] * beta_vals[s] * V_reg[s];
        }
        int gt = chunk_start + t;
        U_out[(size_t)gt * ssm_inner + head * dk + j] = __float2half(u_val);
    }
}

// Kernel 2: State propagation — sequential over chunks, parallel over V-tiles.
// Grid: (ceil(dv/BV), B*n_heads) with BV=32 → (4, B*16), 128 threads per block.
// Each block handles one dk×BV tile of h, iterating over NT chunks.
// Thread tid (0..127) = row index in dk dimension, holds h[tid][v_base+0..31] in 32 registers.
//
// Per chunk:
//   1. Store h at chunk boundary
//   2. W@h matmul [C,dk]×[dk,BV]→[C,BV] via shared memory (h→shared, W tiled)
//   3. v_new = U - W@h, store to global + shared
//   4. Decay h, then h += K^T @ gated_v_new (per-thread, no reduction)
__global__ void __launch_bounds__(128, 3)
kernel_chunkwise_state_propagation(
    float* __restrict__ S_out,           // [B*n_heads, dk, dv] final S state for this layer
    const half* __restrict__ qkv,        // [B*L, 3*ssm_inner] for K access
    const half* __restrict__ W,          // [B*L, ssm_inner]
    const half* __restrict__ U,          // [B*L, ssm_inner]
    const float* __restrict__ gate_cumul,// [B*L, n_heads]
    float* __restrict__ h_states,        // [B*NT*n_heads, dk, dv] output h at each chunk boundary
    half* __restrict__ v_new,            // [B*L, ssm_inner] output corrected values
    int B, int L, int NT, int n_heads, int dk, int dv, int ssm_inner)
{
    const int BV = 32;
    const int C = 64;
    const int BK = 16;  // K-tile size for W@h matmul
    // Thread mapping for matmul output [C=64, BV=32]:
    // 128 threads → 16 in M (C/TM=64/4), 8 in N (BV/TN=32/4)
    const int TM = 4, TN = 4;

    int v_tile = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / n_heads;
    int head = bh % n_heads;
    if (b >= B) return;

    int tid = threadIdx.x;
    int v_base = v_tile * BV;
    int m_idx = tid / 8;   // 0..15, covers M=64 in strides of TM=4
    int n_idx = tid % 8;   // 0..7,  covers N=32 in strides of TN=4

    // h state in registers: h[tid][v_base+0..31]
    float h_reg[32];
    #pragma unroll
    for (int v = 0; v < 32; v++) h_reg[v] = 0.0f;

    // Shared memory: h tile + W tile + v_new buffer
    __shared__ float sh_h[128][32];    // [dk, BV] h tile (16 KB)
    __shared__ float sh_W[64][16];     // [C, BK] W tile (4 KB) - reused per K-tile
    __shared__ float sh_vnew[64][32];  // [C, BV] v_new buffer for pass 2 (8 KB)
    // Total: 28 KB

    float g_offset = 0.0f;

    for (int chunk = 0; chunk < NT; chunk++) {
        int chunk_start = b * L + chunk * C;
        int chunk_len = min(C, L - chunk * C);
        if (chunk_len <= 0) break;

        // 1) Store h state at chunk boundary (BEFORE any modification)
        size_t h_off = ((size_t)(b * NT + chunk) * n_heads + head) * dk * dv;
        #pragma unroll
        for (int v = 0; v < 32; v++)
            h_states[h_off + tid * dv + v_base + v] = h_reg[v];

        // Load h into shared memory for the matmul
        #pragma unroll
        for (int v = 0; v < 32; v++)
            sh_h[tid][v] = h_reg[v];

        // Load g_last — all threads read it after syncthreads
        __shared__ float sh_g_last;
        if (tid == 0) {
            sh_g_last = gate_cumul[(chunk_start + chunk_len - 1) * n_heads + head];
        }
        __syncthreads();
        float g_last = sh_g_last;

        // === W@h matmul: [chunk_len, dk=128] × [dk=128, BV=32] → [chunk_len, BV=32] ===
        // Thread (m_idx, n_idx) computes output tile [m_idx*TM .. +TM-1][n_idx*TN .. +TN-1]
        float acc[TM][TN];
        #pragma unroll
        for (int m = 0; m < TM; m++)
            #pragma unroll
            for (int n = 0; n < TN; n++)
                acc[m][n] = 0.0f;

        for (int k_tile = 0; k_tile < dk; k_tile += BK) {
            // Cooperative load of W tile: W[chunk_start+t][head*dk + k_tile + k]
            // 64*16 = 1024 elements, 128 threads → 8 per thread
            #pragma unroll
            for (int i = tid; i < chunk_len * BK; i += 128) {
                int t = i / BK;
                int k = i % BK;
                int gt = chunk_start + t;
                sh_W[t][k] = __half2float(W[(size_t)gt * ssm_inner + head * dk + k_tile + k]);
            }
            // Zero remaining rows if chunk_len < C
            for (int i = chunk_len * BK + tid; i < C * BK; i += 128) {
                sh_W[i / BK][i % BK] = 0.0f;
            }
            __syncthreads();

            // Compute tile of W@h
            #pragma unroll
            for (int bk = 0; bk < BK; bk++) {
                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    float w = sh_W[m_idx * TM + m][bk];
                    #pragma unroll
                    for (int n = 0; n < TN; n++) {
                        acc[m][n] += w * sh_h[k_tile + bk][n_idx * TN + n];
                    }
                }
            }
            __syncthreads();
        }

        // v_new[t][v] = U[t][v] - wh[t][v], store to global + shared
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            int t = m_idx * TM + m;
            if (t >= chunk_len) continue;
            int gt = chunk_start + t;
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int v = n_idx * TN + n;
                float u_val = __half2float(U[(size_t)gt * ssm_inner + head * dk + v_base + v]);
                float vn = u_val - acc[m][n];
                v_new[(size_t)gt * ssm_inner + head * dk + v_base + v] = __float2half(vn);
                sh_vnew[t][v] = vn;
            }
        }
        __syncthreads();

        // === PASS 2: Decay h, then h += K^T @ gated_v_new ===
        float decay = expf(g_last - g_offset);
        #pragma unroll
        for (int v = 0; v < 32; v++)
            h_reg[v] *= decay;

        // h[tid][v] += K[t][tid] * gated_v_new[t][v] — per-thread, no reduction needed
        for (int t = 0; t < chunk_len; t++) {
            int gt = chunk_start + t;
            float k_val = __half2float(qkv[(size_t)gt * ssm_inner * 3 + ssm_inner + head * dk + tid]);
            float g_t = gate_cumul[gt * n_heads + head];
            float g_factor = expf(g_last - g_t);
            #pragma unroll
            for (int v = 0; v < 32; v++)
                h_reg[v] += k_val * sh_vnew[t][v] * g_factor;
        }
        __syncthreads();

        g_offset = g_last;
    }

    // Store final S state
    size_t s_base = ((size_t)b * n_heads + head) * dk * dv;
    #pragma unroll
    for (int v = 0; v < 32; v++)
        S_out[s_base + tid * dv + v_base + v] = h_reg[v];
}

// Kernel 3: Chunkwise output — per chunk, per head.
// o[t][v] = Q[t] @ h_chunk * exp(g_local[t]) + sum_{s<=t} (Q[t]@K[s] * exp(g[t]-g[s])) * v_new[s][v]
// q_scale already baked into Q during L2 normalization.
//
// Phase 1: Compute gated causal QK^T [C,C] via tiled matmul in shared memory.
// Phase 2: For each token, compute output per-v-column independently (no reduction).
//
// Grid: (1, NT*B, n_heads), 128 threads.
// Thread tid = v dimension in phase 2 (dv=128).
__global__ void __launch_bounds__(128, 3)
kernel_chunkwise_output(
    const half* __restrict__ qkv,        // [B*L, 3*ssm_inner]
    const float* __restrict__ h_states,  // [B*NT*n_heads, dk, dv]
    const half* __restrict__ v_new,      // [B*L, ssm_inner]
    const float* __restrict__ gate_cumul,// [B*L, n_heads]
    half* __restrict__ output,           // [B*L, ssm_inner]
    int B, int L, int NT, int n_heads, int dk, int dv, int ssm_inner)
{
    int cb = blockIdx.y;
    int head = blockIdx.z;
    int b = cb / NT;
    int chunk = cb % NT;
    if (b >= B || head >= n_heads) return;

    int tid = threadIdx.x;
    const int C = 64;
    const int BK = 16;
    int chunk_start = b * L + chunk * C;
    int chunk_len = min(C, L - chunk * C);
    if (chunk_len <= 0) return;

    size_t h_off = ((size_t)(b * NT + chunk) * n_heads + head) * dk * dv;

    // Shared memory for QK^T matmul
    __shared__ float sh_Q_tile[64][16 + 1]; // [C, BK] with padding to avoid bank conflicts (4.25 KB)
    __shared__ float sh_K_tile[64][16 + 1]; // [C, BK] with padding (4.25 KB)
    __shared__ float sh_QK[64][64 + 1];     // [C, C] gated causal attention matrix (16.25 KB)
    // sh_Q_row reuses sh_Q_tile space (only need 128 floats, share with first row)
    // Total: ~25 KB

    // === Phase 1: QK^T matmul [C,dk] × [dk,C] → [C,C] ===
    // Thread layout: m_idx = tid/4 (0..31), n_idx = tid%4 (0..3)
    // Each thread computes QK[m_idx*2..+1][n_idx*16..+15] (TM=2, TN=16, 32 elements)
    const int TM = 2, TN = 16;
    int m_idx = tid / 4;   // 0..31
    int n_idx = tid % 4;   // 0..3
    float acc[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; m++)
        #pragma unroll
        for (int n = 0; n < TN; n++)
            acc[m][n] = 0.0f;

    for (int k_tile = 0; k_tile < dk; k_tile += BK) {
        // Cooperative load Q tile [C, BK] and K tile [C, BK]
        // 128 threads, 64*16=1024 elements → 8 per thread
        for (int i = tid; i < C * BK; i += 128) {
            int row = i / BK, col = i % BK;
            int gt = chunk_start + row;
            if (row < chunk_len) {
                sh_Q_tile[row][col] = __half2float(qkv[(size_t)gt * ssm_inner * 3 + head * dk + k_tile + col]);
                sh_K_tile[row][col] = __half2float(qkv[(size_t)gt * ssm_inner * 3 + ssm_inner + head * dk + k_tile + col]);
            } else {
                sh_Q_tile[row][col] = 0.0f;
                sh_K_tile[row][col] = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int bk = 0; bk < BK; bk++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                float q = sh_Q_tile[m_idx * TM + m][bk];
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[m][n] += q * sh_K_tile[n_idx * TN + n][bk];
                }
            }
        }
        __syncthreads();
    }

    // Apply causal mask + gating, store to shared
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int t = m_idx * TM + m;
        float g_t = (t < chunk_len) ? gate_cumul[(chunk_start + t) * n_heads + head] : 0.0f;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int s = n_idx * TN + n;
            float val = 0.0f;
            if (s <= t && t < chunk_len && s < chunk_len) {
                float g_s = gate_cumul[(chunk_start + s) * n_heads + head];
                val = acc[m][n] * expf(g_t - g_s);
            }
            sh_QK[t][s] = val;
        }
    }
    __syncthreads();

    // === Phase 2: Compute output per column ===
    // Thread tid = v dimension (dv=128). Each thread computes output[t][v] independently.
    float g_offset = (chunk > 0) ? gate_cumul[(chunk_start - 1) * n_heads + head] : 0.0f;
    int v = tid;

    for (int t = 0; t < chunk_len; t++) {
        int gt = chunk_start + t;
        float g_t = gate_cumul[gt * n_heads + head];

        // Inter-chunk: Q[t] @ h[:,v] * exp(g_local_t)
        float o_inter = 0.0f;
        for (int k = 0; k < dk; k++) {
            float q_k = __half2float(qkv[(size_t)gt * ssm_inner * 3 + head * dk + k]);
            o_inter += q_k * h_states[h_off + k * dv + v];
        }
        o_inter *= expf(g_t - g_offset);

        // Intra-chunk: sum_{s<=t} QK[t][s] * v_new[s][v]
        float o_intra = 0.0f;
        for (int s = 0; s <= t; s++) {
            int gs = chunk_start + s;
            o_intra += sh_QK[t][s] * __half2float(v_new[(size_t)gs * ssm_inner + head * dk + v]);
        }

        output[(size_t)gt * ssm_inner + head * dk + v] = __float2half(o_inter + o_intra);
    }
}

// Batched RoPE for prefill: apply RoPE to all B*L tokens.
// Position for token i = i % L (each sequence starts at pos 0).
// Grid.x: n_heads + n_kv_heads, Grid.y: B*L tokens.
__global__ void __launch_bounds__(256)
kernel_rope_batch(
    half* __restrict__ q,    // [B*L, n_heads * head_dim]
    half* __restrict__ k,    // [B*L, n_kv_heads * head_dim]
    int n_heads, int n_kv_heads, int head_dim,
    int L, float theta, int rope_dim)
{
    int head = blockIdx.x;
    int token_idx = blockIdx.y;
    int pos = token_idx % L;

    int is_k = (head >= n_heads);
    int actual_head;
    half* vec;

    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    if (!is_k) {
        actual_head = head;
        vec = q + (size_t)token_idx * q_stride + actual_head * head_dim;
    } else {
        actual_head = head - n_heads;
        if (actual_head >= n_kv_heads) return;
        vec = k + (size_t)token_idx * kv_stride + actual_head * head_dim;
    }

    int tid = threadIdx.x;
    int n_pairs = rope_dim / 2;
    if (tid >= n_pairs) return;

    float freq_exp = -2.0f * tid / (float)rope_dim;
    float freq = (float)pos * powf(theta, freq_exp);
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);

    int d0 = 2 * tid, d1 = 2 * tid + 1;
    float x0 = __half2float(vec[d0]);
    float x1 = __half2float(vec[d1]);
    vec[d0] = __float2half(x0 * cos_val - x1 * sin_val);
    vec[d1] = __float2half(x0 * sin_val + x1 * cos_val);
}

// Batched deinterleave Q+gate for all B*L tokens.
// Input: [B*L, n_head * head_dim * 2] interleaved
// Output: [B*L, n_head * head_dim] Q, [B*L, n_head * head_dim] gate
// Grid: (ceil(n_head*head_dim / 256), B*L)
__global__ void __launch_bounds__(256)
kernel_deinterleave_qgate_batch(
    const half* __restrict__ interleaved,  // [B*L, n_head * head_dim * 2]
    half* __restrict__ q_out,              // [B*L, n_head * head_dim]
    half* __restrict__ gate_out,           // [B*L, n_head * head_dim]
    int n_tokens, int n_head, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    int total = n_head * head_dim;
    if (idx >= total || token >= n_tokens) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    size_t src_base = (size_t)token * n_head * head_dim * 2;
    int src_q = h * head_dim * 2 + d;
    int src_g = h * head_dim * 2 + head_dim + d;

    size_t dst_offset = (size_t)token * total + idx;
    q_out[dst_offset] = interleaved[src_base + src_q];
    gate_out[dst_offset] = interleaved[src_base + src_g];
}

// Flash-attention-style batched causal self-attention for prefill.
// One warp (32 threads) per (batch, q_head) pair.
// Each warp processes all L query positions with online softmax.
// No materialized score matrix — O(1) extra memory per block.
// GQA: n_head Q heads share n_kv_heads K/V heads.
__global__ void __launch_bounds__(32)
kernel_batch_causal_attn(
    const half* __restrict__ Q,       // [B*L, n_head * head_dim]
    const half* __restrict__ K,       // [B*L, n_kv_heads * head_dim]
    const half* __restrict__ V,       // [B*L, n_kv_heads * head_dim]
    half* __restrict__ output,        // [B*L, n_head * head_dim]
    int L, int n_head, int n_kv_heads, int head_dim, float scale)
{
    int block_id = blockIdx.x;
    int b = block_id / n_head;
    int h = block_id % n_head;
    int kv_h = h * n_kv_heads / n_head;  // GQA mapping
    int lane = threadIdx.x;  // 0..31

    int q_stride = n_head * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    int elems = head_dim / 32;  // 8 for head_dim=256

    for (int t = 0; t < L; t++) {
        const half* q_ptr = Q + (size_t)(b * L + t) * q_stride + h * head_dim + lane * elems;

        float q_local[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) q_local[e] = __half2float(q_ptr[e]) * scale;

        float max_score = -FLT_MAX;
        float sum_exp = 0.0f;
        float o_local[8] = {};

        for (int k = 0; k <= t; k++) {
            const half* k_ptr = K + (size_t)(b * L + k) * kv_stride + kv_h * head_dim + lane * elems;
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; e++) dot += q_local[e] * __half2float(k_ptr[e]);

            // Warp reduction — all 32 lanes get the full dot product
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

            // Online softmax update
            float old_max = max_score;
            max_score = fmaxf(max_score, dot);
            float rescale = expf(old_max - max_score);
            sum_exp = sum_exp * rescale + expf(dot - max_score);

            #pragma unroll
            for (int e = 0; e < 8; e++) o_local[e] *= rescale;

            float w = expf(dot - max_score);
            const half* v_ptr = V + (size_t)(b * L + k) * kv_stride + kv_h * head_dim + lane * elems;
            #pragma unroll
            for (int e = 0; e < 8; e++) o_local[e] += w * __half2float(v_ptr[e]);
        }

        float inv_sum = 1.0f / sum_exp;
        half* o_ptr = output + (size_t)(b * L + t) * q_stride + h * head_dim + lane * elems;
        #pragma unroll
        for (int e = 0; e < 8; e++) o_ptr[e] = __float2half(o_local[e] * inv_sum);
    }
}

// Batched sigmoid-mul: output[i] = attn[i] * sigmoid(gate[i]) for all B*L tokens
__global__ void __launch_bounds__(256)
kernel_sigmoid_mul_batch(
    const half* __restrict__ attn,     // [N, dim]
    const half* __restrict__ gate,     // [N, dim]
    half* __restrict__ output,         // [N, dim]
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float a = __half2float(attn[idx]);
    float g = __half2float(gate[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    output[idx] = __float2half(a * sig);
}

// ============================================================
// GQA attention decode kernel (multi-warp)
// ============================================================
// 8 warps (256 threads) per query head for better SM utilization
// Warps split scoring across seq_len; all threads cover head_dim for value accumulation
__global__ void __launch_bounds__(256)
kernel_gqa_attention_decode(
    const half* __restrict__ q,        // [n_head, head_dim]
    const half* __restrict__ k_cache,  // [max_seq, n_kv_heads, head_dim]
    const half* __restrict__ v_cache,  // [max_seq, n_kv_heads, head_dim]
    half* __restrict__ output,         // [n_head, head_dim]
    float* __restrict__ scores_buf,    // [n_head, max_seq] scratch (fixed stride)
    const int* __restrict__ d_pos,     // device pointer to current position
    int n_head, int n_kv_heads, int head_dim, int max_seq, float scale)
{
    int qh = blockIdx.x;
    if (qh >= n_head) return;

    constexpr int N_WARPS = 8;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    int seq_len = *d_pos + 1;
    int kv_head = qh / (n_head / n_kv_heads);

    const half* q_head = q + qh * head_dim;
    float* scores = scores_buf + qh * max_seq;

    // Step 1: Compute attention scores — warps split across time steps
    // Each warp handles t = warp_id, warp_id+N_WARPS, warp_id+2*N_WARPS, ...
    for (int t = warp_id; t < seq_len; t += N_WARPS) {
        const half* k_t = k_cache + (size_t)t * n_kv_heads * head_dim + kv_head * head_dim;

        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            dot += __half2float(q_head[d]) * __half2float(k_t[d]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
        }

        if (lane == 0) scores[t] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax — all 256 threads cooperate
    float max_val = -FLT_MAX;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        max_val = fmaxf(max_val, scores[t]);
    }
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }
    // Cross-warp reduction via shared memory (no cross-warp shuffles)
    __shared__ float s_reduce[N_WARPS];
    if (lane == 0) s_reduce[warp_id] = max_val;
    __syncthreads();
    if (tid == 0) {
        float m = s_reduce[0];
        for (int i = 1; i < N_WARPS; i++) m = fmaxf(m, s_reduce[i]);
        s_reduce[0] = m;
    }
    __syncthreads();
    max_val = s_reduce[0];

    float sum_exp = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        sum_exp += e;
    }
    // Warp-level sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_xor_sync(0xFFFFFFFF, sum_exp, offset);
    }
    if (lane == 0) s_reduce[warp_id] = sum_exp;
    __syncthreads();
    if (tid == 0) {
        float s = s_reduce[0];
        for (int i = 1; i < N_WARPS; i++) s += s_reduce[i];
        s_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_reduce[0];

    for (int t = tid; t < seq_len; t += blockDim.x) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of values — 256 threads directly cover head_dim=256
    half* out_head = output + qh * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const half* v_t = v_cache + (size_t)t * n_kv_heads * head_dim + kv_head * head_dim;
            acc += scores[t] * __half2float(v_t[d]);
        }
        out_head[d] = __float2half(acc);
    }
}

// ============================================================
// Argmax kernels for greedy decoding (multi-block)
// ============================================================
// Phase 1: Each block finds max in its range, writes to scratch
constexpr int ARGMAX_BLOCKS = 256;

__global__ void __launch_bounds__(256)
kernel_argmax_partial(const float* __restrict__ logits,
                      float* __restrict__ partial_max,
                      int* __restrict__ partial_idx,
                      int n) {
    __shared__ float s_max[256];
    __shared__ int s_idx[256];

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int global_tid = bid * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = global_tid; i < n; i += stride) {
        float v = logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max[tid + s] > s_max[tid]) {
            s_max[tid] = s_max[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[bid] = s_max[0];
        partial_idx[bid] = s_idx[0];
    }
}

// Phase 2: Reduce partial results (ARGMAX_BLOCKS entries)
__global__ void __launch_bounds__(256)
kernel_argmax_reduce(const float* __restrict__ partial_max,
                     const int* __restrict__ partial_idx,
                     int* __restrict__ result,
                     int n_partials) {
    __shared__ float s_max[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = 0;

    if (tid < n_partials) {
        local_max = partial_max[tid];
        local_idx = partial_idx[tid];
    }

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max[tid + s] > s_max[tid]) {
            s_max[tid] = s_max[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) *result = s_idx[0];
}

// Remap argmax result through a token ID lookup table (for reduced LM head)
__global__ void kernel_remap_token(const int* __restrict__ token_map,
                                   int* __restrict__ argmax_result) {
    *argmax_result = token_map[*argmax_result];
}

// Remap with IDK: indices >= K or < 0 map to -1 (IDK = skip speculation)
__global__ void kernel_remap_token_idk(const int* __restrict__ token_map,
                                       int* __restrict__ argmax_result,
                                       int K) {
    int idx = *argmax_result;
    *argmax_result = (idx < 0 || idx >= K) ? -1 : token_map[idx];
}

// Confidence gate: if softmax probability of argmax token < threshold, set to -1 (skip)
__global__ void kernel_confidence_gate(const float* __restrict__ logits,
                                       int* __restrict__ argmax_result,
                                       int N, float threshold) {
    int idx = *argmax_result;
    if (idx < 0) return;  // already gated

    float max_val = logits[idx];
    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_exp += expf(logits[i] - max_val);
    }
    float prob = 1.0f / sum_exp;

    if (prob < threshold) {
        *argmax_result = -1;
    }
}

// ============================================================
// KV cache store kernel (reads pos from device memory for graph capture)
// ============================================================
__global__ void __launch_bounds__(256)
kernel_kv_cache_store(
    half* __restrict__ k_cache,        // [max_seq, kv_dim]
    half* __restrict__ v_cache,        // [max_seq, kv_dim]
    const half* __restrict__ k_src,    // [kv_dim]
    const half* __restrict__ v_src,    // [kv_dim]
    const int* __restrict__ d_pos,     // device pointer to position
    int kv_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kv_dim) return;
    int p = *d_pos;
    k_cache[(size_t)p * kv_dim + idx] = k_src[idx];
    v_cache[(size_t)p * kv_dim + idx] = v_src[idx];
}

// ============================================================
// Deinterleave Q + gate: [Q_h0, gate_h0, Q_h1, gate_h1, ...] → [Q_all, gate_all]
// ============================================================
__global__ void __launch_bounds__(256)
kernel_deinterleave_qgate(
    const half* __restrict__ interleaved,  // [n_head * head_dim * 2]
    half* __restrict__ q_out,              // [n_head * head_dim]
    half* __restrict__ gate_out,           // [n_head * head_dim]
    int n_head, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_head * head_dim;
    if (idx >= total) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    // In interleaved layout: head h Q starts at h * head_dim * 2
    //                        head h gate starts at h * head_dim * 2 + head_dim
    int src_q = h * head_dim * 2 + d;
    int src_g = h * head_dim * 2 + head_dim + d;

    q_out[idx] = interleaved[src_q];
    gate_out[idx] = interleaved[src_g];
}

// FP16 to FP32 conversion for logits
__global__ void __launch_bounds__(256)
kernel_half_to_float(const half* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

// ============================================================
// InferenceState allocation
// ============================================================

void InferenceState::allocate(const ModelConfig& cfg, CudaAllocator& alloc, int max_seq) {
    auto a = [&](size_t bytes) -> void* { return alloc.alloc(bytes); };

    // Main buffers — two backing buffers for pointer-swap residual pattern
    buf_a    = static_cast<half*>(a(cfg.n_embed * sizeof(half)));
    buf_b    = static_cast<half*>(a(cfg.n_embed * sizeof(half)));
    x_norm   = static_cast<half*>(a(cfg.n_embed * sizeof(half)));
    x = buf_a;
    residual = buf_b;

    // DeltaNet scratch — Q/K/V alias into qkv (no separate allocation)
    int qkv_dim = cfg.ssm_inner_size * 3;  // 6144
    qkv      = static_cast<half*>(a(qkv_dim * sizeof(half)));
    gate_z   = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    attn_out = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    gated_out= static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));

    // Full attention scratch
    int q_dim = cfg.n_head * cfg.head_dim * 2;  // Q + gate = 4096
    fa_q     = static_cast<half*>(a(q_dim * sizeof(half)));
    fa_k     = static_cast<half*>(a(cfg.n_head_kv * cfg.head_dim * sizeof(half)));
    fa_v     = static_cast<half*>(a(cfg.n_head_kv * cfg.head_dim * sizeof(half)));
    attn_scores = static_cast<float*>(a(cfg.n_head * max_seq * sizeof(float)));

    // FFN scratch
    ffn_gate = static_cast<half*>(a(cfg.n_ff * sizeof(half)));
    ffn_up   = static_cast<half*>(a(cfg.n_ff * sizeof(half)));
    ffn_out  = static_cast<half*>(a(cfg.n_ff * sizeof(half)));

    // Output
    logits_h = static_cast<half*>(a(cfg.n_vocab * sizeof(half)));
    logits_f = static_cast<float*>(a(cfg.n_vocab * sizeof(float)));

    // DeltaNet scratch
    d_alpha = static_cast<float*>(a(cfg.ssm_n_heads * sizeof(float)));
    d_beta  = static_cast<float*>(a(cfg.ssm_n_heads * sizeof(float)));

    // dp4a Q8_1 scratch buffers — sized for the largest input vector
    // Largest: n_ff=3584 (FFN down input), n_embed=1024 (most projections)
    int max_q8_dim = std::max(cfg.n_ff, std::max(cfg.n_embed, cfg.ssm_inner_size));
    int q8_blocks = (max_q8_dim + 31) / 32;
    x_q8_a = a(q8_blocks * 36);  // 36 bytes per block_q8_1
    x_q8_b = a(q8_blocks * 36);

    // Pre-allocated argmax results: d_argmax_token[0] = token A, d_argmax_token[1] = token B
    // Adjacent allocation so forward_2tok can read both in one 8-byte D2H memcpy
    d_argmax_token = static_cast<int*>(a(2 * sizeof(int)));
    argmax_partial_max = static_cast<float*>(a(ARGMAX_BLOCKS * sizeof(float)));
    argmax_partial_idx = static_cast<int*>(a(ARGMAX_BLOCKS * sizeof(int)));
    // d_token_id layout: [tok_a, pos_a, tok_b, pos_b] — 4 ints for batch2 support
    d_token_id = static_cast<int*>(a(4 * sizeof(int)));
    d_pos = d_token_id + 1;
    max_seq_alloc = max_seq;

    // Create non-blocking streams for graph capture + overlap
    GWEN_CHECK_CUDA(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    GWEN_CHECK_CUDA(cudaStreamCreateWithFlags(&overlap_stream, cudaStreamNonBlocking));
    GWEN_CHECK_CUDA(cudaEventCreate(&ev_conv_done));
    GWEN_CHECK_CUDA(cudaEventCreate(&ev_overlap_done));

    // DeltaNet states (18 layers)
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        if (!cfg.is_full_attention_layer(i)) {
            DeltaNetState state;
            state.n_heads = cfg.ssm_n_heads;
            state.state_size = cfg.ssm_state_size;
            state.qkv_dim = qkv_dim;
            state.conv_kernel = cfg.ssm_conv_kernel;

            // State matrix: n_heads × d_k × d_v in FP32
            size_t S_bytes = (size_t)state.n_heads * state.state_size * state.state_size * sizeof(float);
            state.S = static_cast<float*>(a(S_bytes));
            GWEN_CHECK_CUDA(cudaMemset(state.S, 0, S_bytes));

            // Conv state: (kernel-1) × qkv_dim in FP32
            size_t conv_bytes = (size_t)(state.conv_kernel - 1) * state.qkv_dim * sizeof(float);
            state.conv_state = static_cast<float*>(a(conv_bytes));
            GWEN_CHECK_CUDA(cudaMemset(state.conv_state, 0, conv_bytes));

            deltanet_states.push_back(state);
        }
    }

    // KV caches (6 layers)
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        if (cfg.is_full_attention_layer(i)) {
            KVCache cache;
            cache.max_seq = max_seq;
            cache.n_kv_heads = cfg.n_head_kv;
            cache.head_dim = cfg.head_dim;

            size_t kv_bytes = (size_t)max_seq * cfg.n_head_kv * cfg.head_dim * sizeof(half);
            cache.k_cache = static_cast<half*>(a(kv_bytes));
            cache.v_cache = static_cast<half*>(a(kv_bytes));
            GWEN_CHECK_CUDA(cudaMemset(cache.k_cache, 0, kv_bytes));
            GWEN_CHECK_CUDA(cudaMemset(cache.v_cache, 0, kv_bytes));

            kv_caches.push_back(cache);
        }
    }

    pos = 0;

    // Allow 64 KB dynamic shared memory for DeltaNet fused kernels (S matrix in shared)
    GWEN_CHECK_CUDA(cudaFuncSetAttribute(
        (const void*)kernel_deltanet_fused,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    GWEN_CHECK_CUDA(cudaFuncSetAttribute(
        (const void*)kernel_deltanet_fused_2tok,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
}

void InferenceState::allocate_prefill(const ModelConfig& cfg, CudaAllocator& alloc,
                                      int max_tokens, bool f32_path) {
    auto a = [&](size_t bytes) -> void* { return alloc.alloc(bytes); };
    max_prefill = max_tokens;

    // Batch activation buffers: [max_tokens, dim]
    prefill_x     = static_cast<half*>(a(max_tokens * cfg.n_embed * sizeof(half)));
    prefill_out   = static_cast<half*>(a(max_tokens * cfg.n_embed * sizeof(half)));
    prefill_norm  = static_cast<half*>(a(max_tokens * cfg.n_embed * sizeof(half)));

    // F32 buffers for verified reference path (GWEN_GEMM_DECODE=1 / extract_hidden)
    // Only needed for single-sequence forward_prefill, not for batch extraction.
    if (f32_path) {
        prefill_x_f32   = static_cast<float*>(a(max_tokens * cfg.n_embed * sizeof(float)));
        prefill_out_f32 = static_cast<float*>(a(max_tokens * cfg.n_embed * sizeof(float)));

        size_t qkv_dim_total = cfg.ssm_inner_size * 3;
        prefill_proj_qkv_f32  = static_cast<float*>(a(max_tokens * qkv_dim_total * sizeof(float)));
        prefill_proj_gate_f32 = static_cast<float*>(a(max_tokens * cfg.ssm_inner_size * sizeof(float)));
        prefill_dn_out_f32    = static_cast<float*>(a(max_tokens * cfg.ssm_inner_size * sizeof(float)));
        prefill_ffn_gate_f32  = static_cast<float*>(a(max_tokens * cfg.n_ff * sizeof(float)));
        prefill_ffn_up_f32    = static_cast<float*>(a(max_tokens * cfg.n_ff * sizeof(float)));
        prefill_ffn_out_f32   = static_cast<float*>(a(max_tokens * cfg.n_ff * sizeof(float)));
    }

    // Dequantized weight scratch — large enough for biggest weight matrix
    // Biggest: QKV (6144 × 1024) or Q+gate (4096 × 1024) or FFN gate/up (3584 × 1024)
    // or LM head (248320 × 1024)
    // We'll cap at FFN-sized since LM head uses GEMV (single token at end)
    size_t max_weight_elems = (size_t)std::max({
        cfg.ssm_inner_size * 3 * cfg.n_embed,  // QKV: 6144 × 1024
        cfg.n_head * cfg.head_dim * 2 * cfg.n_embed,  // Q+gate: 4096 × 1024
        cfg.n_ff * cfg.n_embed                  // FFN: 3584 × 1024
    });
    prefill_temp_w = static_cast<half*>(a(max_weight_elems * sizeof(half)));

    // Batch FFN buffers: [max_tokens, n_ff]
    // Cannot alias ffn_up/ffn_out with proj_qkv: full attention layers use ffn_up for V
    // projection and ffn_out for deinterleaved gate while proj_qkv still holds Q+gate.
    prefill_ffn_gate = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));
    prefill_ffn_up   = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));
    prefill_ffn_out  = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));

    // Batch projection buffers for DeltaNet QKV/gate and full attention Q/K/V
    size_t qkv_dim = cfg.ssm_inner_size * 3;  // 6144 (DeltaNet) — also big enough for attn Q+gate (4096)
    prefill_proj_qkv  = static_cast<half*>(a(max_tokens * qkv_dim * sizeof(half)));
    prefill_proj_gate = static_cast<half*>(a(max_tokens * cfg.ssm_inner_size * sizeof(half)));

    // Batch DeltaNet gate/beta buffers and pre-allocated token IDs
    prefill_dn_gate = static_cast<float*>(a(max_tokens * cfg.ssm_n_heads * sizeof(float)));
    prefill_dn_beta = static_cast<float*>(a(max_tokens * cfg.ssm_n_heads * sizeof(float)));
    d_prefill_tokens = static_cast<int*>(a(max_tokens * sizeof(int)));
}

// ============================================================
// Batch prefill allocation (B independent sequences)
// ============================================================

void InferenceState::allocate_batch_prefill(const ModelConfig& cfg, CudaAllocator& alloc,
                                             int max_total_tokens, int max_seqs, bool f32_path) {
    // Allocate prefill buffers for max_total_tokens
    allocate_prefill(cfg, alloc, max_total_tokens, f32_path);
    max_batch_seqs = max_seqs;

    auto a = [&](size_t bytes) -> void* { return alloc.alloc(bytes); };

    // Count DeltaNet layers
    n_batch_dn_layers = 0;
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        if (!cfg.is_full_attention_layer(i)) n_batch_dn_layers++;
    }

    // Allocate B independent DeltaNet S states: [max_seqs * n_dn_layers * n_heads * dk * dv]
    int n_heads = cfg.ssm_n_heads;
    int dk = cfg.ssm_state_size;
    int dv = cfg.ssm_state_size;
    size_t S_per_layer = (size_t)n_heads * dk * dv * sizeof(float);
    size_t S_total = (size_t)max_seqs * n_batch_dn_layers * S_per_layer;
    batch_dn_S = static_cast<float*>(a(S_total));

    // Allocate B independent conv states: [max_seqs * n_dn_layers * (conv_kernel-1) * qkv_dim]
    int qkv_dim = cfg.ssm_inner_size * 3;
    int conv_km1 = cfg.ssm_conv_kernel - 1;  // 3
    size_t conv_per_layer = (size_t)conv_km1 * qkv_dim * sizeof(float);
    size_t conv_total = (size_t)max_seqs * n_batch_dn_layers * conv_per_layer;
    batch_dn_conv = static_cast<float*>(a(conv_total));

    // --- Chunkwise DeltaNet intermediate buffers ---
    int max_seq_len = max_total_tokens / max_seqs;  // L
    chunk_NT_max = (max_seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Cumulative gate: [max_tokens, n_heads]
    chunk_gate_cumul = static_cast<float*>(a(max_total_tokens * n_heads * sizeof(float)));

    // W, U: [max_tokens, ssm_inner] (same layout as K, V)
    chunk_W = static_cast<half*>(a(max_total_tokens * cfg.ssm_inner_size * sizeof(half)));
    chunk_U = static_cast<half*>(a(max_total_tokens * cfg.ssm_inner_size * sizeof(half)));

    // h_states: [max_seqs * NT_max * n_heads, dk, dv] FP32
    size_t h_total = (size_t)max_seqs * chunk_NT_max * n_heads * dk * dv * sizeof(float);
    chunk_h_states = static_cast<float*>(a(h_total));

    // v_new: [max_tokens, ssm_inner]
    chunk_v_new = static_cast<half*>(a(max_total_tokens * cfg.ssm_inner_size * sizeof(half)));

    size_t chunk_total = max_total_tokens * n_heads * sizeof(float)  // gate_cumul
        + 2 * max_total_tokens * cfg.ssm_inner_size * sizeof(half)   // W + U
        + h_total                                                     // h_states
        + max_total_tokens * cfg.ssm_inner_size * sizeof(half);       // v_new

    fprintf(stderr, "Batch prefill: max_seqs=%d, max_tokens=%d\n", max_seqs, max_total_tokens);
    fprintf(stderr, "  DeltaNet S states: %.1f MB (%d seqs × %d layers × %.0f KB/layer)\n",
           S_total / 1024.0 / 1024.0, max_seqs, n_batch_dn_layers, S_per_layer / 1024.0);
    fprintf(stderr, "  Conv states: %.1f MB\n", conv_total / 1024.0 / 1024.0);
    fprintf(stderr, "  Chunkwise buffers: %.1f MB (NT_max=%d)\n", chunk_total / 1024.0 / 1024.0, chunk_NT_max);
}

// ============================================================
// MTP allocation
// ============================================================

void InferenceState::allocate_mtp(const ModelConfig& cfg, CudaAllocator& alloc, int max_seq) {
    auto a = [&](size_t bytes) -> void* { return alloc.alloc(bytes); };

    // Hidden state buffer (saved from main forward for MTP input)
    mtp_hidden = static_cast<half*>(a(cfg.n_embed * sizeof(half)));

    // Concat buffer for FC input: [norm_embed; norm_hidden] = [2 * n_embed]
    mtp_concat = static_cast<half*>(a(2 * cfg.n_embed * sizeof(half)));

    // MTP KV cache (1 attention layer)
    mtp_kv_cache.max_seq = max_seq;
    mtp_kv_cache.n_kv_heads = cfg.n_head_kv;
    mtp_kv_cache.head_dim = cfg.head_dim;
    size_t kv_bytes = (size_t)max_seq * cfg.n_head_kv * cfg.head_dim * sizeof(half);
    mtp_kv_cache.k_cache = static_cast<half*>(a(kv_bytes));
    mtp_kv_cache.v_cache = static_cast<half*>(a(kv_bytes));
    GWEN_CHECK_CUDA(cudaMemset(mtp_kv_cache.k_cache, 0, kv_bytes));
    GWEN_CHECK_CUDA(cudaMemset(mtp_kv_cache.v_cache, 0, kv_bytes));

    // Device scalars for MTP (token_id + pos, adjacent)
    d_mtp_token = static_cast<int*>(a(2 * sizeof(int)));
    d_mtp_pos = d_mtp_token + 1;

    // Token A hidden state backup (for reject path: swap into mtp_hidden)
    mtp_hidden_b = static_cast<half*>(a(cfg.n_embed * sizeof(half)));

    // Activation replay buffers (replaces 19.3 MB checkpoint with ~578 KB cache)
    n_dn_layers = (int)deltanet_states.size();
    if (n_dn_layers > 0) {
        const auto& s0 = deltanet_states[0];
        int dk = s0.state_size, dv = s0.state_size;
        int n_heads = s0.n_heads;
        int qkv_dim = s0.qkv_dim;

        // S snapshots: one per DeltaNet layer, saved after token A in 2tok loop
        size_t S_bytes_per_layer = (size_t)n_heads * dk * dv * sizeof(float);
        for (int i = 0; i < n_dn_layers; i++) {
            dn_S_snapshot.push_back(static_cast<float*>(a(S_bytes_per_layer)));
        }

        // conv_state[0] per layer: [qkv_dim] float
        size_t conv_replay_bytes = (size_t)n_dn_layers * qkv_dim * sizeof(float);
        dn_replay_conv_row = static_cast<float*>(a(conv_replay_bytes));

        // Device pointer array for batched conv undo kernel
        d_conv_ptrs = static_cast<float**>(a(n_dn_layers * sizeof(float*)));
        std::vector<float*> h_conv_ptrs(n_dn_layers);
        for (int i = 0; i < n_dn_layers; i++) {
            h_conv_ptrs[i] = deltanet_states[i].conv_state;
        }
        GWEN_CHECK_CUDA(cudaMemcpy(d_conv_ptrs, h_conv_ptrs.data(),
                                    n_dn_layers * sizeof(float*), cudaMemcpyHostToDevice));

        size_t replay_total = n_dn_layers * S_bytes_per_layer + conv_replay_bytes;
        fprintf(stderr, "MTP state allocated (KV cache: %.1f MB, replay buffers: %.1f KB)\n",
               2 * kv_bytes / 1024.0 / 1024.0, replay_total / 1024.0);
    } else {
        fprintf(stderr, "MTP state allocated (KV cache: %.1f MB, no DeltaNet layers)\n",
               2 * kv_bytes / 1024.0 / 1024.0);
    }
}

// ============================================================
// Batch-2 scratch allocation (token B buffers for 2-token verify)
// ============================================================

void InferenceState::allocate_batch2(const ModelConfig& cfg, CudaAllocator& alloc) {
    auto a = [&](size_t bytes) -> void* { return alloc.alloc(bytes); };

    b2_buf_a     = static_cast<half*>(a(cfg.n_embed * sizeof(half)));
    b2_buf_b     = static_cast<half*>(a(cfg.n_embed * sizeof(half)));
    b2_x_norm    = static_cast<half*>(a(cfg.n_embed * sizeof(half)));

    int qkv_dim = cfg.ssm_inner_size * 3;
    b2_qkv       = static_cast<half*>(a(qkv_dim * sizeof(half)));
    b2_gate_z    = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    b2_attn_out  = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    b2_gated_out = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));

    int q_dim = cfg.n_head * cfg.head_dim * 2;
    b2_fa_q      = static_cast<half*>(a(q_dim * sizeof(half)));
    b2_fa_k      = static_cast<half*>(a(cfg.n_head_kv * cfg.head_dim * sizeof(half)));
    b2_fa_v      = static_cast<half*>(a(cfg.n_head_kv * cfg.head_dim * sizeof(half)));

    b2_ffn_gate  = static_cast<half*>(a(cfg.n_ff * sizeof(half)));
    b2_ffn_up    = static_cast<half*>(a(cfg.n_ff * sizeof(half)));
    b2_ffn_out   = static_cast<half*>(a(cfg.n_ff * sizeof(half)));

    b2_logits_h  = static_cast<half*>(a(cfg.n_vocab * sizeof(half)));
    b2_logits_f  = static_cast<float*>(a(cfg.n_vocab * sizeof(float)));

    int max_q8_dim = std::max(cfg.n_ff, std::max(cfg.n_embed, cfg.ssm_inner_size));
    int q8_blocks = (max_q8_dim + 31) / 32;
    b2_x_q8_a = a(q8_blocks * 36);
    b2_x_q8_b = a(q8_blocks * 36);

    // b2_d_argmax is adjacent to d_argmax_token (allocated as 2 ints in allocate())
    b2_d_argmax = d_argmax_token + 1;
    b2_argmax_partial_max = static_cast<float*>(a(ARGMAX_BLOCKS * sizeof(float)));
    b2_argmax_partial_idx = static_cast<int*>(a(ARGMAX_BLOCKS * sizeof(int)));

    batch2_allocated = true;
    fprintf(stderr, "Batch-2 verify buffers allocated\n");
}

// ============================================================
// DeltaNet state checkpoint / restore (for speculative decode rollback)
// ============================================================

// ============================================================
// Conv1d state undo kernel (for speculative decode reject)
// ============================================================

__global__ void __launch_bounds__(256)
kernel_conv1d_undo_batch(
    float** __restrict__ conv_ptrs,        // [n_layers] conv_state pointers
    const float* __restrict__ saved_rows,  // [n_layers, dim] saved row 0
    int dim, int n_layers)
{
    int layer = blockIdx.y;
    if (layer >= n_layers) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    float* conv = conv_ptrs[layer];

    // Reverse shift: current [row2, x_A, x_B] → restore [saved_row0, row2, x_A]
    float row2 = conv[0 * dim + idx];
    float x_A  = conv[1 * dim + idx];

    conv[0 * dim + idx] = saved_rows[layer * dim + idx];
    conv[1 * dim + idx] = row2;
    conv[2 * dim + idx] = x_A;
}

void InferenceState::undo_deltanet_token_b(cudaStream_t stream) {
    const auto& s0 = deltanet_states[0];
    int qkv_dim = s0.qkv_dim;

    // Restore S from snapshots (exact — saved after token A's decode)
    for (int i = 0; i < n_dn_layers; i++) {
        auto& state = deltanet_states[i];
        size_t S_bytes = (size_t)state.n_heads * state.state_size * state.state_size * sizeof(float);
        GWEN_CHECK_CUDA(cudaMemcpyAsync(state.S, dn_S_snapshot[i], S_bytes,
                                          cudaMemcpyDeviceToDevice, stream));
    }

    // Undo conv1d state shift: 1 kernel launch
    int conv_blocks = (qkv_dim + 255) / 256;
    dim3 grid_c(conv_blocks, n_dn_layers);
    kernel_conv1d_undo_batch<<<grid_c, 256, 0, stream>>>(
        d_conv_ptrs, dn_replay_conv_row,
        qkv_dim, n_dn_layers);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// Reset all recurrent state for independent sequence processing
// ============================================================

void InferenceState::reset_state() {
    for (auto& state : deltanet_states) {
        size_t S_bytes = (size_t)state.n_heads * state.state_size * state.state_size * sizeof(float);
        size_t conv_bytes = (size_t)(state.conv_kernel - 1) * state.qkv_dim * sizeof(float);
        GWEN_CHECK_CUDA(cudaMemset(state.S, 0, S_bytes));
        GWEN_CHECK_CUDA(cudaMemset(state.conv_state, 0, conv_bytes));
    }
    for (auto& cache : kv_caches) {
        size_t kv_bytes = (size_t)cache.max_seq * cache.n_kv_heads * cache.head_dim * sizeof(half);
        GWEN_CHECK_CUDA(cudaMemset(cache.k_cache, 0, kv_bytes));
        GWEN_CHECK_CUDA(cudaMemset(cache.v_cache, 0, kv_bytes));
    }
    if (mtp_kv_cache.k_cache) {
        size_t kv_bytes = (size_t)mtp_kv_cache.max_seq * mtp_kv_cache.n_kv_heads * mtp_kv_cache.head_dim * sizeof(half);
        GWEN_CHECK_CUDA(cudaMemset(mtp_kv_cache.k_cache, 0, kv_bytes));
        GWEN_CHECK_CUDA(cudaMemset(mtp_kv_cache.v_cache, 0, kv_bytes));
    }
    pos = 0;
    mtp_pos = 0;
}

// Forward declaration for kernel used in extract_hidden (defined before forward_prefill)
__global__ void kernel_f32_to_half(const float* __restrict__ in, half* __restrict__ out, int total);

// ============================================================
// Extract hidden states for all tokens (prefill layers only)
// ============================================================

void InferenceState::extract_hidden(Model& model, const std::vector<int>& tokens, void* output_host) {
    GWEN_CHECK(prefill_x_f32 != nullptr, "extract_hidden requires F32 prefill buffers (not available in batch-only mode)");
    reset_state();
    forward_prefill(model, tokens);
    // After forward_prefill, prefill_x_f32 holds [N, n_embed] hidden states (F32)
    // Convert to FP16 for output
    int N = (int)tokens.size();
    int total = N * model.config.n_embed;
    int blocks = (total + 255) / 256;
    kernel_f32_to_half<<<blocks, 256, 0, compute_stream>>>(prefill_x_f32, prefill_x, total);
    GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    size_t bytes = (size_t)total * sizeof(half);
    GWEN_CHECK_CUDA(cudaMemcpy(output_host, prefill_x, bytes, cudaMemcpyDeviceToHost));
}

// ============================================================
// Forward pass body — all GPU work (CUDA graph capturable)
// ============================================================
// Reads token_id from d_token_id and pos from d_pos (device memory).
// Buffer convention: buf_a holds hidden state at layer start/end.

void InferenceState::forward_body(Model& model, cudaStream_t s) {
    const auto& cfg = model.config;
    const float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);

    // 1. Embedding lookup → buf_a (reads token_id from device memory)
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      d_token_id, buf_a, cfg.n_embed, s);

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // 2. Process each layer
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            const auto& w = layer.deltanet;
            auto& state = deltanet_states[dn_state_idx++];

            // Fused RMSNorm + Q8_1 quantize (also writes FP16 x_norm for gate/beta computation)
            gwen_rmsnorm_quantize_q8_1(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_q8_a, x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_gemv_dp4a(w.attn_qkv.device_data, x_q8_a, qkv,
                      w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type, s);
            gwen_gemv_dp4a(w.attn_gate.device_data, x_q8_a, gate_z,
                      w.attn_gate.shape[1], w.attn_gate.shape[0], w.attn_gate.type, s);

            int qkv_dim = cfg.ssm_inner_size * 3;
            int conv_blocks = (qkv_dim + 255) / 256;
            kernel_conv1d_silu<<<conv_blocks, 256, 0, s>>>(
                qkv, qkv, state.conv_state,
                static_cast<const float*>(w.ssm_conv1d.device_data),
                qkv_dim, cfg.ssm_conv_kernel);

            half* q = qkv;
            half* k = qkv + cfg.ssm_inner_size;
            half* v = qkv + 2 * cfg.ssm_inner_size;

            // Fused: L2-norm(Q,K) + gate/beta + S update
            kernel_deltanet_fused<<<cfg.ssm_n_heads, 128, 65536, s>>>(
                state.S, q, k, v, x_norm,
                w.ssm_alpha.device_data, w.ssm_beta.device_data,
                static_cast<const float*>(w.ssm_a.device_data),
                static_cast<const float*>(w.ssm_dt_bias.device_data),
                attn_out, q_scale,
                cfg.ssm_n_heads, cfg.ssm_state_size, cfg.n_embed);

            // Fused gated RMSNorm + Q8_1 quantize
            kernel_gated_rmsnorm_quantize_q8_1<<<cfg.ssm_n_heads, 128, 0, s>>>(
                attn_out,
                static_cast<const float*>(w.ssm_norm.device_data),
                gate_z, static_cast<block_q8_1*>(x_q8_b),
                cfg.ssm_n_heads, cfg.ssm_state_size, cfg.rms_norm_eps);
            gwen_gemv_dp4a_residual(w.ssm_out.device_data, x_q8_b, buf_b, buf_a,
                      w.ssm_out.shape[1], w.ssm_out.shape[0], w.ssm_out.type, s);

            // Fused RMSNorm + Q8_1 quantize (no FP16 output needed)
            gwen_rmsnorm_quantize_q8_1(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_gemv_dp4a(w.ffn_gate.device_data, x_q8_a, ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type, s);
            gwen_gemv_dp4a(w.ffn_up.device_data, x_q8_a, ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type, s);
            // Fused SwiGLU + Q8_1 quantize (skips FP16 intermediate)
            gwen_swiglu_quantize_q8_1(ffn_gate, ffn_up, x_q8_b, cfg.n_ff, s);
            gwen_gemv_dp4a_residual(w.ffn_down.device_data, x_q8_b, buf_a, buf_b,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type, s);

        } else {
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            // Fused RMSNorm + Q8_1 quantize (no FP16 output needed)
            gwen_rmsnorm_quantize_q8_1(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_gemv_dp4a(w.attn_q.device_data, x_q8_a, qkv,
                      w.attn_q.shape[1], w.attn_q.shape[0], w.attn_q.type, s);

            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                int deint_blocks = (attn_dim + 255) / 256;
                kernel_deinterleave_qgate<<<deint_blocks, 256, 0, s>>>(
                    qkv, fa_q, gated_out,
                    cfg.n_head, cfg.head_dim);
            }

            gwen_gemv_dp4a(w.attn_k.device_data, x_q8_a, fa_k,
                      w.attn_k.shape[1], w.attn_k.shape[0], w.attn_k.type, s);
            gwen_gemv_dp4a(w.attn_v.device_data, x_q8_a, fa_v,
                      w.attn_v.shape[1], w.attn_v.shape[0], w.attn_v.type, s);

            gwen_rmsnorm_batched_f32w(fa_q, static_cast<const float*>(w.attn_q_norm.device_data),
                                      fa_q, cfg.n_head, cfg.head_dim, cfg.rms_norm_eps, s);
            gwen_rmsnorm_batched_f32w(fa_k, static_cast<const float*>(w.attn_k_norm.device_data),
                                      fa_k, cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps, s);

            gwen_rope(fa_q, fa_k,
                      cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                      d_pos, cfg.rope_theta, cfg.rope_sections, cfg.rope_dim, s);

            {
                int kv_dim = cfg.n_head_kv * cfg.head_dim;
                int kv_blocks = (kv_dim + 255) / 256;
                kernel_kv_cache_store<<<kv_blocks, 256, 0, s>>>(
                    cache.k_cache, cache.v_cache, fa_k, fa_v, d_pos, kv_dim);
            }

            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            kernel_gqa_attention_decode<<<cfg.n_head, 256, 0, s>>>(
                fa_q, cache.k_cache, cache.v_cache,
                attn_out, attn_scores, d_pos,
                cfg.n_head, cfg.n_head_kv, cfg.head_dim, max_seq_alloc, scale);

            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                gwen_sigmoid_mul(attn_out, gated_out, gated_out, attn_dim, s);
            }

            // Quantize gated_out for output projection (fused residual add)
            gwen_quantize_q8_1(gated_out, x_q8_b, cfg.n_head * cfg.head_dim, s);
            gwen_gemv_dp4a_residual(w.attn_output.device_data, x_q8_b, buf_b, buf_a,
                      w.attn_output.shape[1], w.attn_output.shape[0], w.attn_output.type, s);

            // Fused RMSNorm + Q8_1 quantize (no FP16 output needed)
            gwen_rmsnorm_quantize_q8_1(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_gemv_dp4a(w.ffn_gate.device_data, x_q8_a, ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type, s);
            gwen_gemv_dp4a(w.ffn_up.device_data, x_q8_a, ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type, s);
            // Fused SwiGLU + Q8_1 quantize (skips FP16 intermediate)
            gwen_swiglu_quantize_q8_1(ffn_gate, ffn_up, x_q8_b, cfg.n_ff, s);
            gwen_gemv_dp4a_residual(w.ffn_down.device_data, x_q8_b, buf_a, buf_b,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type, s);
        }
    }

    // 3. Save hidden state for MTP (before LM head destroys it)
    // buf_a holds the final hidden state after all 24 layers.
    // Copy to mtp_hidden for use by forward_mtp_body().
    // This is a 2KB D2D copy — negligible cost, CUDA graph capturable.
    if (mtp_hidden) {
        GWEN_CHECK_CUDA(cudaMemcpyAsync(mtp_hidden, buf_a, cfg.n_embed * sizeof(half),
                                          cudaMemcpyDeviceToDevice, s));
    }

    // 4. Fused RMSNorm + Q8_1 quantize + LM Head + argmax
    gwen_rmsnorm_quantize_q8_1(buf_a, static_cast<const float*>(model.output_norm.device_data),
                      x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);
    gwen_gemv_dp4a(model.token_embd.device_data, x_q8_a, logits_h,
              cfg.n_vocab, cfg.n_embed, model.token_embd.type, s);

    int logit_blocks = (cfg.n_vocab + 255) / 256;
    kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, cfg.n_vocab);

    kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(logits_f, argmax_partial_max, argmax_partial_idx, cfg.n_vocab);
    kernel_argmax_reduce<<<1, 256, 0, s>>>(argmax_partial_max, argmax_partial_idx, d_argmax_token, ARGMAX_BLOCKS);
}

// ============================================================
// 2-token forward body — batch GEMV reads weights once for 2 tokens
// ============================================================
// Token A: d_token_id[0], d_pos[0] → buf_a (hidden) → d_argmax_token (result)
// Token B: d_token_id[1], d_pos[1] → b2_buf_a (hidden) → b2_d_argmax (result)
// d_token_id layout: [tok_a, pos_a, tok_b, pos_b] (4 ints, one cudaMemcpy)

void InferenceState::forward_body_2tok(Model& model, cudaStream_t s) {
    const auto& cfg = model.config;
    const float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);

    // Aliases for token A (existing buffers) and token B (b2_ buffers)
    // Token A hidden: buf_a/buf_b (pointer-swap residual), x_norm, qkv, gate_z, ...
    // Token B hidden: b2_buf_a/b2_buf_b, b2_x_norm, b2_qkv, b2_gate_z, ...

    // We use d_token_id / d_pos for token A, and a separate pair for token B.
    // Layout: d_token_id[0]=tok_a, d_token_id[1]=pos_a (d_pos), d_token_id[2]=tok_b, d_token_id[3]=pos_b
    int* d_token_b = d_token_id + 2;
    int* d_pos_b   = d_token_id + 3;

    // 1. Embedding lookup for both tokens
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      d_token_id, buf_a, cfg.n_embed, s);
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      d_token_b, b2_buf_a, cfg.n_embed, s);

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // 2. Process each layer — token A first (state update), then token B, batch GEMV where possible
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            const auto& w = layer.deltanet;
            int dn_idx = dn_state_idx++;
            auto& state = deltanet_states[dn_idx];

            // --- Batch2 RMSNorm + Q8_1 quantize for both tokens ---
            gwen_rmsnorm_quantize_q8_1_batch2(
                              buf_a, b2_buf_a,
                              static_cast<const float*>(w.attn_norm.device_data),
                              x_q8_a, b2_x_q8_a, x_norm, b2_x_norm,
                              cfg.n_embed, cfg.rms_norm_eps, s);

            // --- Batch2 GEMV: QKV and gate projections (read weights once) ---
            gwen_gemv_dp4a_batch2(w.attn_qkv.device_data, x_q8_a, b2_x_q8_a,
                      qkv, b2_qkv,
                      w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type, s);
            gwen_gemv_dp4a_batch2(w.attn_gate.device_data, x_q8_a, b2_x_q8_a,
                      gate_z, b2_gate_z,
                      w.attn_gate.shape[1], w.attn_gate.shape[0], w.attn_gate.type, s);

            int qkv_dim = cfg.ssm_inner_size * 3;
            int conv_blocks = (qkv_dim + 255) / 256;

            // --- Conv1D for token A ---
            kernel_conv1d_silu<<<conv_blocks, 256, 0, s>>>(
                qkv, qkv, state.conv_state,
                static_cast<const float*>(w.ssm_conv1d.device_data),
                qkv_dim, cfg.ssm_conv_kernel);

            // --- Save conv_state[0] right after conv1d A (before B shifts it) ---
            if (dn_replay_conv_row) {
                GWEN_CHECK_CUDA(cudaMemcpyAsync(
                    dn_replay_conv_row + dn_idx * qkv_dim,
                    state.conv_state,
                    qkv_dim * sizeof(float),
                    cudaMemcpyDeviceToDevice, s));
            }

            // --- Fork: conv1d B on overlap_stream (overlaps with deltanet A) ---
            GWEN_CHECK_CUDA(cudaEventRecord(ev_conv_done, s));
            GWEN_CHECK_CUDA(cudaStreamWaitEvent(overlap_stream, ev_conv_done));
            kernel_conv1d_silu<<<conv_blocks, 256, 0, overlap_stream>>>(
                b2_qkv, b2_qkv, state.conv_state,
                static_cast<const float*>(w.ssm_conv1d.device_data),
                qkv_dim, cfg.ssm_conv_kernel);
            GWEN_CHECK_CUDA(cudaEventRecord(ev_overlap_done, overlap_stream));

            // --- Join: wait for conv1d B before fused 2-token deltanet ---
            GWEN_CHECK_CUDA(cudaStreamWaitEvent(s, ev_overlap_done));

            // --- Fused 2-token DeltaNet: A + snapshot + B in one kernel launch ---
            // S stays in shared memory across both tokens. Snapshot is written
            // between A and B, overlapped with B's Phase 1-2 (L2-norm + gate/beta).
            half *qa = qkv, *ka = qkv + cfg.ssm_inner_size, *va = qkv + 2 * cfg.ssm_inner_size;
            half *qb = b2_qkv, *kb = b2_qkv + cfg.ssm_inner_size, *vb = b2_qkv + 2 * cfg.ssm_inner_size;
            float* snap = (dn_idx < (int)dn_S_snapshot.size()) ? dn_S_snapshot[dn_idx] : nullptr;

            kernel_deltanet_fused_2tok<<<cfg.ssm_n_heads, 128, 65536, s>>>(
                state.S,
                qa, ka, va, x_norm,
                qb, kb, vb, b2_x_norm,
                w.ssm_alpha.device_data, w.ssm_beta.device_data,
                static_cast<const float*>(w.ssm_a.device_data),
                static_cast<const float*>(w.ssm_dt_bias.device_data),
                attn_out, b2_attn_out, snap,
                q_scale, cfg.ssm_n_heads, cfg.ssm_state_size, cfg.n_embed);

            // --- Batch2 Gated RMSNorm + Q8_1 for both tokens ---
            {
                dim3 grid(cfg.ssm_n_heads, 2);
                kernel_gated_rmsnorm_quantize_q8_1_batch2<<<grid, 128, 0, s>>>(
                    attn_out, b2_attn_out,
                    static_cast<const float*>(w.ssm_norm.device_data),
                    gate_z, b2_gate_z,
                    static_cast<block_q8_1*>(x_q8_b), static_cast<block_q8_1*>(b2_x_q8_b),
                    cfg.ssm_n_heads, cfg.ssm_state_size, cfg.rms_norm_eps);
            }

            // --- Batch2 GEMV: ssm_out with residual ---
            gwen_gemv_dp4a_residual_batch2(w.ssm_out.device_data, x_q8_b, b2_x_q8_b,
                      buf_b, b2_buf_b, buf_a, b2_buf_a,
                      w.ssm_out.shape[1], w.ssm_out.shape[0], w.ssm_out.type, s);

            // --- Batch2 RMSNorm + Q8_1 for both tokens (FFN input) ---
            gwen_rmsnorm_quantize_q8_1_batch2(
                              buf_b, b2_buf_b,
                              static_cast<const float*>(w.post_attn_norm.device_data),
                              x_q8_a, b2_x_q8_a, nullptr, nullptr,
                              cfg.n_embed, cfg.rms_norm_eps, s);

            // --- Batch2 GEMV: FFN gate and up ---
            gwen_gemv_dp4a_batch2(w.ffn_gate.device_data, x_q8_a, b2_x_q8_a,
                      ffn_gate, b2_ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type, s);
            gwen_gemv_dp4a_batch2(w.ffn_up.device_data, x_q8_a, b2_x_q8_a,
                      ffn_up, b2_ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type, s);

            // --- Batch2 SwiGLU + Q8_1 for both tokens ---
            gwen_swiglu_quantize_q8_1_batch2(ffn_gate, b2_ffn_gate,
                      ffn_up, b2_ffn_up,
                      x_q8_b, b2_x_q8_b, cfg.n_ff, s);

            // --- Batch2 GEMV: FFN down with residual ---
            gwen_gemv_dp4a_residual_batch2(w.ffn_down.device_data, x_q8_b, b2_x_q8_b,
                      buf_a, b2_buf_a, buf_b, b2_buf_b,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type, s);

        } else {
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            // --- Batch2 RMSNorm + Q8_1 for both tokens ---
            gwen_rmsnorm_quantize_q8_1_batch2(
                              buf_a, b2_buf_a,
                              static_cast<const float*>(w.attn_norm.device_data),
                              x_q8_a, b2_x_q8_a, nullptr, nullptr,
                              cfg.n_embed, cfg.rms_norm_eps, s);

            // --- Batch2 GEMV: Q, K, V projections ---
            gwen_gemv_dp4a_batch2(w.attn_q.device_data, x_q8_a, b2_x_q8_a,
                      qkv, b2_qkv,
                      w.attn_q.shape[1], w.attn_q.shape[0], w.attn_q.type, s);
            gwen_gemv_dp4a_batch2(w.attn_k.device_data, x_q8_a, b2_x_q8_a,
                      fa_k, b2_fa_k,
                      w.attn_k.shape[1], w.attn_k.shape[0], w.attn_k.type, s);
            gwen_gemv_dp4a_batch2(w.attn_v.device_data, x_q8_a, b2_x_q8_a,
                      fa_v, b2_fa_v,
                      w.attn_v.shape[1], w.attn_v.shape[0], w.attn_v.type, s);

            // --- Token A: deinterleave, norms, RoPE, KV store, attention ---
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                int deint_blocks = (attn_dim + 255) / 256;
                kernel_deinterleave_qgate<<<deint_blocks, 256, 0, s>>>(
                    qkv, fa_q, gated_out, cfg.n_head, cfg.head_dim);
            }
            gwen_rmsnorm_batched_f32w(fa_q, static_cast<const float*>(w.attn_q_norm.device_data),
                                      fa_q, cfg.n_head, cfg.head_dim, cfg.rms_norm_eps, s);
            gwen_rmsnorm_batched_f32w(fa_k, static_cast<const float*>(w.attn_k_norm.device_data),
                                      fa_k, cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps, s);
            gwen_rope(fa_q, fa_k, cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                      d_pos, cfg.rope_theta, cfg.rope_sections, cfg.rope_dim, s);
            {
                int kv_dim = cfg.n_head_kv * cfg.head_dim;
                int kv_blocks = (kv_dim + 255) / 256;
                kernel_kv_cache_store<<<kv_blocks, 256, 0, s>>>(
                    cache.k_cache, cache.v_cache, fa_k, fa_v, d_pos, kv_dim);
            }
            {
                float scale = 1.0f / sqrtf((float)cfg.head_dim);
                kernel_gqa_attention_decode<<<cfg.n_head, 256, 0, s>>>(
                    fa_q, cache.k_cache, cache.v_cache,
                    attn_out, attn_scores, d_pos,
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim, max_seq_alloc, scale);
            }
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                gwen_sigmoid_mul(attn_out, gated_out, gated_out, attn_dim, s);
            }

            // --- Token B: deinterleave, norms, RoPE, KV store, attention ---
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                int deint_blocks = (attn_dim + 255) / 256;
                kernel_deinterleave_qgate<<<deint_blocks, 256, 0, s>>>(
                    b2_qkv, b2_fa_q, b2_gated_out, cfg.n_head, cfg.head_dim);
            }
            gwen_rmsnorm_batched_f32w(b2_fa_q, static_cast<const float*>(w.attn_q_norm.device_data),
                                      b2_fa_q, cfg.n_head, cfg.head_dim, cfg.rms_norm_eps, s);
            gwen_rmsnorm_batched_f32w(b2_fa_k, static_cast<const float*>(w.attn_k_norm.device_data),
                                      b2_fa_k, cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps, s);
            gwen_rope(b2_fa_q, b2_fa_k, cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                      d_pos_b, cfg.rope_theta, cfg.rope_sections, cfg.rope_dim, s);
            {
                int kv_dim = cfg.n_head_kv * cfg.head_dim;
                int kv_blocks = (kv_dim + 255) / 256;
                kernel_kv_cache_store<<<kv_blocks, 256, 0, s>>>(
                    cache.k_cache, cache.v_cache, b2_fa_k, b2_fa_v, d_pos_b, kv_dim);
            }
            {
                float scale = 1.0f / sqrtf((float)cfg.head_dim);
                kernel_gqa_attention_decode<<<cfg.n_head, 256, 0, s>>>(
                    b2_fa_q, cache.k_cache, cache.v_cache,
                    b2_attn_out, attn_scores, d_pos_b,
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim, max_seq_alloc, scale);
            }
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                gwen_sigmoid_mul(b2_attn_out, b2_gated_out, b2_gated_out, attn_dim, s);
            }

            // --- Batch2 quantize gated outputs + batch2 output proj with residual ---
            gwen_quantize_q8_1_batch2(gated_out, b2_gated_out,
                      x_q8_b, b2_x_q8_b, cfg.n_head * cfg.head_dim, s);
            gwen_gemv_dp4a_residual_batch2(w.attn_output.device_data, x_q8_b, b2_x_q8_b,
                      buf_b, b2_buf_b, buf_a, b2_buf_a,
                      w.attn_output.shape[1], w.attn_output.shape[0], w.attn_output.type, s);

            // --- Batch2 RMSNorm + Q8_1 for FFN ---
            gwen_rmsnorm_quantize_q8_1_batch2(
                              buf_b, b2_buf_b,
                              static_cast<const float*>(w.post_attn_norm.device_data),
                              x_q8_a, b2_x_q8_a, nullptr, nullptr,
                              cfg.n_embed, cfg.rms_norm_eps, s);

            // --- Batch2 GEMV: FFN ---
            gwen_gemv_dp4a_batch2(w.ffn_gate.device_data, x_q8_a, b2_x_q8_a,
                      ffn_gate, b2_ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type, s);
            gwen_gemv_dp4a_batch2(w.ffn_up.device_data, x_q8_a, b2_x_q8_a,
                      ffn_up, b2_ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type, s);
            gwen_swiglu_quantize_q8_1_batch2(ffn_gate, b2_ffn_gate,
                      ffn_up, b2_ffn_up,
                      x_q8_b, b2_x_q8_b, cfg.n_ff, s);
            gwen_gemv_dp4a_residual_batch2(w.ffn_down.device_data, x_q8_b, b2_x_q8_b,
                      buf_a, b2_buf_a, buf_b, b2_buf_b,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type, s);
        }
    }

    // 3. Save hidden states for MTP
    // mtp_hidden  ← token B's hidden (used on accept path)
    // mtp_hidden_b ← token A's hidden (swapped in on reject path)
    if (mtp_hidden) {
        GWEN_CHECK_CUDA(cudaMemcpyAsync(mtp_hidden, b2_buf_a, cfg.n_embed * sizeof(half),
                                          cudaMemcpyDeviceToDevice, s));
    }
    if (mtp_hidden_b) {
        GWEN_CHECK_CUDA(cudaMemcpyAsync(mtp_hidden_b, buf_a, cfg.n_embed * sizeof(half),
                                          cudaMemcpyDeviceToDevice, s));
    }

    // 4. LM head for both tokens — batch2 GEMV for the big 248K×1024 projection
    gwen_rmsnorm_quantize_q8_1(buf_a, static_cast<const float*>(model.output_norm.device_data),
                      x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);
    gwen_rmsnorm_quantize_q8_1(b2_buf_a, static_cast<const float*>(model.output_norm.device_data),
                      b2_x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);
    gwen_gemv_dp4a_batch2(model.token_embd.device_data, x_q8_a, b2_x_q8_a,
              logits_h, b2_logits_h,
              cfg.n_vocab, cfg.n_embed, model.token_embd.type, s);

    int logit_blocks = (cfg.n_vocab + 255) / 256;
    kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, cfg.n_vocab);
    kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(b2_logits_h, b2_logits_f, cfg.n_vocab);

    kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(logits_f, argmax_partial_max, argmax_partial_idx, cfg.n_vocab);
    kernel_argmax_reduce<<<1, 256, 0, s>>>(argmax_partial_max, argmax_partial_idx, d_argmax_token, ARGMAX_BLOCKS);
    kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(b2_logits_f, b2_argmax_partial_max, b2_argmax_partial_idx, cfg.n_vocab);
    kernel_argmax_reduce<<<1, 256, 0, s>>>(b2_argmax_partial_max, b2_argmax_partial_idx, b2_d_argmax, ARGMAX_BLOCKS);
}

std::pair<int,int> InferenceState::forward_2tok(Model& model, int token_id_a, int token_id_b) {
    // Pack: [tok_a, pos_a, tok_b, pos_b] — 4 ints, one memcpy
    int params[4] = {token_id_a, pos, token_id_b, pos + 1};
    GWEN_CHECK_CUDA(cudaMemcpyAsync(d_token_id, params, 4 * sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    if (!graph_2tok_captured) {
        cudaGraph_t graph;
        GWEN_CHECK_CUDA(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal));
        forward_body_2tok(model, compute_stream);
        GWEN_CHECK_CUDA(cudaStreamEndCapture(compute_stream, &graph));
        GWEN_CHECK_CUDA(cudaGraphInstantiate(&graph_2tok_exec, graph, nullptr, nullptr, 0));
        GWEN_CHECK_CUDA(cudaGraphDestroy(graph));
        graph_2tok_captured = true;
    }

    GWEN_CHECK_CUDA(cudaGraphLaunch(graph_2tok_exec, compute_stream));
    GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));

    // d_argmax_token[0] = pred_a, d_argmax_token[1] = pred_b (adjacent in device memory)
    int results[2];
    GWEN_CHECK_CUDA(cudaMemcpy(results, d_argmax_token, 2 * sizeof(int), cudaMemcpyDeviceToHost));

    pos += 2;
    return {results[0], results[1]};
}

// ============================================================
// Forward pass — CUDA graph capture + replay
// ============================================================

int InferenceState::forward(Model& model, int token_id) {
    // Write token_id and pos to device memory in one copy (adjacent layout)
    int params[2] = {token_id, pos};
    GWEN_CHECK_CUDA(cudaMemcpyAsync(d_token_id, params, 2 * sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    if (!graph_captured) {
        // First call: capture the entire forward pass as a CUDA graph
        cudaGraph_t graph;

        GWEN_CHECK_CUDA(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal));
        forward_body(model, compute_stream);
        GWEN_CHECK_CUDA(cudaStreamEndCapture(compute_stream, &graph));
        GWEN_CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        GWEN_CHECK_CUDA(cudaGraphDestroy(graph));
        graph_captured = true;
    }

    // Replay the graph
    GWEN_CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));

    // Sync and get result
    GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    int next_token;
    GWEN_CHECK_CUDA(cudaMemcpy(&next_token, d_argmax_token, sizeof(int), cudaMemcpyDeviceToHost));

    pos++;
    return next_token;
}

// ============================================================
// Prefill — process all prompt tokens, using GEMM for projections
// ============================================================
// Row dequantization: extract K rows from Q6K table → FP16
// ============================================================
// Same dequant math as kernel_embed_lookup_batch_q6k, but uses a row_ids
// indirection array instead of sequential token IDs.
// Grid: (1, K), Block: 256 — one block per row.

__global__ void __launch_bounds__(256)
kernel_dequant_rows_q6k(const void* __restrict__ table,
                         const int* __restrict__ row_ids,
                         half* __restrict__ dst,
                         int dim, int K) {
    int row_idx = blockIdx.y;
    if (row_idx >= K) return;

    int token_id = row_ids[row_idx];
    int blocks_per_row = dim / 256;
    int tid = threadIdx.x;

    for (int blk_local = 0; blk_local < blocks_per_row; blk_local++) {
        int blk_idx = token_id * blocks_per_row + blk_local;
        const uint8_t* base = static_cast<const uint8_t*>(table) + (size_t)blk_idx * 210;

        const uint8_t* ql = base;
        const uint8_t* qh = base + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(base + 192);
        half d_half;
        memcpy(&d_half, base + 208, sizeof(half));
        float d = __half2float(d_half);

        int half_idx = tid / 128;
        int j = tid % 128;
        int quarter = j / 32;
        int pos = j % 32;

        int ql_byte = half_idx * 64 + (quarter & 1) * 32 + pos;
        int ql_nibble = (quarter >= 2) ? (ql[ql_byte] >> 4) : (ql[ql_byte] & 0xF);

        int qh_byte = half_idx * 32 + pos;
        int qh_shift = quarter * 2;
        int qh_bits = (qh[qh_byte] >> qh_shift) & 0x3;

        int q_val = ql_nibble | (qh_bits << 4);
        int scale_idx = half_idx * 8 + quarter * 2 + pos / 16;
        int8_t scale = scales[scale_idx];
        float result = d * scale * (q_val - 32);
        dst[(size_t)row_idx * dim + blk_local * 256 + tid] = __float2half(result);
    }
}

void gwen_dequant_rows_q6k(const void* table, const int* row_ids, half* dst,
                            int K, int dim, cudaStream_t stream) {
    dim3 grid(1, K);
    kernel_dequant_rows_q6k<<<grid, 256, 0, stream>>>(table, row_ids, dst, dim, K);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// Strategy: batch GEMMs for linear projections, sequential for state-dependent ops.
// Layout: prefill_x[t, dim] = hidden state of token t

// Batched embedding lookup
__global__ void __launch_bounds__(256)
kernel_embed_lookup_batch_q6k(const void* __restrict__ table,
                               const int* __restrict__ token_ids,
                               half* __restrict__ y,
                               int dim, int n_tokens) {
    int token_idx = blockIdx.y;
    if (token_idx >= n_tokens) return;

    int token_id = token_ids[token_idx];
    int blocks_per_row = dim / 256;
    int tid = threadIdx.x;

    for (int blk_local = 0; blk_local < blocks_per_row; blk_local++) {
        int blk_idx = token_id * blocks_per_row + blk_local;
        const uint8_t* base = static_cast<const uint8_t*>(table) + (size_t)blk_idx * 210;

        const uint8_t* ql = base;
        const uint8_t* qh = base + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(base + 192);
        half d_half;
        memcpy(&d_half, base + 208, sizeof(half));
        float d = __half2float(d_half);

        int half_idx = tid / 128;
        int j = tid % 128;
        int quarter = j / 32;
        int pos = j % 32;

        int ql_byte = half_idx * 64 + (quarter & 1) * 32 + pos;
        int ql_nibble = (quarter >= 2) ? (ql[ql_byte] >> 4) : (ql[ql_byte] & 0xF);

        int qh_byte = half_idx * 32 + pos;
        int qh_shift = quarter * 2;
        int qh_bits = (qh[qh_byte] >> qh_shift) & 0x3;

        int q_val = ql_nibble | (qh_bits << 4);
        int scale_idx = half_idx * 8 + quarter * 2 + pos / 16;
        int8_t scale = scales[scale_idx];
        float result = d * scale * (q_val - 32);
        y[(size_t)token_idx * dim + blk_local * 256 + tid] = __float2half(result);
    }
}

// Batched RMSNorm for prefill: normalize N vectors of length dim
__global__ void __launch_bounds__(256)
kernel_rmsnorm_batch_f32w(const half* __restrict__ x, const float* __restrict__ weight,
                          half* __restrict__ y, int n_tokens, int dim, float eps) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;
    int tid = threadIdx.x;

    const half* xv = x + (size_t)token_idx * dim;
    half* yv = y + (size_t)token_idx * dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(xv[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[8];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    float inv_rms = rsqrtf(sum_sq / dim + eps);

    // Broadcast and apply
    __shared__ float s_inv_rms;
    if (tid == 0) s_inv_rms = inv_rms;
    __syncthreads();
    inv_rms = s_inv_rms;

    for (int i = tid; i < dim; i += blockDim.x) {
        yv[i] = __float2half(__half2float(xv[i]) * inv_rms * weight[i]);
    }
}

// Batched SwiGLU: y[t] = SiLU(gate[t]) * up[t]
__global__ void __launch_bounds__(256)
kernel_swiglu_batch(const half* __restrict__ gate, const half* __restrict__ up,
                    half* __restrict__ y, int n_tokens, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * dim;
    if (idx >= total) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    y[idx] = __float2half(g * sig * u);
}

// Batched residual add: y[i] += residual[i]
__global__ void __launch_bounds__(256)
kernel_add_inplace_batch(half* __restrict__ y, const half* __restrict__ residual, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    y[idx] = __hadd(y[idx], residual[idx]);
}

// ============================================================
// F32 residual stream kernels (for verified GEMM decode path)
// ============================================================

// Embedding lookup with F32 output (eliminates initial FP16 truncation)
__global__ void __launch_bounds__(256)
kernel_embed_lookup_batch_q6k_f32(const void* __restrict__ table,
                                   const int* __restrict__ token_ids,
                                   float* __restrict__ y,
                                   int dim, int n_tokens) {
    int token_idx = blockIdx.y;
    if (token_idx >= n_tokens) return;

    int token_id = token_ids[token_idx];
    int blocks_per_row = dim / 256;
    int tid = threadIdx.x;

    for (int blk_local = 0; blk_local < blocks_per_row; blk_local++) {
        int blk_idx = token_id * blocks_per_row + blk_local;
        const uint8_t* base = static_cast<const uint8_t*>(table) + (size_t)blk_idx * 210;

        const uint8_t* ql = base;
        const uint8_t* qh = base + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(base + 192);
        half d_half;
        memcpy(&d_half, base + 208, sizeof(half));
        float d = __half2float(d_half);

        int half_idx = tid / 128;
        int j = tid % 128;
        int quarter = j / 32;
        int pos_in_block = j % 32;

        int ql_byte = half_idx * 64 + (quarter & 1) * 32 + pos_in_block;
        int ql_nibble = (quarter >= 2) ? (ql[ql_byte] >> 4) : (ql[ql_byte] & 0xF);

        int qh_byte = half_idx * 32 + pos_in_block;
        int qh_shift = quarter * 2;
        int qh_bits = (qh[qh_byte] >> qh_shift) & 0x3;

        int q_val = ql_nibble | (qh_bits << 4);
        int scale_idx = half_idx * 8 + quarter * 2 + pos_in_block / 16;
        int8_t scale = scales[scale_idx];
        float result = d * scale * (q_val - 32);
        y[(size_t)token_idx * dim + blk_local * 256 + tid] = result;  // F32 output
    }
}

// RMSNorm with F32 input, FP16 output (for feeding GEMM which expects half*)
__global__ void __launch_bounds__(256)
kernel_rmsnorm_batch_f32in_f32w(const float* __restrict__ x, const float* __restrict__ weight,
                                 half* __restrict__ y, int n_tokens, int dim, float eps) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;
    int tid = threadIdx.x;

    const float* xv = x + (size_t)token_idx * dim;
    half* yv = y + (size_t)token_idx * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = xv[i];
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    float inv_rms = rsqrtf(sum_sq / dim + eps);

    __shared__ float s_inv_rms;
    if (tid == 0) s_inv_rms = inv_rms;
    __syncthreads();
    inv_rms = s_inv_rms;

    for (int i = tid; i < dim; i += blockDim.x) {
        yv[i] = __float2half(xv[i] * inv_rms * weight[i]);
    }
}

// F32 to half conversion
__global__ void __launch_bounds__(256)
kernel_f32_to_half(const float* __restrict__ in, half* __restrict__ out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __float2half(in[idx]);
}

// F32 inplace add: y[i] += x[i]
__global__ void __launch_bounds__(256)
kernel_add_inplace_f32(float* __restrict__ y, const float* __restrict__ x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    y[idx] += x[idx];
}

// Batch Conv1D + SiLU with F32 data (same as kernel_batch_conv1d_silu but float I/O)
__global__ void __launch_bounds__(256)
kernel_batch_conv1d_silu_f32(
    float* __restrict__ data,          // [N, dim] in-place (F32)
    float* __restrict__ conv_state,    // [kernel_size-1, dim] rolling state
    const float* __restrict__ weight,  // [dim, kernel_size] conv weights
    int N, int dim, int kernel_size)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    float st0 = conv_state[0 * dim + d];
    float st1 = conv_state[1 * dim + d];
    float st2 = conv_state[2 * dim + d];

    float w0 = weight[d * kernel_size + 0];
    float w1 = weight[d * kernel_size + 1];
    float w2 = weight[d * kernel_size + 2];
    float w3 = weight[d * kernel_size + 3];

    for (int t = 0; t < N; t++) {
        float x_val = data[(size_t)t * dim + d];
        float acc = w3 * x_val + w2 * st2 + w1 * st1 + w0 * st0;
        float silu = acc / (1.0f + expf(-acc));
        data[(size_t)t * dim + d] = silu;  // F32 output

        st0 = st1;
        st1 = st2;
        st2 = x_val;
    }

    conv_state[0 * dim + d] = st0;
    conv_state[1 * dim + d] = st1;
    conv_state[2 * dim + d] = st2;
}

// DeltaNet prefill with F32 QKV input and F32 output
__global__ void __launch_bounds__(128)
kernel_deltanet_prefill_f32(
    float* __restrict__ S,              // [n_heads, dk, dv] recurrent state
    const float* __restrict__ qkv_batch, // [N, 3*ssm_inner] F32 (after conv1d)
    const float* __restrict__ gate_batch, // [N, n_heads]
    const float* __restrict__ beta_batch, // [N, n_heads]
    float* __restrict__ output_batch,    // [N, ssm_inner] F32 output
    int N, int n_heads, int dk, int dv, int ssm_inner,
    float q_scale)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int j = threadIdx.x;
    int warp_id = j / 32;
    int lane = j % 32;

    float* S_head = S + (size_t)head * dk * dv;

    __shared__ float sh_k[128];
    __shared__ float sh_q[128];
    __shared__ float sh_reduce[4];

    for (int t = 0; t < N; t++) {
        const float* q_ptr = qkv_batch + (size_t)t * ssm_inner * 3 + head * dk;
        const float* k_ptr = q_ptr + ssm_inner;
        const float* v_ptr = k_ptr + ssm_inner;

        // L2-normalize Q
        float q_raw = q_ptr[j];
        float q_sq = q_raw * q_raw;
        for (int off = 16; off > 0; off >>= 1)
            q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, off);
        if (lane == 0) sh_reduce[warp_id] = q_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        float q_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f)) * q_scale;

        // L2-normalize K
        float k_raw = k_ptr[j];
        float k_sq = k_raw * k_raw;
        for (int off = 16; off > 0; off >>= 1)
            k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, off);
        if (lane == 0) sh_reduce[warp_id] = k_sq;
        __syncthreads();
        if (j < 4) {
            float s = sh_reduce[j];
            for (int off = 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xF, s, off);
            if (j == 0) sh_reduce[0] = s;
        }
        __syncthreads();
        float k_inv = rsqrtf(fmaxf(sh_reduce[0], 1e-12f));

        sh_q[j] = q_raw * q_inv;
        sh_k[j] = k_raw * k_inv;
        __syncthreads();

        float decay = expf(gate_batch[t * n_heads + head]);
        float b = beta_batch[t * n_heads + head];

        float sk_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float val = S_head[i * 128 + j] * decay;
            S_head[i * 128 + j] = val;
            sk_j += val * sh_k[i];
        }

        float d_j = (v_ptr[j] - sk_j) * b;
        float o_j = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 128; i++) {
            float updated = S_head[i * 128 + j] + sh_k[i] * d_j;
            S_head[i * 128 + j] = updated;
            o_j += updated * sh_q[i];
        }

        output_batch[(size_t)t * ssm_inner + head * dv + j] = o_j;  // F32 output
        __syncthreads();
    }
}

// Batch gated RMSNorm with F32 I/O
__global__ void __launch_bounds__(32)
kernel_batch_gated_rmsnorm_f32(
    const float* __restrict__ x,       // [N, n_heads * dim_per_head] F32
    const float* __restrict__ weight,  // [dim_per_head]
    const float* __restrict__ gate,    // [N, n_heads * dim_per_head] F32
    float* __restrict__ output,        // [N, n_heads * dim_per_head] F32
    int N, int n_heads, int dim_per_head, float eps)
{
    int block_id = blockIdx.x;
    int t = block_id / n_heads;
    int head = block_id % n_heads;
    if (t >= N) return;

    int lane = threadIdx.x;
    int stride = n_heads * dim_per_head;
    int base = t * stride + head * dim_per_head;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim_per_head; i += 32) {
        float val = x[base + i];
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }
    float rms_inv = rsqrtf(sum_sq / dim_per_head + eps);

    for (int i = lane; i < dim_per_head; i += 32) {
        float x_val = x[base + i] * rms_inv;
        float w = weight[i];
        float g = gate[base + i];
        float silu_g = g / (1.0f + expf(-g));
        output[base + i] = x_val * w * silu_g;  // F32 output
    }
}

// Batched SwiGLU with F32 I/O
__global__ void __launch_bounds__(256)
kernel_swiglu_batch_f32(const float* __restrict__ gate, const float* __restrict__ up,
                         float* __restrict__ y, int n_tokens, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * dim;
    if (idx >= total) return;

    float g = gate[idx];
    float u = up[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    y[idx] = g * sig * u;
}

int InferenceState::forward_prefill(Model& model, const std::vector<int>& tokens) {
    const auto& cfg = model.config;
    const float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);
    int N = (int)tokens.size();
    cudaStream_t s = compute_stream;

    GWEN_CHECK(N <= max_prefill, "Prompt too long for prefill buffer");
    GWEN_CHECK(N > 0, "Empty prompt");

    // Upload token IDs to device (use pre-allocated buffer)
    int* d_token_ids = d_prefill_tokens;
    GWEN_CHECK_CUDA(cudaMemcpyAsync(d_token_ids, tokens.data(), N * sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    // 1. Batch embedding lookup → F32 residual accumulator [N, 1024]
    {
        dim3 grid(1, N);
        kernel_embed_lookup_batch_q6k_f32<<<grid, 256, 0, s>>>(
            model.token_embd.device_data, d_token_ids, prefill_x_f32, cfg.n_embed, N);
    }

    // Working buffers: F32 residual accumulators swap between pf_a/pf_b each half-layer.
    // pf_norm stays FP16 — it's the RMSNorm output feeding GEMMs (which expect half*).
    // GEMM outputs go to pf_norm (reused as temp), then are added to the F32 residual.
    // This eliminates 48 accumulated FP16 truncations (2 adds × 24 layers) in the residual stream.
    float* pf_a = prefill_x_f32;     // F32 residual accumulator A
    float* pf_b = prefill_out_f32;   // F32 residual accumulator B
    half* pf_norm = prefill_norm;     // FP16 normalized (GEMM input/output temp)

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // Per-layer dump for debugging (GWEN_DUMP_LAYERS=1)
    bool dump_layers = (getenv("GWEN_DUMP_LAYERS") != nullptr);
    auto dump_f32_buf = [&](const float* d_ptr, int offset_elems, int count, const std::string& path) {
        if (!dump_layers) return;
        GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
        std::vector<float> host(count);
        GWEN_CHECK_CUDA(cudaMemcpy(host.data(), d_ptr + offset_elems,
                                    count * sizeof(float), cudaMemcpyDeviceToHost));
        FILE* fp = fopen(path.c_str(), "wb");
        fwrite(&count, sizeof(int), 1, fp);
        fwrite(host.data(), sizeof(float), count, fp);
        fclose(fp);
    };

    // Dump embedding before any layers
    dump_f32_buf(pf_a, (N-1)*cfg.n_embed, cfg.n_embed, "/tmp/gwen_embed.bin");

    // 2. Process each layer
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            // ===== DeltaNet Layer — Full F32 internal pipeline =====
            // RMSNorm(F32→half) → GEMM(half→F32) → Conv1d/DeltaNet/GatedRMSNorm(F32) →
            // convert(F32→half) → GEMM(half→F32, direct to residual) → F32 add
            const auto& w = layer.deltanet;
            auto& state = deltanet_states[dn_state_idx++];

            // RMSNorm: pf_a(F32) → pf_norm(half) for GEMM input
            kernel_rmsnorm_batch_f32in_f32w<<<N, 256, 0, s>>>(
                pf_a, static_cast<const float*>(w.attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // Dump layer 0 intermediates for bisection
            if (layer_idx == 0 && dump_layers) {
                // Dump RMSNorm output (FP16 → F32 for comparison)
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                {
                    std::vector<half> h16(cfg.n_embed);
                    GWEN_CHECK_CUDA(cudaMemcpy(h16.data(), pf_norm + (N-1)*cfg.n_embed,
                        cfg.n_embed * sizeof(half), cudaMemcpyDeviceToHost));
                    std::vector<float> h32(cfg.n_embed);
                    for (int j = 0; j < (int)cfg.n_embed; j++) h32[j] = __half2float(h16[j]);
                    int count = cfg.n_embed;
                    FILE* fp = fopen("/tmp/gwen_l0_rmsnorm.bin", "wb");
                    fwrite(&count, sizeof(int), 1, fp);
                    fwrite(h32.data(), sizeof(float), count, fp);
                    fclose(fp);
                }
            }

            // QKV and gate GEMM with F32 output
            int qkv_dim = cfg.ssm_inner_size * 3;  // 6144
            gwen_gemm_f32out(w.attn_qkv.device_data, w.attn_qkv.type, prefill_temp_w,
                             pf_norm, prefill_proj_qkv_f32,
                             w.attn_qkv.shape[1], w.attn_qkv.shape[0], N, s);
            gwen_gemm_f32out(w.attn_gate.device_data, w.attn_gate.type, prefill_temp_w,
                             pf_norm, prefill_proj_gate_f32,
                             w.attn_gate.shape[1], w.attn_gate.shape[0], N, s);

            // Dump layer 0 QKV GEMM output
            if (layer_idx == 0) {
                dump_f32_buf(prefill_proj_qkv_f32, (N-1)*qkv_dim, qkv_dim,
                             "/tmp/gwen_l0_qkv.bin");
                dump_f32_buf(prefill_proj_gate_f32, (N-1)*cfg.ssm_inner_size, cfg.ssm_inner_size,
                             "/tmp/gwen_l0_gate.bin");
            }

            // F32 DeltaNet state ops
            {
                // Conv1d + SiLU on F32 QKV data
                int conv_grid = (qkv_dim + 255) / 256;
                kernel_batch_conv1d_silu_f32<<<conv_grid, 256, 0, s>>>(
                    prefill_proj_qkv_f32, state.conv_state,
                    static_cast<const float*>(w.ssm_conv1d.device_data),
                    N, qkv_dim, cfg.ssm_conv_kernel);

                // Gate/beta (reads FP16 pf_norm, outputs F32 — unchanged)
                kernel_batch_compute_gate_beta<<<dim3(cfg.ssm_n_heads, N), 32, 0, s>>>(
                    pf_norm,
                    w.ssm_alpha.device_data, w.ssm_beta.device_data,
                    static_cast<const float*>(w.ssm_a.device_data),
                    static_cast<const float*>(w.ssm_dt_bias.device_data),
                    prefill_dn_gate, prefill_dn_beta,
                    N, cfg.n_embed, cfg.ssm_n_heads);

                // DeltaNet with F32 I/O
                kernel_deltanet_prefill_f32<<<cfg.ssm_n_heads, 128, 0, s>>>(
                    state.S, prefill_proj_qkv_f32,
                    prefill_dn_gate, prefill_dn_beta,
                    prefill_dn_out_f32,
                    N, cfg.ssm_n_heads, cfg.ssm_state_size, cfg.ssm_state_size,
                    cfg.ssm_inner_size, q_scale);

                // Gated RMSNorm with F32 I/O
                kernel_batch_gated_rmsnorm_f32<<<N * cfg.ssm_n_heads, 32, 0, s>>>(
                    prefill_dn_out_f32,
                    static_cast<const float*>(w.ssm_norm.device_data),
                    prefill_proj_gate_f32,
                    prefill_proj_gate_f32,
                    N, cfg.ssm_n_heads, cfg.ssm_state_size, cfg.rms_norm_eps);
            }

            // Convert F32→FP16 for Out GEMM input, then GEMM→F32 directly to residual
            {
                int total_gate = N * cfg.ssm_inner_size;
                int blocks_gate = (total_gate + 255) / 256;
                kernel_f32_to_half<<<blocks_gate, 256, 0, s>>>(
                    prefill_proj_gate_f32, prefill_proj_gate, total_gate);
            }
            gwen_gemm_f32out(w.ssm_out.device_data, w.ssm_out.type, prefill_temp_w,
                             prefill_proj_gate, pf_b,
                             w.ssm_out.shape[1], w.ssm_out.shape[0], N, s);

            // F32 residual add: pf_b += pf_a
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_f32<<<blocks, 256, 0, s>>>(pf_b, pf_a, total);
            }
            // Dump attn_residual (last token in pf_b)
            dump_f32_buf(pf_b, (N-1)*cfg.n_embed, cfg.n_embed,
                         "/tmp/gwen_attn_res_" + std::to_string(layer_idx) + ".bin");

            // Post-attention RMSNorm: pf_b(F32) → pf_norm(half)
            kernel_rmsnorm_batch_f32in_f32w<<<N, 256, 0, s>>>(
                pf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // FFN with F32 output GEMMs and F32 SwiGLU
            gwen_gemm_f32out(w.ffn_gate.device_data, w.ffn_gate.type, prefill_temp_w,
                             pf_norm, prefill_ffn_gate_f32,
                             w.ffn_gate.shape[1], w.ffn_gate.shape[0], N, s);
            gwen_gemm_f32out(w.ffn_up.device_data, w.ffn_up.type, prefill_temp_w,
                             pf_norm, prefill_ffn_up_f32,
                             w.ffn_up.shape[1], w.ffn_up.shape[0], N, s);

            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch_f32<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate_f32, prefill_ffn_up_f32, prefill_ffn_out_f32, N, cfg.n_ff);
            }

            // Convert F32 SwiGLU output → FP16 for FFN down GEMM, then GEMM→F32 to residual
            {
                int total_ffn = N * cfg.n_ff;
                int blocks_ffn = (total_ffn + 255) / 256;
                kernel_f32_to_half<<<blocks_ffn, 256, 0, s>>>(
                    prefill_ffn_out_f32, prefill_ffn_out, total_ffn);
            }
            gwen_gemm_f32out(w.ffn_down.device_data, w.ffn_down.type, prefill_temp_w,
                             prefill_ffn_out, pf_a,
                             w.ffn_down.shape[1], w.ffn_down.shape[0], N, s);

            // F32 residual add: pf_a += pf_b
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_f32<<<blocks, 256, 0, s>>>(pf_a, pf_b, total);
            }
            // Dump post_ffn (last token in pf_a)
            dump_f32_buf(pf_a, (N-1)*cfg.n_embed, cfg.n_embed,
                         "/tmp/gwen_post_ffn_" + std::to_string(layer_idx) + ".bin");

        } else {
            // ===== Full Attention Layer — F32 residual + F32 FFN =====
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            // RMSNorm: pf_a(F32) → pf_norm(half)
            kernel_rmsnorm_batch_f32in_f32w<<<N, 256, 0, s>>>(
                pf_a, static_cast<const float*>(w.attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            int attn_dim = cfg.n_head * cfg.head_dim;  // 2048
            int kv_dim = cfg.n_head_kv * cfg.head_dim;  // 512
            int q_proj_dim = w.attn_q.shape[1];  // 4096

            // Q, K, V projections (FP16 — per-token attention ops are FP16)
            gwen_gemm(w.attn_q.device_data, w.attn_q.type, prefill_temp_w,
                      pf_norm, prefill_proj_qkv,
                      q_proj_dim, w.attn_q.shape[0], N, s);
            gwen_gemm(w.attn_k.device_data, w.attn_k.type, prefill_temp_w,
                      pf_norm, prefill_ffn_gate,
                      kv_dim, w.attn_k.shape[0], N, s);
            gwen_gemm(w.attn_v.device_data, w.attn_v.type, prefill_temp_w,
                      pf_norm, prefill_ffn_up,
                      kv_dim, w.attn_v.shape[0], N, s);

            // Per-token attention (FP16 internal — only 6 layers, non-recurrent)
            for (int t = 0; t < N; t++) {
                int cur_pos = pos + t;

                int deint_blocks = (attn_dim + 255) / 256;
                kernel_deinterleave_qgate<<<deint_blocks, 256, 0, s>>>(
                    prefill_proj_qkv + (size_t)t * q_proj_dim,
                    fa_q, gated_out, cfg.n_head, cfg.head_dim);

                GWEN_CHECK_CUDA(cudaMemcpyAsync(fa_k, prefill_ffn_gate + (size_t)t * kv_dim,
                    kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, s));
                GWEN_CHECK_CUDA(cudaMemcpyAsync(fa_v, prefill_ffn_up + (size_t)t * kv_dim,
                    kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, s));

                gwen_rmsnorm_batched_f32w(fa_q, static_cast<const float*>(w.attn_q_norm.device_data),
                                          fa_q, cfg.n_head, cfg.head_dim, cfg.rms_norm_eps, s);
                gwen_rmsnorm_batched_f32w(fa_k, static_cast<const float*>(w.attn_k_norm.device_data),
                                          fa_k, cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps, s);

                GWEN_CHECK_CUDA(cudaMemcpyAsync(d_pos, &cur_pos, sizeof(int), cudaMemcpyHostToDevice, s));

                gwen_rope(fa_q, fa_k,
                          cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                          d_pos, cfg.rope_theta, cfg.rope_sections, cfg.rope_dim, s);

                int kv_blocks = (kv_dim + 255) / 256;
                kernel_kv_cache_store<<<kv_blocks, 256, 0, s>>>(
                    cache.k_cache, cache.v_cache, fa_k, fa_v, d_pos, kv_dim);

                float scale = 1.0f / sqrtf((float)cfg.head_dim);
                kernel_gqa_attention_decode<<<cfg.n_head, 256, 0, s>>>(
                    fa_q, cache.k_cache, cache.v_cache,
                    attn_out, attn_scores, d_pos,
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim, max_seq_alloc, scale);

                gwen_sigmoid_mul(attn_out, gated_out, gated_out, attn_dim, s);

                GWEN_CHECK_CUDA(cudaMemcpyAsync(
                    prefill_proj_gate + (size_t)t * cfg.ssm_inner_size,
                    gated_out, attn_dim * sizeof(half),
                    cudaMemcpyDeviceToDevice, s));
            }

            // Out projection → F32 directly to residual
            gwen_gemm_f32out(w.attn_output.device_data, w.attn_output.type, prefill_temp_w,
                             prefill_proj_gate, pf_b,
                             w.attn_output.shape[1], w.attn_output.shape[0], N, s);

            // F32 residual add: pf_b += pf_a
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_f32<<<blocks, 256, 0, s>>>(pf_b, pf_a, total);
            }
            // Dump attn_residual (last token in pf_b)
            dump_f32_buf(pf_b, (N-1)*cfg.n_embed, cfg.n_embed,
                         "/tmp/gwen_attn_res_" + std::to_string(layer_idx) + ".bin");

            // Post-attention RMSNorm: pf_b(F32) → pf_norm(half)
            kernel_rmsnorm_batch_f32in_f32w<<<N, 256, 0, s>>>(
                pf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // FFN with F32 output GEMMs and F32 SwiGLU
            gwen_gemm_f32out(w.ffn_gate.device_data, w.ffn_gate.type, prefill_temp_w,
                             pf_norm, prefill_ffn_gate_f32,
                             w.ffn_gate.shape[1], w.ffn_gate.shape[0], N, s);
            gwen_gemm_f32out(w.ffn_up.device_data, w.ffn_up.type, prefill_temp_w,
                             pf_norm, prefill_ffn_up_f32,
                             w.ffn_up.shape[1], w.ffn_up.shape[0], N, s);

            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch_f32<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate_f32, prefill_ffn_up_f32, prefill_ffn_out_f32, N, cfg.n_ff);
            }

            // Convert F32→FP16 for FFN down GEMM, then GEMM→F32 to residual
            {
                int total_ffn = N * cfg.n_ff;
                int blocks_ffn = (total_ffn + 255) / 256;
                kernel_f32_to_half<<<blocks_ffn, 256, 0, s>>>(
                    prefill_ffn_out_f32, prefill_ffn_out, total_ffn);
            }
            gwen_gemm_f32out(w.ffn_down.device_data, w.ffn_down.type, prefill_temp_w,
                             prefill_ffn_out, pf_a,
                             w.ffn_down.shape[1], w.ffn_down.shape[0], N, s);

            // F32 residual add: pf_a += pf_b
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_f32<<<blocks, 256, 0, s>>>(pf_a, pf_b, total);
            }
            // Dump post_ffn (last token in pf_a)
            dump_f32_buf(pf_a, (N-1)*cfg.n_embed, cfg.n_embed,
                         "/tmp/gwen_post_ffn_" + std::to_string(layer_idx) + ".bin");
        }
    }

    // 3. Final RMSNorm on last token: convert F32→half, then normalize
    // This also leaves buf_a with the FP16 hidden state for subsequent decode tokens
    float* last_hidden_f32 = pf_a + (size_t)(N - 1) * cfg.n_embed;
    {
        int blocks = (cfg.n_embed + 255) / 256;
        kernel_f32_to_half<<<blocks, 256, 0, s>>>(last_hidden_f32, buf_a, cfg.n_embed);
    }
    gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(model.output_norm.device_data),
                      x_norm, cfg.n_embed, cfg.rms_norm_eps, s);

    // 4. LM Head GEMV on last token
    gwen_gemv(model.token_embd.device_data, x_norm, logits_h,
              cfg.n_vocab, cfg.n_embed, model.token_embd.type, s);

    int logit_blocks = (cfg.n_vocab + 255) / 256;
    kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, cfg.n_vocab);

    // Dump result_norm (F32 hidden state before LM head) and result_output (F32 logits)
    dump_f32_buf(last_hidden_f32, 0, cfg.n_embed, "/tmp/gwen_result_norm.bin");
    dump_f32_buf(logits_f, 0, cfg.n_vocab, "/tmp/gwen_result_output.bin");

    kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(logits_f, argmax_partial_max, argmax_partial_idx, cfg.n_vocab);
    kernel_argmax_reduce<<<1, 256, 0, s>>>(argmax_partial_max, argmax_partial_idx, d_argmax_token, ARGMAX_BLOCKS);

    GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
    int next_token;
    GWEN_CHECK_CUDA(cudaMemcpy(&next_token, d_argmax_token, sizeof(int), cudaMemcpyDeviceToHost));

    pos += N;
    return next_token;
}

// ============================================================
// Batch extract: B independent sequences, GEMM-batched forward
// ============================================================
// Processes B sequences of length L through all 24 layers.
// GEMM projections operate on all B*L tokens at once (one weight read).
// DeltaNet uses B independent S/conv states in parallel.
// Full attention uses flash-attention-style causal attention (no KV cache).
// Output: prefill_x holds [B*L, n_embed] hidden states.

void InferenceState::extract_hidden_batch(Model& model, const int32_t* all_tokens,
                                           int B, int L, void* output_host) {
    const auto& cfg = model.config;
    int N = B * L;  // total tokens
    cudaStream_t s = compute_stream;

    GWEN_CHECK(N <= max_prefill, "Batch too large for prefill buffers");
    GWEN_CHECK(B <= max_batch_seqs, "Too many sequences for batch state");
    GWEN_CHECK(B > 0 && L > 0, "Empty batch");

    const float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);

    // Zero batch DeltaNet states (S and conv)

    {
        int n_heads = cfg.ssm_n_heads;
        int dk = cfg.ssm_state_size, dv = cfg.ssm_state_size;
        size_t S_per_layer = (size_t)n_heads * dk * dv * sizeof(float);
        size_t S_total = (size_t)B * n_batch_dn_layers * S_per_layer;
        GWEN_CHECK_CUDA(cudaMemsetAsync(batch_dn_S, 0, S_total, s));

        int qkv_dim = cfg.ssm_inner_size * 3;
        int conv_km1 = cfg.ssm_conv_kernel - 1;
        size_t conv_per_layer = (size_t)conv_km1 * qkv_dim * sizeof(float);
        size_t conv_total = (size_t)B * n_batch_dn_layers * conv_per_layer;
        GWEN_CHECK_CUDA(cudaMemsetAsync(batch_dn_conv, 0, conv_total, s));
    }

    // Upload token IDs to device
    GWEN_CHECK_CUDA(cudaMemcpyAsync(d_prefill_tokens, all_tokens,
                                     N * sizeof(int), cudaMemcpyHostToDevice, s));


    // 1. Batch embedding lookup → prefill_x [B*L, 1024]

    {
        dim3 grid(1, N);
        kernel_embed_lookup_batch_q6k<<<grid, 256, 0, s>>>(
            model.token_embd.device_data, d_prefill_tokens, prefill_x, cfg.n_embed, N);
    }

    half* pf_a = prefill_x;
    half* pf_b = prefill_out;
    half* pf_norm = prefill_norm;

    int dn_layer_idx = 0;

    // 2. Process each layer — all B*L tokens through GEMM, per-sequence for recurrent ops
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            // ===== DeltaNet Layer =====
            const auto& w = layer.deltanet;
            int dn_idx = dn_layer_idx++;

            // Batch RMSNorm: [B*L, 1024] → [B*L, 1024]
        
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_a, static_cast<const float*>(w.attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // Batch GEMM projections on all B*L tokens (ONE weight read each)
            int qkv_dim = cfg.ssm_inner_size * 3;  // 6144
        
            gwen_gemm(w.attn_qkv.device_data, w.attn_qkv.type, prefill_temp_w,
                      pf_norm, prefill_proj_qkv,
                      w.attn_qkv.shape[1], w.attn_qkv.shape[0], N, s);
            gwen_gemm(w.attn_gate.device_data, w.attn_gate.type, prefill_temp_w,
                      pf_norm, prefill_proj_gate,
                      w.attn_gate.shape[1], w.attn_gate.shape[0], N, s);

            // Batch conv1d + SiLU: B independent conv states
            int n_heads = cfg.ssm_n_heads;
            int dk = cfg.ssm_state_size;
            int dv = cfg.ssm_state_size;
            size_t S_per_layer_elems = (size_t)n_heads * dk * dv;
            size_t conv_per_layer_elems = (size_t)(cfg.ssm_conv_kernel - 1) * qkv_dim;

            // Pointer to this layer's batch of S states and conv states
            float* batch_S_layer = batch_dn_S + (size_t)dn_idx * max_batch_seqs * S_per_layer_elems;
            float* batch_conv_layer = batch_dn_conv + (size_t)dn_idx * max_batch_seqs * conv_per_layer_elems;

        
            {
                int conv_grid_x = (qkv_dim + 255) / 256;
                dim3 conv_grid(conv_grid_x, B);
                kernel_batch_conv1d_silu_multi<<<conv_grid, 256, 0, s>>>(
                    prefill_proj_qkv, batch_conv_layer,
                    static_cast<const float*>(w.ssm_conv1d.device_data),
                    B, L, qkv_dim, cfg.ssm_conv_kernel);
            }

            // Batch gate/beta for all B*L tokens
        
            kernel_batch_compute_gate_beta<<<dim3(n_heads, N), 32, 0, s>>>(
                pf_norm,
                w.ssm_alpha.device_data, w.ssm_beta.device_data,
                static_cast<const float*>(w.ssm_a.device_data),
                static_cast<const float*>(w.ssm_dt_bias.device_data),
                prefill_dn_gate, prefill_dn_beta,
                N, cfg.n_embed, n_heads);

            // DeltaNet kernel: chunkwise (default) or old sequential (GWEN_USE_SEQ_DN=1)
            {
                static int use_seq = -1;
                if (use_seq < 0) use_seq = (getenv("GWEN_USE_SEQ_DN") != nullptr) ? 1 : 0;

                if (use_seq) {
                    kernel_deltanet_prefill_batch<<<n_heads * B, 128, 0, s>>>(
                        batch_S_layer, prefill_proj_qkv, prefill_dn_gate, prefill_dn_beta,
                        prefill_ffn_gate, B, L, n_heads, dk, dv, cfg.ssm_inner_size, q_scale);
                } else {
                    int NT = (L + 63) / 64;

                    kernel_cumgate_l2norm_batch<<<dim3(n_heads, B), 128, 0, s>>>(
                        prefill_proj_qkv, prefill_dn_gate, chunk_gate_cumul,
                        B, L, n_heads, dk, cfg.ssm_inner_size, q_scale);

                    kernel_chunkwise_wy_repr<<<dim3(NT * B, n_heads), 128, 0, s>>>(
                        prefill_proj_qkv, prefill_dn_beta, chunk_gate_cumul,
                        chunk_W, chunk_U,
                        B, L, NT, n_heads, dk, cfg.ssm_inner_size);

                    int BV = 32;
                    dim3 prop_grid((dv + BV - 1) / BV, B * n_heads);
                    kernel_chunkwise_state_propagation<<<prop_grid, 128, 0, s>>>(
                        batch_S_layer, prefill_proj_qkv,
                        chunk_W, chunk_U, chunk_gate_cumul,
                        chunk_h_states, chunk_v_new,
                        B, L, NT, n_heads, dk, dv, cfg.ssm_inner_size);

                    dim3 out_grid(1, NT * B, n_heads);
                    kernel_chunkwise_output<<<out_grid, 128, 0, s>>>(
                        prefill_proj_qkv, chunk_h_states, chunk_v_new, chunk_gate_cumul,
                        prefill_ffn_gate,
                        B, L, NT, n_heads, dk, dv, cfg.ssm_inner_size);
                }
            }

            // Batch gated RMSNorm
        
            kernel_batch_gated_rmsnorm<<<N * n_heads, 32, 0, s>>>(
                prefill_ffn_gate,
                static_cast<const float*>(w.ssm_norm.device_data),
                prefill_proj_gate,
                prefill_proj_gate,
                N, n_heads, dk, cfg.rms_norm_eps);

            // Batch output projection GEMM → pf_b
        
            gwen_gemm(w.ssm_out.device_data, w.ssm_out.type, prefill_temp_w,
                      prefill_proj_gate, pf_b,
                      w.ssm_out.shape[1], w.ssm_out.shape[0], N, s);

            // Batch residual add: pf_b += pf_a
        
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_b, pf_a, total);
            }

            // Batch post-attention RMSNorm
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // Batch FFN GEMMs
        
            gwen_gemm(w.ffn_gate.device_data, w.ffn_gate.type, prefill_temp_w,
                      pf_norm, prefill_ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], N, s);
            gwen_gemm(w.ffn_up.device_data, w.ffn_up.type, prefill_temp_w,
                      pf_norm, prefill_ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], N, s);

        
            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate, prefill_ffn_up, prefill_ffn_out, N, cfg.n_ff);
            }

        
            gwen_gemm(w.ffn_down.device_data, w.ffn_down.type, prefill_temp_w,
                      prefill_ffn_out, pf_a,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], N, s);

        
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_a, pf_b, total);
            }

        } else {
            // ===== Full Attention Layer (flash-attention, no KV cache) =====
            const auto& w = layer.full_attn;

            int attn_dim = cfg.n_head * cfg.head_dim;  // 2048
            int kv_dim = cfg.n_head_kv * cfg.head_dim;  // 512
            int q_proj_dim = w.attn_q.shape[1];  // 4096 (Q + gate interleaved)

            // Batch RMSNorm
        
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_a, static_cast<const float*>(w.attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // Batch Q, K, V projections via GEMM (ONE weight read each)
        
            gwen_gemm(w.attn_q.device_data, w.attn_q.type, prefill_temp_w,
                      pf_norm, prefill_proj_qkv,  // [N, q_proj_dim=4096]
                      q_proj_dim, w.attn_q.shape[0], N, s);
            gwen_gemm(w.attn_k.device_data, w.attn_k.type, prefill_temp_w,
                      pf_norm, prefill_ffn_gate,  // [N, kv_dim=512]
                      kv_dim, w.attn_k.shape[0], N, s);
            gwen_gemm(w.attn_v.device_data, w.attn_v.type, prefill_temp_w,
                      pf_norm, prefill_ffn_up,    // [N, kv_dim=512]
                      kv_dim, w.attn_v.shape[0], N, s);

            // Batch deinterleave Q+gate → prefill_proj_gate (Q) and prefill_ffn_out (gate)
        
            {
                int deint_blocks = (attn_dim + 255) / 256;
                dim3 grid(deint_blocks, N);
                kernel_deinterleave_qgate_batch<<<grid, 256, 0, s>>>(
                    prefill_proj_qkv, prefill_proj_gate, prefill_ffn_out,
                    N, cfg.n_head, cfg.head_dim);
            }

            // Batch Q RMSNorm (per-head): [N * n_head, head_dim]
            gwen_rmsnorm_batched_f32w(
                prefill_proj_gate, static_cast<const float*>(w.attn_q_norm.device_data),
                prefill_proj_gate, N * cfg.n_head, cfg.head_dim, cfg.rms_norm_eps, s);

            // Batch K RMSNorm (per-head): [N * n_kv_heads, head_dim]
            gwen_rmsnorm_batched_f32w(
                prefill_ffn_gate, static_cast<const float*>(w.attn_k_norm.device_data),
                prefill_ffn_gate, N * cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps, s);

            // Batch RoPE: positions = t % L
            {
                dim3 rope_grid(cfg.n_head + cfg.n_head_kv, N);
                int n_pairs = cfg.rope_dim / 2;
                int rope_threads = ((n_pairs + 31) / 32) * 32;
                if (rope_threads < 32) rope_threads = 32;
                if (rope_threads > 256) rope_threads = 256;
                kernel_rope_batch<<<rope_grid, rope_threads, 0, s>>>(
                    prefill_proj_gate, prefill_ffn_gate,
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                    L, cfg.rope_theta, cfg.rope_dim);
            }

            // Flash causal attention: one warp per (batch, head) pair
        
            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            kernel_batch_causal_attn<<<B * cfg.n_head, 32, 0, s>>>(
                prefill_proj_gate, prefill_ffn_gate, prefill_ffn_up,
                prefill_proj_qkv,  // reuse for attention output [N, n_head*head_dim]
                L, cfg.n_head, cfg.n_head_kv, cfg.head_dim, scale);

            // Batch sigmoid-mul: output = attn_out * sigmoid(gate)
        
            {
                int total = N * attn_dim;
                int blocks = (total + 255) / 256;
                kernel_sigmoid_mul_batch<<<blocks, 256, 0, s>>>(
                    prefill_proj_qkv, prefill_ffn_out, prefill_proj_gate,
                    total);
            }

            // Batch output projection GEMM → pf_b
        
            gwen_gemm(w.attn_output.device_data, w.attn_output.type, prefill_temp_w,
                      prefill_proj_gate, pf_b,
                      w.attn_output.shape[1], w.attn_output.shape[0], N, s);

            // Batch residual add: pf_b += pf_a
        
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_b, pf_a, total);
            }

            // Batch post-attention RMSNorm
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // Batch FFN GEMMs
        
            gwen_gemm(w.ffn_gate.device_data, w.ffn_gate.type, prefill_temp_w,
                      pf_norm, prefill_ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], N, s);
            gwen_gemm(w.ffn_up.device_data, w.ffn_up.type, prefill_temp_w,
                      pf_norm, prefill_ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], N, s);

        
            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate, prefill_ffn_up, prefill_ffn_out, N, cfg.n_ff);
            }

        
            gwen_gemm(w.ffn_down.device_data, w.ffn_down.type, prefill_temp_w,
                      prefill_ffn_out, pf_a,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], N, s);

        
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_a, pf_b, total);
            }
        }
    }

    // 3. Copy hidden states to host: pf_a holds [B*L, n_embed]

    // 3. Copy hidden states to host: pf_a holds [B*L, n_embed]
    GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
    size_t bytes = (size_t)N * cfg.n_embed * sizeof(half);
    GWEN_CHECK_CUDA(cudaMemcpy(output_host, pf_a, bytes, cudaMemcpyDeviceToHost));
}

// ============================================================
// Batch predictions from hidden states
// ============================================================

void InferenceState::predict_from_hidden(Model& model, half* hidden_gpu, int N, int32_t* preds_host) {
    const auto& cfg = model.config;
    cudaStream_t s = 0;

    // 1. Batch RMSNorm: hidden → prefill_norm [N, n_embed]
    kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
        hidden_gpu, static_cast<const float*>(model.output_norm.device_data),
        prefill_norm, N, cfg.n_embed, cfg.rms_norm_eps);

    // 2. Per-token: Q8_1 quantize + GEMV + argmax
    // Reuses single-token scratch buffers (x_q8_a, logits_h, logits_f, argmax_*)
    for (int t = 0; t < N; t++) {
        half* norm_t = prefill_norm + (size_t)t * cfg.n_embed;

        // Quantize normed hidden to Q8_1
        gwen_quantize_q8_1(norm_t, x_q8_a, cfg.n_embed, s);

        // GEMV: embed_tokens × normed → logits [n_vocab]
        gwen_gemv_dp4a(model.token_embd.device_data, x_q8_a, logits_h,
                       cfg.n_vocab, cfg.n_embed, model.token_embd.type, s);

        // FP16 → FP32 for argmax
        int logit_blocks = (cfg.n_vocab + 255) / 256;
        kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, cfg.n_vocab);

        // Argmax
        kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(
            logits_f, argmax_partial_max, argmax_partial_idx, cfg.n_vocab);
        kernel_argmax_reduce<<<1, 256, 0, s>>>(
            argmax_partial_max, argmax_partial_idx, d_argmax_token, ARGMAX_BLOCKS);

        // Copy single prediction to host
        GWEN_CHECK_CUDA(cudaMemcpyAsync(&preds_host[t], d_argmax_token,
                                          sizeof(int32_t), cudaMemcpyDeviceToHost, s));
    }

    GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
}

// ============================================================
// Generation loop
// ============================================================

std::vector<int> InferenceState::generate(Model& model, const std::vector<int>& prompt_tokens,
                                           int n_predict, bool greedy, float temperature,
                                           const std::vector<int>& teacher_tokens) {
    std::vector<int> output_tokens;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Prefill: process all prompt tokens at once
    if (max_prefill > 0 && (int)prompt_tokens.size() <= max_prefill) {
        int next = forward_prefill(model, prompt_tokens);
        output_tokens.push_back(next);
    } else {
        // Fallback: sequential processing
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            int next = forward(model, prompt_tokens[i]);
            if (i == (int)prompt_tokens.size() - 1) {
                output_tokens.push_back(next);
            }
        }
    }

    GWEN_CHECK_CUDA(cudaDeviceSynchronize());
    auto t_prefill = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
    fprintf(stderr, "TTFT: %.1f ms (%.0f prompt tok/s)\n", ttft_ms,
           prompt_tokens.size() / (ttft_ms / 1000.0));

    // Decode loop
    // forward(): optimized GEMV + CUDA graph path (default, 6x faster).
    // forward_prefill(N=1): verified GEMM reference path, enable with GWEN_GEMM_DECODE=1.
    bool gemm_decode = (getenv("GWEN_GEMM_DECODE") != nullptr);
    bool dump_logits = (getenv("GWEN_DUMP_LOGITS") != nullptr);
    bool teacher_forcing = !teacher_tokens.empty();

    // Binary logit dump: GWEN_LOGITS_BIN=path writes same format as llama_golden
    const char* logits_bin_path = getenv("GWEN_LOGITS_BIN");
    FILE* logits_bin_fp = nullptr;
    if (logits_bin_path) {
        logits_bin_fp = fopen(logits_bin_path, "wb");
        int32_t header[2] = {n_predict, (int32_t)model.config.n_vocab};
        fwrite(header, sizeof(int32_t), 2, logits_bin_fp);
        // Write position 0 (from prefill)
        int32_t tok0 = output_tokens[0];
        fwrite(&tok0, sizeof(int32_t), 1, logits_bin_fp);
        std::vector<float> host_logits(model.config.n_vocab);
        GWEN_CHECK_CUDA(cudaMemcpy(host_logits.data(), logits_f,
            model.config.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(host_logits.data(), sizeof(float), model.config.n_vocab, logits_bin_fp);
    }

    for (int i = 1; i < n_predict; i++) {
        // In teacher-forcing mode, feed the reference token instead of own prediction.
        // This keeps GWEN's state synchronized with the reference engine so we can
        // compare logits at every position without cascade divergence.
        int input_token;
        if (teacher_forcing && (i - 1) < (int)teacher_tokens.size()) {
            input_token = teacher_tokens[i - 1];
        } else {
            input_token = output_tokens.back();
        }

        int next;
        if (gemm_decode) {
            next = forward_prefill(model, {input_token});
        } else {
            next = forward(model, input_token);
        }
        output_tokens.push_back(next);

        if (dump_logits) {
            // Copy logits to host and print top-5
            std::vector<float> host_logits(model.config.n_vocab);
            GWEN_CHECK_CUDA(cudaMemcpy(host_logits.data(), logits_f,
                model.config.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<std::pair<float,int>> scored;
            for (int j = 0; j < (int)model.config.n_vocab; j++)
                scored.push_back({host_logits[j], j});
            std::sort(scored.begin(), scored.end(),
                [](auto& a, auto& b) { return a.first > b.first; });
            fprintf(stderr, "  [%d] token=%d logit=%.4f  top5:", i, next, scored[0].first);
            for (int j = 0; j < 5; j++)
                fprintf(stderr, " %d(%.2f)", scored[j].second, scored[j].first);
            fprintf(stderr, "\n");
        }

        if (logits_bin_fp) {
            int32_t tok = next;
            fwrite(&tok, sizeof(int32_t), 1, logits_bin_fp);
            std::vector<float> hl(model.config.n_vocab);
            GWEN_CHECK_CUDA(cudaMemcpy(hl.data(), logits_f,
                model.config.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
            fwrite(hl.data(), sizeof(float), model.config.n_vocab, logits_bin_fp);
        }

        if (!teacher_forcing && next == (int)model.config.eos_token_id) break;
    }
    if (logits_bin_fp) fclose(logits_bin_fp);

    return output_tokens;
}

// ============================================================
// MTP Forward Pass
// ============================================================
// Computes: h' = FC( concat( RMSNorm(hidden), RMSNorm(embed(token)) ) )
//           h_mtp = FullAttnLayer(h')
//           logits = lm_head( RMSNorm(h_mtp) )
// Uses mtp_hidden (saved by forward_body after all 24 layers).
// Reuses scratch buffers (safe since MTP runs after main forward completes).

// Kernel: concat two half vectors into one
__global__ void __launch_bounds__(256)
kernel_concat_half(const half* __restrict__ a, const half* __restrict__ b,
                   half* __restrict__ out, int dim_a, int dim_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim_a + dim_b;
    if (idx >= total) return;
    out[idx] = (idx < dim_a) ? a[idx] : b[idx - dim_a];
}

void InferenceState::forward_mtp_body(Model& model, cudaStream_t s) {
    const auto& cfg = model.config;
    const auto& mtp_w = model.mtp;
    const bool quantized = (mtp_w.fc.type == GGMLType::Q8_0);

    // 1. Embed the input token → buf_b
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      d_mtp_token, buf_b, cfg.n_embed, s);

    // 2. RMSNorm both streams → mtp_concat = [norm_embed, norm_hidden]
    half* norm_embed = mtp_concat;              // [0 .. n_embed)
    half* norm_hidden = mtp_concat + cfg.n_embed; // [n_embed .. 2*n_embed)

    gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(mtp_w.pre_fc_norm_embed.device_data),
                      norm_embed, cfg.n_embed, cfg.rms_norm_eps, s);
    gwen_rmsnorm_f32w(mtp_hidden, static_cast<const float*>(mtp_w.pre_fc_norm_hidden.device_data),
                      norm_hidden, cfg.n_embed, cfg.rms_norm_eps, s);

    // 3. FC projection: mtp_concat [2*n_embed] → buf_a [n_embed]
    if (quantized) {
        // Q8_0 GEMV takes FP16 input directly (no Q8_1 pre-quantization needed)
        gwen_gemv(mtp_w.fc.device_data, mtp_concat, buf_a,
                  cfg.n_embed, 2 * cfg.n_embed, mtp_w.fc.type, s);
    } else {
        gwen_gemv_fp16(static_cast<const half*>(mtp_w.fc.device_data),
                       mtp_concat, buf_a,
                       cfg.n_embed, 2 * cfg.n_embed, s);
    }

    // 4. Full attention layer
    {
        const auto& w = mtp_w.layer;
        auto& cache = mtp_kv_cache;

        int attn_dim = cfg.n_head * cfg.head_dim;  // 2048
        int kv_dim = cfg.n_head_kv * cfg.head_dim;  // 512
        int q_proj_dim = attn_dim * 2;  // 4096 (Q + gate)

        // RMSNorm → x_norm
        gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                          x_norm, cfg.n_embed, cfg.rms_norm_eps, s);

        // Q/K/V projections
        if (quantized) {
            gwen_gemv(w.attn_q.device_data, x_norm, qkv,
                      q_proj_dim, cfg.n_embed, w.attn_q.type, s);
            gwen_gemv(w.attn_k.device_data, x_norm, fa_k,
                      kv_dim, cfg.n_embed, w.attn_k.type, s);
            gwen_gemv(w.attn_v.device_data, x_norm, fa_v,
                      kv_dim, cfg.n_embed, w.attn_v.type, s);
        } else {
            gwen_gemv_fp16(static_cast<const half*>(w.attn_q.device_data),
                           x_norm, qkv, q_proj_dim, cfg.n_embed, s);
            gwen_gemv_fp16(static_cast<const half*>(w.attn_k.device_data),
                           x_norm, fa_k, kv_dim, cfg.n_embed, s);
            gwen_gemv_fp16(static_cast<const half*>(w.attn_v.device_data),
                           x_norm, fa_v, kv_dim, cfg.n_embed, s);
        }

        // Deinterleave Q and gate
        {
            int deint_blocks = (attn_dim + 255) / 256;
            kernel_deinterleave_qgate<<<deint_blocks, 256, 0, s>>>(
                qkv, fa_q, gated_out, cfg.n_head, cfg.head_dim);
        }

        // QK norms
        gwen_rmsnorm_batched_f32w(fa_q, static_cast<const float*>(w.attn_q_norm.device_data),
                                  fa_q, cfg.n_head, cfg.head_dim, cfg.rms_norm_eps, s);
        gwen_rmsnorm_batched_f32w(fa_k, static_cast<const float*>(w.attn_k_norm.device_data),
                                  fa_k, cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps, s);

        // RoPE
        gwen_rope(fa_q, fa_k,
                  cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                  d_mtp_pos, cfg.rope_theta, cfg.rope_sections, cfg.rope_dim, s);

        // KV cache store
        int kv_blocks = (kv_dim + 255) / 256;
        kernel_kv_cache_store<<<kv_blocks, 256, 0, s>>>(
            cache.k_cache, cache.v_cache, fa_k, fa_v, d_mtp_pos, kv_dim);

        // GQA attention
        float scale = 1.0f / sqrtf((float)cfg.head_dim);
        kernel_gqa_attention_decode<<<cfg.n_head, 256, 0, s>>>(
            fa_q, cache.k_cache, cache.v_cache,
            attn_out, attn_scores, d_mtp_pos,
            cfg.n_head, cfg.n_head_kv, cfg.head_dim, mtp_kv_cache.max_seq, scale);

        // Apply gate: attn_out *= sigmoid(gated_out)
        gwen_sigmoid_mul(attn_out, gated_out, gated_out, attn_dim, s);

        // Output projection + residual add → buf_b = o_proj(gated_out) + buf_a
        if (quantized) {
            gwen_gemv(w.attn_output.device_data, gated_out, buf_b,
                      cfg.n_embed, attn_dim, w.attn_output.type, s);
            gwen_add_inplace(buf_b, buf_a, cfg.n_embed, s);
        } else {
            gwen_gemv_fp16_residual(static_cast<const half*>(w.attn_output.device_data),
                                    gated_out, buf_b, buf_a,
                                    cfg.n_embed, attn_dim, s);
        }

        // Post-attention RMSNorm
        gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                          x_norm, cfg.n_embed, cfg.rms_norm_eps, s);

        // FFN
        if (quantized) {
            gwen_gemv(w.ffn_gate.device_data, x_norm, ffn_gate,
                      cfg.n_ff, cfg.n_embed, w.ffn_gate.type, s);
            gwen_gemv(w.ffn_up.device_data, x_norm, ffn_up,
                      cfg.n_ff, cfg.n_embed, w.ffn_up.type, s);
            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff, s);
            gwen_gemv(w.ffn_down.device_data, ffn_out, buf_a,
                      cfg.n_embed, cfg.n_ff, w.ffn_down.type, s);
            gwen_add_inplace(buf_a, buf_b, cfg.n_embed, s);
        } else {
            gwen_gemv_fp16(static_cast<const half*>(w.ffn_gate.device_data),
                           x_norm, ffn_gate, cfg.n_ff, cfg.n_embed, s);
            gwen_gemv_fp16(static_cast<const half*>(w.ffn_up.device_data),
                           x_norm, ffn_up, cfg.n_ff, cfg.n_embed, s);
            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff, s);
            gwen_gemv_fp16_residual(static_cast<const half*>(w.ffn_down.device_data),
                                    ffn_out, buf_a, buf_b,
                                    cfg.n_embed, cfg.n_ff, s);
        }
    }

    // 5. Output norm + LM head + argmax
    gwen_rmsnorm_quantize_q8_1(buf_a, static_cast<const float*>(mtp_w.output_norm.device_data),
                      x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);

    if (model.has_reduced_lm_head) {
        // Reduced LM head: only score top-K (or K+1 with IDK) tokens
        const auto& rl = model.reduced_lm_head;
        int K = rl.K;
        int lm_rows = rl.has_idk ? K + 1 : K;
        if (rl.type == GGMLType::F16) {
            // FP16 reduced lm_head: use RMSNorm output directly (not Q8_1)
            half* norm_out = x_norm;
            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(mtp_w.output_norm.device_data),
                              norm_out, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_gemv_fp16(static_cast<const half*>(rl.weights.device_data),
                           norm_out, logits_h, lm_rows, cfg.n_embed, s);
        } else {
            gwen_gemv_dp4a(rl.weights.device_data, x_q8_a, logits_h,
                      lm_rows, cfg.n_embed, rl.type, s);
        }

        int logit_blocks = (lm_rows + 255) / 256;
        kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, lm_rows);
        kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(logits_f, argmax_partial_max, argmax_partial_idx, lm_rows);
        kernel_argmax_reduce<<<1, 256, 0, s>>>(argmax_partial_max, argmax_partial_idx, d_argmax_token, ARGMAX_BLOCKS);
        // Confidence gate: skip speculation if max softmax prob < threshold
        if (mtp_confidence_threshold > 0.0f) {
            kernel_confidence_gate<<<1, 1, 0, s>>>(logits_f, d_argmax_token, lm_rows, mtp_confidence_threshold);
        }
        // Remap index → real token ID (IDK: index < 0 or >= K maps to -1)
        if (rl.has_idk || mtp_confidence_threshold > 0.0f) {
            kernel_remap_token_idk<<<1, 1, 0, s>>>(rl.d_token_ids, d_argmax_token, K);
        } else {
            kernel_remap_token<<<1, 1, 0, s>>>(rl.d_token_ids, d_argmax_token);
        }
    } else {
        // Full LM head: score all vocab tokens
        gwen_gemv_dp4a(model.token_embd.device_data, x_q8_a, logits_h,
                  cfg.n_vocab, cfg.n_embed, model.token_embd.type, s);

        int logit_blocks = (cfg.n_vocab + 255) / 256;
        kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, cfg.n_vocab);
        kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(logits_f, argmax_partial_max, argmax_partial_idx, cfg.n_vocab);
        kernel_argmax_reduce<<<1, 256, 0, s>>>(argmax_partial_max, argmax_partial_idx, d_argmax_token, ARGMAX_BLOCKS);
    }
}

int InferenceState::forward_mtp(Model& model, int token_id) {
    int mtp_params[2] = {token_id, mtp_pos};
    GWEN_CHECK_CUDA(cudaMemcpyAsync(d_mtp_token, mtp_params, 2 * sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    if (!mtp_graph_captured) {
        cudaGraph_t graph;
        GWEN_CHECK_CUDA(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal));
        forward_mtp_body(model, compute_stream);
        GWEN_CHECK_CUDA(cudaStreamEndCapture(compute_stream, &graph));
        GWEN_CHECK_CUDA(cudaGraphInstantiate(&mtp_graph_exec, graph, nullptr, nullptr, 0));
        GWEN_CHECK_CUDA(cudaGraphDestroy(graph));
        mtp_graph_captured = true;
    }

    GWEN_CHECK_CUDA(cudaGraphLaunch(mtp_graph_exec, compute_stream));
    GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));

    int draft_token;
    GWEN_CHECK_CUDA(cudaMemcpy(&draft_token, d_argmax_token, sizeof(int), cudaMemcpyDeviceToHost));
    mtp_pos++;
    return draft_token;
}

// ============================================================
// Speculative Generation with MTP + activation replay
// ============================================================
// Uses forward_2tok() to verify draft + get bonus in ONE pass that reads
// model weights only once. This halves the bandwidth of the verify step.
//
// Each cycle: forward_2tok(last_emitted, draft) → pred_a, pred_b
//   Accept: pred_a == draft → emit draft, emit pred_b → MTP(pred_b) → new draft
//   Reject: pred_a != draft → undo token B state (~0.05ms), emit pred_a → MTP(pred_a)
//
// Activation replay eliminates the 1.66ms re-forward on reject by inverting
// the DeltaNet state update using saved activations from forward_body_2tok.

std::vector<int> InferenceState::generate_speculative(Model& model,
                                                       const std::vector<int>& prompt_tokens,
                                                       int n_predict) {
    std::vector<int> output_tokens;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Prefill
    if (max_prefill > 0 && (int)prompt_tokens.size() <= max_prefill) {
        int next = forward_prefill(model, prompt_tokens);
        output_tokens.push_back(next);
    } else {
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            int next = forward(model, prompt_tokens[i]);
            if (i == (int)prompt_tokens.size() - 1) {
                output_tokens.push_back(next);
            }
        }
    }

    GWEN_CHECK_CUDA(cudaDeviceSynchronize());
    auto t_prefill = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
    fprintf(stderr, "TTFT: %.1f ms (%.0f prompt tok/s)\n", ttft_ms,
           prompt_tokens.size() / (ttft_ms / 1000.0));

    // First decode step: process the prefill prediction to set mtp_hidden
    if (output_tokens.size() < (size_t)n_predict) {
        int next = forward(model, output_tokens.back());
        output_tokens.push_back(next);
        if (next == (int)model.config.eos_token_id) return output_tokens;
    }

    // Generate first MTP draft
    int draft = forward_mtp(model, output_tokens.back());

    if (!batch2_allocated) {
        // === FALLBACK: CPU-driven speculative decode (no batch2 buffers) ===
        int idk_count = 0;
        while ((int)output_tokens.size() < n_predict) {
            if (draft == -1) {
                // IDK: skip speculation, fall back to regular forward
                int next = forward(model, output_tokens.back());
                output_tokens.push_back(next);
                idk_count++;
                if (next == (int)model.config.eos_token_id) break;
                if ((int)output_tokens.size() >= n_predict) break;
                draft = forward_mtp(model, next);
                continue;
            }
            int last_token = output_tokens.back();
            int pred = forward(model, last_token);
            if (pred == draft) {
                output_tokens.push_back(draft);
            } else {
                output_tokens.push_back(pred);
            }
            int emitted = output_tokens.back();
            if (emitted == (int)model.config.eos_token_id) break;
            if ((int)output_tokens.size() >= n_predict) break;
            int bonus = forward(model, emitted);
            output_tokens.push_back(bonus);
            if (bonus == (int)model.config.eos_token_id) break;
            draft = forward_mtp(model, bonus);
        }
        if (idk_count > 0) {
            fprintf(stderr, "MTP IDK: %d tokens skipped speculation\n", idk_count);
        }
        if ((int)output_tokens.size() > n_predict) output_tokens.resize(n_predict);
        return output_tokens;
    }

    // Speculative decode loop: forward_2tok verifies draft + gets bonus token in one pass.
    // CPU dispatches two CUDA graphs per cycle (forward_2tok + MTP) with one int comparison
    // between them. CPU overhead is ~30 μs per 1.6 ms cycle (~2% of wall time).
    {
        int accepted = 0, rejected = 0, idk_count = 0;
        while ((int)output_tokens.size() < n_predict) {
            if (draft == -1) {
                // IDK: skip forward_2tok, fall back to regular forward
                int next = forward(model, output_tokens.back());
                output_tokens.push_back(next);
                idk_count++;
                if (next == (int)model.config.eos_token_id) break;
                if ((int)output_tokens.size() >= n_predict) break;
                draft = forward_mtp(model, next);
                continue;
            }
            int last_token = output_tokens.back();
            auto [pred_a, pred_b] = forward_2tok(model, last_token, draft);
            if (pred_a == draft) {
                output_tokens.push_back(draft);
                output_tokens.push_back(pred_b);
                accepted++;
                if (pred_b == (int)model.config.eos_token_id) break;
                if ((int)output_tokens.size() >= n_predict) break;
                draft = forward_mtp(model, pred_b);
            } else {
                rejected++;
                undo_deltanet_token_b(compute_stream);
                pos -= 1;
                GWEN_CHECK_CUDA(cudaMemcpyAsync(mtp_hidden, mtp_hidden_b,
                    model.config.n_embed * sizeof(half),
                    cudaMemcpyDeviceToDevice, compute_stream));
                GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));
                output_tokens.push_back(pred_a);
                if (pred_a == (int)model.config.eos_token_id) break;
                if ((int)output_tokens.size() >= n_predict) break;
                draft = forward_mtp(model, pred_a);
            }
        }
        if ((int)output_tokens.size() > n_predict) output_tokens.resize(n_predict);
        int total = accepted + rejected;
        fprintf(stderr, "MTP stats: %d accepted, %d rejected, %d IDK (%.1f%% acceptance rate, %.1f%% IDK), %d cycles\n",
               accepted, rejected, idk_count,
               total > 0 ? 100.0 * accepted / total : 0.0,
               (total + idk_count) > 0 ? 100.0 * idk_count / (total + idk_count) : 0.0,
               total + idk_count);
        return output_tokens;
    }
}

} // namespace gwen
