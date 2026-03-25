#include "gwen/inference.h"
#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

#include <cuda_fp8.h>  // __nv_fp8_e4m3 used in DeltaNet fused kernel FP8 paths (legacy, never called for K-quant)
#include <cublas_v2.h>
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
// Convert alpha/beta GEMV outputs → decay/beta scalars
// ============================================================
// alpha_proj: [n_heads] half — dot product output
// beta_proj: [n_heads] half — dot product output
// ssm_a: [n_heads] float — log decay rate
// dt_bias: [n_heads] float — dt bias
// decay_out: [n_heads] float — exp(ssm_a * softplus(alpha_proj + dt_bias))
// beta_out: [n_heads] float — sigmoid(beta_proj)
__global__ void __launch_bounds__(32)
kernel_alpha_beta_to_decay(
    const half* __restrict__ alpha_proj,
    const half* __restrict__ beta_proj,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ decay_out,
    float* __restrict__ beta_out,
    int n_heads)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;
    float alpha_val = __half2float(alpha_proj[h]);
    float beta_val = __half2float(beta_proj[h]);
    float biased = alpha_val + dt_bias[h];
    float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
    decay_out[h] = expf(ssm_a[h] * sp);
    beta_out[h] = 1.0f / (1.0f + expf(-beta_val));
}

// Batch version: converts [N, n_heads] FP16 alpha/beta projections to gate/beta
// Gate is stored as ssm_a * softplus(alpha + dt_bias) — NOT exp'd yet.
// The DeltaNet prefill kernel applies exp() internally.
// Grid: (ceil(n_heads/32), N), Block: 32
__global__ void __launch_bounds__(32)
kernel_batch_alpha_beta_to_gate(
    const half* __restrict__ alpha_proj,  // [N, n_heads] FP16
    const half* __restrict__ beta_proj,   // [N, n_heads] FP16
    const float* __restrict__ ssm_a,      // [n_heads]
    const float* __restrict__ dt_bias,    // [n_heads]
    float* __restrict__ gate_out,         // [N, n_heads] — ssm_a * softplus(alpha + bias)
    float* __restrict__ beta_out,         // [N, n_heads] — sigmoid(beta)
    int N, int n_heads)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= N) return;
    int idx = t * n_heads + h;
    float alpha_val = __half2float(alpha_proj[idx]);
    float beta_val = __half2float(beta_proj[idx]);
    float biased = alpha_val + dt_bias[h];
    float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
    gate_out[idx] = ssm_a[h] * sp;  // NOT exp'd — DeltaNet kernel applies exp()
    beta_out[idx] = 1.0f / (1.0f + expf(-beta_val));
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

// Batch2: Gated RMSNorm for two tokens in one launch
// Grid: (n_heads, 2) — blockIdx.y selects token
__global__ void __launch_bounds__(32)
kernel_gated_rmsnorm_batch2(
    const half* __restrict__ x0, const half* __restrict__ x1,
    const float* __restrict__ weight,
    const half* __restrict__ gate0, const half* __restrict__ gate1,
    half* __restrict__ out0, half* __restrict__ out1,
    int n_heads, int dim_per_head, float eps)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    const half* x = (blockIdx.y == 0) ? x0 : x1;
    const half* gate = (blockIdx.y == 0) ? gate0 : gate1;
    half* output = (blockIdx.y == 0) ? out0 : out1;
    int lane = threadIdx.x;
    int base = head * dim_per_head;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim_per_head; i += 32) {
        float val = __half2float(x[base + i]);
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
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
    const float* __restrict__ alpha_scales,
    const float* __restrict__ beta_scales,
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

    // Gate and beta dot products (FP8 E4M3 with per-head scales)
    const uint8_t* alpha_fp8 = static_cast<const uint8_t*>(alpha_w);
    const uint8_t* beta_fp8 = static_cast<const uint8_t*>(beta_w);
    float a_scale = alpha_scales[head];
    float b_scale = beta_scales[head];
    float aa = 0.0f, ba = 0.0f;
    for (int i = tid; i < n_embed; i += 128) {
        __nv_fp8_e4m3 a_fp8, b_fp8;
        *reinterpret_cast<uint8_t*>(&a_fp8) = alpha_fp8[head * n_embed + i];
        *reinterpret_cast<uint8_t*>(&b_fp8) = beta_fp8[head * n_embed + i];
        float x_val = __half2float(x_norm[i]);
        aa += (float(a_fp8) * a_scale) * x_val;
        ba += (float(b_fp8) * b_scale) * x_val;
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
    float* __restrict__ S,              // [n_v_heads, dk, dk] recurrent state
    half* __restrict__ q_in,            // [n_k_heads * dk] Q from conv1d
    half* __restrict__ k_in,            // [n_k_heads * dk] K from conv1d
    const half* __restrict__ v_in,      // [n_v_heads * dk] V from conv1d
    const half* __restrict__ x_norm,    // [n_embed] pre-attention norm (for gate/beta)
    const void* __restrict__ alpha_w,   // FP8 E4M3 [n_v_heads, n_embed] (NULL if pre-computed)
    const void* __restrict__ beta_w,    // FP8 E4M3 [n_v_heads, n_embed] (NULL if pre-computed)
    const float* __restrict__ alpha_scales, // [n_v_heads] per-row scales (or pre-computed decay)
    const float* __restrict__ beta_scales,  // [n_v_heads] per-row scales (or pre-computed beta)
    const float* __restrict__ ssm_a,    // [n_v_heads] (NULL if pre-computed)
    const float* __restrict__ dt_bias,  // [n_v_heads] (NULL if pre-computed)
    half* __restrict__ output,          // [n_v_heads * dk]
    float q_scale,                      // 1/sqrt(dk) for Q normalization
    int n_v_heads, int n_k_heads, int dk, int n_embed)
{
    int head = blockIdx.x;  // V head index
    if (head >= n_v_heads) return;
    // GGUF stores V-heads in tiled order: [K0_v0, K1_v0, ..., K0_v1, K1_v1, ...]
    // so the K-head for V-head `head` is head % n_k_heads (not head * n_k_heads / n_v_heads)
    int k_head = (n_k_heads == n_v_heads) ? head : head % n_k_heads;
    int tid = threadIdx.x;  // 0..127
    int warp_id = tid / 32;
    int lane = tid % 32;
    half* q = q_in + k_head * dk;
    half* k = k_in + k_head * dk;

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
    if (alpha_w != nullptr) {
        // FP8 E4M3 path: compute gate/beta dot products inline
        const uint8_t* alpha_fp8 = static_cast<const uint8_t*>(alpha_w);
        const uint8_t* beta_fp8 = static_cast<const uint8_t*>(beta_w);
        float alpha_scale = alpha_scales[head];
        float beta_scale = beta_scales[head];

        float alpha_acc = 0.0f, beta_acc = 0.0f;
        for (int i = tid; i < n_embed; i += 128) {
            __nv_fp8_e4m3 a_fp8, b_fp8;
            *reinterpret_cast<uint8_t*>(&a_fp8) = alpha_fp8[head * n_embed + i];
            *reinterpret_cast<uint8_t*>(&b_fp8) = beta_fp8[head * n_embed + i];
            float x_val = __half2float(x_norm[i]);
            alpha_acc += (float(a_fp8) * alpha_scale) * x_val;
            beta_acc += (float(b_fp8) * beta_scale) * x_val;
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
    } else {
        // Pre-computed path: decay and beta passed via alpha_scales/beta_scales
        if (tid == 0) {
            s_decay = alpha_scales[head];
            s_beta = beta_scales[head];
        }
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
    float* __restrict__ S,              // [n_v_heads, dk, dk] recurrent state
    // Token A inputs
    half* __restrict__ q_in_a,          // [n_k_heads * dk]
    half* __restrict__ k_in_a,          // [n_k_heads * dk]
    const half* __restrict__ v_in_a,    // [n_v_heads * dk]
    const half* __restrict__ x_norm_a,  // [n_embed]
    // Token B inputs
    half* __restrict__ q_in_b,          // [n_k_heads * dk]
    half* __restrict__ k_in_b,          // [n_k_heads * dk]
    const half* __restrict__ v_in_b,    // [n_v_heads * dk]
    const half* __restrict__ x_norm_b,  // [n_embed]
    // Shared weights
    const void* __restrict__ alpha_w,   // [n_v_heads, n_embed]
    const void* __restrict__ beta_w,    // [n_v_heads, n_embed]
    const float* __restrict__ alpha_scales, // [n_v_heads] per-row scales
    const float* __restrict__ beta_scales,  // [n_v_heads] per-row scales
    const float* __restrict__ ssm_a,    // [n_v_heads]
    const float* __restrict__ dt_bias,  // [n_v_heads]
    // Outputs
    half* __restrict__ output_a,        // [n_v_heads * dk]
    half* __restrict__ output_b,        // [n_v_heads * dk]
    // S snapshot buffer (written between A and B)
    float* __restrict__ S_snapshot,     // [n_v_heads * dk * dk] or nullptr
    float q_scale,
    int n_v_heads, int n_k_heads, int dk, int n_embed)
{
    int head = blockIdx.x;  // V head index
    if (head >= n_v_heads) return;
    int k_head = (n_k_heads == n_v_heads) ? head : head % n_k_heads;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Dynamic shared memory: S matrix [128][128] = 64 KB
    extern __shared__ float sh_S[];
    __shared__ float sh_q[128], sh_k[128];
    __shared__ float warp_buf[4];
    __shared__ float s_q_inv, s_k_inv, s_decay, s_beta;

    float* S_head = S + head * dk * dk;
    const bool precomputed = (alpha_w == nullptr);

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

    // Phase 1: L2-normalize Q_a and K_a (indexed by K head)
    {
        half* q = q_in_a + k_head * dk;
        half* k = k_in_a + k_head * dk;
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
    if (precomputed) {
        // Pre-computed path: decay_a at alpha_scales[head], beta_a at beta_scales[head]
        if (tid == 0) {
            s_decay = alpha_scales[head];
            s_beta = beta_scales[head];
        }
    } else {
        const uint8_t* alpha_fp8 = static_cast<const uint8_t*>(alpha_w);
        const uint8_t* beta_fp8 = static_cast<const uint8_t*>(beta_w);
        float a_scale = alpha_scales[head];
        float b_scale = beta_scales[head];
        float alpha_acc = 0.0f, beta_acc = 0.0f;
        for (int i = tid; i < n_embed; i += 128) {
            __nv_fp8_e4m3 af, bf;
            *reinterpret_cast<uint8_t*>(&af) = alpha_fp8[head * n_embed + i];
            *reinterpret_cast<uint8_t*>(&bf) = beta_fp8[head * n_embed + i];
            float x_val = __half2float(x_norm_a[i]);
            alpha_acc += (float(af) * a_scale) * x_val;
            beta_acc += (float(bf) * b_scale) * x_val;
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
        half* q = q_in_b + k_head * dk;
        half* k = k_in_b + k_head * dk;
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
    if (precomputed) {
        // Pre-computed path: decay_b at alpha_scales[n_v_heads + head], beta_b at beta_scales[n_v_heads + head]
        if (tid == 0) {
            s_decay = alpha_scales[n_v_heads + head];
            s_beta = beta_scales[n_v_heads + head];
        }
    } else {
        const uint8_t* alpha_fp8 = static_cast<const uint8_t*>(alpha_w);
        const uint8_t* beta_fp8 = static_cast<const uint8_t*>(beta_w);
        float a_scale = alpha_scales[head];
        float b_scale = beta_scales[head];
        float alpha_acc = 0.0f, beta_acc = 0.0f;
        for (int i = tid; i < n_embed; i += 128) {
            __nv_fp8_e4m3 af, bf;
            *reinterpret_cast<uint8_t*>(&af) = alpha_fp8[head * n_embed + i];
            *reinterpret_cast<uint8_t*>(&bf) = beta_fp8[head * n_embed + i];
            float x_val = __half2float(x_norm_b[i]);
            alpha_acc += (float(af) * a_scale) * x_val;
            beta_acc += (float(bf) * b_scale) * x_val;
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
    const void* __restrict__ alpha_w,    // FP8 E4M3 [n_heads, n_embed]
    const void* __restrict__ beta_w,     // FP8 E4M3 [n_heads, n_embed]
    const float* __restrict__ alpha_scales, // [n_heads]
    const float* __restrict__ beta_scales,  // [n_heads]
    const float* __restrict__ ssm_a,     // [n_heads] A parameter (negative)
    const float* __restrict__ dt_bias,   // [n_heads]
    float* __restrict__ gate_out,        // [n_heads] gate = ssm_a * softplus(alpha_proj + dt_bias)
    float* __restrict__ beta_out,        // [n_heads] sigmoid(beta_proj)
    int n_embed, int n_heads)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int lane = threadIdx.x;

    const uint8_t* alpha_fp8 = static_cast<const uint8_t*>(alpha_w);
    const uint8_t* beta_fp8 = static_cast<const uint8_t*>(beta_w);
    float a_scale = alpha_scales[head];
    float b_scale = beta_scales[head];

    float alpha_acc = 0.0f;
    float beta_acc = 0.0f;

    for (int i = lane; i < n_embed; i += 32) {
        __nv_fp8_e4m3 af, bf;
        *reinterpret_cast<uint8_t*>(&af) = alpha_fp8[head * n_embed + i];
        *reinterpret_cast<uint8_t*>(&bf) = beta_fp8[head * n_embed + i];
        float x_val = __half2float(x[i]);
        alpha_acc += (float(af) * a_scale) * x_val;
        beta_acc += (float(bf) * b_scale) * x_val;
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
    const void* __restrict__ alpha_w,     // FP8 E4M3 [n_heads, n_embed]
    const void* __restrict__ beta_w,      // FP8 E4M3 [n_heads, n_embed]
    const float* __restrict__ alpha_scales, // [n_heads] per-row scales
    const float* __restrict__ beta_scales,  // [n_heads] per-row scales
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
    const uint8_t* alpha_fp8 = static_cast<const uint8_t*>(alpha_w);
    const uint8_t* beta_fp8 = static_cast<const uint8_t*>(beta_w);
    float a_scale = alpha_scales[head];
    float b_scale = beta_scales[head];

    float alpha_acc = 0.0f;
    float beta_acc = 0.0f;

    for (int i = lane; i < n_embed; i += 32) {
        __nv_fp8_e4m3 a_fp8, b_fp8;
        *reinterpret_cast<uint8_t*>(&a_fp8) = alpha_fp8[head * n_embed + i];
        *reinterpret_cast<uint8_t*>(&b_fp8) = beta_fp8[head * n_embed + i];
        float x_val = __half2float(x[i]);
        alpha_acc += (float(a_fp8) * a_scale) * x_val;
        beta_acc += (float(b_fp8) * b_scale) * x_val;
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

// L2-normalize Q and K heads within interleaved QKV tensor.
// One warp per (token, K-head). Q gets extra_scale, K gets scale 1.0.
// Grid: N * n_k_heads, Block: 32
__global__ void __launch_bounds__(32)
kernel_l2_normalize_qkv_batch(
    half* __restrict__ qkv,       // [N, qkv_dim] after conv1d+silu
    int N, int qkv_stride, int q_width, int n_k_heads, int dk, float q_scale)
{
    int idx = blockIdx.x;
    int t = idx / n_k_heads;
    int h = idx % n_k_heads;
    int lane = threadIdx.x;
    if (t >= N) return;

    half* q = qkv + (size_t)t * qkv_stride + h * dk;
    half* k = q + q_width;

    // L2 normalize Q with q_scale
    float sq = 0.0f;
    for (int i = lane; i < dk; i += 32) {
        float v = __half2float(q[i]);
        sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_xor_sync(0xFFFFFFFF, sq, off);
    float q_inv = rsqrtf(fmaxf(sq, 1e-12f)) * q_scale;
    for (int i = lane; i < dk; i += 32)
        q[i] = __float2half(__half2float(q[i]) * q_inv);

    // L2 normalize K
    sq = 0.0f;
    for (int i = lane; i < dk; i += 32) {
        float v = __half2float(k[i]);
        sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_xor_sync(0xFFFFFFFF, sq, off);
    float k_inv = rsqrtf(fmaxf(sq, 1e-12f));
    for (int i = lane; i < dk; i += 32)
        k[i] = __float2half(__half2float(k[i]) * k_inv);
}

// Fast DeltaNet prefill — S in registers, warp-per-column layout.
// Matches llama.cpp's gated_delta_net_cuda: pointer-increment inner loop,
// separate Q/K/V pointers (FP16, converted inline), stride-based addressing.
// Grid: (n_heads, 1, dv / num_warps), Block: (32, num_warps=4)
// Q and K MUST be pre-normalized (L2 norm + q_scale) before calling this kernel.
template <int DK = 128>
__global__ void __launch_bounds__(128, 2)
kernel_deltanet_prefill_fast(
    float* __restrict__ S,              // [n_v_heads, dk, dv] recurrent state (row-major)
    const half* __restrict__ qkv_batch, // [N, qkv_dim] after conv1d+silu
    const float* __restrict__ gate_batch, // [N, n_v_heads]
    const float* __restrict__ beta_batch, // [N, n_v_heads]
    half* __restrict__ output_batch,    // [N, ssm_inner]
    int N, int n_v_heads, int n_k_heads, int dk, int dv, int qkv_dim, int ssm_inner,
    float q_scale)
{
    const int head = blockIdx.x;  // V-head index
    const int col  = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    if (col >= dv) return;

    constexpr int WARP_SIZE = 32;
    constexpr int ROWS_PER_LANE = DK / WARP_SIZE;  // 4

    // Load S column into registers
    float* S_col = S + (size_t)head * dk * dv + (size_t)col;
    float s_reg[ROWS_PER_LANE];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
        s_reg[r] = S_col[(r * WARP_SIZE + lane) * dv];
    }

    // QKV layout: [Q(n_k_heads*dk), K(n_k_heads*dk), V(n_v_heads*dv)]
    // GGUF tiled order: k_head = head % n_k_heads
    const int k_head = (n_k_heads == n_v_heads) ? head : head % n_k_heads;
    const int q_width = n_k_heads * dk;
    const half* q_ptr = qkv_batch + k_head * dk;      // Q for K-head, token 0
    const half* k_ptr = qkv_batch + q_width + k_head * dk; // K for K-head, token 0
    const half* v_ptr = qkv_batch + 2 * q_width + head * dv; // V for V-head, token 0
    const float* g_ptr = gate_batch + head;
    const float* b_ptr = beta_batch + head;
    half* o_ptr = output_batch + head * dv;

    for (int t = 0; t < N; t++) {
        // Load Q and K from FP16 (pre-normalized)
        float q_reg[ROWS_PER_LANE];
        float k_reg[ROWS_PER_LANE];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; r++) {
            int i = r * WARP_SIZE + lane;
            q_reg[r] = __half2float(q_ptr[i]);
            k_reg[r] = __half2float(k_ptr[i]);
        }

        float decay = expf(*g_ptr);
        float beta_val = *b_ptr;
        float v_col = __half2float(v_ptr[col]);

        // kv_col = sum_i S[i][col] * k[i]
        float kv_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; r++) {
            kv_partial += s_reg[r] * k_reg[r];
        }
        float kv_col = kv_partial;
        for (int off = 16; off > 0; off >>= 1)
            kv_col += __shfl_xor_sync(0xFFFFFFFF, kv_col, off);

        float delta_col = (v_col - decay * kv_col) * beta_val;

        // S update + attn
        float attn_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; r++) {
            s_reg[r] = decay * s_reg[r] + k_reg[r] * delta_col;
            attn_partial += s_reg[r] * q_reg[r];
        }
        float attn_col = attn_partial;
        for (int off = 16; off > 0; off >>= 1)
            attn_col += __shfl_xor_sync(0xFFFFFFFF, attn_col, off);

        if (lane == 0) {
            o_ptr[col] = __float2half(attn_col);
        }

        // Advance pointers (simple adds, no multiplications)
        q_ptr += qkv_dim;
        k_ptr += qkv_dim;
        v_ptr += qkv_dim;
        g_ptr += n_v_heads;
        b_ptr += n_v_heads;
        o_ptr += ssm_inner;
    }

    // Write S column back
    #pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
        S_col[(r * WARP_SIZE + lane) * dv] = s_reg[r];
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

// ============================================================
// Multi-query Flash Attention for prefill
// ============================================================
// Each warp processes QR consecutive query positions for the same Q head.
// K/V are loaded once per key position and reused across QR queries.
// This cuts L2 bandwidth by QR× compared to one-query-per-warp.
// GQA grouping: 4 Q heads per KV head share a block for L1 reuse.
//
// Grid: (n_kv_heads, ceil(N/QR), batch)
// Block: GQA_RATIO warps, each processes QR queries on its Q head
//
// QR=4: 4 queries per warp, 4 Q heads per block → 16 outputs per block
// Register budget: q[4][8] + o[4][8] + softmax[4×2] + k[8] + temps ≈ 90 regs/thread
template<int QR = 4>
__global__ void __launch_bounds__(128)
kernel_flash_attn_multi(
    const half* __restrict__ Q,       // [B*L, n_head * head_dim]
    const half* __restrict__ K,       // [B*L, n_kv_heads * head_dim]
    const half* __restrict__ V,       // [B*L, n_kv_heads * head_dim]
    half* __restrict__ output,        // [B*L, n_head * head_dim]
    int L, int n_head, int n_kv_heads, int head_dim, float scale)
{
    constexpr int ELEMS = 8;  // head_dim / 32 = 256 / 32

    int kv_h = blockIdx.x;
    int q_tile = blockIdx.y;
    int b = blockIdx.z;
    int gqa_ratio = n_head / n_kv_heads;  // 4
    int q_h = kv_h * gqa_ratio + threadIdx.x / 32;  // warp selects Q head
    int lane = threadIdx.x % 32;

    int q_start = q_tile * QR;

    int q_stride = n_head * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    // Load Q for QR consecutive query positions
    float q_reg[QR][ELEMS];
    float o_reg[QR][ELEMS];
    float m_reg[QR];
    float l_reg[QR];

    #pragma unroll
    for (int qi = 0; qi < QR; qi++) {
        int q_pos = q_start + qi;
        m_reg[qi] = -FLT_MAX;
        l_reg[qi] = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) o_reg[qi][e] = 0.0f;
        if (q_pos < L) {
            const half* q_ptr = Q + (size_t)(b * L + q_pos) * q_stride + q_h * head_dim + lane * ELEMS;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                q_reg[qi][e] = __half2float(q_ptr[e]) * scale;
        } else {
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                q_reg[qi][e] = 0.0f;
        }
    }

    // Process all key positions up to the last valid query in this tile
    int max_q = min(q_start + QR - 1, L - 1);

    for (int k = 0; k <= max_q; k++) {
        // Load K and V once — reused across QR queries
        const half* k_ptr = K + (size_t)(b * L + k) * kv_stride + kv_h * head_dim + lane * ELEMS;
        float k_local[ELEMS];
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) k_local[e] = __half2float(k_ptr[e]);

        const half* v_ptr = V + (size_t)(b * L + k) * kv_stride + kv_h * head_dim + lane * ELEMS;
        float v_local[ELEMS];
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) v_local[e] = __half2float(v_ptr[e]);

        // Process each query
        #pragma unroll
        for (int qi = 0; qi < QR; qi++) {
            int q_pos = q_start + qi;
            if (q_pos >= L || k > q_pos) continue;  // causal mask

            // Dot product Q[qi] · K[k]
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                dot += q_reg[qi][e] * k_local[e];

            // Warp reduction
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

            // Online softmax update
            float old_max = m_reg[qi];
            float new_max = fmaxf(old_max, dot);
            float rescale = expf(old_max - new_max);

            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                o_reg[qi][e] *= rescale;
            l_reg[qi] = l_reg[qi] * rescale + expf(dot - new_max);
            m_reg[qi] = new_max;

            // Accumulate weighted V (reused from v_local)
            float w = expf(dot - new_max);
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                o_reg[qi][e] += w * v_local[e];
        }
    }

    // Output
    #pragma unroll
    for (int qi = 0; qi < QR; qi++) {
        int q_pos = q_start + qi;
        if (q_pos >= L) continue;
        float inv_sum = (l_reg[qi] > 0.0f) ? 1.0f / l_reg[qi] : 0.0f;
        half* o_ptr = output + (size_t)(b * L + q_pos) * q_stride + q_h * head_dim + lane * ELEMS;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            o_ptr[e] = __float2half(o_reg[qi][e] * inv_sum);
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

// Batch KV cache store: copy N tokens of K and V into cache at positions pos_offset..pos_offset+N-1
// Grid: (ceil(kv_dim/256), N)
__global__ void __launch_bounds__(256)
kernel_kv_cache_store_batch(
    half* __restrict__ k_cache,         // [max_seq, kv_dim]
    half* __restrict__ v_cache,         // [max_seq, kv_dim]
    const half* __restrict__ k_batch,   // [N, kv_dim]
    const half* __restrict__ v_batch,   // [N, kv_dim]
    int kv_dim, int pos_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kv_dim) return;
    int t = blockIdx.y;
    int pos = pos_offset + t;
    size_t cache_off = (size_t)pos * kv_dim + idx;
    size_t src_off = (size_t)t * kv_dim + idx;
    k_cache[cache_off] = k_batch[src_off];
    v_cache[cache_off] = v_batch[src_off];
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
    int qkv_dim = cfg.ssm_qkv_dim();  // 2*k_heads*dk + v_heads*dv
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

    // DeltaNet scratch (alpha/beta are per V-head)
    // 2x n_v_heads: first n_vh for token A (or single token), second n_vh for token B in 2tok mode
    d_alpha = static_cast<float*>(a(2 * cfg.ssm_n_v_heads * sizeof(float)));
    d_beta  = static_cast<float*>(a(2 * cfg.ssm_n_v_heads * sizeof(float)));


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

    // DeltaNet states
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        if (!cfg.is_full_attention_layer(i)) {
            DeltaNetState state;
            state.n_heads = cfg.ssm_n_v_heads;  // S matrix count = V heads
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
                                      int max_tokens) {
    auto a = [&](size_t bytes) -> void* { return alloc.alloc(bytes); };
    max_prefill = max_tokens;

    // Batch activation buffers: [max_tokens, dim]
    prefill_x     = static_cast<half*>(a(max_tokens * cfg.n_embed * sizeof(half)));
    prefill_out   = static_cast<half*>(a(max_tokens * cfg.n_embed * sizeof(half)));
    prefill_norm  = static_cast<half*>(a(max_tokens * cfg.n_embed * sizeof(half)));

    // Batch FFN buffers: [max_tokens, n_ff]
    // Cannot alias ffn_up/ffn_out with proj_qkv: full attention layers use ffn_up for V
    // projection and ffn_out for deinterleaved gate while proj_qkv still holds Q+gate.
    prefill_ffn_gate = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));
    prefill_ffn_up   = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));
    prefill_ffn_out  = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));

    // Batch projection buffers for DeltaNet QKV/gate and full attention Q/K/V
    size_t qkv_dim_alloc = cfg.ssm_qkv_dim();  // 2*k_heads*dk + v_heads*dv
    prefill_proj_qkv  = static_cast<half*>(a(max_tokens * qkv_dim_alloc * sizeof(half)));
    prefill_proj_gate = static_cast<half*>(a(max_tokens * cfg.ssm_inner_size * sizeof(half)));

    // Batch DeltaNet gate/beta buffers and pre-allocated token IDs
    prefill_dn_gate = static_cast<float*>(a(max_tokens * cfg.ssm_n_v_heads * sizeof(float)));
    prefill_dn_beta = static_cast<float*>(a(max_tokens * cfg.ssm_n_v_heads * sizeof(float)));
    d_prefill_tokens = static_cast<int*>(a(max_tokens * sizeof(int)));

    // MMQ (K-quant) GEMM prefill scratch
    int max_K = (int)std::max({cfg.n_embed, cfg.n_ff, (uint32_t)cfg.ssm_inner_size,
                                cfg.ssm_qkv_dim()});
    mmq_scratch_size = gwen_gemm_mmq_scratch_size(max_K, max_tokens);
    mmq_scratch = a(mmq_scratch_size);
    // cuBLAS handle for F16 GEMM (IQ4_XS→F16 converted weights)
    cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle));
    // Flash attention F32 scratch buffer
    size_t fa_scratch_size = (size_t)max_tokens * cfg.n_head * cfg.head_dim * sizeof(float);
    prefill_fa_scratch = static_cast<float*>(a(fa_scratch_size));
}

// ============================================================
// Batch prefill allocation (B independent sequences)
// ============================================================

void InferenceState::allocate_batch_prefill(const ModelConfig& cfg, CudaAllocator& alloc,
                                             int max_total_tokens, int max_seqs) {
    // Allocate prefill buffers for max_total_tokens
    allocate_prefill(cfg, alloc, max_total_tokens);
    max_batch_seqs = max_seqs;

    auto a = [&](size_t bytes) -> void* { return alloc.alloc(bytes); };

    // Count DeltaNet layers
    n_batch_dn_layers = 0;
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        if (!cfg.is_full_attention_layer(i)) n_batch_dn_layers++;
    }

    // Allocate B independent DeltaNet S states: [max_seqs * n_dn_layers * n_v_heads * dk * dv]
    int n_heads = cfg.ssm_n_v_heads;
    int dk = cfg.ssm_state_size;
    int dv = cfg.ssm_state_size;
    size_t S_per_layer = (size_t)n_heads * dk * dv * sizeof(float);
    size_t S_total = (size_t)max_seqs * n_batch_dn_layers * S_per_layer;
    batch_dn_S = static_cast<float*>(a(S_total));

    // Allocate B independent conv states: [max_seqs * n_dn_layers * (conv_kernel-1) * qkv_dim]
    int qkv_dim = cfg.ssm_qkv_dim();
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

    GWEN_LOG("Batch prefill: max_seqs=%d, max_tokens=%d\n", max_seqs, max_total_tokens);
    GWEN_LOG("  DeltaNet S states: %.1f MB (%d seqs × %d layers × %.0f KB/layer)\n",
           S_total / 1024.0 / 1024.0, max_seqs, n_batch_dn_layers, S_per_layer / 1024.0);
    GWEN_LOG("  Conv states: %.1f MB\n", conv_total / 1024.0 / 1024.0);
    GWEN_LOG("  Chunkwise buffers: %.1f MB (NT_max=%d)\n", chunk_total / 1024.0 / 1024.0, chunk_NT_max);
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

#ifdef GWEN_DEBUG
        size_t replay_total = n_dn_layers * S_bytes_per_layer + conv_replay_bytes;
        GWEN_LOG("MTP state allocated (KV cache: %.1f MB, replay buffers: %.1f KB)\n",
               2 * kv_bytes / 1024.0 / 1024.0, replay_total / 1024.0);
#endif
    } else {
        GWEN_LOG("MTP state allocated (KV cache: %.1f MB, no DeltaNet layers)\n",
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

    int qkv_dim = cfg.ssm_qkv_dim();
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
    GWEN_LOG("Batch-2 verify buffers allocated\n");
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

// ============================================================
// Extract hidden states for all tokens (prefill layers only)
// ============================================================

void InferenceState::extract_hidden(Model& model, const std::vector<int>& tokens, void* output_host) {
    reset_state();
    forward_prefill(model, tokens);
    // After forward_prefill, prefill_x holds [N, n_embed] hidden states (FP16)
    GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    int N = (int)tokens.size();
    size_t bytes = (size_t)N * model.config.n_embed * sizeof(half);
    GWEN_CHECK_CUDA(cudaMemcpy(output_host, prefill_x, bytes, cudaMemcpyDeviceToHost));
}

// ============================================================
// Forward pass body — all GPU work (CUDA graph capturable)
// ============================================================
// Reads token_id from d_token_id and pos from d_pos (device memory).
// Buffer convention: buf_a holds hidden state at layer start/end.

// Helper: get the LM head weight (separate output.weight or tied token_embd)
static inline const WeightRef& lm_head_weight(const Model& model) {
    return model.config.tie_word_embeddings ? model.token_embd : model.output_weight;
}

// Helper: is this a K-quant GGUF type?
static inline bool is_kquant_type(GGMLType t) {
    return t == GGMLType::Q4_K || t == GGMLType::Q5_K ||
           t == GGMLType::Q6_K || t == GGMLType::Q8_0 ||
           t == GGMLType::IQ4_XS;
}

// Helper: is this a dp4a-compatible type?
static inline bool is_dp4a_type(GGMLType t) {
    return t == GGMLType::Q4_K || t == GGMLType::Q5_K ||
           t == GGMLType::Q6_K || t == GGMLType::Q8_0 ||
           t == GGMLType::IQ4_XS;
}

// GEMV dispatch: F16 and K-quant (Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS)
// When x_q8 is provided and weight is K-quant, uses dp4a (INT8 dot product) path.
static inline void gemv_dispatch(const WeightRef& w, const half* x, half* y,
                                  int out_features, int in_features, cudaStream_t s,
                                  const void* x_q8 = nullptr) {
    if (w.type == GGMLType::F16) {
        gwen_gemv_fp16(static_cast<const half*>(w.device_data), x, y,
                        out_features, in_features, s);
    } else if (x_q8 && is_dp4a_type(w.type)) {
        gwen_gemv_dp4a(w.device_data, x_q8, y, out_features, in_features, w.type, s);
    } else {
        gwen_gemv(w.device_data, x, y, out_features, in_features, w.type, s);
    }
}

// GEMV + FP16 residual: y = W*x + residual (all FP16)
// dp4a path uses fused residual kernel; fallback does GEMV then add_inplace.
static inline void gemv_dispatch_residual(const WeightRef& w, const half* x,
                                           half* y, const half* residual,
                                           int out_features, int in_features,
                                           cudaStream_t s, const void* x_q8 = nullptr) {
    if (x_q8 && is_dp4a_type(w.type)) {
        gwen_gemv_dp4a_residual(w.device_data, x_q8, y, residual,
                                 out_features, in_features, w.type, s);
    } else {
        gemv_dispatch(w, x, y, out_features, in_features, s, x_q8);
        gwen_add_inplace(y, residual, out_features, s);
    }
}

// Batch-2 GEMV: read weights once, produce 2 outputs
static inline void gemv_dispatch_batch2(const WeightRef& w,
                                         const half* x0, const half* x1,
                                         half* y0, half* y1,
                                         int out_features, int in_features, cudaStream_t s,
                                         const void* x_q8_0 = nullptr, const void* x_q8_1 = nullptr) {
    if (is_dp4a_type(w.type) && x_q8_0 && x_q8_1) {
        gwen_gemv_dp4a_batch2(w.device_data, x_q8_0, x_q8_1, y0, y1,
                               out_features, in_features, w.type, s);
    } else {
        // Fallback: two sequential GEMVs (F16 weights, Q8_0, or K-quant without Q8_1 input)
        gemv_dispatch(w, x0, y0, out_features, in_features, s, x_q8_0);
        gemv_dispatch(w, x1, y1, out_features, in_features, s, x_q8_1);
    }
}

// Batch-2 GEMV + FP16 residual: y0 = W*x0 + r0, y1 = W*x1 + r1
static inline void gemv_dispatch_batch2_residual(const WeightRef& w,
                                                   const half* x0, const half* x1,
                                                   half* y0, half* y1,
                                                   const half* r0, const half* r1,
                                                   int out_features, int in_features,
                                                   cudaStream_t s,
                                                   const void* x_q8_0 = nullptr, const void* x_q8_1 = nullptr) {
    if (is_dp4a_type(w.type) && x_q8_0 && x_q8_1) {
        gwen_gemv_dp4a_residual_batch2(w.device_data, x_q8_0, x_q8_1, y0, y1, r0, r1,
                                        out_features, in_features, w.type, s);
    } else {
        // Fallback: two sequential residual dispatches
        gemv_dispatch_residual(w, x0, y0, r0, out_features, in_features, s, x_q8_0);
        gemv_dispatch_residual(w, x1, y1, r1, out_features, in_features, s, x_q8_1);
    }
}

void InferenceState::forward_body(Model& model, cudaStream_t s) {
    const auto& cfg = model.config;
    const float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);

    // Detect K-quant model: use dp4a path with Q8_1-quantized inputs
    const bool use_dp4a = is_dp4a_type(model.layers[0].deltanet.attn_qkv.type);

    // 1. Embedding lookup → buf_a (FP16 residual)
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      d_token_id, buf_a, cfg.n_embed, s, nullptr);

#ifdef GWEN_DEBUG
    {   // Debug: check embedding output for NaN
        GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
        half h4[4]; float f4[4];
        GWEN_CHECK_CUDA(cudaMemcpy(h4, buf_a, 8, cudaMemcpyDeviceToHost));
        for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
        GWEN_LOG("[DBG embed] %.6f %.6f %.6f %.6f\n", f4[0], f4[1], f4[2], f4[3]);
    }
#endif

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // 2. Process each layer (GWEN_LAYER_LIMIT env var to run subset for debugging)
    uint32_t n_layers_run = cfg.n_layers;
    const char* layer_limit_env = getenv("GWEN_LAYER_LIMIT");
    if (layer_limit_env) n_layers_run = std::min((uint32_t)atoi(layer_limit_env), cfg.n_layers);
    for (uint32_t layer_idx = 0; layer_idx < n_layers_run; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            const auto& w = layer.deltanet;
            auto& state = deltanet_states[dn_state_idx++];

            // RMSNorm: FP16 input → FP16 x_norm
            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize x_norm to Q8_1 for dp4a GEMV path
            if (use_dp4a) gwen_quantize_q8_1(x_norm, x_q8_a, cfg.n_embed, s);
            // QKV + gate projections
            gemv_dispatch(w.attn_qkv, x_norm, qkv, w.attn_qkv.shape[0], w.attn_qkv.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);
            gemv_dispatch(w.attn_gate, x_norm, gate_z, w.attn_gate.shape[0], w.attn_gate.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);

#ifdef GWEN_DEBUG
            if (layer_idx <= 1) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, x_norm, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u norm] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
                GWEN_CHECK_CUDA(cudaMemcpy(h4, qkv, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u qkv] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
                GWEN_CHECK_CUDA(cudaMemcpy(h4, gate_z, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u gate] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif

            int qkv_dim = cfg.ssm_qkv_dim();
            int conv_blocks = (qkv_dim + 255) / 256;
            kernel_conv1d_silu<<<conv_blocks, 256, 0, s>>>(
                qkv, qkv, state.conv_state,
                static_cast<const float*>(w.ssm_conv1d.device_data),
                qkv_dim, cfg.ssm_conv_kernel);


#ifdef GWEN_DEBUG
            if (layer_idx <= 1) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, qkv, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u conv] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif
            int q_width = cfg.ssm_n_k_heads * cfg.ssm_state_size;
            half* q = qkv;
            half* k = qkv + q_width;
            half* v = qkv + 2 * q_width;

            // Fused: L2-norm(Q,K) + gate/beta + S update
            {
                // Compute alpha/beta projections via GEMV, convert to decay/beta
                // Use attn_out as scratch for half-precision projections (avoid aliasing with d_alpha/d_beta)
                half* alpha_h = attn_out;  // [n_v_heads] half scratch
                half* beta_h = attn_out + cfg.ssm_n_v_heads;  // [n_v_heads] half scratch
                gemv_dispatch(w.ssm_alpha, x_norm, alpha_h,
                              w.ssm_alpha.shape[0], w.ssm_alpha.shape[1], s,
                              use_dp4a ? x_q8_a : nullptr);
                gemv_dispatch(w.ssm_beta, x_norm, beta_h,
                              w.ssm_beta.shape[0], w.ssm_beta.shape[1], s,
                              use_dp4a ? x_q8_a : nullptr);
                int n_vh = cfg.ssm_n_v_heads;
                kernel_alpha_beta_to_decay<<<(n_vh+31)/32, 32, 0, s>>>(
                    alpha_h, beta_h,
                    static_cast<const float*>(w.ssm_a.device_data),
                    static_cast<const float*>(w.ssm_dt_bias.device_data),
                    d_alpha, d_beta, n_vh);
                kernel_deltanet_fused<<<cfg.ssm_n_v_heads, 128, 65536, s>>>(
                    state.S, q, k, v, x_norm,
                    nullptr, nullptr,  // pre-computed: skip internal dot products
                    d_alpha, d_beta,   // pre-computed decay and beta
                    nullptr, nullptr,  // not used in pre-computed mode
                    attn_out, q_scale,
                    cfg.ssm_n_v_heads, cfg.ssm_n_k_heads, cfg.ssm_state_size, cfg.n_embed);
            }


#ifdef GWEN_DEBUG
            if (layer_idx <= 1) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, attn_out, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u deltanet] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif
            // Gated RMSNorm
            kernel_gated_rmsnorm<<<cfg.ssm_n_v_heads, 32, 0, s>>>(
                attn_out, static_cast<const float*>(w.ssm_norm.device_data),
                gate_z, gated_out,
                cfg.ssm_n_v_heads, cfg.ssm_state_size, cfg.rms_norm_eps);

#ifdef GWEN_DEBUG
            if (layer_idx <= 1) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, gated_out, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u gated] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif
            // Quantize gated_out to Q8_1 for ssm_out dp4a path
            if (use_dp4a) gwen_quantize_q8_1(gated_out, x_q8_a, cfg.ssm_inner_size, s);
            // Output projection with FP16 residual: buf_b = ssm_out(gated_out) + buf_a
            gemv_dispatch_residual(w.ssm_out, gated_out, buf_b, buf_a,
                          w.ssm_out.shape[0], w.ssm_out.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);

#ifdef GWEN_DEBUG
            if (layer_idx == 0) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, buf_b, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u ssm_out_res] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif

            // FFN: RMSNorm FP16→FP16, GEMV gate/up, SwiGLU, GEMV down
            gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize x_norm for FFN gate/up dp4a
            if (use_dp4a) gwen_quantize_q8_1(x_norm, x_q8_a, cfg.n_embed, s);

#ifdef GWEN_DEBUG
            if (layer_idx <= 1) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, x_norm, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u ffn_norm] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif

            gemv_dispatch(w.ffn_gate, x_norm, ffn_gate, w.ffn_gate.shape[0], w.ffn_gate.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);
            gemv_dispatch(w.ffn_up, x_norm, ffn_up, w.ffn_up.shape[0], w.ffn_up.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);

#ifdef GWEN_DEBUG
            if (layer_idx <= 1) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, ffn_gate, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u ffn_gate] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
                GWEN_CHECK_CUDA(cudaMemcpy(h4, ffn_up, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u ffn_up] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif

            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff, s);
            // Quantize ffn_out for ffn_down dp4a
            if (use_dp4a) gwen_quantize_q8_1(ffn_out, x_q8_a, cfg.n_ff, s);

#ifdef GWEN_DEBUG
            if (layer_idx <= 1) {
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
                half h4[4]; float f4[4];
                GWEN_CHECK_CUDA(cudaMemcpy(h4, ffn_out, 8, cudaMemcpyDeviceToHost));
                for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
                GWEN_LOG("[DBG L%u swiglu] %.6f %.6f %.6f %.6f\n", layer_idx, f4[0], f4[1], f4[2], f4[3]);
            }
#endif

            // FFN down + FP16 residual: buf_a = ffn_down(ffn_out) + buf_b
            gemv_dispatch_residual(w.ffn_down, ffn_out, buf_a, buf_b,
                          w.ffn_down.shape[0], w.ffn_down.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);

        } else {
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize x_norm for Q/K/V dp4a
            if (use_dp4a) gwen_quantize_q8_1(x_norm, x_q8_a, cfg.n_embed, s);
            gemv_dispatch(w.attn_q, x_norm, qkv, w.attn_q.shape[0], w.attn_q.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                int deint_blocks = (attn_dim + 255) / 256;
                kernel_deinterleave_qgate<<<deint_blocks, 256, 0, s>>>(
                    qkv, fa_q, gated_out, cfg.n_head, cfg.head_dim);
            }
            gemv_dispatch(w.attn_k, x_norm, fa_k, w.attn_k.shape[0], w.attn_k.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);
            gemv_dispatch(w.attn_v, x_norm, fa_v, w.attn_v.shape[0], w.attn_v.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);

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

            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            kernel_gqa_attention_decode<<<cfg.n_head, 256, 0, s>>>(
                fa_q, cache.k_cache, cache.v_cache,
                attn_out, attn_scores, d_pos,
                cfg.n_head, cfg.n_head_kv, cfg.head_dim, max_seq_alloc, scale);
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                gwen_sigmoid_mul(attn_out, gated_out, gated_out, attn_dim, s);
            }

            // Quantize gated_out for attn_output dp4a
            if (use_dp4a) gwen_quantize_q8_1(gated_out, x_q8_a, cfg.n_head * cfg.head_dim, s);
            // Output projection with FP16 residual: buf_b = attn_output(gated_out) + buf_a
            gemv_dispatch_residual(w.attn_output, gated_out, buf_b, buf_a,
                          w.attn_output.shape[0], w.attn_output.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);

            // FFN
            gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize x_norm for FFN dp4a
            if (use_dp4a) gwen_quantize_q8_1(x_norm, x_q8_a, cfg.n_embed, s);
            gemv_dispatch(w.ffn_gate, x_norm, ffn_gate, w.ffn_gate.shape[0], w.ffn_gate.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);
            gemv_dispatch(w.ffn_up, x_norm, ffn_up, w.ffn_up.shape[0], w.ffn_up.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);
            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff, s);
            // Quantize ffn_out for ffn_down dp4a
            if (use_dp4a) gwen_quantize_q8_1(ffn_out, x_q8_a, cfg.n_ff, s);
            // FFN down + FP16 residual: buf_a = ffn_down(ffn_out) + buf_b
            gemv_dispatch_residual(w.ffn_down, ffn_out, buf_a, buf_b,
                          w.ffn_down.shape[0], w.ffn_down.shape[1], s,
                          use_dp4a ? x_q8_a : nullptr);

        }

#ifdef GWEN_DEBUG
        {
            GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
            half h4[4]; float f4[4];
            GWEN_CHECK_CUDA(cudaMemcpy(h4, buf_a, 8, cudaMemcpyDeviceToHost));
            for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
            bool has_nan = (f4[0] != f4[0] || f4[1] != f4[1] || f4[2] != f4[2] || f4[3] != f4[3]);
            if (layer_idx == 0 || layer_idx == cfg.n_layers - 1 || has_nan)
                GWEN_LOG("[DBG L%u] %.6f %.6f %.6f %.6f%s\n", layer_idx, f4[0], f4[1], f4[2], f4[3], has_nan ? " NAN!" : "");
        }
#endif
    }

    // 3. Save hidden state for MTP (before LM head destroys it)
    if (mtp_hidden) {
        GWEN_CHECK_CUDA(cudaMemcpyAsync(mtp_hidden, buf_a, cfg.n_embed * sizeof(half),
                                         cudaMemcpyDeviceToDevice, s));
    }

    // 4. LM Head + argmax
    gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(model.output_norm.device_data),
                      x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
    // Quantize x_norm for LM head dp4a
    if (use_dp4a) gwen_quantize_q8_1(x_norm, x_q8_a, cfg.n_embed, s);

#ifdef GWEN_DEBUG
    {
        GWEN_CHECK_CUDA(cudaStreamSynchronize(s));
        half h4[4]; float f4[4];
        GWEN_CHECK_CUDA(cudaMemcpy(h4, x_norm, 8, cudaMemcpyDeviceToHost));
        for (int _i=0;_i<4;_i++) f4[_i]=__half2float(h4[_i]);
        GWEN_LOG("[DBG x_norm_final] %.6f %.6f %.6f %.6f\n", f4[0], f4[1], f4[2], f4[3]);

        const auto& lmw = lm_head_weight(model);
        GWEN_LOG("[DBG lm_head] type=%s shape=[%lu,%lu] size=%zu on_device=%d\n",
                 ggml_type_name(lmw.type), lmw.shape[0], lmw.shape[1],
                 lmw.size_bytes, lmw.on_device());
        // Verify first 16 bytes of LM head GPU data match host
        uint8_t gpu_bytes[16], host_bytes[16];
        GWEN_CHECK_CUDA(cudaMemcpy(gpu_bytes, lmw.device_data, 16, cudaMemcpyDeviceToHost));
        memcpy(host_bytes, lmw.host_data, 16);
        bool match = memcmp(gpu_bytes, host_bytes, 16) == 0;
        GWEN_LOG("[DBG lm_head data] gpu=%02x%02x%02x%02x host=%02x%02x%02x%02x %s\n",
                 gpu_bytes[0], gpu_bytes[1], gpu_bytes[2], gpu_bytes[3],
                 host_bytes[0], host_bytes[1], host_bytes[2], host_bytes[3],
                 match ? "MATCH" : "MISMATCH!");
    }
#endif

    gemv_dispatch(lm_head_weight(model), x_norm, logits_h, cfg.n_vocab, cfg.n_embed, s,
                  use_dp4a ? x_q8_a : nullptr);

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

    // Detect K-quant model: use dp4a path with Q8_1-quantized inputs
    const bool use_dp4a = is_dp4a_type(model.layers[0].deltanet.attn_qkv.type);

    // Aliases for token A (existing buffers) and token B (b2_ buffers)
    // Token A hidden: buf_a/buf_b (pointer-swap residual), x_norm, qkv, gate_z, ...
    // Token B hidden: b2_buf_a/b2_buf_b, b2_x_norm, b2_qkv, b2_gate_z, ...

    // We use d_token_id / d_pos for token A, and a separate pair for token B.
    // Layout: d_token_id[0]=tok_a, d_token_id[1]=pos_a (d_pos), d_token_id[2]=tok_b, d_token_id[3]=pos_b
    int* d_token_b = d_token_id + 2;
    int* d_pos_b   = d_token_id + 3;

    // 1. Embedding lookup for both tokens → FP16 residual
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      d_token_id, buf_a, cfg.n_embed, s, nullptr);
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      d_token_b, b2_buf_a, cfg.n_embed, s, nullptr);

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // 2. Process each layer — token A first (state update), then token B, batch GEMV where possible
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            const auto& w = layer.deltanet;
            int dn_idx = dn_state_idx++;
            auto& state = deltanet_states[dn_idx];

            // --- Batch2 RMSNorm: FP16 input → FP16 output ---
            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_rmsnorm_f32w(b2_buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              b2_x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize both x_norm vectors for dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(x_norm, b2_x_norm, x_q8_a, b2_x_q8_a, cfg.n_embed, s);

            // --- Batch2 GEMV: QKV and gate projections (read weights once) ---
            gemv_dispatch_batch2(w.attn_qkv, x_norm, b2_x_norm, qkv, b2_qkv,
                      w.attn_qkv.shape[0], w.attn_qkv.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);
            gemv_dispatch_batch2(w.attn_gate, x_norm, b2_x_norm, gate_z, b2_gate_z,
                      w.attn_gate.shape[0], w.attn_gate.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);

            int qkv_dim = cfg.ssm_qkv_dim();
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

            // --- Pre-compute alpha/beta for both tokens ---
            {
                int n_vh = cfg.ssm_n_v_heads;
                // Use attn_out as scratch for half-precision projections
                half* alpha_h_a = attn_out;
                half* beta_h_a = attn_out + n_vh;
                half* alpha_h_b = attn_out + 2 * n_vh;
                half* beta_h_b = attn_out + 3 * n_vh;
                // Token A projections
                gemv_dispatch(w.ssm_alpha, x_norm, alpha_h_a,
                              w.ssm_alpha.shape[0], w.ssm_alpha.shape[1], s,
                              use_dp4a ? x_q8_a : nullptr);
                gemv_dispatch(w.ssm_beta, x_norm, beta_h_a,
                              w.ssm_beta.shape[0], w.ssm_beta.shape[1], s,
                              use_dp4a ? x_q8_a : nullptr);
                // Token B projections
                gemv_dispatch(w.ssm_alpha, b2_x_norm, alpha_h_b,
                              w.ssm_alpha.shape[0], w.ssm_alpha.shape[1], s,
                              use_dp4a ? b2_x_q8_a : nullptr);
                gemv_dispatch(w.ssm_beta, b2_x_norm, beta_h_b,
                              w.ssm_beta.shape[0], w.ssm_beta.shape[1], s,
                              use_dp4a ? b2_x_q8_a : nullptr);
                // Convert to decay/beta: d_alpha[0..n_vh) = token A, d_alpha[n_vh..2*n_vh) = token B
                kernel_alpha_beta_to_decay<<<(n_vh+31)/32, 32, 0, s>>>(
                    alpha_h_a, beta_h_a,
                    static_cast<const float*>(w.ssm_a.device_data),
                    static_cast<const float*>(w.ssm_dt_bias.device_data),
                    d_alpha, d_beta, n_vh);
                kernel_alpha_beta_to_decay<<<(n_vh+31)/32, 32, 0, s>>>(
                    alpha_h_b, beta_h_b,
                    static_cast<const float*>(w.ssm_a.device_data),
                    static_cast<const float*>(w.ssm_dt_bias.device_data),
                    d_alpha + n_vh, d_beta + n_vh, n_vh);
            }

            // --- Fused 2-token DeltaNet: A + snapshot + B in one kernel launch ---
            // S stays in shared memory across both tokens. Snapshot is written
            // between A and B, overlapped with B's Phase 1-2 (L2-norm + gate/beta).
            {
            int q_w = cfg.ssm_n_k_heads * cfg.ssm_state_size;
            half *qa = qkv, *ka = qkv + q_w, *va = qkv + 2 * q_w;
            half *qb = b2_qkv, *kb = b2_qkv + q_w, *vb = b2_qkv + 2 * q_w;
            float* snap = (dn_idx < (int)dn_S_snapshot.size()) ? dn_S_snapshot[dn_idx] : nullptr;

            kernel_deltanet_fused_2tok<<<cfg.ssm_n_v_heads, 128, 65536, s>>>(
                state.S,
                qa, ka, va, x_norm,
                qb, kb, vb, b2_x_norm,
                nullptr, nullptr,    // pre-computed: skip internal dot products
                d_alpha, d_beta,     // pre-computed decay and beta (A at [0..n_vh), B at [n_vh..2*n_vh))
                nullptr, nullptr,    // not used in pre-computed mode
                attn_out, b2_attn_out, snap,
                q_scale, cfg.ssm_n_v_heads, cfg.ssm_n_k_heads, cfg.ssm_state_size, cfg.n_embed);
            }

            // --- Batch2 Gated RMSNorm (FP16 output, no Q8_1) ---
            {
                dim3 grid(cfg.ssm_n_v_heads, 2);
                kernel_gated_rmsnorm_batch2<<<grid, 32, 0, s>>>(
                    attn_out, b2_attn_out,
                    static_cast<const float*>(w.ssm_norm.device_data),
                    gate_z, b2_gate_z,
                    gated_out, b2_gated_out,
                    cfg.ssm_n_v_heads, cfg.ssm_state_size, cfg.rms_norm_eps);
            }

            // Quantize gated_out for ssm_out dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(gated_out, b2_gated_out, x_q8_a, b2_x_q8_a, cfg.ssm_inner_size, s);
            // --- Batch2 GEMV: ssm_out with FP16 residual ---
            gemv_dispatch_batch2_residual(w.ssm_out, gated_out, b2_gated_out,
                      buf_b, b2_buf_b, buf_a, b2_buf_a,
                      w.ssm_out.shape[0], w.ssm_out.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);

            // --- Batch2 RMSNorm FP16→FP16 (FFN input) ---
            gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_rmsnorm_f32w(b2_buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              b2_x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize x_norm for FFN dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(x_norm, b2_x_norm, x_q8_a, b2_x_q8_a, cfg.n_embed, s);

            // --- Batch2 GEMV: FFN gate and up ---
            gemv_dispatch_batch2(w.ffn_gate, x_norm, b2_x_norm, ffn_gate, b2_ffn_gate,
                      w.ffn_gate.shape[0], w.ffn_gate.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);
            gemv_dispatch_batch2(w.ffn_up, x_norm, b2_x_norm, ffn_up, b2_ffn_up,
                      w.ffn_up.shape[0], w.ffn_up.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);

            // --- Batch2 SwiGLU ---
            gwen_swiglu_batch2(ffn_gate, b2_ffn_gate, ffn_up, b2_ffn_up,
                      ffn_out, b2_ffn_out, cfg.n_ff, s);
            // Quantize ffn_out for ffn_down dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(ffn_out, b2_ffn_out, x_q8_a, b2_x_q8_a, cfg.n_ff, s);

            // --- Batch2 GEMV: FFN down with FP16 residual ---
            gemv_dispatch_batch2_residual(w.ffn_down, ffn_out, b2_ffn_out,
                      buf_a, b2_buf_a, buf_b, b2_buf_b,
                      w.ffn_down.shape[0], w.ffn_down.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);

        } else {
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            // --- Batch2 RMSNorm FP16→FP16 ---
            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_rmsnorm_f32w(b2_buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              b2_x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize both x_norm for Q/K/V dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(x_norm, b2_x_norm, x_q8_a, b2_x_q8_a, cfg.n_embed, s);

            // --- Batch2 GEMV: Q, K, V projections ---
            gemv_dispatch_batch2(w.attn_q, x_norm, b2_x_norm, qkv, b2_qkv,
                      w.attn_q.shape[0], w.attn_q.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);
            gemv_dispatch_batch2(w.attn_k, x_norm, b2_x_norm, fa_k, b2_fa_k,
                      w.attn_k.shape[0], w.attn_k.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);
            gemv_dispatch_batch2(w.attn_v, x_norm, b2_x_norm, fa_v, b2_fa_v,
                      w.attn_v.shape[0], w.attn_v.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);

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

            // Quantize gated_out for attn_output dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(gated_out, b2_gated_out, x_q8_a, b2_x_q8_a, cfg.n_head * cfg.head_dim, s);
            // --- Batch2 output proj with FP16 residual ---
            gemv_dispatch_batch2_residual(w.attn_output, gated_out, b2_gated_out,
                      buf_b, b2_buf_b, buf_a, b2_buf_a,
                      w.attn_output.shape[0], w.attn_output.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);

            // --- Batch2 RMSNorm FP16→FP16 for FFN ---
            gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_rmsnorm_f32w(b2_buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              b2_x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            // Quantize x_norm for FFN dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(x_norm, b2_x_norm, x_q8_a, b2_x_q8_a, cfg.n_embed, s);

            // --- Batch2 GEMV: FFN ---
            gemv_dispatch_batch2(w.ffn_gate, x_norm, b2_x_norm, ffn_gate, b2_ffn_gate,
                      w.ffn_gate.shape[0], w.ffn_gate.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);
            gemv_dispatch_batch2(w.ffn_up, x_norm, b2_x_norm, ffn_up, b2_ffn_up,
                      w.ffn_up.shape[0], w.ffn_up.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);
            gwen_swiglu_batch2(ffn_gate, b2_ffn_gate, ffn_up, b2_ffn_up,
                      ffn_out, b2_ffn_out, cfg.n_ff, s);
            // Quantize ffn_out for ffn_down dp4a
            if (use_dp4a) gwen_quantize_q8_1_batch2(ffn_out, b2_ffn_out, x_q8_a, b2_x_q8_a, cfg.n_ff, s);
            gemv_dispatch_batch2_residual(w.ffn_down, ffn_out, b2_ffn_out,
                      buf_a, b2_buf_a, buf_b, b2_buf_b,
                      w.ffn_down.shape[0], w.ffn_down.shape[1], s,
                      use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);
        }
    }

    // 3. Save hidden states for MTP (already FP16)
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

    // 4. LM head for both tokens — batch2 GEMV
    gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(model.output_norm.device_data),
                      x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
    gwen_rmsnorm_f32w(b2_buf_a, static_cast<const float*>(model.output_norm.device_data),
                      b2_x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
    // Quantize x_norm for LM head dp4a
    if (use_dp4a) gwen_quantize_q8_1_batch2(x_norm, b2_x_norm, x_q8_a, b2_x_q8_a, cfg.n_embed, s);
    gemv_dispatch_batch2(lm_head_weight(model), x_norm, b2_x_norm, logits_h, b2_logits_h,
              cfg.n_vocab, cfg.n_embed, s,
              use_dp4a ? x_q8_a : nullptr, use_dp4a ? b2_x_q8_a : nullptr);

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

#ifdef GWEN_DEBUG
    // Debug: run without CUDA graph to allow sync/memcpy debug checks
    forward_body(model, compute_stream);
    GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));
#else
    // CUDA graph: captures the full forward pass on first call, replays thereafter.
    // Host-side type dispatches (F16/K-quant) are invariant per model — safe to capture.
    if (!graph_captured) {
        cudaGraph_t graph;
        GWEN_CHECK_CUDA(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal));
        forward_body(model, compute_stream);
        GWEN_CHECK_CUDA(cudaStreamEndCapture(compute_stream, &graph));
        GWEN_CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        GWEN_CHECK_CUDA(cudaGraphDestroy(graph));
        graph_captured = true;
    }
    GWEN_CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));
    GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));
#endif
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

// Batched Q4_K embedding lookup
__global__ void __launch_bounds__(256)
kernel_embed_lookup_batch_q4k(const void* __restrict__ table,
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
        const uint8_t* base = static_cast<const uint8_t*>(table) + (size_t)blk_idx * 144;

        half d_half, dmin_half;
        memcpy(&d_half, base, sizeof(half));
        memcpy(&dmin_half, base + 2, sizeof(half));
        float d = __half2float(d_half);
        float dmin = __half2float(dmin_half);
        const uint8_t* scales_p = base + 4;
        const uint8_t* qs = base + 16;

        int sub_block = tid / 32;
        uint8_t sc_lo, m_lo;
        if (sub_block < 4) {
            sc_lo = scales_p[sub_block] & 0x3F;
            m_lo  = scales_p[sub_block + 4] & 0x3F;
        } else {
            sc_lo = (scales_p[sub_block + 4] & 0xF) | ((scales_p[sub_block - 4] >> 6) << 4);
            m_lo  = (scales_p[sub_block + 4] >> 4) | ((scales_p[sub_block] >> 6) << 4);
        }

        float scale = d * sc_lo;
        float min_val = dmin * m_lo;

        int group = tid / 64;
        int within = tid % 64;
        int is_high = within / 32;
        int pos = within % 32;
        int qs_byte_idx = group * 32 + pos;
        int q_val = is_high ? (qs[qs_byte_idx] >> 4) : (qs[qs_byte_idx] & 0xF);

        y[(size_t)token_idx * dim + blk_local * 256 + tid] = __float2half(scale * q_val - min_val);
    }
}

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

// Batch FP8 embedding lookup: for each token, look up FP8 row and dequant to FP16
__global__ void __launch_bounds__(256)
kernel_embed_lookup_batch_fp8(const uint8_t* __restrict__ table,
                                const float* __restrict__ scales,
                                const int* __restrict__ token_ids,
                                half* __restrict__ y,
                                int dim, int n_tokens) {
    int token_idx = blockIdx.y;
    if (token_idx >= n_tokens) return;

    int token_id = token_ids[token_idx];
    float scale = scales[token_id];

    const uint8_t* row = table + (size_t)token_id * dim;
    half* out = y + (size_t)token_idx * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        // Convert FP8 E4M3 byte to float via hardware
        __nv_fp8_e4m3 fp8_val;
        *reinterpret_cast<uint8_t*>(&fp8_val) = row[i];
        float val = float(fp8_val) * scale;
        out[i] = __float2half(val);
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

    // GEMM helper: routes to MMQ (K-quant) or cuBLAS (F16)
    auto do_gemm = [&](const WeightRef& w, const half* X, half* Y) {
        int M = (int)w.shape[0];
        int K = (int)w.shape[1];
        if (is_kquant_type(w.type)) {
            gwen_gemm_mmq(w.device_data, w.type, X, Y, mmq_scratch, M, K, N, s);
        } else if (w.type == GGMLType::F16) {
            // cuBLAS FP16 GEMM for IQ4_XS→F16 converted weights
            // Y[M,N] = W[M,K] @ X[K,N] where X is [N,K] row-major (= [K,N] col-major)
            cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
            cublasSetStream(handle, s);
            __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        M, N, K,
                        &alpha_h,
                        static_cast<const __half*>(w.device_data), K,  // W^T: [K,M] → [M,K] after transpose
                        X, K,                                           // X: [K,N] col-major
                        &beta_h,
                        Y, M);                                          // Y: [M,N] col-major
        } else {
            GWEN_CHECK(false, "Unsupported weight type for prefill GEMM");
        }
    };

    // 1. Batch embedding lookup → FP16 residual [N, n_embed]
    {
        dim3 grid(1, N);
        if (model.token_embd.type == GGMLType::Q4_K) {
            kernel_embed_lookup_batch_q4k<<<grid, 256, 0, s>>>(
                model.token_embd.device_data,
                d_token_ids, prefill_x, cfg.n_embed, N);
        } else {
            kernel_embed_lookup_batch_q6k<<<grid, 256, 0, s>>>(
                model.token_embd.device_data,
                d_token_ids, prefill_x, cfg.n_embed, N);
        }
    }

    // FP16 residual accumulators (matching extract_hidden_batch pipeline)
    half* pf_a = prefill_x;       // FP16 residual accumulator A
    half* pf_b = prefill_out;     // FP16 residual accumulator B
    half* pf_norm = prefill_norm;  // FP16 normalized (GEMM input/output temp)

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // 2. Process each layer
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            // ===== DeltaNet Layer — FP16 pipeline with fast register-based kernel =====
            const auto& w = layer.deltanet;
            auto& state = deltanet_states[dn_state_idx++];

            // RMSNorm: pf_a(FP16) → pf_norm(FP16)
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_a, static_cast<const float*>(w.attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // QKV and gate GEMMs → FP16 output
            int qkv_dim = cfg.ssm_qkv_dim();  // 6144
            do_gemm(w.attn_qkv, pf_norm, prefill_proj_qkv);
            do_gemm(w.attn_gate, pf_norm, prefill_proj_gate);

            // Conv1d + SiLU (FP16 QKV, F32 conv state)
            {
                int conv_grid_x = (qkv_dim + 255) / 256;
                kernel_batch_conv1d_silu_multi<<<conv_grid_x, 256, 0, s>>>(
                    prefill_proj_qkv, state.conv_state,
                    static_cast<const float*>(w.ssm_conv1d.device_data),
                    1, N, qkv_dim, cfg.ssm_conv_kernel);
            }

            // Gate/beta computation: batch GEMM for alpha/beta projections, then convert
            {
                int n_vh = cfg.ssm_n_v_heads;
                half* alpha_batch = prefill_ffn_out;  // reuse as scratch (n_vh << n_ff)
                half* beta_batch = alpha_batch + (size_t)N * n_vh;
                do_gemm(w.ssm_alpha, pf_norm, alpha_batch);
                do_gemm(w.ssm_beta, pf_norm, beta_batch);
                kernel_batch_alpha_beta_to_gate<<<dim3((n_vh+31)/32, N), 32, 0, s>>>(
                    alpha_batch, beta_batch,
                    static_cast<const float*>(w.ssm_a.device_data),
                    static_cast<const float*>(w.ssm_dt_bias.device_data),
                    prefill_dn_gate, prefill_dn_beta, N, n_vh);
            }

            // Pre-normalize Q and K (L2 norm) — avoids 10 warp reductions per token inside DeltaNet
            // QKV layout: [N, Q(k*dk) | K(k*dk) | V(v*dv)]
            {
                int total_vecs = N * cfg.ssm_n_k_heads;
                int q_width = cfg.ssm_n_k_heads * cfg.ssm_state_size;
                kernel_l2_normalize_qkv_batch<<<total_vecs, 32, 0, s>>>(
                    prefill_proj_qkv, N, cfg.ssm_qkv_dim(), q_width,
                    cfg.ssm_n_k_heads, cfg.ssm_state_size, q_scale);
            }

            // Fast DeltaNet: S in registers, warp-per-column (512 blocks vs old 16)
            {
                int num_warps = 4;
                int dv = cfg.ssm_state_size;  // 128
                dim3 grid(cfg.ssm_n_v_heads, 1, (dv + num_warps - 1) / num_warps);
                dim3 block(32, num_warps);
                kernel_deltanet_prefill_fast<<<grid, block, 0, s>>>(
                    state.S, prefill_proj_qkv,
                    prefill_dn_gate, prefill_dn_beta,
                    prefill_ffn_gate,  // reuse as DeltaNet output
                    N, cfg.ssm_n_v_heads, cfg.ssm_n_k_heads,
                    cfg.ssm_state_size, cfg.ssm_state_size,
                    cfg.ssm_qkv_dim(), cfg.ssm_inner_size, q_scale);
            }

            // Gated RMSNorm (FP16)
            kernel_batch_gated_rmsnorm<<<N * cfg.ssm_n_v_heads, 32, 0, s>>>(
                prefill_ffn_gate,
                static_cast<const float*>(w.ssm_norm.device_data),
                prefill_proj_gate,
                prefill_proj_gate,
                N, cfg.ssm_n_v_heads, cfg.ssm_state_size, cfg.rms_norm_eps);
            // Output projection GEMM → pf_b (FP16)
            do_gemm(w.ssm_out, prefill_proj_gate, pf_b);

            // FP16 residual add: pf_b += pf_a
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_b, pf_a, total);
            }

            // Post-attention RMSNorm: pf_b(FP16) → pf_norm(FP16)
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // FFN GEMMs (FP16)
            do_gemm(w.ffn_gate, pf_norm, prefill_ffn_gate);
            do_gemm(w.ffn_up, pf_norm, prefill_ffn_up);

            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate, prefill_ffn_up, prefill_ffn_out, N, cfg.n_ff);
            }

            do_gemm(w.ffn_down, prefill_ffn_out, pf_a);

            // FP16 residual add: pf_a += pf_b
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_a, pf_b, total);
            }
            // (debug dumps removed — FP16 pipeline)

        } else {
            // ===== Full Attention Layer — F32 residual + F32 FFN =====
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            // RMSNorm: pf_a(FP16) → pf_norm(FP16)
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_a, static_cast<const float*>(w.attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            int attn_dim = cfg.n_head * cfg.head_dim;  // 2048
            int kv_dim = cfg.n_head_kv * cfg.head_dim;  // 512
            int q_proj_dim = w.attn_q.shape[1];  // 4096

            // Q, K, V projections (FP16 — per-token attention ops are FP16)
            do_gemm(w.attn_q, pf_norm, prefill_proj_qkv);
            do_gemm(w.attn_k, pf_norm, prefill_ffn_gate);
            do_gemm(w.attn_v, pf_norm, prefill_ffn_up);

            // Batched attention (replaces per-token loop)
            // 1. Batch deinterleave Q+gate
            {
                int deint_blocks = (attn_dim + 255) / 256;
                dim3 grid(deint_blocks, N);
                kernel_deinterleave_qgate_batch<<<grid, 256, 0, s>>>(
                    prefill_proj_qkv, prefill_proj_gate, prefill_ffn_out,
                    N, cfg.n_head, cfg.head_dim);
            }

            // 2. Batch Q/K RMSNorm
            gwen_rmsnorm_batched_f32w(
                prefill_proj_gate, static_cast<const float*>(w.attn_q_norm.device_data),
                prefill_proj_gate, N * cfg.n_head, cfg.head_dim, cfg.rms_norm_eps, s);
            gwen_rmsnorm_batched_f32w(
                prefill_ffn_gate, static_cast<const float*>(w.attn_k_norm.device_data),
                prefill_ffn_gate, N * cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps, s);

            // 3. Batch RoPE
            {
                dim3 rope_grid(cfg.n_head + cfg.n_head_kv, N);
                int n_pairs = cfg.rope_dim / 2;
                int rope_threads = ((n_pairs + 31) / 32) * 32;
                if (rope_threads < 32) rope_threads = 32;
                if (rope_threads > 256) rope_threads = 256;
                kernel_rope_batch<<<rope_grid, rope_threads, 0, s>>>(
                    prefill_proj_gate, prefill_ffn_gate,
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                    N, cfg.rope_theta, cfg.rope_dim);
            }

            // 4. Store K, V into KV cache for subsequent decode
            {
                int kv_blocks = (kv_dim + 255) / 256;
                dim3 grid(kv_blocks, N);
                kernel_kv_cache_store_batch<<<grid, 256, 0, s>>>(
                    cache.k_cache, cache.v_cache,
                    prefill_ffn_gate, prefill_ffn_up,
                    kv_dim, pos);
            }

            // 5. MMA flash attention (ported from llama.cpp — tensor core m16n8k16)
            {
                float scale = 1.0f / sqrtf((float)cfg.head_dim);
                gwen_flash_attn_mma(
                    prefill_proj_gate,   // Q [N, n_head * head_dim] FP16
                    prefill_ffn_gate,    // K [N, n_kv_heads * head_dim] FP16
                    prefill_ffn_up,      // V [N, n_kv_heads * head_dim] FP16
                    prefill_proj_qkv,    // output [N, n_head * head_dim] FP16
                    prefill_fa_scratch,// F32 scratch (reuse DeltaNet QKV buffer)
                    N, cfg.n_head, cfg.n_head_kv, cfg.head_dim, scale, s);
            }

            // 6. Batch sigmoid-mul: output = attn_out * sigmoid(gate)
            {
                int total = N * attn_dim;
                int blocks = (total + 255) / 256;
                kernel_sigmoid_mul_batch<<<blocks, 256, 0, s>>>(
                    prefill_proj_qkv, prefill_ffn_out, prefill_proj_gate,
                    total);
            }

            // Out projection GEMM → pf_b (FP16)
            do_gemm(w.attn_output, prefill_proj_gate, pf_b);

            // FP16 residual add: pf_b += pf_a
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_b, pf_a, total);
            }

            // Post-attention RMSNorm: pf_b(FP16) → pf_norm(FP16)
            kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
                pf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                pf_norm, N, cfg.n_embed, cfg.rms_norm_eps);

            // FFN GEMMs (FP16)
            do_gemm(w.ffn_gate, pf_norm, prefill_ffn_gate);
            do_gemm(w.ffn_up, pf_norm, prefill_ffn_up);

            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate, prefill_ffn_up, prefill_ffn_out, N, cfg.n_ff);
            }

            do_gemm(w.ffn_down, prefill_ffn_out, pf_a);

            // FP16 residual add: pf_a += pf_b
            {
                int total = N * cfg.n_embed;
                int blocks = (total + 255) / 256;
                kernel_add_inplace_batch<<<blocks, 256, 0, s>>>(pf_a, pf_b, total);
            }
        }
    }

    // 3. Copy last token hidden → buf_a (FP16), RMSNorm, LM head
    GWEN_CHECK_CUDA(cudaMemcpyAsync(buf_a, pf_a + (size_t)(N - 1) * cfg.n_embed,
                                     cfg.n_embed * sizeof(half), cudaMemcpyDeviceToDevice, s));

    gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(model.output_norm.device_data),
                      x_norm, cfg.n_embed, cfg.rms_norm_eps, s);

    // 4. LM Head GEMV on last token
    gemv_dispatch(lm_head_weight(model), x_norm, logits_h, cfg.n_vocab, cfg.n_embed, s);

    int logit_blocks = (cfg.n_vocab + 255) / 256;
    kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, cfg.n_vocab);

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
        int n_heads = cfg.ssm_n_v_heads;
        int dk = cfg.ssm_state_size, dv = cfg.ssm_state_size;
        size_t S_per_layer = (size_t)n_heads * dk * dv * sizeof(float);
        size_t S_total = (size_t)B * n_batch_dn_layers * S_per_layer;
        GWEN_CHECK_CUDA(cudaMemsetAsync(batch_dn_S, 0, S_total, s));

        int qkv_dim = cfg.ssm_qkv_dim();
        int conv_km1 = cfg.ssm_conv_kernel - 1;
        size_t conv_per_layer = (size_t)conv_km1 * qkv_dim * sizeof(float);
        size_t conv_total = (size_t)B * n_batch_dn_layers * conv_per_layer;
        GWEN_CHECK_CUDA(cudaMemsetAsync(batch_dn_conv, 0, conv_total, s));
    }

    // Upload token IDs to device
    GWEN_CHECK_CUDA(cudaMemcpyAsync(d_prefill_tokens, all_tokens,
                                     N * sizeof(int), cudaMemcpyHostToDevice, s));


    // 1. Batch embedding lookup → prefill_x [B*L, n_embed]
    {
        dim3 grid(1, N);
        if (model.token_embd.type == GGMLType::Q4_K) {
            kernel_embed_lookup_batch_q4k<<<grid, 256, 0, s>>>(
                model.token_embd.device_data,
                d_prefill_tokens, prefill_x, cfg.n_embed, N);
        } else {
            kernel_embed_lookup_batch_q6k<<<grid, 256, 0, s>>>(
                model.token_embd.device_data,
                d_prefill_tokens, prefill_x, cfg.n_embed, N);
        }
    }

    // GEMM dispatch: routes to MMQ (K-quant) or cuBLAS (F16)
    auto do_gemm = [&](const WeightRef& w, const half* X, half* Y) {
        int M = (int)w.shape[0];
        int K = (int)w.shape[1];
        if (is_kquant_type(w.type)) {
            gwen_gemm_mmq(w.device_data, w.type, X, Y, mmq_scratch, M, K, N, s);
        } else if (w.type == GGMLType::F16) {
            cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
            cublasSetStream(handle, s);
            __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        M, N, K,
                        &alpha_h,
                        static_cast<const __half*>(w.device_data), K,
                        X, K,
                        &beta_h,
                        Y, M);
        } else {
            GWEN_CHECK(false, "Unsupported weight type for batch GEMM");
        }
    };

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
            int qkv_dim = cfg.ssm_qkv_dim();  // 6144
        
            do_gemm(w.attn_qkv, pf_norm, prefill_proj_qkv);
            do_gemm(w.attn_gate, pf_norm, prefill_proj_gate);

            // Batch conv1d + SiLU: B independent conv states
            int n_heads = cfg.ssm_n_v_heads;
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

            // Batch gate/beta for all B*L tokens: GEMM for alpha/beta projections, then convert
            {
                half* alpha_batch = prefill_ffn_out;  // reuse as scratch (n_heads << n_ff)
                half* beta_batch = alpha_batch + (size_t)N * n_heads;
                do_gemm(w.ssm_alpha, pf_norm, alpha_batch);
                do_gemm(w.ssm_beta, pf_norm, beta_batch);
                kernel_batch_alpha_beta_to_gate<<<dim3((n_heads+31)/32, N), 32, 0, s>>>(
                    alpha_batch, beta_batch,
                    static_cast<const float*>(w.ssm_a.device_data),
                    static_cast<const float*>(w.ssm_dt_bias.device_data),
                    prefill_dn_gate, prefill_dn_beta, N, n_heads);
            }

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
        
            do_gemm(w.ssm_out, prefill_proj_gate, pf_b);

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
        
            do_gemm(w.ffn_gate, pf_norm, prefill_ffn_gate);
            do_gemm(w.ffn_up, pf_norm, prefill_ffn_up);

            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate, prefill_ffn_up, prefill_ffn_out, N, cfg.n_ff);
            }

            do_gemm(w.ffn_down, prefill_ffn_out, pf_a);

        
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
        
            do_gemm(w.attn_q, pf_norm, prefill_proj_qkv);
            do_gemm(w.attn_k, pf_norm, prefill_ffn_gate);
            do_gemm(w.attn_v, pf_norm, prefill_ffn_up);

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

            // Multi-query flash attention (4 queries/warp × 4 GQA heads/block)
            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            {
                constexpr int QR = 4;
                int gqa_ratio = cfg.n_head / cfg.n_head_kv;
                int n_q_tiles = (L + QR - 1) / QR;
                dim3 attn_grid(cfg.n_head_kv, n_q_tiles, B);
                kernel_flash_attn_multi<QR><<<attn_grid, gqa_ratio * 32, 0, s>>>(
                    prefill_proj_gate, prefill_ffn_gate, prefill_ffn_up,
                    prefill_proj_qkv,
                    L, cfg.n_head, cfg.n_head_kv, cfg.head_dim, scale);
            }

            // Batch sigmoid-mul: output = attn_out * sigmoid(gate)
        
            {
                int total = N * attn_dim;
                int blocks = (total + 255) / 256;
                kernel_sigmoid_mul_batch<<<blocks, 256, 0, s>>>(
                    prefill_proj_qkv, prefill_ffn_out, prefill_proj_gate,
                    total);
            }

            // Batch output projection GEMM → pf_b
        
            do_gemm(w.attn_output, prefill_proj_gate, pf_b);

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
        
            do_gemm(w.ffn_gate, pf_norm, prefill_ffn_gate);
            do_gemm(w.ffn_up, pf_norm, prefill_ffn_up);

            {
                int total = N * cfg.n_ff;
                int blocks = (total + 255) / 256;
                kernel_swiglu_batch<<<blocks, 256, 0, s>>>(
                    prefill_ffn_gate, prefill_ffn_up, prefill_ffn_out, N, cfg.n_ff);
            }

            do_gemm(w.ffn_down, prefill_ffn_out, pf_a);

        
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

    // 2. Per-token: GEMV + argmax
    // Reuses single-token scratch buffers (logits_h, logits_f, argmax_*)
    for (int t = 0; t < N; t++) {
        half* norm_t = prefill_norm + (size_t)t * cfg.n_embed;

        // GEMV: embed_tokens × normed → logits [n_vocab]
        gemv_dispatch(lm_head_weight(model), norm_t, logits_h, cfg.n_vocab, cfg.n_embed, s);

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
    GWEN_LOG("TTFT: %.1f ms (%.0f prompt tok/s)\n", ttft_ms,
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
                      d_mtp_token, buf_b, cfg.n_embed, s, nullptr);

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
    if (model.has_reduced_lm_head) {
        const auto& rl = model.reduced_lm_head;
        int K = rl.K;
        int lm_rows = rl.has_idk ? K + 1 : K;
        if (rl.type == GGMLType::F16) {
            // FP16 reduced lm_head
            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(mtp_w.output_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_gemv_fp16(static_cast<const half*>(rl.weights.device_data),
                           x_norm, logits_h, lm_rows, cfg.n_embed, s);
        } else {
            // Q6_K reduced lm_head — still needs Q8_1 for dp4a
            gwen_rmsnorm_quantize_q8_1(buf_a, static_cast<const float*>(mtp_w.output_norm.device_data),
                              x_q8_a, nullptr, cfg.n_embed, cfg.rms_norm_eps, s);
            gwen_gemv_dp4a(rl.weights.device_data, x_q8_a, logits_h,
                      lm_rows, cfg.n_embed, rl.type, s);
        }

        int logit_blocks = (lm_rows + 255) / 256;
        kernel_half_to_float<<<logit_blocks, 256, 0, s>>>(logits_h, logits_f, lm_rows);
        kernel_argmax_partial<<<ARGMAX_BLOCKS, 256, 0, s>>>(logits_f, argmax_partial_max, argmax_partial_idx, lm_rows);
        kernel_argmax_reduce<<<1, 256, 0, s>>>(argmax_partial_max, argmax_partial_idx, d_argmax_token, ARGMAX_BLOCKS);
        if (mtp_confidence_threshold > 0.0f) {
            kernel_confidence_gate<<<1, 1, 0, s>>>(logits_f, d_argmax_token, lm_rows, mtp_confidence_threshold);
        }
        if (rl.has_idk || mtp_confidence_threshold > 0.0f) {
            kernel_remap_token_idk<<<1, 1, 0, s>>>(rl.d_token_ids, d_argmax_token, K);
        } else {
            kernel_remap_token<<<1, 1, 0, s>>>(rl.d_token_ids, d_argmax_token);
        }
    } else {
        // Full LM head
        gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(mtp_w.output_norm.device_data),
                          x_norm, cfg.n_embed, cfg.rms_norm_eps, s);
        gemv_dispatch(lm_head_weight(model), x_norm, logits_h, cfg.n_vocab, cfg.n_embed, s);

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
    GWEN_LOG("TTFT: %.1f ms (%.0f prompt tok/s)\n", ttft_ms,
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
        GWEN_LOG("MTP IDK: %d tokens skipped speculation\n", idk_count);
        if ((int)output_tokens.size() > n_predict) output_tokens.resize(n_predict);
        return output_tokens;
    }

    // Speculative decode loop: forward_2tok verifies draft + gets bonus token in one pass.
    // CPU dispatches two CUDA graphs per cycle (forward_2tok + MTP) with one int comparison
    // between them. CPU overhead is ~30 μs per 1.6 ms cycle (~2% of wall time).
    {
        // Cycle profiling: CUDA events for per-operation timing
        cudaEvent_t ev_start = nullptr, ev_end = nullptr;
        struct TimingStats {
            double sum = 0, sum_sq = 0, min_val = 1e9, max_val = 0;
            int count = 0;
            void record(float ms) {
                sum += ms; sum_sq += (double)ms * ms;
                if (ms < min_val) min_val = ms;
                if (ms > max_val) max_val = ms;
                count++;
            }
            double avg() const { return count > 0 ? sum / count : 0; }
            double stddev() const {
                if (count < 2) return 0;
                double mean = avg();
                return sqrt(sum_sq / count - mean * mean);
            }
        };
        TimingStats t_1tok, t_2tok, t_mtp, t_rollback;

        if (profile_cycles) {
            GWEN_CHECK_CUDA(cudaEventCreate(&ev_start));
            GWEN_CHECK_CUDA(cudaEventCreate(&ev_end));
        }

        auto time_op_begin = [&]() {
            if (profile_cycles) {
                GWEN_CHECK_CUDA(cudaEventRecord(ev_start, compute_stream));
            }
        };
        auto time_op_end = [&](TimingStats& stats) {
            if (profile_cycles) {
                GWEN_CHECK_CUDA(cudaEventRecord(ev_end, compute_stream));
                GWEN_CHECK_CUDA(cudaEventSynchronize(ev_end));
                float ms;
                GWEN_CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_end));
                stats.record(ms);
            }
        };

        int accepted = 0, rejected = 0, idk_count = 0;
        while ((int)output_tokens.size() < n_predict) {
            if (draft == -1) {
                // IDK: skip forward_2tok, fall back to regular forward
                time_op_begin();
                int next = forward(model, output_tokens.back());
                time_op_end(t_1tok);
                output_tokens.push_back(next);
                idk_count++;
                if (next == (int)model.config.eos_token_id) break;
                if ((int)output_tokens.size() >= n_predict) break;
                time_op_begin();
                draft = forward_mtp(model, next);
                time_op_end(t_mtp);
                continue;
            }
            int last_token = output_tokens.back();
            time_op_begin();
            auto [pred_a, pred_b] = forward_2tok(model, last_token, draft);
            time_op_end(t_2tok);
            if (pred_a == draft) {
                output_tokens.push_back(draft);
                output_tokens.push_back(pred_b);
                accepted++;
                if (pred_b == (int)model.config.eos_token_id) break;
                if ((int)output_tokens.size() >= n_predict) break;
                time_op_begin();
                draft = forward_mtp(model, pred_b);
                time_op_end(t_mtp);
            } else {
                rejected++;
                time_op_begin();
                undo_deltanet_token_b(compute_stream);
                pos -= 1;
                GWEN_CHECK_CUDA(cudaMemcpyAsync(mtp_hidden, mtp_hidden_b,
                    model.config.n_embed * sizeof(half),
                    cudaMemcpyDeviceToDevice, compute_stream));
                GWEN_CHECK_CUDA(cudaStreamSynchronize(compute_stream));
                time_op_end(t_rollback);
                output_tokens.push_back(pred_a);
                if (pred_a == (int)model.config.eos_token_id) break;
                if ((int)output_tokens.size() >= n_predict) break;
                time_op_begin();
                draft = forward_mtp(model, pred_a);
                time_op_end(t_mtp);
            }
        }
        if ((int)output_tokens.size() > n_predict) output_tokens.resize(n_predict);
#ifdef GWEN_DEBUG
        int total = accepted + rejected;
        GWEN_LOG("MTP stats: %d accepted, %d rejected, %d IDK (%.1f%% acceptance rate, %.1f%% IDK), %d cycles\n",
               accepted, rejected, idk_count,
               total > 0 ? 100.0 * accepted / total : 0.0,
               (total + idk_count) > 0 ? 100.0 * idk_count / (total + idk_count) : 0.0,
               total + idk_count);
#endif

        if (profile_cycles) {
            fprintf(stderr, "\n--- Cycle Timing (ms) ---\n");
            fprintf(stderr, "%-14s  %6s  %6s  %6s  %6s  %5s\n", "Operation", "avg", "min", "max", "stddev", "count");
            auto print_row = [](const char* name, const TimingStats& s) {
                if (s.count > 0) {
                    fprintf(stderr, "%-14s  %6.3f  %6.3f  %6.3f  %6.3f  %5d\n",
                            name, s.avg(), s.min_val, s.max_val, s.stddev(), s.count);
                }
            };
            print_row("forward_1tok", t_1tok);
            print_row("forward_2tok", t_2tok);
            print_row("forward_mtp", t_mtp);
            print_row("rollback", t_rollback);
            fprintf(stderr, "\n");
            GWEN_CHECK_CUDA(cudaEventDestroy(ev_start));
            GWEN_CHECK_CUDA(cudaEventDestroy(ev_end));
        }

        return output_tokens;
    }
}


} // namespace gwen
