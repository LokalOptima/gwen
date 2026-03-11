#include "gwen/inference.h"
#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

#include <cfloat>
#include <cstdio>
#include <algorithm>

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
__global__ void __launch_bounds__(128)
kernel_deltanet_decode(
    float* __restrict__ S,           // [n_heads, d_k, d_v] recurrent state
    const half* __restrict__ q_in,   // [n_heads * d_k] query vectors (L2-normalized)
    const half* __restrict__ k_in,   // [n_heads * d_k] key vectors (L2-normalized)
    const half* __restrict__ v_in,   // [n_heads * d_v] value vectors
    const float* __restrict__ gate,  // [n_heads] gate = ssm_a * softplus(alpha_proj + dt_bias)
    const float* __restrict__ beta,  // [n_heads] sigmoid(beta_proj)
    half* __restrict__ output,       // [n_heads * d_v] output
    int n_heads, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;  // 0..127

    float decay = expf(gate[head]);
    float b = beta[head];

    float* S_head = S + head * dk * dv;
    const half* q = q_in + head * dk;
    const half* k = k_in + head * dk;
    const half* v = v_in + head * dv;
    half* out = output + head * dv;

    int total = dk * dv;  // 128 * 128 = 16384

    // Step 1: Decay state — S *= exp(gate)
    for (int idx = tid; idx < total; idx += blockDim.x) {
        S_head[idx] *= decay;
    }

    __syncthreads();

    // Step 2: Compute sk = S^T @ k (dv-dimensional)
    // sk[j] = sum_i S[i*dv+j] * k[i]
    __shared__ float sk[128];  // dv = 128
    for (int j = tid; j < dv; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < dk; i++) {
            acc += S_head[i * dv + j] * __half2float(k[i]);
        }
        sk[j] = acc;
    }

    __syncthreads();

    // Step 3+4: Compute delta d = (v - sk) * beta, and update S += outer(k, d)
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int row = idx / dv;  // k dimension
        int col = idx % dv;  // v dimension
        float d_col = (__half2float(v[col]) - sk[col]) * b;
        S_head[idx] += __half2float(k[row]) * d_col;
    }

    __syncthreads();

    // Step 5: Compute output o = S^T @ q
    // o[j] = sum_i S[i*dv+j] * q[i]
    for (int j = tid; j < dv; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < dk; i++) {
            acc += S_head[i * dv + j] * __half2float(q[i]);
        }
        out[j] = __float2half(acc);
    }
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
// Gate/Beta computation for DeltaNet
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
        float sp = logf(1.0f + expf(alpha_biased));  // softplus
        gate_out[head] = ssm_a[head] * sp;

        // beta = sigmoid(beta_proj)
        beta_out[head] = 1.0f / (1.0f + expf(-beta_acc));
    }
}

// ============================================================
// GQA attention decode kernel
// ============================================================
// For each query head, compute attention over the KV cache
// Output [n_head, head_dim]
__global__ void __launch_bounds__(32)
kernel_gqa_attention_decode(
    const half* __restrict__ q,        // [n_head, head_dim]
    const half* __restrict__ k_cache,  // [seq_len, n_kv_heads, head_dim]
    const half* __restrict__ v_cache,  // [seq_len, n_kv_heads, head_dim]
    half* __restrict__ output,         // [n_head, head_dim]
    float* __restrict__ scores_buf,    // [n_head, max_seq] scratch
    int n_head, int n_kv_heads, int head_dim, int seq_len, float scale)
{
    int qh = blockIdx.x;  // query head index
    if (qh >= n_head) return;
    int lane = threadIdx.x;

    int kv_head = qh / (n_head / n_kv_heads);  // GQA mapping

    const half* q_head = q + qh * head_dim;
    float* scores = scores_buf + qh * seq_len;

    // Step 1: Compute attention scores: score[t] = q @ k_cache[t] * scale
    for (int t = 0; t < seq_len; t++) {
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

    // Step 2: Softmax over scores
    float max_val = -FLT_MAX;
    for (int t = lane; t < seq_len; t += 32) {
        max_val = fmaxf(max_val, scores[t]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    float sum_exp = 0.0f;
    for (int t = lane; t < seq_len; t += 32) {
        scores[t] = expf(scores[t] - max_val);
        sum_exp += scores[t];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_xor_sync(0xFFFFFFFF, sum_exp, offset);
    }

    float inv_sum = 1.0f / sum_exp;
    for (int t = lane; t < seq_len; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of values
    half* out_head = output + qh * head_dim;
    for (int d = lane; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const half* v_t = v_cache + (size_t)t * n_kv_heads * head_dim + kv_head * head_dim;
            acc += scores[t] * __half2float(v_t[d]);
        }
        out_head[d] = __float2half(acc);
    }
}

// ============================================================
// Argmax kernel for greedy decoding
// ============================================================
__global__ void __launch_bounds__(256)
kernel_argmax(const float* __restrict__ logits, int* __restrict__ result, int n) {
    __shared__ float s_max[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < n; i += blockDim.x) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max[tid + s] > s_max[tid]) {
            s_max[tid] = s_max[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) *result = s_idx[0];
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

    // Pre-allocated argmax result
    d_argmax_token = static_cast<int*>(a(sizeof(int)));

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
}

// ============================================================
// Forward pass — single token decode (optimized)
// ============================================================
// Buffer convention (no pointer swaps):
//   buf_a: hidden state at layer start and end
//   buf_b: alternate buffer for GEMV output before residual add
//   x_norm: RMSNorm output
// After each complete layer, result is always in buf_a.

int InferenceState::forward(Model& model, int token_id) {
    const auto& cfg = model.config;
    const float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);

    // Ensure deterministic buffer assignment
    x = buf_a;
    residual = buf_b;

    // 1. Embedding lookup → buf_a
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      token_id, buf_a, cfg.n_embed);

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // 2. Process each layer
    // Invariant: at loop start, hidden state is always in buf_a
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            // ===== DeltaNet Layer =====
            const auto& w = layer.deltanet;
            auto& state = deltanet_states[dn_state_idx++];

            // Pre-attention RMSNorm: buf_a → x_norm
            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            // QKV projection: x_norm [1024] → qkv [6144]
            gwen_gemv(w.attn_qkv.device_data, x_norm, qkv,
                      w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type);

            // Gate projection: x_norm [1024] → gate_z [2048]
            gwen_gemv(w.attn_gate.device_data, x_norm, gate_z,
                      w.attn_gate.shape[1], w.attn_gate.shape[0], w.attn_gate.type);

            // Fused Conv1D + SiLU on QKV
            int qkv_dim = cfg.ssm_inner_size * 3;
            int conv_blocks = (qkv_dim + 255) / 256;
            kernel_conv1d_silu<<<conv_blocks, 256>>>(
                qkv, qkv, state.conv_state,
                static_cast<const float*>(w.ssm_conv1d.device_data),
                qkv_dim, cfg.ssm_conv_kernel);

            // Q/K/V aliased into qkv buffer
            half* q = qkv;
            half* k = qkv + cfg.ssm_inner_size;
            half* v = qkv + 2 * cfg.ssm_inner_size;

            // L2-normalize Q (with fused 1/sqrt(d_k) scaling) and K
            gwen_l2_normalize(q, q, cfg.ssm_n_heads, cfg.ssm_state_size, q_scale);
            gwen_l2_normalize(k, k, cfg.ssm_n_heads, cfg.ssm_state_size);

            // Compute gate and beta
            kernel_compute_gate_beta<<<cfg.ssm_n_heads, 32>>>(
                x_norm,
                w.ssm_alpha.device_data, w.ssm_beta.device_data,
                static_cast<const float*>(w.ssm_a.device_data),
                static_cast<const float*>(w.ssm_dt_bias.device_data),
                d_alpha, d_beta,
                cfg.n_embed, cfg.ssm_n_heads);

            // DeltaNet recurrence (delta rule)
            kernel_deltanet_decode<<<cfg.ssm_n_heads, 128>>>(
                state.S, q, k, v, d_alpha, d_beta,
                attn_out, cfg.ssm_n_heads, cfg.ssm_state_size, cfg.ssm_state_size);

            // Gated RMSNorm: output = RMSNorm(attn_out) * SiLU(gate_z)
            kernel_gated_rmsnorm<<<cfg.ssm_n_heads, 32>>>(
                attn_out,
                static_cast<const float*>(w.ssm_norm.device_data),
                gate_z, gated_out,
                cfg.ssm_n_heads, cfg.ssm_state_size, cfg.rms_norm_eps);

            // Output projection: [2048] → buf_b
            gwen_gemv(w.ssm_out.device_data, gated_out, buf_b,
                      w.ssm_out.shape[1], w.ssm_out.shape[0], w.ssm_out.type);

            // Residual add: buf_b += buf_a → result in buf_b
            gwen_add_inplace(buf_b, buf_a, cfg.n_embed);

            // Post-attention RMSNorm: buf_b → x_norm
            gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            // SwiGLU FFN
            gwen_gemv(w.ffn_gate.device_data, x_norm, ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
            gwen_gemv(w.ffn_up.device_data, x_norm, ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff);

            // FFN down → buf_a
            gwen_gemv(w.ffn_down.device_data, ffn_out, buf_a,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);

            // Residual add: buf_a += buf_b → result in buf_a
            gwen_add_inplace(buf_a, buf_b, cfg.n_embed);

        } else {
            // ===== Full Attention Layer =====
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            // Pre-attention RMSNorm: buf_a → x_norm
            gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            // Q+gate projection: [1024] → [4096] (interleaved) — write to qkv as temp
            gwen_gemv(w.attn_q.device_data, x_norm, qkv,
                      w.attn_q.shape[1], w.attn_q.shape[0], w.attn_q.type);

            // Deinterleave: qkv (interleaved) → fa_q (Q) + gated_out (gate)
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                int deint_blocks = (attn_dim + 255) / 256;
                kernel_deinterleave_qgate<<<deint_blocks, 256>>>(
                    qkv, fa_q, gated_out,
                    cfg.n_head, cfg.head_dim);
            }

            // K projection: [1024] → [512]
            gwen_gemv(w.attn_k.device_data, x_norm, fa_k,
                      w.attn_k.shape[1], w.attn_k.shape[0], w.attn_k.type);

            // V projection: [1024] → [512]
            gwen_gemv(w.attn_v.device_data, x_norm, fa_v,
                      w.attn_v.shape[1], w.attn_v.shape[0], w.attn_v.type);

            // Per-head Q and K RMSNorm (batched)
            gwen_rmsnorm_batched_f32w(fa_q, static_cast<const float*>(w.attn_q_norm.device_data),
                                      fa_q, cfg.n_head, cfg.head_dim, cfg.rms_norm_eps);
            gwen_rmsnorm_batched_f32w(fa_k, static_cast<const float*>(w.attn_k_norm.device_data),
                                      fa_k, cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps);

            // RoPE on Q and K
            gwen_rope(fa_q, fa_k,
                      cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                      pos, cfg.rope_theta, cfg.rope_sections, cfg.rope_dim);

            // Store K and V in cache (async to avoid runtime overhead)
            size_t kv_row_bytes = cfg.n_head_kv * cfg.head_dim * sizeof(half);
            GWEN_CHECK_CUDA(cudaMemcpyAsync(
                cache.k_cache + (size_t)pos * cfg.n_head_kv * cfg.head_dim,
                fa_k, kv_row_bytes, cudaMemcpyDeviceToDevice, 0));
            GWEN_CHECK_CUDA(cudaMemcpyAsync(
                cache.v_cache + (size_t)pos * cfg.n_head_kv * cfg.head_dim,
                fa_v, kv_row_bytes, cudaMemcpyDeviceToDevice, 0));

            // GQA Attention decode
            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            int seq_len = pos + 1;

            kernel_gqa_attention_decode<<<cfg.n_head, 32>>>(
                fa_q, cache.k_cache, cache.v_cache,
                attn_out, attn_scores,
                cfg.n_head, cfg.n_head_kv, cfg.head_dim, seq_len, scale);

            // Gated attention: output = attn_result * sigmoid(gate)
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                gwen_sigmoid_mul(attn_out, gated_out, gated_out, attn_dim);
            }

            // Output projection: [2048] → buf_b
            gwen_gemv(w.attn_output.device_data, gated_out, buf_b,
                      w.attn_output.shape[1], w.attn_output.shape[0], w.attn_output.type);

            // Residual add: buf_b += buf_a → result in buf_b
            gwen_add_inplace(buf_b, buf_a, cfg.n_embed);

            // Post-attention RMSNorm + FFN
            gwen_rmsnorm_f32w(buf_b, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            gwen_gemv(w.ffn_gate.device_data, x_norm, ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
            gwen_gemv(w.ffn_up.device_data, x_norm, ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff);

            // FFN down → buf_a
            gwen_gemv(w.ffn_down.device_data, ffn_out, buf_a,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);

            // Residual add: buf_a += buf_b → result in buf_a
            gwen_add_inplace(buf_a, buf_b, cfg.n_embed);
        }
    }

    // 3. Final RMSNorm: buf_a → x_norm
    gwen_rmsnorm_f32w(buf_a, static_cast<const float*>(model.output_norm.device_data),
                      x_norm, cfg.n_embed, cfg.rms_norm_eps);

    // 4. LM Head GEMV: x_norm [1024] → logits [248320]
    gwen_gemv(model.token_embd.device_data, x_norm, logits_h,
              cfg.n_vocab, cfg.n_embed, model.token_embd.type);

    // 5. Convert logits to FP32 for sampling
    int logit_blocks = (cfg.n_vocab + 255) / 256;
    kernel_half_to_float<<<logit_blocks, 256>>>(logits_h, logits_f, cfg.n_vocab);

    // 6. Greedy decode (argmax) — pre-allocated buffer
    kernel_argmax<<<1, 256>>>(logits_f, d_argmax_token, cfg.n_vocab);

    int next_token;
    GWEN_CHECK_CUDA(cudaMemcpy(&next_token, d_argmax_token, sizeof(int), cudaMemcpyDeviceToHost));

    pos++;
    return next_token;
}

// ============================================================
// Generation loop
// ============================================================

std::vector<int> InferenceState::generate(Model& model, const std::vector<int>& prompt_tokens,
                                           int n_predict, bool greedy, float temperature) {
    std::vector<int> output_tokens;

    // Process prompt tokens (sequential for now — prefill optimization is Phase 5)
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        int next = forward(model, prompt_tokens[i]);
        if (i == (int)prompt_tokens.size() - 1) {
            output_tokens.push_back(next);
        }
    }

    // Decode loop
    for (int i = 1; i < n_predict; i++) {
        int next = forward(model, output_tokens.back());
        output_tokens.push_back(next);

        if (next == (int)model.config.eos_token_id) break;
    }

    return output_tokens;
}

} // namespace gwen
