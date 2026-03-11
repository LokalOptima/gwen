#include "gwen/inference.h"
#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

#include <cfloat>
#include <cstdio>
#include <algorithm>

namespace gwen {

// Debug helper: print first N values of a GPU half array
static void debug_print_half(const char* label, const half* d_ptr, int n, int total = 0) {
    std::vector<half> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(half), cudaMemcpyDeviceToHost);
    printf("  [DEBUG] %s:", label);
    for (int i = 0; i < n; i++) printf(" %.6f", __half2float(h[i]));
    if (total > 0) {
        // compute norm
        std::vector<half> all(total);
        cudaMemcpy(all.data(), d_ptr, total * sizeof(half), cudaMemcpyDeviceToHost);
        float norm = 0;
        for (int i = 0; i < total; i++) { float v = __half2float(all[i]); norm += v*v; }
        printf(" (norm=%.4f)", sqrtf(norm));
    }
    printf("\n");
}
static void debug_print_float(const char* label, const float* d_ptr, int n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  [DEBUG] %s:", label);
    for (int i = 0; i < n; i++) printf(" %.6f", h[i]);
    printf("\n");
}

// ============================================================
// Simple element-wise scale kernel for half arrays
// ============================================================
__global__ void kernel_scale_half(half* __restrict__ x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = __float2half(__half2float(x[idx]) * scale);
    }
}

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
// Conv1D with rolling state
// ============================================================
// Applies 1D convolution with kernel_size=4 on the QKV features
// State stores the last (kernel_size-1)=3 values
__global__ void __launch_bounds__(256)
kernel_conv1d(
    half* __restrict__ output,           // [dim] output (may alias input)
    const half* __restrict__ input,      // [dim] current input
    float* __restrict__ conv_state,      // [kernel_size-1, dim] rolling state
    const float* __restrict__ weight,    // [kernel_size, dim] conv weights (stored as [kernel, dim])
    int dim, int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    // GGUF stores conv weight as [dim, kernel_size] — weight[feature][kernel_pos]
    // Convolution: output[i] = sum_{k=0}^{K-1} weight[i, k] * input_at_time(t-K+1+k, i)
    float x_val = __half2float(input[idx]);  // Save input BEFORE output write (may alias)

    float acc = 0.0f;
    for (int k = 0; k < kernel_size - 1; k++) {
        acc += weight[idx * kernel_size + k] * conv_state[k * dim + idx];
    }
    acc += weight[idx * kernel_size + (kernel_size - 1)] * x_val;

    output[idx] = __float2half(acc);

    // Shift state: state[0] = state[1], state[1] = state[2], state[2] = original input
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

    // Main buffers
    x        = static_cast<half*>(a(cfg.n_embed * sizeof(half)));
    x_norm   = static_cast<half*>(a(cfg.n_embed * sizeof(half)));
    residual = static_cast<half*>(a(cfg.n_embed * sizeof(half)));

    // DeltaNet scratch
    int qkv_dim = cfg.ssm_inner_size * 3;  // 6144
    qkv      = static_cast<half*>(a(qkv_dim * sizeof(half)));
    gate_z   = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    q        = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    k        = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    v        = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    attn_out = static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));
    gated_out= static_cast<half*>(a(cfg.ssm_inner_size * sizeof(half)));

    // Full attention scratch
    int q_dim = cfg.n_head * cfg.head_dim * 2;  // Q + gate = 4096
    fa_q     = static_cast<half*>(a(q_dim * sizeof(half)));
    fa_k     = static_cast<half*>(a(cfg.n_head_kv * cfg.head_dim * sizeof(half)));
    fa_v     = static_cast<half*>(a(cfg.n_head_kv * cfg.head_dim * sizeof(half)));
    attn_scores = static_cast<float*>(a(cfg.n_head * max_seq * sizeof(float)));
    attn_probs  = static_cast<half*>(a(cfg.n_head * max_seq * sizeof(half)));

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

    // DeltaNet states (18 layers)
    int dn_idx = 0;
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
// Forward pass — single token decode
// ============================================================

int InferenceState::forward(Model& model, int token_id) {
    const auto& cfg = model.config;

    bool debug = (pos == 0);  // Debug first token only

    // 1. Embedding lookup
    gwen_embed_lookup(model.token_embd.device_data, model.token_embd.type,
                      token_id, x, cfg.n_embed);

    if (debug) {
        cudaDeviceSynchronize();
        printf("[DEBUG] Token %d embedding:\n", token_id);
        debug_print_half("embd[:10]", x, 10, cfg.n_embed);
    }

    int dn_state_idx = 0;
    int kv_cache_idx = 0;

    // 2. Process each layer
    for (uint32_t layer_idx = 0; layer_idx < cfg.n_layers; layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (!layer.is_full_attention) {
            // ===== DeltaNet Layer =====
            const auto& w = layer.deltanet;
            auto& state = deltanet_states[dn_state_idx++];

            // 2a. Pre-attention RMSNorm
            gwen_rmsnorm_f32w(x, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 after RMSNorm:\n");
                debug_print_half("x_norm[:10]", x_norm, 10, cfg.n_embed);
            }

            // Save residual
            GWEN_CHECK_CUDA(cudaMemcpy(residual, x, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToDevice));

            // 2b. QKV projection: x_norm [1024] → qkv [6144]
            gwen_gemv(w.attn_qkv.device_data, x_norm, qkv,
                      w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 QKV projection:\n");
                debug_print_half("qkv[:10]", qkv, 10, cfg.ssm_inner_size * 3);
            }

            // 2c. Gate projection: x_norm [1024] → gate_z [2048]
            gwen_gemv(w.attn_gate.device_data, x_norm, gate_z,
                      w.attn_gate.shape[1], w.attn_gate.shape[0], w.attn_gate.type);

            // 2d. Conv1D on QKV
            int qkv_dim = cfg.ssm_inner_size * 3;
            int conv_blocks = (qkv_dim + 255) / 256;
            kernel_conv1d<<<conv_blocks, 256>>>(
                qkv, qkv, state.conv_state,
                static_cast<const float*>(w.ssm_conv1d.device_data),
                qkv_dim, cfg.ssm_conv_kernel);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 after conv1d:\n");
                debug_print_half("conv[:10]", qkv, 10);
            }

            // 2e. SiLU activation on QKV
            gwen_silu_inplace(qkv, qkv_dim);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 after SiLU:\n");
                debug_print_half("silu[:10]", qkv, 10);
            }

            // 2f. Split QKV into Q, K, V (each ssm_inner_size = 2048)
            GWEN_CHECK_CUDA(cudaMemcpy(q, qkv, cfg.ssm_inner_size * sizeof(half), cudaMemcpyDeviceToDevice));
            GWEN_CHECK_CUDA(cudaMemcpy(k, qkv + cfg.ssm_inner_size,
                            cfg.ssm_inner_size * sizeof(half), cudaMemcpyDeviceToDevice));
            GWEN_CHECK_CUDA(cudaMemcpy(v, qkv + 2 * cfg.ssm_inner_size,
                            cfg.ssm_inner_size * sizeof(half), cudaMemcpyDeviceToDevice));

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 Q/K/V split:\n");
                debug_print_half("Q[:5]", q, 5);
                debug_print_half("K[:5]", k, 5);
                debug_print_half("V[:5]", v, 5);
            }

            // 2g. L2-normalize Q and K
            gwen_l2_normalize(q, q, cfg.ssm_n_heads, cfg.ssm_state_size);
            gwen_l2_normalize(k, k, cfg.ssm_n_heads, cfg.ssm_state_size);

            // 2g2. Scale Q by 1/sqrt(d_k) — matches llama.cpp's DeltaNet implementation
            {
                float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);
                int n_q = cfg.ssm_inner_size;
                kernel_scale_half<<<(n_q + 255) / 256, 256>>>(q, q_scale, n_q);
            }

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 Q/K after L2 norm:\n");
                debug_print_half("Q_norm[:5]", q, 5);
                debug_print_half("K_norm[:5]", k, 5);
            }

            // 2h. Compute gate and beta
            kernel_compute_gate_beta<<<cfg.ssm_n_heads, 32>>>(
                x_norm,
                w.ssm_alpha.device_data, w.ssm_beta.device_data,
                static_cast<const float*>(w.ssm_a.device_data),
                static_cast<const float*>(w.ssm_dt_bias.device_data),
                d_alpha, d_beta,
                cfg.n_embed, cfg.ssm_n_heads);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 gate/beta:\n");
                debug_print_float("gate[:4]", d_alpha, 4);
                debug_print_float("beta[:4]", d_beta, 4);
            }

            // 2i. DeltaNet recurrence: update state and compute output (delta rule)
            kernel_deltanet_decode<<<cfg.ssm_n_heads, 128>>>(
                state.S, q, k, v, d_alpha, d_beta,
                attn_out, cfg.ssm_n_heads, cfg.ssm_state_size, cfg.ssm_state_size);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 DeltaNet output:\n");
                debug_print_half("attn_out[:10]", attn_out, 10);
            }

            // 2j. Gated RMSNorm: output = RMSNorm(attn_out) * SiLU(gate_z)
            kernel_gated_rmsnorm<<<cfg.ssm_n_heads, 32>>>(
                attn_out,
                static_cast<const float*>(w.ssm_norm.device_data),
                gate_z, gated_out,
                cfg.ssm_n_heads, cfg.ssm_state_size, cfg.rms_norm_eps);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 gated RMSNorm output:\n");
                debug_print_half("gated[:10]", gated_out, 10);
            }

            // 2k. Output projection: [2048] → [1024]
            gwen_gemv(w.ssm_out.device_data, gated_out, x,
                      w.ssm_out.shape[1], w.ssm_out.shape[0], w.ssm_out.type);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 after output proj:\n");
                debug_print_half("out_proj[:10]", x, 10);
            }

            // 2l. Residual add
            gwen_add_inplace(x, residual, cfg.n_embed);

            if (debug && layer_idx == 0) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 0 after residual:\n");
                debug_print_half("x[:10]", x, 10, cfg.n_embed);
            }

            // Save residual for FFN
            GWEN_CHECK_CUDA(cudaMemcpy(residual, x, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToDevice));

            // 2m. Post-attention RMSNorm
            gwen_rmsnorm_f32w(x, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            // 2n. SwiGLU FFN
            gwen_gemv(w.ffn_gate.device_data, x_norm, ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
            gwen_gemv(w.ffn_up.device_data, x_norm, ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff);
            gwen_gemv(w.ffn_down.device_data, ffn_out, x,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);

            // 2o. Residual add
            gwen_add_inplace(x, residual, cfg.n_embed);

            if (debug) {
                cudaDeviceSynchronize();
                // compute norm and early logit probe
                std::vector<half> hbuf(cfg.n_embed);
                cudaMemcpy(hbuf.data(), x, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToHost);
                float norm = 0;
                for (uint32_t i = 0; i < cfg.n_embed; i++) { float v = __half2float(hbuf[i]); norm += v*v; }
                // Compute early logits: apply final RMSNorm to current x, dot with a few embd rows
                gwen_rmsnorm_f32w(x, static_cast<const float*>(model.output_norm.device_data),
                                  x_norm, cfg.n_embed, cfg.rms_norm_eps);
                // Compute logit for token 0 and token 11
                half logit_tok0_h, logit_tok11_h;
                gwen_gemv(model.token_embd.device_data, x_norm, logits_h,
                          1, cfg.n_embed, model.token_embd.type);  // just row 0
                cudaMemcpy(&logit_tok0_h, logits_h, sizeof(half), cudaMemcpyDeviceToHost);
                // For token 11: compute as separate single-row GEMV
                // We need the 11th row. Token_embd is [n_embed, n_vocab], row 11 starts at offset 11
                {
                    // Point to row 11 of token_embd: skip 11 * blocks_per_row blocks
                    int bpr = cfg.n_embed / 256;  // Q6_K blocks per row
                    const void* row11 = static_cast<const uint8_t*>(model.token_embd.device_data)
                                        + (size_t)11 * bpr * 210;
                    gwen_gemv(row11, x_norm, logits_h, 1, cfg.n_embed, model.token_embd.type);
                    cudaMemcpy(&logit_tok11_h, logits_h, sizeof(half), cudaMemcpyDeviceToHost);
                }
                cudaDeviceSynchronize();
                printf("[DEBUG] After DeltaNet layer %d: x norm=%.4f, early_logit[0]=%.2f, early_logit[11]=%.2f\n",
                       layer_idx, sqrtf(norm),
                       __half2float(logit_tok0_h), __half2float(logit_tok11_h));
            }

        } else {
            // ===== Full Attention Layer =====
            const auto& w = layer.full_attn;
            auto& cache = kv_caches[kv_cache_idx++];

            // 3a. Pre-attention RMSNorm
            gwen_rmsnorm_f32w(x, static_cast<const float*>(w.attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            GWEN_CHECK_CUDA(cudaMemcpy(residual, x, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToDevice));

            if (debug && layer_idx == 3) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 3 (FullAttn) x_norm before Q proj:\n");
                debug_print_half("x_norm[:10]", x_norm, 10, cfg.n_embed);
            }

            // 3b. Q+gate projection: [1024] → [4096] (interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...])
            gwen_gemv(w.attn_q.device_data, x_norm, fa_q,
                      w.attn_q.shape[1], w.attn_q.shape[0], w.attn_q.type);

            if (debug && layer_idx == 3) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 3 raw Q+gate projection:\n");
                debug_print_half("fa_q[:10]", fa_q, 10);
                debug_print_half("fa_q[256:266]", fa_q + 256, 10);
            }

            // 3b2. Deinterleave Q and gate into separate buffers
            // fa_q[0:4096] interleaved → attn_out[0:2048] = Q, gated_out[0:2048] = gate
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                int deint_blocks = (attn_dim + 255) / 256;
                kernel_deinterleave_qgate<<<deint_blocks, 256>>>(
                    fa_q, attn_out, gated_out,
                    cfg.n_head, cfg.head_dim);
                // Copy Q back to fa_q (first 2048 elements only)
                GWEN_CHECK_CUDA(cudaMemcpy(fa_q, attn_out, attn_dim * sizeof(half), cudaMemcpyDeviceToDevice));
                // Keep gate in gated_out for later
            }

            // 3c. K projection: [1024] → [512]
            gwen_gemv(w.attn_k.device_data, x_norm, fa_k,
                      w.attn_k.shape[1], w.attn_k.shape[0], w.attn_k.type);

            // 3d. V projection: [1024] → [512]
            gwen_gemv(w.attn_v.device_data, x_norm, fa_v,
                      w.attn_v.shape[1], w.attn_v.shape[0], w.attn_v.type);

            if (debug && layer_idx == 3) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 3 after deinterleave:\n");
                debug_print_half("Q[:10]", fa_q, 10);
                debug_print_half("gate[:10]", gated_out, 10);
            }

            // 3e. Per-head Q and K RMSNorm
            for (int h = 0; h < (int)cfg.n_head; h++) {
                gwen_rmsnorm_f32w(fa_q + h * cfg.head_dim,
                                  static_cast<const float*>(w.attn_q_norm.device_data),
                                  fa_q + h * cfg.head_dim,
                                  cfg.head_dim, cfg.rms_norm_eps);
            }
            for (int h = 0; h < (int)cfg.n_head_kv; h++) {
                gwen_rmsnorm_f32w(fa_k + h * cfg.head_dim,
                                  static_cast<const float*>(w.attn_k_norm.device_data),
                                  fa_k + h * cfg.head_dim,
                                  cfg.head_dim, cfg.rms_norm_eps);
            }

            if (debug && layer_idx == 3) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 3 after Q/K RMSNorm:\n");
                debug_print_half("Q_normed[:10]", fa_q, 10);
                debug_print_half("K_normed[:10]", fa_k, 10);
                debug_print_half("V[:10]", fa_v, 10);
            }

            // 3f. RoPE on Q and K
            gwen_rope(fa_q, fa_k,
                      cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                      pos, cfg.rope_theta, cfg.rope_sections, cfg.rope_dim);

            // 3g. Store K and V in cache at current position
            size_t kv_row_bytes = cfg.n_head_kv * cfg.head_dim * sizeof(half);
            GWEN_CHECK_CUDA(cudaMemcpy(
                cache.k_cache + (size_t)pos * cfg.n_head_kv * cfg.head_dim,
                fa_k, kv_row_bytes, cudaMemcpyDeviceToDevice));
            GWEN_CHECK_CUDA(cudaMemcpy(
                cache.v_cache + (size_t)pos * cfg.n_head_kv * cfg.head_dim,
                fa_v, kv_row_bytes, cudaMemcpyDeviceToDevice));

            // 3h. GQA Attention decode
            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            int seq_len = pos + 1;

            kernel_gqa_attention_decode<<<cfg.n_head, 32>>>(
                fa_q, cache.k_cache, cache.v_cache,
                attn_out,  // attn output [n_head * head_dim] = [2048]
                attn_scores,
                cfg.n_head, cfg.n_head_kv, cfg.head_dim, seq_len, scale);

            if (debug && layer_idx == 3) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 3 attention output:\n");
                debug_print_half("attn_out[:10]", attn_out, 10);
            }

            // 3i. Gated attention: output = attn_result * sigmoid(gate)
            // gate is already in gated_out from step 3b2
            {
                int attn_dim = cfg.n_head * cfg.head_dim;
                gwen_sigmoid(gated_out, gated_out, attn_dim);  // sigmoid in-place
                gwen_mul(attn_out, gated_out, gated_out, attn_dim);
            }

            if (debug && layer_idx == 3) {
                cudaDeviceSynchronize();
                printf("[DEBUG] Layer 3 gated output:\n");
                debug_print_half("gated[:10]", gated_out, 10);
            }

            // 3j. Output projection: [2048] → [1024]
            gwen_gemv(w.attn_output.device_data, gated_out, x,
                      w.attn_output.shape[1], w.attn_output.shape[0], w.attn_output.type);

            // 3k. Residual add
            gwen_add_inplace(x, residual, cfg.n_embed);

            // Save residual
            GWEN_CHECK_CUDA(cudaMemcpy(residual, x, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToDevice));

            // 3l. Post-attention RMSNorm + FFN
            gwen_rmsnorm_f32w(x, static_cast<const float*>(w.post_attn_norm.device_data),
                              x_norm, cfg.n_embed, cfg.rms_norm_eps);

            gwen_gemv(w.ffn_gate.device_data, x_norm, ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
            gwen_gemv(w.ffn_up.device_data, x_norm, ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
            gwen_swiglu(ffn_gate, ffn_up, ffn_out, cfg.n_ff);
            gwen_gemv(w.ffn_down.device_data, ffn_out, x,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);

            // 3m. Residual add
            gwen_add_inplace(x, residual, cfg.n_embed);

            if (debug) {
                cudaDeviceSynchronize();
                std::vector<half> hbuf(cfg.n_embed);
                cudaMemcpy(hbuf.data(), x, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToHost);
                float norm = 0;
                for (uint32_t i = 0; i < cfg.n_embed; i++) { float v = __half2float(hbuf[i]); norm += v*v; }
                // Early logit probe
                gwen_rmsnorm_f32w(x, static_cast<const float*>(model.output_norm.device_data),
                                  x_norm, cfg.n_embed, cfg.rms_norm_eps);
                half logit_tok0_h, logit_tok11_h;
                gwen_gemv(model.token_embd.device_data, x_norm, logits_h,
                          1, cfg.n_embed, model.token_embd.type);
                cudaMemcpy(&logit_tok0_h, logits_h, sizeof(half), cudaMemcpyDeviceToHost);
                {
                    int bpr = cfg.n_embed / 256;
                    const void* row11 = static_cast<const uint8_t*>(model.token_embd.device_data)
                                        + (size_t)11 * bpr * 210;
                    gwen_gemv(row11, x_norm, logits_h, 1, cfg.n_embed, model.token_embd.type);
                    cudaMemcpy(&logit_tok11_h, logits_h, sizeof(half), cudaMemcpyDeviceToHost);
                }
                cudaDeviceSynchronize();
                printf("[DEBUG] After FullAttn layer %d: x norm=%.4f, early_logit[0]=%.2f, early_logit[11]=%.2f\n",
                       layer_idx, sqrtf(norm),
                       __half2float(logit_tok0_h), __half2float(logit_tok11_h));
            }
        }
    }

    // 4. Final RMSNorm
    gwen_rmsnorm_f32w(x, static_cast<const float*>(model.output_norm.device_data),
                      x_norm, cfg.n_embed, cfg.rms_norm_eps);

    // 5. LM Head: x_norm [1024] → logits [248320]
    // LM head uses tied weights (token_embd transposed)
    // token_embd is [n_embed, n_vocab] = [1024, 248320] in Q6_K
    // For the LM head we need y = W^T @ x, which is just the GEMV with transposed semantics
    // But GGUF stores it as [1024, 248320] so GEMV(248320 rows, 1024 cols) gives us logits directly
    gwen_gemv(model.token_embd.device_data, x_norm, logits_h,
              cfg.n_vocab, cfg.n_embed, model.token_embd.type);

    if (debug) {
        cudaDeviceSynchronize();
        printf("[DEBUG] Final x_norm[:10]:\n");
        debug_print_half("x_norm[:10]", x_norm, 10, cfg.n_embed);
        printf("[DEBUG] Logits (first 10, around 9419):\n");
        debug_print_half("logits_h[:10]", logits_h, 10);
        debug_print_half("logits[9415:9425]", logits_h + 9415, 10);

        // Dump final x_norm to binary file for Python verification
        {
            std::vector<half> h_xnorm(cfg.n_embed);
            cudaMemcpy(h_xnorm.data(), x_norm, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToHost);
            FILE* fp = fopen("/tmp/gwen_x_norm.bin", "wb");
            if (fp) {
                // Write as float32 for easy Python reading
                for (uint32_t i = 0; i < cfg.n_embed; i++) {
                    float v = __half2float(h_xnorm[i]);
                    fwrite(&v, sizeof(float), 1, fp);
                }
                fclose(fp);
                printf("[DEBUG] Dumped x_norm (%d floats) to /tmp/gwen_x_norm.bin\n", cfg.n_embed);
            }
        }
        // Also dump x (pre-norm) for comparison
        {
            std::vector<half> h_x(cfg.n_embed);
            cudaMemcpy(h_x.data(), x, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToHost);
            FILE* fp = fopen("/tmp/gwen_x_final.bin", "wb");
            if (fp) {
                for (uint32_t i = 0; i < cfg.n_embed; i++) {
                    float v = __half2float(h_x[i]);
                    fwrite(&v, sizeof(float), 1, fp);
                }
                fclose(fp);
                printf("[DEBUG] Dumped x (%d floats) to /tmp/gwen_x_final.bin\n", cfg.n_embed);
            }
        }
        // Dump per-layer x values
        printf("[DEBUG] Logit for token 11 (llama top): ");
        debug_print_half("logits[11]", logits_h + 11, 1);
    }

    // Convert logits to FP32 for sampling
    int logit_blocks = (cfg.n_vocab + 255) / 256;
    kernel_half_to_float<<<logit_blocks, 256>>>(logits_h, logits_f, cfg.n_vocab);

    // Dump logits if requested
    if (debug) {
        const char* dump_path = getenv("GWEN_DUMP_LOGITS");
        if (dump_path) {
            std::vector<float> h_logits(cfg.n_vocab);
            cudaMemcpy(h_logits.data(), logits_f, cfg.n_vocab * sizeof(float), cudaMemcpyDeviceToHost);
            FILE* fp = fopen(dump_path, "wb");
            if (fp) {
                int nv = cfg.n_vocab;
                fwrite(&nv, sizeof(int), 1, fp);
                fwrite(h_logits.data(), sizeof(float), cfg.n_vocab, fp);
                fclose(fp);
                printf("[DEBUG] Dumped %d logits to %s\n", cfg.n_vocab, dump_path);
            }
        }
    }

    // 6. Greedy decode (argmax)
    int* d_token;
    GWEN_CHECK_CUDA(cudaMalloc(&d_token, sizeof(int)));
    kernel_argmax<<<1, 256>>>(logits_f, d_token, cfg.n_vocab);

    int next_token;
    GWEN_CHECK_CUDA(cudaMemcpy(&next_token, d_token, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_token);

    if (debug) {
        // Print top-5 logits
        std::vector<float> h_logits(cfg.n_vocab);
        cudaMemcpy(h_logits.data(), logits_f, cfg.n_vocab * sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < (int)cfg.n_vocab; i++) scored.push_back({h_logits[i], i});
        std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
        printf("[DEBUG] Top-10 logits:\n");
        for (int i = 0; i < 10; i++) {
            printf("  token=%d logit=%.4f\n", scored[i].second, scored[i].first);
        }
    }

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
