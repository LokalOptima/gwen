#pragma once

#include "gwen/common.h"
#include "gwen/model.h"
#include "gwen/memory.h"
#include <cuda_runtime.h>
#include <functional>

namespace gwen {

// DeltaNet recurrent state for one layer
struct DeltaNetState {
    float* S = nullptr;       // [n_heads, d_k, d_v] recurrent state matrix (FP32 for stability)
    float* conv_state = nullptr; // [conv_kernel-1, qkv_dim] conv1d rolling state
    int n_heads;
    int state_size;            // d_k = d_v = 128
    int qkv_dim;               // 3 * ssm_inner = 6144
    int conv_kernel;
};

// KV cache for one full attention layer
struct KVCache {
    half* k_cache = nullptr;  // [max_seq, n_kv_heads, head_dim]
    half* v_cache = nullptr;  // [max_seq, n_kv_heads, head_dim]
    int max_seq;
    int n_kv_heads;
    int head_dim;
};

// Complete inference state
struct InferenceState {
    // Scratch buffers (reused across layers) — x and residual are pointer-swapped
    half* x = nullptr;          // [n_embed] current hidden state
    half* x_norm = nullptr;     // [n_embed] after RMSNorm
    half* residual = nullptr;   // [n_embed] residual connection
    half* buf_a = nullptr;      // backing storage (one of x/residual points here)
    half* buf_b = nullptr;      // backing storage (the other of x/residual points here)

    // DeltaNet scratch
    half* qkv = nullptr;        // [3 * ssm_inner] — Q/K/V aliased into this
    half* gate_z = nullptr;     // [ssm_inner]
    half* attn_out = nullptr;   // [ssm_inner]
    half* gated_out = nullptr;  // [ssm_inner]

    // Full attention scratch
    half* fa_q = nullptr;       // [n_head * head_dim * 2] = Q + gate
    half* fa_k = nullptr;       // [n_kv_heads * head_dim]
    half* fa_v = nullptr;       // [n_kv_heads * head_dim]
    float* attn_scores = nullptr; // [n_head, max_seq] attention scores (FP32)

    // FFN scratch
    half* ffn_gate = nullptr;   // [n_ff]
    half* ffn_up = nullptr;     // [n_ff]
    half* ffn_out = nullptr;    // [n_ff]

    // Output
    half* logits_h = nullptr;   // [n_vocab] logits (on GPU)
    float* logits_f = nullptr;  // [n_vocab] logits in FP32 for sampling

    // Per-layer state
    std::vector<DeltaNetState> deltanet_states;
    std::vector<KVCache> kv_caches;

    // DeltaNet scratch (pre-allocated)
    float* d_alpha = nullptr;    // [ssm_n_heads]
    float* d_beta = nullptr;     // [ssm_n_heads]

    // dp4a GEMV scratch — Q8_1 quantized input vectors
    void* x_q8_a = nullptr;        // Q8_1 of buf_a/x_norm
    void* x_q8_b = nullptr;        // Q8_1 of ffn_out/gated_out

    int* d_argmax_token = nullptr;
    float* argmax_partial_max = nullptr; // [256] scratch for multi-block argmax
    int* argmax_partial_idx = nullptr;   // [256] scratch for multi-block argmax
    int* d_pos = nullptr;          // device-side position (for CUDA graph)
    int* d_token_id = nullptr;     // device-side token ID (for CUDA graph)
    int pos = 0;                   // current position in sequence
    int max_seq_alloc = 0;         // max sequence length allocated

    // CUDA Graph state
    cudaStream_t compute_stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_captured = false;

    // Allocate all buffers
    void allocate(const ModelConfig& cfg, CudaAllocator& alloc, int max_seq = 4096);

    // Prefill scratch buffers
    half* prefill_x = nullptr;       // [max_prefill, n_embed]
    half* prefill_out = nullptr;     // [max_prefill, n_embed]
    half* prefill_norm = nullptr;    // [max_prefill, n_embed]
    void* mmq_scratch = nullptr;
    void* cublas_handle = nullptr;
    half* prefill_ffn_gate = nullptr;
    half* prefill_ffn_up = nullptr;
    half* prefill_ffn_out = nullptr;
    half* prefill_proj_qkv = nullptr;
    half* prefill_proj_gate = nullptr;
    float* prefill_fa_scratch = nullptr;
    float* prefill_dn_gate = nullptr;
    float* prefill_dn_beta = nullptr;
    int* d_prefill_tokens = nullptr;
    int max_prefill = 0;

    void allocate_prefill(const ModelConfig& cfg, CudaAllocator& alloc, int max_tokens);

    // Reset all recurrent state (DeltaNet S/conv, KV caches, position counters)
    void reset_state();

    // Extract hidden states for all tokens via prefill (resets state first)
    void extract_hidden(Model& model, const std::vector<int>& tokens, void* output_host);

    // Run the forward pass (all GPU work on given stream)
    void forward_body(Model& model, cudaStream_t stream);

    // Run one forward pass (single token decode)
    int forward(Model& model, int token_id);

    // Process all prompt tokens at once (prefill), return last token's prediction
    int forward_prefill(Model& model, const std::vector<int>& tokens);

    // Generate tokens (standard greedy)
    // teacher_tokens: if non-empty, feed these tokens as input instead of own predictions
    //                 (for teacher-forced comparison against a reference engine)
    // on_token: if non-null, called with each generated token ID for streaming output
    std::vector<int> generate(Model& model, const std::vector<int>& prompt_tokens,
                              int n_predict, bool greedy = true, float temperature = 1.0f,
                              const std::vector<int>& teacher_tokens = {},
                              std::function<void(int)> on_token = nullptr);
};

} // namespace gwen
