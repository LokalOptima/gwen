#pragma once

#include "gwen/common.h"
#include "gwen/model.h"
#include "gwen/memory.h"
#include <cuda_runtime.h>

namespace gwen {

// DeltaNet recurrent state for one layer (16 heads × 128 × 128 = 32MB per layer at FP32)
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
    half* qkv = nullptr;        // [3 * ssm_inner] = [6144] — Q/K/V aliased into this
    half* gate_z = nullptr;     // [ssm_inner] = [2048]
    half* attn_out = nullptr;   // [ssm_inner] = [2048]
    half* gated_out = nullptr;  // [ssm_inner] = [2048]

    // Full attention scratch
    half* fa_q = nullptr;       // [n_head * head_dim * 2] = Q + gate = [4096]
    half* fa_k = nullptr;       // [n_kv_heads * head_dim] = [512]
    half* fa_v = nullptr;       // [n_kv_heads * head_dim] = [512]
    float* attn_scores = nullptr; // [n_head, max_seq] attention scores (FP32)

    // FFN scratch
    half* ffn_gate = nullptr;   // [n_ff] = [3584]
    half* ffn_up = nullptr;     // [n_ff] = [3584]
    half* ffn_out = nullptr;    // [n_ff] = [3584]

    // Output
    half* logits_h = nullptr;   // [n_vocab] logits (on GPU)
    float* logits_f = nullptr;  // [n_vocab] logits in FP32 for sampling

    // Per-layer state
    std::vector<DeltaNetState> deltanet_states;  // 18 states
    std::vector<KVCache> kv_caches;              // 6 caches

    // DeltaNet scratch (pre-allocated)
    float* d_alpha = nullptr;    // [ssm_n_heads]
    float* d_beta = nullptr;     // [ssm_n_heads]

    int* d_argmax_token = nullptr; // pre-allocated argmax result
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

    // Run the forward pass (all GPU work on given stream)
    void forward_body(Model& model, cudaStream_t stream);

    // Run one forward pass (single token decode)
    int forward(Model& model, int token_id);

    // Generate tokens
    std::vector<int> generate(Model& model, const std::vector<int>& prompt_tokens,
                              int n_predict, bool greedy = true, float temperature = 1.0f);
};

} // namespace gwen
