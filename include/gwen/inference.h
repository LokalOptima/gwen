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

    // dp4a GEMV scratch — Q8_1 quantized input vectors
    void* x_q8_a = nullptr;        // Q8_1 of buf_a/x_norm (max: n_ff/32 blocks for FFN down)
    void* x_q8_b = nullptr;        // Q8_1 of ffn_out/gated_out (max: n_ff/32 blocks)

    int* d_argmax_token = nullptr; // pre-allocated argmax result
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
    half* prefill_x = nullptr;       // [max_prefill, n_embed] input embeddings
    half* prefill_out = nullptr;     // [max_prefill, n_embed] layer output
    half* prefill_norm = nullptr;    // [max_prefill, n_embed] after norm
    half* prefill_temp_w = nullptr;  // scratch for dequantized weights
    half* prefill_ffn_gate = nullptr; // [max_prefill, n_ff]
    half* prefill_ffn_up = nullptr;   // [max_prefill, n_ff]
    half* prefill_ffn_out = nullptr;  // [max_prefill, n_ff]
    half* prefill_proj_qkv = nullptr; // [max_prefill, ssm_inner*3] batch projection output
    half* prefill_proj_gate = nullptr; // [max_prefill, ssm_inner] batch gate/output proj
    int max_prefill = 0;

    void allocate_prefill(const ModelConfig& cfg, CudaAllocator& alloc, int max_tokens);

    // --- MTP (Multi-Token Prediction) state ---
    half* mtp_hidden = nullptr;     // [n_embed] saved hidden state after all layers (for MTP input)
    half* mtp_concat = nullptr;     // [2*n_embed] concat buffer for FC input
    KVCache mtp_kv_cache;           // KV cache for MTP's attention layer
    int* d_mtp_token = nullptr;     // device token ID for MTP embedding lookup
    int* d_mtp_pos = nullptr;       // device position for MTP (aliased to d_mtp_token + 1)
    int mtp_pos = 0;                // MTP's own position counter (separate from main model)
    cudaGraphExec_t mtp_graph_exec = nullptr;
    bool mtp_graph_captured = false;

    // DeltaNet state checkpoint buffers (for speculative decode rollback)
    std::vector<float*> dn_S_checkpoint;     // saved S matrices (one per DeltaNet layer)
    std::vector<float*> dn_conv_checkpoint;  // saved conv states (one per DeltaNet layer)

    void allocate_mtp(const ModelConfig& cfg, CudaAllocator& alloc, int max_seq);

    // Run the forward pass (all GPU work on given stream)
    void forward_body(Model& model, cudaStream_t stream);

    // Run one forward pass (single token decode)
    int forward(Model& model, int token_id);

    // Process all prompt tokens at once (prefill), return last token's prediction
    int forward_prefill(Model& model, const std::vector<int>& tokens);

    // MTP forward: predict next token from saved hidden state + current token embedding
    // Returns draft token ID. Uses mtp_hidden (set by forward_body).
    int forward_mtp(Model& model, int token_id);
    void forward_mtp_body(Model& model, cudaStream_t stream);

    // Verification forward: process 2 tokens [accepted, draft] in a batch.
    // Returns prediction at position 0 (to verify draft) and prediction at position 1 (bonus).
    // Also saves hidden state at both positions for MTP use.
    // Checkpoints DeltaNet state after token 0 for rollback on rejection.
    struct VerifyResult {
        int pred_0;      // target prediction after token 0 (correct next token)
        int pred_1;      // target prediction after token 1 (bonus if accepted)
    };
    VerifyResult forward_verify(Model& model, int token_0, int token_1);

    // Save/restore DeltaNet state for speculative decode rollback
    void save_deltanet_checkpoint(cudaStream_t stream);
    void restore_deltanet_checkpoint(cudaStream_t stream);

    // Generate tokens (standard greedy)
    std::vector<int> generate(Model& model, const std::vector<int>& prompt_tokens,
                              int n_predict, bool greedy = true, float temperature = 1.0f);

    // Generate tokens with MTP speculative decoding
    std::vector<int> generate_speculative(Model& model, const std::vector<int>& prompt_tokens,
                                           int n_predict);
};

} // namespace gwen
