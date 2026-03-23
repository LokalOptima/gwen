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

    // F32 residual accumulators (FP8 path — avoids FP16 precision loss over 24 layers)
    float* buf_a_f32 = nullptr; // [n_embed] F32 residual buffer A
    float* buf_b_f32 = nullptr; // [n_embed] F32 residual buffer B

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

    // --- Batch-2 scratch buffers (token B in 2-token verify) ---
    half* b2_buf_a = nullptr;       // [n_embed] token B hidden state
    half* b2_buf_b = nullptr;       // [n_embed] token B residual
    half* b2_x_norm = nullptr;      // [n_embed] token B after norm
    half* b2_qkv = nullptr;         // [3 * ssm_inner]
    half* b2_gate_z = nullptr;      // [ssm_inner]
    half* b2_attn_out = nullptr;    // [ssm_inner]
    half* b2_gated_out = nullptr;   // [ssm_inner]
    half* b2_fa_q = nullptr;        // [n_head * head_dim * 2]
    half* b2_fa_k = nullptr;        // [n_kv_heads * head_dim]
    half* b2_fa_v = nullptr;        // [n_kv_heads * head_dim]
    half* b2_ffn_gate = nullptr;    // [n_ff]
    half* b2_ffn_up = nullptr;      // [n_ff]
    half* b2_ffn_out = nullptr;     // [n_ff]
    half* b2_logits_h = nullptr;    // [n_vocab]
    float* b2_logits_f = nullptr;   // [n_vocab]
    void* b2_x_q8_a = nullptr;     // Q8_1 scratch for token B
    void* b2_x_q8_b = nullptr;     // Q8_1 scratch for token B
    int* b2_d_argmax = nullptr;     // argmax result for token B
    float* b2_argmax_partial_max = nullptr;
    int* b2_argmax_partial_idx = nullptr;
    bool batch2_allocated = false;

    void allocate_batch2(const ModelConfig& cfg, CudaAllocator& alloc);

    // CUDA Graph state
    cudaStream_t compute_stream = nullptr;
    cudaStream_t overlap_stream = nullptr;   // secondary stream for DeltaNet pipelining
    cudaEvent_t ev_conv_done = nullptr;      // signals conv1d A + conv_snap finished
    cudaEvent_t ev_overlap_done = nullptr;   // signals conv1d B finished on overlap_stream
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_captured = false;
    cudaGraphExec_t graph_2tok_exec = nullptr;
    bool graph_2tok_captured = false;

    // Cycle profiling (--profile-cycles)
    bool profile_cycles = false;

    // Allocate all buffers
    void allocate(const ModelConfig& cfg, CudaAllocator& alloc, int max_seq = 4096);

    // Prefill scratch buffers
    half* prefill_x = nullptr;       // [max_prefill, n_embed] input embeddings (FP16, for extract_hidden)
    half* prefill_out = nullptr;     // [max_prefill, n_embed] layer output (FP16, unused in F32 path)
    half* prefill_norm = nullptr;    // [max_prefill, n_embed] after norm (FP16, GEMM I/O)
    float* prefill_x_f32 = nullptr;       // [max_prefill, n_embed] F32 residual accumulator A
    float* prefill_out_f32 = nullptr;     // [max_prefill, n_embed] F32 residual accumulator B
    float* prefill_proj_qkv_f32 = nullptr; // [max_prefill, ssm_inner*3] F32 QKV projection
    float* prefill_proj_gate_f32 = nullptr; // [max_prefill, ssm_inner] F32 gate/output projection
    float* prefill_dn_out_f32 = nullptr;  // [max_prefill, ssm_inner] F32 DeltaNet output temp
    float* prefill_ffn_gate_f32 = nullptr; // [max_prefill, n_ff] F32 FFN gate
    float* prefill_ffn_up_f32 = nullptr;   // [max_prefill, n_ff] F32 FFN up
    float* prefill_ffn_out_f32 = nullptr;  // [max_prefill, n_ff] F32 FFN SwiGLU output
    // FP8 GEMM prefill scratch
    uint8_t* fp8_act_buf = nullptr;       // [max_prefill * max_K] FP8 quantized activations
    float* sfb_act_buf = nullptr;         // [ceil(max_prefill/128) * ceil(max_K/128)] scale factors
    void* gemm_fp8_workspace = nullptr;   // CUTLASS workspace
    size_t gemm_fp8_ws_size = 0;
    half* prefill_ffn_gate = nullptr; // [max_prefill, n_ff]
    half* prefill_ffn_up = nullptr;   // [max_prefill, n_ff]
    half* prefill_ffn_out = nullptr;  // [max_prefill, n_ff]
    half* prefill_proj_qkv = nullptr; // [max_prefill, ssm_inner*3] batch projection output
    half* prefill_proj_gate = nullptr; // [max_prefill, ssm_inner] batch gate/output proj
    float* prefill_dn_gate = nullptr; // [max_prefill, ssm_n_heads] DeltaNet gate values
    float* prefill_dn_beta = nullptr; // [max_prefill, ssm_n_heads] DeltaNet beta values
    int* d_prefill_tokens = nullptr;  // [max_prefill] pre-allocated device token IDs
    int max_prefill = 0;

    void allocate_prefill(const ModelConfig& cfg, CudaAllocator& alloc, int max_tokens, bool f32_path = true);

    // --- Batch prefill state (for extract_hidden_batch) ---
    // B independent DeltaNet states and conv states, contiguous for batch kernels
    float* batch_dn_S = nullptr;       // [max_batch * n_dn_layers * n_heads * dk * dv] all S states
    float* batch_dn_conv = nullptr;    // [max_batch * n_dn_layers * (conv_kernel-1) * qkv_dim] all conv states
    int max_batch_seqs = 0;            // max B for batch prefill
    int n_batch_dn_layers = 0;         // number of DeltaNet layers

    // --- Chunkwise DeltaNet intermediate buffers ---
    static constexpr int CHUNK_SIZE = 64;
    float* chunk_gate_cumul = nullptr; // [max_tokens, n_heads] cumulative gate prefix sum
    half* chunk_W = nullptr;           // [max_tokens, ssm_inner] WY W vectors (transformed beta*K)
    half* chunk_U = nullptr;           // [max_tokens, ssm_inner] WY U vectors (transformed beta*V)
    float* chunk_h_states = nullptr;   // [max_batch * NT_max * n_heads, dk, dv] h state per chunk boundary
    half* chunk_v_new = nullptr;       // [max_tokens, ssm_inner] corrected values after state propagation
    int chunk_NT_max = 0;              // max number of chunks per sequence

    void allocate_batch_prefill(const ModelConfig& cfg, CudaAllocator& alloc, int max_total_tokens, int max_seqs, bool f32_path = false);

    // Batch extract: process B independent sequences of length L, output all hidden states
    // all_tokens: [B * L] flat array of token IDs (host)
    // output_host: [B * L * n_embed] FP16 hidden states (host)
    void extract_hidden_batch(Model& model, const int32_t* all_tokens, int B, int L, void* output_host);

    // Dev server: restricted embed for teacher logit computation
    half* restricted_embed_fp16 = nullptr;  // [K, n_embed] FP16 on GPU (persistent)
    int restricted_vocab_K = 0;             // number of restricted vocab entries

    // Compute main model predictions from hidden states on GPU (after extract_hidden_batch).
    // Applies output_norm + lm_head (embed_tokens GEMV) + argmax per token.
    // hidden_gpu: [N, n_embed] FP16 on GPU (e.g., pf_a after batch extraction)
    // preds_host: [N] int32 predictions on host
    void predict_from_hidden(Model& model, half* hidden_gpu, int N, int32_t* preds_host);

    // --- MTP (Multi-Token Prediction) state ---
    half* mtp_hidden = nullptr;     // [n_embed] saved hidden state after all layers (for MTP input)
    half* mtp_hidden_b = nullptr;   // [n_embed] token B's hidden (for 2tok accept path)
    half* mtp_concat = nullptr;     // [2*n_embed] concat buffer for FC input
    KVCache mtp_kv_cache;           // KV cache for MTP's attention layer
    int* d_mtp_token = nullptr;     // device token ID for MTP embedding lookup
    int* d_mtp_pos = nullptr;       // device position for MTP (aliased to d_mtp_token + 1)
    int mtp_pos = 0;                // MTP's own position counter (separate from main model)
    cudaGraphExec_t mtp_graph_exec = nullptr;
    bool mtp_graph_captured = false;

    // Activation replay buffers (for speculative decode rollback)
    // S snapshots: saved between token A and B in forward_body_2tok (baked into CUDA graph).
    // On reject, S is restored from snapshots (~0.02 ms) + conv1d undo (~0.002 ms).
    // Eliminates the 1.66 ms re-forward of the old checkpoint approach.
    std::vector<float*> dn_S_snapshot;    // [n_dn_layers] S state after token A (for exact restore)
    float* dn_replay_conv_row = nullptr;  // [n_dn_layers, qkv_dim] saved conv_state[0] per layer
    float** d_conv_ptrs = nullptr;        // [n_dn_layers] device array of conv_state pointers
    int n_dn_layers = 0;                  // number of DeltaNet layers (18 for Qwen3.5)

    void allocate_mtp(const ModelConfig& cfg, CudaAllocator& alloc, int max_seq);

    // Reset all recurrent state (DeltaNet S/conv, KV caches, position counters)
    // Call before processing an independent sequence
    void reset_state();

    // Extract hidden states for all tokens via prefill (resets state first)
    // Output: host buffer receiving N * n_embed FP16 values
    void extract_hidden(Model& model, const std::vector<int>& tokens, void* output_host);

    // Run the forward pass (all GPU work on given stream)
    void forward_body(Model& model, cudaStream_t stream);

    // Run one forward pass (single token decode)
    int forward(Model& model, int token_id);

    // 2-token forward: process two tokens through all layers, reading weights once
    // Returns {token_A_prediction, token_B_prediction}
    // Token A is processed first through each layer (state-updating), then token B
    // All GEMV projections use batch2 kernels to halve bandwidth
    // Also saves token B's activations to replay buffers for undo on reject
    void forward_body_2tok(Model& model, cudaStream_t stream);
    std::pair<int,int> forward_2tok(Model& model, int token_id_a, int token_id_b);

    // Process all prompt tokens at once (prefill), return last token's prediction
    int forward_prefill(Model& model, const std::vector<int>& tokens);

    // MTP forward: predict next token from saved hidden state + current token embedding
    // Returns draft token ID. Uses mtp_hidden (set by forward_body).
    int forward_mtp(Model& model, int token_id);
    void forward_mtp_body(Model& model, cudaStream_t stream);

    // Undo token B's state changes: restore S from snapshots + reverse conv1d shift.
    // Cost: ~0.04 ms (18 MB S restore + conv undo) vs 1.7 ms for old checkpoint+re-forward.
    void undo_deltanet_token_b(cudaStream_t stream);

    // Generate tokens (standard greedy)
    // teacher_tokens: if non-empty, feed these tokens as input instead of own predictions
    //                 (for teacher-forced comparison against a reference engine)
    std::vector<int> generate(Model& model, const std::vector<int>& prompt_tokens,
                              int n_predict, bool greedy = true, float temperature = 1.0f,
                              const std::vector<int>& teacher_tokens = {});

    // Generate tokens with MTP speculative decoding
    float mtp_confidence_threshold = 0.0f;  // softmax prob below this → skip speculation
    std::vector<int> generate_speculative(Model& model, const std::vector<int>& prompt_tokens,
                                           int n_predict);
};

} // namespace gwen
