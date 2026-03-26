#pragma once

#include "gwen/common.h"
#include "gwen/gguf.h"
#include "gwen/memory.h"

namespace gwen {

// A reference to a weight tensor on GPU (or mmap'd host for lazy upload)
struct WeightRef {
    const void* host_data = nullptr;  // mmap'd pointer (stays valid as long as GGUFFile lives)
    void* device_data = nullptr;      // GPU pointer (after upload)
    GGMLType type = GGMLType::F32;
    size_t n_elements = 0;
    size_t size_bytes = 0;
    std::vector<uint64_t> shape;

    bool on_device() const { return device_data != nullptr; }
};

// DeltaNet layer weights (18 of 24 layers)
struct DeltaNetLayerWeights {
    WeightRef attn_norm;       // [n_embed] F32 — pre-attention RMSNorm
    WeightRef attn_qkv;        // [n_embed, 3*ssm_inner] Q5_K — joint QKV
    WeightRef attn_gate;       // [n_embed, ssm_inner] Q4_K — gate (Z) for gated RMSNorm
    WeightRef ssm_conv1d;      // [conv_kernel, 3*ssm_inner] F32 — 1D conv
    WeightRef ssm_a;           // [ssm_n_heads] F32 — log decay
    WeightRef ssm_dt_bias;     // [ssm_n_heads] F32 — dt bias
    WeightRef ssm_alpha;       // [n_embed, ssm_n_heads] Q8_0 — alpha projection
    WeightRef ssm_beta;        // [n_embed, ssm_n_heads] Q8_0 — beta projection
    WeightRef ssm_norm;        // [ssm_state_size] F32 — per-head RMSNorm
    WeightRef ssm_out;         // [ssm_inner, n_embed] Q5_K — output projection
    WeightRef post_attn_norm;  // [n_embed] F32 — post-attention RMSNorm
    WeightRef ffn_gate;        // [n_embed, n_ff] Q4_K
    WeightRef ffn_up;          // [n_embed, n_ff] Q4_K
    WeightRef ffn_down;        // [n_ff, n_embed] Q6_K/Q4_K
};

// Full attention layer weights (6 of 24 layers)
struct FullAttnLayerWeights {
    WeightRef attn_norm;       // [n_embed] F32 — pre-attention RMSNorm
    WeightRef attn_q;          // [n_embed, n_head*(head_dim+head_dim)] Q4_K — Q + gate
    WeightRef attn_k;          // [n_embed, n_head_kv*head_dim] Q4_K
    WeightRef attn_v;          // [n_embed, n_head_kv*head_dim] Q6_K
    WeightRef attn_q_norm;     // [head_dim] F32 — per-head Q RMSNorm
    WeightRef attn_k_norm;     // [head_dim] F32 — per-head K RMSNorm
    WeightRef attn_output;     // [n_head*head_dim, n_embed] Q4_K — output projection
    WeightRef post_attn_norm;  // [n_embed] F32 — post-attention RMSNorm
    WeightRef ffn_gate;        // [n_embed, n_ff] Q4_K
    WeightRef ffn_up;          // [n_embed, n_ff] Q4_K
    WeightRef ffn_down;        // [n_ff, n_embed] Q4_K
};

// One layer — either DeltaNet or Full Attention
struct Layer {
    bool is_full_attention = false;
    DeltaNetLayerWeights deltanet;
    FullAttnLayerWeights full_attn;
};

// Complete model
struct Model {
    ModelConfig config;
    std::unique_ptr<GGUFFile> gguf;

    // Global weights
    WeightRef token_embd;       // [n_embed, n_vocab] Q6_K / Q4_K
    WeightRef output_weight;    // [n_embed, n_vocab] Q6_K — separate lm_head (when tie_word_embeddings=false)
    WeightRef output_norm;      // [n_embed] F32

    // Per-layer weights
    std::vector<Layer> layers;

    // Load from GGUF file
    static std::unique_ptr<Model> load(const std::string& gguf_path);

    // Upload all weights to GPU
    void upload_weights(CudaAllocator& allocator);

    // Print model info
    void print_info() const;
};

} // namespace gwen
