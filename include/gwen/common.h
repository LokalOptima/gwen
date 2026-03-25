#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- Error checking ---

#define GWEN_CHECK_CUDA(call)                                                  \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

#define GWEN_CHECK(cond, msg)                                                  \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "GWEN error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    (msg));                                                     \
            throw std::runtime_error(std::string("GWEN error: ") + (msg));     \
        }                                                                      \
    } while (0)

namespace gwen {

// --- GGML tensor types (matching GGUF format) ---
enum class GGMLType : uint32_t {
    F32    = 0,
    F16    = 1,
    Q4_0   = 2,
    Q4_1   = 3,
    Q5_0   = 6,
    Q5_1   = 7,
    Q8_0   = 8,
    Q8_1   = 9,
    Q2_K   = 10,
    Q3_K   = 11,
    Q4_K   = 12,
    Q5_K   = 13,
    Q6_K   = 14,
    Q8_K   = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8     = 24,
    I16    = 25,
    I32    = 26,
    I64    = 27,
    F64    = 28,
    BF16   = 30,
    FP8_E4M3 = 31,  // GWEN-specific: 8-bit float (E4M3), 1 byte per element
    FP4_E2M1 = 32,  // GWEN-specific: 4-bit float (E2M1), packed 2 per byte
};

const char* ggml_type_name(GGMLType type);

// Block sizes for quantized types (number of elements per block)
uint32_t ggml_type_block_size(GGMLType type);

// Bytes per block for quantized types
size_t ggml_type_block_bytes(GGMLType type);

// Bytes for a tensor of given type and element count
size_t ggml_type_size(GGMLType type, size_t n_elements);

// --- Model hyperparameters ---
struct ModelConfig {
    uint32_t n_layers        = 24;
    uint32_t n_embed         = 1024;
    uint32_t n_ff            = 3584;
    uint32_t n_vocab         = 248320;
    uint32_t n_head          = 8;       // Q heads (full attention)
    uint32_t n_head_kv       = 2;       // KV heads (full attention, GQA)
    uint32_t head_dim        = 256;     // full attention head dim
    float    rope_theta      = 10000000.0f;
    int      rope_sections[4] = {11, 11, 10, 0};
    uint32_t rope_dim        = 64;      // partial rotary dims
    float    rms_norm_eps    = 1e-6f;
    uint32_t ssm_conv_kernel = 4;
    uint32_t ssm_state_size  = 128;     // d_k = d_v for DeltaNet
    uint32_t ssm_n_heads     = 16;      // DeltaNet V heads (legacy alias for ssm_n_v_heads)
    uint32_t ssm_n_k_heads   = 16;      // DeltaNet K heads (linear_num_key_heads)
    uint32_t ssm_n_v_heads   = 16;      // DeltaNet V heads (linear_num_value_heads)
    uint32_t ssm_inner_size  = 2048;    // DeltaNet hidden (= n_v_heads * state_size)
    uint32_t full_attn_interval = 4;    // every 4th layer is full attention
    uint32_t eos_token_id    = 248046;
    uint32_t pad_token_id    = 248055;
    uint32_t context_length  = 262144;
    bool     tie_word_embeddings = true;  // false for 9B+ (separate output.weight)

    bool is_full_attention_layer(int layer_idx) const {
        return (layer_idx + 1) % full_attn_interval == 0;
    }

    // QKV projection output dimension: Q(k_heads*dk) + K(k_heads*dk) + V(v_heads*dv)
    uint32_t ssm_qkv_dim() const {
        return ssm_n_k_heads * ssm_state_size * 2 + ssm_n_v_heads * ssm_state_size;
    }
};

} // namespace gwen

// Debug logging macro — compiles to nop unless GWEN_DEBUG is defined
#ifdef GWEN_DEBUG
#define GWEN_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define GWEN_LOG(...) ((void)0)
#endif
