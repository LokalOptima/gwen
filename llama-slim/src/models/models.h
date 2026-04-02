#pragma once

#include "llama-model.h"
#include "llama-graph.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

//
// base classes
//

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);
};

//
// models
//

struct llm_build_qwen35 : public llm_build_delta_net_base {
    llm_build_qwen35(const llama_model & model, const llm_graph_params & params);
private:
    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int * sections,
                        int   il);

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    const llama_model & model;
};
