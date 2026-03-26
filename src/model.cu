#include "gwen/model.h"
#include "gwen/kernels.h"
#include <fstream>

namespace gwen {


// Helper: fill WeightRef from a GGUF tensor
// GGML stores 2D shapes as [ne0=cols=in_features, ne1=rows=out_features]
// GWEN convention: shape[0]=out_features, shape[1]=in_features
// So we swap for 2D tensors at load time.
static WeightRef weight_from_tensor(const GGUFTensor& t) {
    WeightRef w;
    w.host_data = t.data;
    w.type = t.type;
    w.n_elements = t.n_elements;
    w.size_bytes = t.size_bytes;
    w.shape = t.shape;
    // Swap GGML [cols, rows] → GWEN [rows, cols] for 2D weight matrices
    if (w.shape.size() == 2) {
        std::swap(w.shape[0], w.shape[1]);
    }
    return w;
}

static WeightRef weight_from_tensor(const GGUFFile& gguf, const std::string& name) {
    return weight_from_tensor(gguf.get_tensor(name));
}

std::unique_ptr<Model> Model::load(const std::string& gguf_path) {
    auto model = std::make_unique<Model>();

    model->gguf = GGUFFile::open(gguf_path);
    model->config = model->gguf->build_config();
    const auto& cfg = model->config;
    const auto& gguf = *model->gguf;

    // Global weights
    model->token_embd = weight_from_tensor(gguf, "token_embd.weight");
    model->output_norm = weight_from_tensor(gguf, "output_norm.weight");

    // Separate output.weight (lm_head) for models with tie_word_embeddings=false
    auto* output_tensor = gguf.find_tensor("output.weight");
    if (output_tensor) {
        model->output_weight = weight_from_tensor(*output_tensor);
        model->config.tie_word_embeddings = false;
        fprintf(stderr, "  output.weight: [%lu, %lu] %s (separate lm_head)\n",
                output_tensor->shape[0], output_tensor->shape[1],
                ggml_type_name(output_tensor->type));
    } else {
        model->config.tie_word_embeddings = true;
    }

    auto load = [&](const std::string& name) -> WeightRef {
        return weight_from_tensor(gguf, name);
    };

    // Per-layer weights
    model->layers.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        auto& layer = model->layers[i];
        std::string prefix = "blk." + std::to_string(i) + ".";

        layer.is_full_attention = cfg.is_full_attention_layer(i);

        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            w.attn_norm      = load(prefix + "attn_norm.weight");
            w.attn_q         = load(prefix + "attn_q.weight");
            w.attn_k         = load(prefix + "attn_k.weight");
            w.attn_v         = load(prefix + "attn_v.weight");
            w.attn_q_norm    = load(prefix + "attn_q_norm.weight");
            w.attn_k_norm    = load(prefix + "attn_k_norm.weight");
            w.attn_output    = load(prefix + "attn_output.weight");
            w.post_attn_norm = load(prefix + "post_attention_norm.weight");
            w.ffn_gate       = load(prefix + "ffn_gate.weight");
            w.ffn_up         = load(prefix + "ffn_up.weight");
            w.ffn_down       = load(prefix + "ffn_down.weight");
        } else {
            auto& w = layer.deltanet;
            w.attn_norm      = load(prefix + "attn_norm.weight");
            w.attn_qkv       = load(prefix + "attn_qkv.weight");
            w.attn_gate      = load(prefix + "attn_gate.weight");
            w.ssm_conv1d     = load(prefix + "ssm_conv1d.weight");
            w.ssm_a          = load(prefix + "ssm_a");
            w.ssm_dt_bias    = load(prefix + "ssm_dt.bias");
            w.ssm_alpha      = load(prefix + "ssm_alpha.weight");
            w.ssm_beta       = load(prefix + "ssm_beta.weight");
            w.ssm_norm       = load(prefix + "ssm_norm.weight");
            w.ssm_out        = load(prefix + "ssm_out.weight");
            w.post_attn_norm = load(prefix + "post_attention_norm.weight");
            w.ffn_gate       = load(prefix + "ffn_gate.weight");
            w.ffn_up         = load(prefix + "ffn_up.weight");
            w.ffn_down       = load(prefix + "ffn_down.weight");
        }
    }

    return model;
}

// Upload all weight tensors to GPU
static void upload_weight(CudaAllocator& alloc, WeightRef& w) {
    if (w.host_data && w.size_bytes > 0 && !w.on_device()) {
        w.device_data = alloc.upload(w.host_data, w.size_bytes);
    }
}

void Model::upload_weights(CudaAllocator& allocator) {
    upload_weight(allocator, token_embd);
    if (!config.tie_word_embeddings) {
        upload_weight(allocator, output_weight);
    }
    upload_weight(allocator, output_norm);

    for (auto& layer : layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            upload_weight(allocator, w.attn_norm);
            upload_weight(allocator, w.attn_q);
            upload_weight(allocator, w.attn_k);
            upload_weight(allocator, w.attn_v);
            upload_weight(allocator, w.attn_q_norm);
            upload_weight(allocator, w.attn_k_norm);
            upload_weight(allocator, w.attn_output);
            upload_weight(allocator, w.post_attn_norm);
            upload_weight(allocator, w.ffn_gate);
            upload_weight(allocator, w.ffn_up);
            upload_weight(allocator, w.ffn_down);
        } else {
            auto& w = layer.deltanet;
            upload_weight(allocator, w.attn_norm);
            upload_weight(allocator, w.attn_qkv);
            upload_weight(allocator, w.attn_gate);
            upload_weight(allocator, w.ssm_conv1d);
            upload_weight(allocator, w.ssm_a);
            upload_weight(allocator, w.ssm_dt_bias);
            upload_weight(allocator, w.ssm_alpha);
            upload_weight(allocator, w.ssm_beta);
            upload_weight(allocator, w.ssm_norm);
            upload_weight(allocator, w.ssm_out);
            upload_weight(allocator, w.post_attn_norm);
            upload_weight(allocator, w.ffn_gate);
            upload_weight(allocator, w.ffn_up);
            upload_weight(allocator, w.ffn_down);
        }
    }
}

void Model::print_info() const {
    fprintf(stderr, "=== GWEN Model Info ===\n");
    fprintf(stderr, "Model: %s\n", gguf ? gguf->path().c_str() : "(unknown)");
    fprintf(stderr, "Layers: %u (%u DeltaNet + %u FullAttn)\n",
           config.n_layers,
           config.n_layers - config.n_layers / config.full_attn_interval,
           config.n_layers / config.full_attn_interval);
    fprintf(stderr, "Embed dim: %u\n", config.n_embed);
    fprintf(stderr, "FFN dim: %u\n", config.n_ff);
    fprintf(stderr, "Vocab: %u\n", config.n_vocab);
    fprintf(stderr, "Full Attn: %u heads (%u KV), head_dim=%u\n",
           config.n_head, config.n_head_kv, config.head_dim);
    fprintf(stderr, "DeltaNet: K=%u V=%u heads, state=%ux%u, inner=%u, qkv=%u\n",
           config.ssm_n_k_heads, config.ssm_n_v_heads,
           config.ssm_state_size, config.ssm_state_size,
           config.ssm_inner_size, config.ssm_qkv_dim());
    fprintf(stderr, "RoPE: theta=%.0f, dim=%u, sections=[%d,%d,%d,%d]\n",
           config.rope_theta, config.rope_dim,
           config.rope_sections[0], config.rope_sections[1],
           config.rope_sections[2], config.rope_sections[3]);
    fprintf(stderr, "Context length: %u\n", config.context_length);
    fprintf(stderr, "Tie embeddings: %s\n", config.tie_word_embeddings ? "yes" : "no");
    fprintf(stderr, "RMSNorm eps: %e\n", config.rms_norm_eps);

    fprintf(stderr, "\nLayer pattern: ");
    for (uint32_t i = 0; i < config.n_layers; i++) {
        fprintf(stderr, "%c", config.is_full_attention_layer(i) ? 'A' : 'D');
    }
    fprintf(stderr, "\n");

    size_t total_bytes = token_embd.size_bytes + output_norm.size_bytes +
                         (config.tie_word_embeddings ? 0 : output_weight.size_bytes);
    for (auto& layer : layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            total_bytes += w.attn_norm.size_bytes + w.attn_q.size_bytes +
                          w.attn_k.size_bytes + w.attn_v.size_bytes +
                          w.attn_q_norm.size_bytes + w.attn_k_norm.size_bytes +
                          w.attn_output.size_bytes + w.post_attn_norm.size_bytes +
                          w.ffn_gate.size_bytes + w.ffn_up.size_bytes +
                          w.ffn_down.size_bytes;
        } else {
            auto& w = layer.deltanet;
            total_bytes += w.attn_norm.size_bytes + w.attn_qkv.size_bytes +
                          w.attn_gate.size_bytes + w.ssm_conv1d.size_bytes +
                          w.ssm_a.size_bytes + w.ssm_dt_bias.size_bytes +
                          w.ssm_alpha.size_bytes + w.ssm_beta.size_bytes +
                          w.ssm_norm.size_bytes + w.ssm_out.size_bytes +
                          w.post_attn_norm.size_bytes +
                          w.ffn_gate.size_bytes + w.ffn_up.size_bytes +
                          w.ffn_down.size_bytes;
        }
    }
    fprintf(stderr, "Total weight size: %.1f MB\n", total_bytes / 1024.0 / 1024.0);
    if (gguf) fprintf(stderr, "Tensors: %zu\n", gguf->n_tensors());
}

} // namespace gwen
