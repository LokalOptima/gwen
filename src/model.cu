#include "gwen/model.h"

namespace gwen {

// Helper: fill WeightRef from a GGUF tensor
static WeightRef weight_from_tensor(const GGUFTensor& t) {
    WeightRef w;
    w.host_data = t.data;
    w.type = t.type;
    w.n_elements = t.n_elements;
    w.size_bytes = t.size_bytes;
    w.shape = t.shape;
    return w;
}

static WeightRef weight_from_tensor(const GGUFFile& gguf, const std::string& name) {
    return weight_from_tensor(gguf.get_tensor(name));
}

std::unique_ptr<Model> Model::load(const std::string& gguf_path) {
    auto model = std::make_unique<Model>();

    // Open GGUF
    model->gguf = GGUFFile::open(gguf_path);
    model->config = model->gguf->build_config();
    const auto& cfg = model->config;
    const auto& gguf = *model->gguf;

    // Global weights
    model->token_embd = weight_from_tensor(gguf, "token_embd.weight");
    model->output_norm = weight_from_tensor(gguf, "output_norm.weight");

    // Per-layer weights
    model->layers.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        auto& layer = model->layers[i];
        std::string prefix = "blk." + std::to_string(i) + ".";

        layer.is_full_attention = cfg.is_full_attention_layer(i);

        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            w.attn_norm      = weight_from_tensor(gguf, prefix + "attn_norm.weight");
            w.attn_q         = weight_from_tensor(gguf, prefix + "attn_q.weight");
            w.attn_k         = weight_from_tensor(gguf, prefix + "attn_k.weight");
            w.attn_v         = weight_from_tensor(gguf, prefix + "attn_v.weight");
            w.attn_q_norm    = weight_from_tensor(gguf, prefix + "attn_q_norm.weight");
            w.attn_k_norm    = weight_from_tensor(gguf, prefix + "attn_k_norm.weight");
            w.attn_output    = weight_from_tensor(gguf, prefix + "attn_output.weight");
            w.post_attn_norm = weight_from_tensor(gguf, prefix + "post_attention_norm.weight");
            w.ffn_gate       = weight_from_tensor(gguf, prefix + "ffn_gate.weight");
            w.ffn_up         = weight_from_tensor(gguf, prefix + "ffn_up.weight");
            w.ffn_down       = weight_from_tensor(gguf, prefix + "ffn_down.weight");
        } else {
            auto& w = layer.deltanet;
            w.attn_norm      = weight_from_tensor(gguf, prefix + "attn_norm.weight");
            w.attn_qkv       = weight_from_tensor(gguf, prefix + "attn_qkv.weight");
            w.attn_gate      = weight_from_tensor(gguf, prefix + "attn_gate.weight");
            w.ssm_conv1d     = weight_from_tensor(gguf, prefix + "ssm_conv1d.weight");
            w.ssm_a          = weight_from_tensor(gguf, prefix + "ssm_a");
            w.ssm_dt_bias    = weight_from_tensor(gguf, prefix + "ssm_dt.bias");
            w.ssm_alpha      = weight_from_tensor(gguf, prefix + "ssm_alpha.weight");
            w.ssm_beta       = weight_from_tensor(gguf, prefix + "ssm_beta.weight");
            w.ssm_norm       = weight_from_tensor(gguf, prefix + "ssm_norm.weight");
            w.ssm_out        = weight_from_tensor(gguf, prefix + "ssm_out.weight");
            w.post_attn_norm = weight_from_tensor(gguf, prefix + "post_attention_norm.weight");
            w.ffn_gate       = weight_from_tensor(gguf, prefix + "ffn_gate.weight");
            w.ffn_up         = weight_from_tensor(gguf, prefix + "ffn_up.weight");
            w.ffn_down       = weight_from_tensor(gguf, prefix + "ffn_down.weight");
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
    printf("=== GWEN Model Info ===\n");
    printf("Model: %s\n", gguf->path().c_str());
    printf("Layers: %u (%u DeltaNet + %u FullAttn)\n",
           config.n_layers,
           config.n_layers - config.n_layers / config.full_attn_interval,
           config.n_layers / config.full_attn_interval);
    printf("Embed dim: %u\n", config.n_embed);
    printf("FFN dim: %u\n", config.n_ff);
    printf("Vocab: %u\n", config.n_vocab);
    printf("Full Attn: %u heads (%u KV), head_dim=%u\n",
           config.n_head, config.n_head_kv, config.head_dim);
    printf("DeltaNet: %u heads, state=%ux%u, inner=%u\n",
           config.ssm_n_heads, config.ssm_state_size, config.ssm_state_size,
           config.ssm_inner_size);
    printf("RoPE: theta=%.0f, dim=%u, sections=[%d,%d,%d,%d]\n",
           config.rope_theta, config.rope_dim,
           config.rope_sections[0], config.rope_sections[1],
           config.rope_sections[2], config.rope_sections[3]);
    printf("Context length: %u\n", config.context_length);
    printf("RMSNorm eps: %e\n", config.rms_norm_eps);

    // Print layer pattern
    printf("\nLayer pattern: ");
    for (uint32_t i = 0; i < config.n_layers; i++) {
        printf("%c", config.is_full_attention_layer(i) ? 'A' : 'D');
    }
    printf("\n");

    // Weight summary
    size_t total_bytes = token_embd.size_bytes + output_norm.size_bytes;
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
    printf("Total weight size: %.1f MB\n", total_bytes / 1024.0 / 1024.0);
    printf("Tensors: %zu\n", gguf->n_tensors());
}

} // namespace gwen
