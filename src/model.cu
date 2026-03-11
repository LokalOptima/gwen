#include "gwen/model.h"
#include <fstream>

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

// ============================================================
// MTP weight loading (GWMT binary format)
// ============================================================

void Model::load_mtp(const std::string& mtp_path) {
    std::ifstream f(mtp_path, std::ios::binary);
    GWEN_CHECK(f.is_open(), ("Failed to open MTP file: " + mtp_path).c_str());

    // Read header
    char magic[4];
    f.read(magic, 4);
    GWEN_CHECK(memcmp(magic, "GWMT", 4) == 0, "Invalid MTP file magic (expected GWMT)");

    uint32_t version, n_tensors;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&n_tensors), 4);
    GWEN_CHECK(version == 1, "Unsupported MTP file version");

    printf("Loading MTP weights: %u tensors from %s\n", n_tensors, mtp_path.c_str());

    // Read tensors
    mtp_host_buffers.resize(n_tensors);
    size_t total_bytes = 0;

    for (uint32_t i = 0; i < n_tensors; i++) {
        // Read name
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);

        // Read dtype, ndims, shape
        uint32_t dtype, ndims;
        f.read(reinterpret_cast<char*>(&dtype), 4);
        f.read(reinterpret_cast<char*>(&ndims), 4);
        std::vector<uint64_t> shape(ndims);
        f.read(reinterpret_cast<char*>(shape.data()), ndims * 8);

        // Read data
        uint64_t data_size;
        f.read(reinterpret_cast<char*>(&data_size), 8);
        mtp_host_buffers[i].resize(data_size);
        f.read(reinterpret_cast<char*>(mtp_host_buffers[i].data()), data_size);

        // Compute n_elements
        size_t n_elements = 1;
        for (auto s : shape) n_elements *= s;

        // Build WeightRef
        WeightRef w;
        w.host_data = mtp_host_buffers[i].data();
        if (dtype == 0)      w.type = GGMLType::F32;
        else if (dtype == 1) w.type = GGMLType::F16;
        else if (dtype == 8) w.type = GGMLType::Q8_0;
        else GWEN_CHECK(false, "Unsupported MTP weight dtype");
        w.n_elements = n_elements;
        w.size_bytes = data_size;
        w.shape = shape;

        total_bytes += data_size;

        // Map to MTP weight fields by name
        if (name == "mtp.fc.weight") {
            mtp.fc = w;
        } else if (name == "mtp.pre_fc_norm_embedding.weight") {
            mtp.pre_fc_norm_embed = w;
        } else if (name == "mtp.pre_fc_norm_hidden.weight") {
            mtp.pre_fc_norm_hidden = w;
        } else if (name == "mtp.layers.0.self_attn.q_proj.weight") {
            mtp.layer.attn_q = w;
        } else if (name == "mtp.layers.0.self_attn.k_proj.weight") {
            mtp.layer.attn_k = w;
        } else if (name == "mtp.layers.0.self_attn.v_proj.weight") {
            mtp.layer.attn_v = w;
        } else if (name == "mtp.layers.0.self_attn.o_proj.weight") {
            mtp.layer.attn_output = w;
        } else if (name == "mtp.layers.0.self_attn.q_norm.weight") {
            mtp.layer.attn_q_norm = w;
        } else if (name == "mtp.layers.0.self_attn.k_norm.weight") {
            mtp.layer.attn_k_norm = w;
        } else if (name == "mtp.layers.0.input_layernorm.weight") {
            mtp.layer.attn_norm = w;
        } else if (name == "mtp.layers.0.post_attention_layernorm.weight") {
            mtp.layer.post_attn_norm = w;
        } else if (name == "mtp.layers.0.mlp.gate_proj.weight") {
            mtp.layer.ffn_gate = w;
        } else if (name == "mtp.layers.0.mlp.up_proj.weight") {
            mtp.layer.ffn_up = w;
        } else if (name == "mtp.layers.0.mlp.down_proj.weight") {
            mtp.layer.ffn_down = w;
        } else if (name == "mtp.norm.weight") {
            mtp.output_norm = w;
        } else {
            printf("  Warning: unknown MTP tensor: %s\n", name.c_str());
        }

        printf("  %-50s [", name.c_str());
        for (uint32_t d = 0; d < ndims; d++) {
            if (d > 0) printf(", ");
            printf("%lu", shape[d]);
        }
        const char* dtype_str = "???";
        if (dtype == 0) dtype_str = "F32";
        else if (dtype == 1) dtype_str = "F16";
        else if (dtype == 8) dtype_str = "Q8_0";
        printf("] %s  %.1f KB\n", dtype_str, data_size / 1024.0f);
    }

    has_mtp = true;
    printf("MTP weights loaded: %.1f MB total\n", total_bytes / 1024.0 / 1024.0);
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

    // Upload MTP weights if loaded
    if (has_mtp) {
        upload_weight(allocator, mtp.fc);
        upload_weight(allocator, mtp.pre_fc_norm_embed);
        upload_weight(allocator, mtp.pre_fc_norm_hidden);
        upload_weight(allocator, mtp.layer.attn_norm);
        upload_weight(allocator, mtp.layer.attn_q);
        upload_weight(allocator, mtp.layer.attn_k);
        upload_weight(allocator, mtp.layer.attn_v);
        upload_weight(allocator, mtp.layer.attn_q_norm);
        upload_weight(allocator, mtp.layer.attn_k_norm);
        upload_weight(allocator, mtp.layer.attn_output);
        upload_weight(allocator, mtp.layer.post_attn_norm);
        upload_weight(allocator, mtp.layer.ffn_gate);
        upload_weight(allocator, mtp.layer.ffn_up);
        upload_weight(allocator, mtp.layer.ffn_down);
        upload_weight(allocator, mtp.output_norm);
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
