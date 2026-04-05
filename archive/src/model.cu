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
    GWEN_CHECK(version >= 1 && version <= 4, "Unsupported MTP file version (expected 1-4)");

    fprintf(stderr, "Loading MTP weights: %u tensors from %s\n", n_tensors, mtp_path.c_str());

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
        } else if (name == "mtp.lm_head.weight") {
            // Reduced lm_head from fine-tuning — wire into reduced_lm_head
            reduced_lm_head.weights = w;
            reduced_lm_head.type = w.type;
            // For FP16: row_bytes = n_embed * 2
            if (w.type == GGMLType::F16) {
                reduced_lm_head.row_bytes = config.n_embed * 2;
            }
        } else {
            fprintf(stderr, "  Warning: unknown MTP tensor: %s\n", name.c_str());
        }

        fprintf(stderr, "  %-50s [", name.c_str());
        for (uint32_t d = 0; d < ndims; d++) {
            if (d > 0) fprintf(stderr, ", ");
            fprintf(stderr, "%lu", shape[d]);
        }
        const char* dtype_str = "???";
        if (dtype == 0) dtype_str = "F32";
        else if (dtype == 1) dtype_str = "F16";
        else if (dtype == 8) dtype_str = "Q8_0";
        fprintf(stderr, "] %s  %.1f KB\n", dtype_str, data_size / 1024.0f);
    }

    has_mtp = true;
    fprintf(stderr, "MTP weights loaded: %.1f MB total\n", total_bytes / 1024.0 / 1024.0);

    // v3+ footer: restricted vocab mapping [K, restricted_ids[K]]
    if (version >= 3 && f.peek() != EOF) {
        uint32_t K;
        f.read(reinterpret_cast<char*>(&K), 4);
        if (K > 0) {
            reduced_lm_head.token_ids.resize(K);
            f.read(reinterpret_cast<char*>(reduced_lm_head.token_ids.data()), K * sizeof(int32_t));
            reduced_lm_head.K = K;
            // lm_head weights are in MTP tensors (mtp.lm_head.weight), FP16
            // The reduced_lm_head.weights will be set during upload from the MTP lm_head tensor
            has_reduced_lm_head = true;
            fprintf(stderr, "GWMT v%u: restricted vocab K=%u embedded in MTP file\n", version, K);
        }

        // v4 footer: has_idk flag
        if (version >= 4 && f.peek() != EOF) {
            uint8_t flag;
            f.read(reinterpret_cast<char*>(&flag), 1);
            reduced_lm_head.has_idk = (flag != 0);
            if (reduced_lm_head.has_idk) {
                fprintf(stderr, "GWMT v4: IDK token enabled (index %u maps to -1)\n", K);
            }
        }
    }
}

// ============================================================
// Reduced LM head loading (GWRL binary format)
// ============================================================

void Model::load_reduced_lm_head(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GWEN_CHECK(f.is_open(), ("Failed to open reduced LM head file: " + path).c_str());

    // Read header
    char magic[4];
    f.read(magic, 4);
    GWEN_CHECK(memcmp(magic, "GWRL", 4) == 0, "Invalid reduced LM head file magic (expected GWRL)");

    uint32_t version, K, n_embed, ggml_type, row_bytes;
    f.read(reinterpret_cast<char*>(&version), 4);
    GWEN_CHECK(version == 1, "Unsupported GWRL version");

    f.read(reinterpret_cast<char*>(&K), 4);
    f.read(reinterpret_cast<char*>(&n_embed), 4);
    f.read(reinterpret_cast<char*>(&ggml_type), 4);
    f.read(reinterpret_cast<char*>(&row_bytes), 4);

    fprintf(stderr, "Loading reduced LM head: %u tokens, %u embed, type=%u, %u bytes/row\n",
           K, n_embed, ggml_type, row_bytes);

    // Read token ID mapping
    reduced_lm_head.token_ids.resize(K);
    f.read(reinterpret_cast<char*>(reduced_lm_head.token_ids.data()), K * sizeof(int32_t));

    // Read weight data
    size_t weight_bytes = (size_t)K * row_bytes;
    reduced_lm_head.host_buffer.resize(weight_bytes);
    f.read(reinterpret_cast<char*>(reduced_lm_head.host_buffer.data()), weight_bytes);

    // Set up WeightRef
    reduced_lm_head.weights.host_data = reduced_lm_head.host_buffer.data();
    reduced_lm_head.weights.type = static_cast<GGMLType>(ggml_type);
    reduced_lm_head.weights.n_elements = (size_t)K * n_embed;
    reduced_lm_head.weights.size_bytes = weight_bytes;
    reduced_lm_head.weights.shape = {n_embed, K};  // GGML convention
    reduced_lm_head.K = K;
    reduced_lm_head.row_bytes = row_bytes;
    reduced_lm_head.type = static_cast<GGMLType>(ggml_type);

    has_reduced_lm_head = true;
    fprintf(stderr, "Reduced LM head loaded: %u tokens, %.1f MB (%.1fx reduction)\n",
           K, weight_bytes / 1024.0 / 1024.0,
           (float)config.n_vocab / K);
}

// Upload all weight tensors to GPU
// Forward declarations for dequant kernels
void gwen_dequant(const void* src, half* dst, int n, GGMLType type, cudaStream_t stream);

static void upload_weight(CudaAllocator& alloc, WeightRef& w) {
    if (w.host_data && w.size_bytes > 0 && !w.on_device()) {
        w.device_data = alloc.upload(w.host_data, w.size_bytes);
    }
}

// Upload + pre-dequantize to FP16 for prefill GEMMs (avoids per-call dequant overhead)
static void upload_weight_with_fp16(CudaAllocator& alloc, WeightRef& w) {
    upload_weight(alloc, w);
    if (w.on_device() && w.type != GGMLType::F32 && w.type != GGMLType::F16 && w.n_elements > 0) {
        size_t fp16_bytes = w.n_elements * sizeof(half);
        w.fp16_data = static_cast<half*>(alloc.alloc(fp16_bytes));
        gwen_dequant(w.device_data, w.fp16_data, w.n_elements, w.type, 0);
    }
}

void Model::upload_weights(CudaAllocator& allocator) {
    upload_weight(allocator, token_embd);
    upload_weight(allocator, output_norm);

    for (auto& layer : layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            upload_weight(allocator, w.attn_norm);       // F32 norm — no FP16 needed
            upload_weight_with_fp16(allocator, w.attn_q);
            upload_weight_with_fp16(allocator, w.attn_k);
            upload_weight_with_fp16(allocator, w.attn_v);
            upload_weight(allocator, w.attn_q_norm);     // F32 norm
            upload_weight(allocator, w.attn_k_norm);     // F32 norm
            upload_weight_with_fp16(allocator, w.attn_output);
            upload_weight(allocator, w.post_attn_norm);  // F32 norm
            upload_weight_with_fp16(allocator, w.ffn_gate);
            upload_weight_with_fp16(allocator, w.ffn_up);
            upload_weight_with_fp16(allocator, w.ffn_down);
        } else {
            auto& w = layer.deltanet;
            upload_weight(allocator, w.attn_norm);       // F32 norm
            upload_weight_with_fp16(allocator, w.attn_qkv);
            upload_weight_with_fp16(allocator, w.attn_gate);
            upload_weight(allocator, w.ssm_conv1d);      // F32 conv weights
            upload_weight(allocator, w.ssm_a);           // F32 scalar
            upload_weight(allocator, w.ssm_dt_bias);     // F32 scalar
            upload_weight_with_fp16(allocator, w.ssm_alpha);  // dequant to FP16 (any quant type)
            upload_weight_with_fp16(allocator, w.ssm_beta);   // dequant to FP16 (any quant type)
            upload_weight(allocator, w.ssm_norm);        // F32 norm
            upload_weight_with_fp16(allocator, w.ssm_out);
            upload_weight(allocator, w.post_attn_norm);  // F32 norm
            upload_weight_with_fp16(allocator, w.ffn_gate);
            upload_weight_with_fp16(allocator, w.ffn_up);
            upload_weight_with_fp16(allocator, w.ffn_down);
        }
    }

    // Upload reduced LM head if loaded
    if (has_reduced_lm_head) {
        upload_weight(allocator, reduced_lm_head.weights);
        // Upload token ID mapping to device
        size_t ids_bytes = reduced_lm_head.K * sizeof(int32_t);
        reduced_lm_head.d_token_ids = static_cast<int*>(allocator.upload(
            reduced_lm_head.token_ids.data(), ids_bytes));
        fprintf(stderr, "Reduced LM head uploaded: %.1f MB weights + %.1f KB token map\n",
               reduced_lm_head.weights.size_bytes / 1024.0 / 1024.0,
               ids_bytes / 1024.0);
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
    fprintf(stderr, "=== GWEN Model Info ===\n");
    fprintf(stderr, "Model: %s\n", gguf->path().c_str());
    fprintf(stderr, "Layers: %u (%u DeltaNet + %u FullAttn)\n",
           config.n_layers,
           config.n_layers - config.n_layers / config.full_attn_interval,
           config.n_layers / config.full_attn_interval);
    fprintf(stderr, "Embed dim: %u\n", config.n_embed);
    fprintf(stderr, "FFN dim: %u\n", config.n_ff);
    fprintf(stderr, "Vocab: %u\n", config.n_vocab);
    fprintf(stderr, "Full Attn: %u heads (%u KV), head_dim=%u\n",
           config.n_head, config.n_head_kv, config.head_dim);
    fprintf(stderr, "DeltaNet: %u heads, state=%ux%u, inner=%u\n",
           config.ssm_n_heads, config.ssm_state_size, config.ssm_state_size,
           config.ssm_inner_size);
    fprintf(stderr, "RoPE: theta=%.0f, dim=%u, sections=[%d,%d,%d,%d]\n",
           config.rope_theta, config.rope_dim,
           config.rope_sections[0], config.rope_sections[1],
           config.rope_sections[2], config.rope_sections[3]);
    fprintf(stderr, "Context length: %u\n", config.context_length);
    fprintf(stderr, "RMSNorm eps: %e\n", config.rms_norm_eps);

    // Print layer pattern
    fprintf(stderr, "\nLayer pattern: ");
    for (uint32_t i = 0; i < config.n_layers; i++) {
        fprintf(stderr, "%c", config.is_full_attention_layer(i) ? 'A' : 'D');
    }
    fprintf(stderr, "\n");

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
    fprintf(stderr, "Total weight size: %.1f MB\n", total_bytes / 1024.0 / 1024.0);
    fprintf(stderr, "Tensors: %zu\n", gguf->n_tensors());
}

} // namespace gwen
