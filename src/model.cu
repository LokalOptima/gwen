#include "gwen/model.h"
#include "gwen/kernels.h"
#include <fstream>
#include <unordered_map>

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

// IQ4_XS lookup table (non-linear 4-bit quantization)
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Dequantize IQ4_XS block data to FP16 (CPU-side, at model load time)
// block_iq4_xs: {half d, uint16_t scales_h, uint8_t scales_l[4], uint8_t qs[128]} = 136 bytes per 256 elements
static void dequant_iq4xs_to_fp16(const void* src, uint16_t* dst_fp16, size_t n_elements) {
    const uint8_t* data = static_cast<const uint8_t*>(src);
    size_t n_blocks = n_elements / 256;

    for (size_t i = 0; i < n_blocks; i++) {
        const uint8_t* block = data + i * 136;

        // Read d (FP16 stored as uint16_t)
        uint16_t d_bits;
        memcpy(&d_bits, block, 2);
        // Convert FP16 bits to float
        uint32_t sign = (d_bits >> 15) & 1;
        uint32_t exp = (d_bits >> 10) & 0x1F;
        uint32_t mant = d_bits & 0x3FF;
        float d;
        if (exp == 0) {
            d = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (1.0f / 16384.0f);  // subnormal
        } else if (exp == 31) {
            d = 0.0f;  // inf/nan, shouldn't happen for scales
        } else {
            uint32_t f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&d, &f32_bits, 4);
        }

        uint16_t scales_h;
        memcpy(&scales_h, block + 2, 2);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;

        for (int ib = 0; ib < 8; ib++) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                     (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (ls - 32);

            for (int j = 0; j < 16; j++) {
                float v0 = dl * kvalues_iq4nl[qs[j] & 0xF];
                float v1 = dl * kvalues_iq4nl[qs[j] >> 4];

                // Convert float to FP16 bits (simple conversion)
                auto f32_to_fp16 = [](float val) -> uint16_t {
                    uint32_t bits;
                    memcpy(&bits, &val, 4);
                    uint32_t s = (bits >> 16) & 0x8000;
                    int32_t e = ((bits >> 23) & 0xFF) - 127 + 15;
                    uint32_t m = bits & 0x7FFFFF;
                    if (e <= 0) return s;  // underflow to zero
                    if (e >= 31) return s | 0x7C00;  // overflow to inf
                    return s | (e << 10) | (m >> 13);
                };

                dst_fp16[ib * 32 + j] = f32_to_fp16(v0);
                dst_fp16[ib * 32 + j + 16] = f32_to_fp16(v1);
            }
            qs += 16;
        }
        dst_fp16 += 256;
    }
}

// Convert an IQ4_XS tensor to FP16, storing result in a new host buffer
// Returns the buffer; caller must keep it alive for the model's lifetime
static std::vector<uint8_t> convert_iq4xs_to_fp16(const GGUFTensor& t) {
    size_t fp16_bytes = t.n_elements * sizeof(uint16_t);
    std::vector<uint8_t> buf(fp16_bytes);
    dequant_iq4xs_to_fp16(t.data, reinterpret_cast<uint16_t*>(buf.data()), t.n_elements);
    return buf;
}

// Load tensor, converting IQ4_XS to FP16 at load time
static WeightRef weight_from_tensor_convert(const GGUFFile& gguf, const std::string& name,
                                             std::vector<std::vector<uint8_t>>& converted_buffers) {
    const auto& t = gguf.get_tensor(name);
    if (t.type == GGMLType::IQ4_XS) {
        converted_buffers.push_back(convert_iq4xs_to_fp16(t));
        auto& buf = converted_buffers.back();
        WeightRef w;
        w.host_data = buf.data();
        w.type = GGMLType::F16;
        w.n_elements = t.n_elements;
        w.size_bytes = buf.size();
        w.shape = t.shape;
        // Swap GGML [cols, rows] → GWEN [rows, cols] for 2D weight matrices
        if (w.shape.size() == 2) {
            std::swap(w.shape[0], w.shape[1]);
        }
        // Sanity check: verify first few converted values
        const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(buf.data());
        bool has_nan = false;
        for (int i = 0; i < 8; i++) {
            uint16_t h = fp16_data[i];
            uint32_t exp = (h >> 10) & 0x1F;
            if (exp == 31) has_nan = true;
        }
        fprintf(stderr, "  Converted %s: IQ4_XS → F16 (%.1f MB → %.1f MB)%s\n",
                name.c_str(), t.size_bytes / 1024.0 / 1024.0, buf.size() / 1024.0 / 1024.0,
                has_nan ? " WARNING: NaN in first 8 values!" : "");
        return w;
    }
    return weight_from_tensor(t);
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

    // Helper: load tensor with automatic IQ4_XS → FP16 conversion
    auto load = [&](const std::string& name) -> WeightRef {
        return weight_from_tensor_convert(gguf, name, model->converted_buffers_);
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

    GWEN_LOG("Loading MTP weights: %u tensors from %s\n", n_tensors, mtp_path.c_str());

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

#ifdef GWEN_DEBUG
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
#endif
    }

    has_mtp = true;
    GWEN_LOG("MTP weights loaded: %.1f MB total\n", total_bytes / 1024.0 / 1024.0);

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
            GWEN_LOG("GWMT v%u: restricted vocab K=%u embedded in MTP file\n", version, K);
        }

        // v4 footer: has_idk flag
        if (version >= 4 && f.peek() != EOF) {
            uint8_t flag;
            f.read(reinterpret_cast<char*>(&flag), 1);
            reduced_lm_head.has_idk = (flag != 0);
            if (reduced_lm_head.has_idk) {
                GWEN_LOG("GWMT v4: IDK token enabled (index %u maps to -1)\n", K);
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

    GWEN_LOG("Loading reduced LM head: %u tokens, %u embed, type=%u, %u bytes/row\n",
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
    GWEN_LOG("Reduced LM head loaded: %u tokens, %.1f MB (%.1fx reduction)\n",
           K, weight_bytes / 1024.0 / 1024.0,
           (float)config.n_vocab / K);
}

// ============================================================
// GWFP8 loader — FP8 E4M3 weights with per-row scales
// ============================================================

std::unique_ptr<Model> Model::load_fp8(const std::string& fp8_path) {
    auto model = std::make_unique<Model>();

    // Read entire file into memory (will be mmap'd in future)
    std::ifstream f(fp8_path, std::ios::binary | std::ios::ate);
    GWEN_CHECK(f.is_open(), ("Failed to open: " + fp8_path).c_str());
    size_t file_size = f.tellg();
    f.seekg(0);
    model->gwfp8_file_data_.resize(file_size);
    f.read(reinterpret_cast<char*>(model->gwfp8_file_data_.data()), file_size);
    f.close();

    const uint8_t* base = model->gwfp8_file_data_.data();
    size_t offset = 0;

    // Parse header
    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        memcpy(&v, base + offset, 4);
        offset += 4;
        return v;
    };
    auto read_u64 = [&]() -> uint64_t {
        uint64_t v;
        memcpy(&v, base + offset, 8);
        offset += 8;
        return v;
    };

    GWEN_CHECK(memcmp(base, "GWF8", 4) == 0, "Invalid GWFP8 magic");
    offset = 4;
    uint32_t version = read_u32();
    GWEN_CHECK(version == 1, "Unsupported GWFP8 version");
    uint32_t n_tensors = read_u32();
    uint64_t header_size = read_u64();
    (void)header_size;

    // Parse tensor headers and build name→WeightRef map
    struct TensorInfo {
        std::string name;
        uint32_t dtype;
        std::vector<uint64_t> shape;
        uint32_t scale_mode;
        uint32_t n_scales;
        uint64_t scale_offset;
        uint64_t data_offset;
        uint64_t data_size;
    };
    std::vector<TensorInfo> tensor_infos;
    tensor_infos.reserve(n_tensors);

    for (uint32_t i = 0; i < n_tensors; i++) {
        TensorInfo ti;
        uint32_t name_len = read_u32();
        ti.name = std::string(reinterpret_cast<const char*>(base + offset), name_len);
        offset += name_len;
        ti.dtype = read_u32();
        uint32_t ndims = read_u32();
        ti.shape.resize(ndims);
        for (uint32_t d = 0; d < ndims; d++)
            ti.shape[d] = read_u64();
        ti.scale_mode = read_u32();
        ti.n_scales = read_u32();
        ti.scale_offset = read_u64();
        ti.data_offset = read_u64();
        ti.data_size = read_u64();
        tensor_infos.push_back(std::move(ti));
    }

    // Build WeightRef from TensorInfo, pointing into the file buffer
    auto make_weight = [&](const TensorInfo& ti) -> WeightRef {
        WeightRef w;
        w.host_data = base + ti.data_offset;
        w.size_bytes = ti.data_size;
        w.shape = ti.shape;

        // Compute n_elements from shape
        w.n_elements = 1;
        for (auto s : ti.shape) w.n_elements *= s;

        if (ti.dtype == 0) {  // FP8_E4M3
            w.type = GGMLType::FP8_E4M3;
            w.host_scales = reinterpret_cast<const float*>(base + ti.scale_offset);
            w.n_scale_rows = ti.n_scales;
        } else if (ti.dtype == 1) {  // F32
            w.type = GGMLType::F32;
        } else if (ti.dtype == 2) {  // F16
            w.type = GGMLType::F16;
        }
        return w;
    };

    // Build name lookup
    std::unordered_map<std::string, size_t> name_to_idx;
    for (size_t i = 0; i < tensor_infos.size(); i++)
        name_to_idx[tensor_infos[i].name] = i;

    auto get = [&](const std::string& name) -> WeightRef {
        auto it = name_to_idx.find(name);
        GWEN_CHECK(it != name_to_idx.end(), ("Missing tensor: " + name).c_str());
        return make_weight(tensor_infos[it->second]);
    };

    auto try_get = [&](const std::string& name) -> WeightRef {
        auto it = name_to_idx.find(name);
        if (it == name_to_idx.end()) return {};
        return make_weight(tensor_infos[it->second]);
    };

    // Set config from tensor shapes (infer from the weight dimensions)
    auto& cfg = model->config;
    {
        auto& te = tensor_infos[name_to_idx.at("token_embd.weight")];
        cfg.n_vocab = te.shape[0];
        cfg.n_embed = te.shape[1];
    }

    // Count layers by finding the highest blk.N index
    uint32_t max_layer = 0;
    for (auto& [name, idx] : name_to_idx) {
        if (name.substr(0, 4) == "blk.") {
            auto dot = name.find('.', 4);
            if (dot != std::string::npos) {
                uint32_t li = std::stoi(name.substr(4, dot - 4));
                if (li > max_layer) max_layer = li;
            }
        }
    }
    cfg.n_layers = max_layer + 1;

    // Infer other params from tensor shapes
    {
        auto& qkv = tensor_infos[name_to_idx.at("blk.0.attn_qkv.weight")];
        uint32_t qkv_out = qkv.shape[0];  // 6144 for 0.8B
        // QKV = 2*k_heads*dk + v_heads*dv. For symmetric (0.8B): 3*ssm_inner.
        // For asymmetric (4B+): we need to figure out k vs v heads.
        // Try symmetric first (qkv_out divisible by 3), else infer from ssm_inner_size.
        cfg.ssm_state_size = 128;  // always 128 for Qwen3.5
        if (qkv_out % 3 == 0) {
            // Symmetric: k_heads == v_heads
            cfg.ssm_inner_size = qkv_out / 3;
            cfg.ssm_n_v_heads = cfg.ssm_inner_size / cfg.ssm_state_size;
            cfg.ssm_n_k_heads = cfg.ssm_n_v_heads;
        } else {
            // Asymmetric: infer from gate tensor (gate_z output = ssm_inner_size = v_heads*dv)
            auto& gate = tensor_infos[name_to_idx.at("blk.0.attn_gate.weight")];
            cfg.ssm_inner_size = gate.shape[0];
            cfg.ssm_n_v_heads = cfg.ssm_inner_size / cfg.ssm_state_size;
            // k_heads*dk*2 + v_heads*dv = qkv_out → k_heads = (qkv_out - v_heads*dv) / (2*dk)
            cfg.ssm_n_k_heads = (qkv_out - cfg.ssm_n_v_heads * cfg.ssm_state_size) /
                                (2 * cfg.ssm_state_size);
        }
        cfg.ssm_n_heads = cfg.ssm_n_v_heads;
    }
    {
        auto& ffn = tensor_infos[name_to_idx.at("blk.0.ffn_gate.weight")];
        cfg.n_ff = ffn.shape[0];
    }
    // Full attention params from the first full attention layer (layer 3)
    {
        std::string fa_prefix = "blk." + std::to_string(cfg.full_attn_interval - 1);
        auto it = name_to_idx.find(fa_prefix + ".attn_q.weight");
        if (it != name_to_idx.end()) {
            auto& q = tensor_infos[it->second];
            // Q shape: [n_head * (head_dim + head_dim), n_embed] = [n_head * 2 * head_dim, n_embed]
            // But actually Q is [n_head * head_dim * 2, n_embed] for Qwen3.5 (Q + gate interleaved)
            uint32_t q_out = q.shape[0];
            cfg.n_head = q_out / (cfg.head_dim * 2);  // divide by 2 for Q+gate
        }
        auto it_k = name_to_idx.find(fa_prefix + ".attn_k.weight");
        if (it_k != name_to_idx.end()) {
            auto& k = tensor_infos[it_k->second];
            cfg.n_head_kv = k.shape[0] / cfg.head_dim;
        }
    }

    fprintf(stderr, "GWFP8: %u layers, embed=%u, ff=%u, vocab=%u, heads=%u/%u, ssm_k=%u/v=%u\n",
            cfg.n_layers, cfg.n_embed, cfg.n_ff, cfg.n_vocab, cfg.n_head, cfg.n_head_kv,
            cfg.ssm_n_k_heads, cfg.ssm_n_v_heads);

    // Load global weights
    model->token_embd = get("token_embd.weight");
    model->output_norm = get("output_norm.weight");

    // Load per-layer weights
    model->layers.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        auto& layer = model->layers[i];
        std::string prefix = "blk." + std::to_string(i) + ".";
        layer.is_full_attention = cfg.is_full_attention_layer(i);

        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            w.attn_norm      = get(prefix + "attn_norm.weight");
            w.attn_q         = get(prefix + "attn_q.weight");
            w.attn_k         = get(prefix + "attn_k.weight");
            w.attn_v         = get(prefix + "attn_v.weight");
            w.attn_q_norm    = get(prefix + "attn_q_norm.weight");
            w.attn_k_norm    = get(prefix + "attn_k_norm.weight");
            w.attn_output    = get(prefix + "attn_output.weight");
            w.post_attn_norm = get(prefix + "post_attention_norm.weight");
            w.ffn_gate       = get(prefix + "ffn_gate.weight");
            w.ffn_up         = get(prefix + "ffn_up.weight");
            w.ffn_down       = get(prefix + "ffn_down.weight");
        } else {
            auto& w = layer.deltanet;
            w.attn_norm      = get(prefix + "attn_norm.weight");
            w.attn_qkv       = get(prefix + "attn_qkv.weight");
            w.attn_gate      = get(prefix + "attn_gate.weight");
            w.ssm_conv1d     = get(prefix + "ssm_conv1d.weight");
            w.ssm_a          = get(prefix + "ssm_a");
            w.ssm_dt_bias    = get(prefix + "ssm_dt.bias");
            w.ssm_alpha      = get(prefix + "ssm_alpha.weight");
            w.ssm_beta       = get(prefix + "ssm_beta.weight");
            w.ssm_norm       = get(prefix + "ssm_norm.weight");
            w.ssm_out        = get(prefix + "ssm_out.weight");
            w.post_attn_norm = get(prefix + "post_attention_norm.weight");
            w.ffn_gate       = get(prefix + "ffn_gate.weight");
            w.ffn_up         = get(prefix + "ffn_up.weight");
            w.ffn_down       = get(prefix + "ffn_down.weight");
        }
    }

    // Count total weight bytes
    size_t total = model->token_embd.size_bytes + model->output_norm.size_bytes;
    for (auto& layer : model->layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            total += w.attn_norm.size_bytes + w.attn_q.size_bytes + w.attn_k.size_bytes +
                     w.attn_v.size_bytes + w.attn_q_norm.size_bytes + w.attn_k_norm.size_bytes +
                     w.attn_output.size_bytes + w.post_attn_norm.size_bytes +
                     w.ffn_gate.size_bytes + w.ffn_up.size_bytes + w.ffn_down.size_bytes;
        } else {
            auto& w = layer.deltanet;
            total += w.attn_norm.size_bytes + w.attn_qkv.size_bytes + w.attn_gate.size_bytes +
                     w.ssm_conv1d.size_bytes + w.ssm_a.size_bytes + w.ssm_dt_bias.size_bytes +
                     w.ssm_alpha.size_bytes + w.ssm_beta.size_bytes + w.ssm_norm.size_bytes +
                     w.ssm_out.size_bytes + w.post_attn_norm.size_bytes +
                     w.ffn_gate.size_bytes + w.ffn_up.size_bytes + w.ffn_down.size_bytes;
        }
    }
    fprintf(stderr, "GWFP8: loaded %.1f MB of weights from %s\n", total / 1024.0 / 1024.0, fp8_path.c_str());

    return model;
}

// ============================================================
// GWFP4 loader — FP4 E2M1 weights with E4M3 block scales + F32 global scale
// ============================================================

std::unique_ptr<Model> Model::load_fp4(const std::string& fp4_path) {
    auto model = std::make_unique<Model>();

    // Read entire file into memory
    std::ifstream f(fp4_path, std::ios::binary | std::ios::ate);
    GWEN_CHECK(f.is_open(), ("Failed to open: " + fp4_path).c_str());
    size_t file_size = f.tellg();
    f.seekg(0);
    model->gwfp4_file_data_.resize(file_size);
    f.read(reinterpret_cast<char*>(model->gwfp4_file_data_.data()), file_size);
    f.close();

    const uint8_t* base = model->gwfp4_file_data_.data();
    size_t offset = 0;

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v; memcpy(&v, base + offset, 4); offset += 4; return v;
    };
    auto read_u64 = [&]() -> uint64_t {
        uint64_t v; memcpy(&v, base + offset, 8); offset += 8; return v;
    };
    auto read_f32 = [&]() -> float {
        float v; memcpy(&v, base + offset, 4); offset += 4; return v;
    };

    // Parse header
    GWEN_CHECK(memcmp(base, "GWF4", 4) == 0, "Invalid GWFP4 magic");
    offset = 4;
    uint32_t version = read_u32();
    GWEN_CHECK(version == 1, "Unsupported GWFP4 version");
    uint32_t n_tensors = read_u32();
    uint32_t header_size = read_u32();
    (void)header_size;

    // Read embedded config JSON
    uint32_t config_len = read_u32();
    std::string config_json(reinterpret_cast<const char*>(base + offset), config_len);
    offset += config_len;

    // Parse config JSON (minimal parser — just extract known keys)
    auto& cfg = model->config;
    auto json_int = [&](const std::string& json, const std::string& key) -> uint32_t {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return 0;
        return std::stoul(json.substr(pos + 1));
    };
    auto json_float = [&](const std::string& json, const std::string& key) -> float {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0f;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return 0.0f;
        return std::stof(json.substr(pos + 1));
    };
    auto json_bool = [&](const std::string& json, const std::string& key) -> bool {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return false;
        auto val = json.substr(pos + 1, 10);
        return val.find("true") != std::string::npos;
    };

    cfg.n_embed = json_int(config_json, "hidden_size");
    cfg.n_ff = json_int(config_json, "intermediate_size");
    cfg.n_layers = json_int(config_json, "num_hidden_layers");
    cfg.n_head = json_int(config_json, "num_attention_heads");
    cfg.n_head_kv = json_int(config_json, "num_key_value_heads");
    cfg.n_vocab = json_int(config_json, "vocab_size");
    cfg.head_dim = json_int(config_json, "head_dim");
    if (cfg.head_dim == 0) cfg.head_dim = 256;  // default for Qwen3.5
    cfg.ssm_state_size = json_int(config_json, "linear_key_head_dim");
    if (cfg.ssm_state_size == 0) cfg.ssm_state_size = 128;
    cfg.ssm_n_k_heads = json_int(config_json, "linear_num_key_heads");
    if (cfg.ssm_n_k_heads == 0) cfg.ssm_n_k_heads = 16;
    cfg.ssm_n_v_heads = json_int(config_json, "linear_num_value_heads");
    if (cfg.ssm_n_v_heads == 0) cfg.ssm_n_v_heads = cfg.ssm_n_k_heads;
    cfg.ssm_inner_size = cfg.ssm_n_v_heads * cfg.ssm_state_size;
    cfg.ssm_n_heads = cfg.ssm_n_v_heads;
    cfg.ssm_conv_kernel = json_int(config_json, "linear_conv_kernel_dim");
    if (cfg.ssm_conv_kernel == 0) cfg.ssm_conv_kernel = 4;
    cfg.full_attn_interval = json_int(config_json, "full_attention_interval");
    if (cfg.full_attn_interval == 0) cfg.full_attn_interval = 4;
    cfg.rms_norm_eps = json_float(config_json, "rms_norm_eps");
    if (cfg.rms_norm_eps == 0.0f) cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = json_float(config_json, "rope_theta");
    if (cfg.rope_theta == 0.0f) cfg.rope_theta = 10000000.0f;
    bool tie_embeddings = json_bool(config_json, "tie_word_embeddings");

    fprintf(stderr, "GWFP4: %u layers, embed=%u, ff=%u, vocab=%u, heads=%u/%u, ssm_k=%u/v=%u\n",
            cfg.n_layers, cfg.n_embed, cfg.n_ff, cfg.n_vocab, cfg.n_head, cfg.n_head_kv,
            cfg.ssm_n_k_heads, cfg.ssm_n_v_heads);

    // Parse tensor headers
    struct FP4TensorInfo {
        std::string name;
        uint32_t dtype;          // 0=FP4_E2M1, 1=F32, 2=BF16
        std::vector<uint64_t> shape;
        bool has_scales;
        float scale2;            // global scale
        std::vector<uint64_t> scales_shape;
        uint64_t data_offset, data_size;
        uint64_t scales_offset, scales_size;
    };
    std::vector<FP4TensorInfo> tensor_infos;
    tensor_infos.reserve(n_tensors);

    for (uint32_t i = 0; i < n_tensors; i++) {
        FP4TensorInfo ti;
        uint32_t name_len = read_u32();
        ti.name = std::string(reinterpret_cast<const char*>(base + offset), name_len);
        offset += name_len;
        ti.dtype = read_u32();
        uint32_t ndims = read_u32();
        ti.shape.resize(ndims);
        for (uint32_t d = 0; d < ndims; d++)
            ti.shape[d] = read_u64();
        uint32_t has_scales = read_u32();
        ti.has_scales = (has_scales != 0);
        ti.scale2 = read_f32();
        if (ti.has_scales) {
            uint32_t s_ndims = read_u32();
            ti.scales_shape.resize(s_ndims);
            for (uint32_t d = 0; d < s_ndims; d++)
                ti.scales_shape[d] = read_u64();
        }
        ti.data_offset = read_u64();
        ti.data_size = read_u64();
        ti.scales_offset = read_u64();
        ti.scales_size = read_u64();
        tensor_infos.push_back(std::move(ti));
    }

    // Build name → index map
    std::unordered_map<std::string, size_t> name_to_idx;
    for (size_t i = 0; i < tensor_infos.size(); i++)
        name_to_idx[tensor_infos[i].name] = i;

    auto make_weight = [&](const FP4TensorInfo& ti) -> WeightRef {
        WeightRef w;
        w.host_data = base + ti.data_offset;
        w.size_bytes = ti.data_size;
        w.shape = ti.shape;
        w.n_elements = 1;
        for (auto s : ti.shape) w.n_elements *= s;

        if (ti.dtype == 0) {  // FP4_E2M1
            w.type = GGMLType::FP4_E2M1;
            w.fp4_global_scale = ti.scale2;
            w.scales_fp4_bytes = ti.scales_size;
        } else if (ti.dtype == 1) {  // F32
            w.type = GGMLType::F32;
        } else if (ti.dtype == 2) {  // BF16
            w.type = GGMLType::BF16;
        } else if (ti.dtype == 3) {  // F16
            w.type = GGMLType::F16;
        } else if (ti.dtype == 4) {  // FP8_E4M3 with per-row F32 scales
            w.type = GGMLType::FP8_E4M3;
        }
        return w;
    };

    // Store per-tensor scales host pointers (FP4 E4M3 block scales)
    // We need these to survive until upload — store offsets relative to file base
    struct FP4ScaleInfo {
        const uint8_t* host_ptr;
        size_t size;
    };
    std::unordered_map<std::string, FP4ScaleInfo> fp4_scales;

    auto get = [&](const std::string& name) -> WeightRef {
        auto it = name_to_idx.find(name);
        GWEN_CHECK(it != name_to_idx.end(), ("Missing tensor: " + name).c_str());
        auto& ti = tensor_infos[it->second];
        auto w = make_weight(ti);
        if (ti.has_scales && ti.scales_size > 0) {
            fp4_scales[name] = {base + ti.scales_offset, ti.scales_size};
        }
        return w;
    };

    auto try_get = [&](const std::string& name) -> WeightRef {
        auto it = name_to_idx.find(name);
        if (it == name_to_idx.end()) return {};
        auto& ti = tensor_infos[it->second];
        auto w = make_weight(ti);
        if (ti.has_scales && ti.scales_size > 0) {
            fp4_scales[name] = {base + ti.scales_offset, ti.scales_size};
        }
        return w;
    };

    // Load global weights
    model->token_embd = get("token_embd.weight");
    model->output_norm = get("output_norm.weight");

    // Load per-layer weights
    model->layers.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        auto& layer = model->layers[i];
        std::string prefix = "blk." + std::to_string(i) + ".";
        layer.is_full_attention = cfg.is_full_attention_layer(i);

        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            w.attn_norm      = get(prefix + "attn_norm.weight");
            w.attn_q         = get(prefix + "attn_q.weight");
            w.attn_k         = get(prefix + "attn_k.weight");
            w.attn_v         = get(prefix + "attn_v.weight");
            w.attn_q_norm    = get(prefix + "attn_q_norm.weight");
            w.attn_k_norm    = get(prefix + "attn_k_norm.weight");
            w.attn_output    = get(prefix + "attn_output.weight");
            w.post_attn_norm = get(prefix + "post_attn_norm.weight");
            w.ffn_gate       = get(prefix + "ffn_gate.weight");
            w.ffn_up         = get(prefix + "ffn_up.weight");
            w.ffn_down       = get(prefix + "ffn_down.weight");
        } else {
            auto& w = layer.deltanet;
            w.attn_norm      = get(prefix + "attn_norm.weight");
            w.attn_qkv       = get(prefix + "attn_qkv.weight");
            w.attn_gate      = get(prefix + "attn_gate.weight");
            w.ssm_conv1d     = get(prefix + "ssm_conv1d.weight");
            w.ssm_a          = get(prefix + "ssm_a.weight");
            w.ssm_dt_bias    = try_get(prefix + "ssm_dt_bias.weight");
            w.ssm_alpha      = get(prefix + "ssm_alpha.weight");
            w.ssm_beta       = get(prefix + "ssm_beta.weight");
            w.ssm_norm       = get(prefix + "ssm_norm.weight");
            w.ssm_out        = get(prefix + "ssm_out.weight");
            w.post_attn_norm = get(prefix + "post_attn_norm.weight");
            w.ffn_gate       = get(prefix + "ffn_gate.weight");
            w.ffn_up         = get(prefix + "ffn_up.weight");
            w.ffn_down       = get(prefix + "ffn_down.weight");
        }
    }

    // Store FP4 scales info on the model for upload_weights to use
    // We store this as a map attached to the model via a lambda capture in upload
    // Actually, we need to set the host pointer for scales on each WeightRef.
    // Walk all weights and set their scales host pointers.
    auto set_scales = [&](WeightRef& w, const std::string& name) {
        auto it = fp4_scales.find(name);
        if (it == fp4_scales.end()) return;
        if (w.type == GGMLType::FP4_E2M1) {
            // Store the host pointer to E4M3 scales in host_scales (recast)
            w.host_scales = reinterpret_cast<const float*>(it->second.host_ptr);
            w.n_scale_rows = it->second.size;  // abuse this field for byte count
        } else if (w.type == GGMLType::FP8_E4M3) {
            // Per-row F32 scales
            w.host_scales = reinterpret_cast<const float*>(it->second.host_ptr);
            w.n_scale_rows = it->second.size / sizeof(float);
        }
    };

    // Set scales on all FP4 tensors
    set_scales(model->token_embd, "token_embd.weight");
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        auto& layer = model->layers[i];
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            set_scales(w.attn_q, prefix + "attn_q.weight");
            set_scales(w.attn_k, prefix + "attn_k.weight");
            set_scales(w.attn_v, prefix + "attn_v.weight");
            set_scales(w.attn_output, prefix + "attn_output.weight");
            set_scales(w.ffn_gate, prefix + "ffn_gate.weight");
            set_scales(w.ffn_up, prefix + "ffn_up.weight");
            set_scales(w.ffn_down, prefix + "ffn_down.weight");
        } else {
            auto& w = layer.deltanet;
            set_scales(w.attn_qkv, prefix + "attn_qkv.weight");
            set_scales(w.attn_gate, prefix + "attn_gate.weight");
            set_scales(w.ssm_alpha, prefix + "ssm_alpha.weight");
            set_scales(w.ssm_beta, prefix + "ssm_beta.weight");
            set_scales(w.ssm_out, prefix + "ssm_out.weight");
            set_scales(w.ffn_gate, prefix + "ffn_gate.weight");
            set_scales(w.ffn_up, prefix + "ffn_up.weight");
            set_scales(w.ffn_down, prefix + "ffn_down.weight");
        }
    }

    // Count total weight bytes
    size_t total = model->token_embd.size_bytes + model->output_norm.size_bytes;
    for (auto& layer : model->layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            total += w.attn_norm.size_bytes + w.attn_q.size_bytes + w.attn_k.size_bytes +
                     w.attn_v.size_bytes + w.attn_q_norm.size_bytes + w.attn_k_norm.size_bytes +
                     w.attn_output.size_bytes + w.post_attn_norm.size_bytes +
                     w.ffn_gate.size_bytes + w.ffn_up.size_bytes + w.ffn_down.size_bytes;
        } else {
            auto& w = layer.deltanet;
            total += w.attn_norm.size_bytes + w.attn_qkv.size_bytes + w.attn_gate.size_bytes +
                     w.ssm_conv1d.size_bytes + w.ssm_a.size_bytes + w.ssm_dt_bias.size_bytes +
                     w.ssm_alpha.size_bytes + w.ssm_beta.size_bytes + w.ssm_norm.size_bytes +
                     w.ssm_out.size_bytes + w.post_attn_norm.size_bytes +
                     w.ffn_gate.size_bytes + w.ffn_up.size_bytes + w.ffn_down.size_bytes;
        }
    }
    fprintf(stderr, "GWFP4: loaded %.1f MB of weights from %s\n", total / 1024.0 / 1024.0, fp4_path.c_str());
    (void)tie_embeddings;

    return model;
}

// Upload all weight tensors to GPU
// Forward declarations for dequant kernels
void gwen_dequant(const void* src, half* dst, int n, GGMLType type, cudaStream_t stream);

static void upload_weight(CudaAllocator& alloc, WeightRef& w) {
    if (w.host_data && w.size_bytes > 0 && !w.on_device()) {
        w.device_data = alloc.upload(w.host_data, w.size_bytes);
    }
    // Upload FP8 per-row scales if present
    if (w.type == GGMLType::FP8_E4M3 && w.host_scales && w.n_scale_rows > 0 && !w.device_scales) {
        w.device_scales = static_cast<float*>(
            alloc.upload(w.host_scales, w.n_scale_rows * sizeof(float)));
    }
    // Upload FP4 E4M3 block scales if present
    // For FP4: host_scales points to raw E4M3 bytes, n_scale_rows stores byte count
    if (w.type == GGMLType::FP4_E2M1 && w.host_scales && w.n_scale_rows > 0 && !w.device_scales_fp4) {
        w.device_scales_fp4 = alloc.upload(w.host_scales, w.n_scale_rows);
        w.scales_fp4_bytes = w.n_scale_rows;
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

    // Precompute CUTLASS SFA scale arrays for FP8 weight tensors
    // Replicates per-row scales across K-blocks: sfa[row * n_k_blocks + kb] = scale[row]
    auto prepare_sfa = [&](WeightRef& w) {
        if (w.type != GGMLType::FP8_E4M3 || !w.device_scales || w.shape.size() < 2) return;
        int M = (int)w.shape[0];      // out_features (rows)
        int K = (int)w.shape[1];      // in_features (cols)
        int n_k_blocks = (K + 127) / 128;
        w.sfa_n_k_blocks = n_k_blocks;
        size_t sfa_bytes = (size_t)M * n_k_blocks * sizeof(float);
        w.device_sfa = static_cast<float*>(allocator.alloc(sfa_bytes));
        gwen_replicate_fp8_scales(w.device_scales, w.device_sfa, M, n_k_blocks);
    };

    for (auto& layer : layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            prepare_sfa(w.attn_q);
            prepare_sfa(w.attn_k);
            prepare_sfa(w.attn_v);
            prepare_sfa(w.attn_output);
            prepare_sfa(w.ffn_gate);
            prepare_sfa(w.ffn_up);
            prepare_sfa(w.ffn_down);
        } else {
            auto& w = layer.deltanet;
            prepare_sfa(w.attn_qkv);
            prepare_sfa(w.attn_gate);
            prepare_sfa(w.ssm_out);
            prepare_sfa(w.ffn_gate);
            prepare_sfa(w.ffn_up);
            prepare_sfa(w.ffn_down);
        }
    }
    prepare_sfa(token_embd);
    if (!config.tie_word_embeddings) {
        prepare_sfa(output_weight);
    }
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    // Upload reduced LM head if loaded
    if (has_reduced_lm_head) {
        upload_weight(allocator, reduced_lm_head.weights);
        // Upload token ID mapping to device
        size_t ids_bytes = reduced_lm_head.K * sizeof(int32_t);
        reduced_lm_head.d_token_ids = static_cast<int*>(allocator.upload(
            reduced_lm_head.token_ids.data(), ids_bytes));
        GWEN_LOG("Reduced LM head uploaded: %.1f MB weights + %.1f KB token map\n",
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
    const char* fmt = gguf ? gguf->path().c_str() :
                       (!gwfp4_file_data_.empty() ? "(GWFP4)" : "(GWFP8)");
    fprintf(stderr, "Model: %s\n", fmt);
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

    // Print layer pattern
    fprintf(stderr, "\nLayer pattern: ");
    for (uint32_t i = 0; i < config.n_layers; i++) {
        fprintf(stderr, "%c", config.is_full_attention_layer(i) ? 'A' : 'D');
    }
    fprintf(stderr, "\n");

    // Weight summary
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
