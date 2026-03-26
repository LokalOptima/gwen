#include "gwen/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gwen {

// --- GGMLType helpers ---

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0";
        case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::Q8_K: return "Q8_K";
        case GGMLType::IQ4_XS: return "IQ4_XS";
        case GGMLType::BF16: return "BF16";
        case GGMLType::FP8_E4M3: return "FP8_E4M3";
        case GGMLType::FP4_E2M1: return "FP4_E2M1";
        default: return "UNKNOWN";
    }
}

// Block size = number of elements per quantization block
uint32_t ggml_type_block_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 1;
        case GGMLType::F16:  return 1;
        case GGMLType::BF16: return 1;
        case GGMLType::FP8_E4M3: return 1;
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q4_1: return 32;
        case GGMLType::Q5_0: return 32;
        case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q8_1: return 32;
        case GGMLType::Q2_K: return 256;
        case GGMLType::Q3_K: return 256;
        case GGMLType::Q4_K: return 256;
        case GGMLType::Q5_K: return 256;
        case GGMLType::Q6_K: return 256;
        case GGMLType::Q8_K: return 256;
        case GGMLType::IQ4_XS: return 256;
        default: return 0;
    }
}

// Bytes per quantization block
size_t ggml_type_block_bytes(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::BF16: return 2;
        case GGMLType::FP8_E4M3: return 1;  // 1 byte per element
        case GGMLType::Q4_0: return 18;   // 2 (d) + 16 (qs)
        case GGMLType::Q4_1: return 20;   // 2 (d) + 2 (m) + 16 (qs)
        case GGMLType::Q5_0: return 22;   // 2 (d) + 4 (qh) + 16 (qs)
        case GGMLType::Q5_1: return 24;   // 2 (d) + 2 (m) + 4 (qh) + 16 (qs)
        case GGMLType::Q8_0: return 34;   // 2 (d) + 32 (qs)
        case GGMLType::Q8_1: return 40;   // 4 (d) + 4 (s) + 32 (qs)
        case GGMLType::Q2_K: return 84;
        case GGMLType::Q3_K: return 110;
        case GGMLType::Q4_K: return 144;  // 2(d) + 2(dmin) + 12(scales) + 128(qs)
        case GGMLType::Q5_K: return 176;  // 2(d) + 2(dmin) + 12(scales) + 32(qh) + 128(qs)
        case GGMLType::Q6_K: return 210;  // 128(ql) + 64(qh) + 16(scales) + 2(d)
        case GGMLType::Q8_K: return 292;
        case GGMLType::IQ4_XS: return 136;  // 2(d) + 2(scales_h) + 4(scales_l) + 128(qs)
        default: return 0;
    }
}

size_t ggml_type_size(GGMLType type, size_t n_elements) {
    uint32_t bs = ggml_type_block_size(type);
    size_t bb = ggml_type_block_bytes(type);
    if (bs == 0 || bb == 0) return 0;
    size_t n_blocks = (n_elements + bs - 1) / bs;
    return n_blocks * bb;
}

// --- GGUF reader helpers ---

struct Reader {
    const uint8_t* data;
    size_t pos;
    size_t size;

    template<typename T>
    T read() {
        GWEN_CHECK(pos + sizeof(T) <= size, "GGUF read past end of file");
        T val;
        memcpy(&val, data + pos, sizeof(T));
        pos += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        GWEN_CHECK(pos + len <= size, "GGUF string read past end of file");
        std::string s(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return s;
    }

    GGUFValue read_value(GGUFValueType vtype) {
        switch (vtype) {
            case GGUFValueType::UINT8:   return (uint32_t)read<uint8_t>();
            case GGUFValueType::INT8:    return (int32_t)read<int8_t>();
            case GGUFValueType::UINT16:  return (uint32_t)read<uint16_t>();
            case GGUFValueType::INT16:   return (int32_t)read<int16_t>();
            case GGUFValueType::UINT32:  return read<uint32_t>();
            case GGUFValueType::INT32:   return read<int32_t>();
            case GGUFValueType::FLOAT32: return read<float>();
            case GGUFValueType::BOOL:    return (bool)(read<uint8_t>() != 0);
            case GGUFValueType::STRING:  return read_string();
            case GGUFValueType::UINT64:  return read<uint64_t>();
            case GGUFValueType::INT64:   return read<int64_t>();
            case GGUFValueType::FLOAT64: return read<double>();
            case GGUFValueType::ARRAY: {
                auto atype = static_cast<GGUFValueType>(read<uint32_t>());
                uint64_t alen = read<uint64_t>();

                // Return typed arrays for common types
                if (atype == GGUFValueType::INT32) {
                    std::vector<int32_t> arr(alen);
                    for (uint64_t i = 0; i < alen; i++)
                        arr[i] = read<int32_t>();
                    return arr;
                }
                if (atype == GGUFValueType::FLOAT32) {
                    std::vector<float> arr(alen);
                    for (uint64_t i = 0; i < alen; i++)
                        arr[i] = read<float>();
                    return arr;
                }
                if (atype == GGUFValueType::STRING) {
                    std::vector<std::string> arr(alen);
                    for (uint64_t i = 0; i < alen; i++)
                        arr[i] = read_string();
                    return arr;
                }
                // For other array types, skip over them
                for (uint64_t i = 0; i < alen; i++)
                    read_value(atype);
                return std::string("<array>");
            }
            default:
                throw std::runtime_error("Unknown GGUF value type: " +
                                         std::to_string((uint32_t)vtype));
        }
    }
};

// --- GGUFFile implementation ---

GGUFFile::~GGUFFile() {
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

std::unique_ptr<GGUFFile> GGUFFile::open(const std::string& path) {
    auto gguf = std::unique_ptr<GGUFFile>(new GGUFFile());
    gguf->path_ = path;

    // Open file
    gguf->fd_ = ::open(path.c_str(), O_RDONLY);
    GWEN_CHECK(gguf->fd_ >= 0, ("Failed to open: " + path).c_str());

    // Get file size
    struct stat st;
    GWEN_CHECK(fstat(gguf->fd_, &st) == 0, "fstat failed");
    gguf->mmap_size_ = st.st_size;

    // Memory map
    gguf->mmap_addr_ = mmap(nullptr, gguf->mmap_size_, PROT_READ, MAP_PRIVATE, gguf->fd_, 0);
    GWEN_CHECK(gguf->mmap_addr_ != MAP_FAILED, "mmap failed");

    // Advise sequential access for header, then random for tensor data
    madvise(gguf->mmap_addr_, gguf->mmap_size_, MADV_SEQUENTIAL);

    Reader r{static_cast<const uint8_t*>(gguf->mmap_addr_), 0, gguf->mmap_size_};

    // Parse header
    uint32_t magic = r.read<uint32_t>();
    GWEN_CHECK(magic == GGUF_MAGIC, "Not a GGUF file");

    uint32_t version = r.read<uint32_t>();
    GWEN_CHECK(version == GGUF_VERSION, "Unsupported GGUF version");

    uint64_t n_tensors = r.read<uint64_t>();
    uint64_t n_kv = r.read<uint64_t>();

    // Parse metadata KV pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = r.read_string();
        auto vtype = static_cast<GGUFValueType>(r.read<uint32_t>());
        GGUFValue val = r.read_value(vtype);
        gguf->metadata_[key] = std::move(val);
    }

    // Parse tensor info
    gguf->tensors_.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        auto& t = gguf->tensors_[i];
        t.name = r.read_string();
        uint32_t n_dims = r.read<uint32_t>();
        t.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            t.shape[d] = r.read<uint64_t>();
        }
        t.type = static_cast<GGMLType>(r.read<uint32_t>());
        t.offset = r.read<uint64_t>();

        t.n_elements = 1;
        for (auto dim : t.shape) t.n_elements *= dim;
        t.size_bytes = ggml_type_size(t.type, t.n_elements);
    }

    // Tensor data starts at next alignment boundary after header
    // GGUF uses 32-byte alignment
    constexpr size_t GGUF_ALIGNMENT = 32;
    size_t data_start = (r.pos + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1);

    // Set data pointers and build index
    const uint8_t* base = static_cast<const uint8_t*>(gguf->mmap_addr_);
    for (size_t i = 0; i < gguf->tensors_.size(); i++) {
        auto& t = gguf->tensors_[i];
        t.data = base + data_start + t.offset;
        gguf->tensor_index_[t.name] = i;
    }

    // Switch to random access for tensor data
    madvise(static_cast<uint8_t*>(gguf->mmap_addr_) + data_start,
            gguf->mmap_size_ - data_start, MADV_RANDOM);

    return gguf;
}

// --- Metadata accessors ---

uint32_t GGUFFile::get_u32(const std::string& key) const {
    auto it = metadata_.find(key);
    GWEN_CHECK(it != metadata_.end(), ("Missing GGUF key: " + key).c_str());
    return std::get<uint32_t>(it->second);
}

float GGUFFile::get_f32(const std::string& key) const {
    auto it = metadata_.find(key);
    GWEN_CHECK(it != metadata_.end(), ("Missing GGUF key: " + key).c_str());
    return std::get<float>(it->second);
}

std::string GGUFFile::get_str(const std::string& key) const {
    auto it = metadata_.find(key);
    GWEN_CHECK(it != metadata_.end(), ("Missing GGUF key: " + key).c_str());
    return std::get<std::string>(it->second);
}

std::vector<int32_t> GGUFFile::get_i32_array(const std::string& key) const {
    auto it = metadata_.find(key);
    GWEN_CHECK(it != metadata_.end(), ("Missing GGUF key: " + key).c_str());
    return std::get<std::vector<int32_t>>(it->second);
}

uint32_t GGUFFile::get_u32(const std::string& key, uint32_t default_val) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return default_val;
    return std::get<uint32_t>(it->second);
}

float GGUFFile::get_f32(const std::string& key, float default_val) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return default_val;
    return std::get<float>(it->second);
}

// --- Config builder ---

ModelConfig GGUFFile::build_config() const {
    ModelConfig cfg;
    cfg.n_layers        = get_u32("qwen35.block_count");
    cfg.n_embed         = get_u32("qwen35.embedding_length");
    cfg.n_ff            = get_u32("qwen35.feed_forward_length");
    cfg.n_head          = get_u32("qwen35.attention.head_count");
    cfg.n_head_kv       = get_u32("qwen35.attention.head_count_kv");
    cfg.head_dim        = get_u32("qwen35.attention.key_length");
    cfg.rope_theta      = get_f32("qwen35.rope.freq_base");
    cfg.rms_norm_eps    = get_f32("qwen35.attention.layer_norm_rms_epsilon");
    cfg.ssm_conv_kernel = get_u32("qwen35.ssm.conv_kernel");
    cfg.ssm_state_size  = get_u32("qwen35.ssm.state_size");
    cfg.ssm_n_heads     = get_u32("qwen35.ssm.group_count");
    cfg.ssm_n_k_heads   = cfg.ssm_n_heads;  // GGUF group_count = K heads
    cfg.ssm_n_v_heads   = cfg.ssm_n_heads;  // Default: symmetric (0.8B). Overridden below for 4B+.
    cfg.ssm_inner_size  = get_u32("qwen35.ssm.inner_size");
    // Derive n_v_heads: ssm_inner_size = n_v_heads * state_size
    cfg.ssm_n_v_heads   = cfg.ssm_inner_size / cfg.ssm_state_size;
    cfg.ssm_n_heads     = cfg.ssm_n_v_heads;  // legacy alias
    cfg.full_attn_interval = get_u32("qwen35.full_attention_interval");
    cfg.rope_dim        = get_u32("qwen35.rope.dimension_count", 64);
    cfg.context_length  = get_u32("qwen35.context_length", 262144);
    cfg.eos_token_id    = get_u32("tokenizer.ggml.eos_token_id", 248046);
    cfg.eot_token_id    = get_u32("tokenizer.ggml.eot_token_id", 248044);
    cfg.pad_token_id    = get_u32("tokenizer.ggml.padding_token_id", 248055);

    // Vocab size from embedding tensor
    auto* embd = find_tensor("token_embd.weight");
    if (embd && embd->shape.size() == 2) {
        cfg.n_vocab = embd->shape[1];
    }

    // RoPE sections
    auto sections = get_i32_array("qwen35.rope.dimension_sections");
    for (int i = 0; i < 4 && i < (int)sections.size(); i++) {
        cfg.rope_sections[i] = sections[i];
    }

    return cfg;
}

// --- Tensor access ---

const GGUFTensor* GGUFFile::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

const GGUFTensor& GGUFFile::get_tensor(const std::string& name) const {
    auto* t = find_tensor(name);
    GWEN_CHECK(t != nullptr, ("Missing tensor: " + name).c_str());
    return *t;
}

} // namespace gwen
