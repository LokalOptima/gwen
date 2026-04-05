#pragma once

#include "gwen/common.h"
#include <unordered_map>
#include <variant>

namespace gwen {

// --- GGUF constants ---
constexpr uint32_t GGUF_MAGIC   = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

// GGUF metadata value types
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// Metadata value (simplified: we extract what we need)
using GGUFValue = std::variant<
    uint32_t,
    int32_t,
    float,
    std::string,
    std::vector<int32_t>,
    std::vector<float>,
    std::vector<std::string>,
    uint64_t,
    int64_t,
    double,
    bool
>;

// Tensor descriptor (parsed from GGUF header, before data alignment)
struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> shape;  // dimensions
    GGMLType type;
    uint64_t offset;              // byte offset from start of tensor data region
    size_t n_elements;            // total elements
    size_t size_bytes;            // total bytes
    const void* data;             // pointer into mmap'd region (set after alignment)
};

// GGUF file reader with memory mapping
class GGUFFile {
public:
    ~GGUFFile();

    // Open and parse a GGUF file (memory-maps it)
    static std::unique_ptr<GGUFFile> open(const std::string& path);

    // Metadata access
    const std::unordered_map<std::string, GGUFValue>& metadata() const { return metadata_; }

    // Get typed metadata values (throws if missing or wrong type)
    uint32_t get_u32(const std::string& key) const;
    float    get_f32(const std::string& key) const;
    std::string get_str(const std::string& key) const;
    std::vector<int32_t> get_i32_array(const std::string& key) const;

    // Get with defaults
    uint32_t get_u32(const std::string& key, uint32_t default_val) const;
    float    get_f32(const std::string& key, float default_val) const;

    // Tensor access
    const std::vector<GGUFTensor>& tensors() const { return tensors_; }
    const GGUFTensor* find_tensor(const std::string& name) const;
    const GGUFTensor& get_tensor(const std::string& name) const;

    // Build model config from metadata
    ModelConfig build_config() const;

    // Info
    uint64_t n_tensors() const { return tensors_.size(); }
    const std::string& path() const { return path_; }

private:
    GGUFFile() = default;

    std::string path_;
    int fd_ = -1;
    void* mmap_addr_ = nullptr;
    size_t mmap_size_ = 0;

    std::unordered_map<std::string, GGUFValue> metadata_;
    std::vector<GGUFTensor> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_; // name → index in tensors_
};

} // namespace gwen
