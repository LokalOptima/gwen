#pragma once

#include "gwen/common.h"

namespace gwen {

// RAII wrapper for a CUDA device allocation
class CudaBuffer {
public:
    CudaBuffer() = default;
    ~CudaBuffer() { free(); }

    // Move only
    CudaBuffer(CudaBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // No copy
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Allocate
    static CudaBuffer alloc(size_t bytes) {
        CudaBuffer buf;
        if (bytes > 0) {
            GWEN_CHECK_CUDA(cudaMalloc(&buf.data_, bytes));
            buf.size_ = bytes;
        }
        return buf;
    }

    void free() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    // Upload from host
    void upload(const void* host_src, size_t bytes) {
        GWEN_CHECK(bytes <= size_, "Upload size exceeds buffer");
        GWEN_CHECK_CUDA(cudaMemcpy(data_, host_src, bytes, cudaMemcpyHostToDevice));
    }

    void upload_async(const void* host_src, size_t bytes, cudaStream_t stream) {
        GWEN_CHECK(bytes <= size_, "Upload size exceeds buffer");
        GWEN_CHECK_CUDA(cudaMemcpyAsync(data_, host_src, bytes, cudaMemcpyHostToDevice, stream));
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }

    template<typename T> T* as() { return static_cast<T*>(data_); }
    template<typename T> const T* as() const { return static_cast<const T*>(data_); }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

// Simple arena-style allocator for inference buffers
// Allocates one big chunk and hands out aligned sub-regions
class CudaAllocator {
public:
    CudaAllocator() = default;
    ~CudaAllocator() = default;

    // Allocate a new buffer (tracked for lifetime)
    void* alloc(size_t bytes, size_t alignment = 256) {
        auto buf = CudaBuffer::alloc(bytes);
        void* ptr = buf.data();
        buffers_.push_back(std::move(buf));
        return ptr;
    }

    // Upload host data to a new GPU buffer, return device pointer
    void* upload(const void* host_data, size_t bytes) {
        void* ptr = alloc(bytes);
        GWEN_CHECK_CUDA(cudaMemcpy(ptr, host_data, bytes, cudaMemcpyHostToDevice));
        return ptr;
    }

    // Total allocated bytes
    size_t total_allocated() const {
        size_t total = 0;
        for (auto& b : buffers_) total += b.size();
        return total;
    }

    size_t n_allocations() const { return buffers_.size(); }

private:
    std::vector<CudaBuffer> buffers_;
};

} // namespace gwen
