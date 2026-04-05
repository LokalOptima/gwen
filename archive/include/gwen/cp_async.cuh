#pragma once
// Simplified API for asynchronous data loading.
// Extracted from llama.cpp ggml/src/ggml-cuda/cp-async.cuh — unchanged except:
//   - Replaced GGML_UNUSED / NO_DEVICE_CODE macros with CUDA-standard equivalents.
//   - Replaced CP_ASYNC_AVAILABLE with the CUDA version check directly.
//   - Removed the ggml_cuda_ prefix from helper name.

#include <cuda_fp16.h>

// cp.async is available on Ampere (SM_80) and later, including SM_120.
#define GWEN_CP_ASYNC_AVAILABLE (__CUDA_ARCH__ >= 800)

static __device__ __forceinline__ unsigned int gwen_cvta_generic_to_shared(void * generic_ptr) {
#if __CUDA_ARCH__ >= 800
    return __cvta_generic_to_shared(generic_ptr);
#else
    (void)generic_ptr;
    return 0;
#endif
}

// Copies 16 bytes from global to shared memory using cp.async.cg.
// Both src and dst must be aligned to 16 bytes.
// dst is passed as a 32-bit shared-memory address (from gwen_cvta_generic_to_shared).
// preload == L2 prefetch hint in bytes (0 = no hint, 64/128/256 = hint sizes).
template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256,
                  "preload must be 0, 64, 128, or 256");
#if __CUDA_ARCH__ >= 800
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif  // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
#else
    // Fallback: synchronous copy (16 bytes = 4 x uint32_t)
    const uint4 * src4 = (const uint4 *) src;
    uint4       * dst4 = (uint4 *) __cvta_shared_to_generic(dst);
    *dst4 = *src4;
    (void)preload;
#endif
}

// Commits all outstanding cp.async operations into the current pipeline group.
// Must be called before cp_async_wait_all (or cp.async.wait_group 0).
static __device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

// Waits until all outstanding cp.async operations have completed.
// Does NOT provide additional thread synchronization — a __syncthreads() is
// still required before the loaded data is visible to other threads.
static __device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;");
#endif
}
