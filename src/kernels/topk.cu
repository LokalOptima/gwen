#include "gwen/kernels.h"
#include <cub/block/block_radix_sort.cuh>

namespace gwen {

// Top-k selection using CUB BlockRadixSort.
// K=4096, 256 threads → 16 items/thread. Sort descending, output first k.
// Pack (sortable_fp16, original_index) into uint32 for single-key sort.
static constexpr int TOPK_BLOCK = 256;
static constexpr int TOPK_IPT = 16;  // items per thread = K / BLOCK

__global__ void __launch_bounds__(TOPK_BLOCK)
kernel_topk(const half* __restrict__ logits,
            uint16_t* __restrict__ topk_indices,
            half* __restrict__ topk_values,
            int n_rows, int K, int k) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int tid = threadIdx.x;
    const half* row_data = logits + (size_t)row * K;

    // Load items in blocked arrangement: thread t owns items [t*IPT .. (t+1)*IPT)
    uint32_t keys[TOPK_IPT];
    #pragma unroll
    for (int i = 0; i < TOPK_IPT; i++) {
        int idx = tid * TOPK_IPT + i;
        if (idx < K) {
            uint16_t raw = __half_as_ushort(row_data[idx]);
            // Convert fp16 to sortable uint16 (ascending = larger fp16 → larger uint16)
            uint16_t sortable = (raw & 0x8000) ? ~raw : (raw ^ 0x8000);
            keys[i] = ((uint32_t)sortable << 16) | (uint32_t)idx;
        } else {
            keys[i] = 0;  // pad with minimum
        }
    }

    // CUB block-level radix sort (descending): largest values first
    using BlockSort = cub::BlockRadixSort<uint32_t, TOPK_BLOCK, TOPK_IPT>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    BlockSort(sort_storage).SortDescending(keys);
    __syncthreads();

    // After descending sort in blocked arrangement:
    // Thread 0 has the top-16, thread 1 has next 16, etc.
    // For k=64: threads 0-3 write their items.
    int base = tid * TOPK_IPT;
    if (base < k) {
        int end = min(base + TOPK_IPT, k);
        for (int i = 0; i < end - base; i++) {
            int pos = base + i;
            uint16_t idx = keys[i] & 0xFFFF;
            topk_indices[(size_t)row * k + pos] = idx;
            topk_values[(size_t)row * k + pos] = row_data[idx];
        }
    }
}

void gwen_topk(const half* logits, uint16_t* topk_indices, half* topk_values,
               int n_rows, int K, int k, cudaStream_t stream) {
    kernel_topk<<<n_rows, TOPK_BLOCK, 0, stream>>>(
        logits, topk_indices, topk_values, n_rows, K, k);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
