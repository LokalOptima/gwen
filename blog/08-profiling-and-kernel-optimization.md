# Profiling and Kernel Optimization: 493 to 599 tok/s

*Blog post #8 in the GWEN series — ncu tells all*

## Finally: Real Profiling

The previous posts relied on CUDA event timing to estimate where time was going. That's useful but coarse — you can't see *why* a kernel is slow. This time I got proper profiling working:

- **nsys** (Nsight Systems) with `--cuda-graph-trace=node` to see individual kernel timing within CUDA graph replays
- **ncu** (Nsight Compute) with `--set full` for per-kernel metrics: achieved bandwidth, occupancy, warp stall reasons, register counts

The `--cuda-graph-trace=node` flag was critical. Without it, nsys just shows a single `cudaGraphLaunch` bar — useless for understanding what's happening inside the graph.

## The nsys Surprise

The first nsys trace revealed something I didn't expect. I was focused on GEMV bandwidth efficiency (37-55%), expecting that to be the dominant gap. But the timeline showed two embarrassing hotspots hiding in plain sight:

| Kernel | Time per decode step | % of forward |
|--------|---------------------|--------------|
| `kernel_argmax` | **139 μs** | 8.2% |
| `kernel_gqa_attention_decode` (×6 layers) | **276 μs** | 16.3% |
| All GEMVs | 1,180 μs | 69.4% |
| Everything else | 105 μs | 6.2% |

The argmax and GQA attention kernels — which I'd written quickly and never profiled — were eating **415 microseconds** per decode step. Nearly a quarter of the forward pass.

## The Argmax Problem: 1 Block on 70 SMs

The argmax kernel was the simplest possible implementation: a single block of 256 threads scanning 248,320 logits sequentially. One block means one SM working while 69 others sit idle. The kernel takes 139 μs, but on 70 SMs the work could be split across all of them.

The fix is a classic two-phase parallel reduction:

**Phase 1** — 256 blocks of 256 threads, each scanning a chunk of the vocab:
```cpp
constexpr int ARGMAX_BLOCKS = 256;

__global__ void __launch_bounds__(256)
kernel_argmax_partial(const float* __restrict__ logits,
                      float* __restrict__ partial_max,
                      int* __restrict__ partial_idx, int n) {
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = global_tid; i < n; i += stride) {
        float v = logits[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    // Shared memory reduction within block → write partial_max[bid], partial_idx[bid]
    ...
}
```

**Phase 2** — 1 block of 256 threads reduces the 256 partial results:
```cpp
__global__ void __launch_bounds__(256)
kernel_argmax_reduce(const float* partial_max, const int* partial_idx,
                     int* result, int n_partials) {
    // 256 threads, 256 partials — one per thread, then shared memory reduction
    ...
}
```

Result: **139 μs → 3.5 μs** (40x faster). The second phase adds negligible overhead since it's just 256 elements.

## The GQA Attention Problem: 32 Threads per Head

The GQA attention kernel used one warp (32 threads) per query head. For head_dim=256, that means each thread handles 8 elements in the value accumulation loop. But worse, for the scoring step, a single warp iterates over the entire sequence length sequentially — at position 100 in the KV cache, that's 100 dot products done by 32 threads one at a time.

The fix: 8 warps (256 threads) per head.

**Scoring** — warps split across time steps. Warp 0 does t=0,8,16,..., warp 1 does t=1,9,17,...:
```cpp
for (int t = warp_id; t < seq_len; t += N_WARPS) {
    // Each warp does one dot product — 32 threads cover head_dim
    float dot = 0.0f;
    for (int d = lane; d < head_dim; d += 32)
        dot += __half2float(q_head[d]) * __half2float(k_t[d]);
    // Warp-level reduction → scores[t]
}
```

**Softmax** — all 256 threads cooperate on the reduction. Cross-warp max/sum via shared memory:
```cpp
__shared__ float s_reduce[N_WARPS];
// Each warp reduces its portion, writes to s_reduce[warp_id]
if (lane == 0) s_reduce[warp_id] = max_val;
__syncthreads();
// Thread 0 does final reduction across 8 values
if (tid == 0) {
    float m = s_reduce[0];
    for (int i = 1; i < N_WARPS; i++) m = fmaxf(m, s_reduce[i]);
    s_reduce[0] = m;
}
__syncthreads();
max_val = s_reduce[0];
```

**Value accumulation** — 256 threads directly cover head_dim=256, one element per thread:
```cpp
for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int t = 0; t < seq_len; t++)
        acc += scores[t] * __half2float(v_t[d]);
    out_head[d] = __float2half(acc);
}
```

Result: **276 μs → 47 μs** (5.9x faster) for 6 full-attention layers combined.

### The Shuffle Deadlock

The first version of the multi-warp kernel hung the GPU. The bug was subtle: in the cross-warp reduction, I used `__shfl_xor_sync(0xFFFFFFFF, ...)` inside an `if (tid < N_WARPS)` block. Only 8 of 32 threads in warp 0 entered the conditional, but the mask `0xFFFFFFFF` requires all 32 threads to participate. That's undefined behavior and caused a deadlock.

The fix was simple — use shared memory sequential reduction by thread 0 instead of shuffles for the cross-warp step. Eight iterations is nothing compared to the cost of getting it wrong.

## ncu Deep Dive

With the bottleneck kernels fixed, I also ran full ncu profiles on the remaining kernels for reference:

### GEMV Q4_K (small per-layer GEMVs)
- **37-55% bandwidth efficiency** — confirmed by ncu
- Root cause: poor sector utilization. `block_q4_k` is 144 bytes, threads access non-contiguous fields (qs[0], qs[4], scales, mins) with 16-byte stride within each block. Only **9.2 of 32 bytes** per DRAM sector are useful
- Fix would require weight relayout to structure-of-arrays format — significant effort, deferred

### LM Head Q6_K (single large GEMV)
- **93% bandwidth efficiency** — essentially optimal
- 248,320 × 1024 matrix, 48 registers/thread (slightly above 42 target)
- Already at roofline; no optimization needed

### Fused Kernels
- All fused kernels (RMSNorm+Q8_1, SwiGLU+Q8_1, GatedRMSNorm+Q8_1) confirmed faster than their unfused counterparts
- Main benefit is launch overhead reduction, not raw kernel speedup

### DeltaNet
- Only 16 blocks for 70 SMs (0.23 waves) — poor SM occupancy
- But inherently sequential recurrence limits parallelism
- State matrix [128×128] FP32 = 64 KB per head fits well in L1

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| E2E decode (100 tokens) | 493 tok/s | **599 tok/s** | **+21.5%** |
| Forward pass | 1.70 ms | **1.41 ms** | -17% |
| GQA attention (6 layers) | 276 μs | **47 μs** | 5.9x faster |
| Argmax | 139 μs | **3.5 μs** | 40x faster |
| vs llama.cpp | +10.6% | **+34.3%** | |
| BW efficiency | 36.9% | **44.5%** | |

30/30 exact greedy token match maintained.

## Where We Stand

```
Forward pass budget (1.41 ms):
  GEMV (all layers + LM head)    1.18 ms   83.7%
  GQA attention (6 layers)       0.047 ms   3.3%
  DeltaNet (18 layers)           0.15 ms   10.6%
  Argmax                         0.004 ms   0.3%
  Other (norms, acts, quant)     0.029 ms   2.1%

Theoretical minimum:             0.627 ms
Current forward:                 1.41 ms (44.5% efficiency)
```

The forward pass is now dominated almost entirely by GEMVs (84%), which is exactly where you want to be for a bandwidth-bound workload. The remaining gap is mostly GEMV coalescing inefficiency (37-55% BW on small matrices) and the inherently sequential DeltaNet recurrence.

## What's Next

The clear next optimization target is GEMV memory coalescing. The Q4_K format stores `qs`, `scales`, and `mins` interleaved within each 144-byte block. When 32 threads in a warp each access a different block's `qs[0]`, they generate scattered 32-byte sector requests with only 9.2 bytes useful per sector. A structure-of-arrays weight relayout could potentially push small GEMVs from 37-55% to 70-80% bandwidth, which would bring the forward pass under 1.0 ms and push decode past 700 tok/s.
