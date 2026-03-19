# Beating llama.cpp: dp4a GEMV and the DeltaNet Fusion

*Blog post #6 in the GWEN series — the one where we catch the reference*

## The Gap

At the end of Phase 5, GWEN decoded at 252 tok/s — 57% of llama.cpp's 443 tok/s. The profiler told a clear story:

| Component | Time | Bandwidth | % Peak |
|-----------|------|-----------|--------|
| LM head GEMV | 0.66 ms | 316 GB/s | 35% |
| QKV GEMV | 0.023 ms | 157 GB/s | 18% |
| FFN gate+up | 0.029 ms | 146 GB/s | 16% |
| Full forward pass | 3.98 ms | — | 14.6% |

Two things jumped out: the GEMV kernels were leaving 65-85% of bandwidth on the table, and the forward pass was 3.98ms when the theoretical minimum (pure bandwidth) was 0.58ms. Something was fundamentally wrong with how we were doing GEMV.

## The Key Insight: dp4a

Our original GEMV kernels dequantized weights to FP16 and used FP16 multiply-accumulate. But llama.cpp does something smarter: it quantizes the *input* vector to int8, then uses `__dp4a` — a CUDA intrinsic that computes 4 int8 multiply-accumulates in a single instruction:

```cpp
// 4 int8 values packed in one int32 each
int result = __dp4a(weight_4x_int8, input_4x_int8, accumulator);
// result += w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]
```

This is pure SIMD on the CUDA cores (not Tensor Cores) — it works with any int8 data and has no special tile/shape requirements. The key advantage: we read the quantized weights directly without dequantizing to FP16, and we get 4 multiply-accumulates per instruction instead of 1.

### The Q8_1 Input Format

To use dp4a, we quantize the FP16 input vector to Q8_1 format before each GEMV:

```cpp
struct block_q8_1 {
    __half2 ds;       // {delta (scale), sum (for min-value correction)}
    int8_t qs[32];    // 32 quantized int8 values
};  // 36 bytes per block of 32 elements
```

The quantization kernel is simple: find max absolute value per block, compute scale, round to int8. One warp handles one block (32 elements = 32 threads), using warp shuffles for the reduction.

The `sum` field stores the sum of quantized values — this is needed by Q4_K and Q5_K formats which have per-sub-block minimum values. The correction term is `dmin * sum_x * m`, where `m` is the minimum scale.

### Implementing the dp4a GEMV Kernels

I implemented dp4a kernels for Q4_K, Q5_K, and Q6_K. The structure follows llama.cpp's `vec_dot` approach:

```cpp
template<int NW>
__global__ void kernel_gemv_q4_k_dp4a(
    const block_q4_k* W, const block_q8_1* x_q8,
    half* y, int out_features, int blocks_per_row)
{
    const int row = blockIdx.x;  // one row per block
    const int tid = threadIdx.x + threadIdx.y * 32;

    constexpr int QI = 32;   // int32 positions per Q4_K block
    constexpr int VDR = 2;   // virtual dequant rate
    constexpr int BLOCKS_PER_ITER = VDR * NW * 32 / QI;

    float sumf = 0.0f;
    for (int kbx = tid / (QI/VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        // Load Q4_K nibbles as packed int32
        int v0 = q4[0], v1 = q4[4];
        // Separate high/low nibbles, dp4a against Q8_1 data
        int dot1 = __dp4a(v1i, u[4], __dp4a(v0i, u[0], 0));
        // Apply scales and accumulate
        sumf += d * sumf_d - dmin * sumf_m;
    }
    // Warp shuffle + shared memory reduction → output
}
```

Each thread handles 16 elements per Q4_K super-block (256 elements). With 2 warps (NW=2, 64 threads), that's 4 blocks per iteration — perfect for `in_features=1024` where `blocks_per_row=4`.

The Q6_K kernel was the trickiest. Its memory layout is different: QI=32 but VDR=1, and the 6-bit values are split across `ql` (low 4 bits) and `qh` (high 2 bits) with a non-trivial indexing scheme. After several failed attempts, I matched llama.cpp's exact `vec_dot_q6_K_q8_1` indexing:

```cpp
const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
const int vh_shift = 2 * ((iqs % 16) / 8);
// ...
const int vi = __vsubss4(vil | vih, 0x20202020);  // center-shift: val - 32
```

### Adaptive Warp Count

A subtle problem: with NW=4 (128 threads) and `in_features=1024` (`blocks_per_row=4`), only half the threads are active — `BLOCKS_PER_ITER=8` exceeds `blocks_per_row=4`. The fix: template the kernels on NW and dispatch adaptively:

```cpp
if (blocks_per_row <= 4)
    kernel_gemv_q4_k_dp4a<2><<<out_features, dim3(32, 2)>>>(...);
else
    kernel_gemv_q4_k_dp4a<4><<<out_features, dim3(32, 4)>>>(...);
```

NW=2 gives 100% thread utilization for the 1024-dim layers (most of the model).

### Integration: Quantize Once, GEMV Many

In the forward pass, we quantize each input vector once and reuse the Q8_1 buffer across multiple GEMVs with the same input:

```cpp
// DeltaNet layer: x_norm is input to both QKV and gate projections
gwen_quantize_q8_1(x_norm, x_q8_a, n_embed, stream);      // quantize once
gwen_gemv_dp4a(W_qkv, x_q8_a, qkv, ..., stream);          // reuse
gwen_gemv_dp4a(W_gate, x_q8_a, gate_z, ..., stream);       // reuse
```

Two pre-allocated Q8_1 scratch buffers (`x_q8_a`, `x_q8_b`) alternate as needed. The quantize calls are included in the CUDA graph.

### Results: dp4a Phase

**252 → 328 tok/s (+30%)**. But the real story is in the per-kernel numbers:

| Kernel | Legacy | dp4a | Speedup | Bandwidth |
|--------|--------|------|---------|-----------|
| LM head (248K×1024) | 0.66 ms | **0.25 ms** | **2.6x** | **93% peak** |
| QKV (1024→6144) | 0.023 ms | 0.008 ms | 2.9x | 49% |
| FFN gate+up (2×) | 0.029 ms | 0.010 ms | 2.9x | 41% |
| FFN down (3584→1024) | 0.010 ms | 0.006 ms | 1.7x | 55% |

The LM head — the single biggest bottleneck — went from 35% to **93% of peak bandwidth**. That's about as good as it gets for a memory-bound operation.

## The Hidden Bottleneck: DeltaNet Recurrence

After dp4a, the profiler revealed something I hadn't noticed before:

```
All 24 layers GEMV (dp4a):  1.17 ms
Non-GEMV overhead:          1.29 ms (52.5%!)
```

Over half the forward pass was in non-GEMV operations. Digging deeper:

```
DeltaNet recurrence:  0.031 ms × 18 layers = 0.558 ms
```

The DeltaNet recurrence kernel — updating the 128×128 FP32 state matrix per head — was taking **0.031 ms per call**, burning 0.56ms across 18 layers. That's 23% of the total forward pass.

The problem was obvious once I looked at the kernel. It made **4 sequential passes** over the 64KB state matrix with 3 `__syncthreads()`:

```
Pass 1: S[i] *= decay                       (read + write S)
sync
Pass 2: sk[j] = sum_i S[i][j] * k[i]       (read S)
sync
Pass 3: S[i][j] += k[row] * delta[col]      (read + write S)
sync
Pass 4: o[j] = sum_i S[i][j] * q[i]         (read S)
```

That's 4 reads + 2 writes of the state matrix. But passes 1+2 can be fused (decay and multiply-accumulate in one read), and passes 3+4 can be fused (update and output in one read/write):

```cpp
// Pass 1: Fused decay + S^T@k
float sk_j = 0.0f;
for (int i = 0; i < 128; i++) {
    float val = S_head[i * 128 + j] * decay;
    S_head[i * 128 + j] = val;       // write decayed
    sk_j += val * sh_k[i];           // accumulate for S^T@k
}

// Pass 2: Fused update + S^T@q
float d_j = (v[j] - sk_j) * beta;
float o_j = 0.0f;
for (int i = 0; i < 128; i++) {
    float updated = S_head[i * 128 + j] + sh_k[i] * d_j;
    S_head[i * 128 + j] = updated;   // write updated
    o_j += updated * sh_q[i];        // accumulate for output
}
```

Key insight: each thread handles one column `j` of the 128×128 matrix. Columns are independent — no inter-thread communication needed between passes. The k and q vectors go into shared memory (256 bytes each, loaded once).

This cuts state memory traffic from 6 passes to 4 (2 read+write), eliminates 2 of 3 syncthreads, and keeps the data pipeline flowing.

**Result: 0.031ms → 0.008ms per call (3.7x faster).** Across 18 layers: 0.56ms → 0.15ms, saving **0.41ms per token**.

## The Final Numbers

```
╔══════════════════════════════════════════════════╗
║     GWEN vs llama.cpp — Performance Report      ║
╠══════════════════════════════════════════════════╣
║ Model:  Qwen3.5-0.8B-Q4_K_M (Q4_K/Q5_K/Q6_K)  ║
║ GPU:    NVIDIA RTX 5070 Ti (SM_120, 16GB GDDR7) ║
╚══════════════════════════════════════════════════╝

Correctness:
  Greedy token match:  30/30 vs llama.cpp (EXACT MATCH)
  dp4a kernel tests:   ALL PASSED (8/8)

Decode Speed:
                              GWEN      llama.cpp     Delta
  Decode throughput        452 t/s      434 t/s      +4.2%
  Per-token latency       2.21 ms      2.31 ms
  TTFT (5 tokens)         10.5 ms

Forward Pass Profile:
  All GEMV (dp4a):         1.17 ms  (68.5%)
  DeltaNet recurrence:     0.15 ms  ( 8.8%)
  Other non-GEMV:          0.39 ms  (22.7%)
  Total:                   1.71 ms  (585 tok/s pure forward)
  Bandwidth efficiency:    34.1% (was 14.6%)
```

## Optimization Timeline

| Phase | Decode tok/s | Key Change |
|-------|-------------|------------|
| Phase 2-3 (baseline) | 175 | First working model |
| Phase 4.1 | 184 | Remove debug overhead |
| Phase 4.2 | 215 | Kernel fusion |
| Phase 4.3 | 217 | Fused Conv1D+SiLU, sigmoid_mul |
| Phase 4.4 | 252 | CUDA Graph capture |
| **Phase 6.1** | **328** | **dp4a GEMV kernels (+30%)** |
| **Phase 6.2** | **452** | **2-pass DeltaNet + adaptive warps (+38%)** |

From 252 to 452: **+79% improvement in one optimization phase.**

## Lessons Learned

### 1. Profile the Non-GEMV Operations

I spent weeks agonizing over GEMV bandwidth (35% → 93% for LM head) without realizing that the DeltaNet recurrence was eating 23% of the total time. The lesson: after optimizing the obvious bottleneck, re-profile *everything* — the new bottleneck might be something you've never looked at.

### 2. dp4a is the Right Abstraction for Quantized GEMV

I initially considered FP8/FP4 Tensor Cores or CUTLASS for GEMV. Both are wrong:
- **Tensor Cores** need matrix tiles (16×16 minimum), useless for matrix×vector
- **CUTLASS** is a GEMM library, not GEMV
- **dp4a** is perfect: it's a SIMD instruction on regular CUDA cores, works with any quantized layout, and reads weights directly without dequantization

### 3. Memory Pass Fusion is Powerful

The DeltaNet kernel went from 4 passes to 2 passes over the state matrix — a simple transformation that yielded 3.7x speedup. No algorithmic change, no parallelism increase, just reading the data fewer times. On bandwidth-bound GPUs, memory traffic is everything.

### 4. Thread Utilization Matters at Small Dimensions

With `in_features=1024` (only 4 quantization super-blocks per row), a 128-thread kernel has 50% of threads idle. The adaptive NW=2 dispatch ensures 100% utilization. Always check that your thread block size matches the actual work available.

## What's Left

At 452 tok/s and 34.1% bandwidth efficiency, there's still room to reach the 1718 tok/s theoretical ceiling. The remaining overhead is:

1. **Small kernel launches** (~0.39ms): dozens of tiny kernels per layer (norms, activations, quantize) that could be fused into fewer, larger kernels
2. **DeltaNet recurrence** (0.15ms): only uses 16 blocks on 70 SMs. Multi-block per head or persistent kernel approaches could improve SM utilization
3. **CUDA graph overhead**: the graph replay + H2D memcpy adds ~0.5ms per token vs the pure forward pass timing

But for now: **GWEN beats llama.cpp by 4.2%**, and the model produces identical outputs. Mission accomplished.
