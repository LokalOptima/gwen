# Squeezing the 5070 Ti

*Blog post #3 in the GWEN series — covering Phase 4 optimization*

## Starting Point

After Phase 2+3, GWEN was generating correct text at 175–184 tok/s on an RTX 5070 Ti. Time to optimize. The model is 497 MB in Q4_K_M, and the 5070 Ti has 896 GB/s bandwidth — theoretical minimum is 0.58 ms/token or ~1700 tok/s. We're at ~12% efficiency. Where's the other 88%?

## Profiling

I built a profiler (`bench/profile_forward.cu`) that isolates each component with CUDA events:

| Component | Time (ms) | Bandwidth | % Peak |
|-----------|----------|-----------|--------|
| LM head GEMV (1024→248K) | 0.659 | 317 GB/s | 35% |
| QKV GEMV (1024→6144) | 0.023 | 159 GB/s | 18% |
| FFN gate+up (2× GEMV) | 0.029 | 146 GB/s | 16% |
| FFN down GEMV | 0.010 | 204 GB/s | 23% |
| All other kernels | <0.003 each | — | — |
| **Full forward pass** | **4.66** | — | **12.5%** |

The story is clear: **kernel launch overhead dominates**. Each forward pass launches ~350 CUDA kernels. At ~5–7 μs each for launch + scheduling, that's ~2 ms of pure overhead. The actual computation and memory transfers are only ~2.5 ms.

## The Optimizations

### 1. Eliminate Per-Token cudaMalloc

The argmax for greedy decoding was doing `cudaMalloc`/`cudaFree` every single token:

```cpp
// Before: malloc/free per token
int* d_token;
cudaMalloc(&d_token, sizeof(int));
kernel_argmax<<<1, 256>>>(logits_f, d_token, n_vocab);
cudaMemcpy(&next_token, d_token, sizeof(int), D2H);
cudaFree(d_token);

// After: pre-allocated in allocate()
kernel_argmax<<<1, 256>>>(logits_f, d_argmax_token, n_vocab);
cudaMemcpy(&next_token, d_argmax_token, sizeof(int), D2H);
```

`cudaMalloc` is expensive — it synchronizes with the GPU and can take 10–50 μs.

### 2. QKV Pointer Aliasing

The DeltaNet layer projects x_norm into a 6144-element QKV buffer, then splits it into Q[2048], K[2048], V[2048]. The original code copied each segment:

```cpp
// Before: 3 device-to-device copies per DeltaNet layer = 54 total
cudaMemcpy(q, qkv, 2048 * sizeof(half), D2D);
cudaMemcpy(k, qkv + 2048, 2048 * sizeof(half), D2D);
cudaMemcpy(v, qkv + 4096, 2048 * sizeof(half), D2D);

// After: just alias pointers (zero cost)
half* q = qkv;
half* k = qkv + cfg.ssm_inner_size;
half* v = qkv + 2 * cfg.ssm_inner_size;
```

This works because L2 normalize is in-place and the DeltaNet kernel reads Q/K/V without writing to them.

### 3. Pointer-Swap-Free Residual Pattern

The residual connection (x → norm → attn → add(x) → norm → FFN → add(x)) originally used pointer swaps with `std::swap(x, residual)`. While correct, this prevented CUDA graph capture (the GPU sees different buffer addresses each time).

I replaced it with a fixed buf_a/buf_b assignment:

```
Layer start: hidden state in buf_a
  RMSNorm(buf_a → x_norm)
  GEMV(x_norm → ...) → attn pipeline
  GEMV(gated_out → buf_b)          // write to alternate buffer
  add_inplace(buf_b, buf_a)        // buf_b = attn_output + residual
  RMSNorm(buf_b → x_norm)
  FFN(x_norm → ...)
  GEMV(ffn_out → buf_a)            // write back to original
  add_inplace(buf_a, buf_b)        // buf_a = FFN_output + residual
Layer end: hidden state back in buf_a ✓
```

Each layer deterministically alternates writes between buf_a and buf_b, always ending with the result in buf_a.

### 4. Fused Q Scaling into L2 Normalize

The DeltaNet Q vector needs to be L2-normalized *and* scaled by 1/√(d_k). Previously two separate kernels:

```cpp
// Before: 2 kernel launches
gwen_l2_normalize(q, q, n_heads, state_size);
kernel_scale_half<<<blocks, 256>>>(q, 1/sqrt(128), n);

// After: single kernel with extra_scale parameter
gwen_l2_normalize(q, q, n_heads, state_size, 1.0f / sqrtf(128.0f));
```

18 fewer kernel launches per token.

### 5. Batched Per-Head RMSNorm

Full attention layers normalize Q and K per-head. With 8 Q heads and 2 K heads, that was 10 separate RMSNorm kernel launches. A new batched kernel does all heads in one launch:

```cpp
// Before: 10 kernel launches per full attention layer
for (int h = 0; h < 8; h++)
    gwen_rmsnorm_f32w(fa_q + h*256, weight, fa_q + h*256, 256, eps);
for (int h = 0; h < 2; h++)
    gwen_rmsnorm_f32w(fa_k + h*256, weight, fa_k + h*256, 256, eps);

// After: 2 kernel launches (one for Q heads, one for K heads)
gwen_rmsnorm_batched_f32w(fa_q, weight, fa_q, 8, 256, eps);
gwen_rmsnorm_batched_f32w(fa_k, weight, fa_k, 2, 256, eps);
```

48 fewer kernel launches per token (8 saved × 6 attention layers).

### 6. Debug Stripping

All `debug_print_half`, `debug_print_float`, early logit probes, and per-layer `cudaDeviceSynchronize()` calls were removed. Each sync point forces the CPU to wait for the GPU, killing pipeline parallelism. The debug code also performed full GEMV operations at every layer for "early logit probes" — that's 24 extra GEMV calls per first token.

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Decode speed | 184 tok/s | 215 tok/s | +17% |
| Forward pass | 5.4 ms | 4.66 ms | -14% |
| Kernel launches | ~500 | ~350 | -30% |
| Bandwidth efficiency | 11% | 12.5% | +14% |

## What's Left on the Table

The remaining gap to theoretical (215 vs 1700 tok/s) is dominated by:

1. **Kernel launch overhead** (~2 ms): Each of ~350 kernel launches costs 5-7 μs in scheduling overhead. CUDA Graphs could eliminate most of this, but the full attention layers have position-dependent parameters (KV cache offsets, sequence length) that complicate graph capture.

2. **GEMV bandwidth underutilization** (18-35% of peak): For small weight matrices like 1024→2048, the 256-thread block processes only 4 Q4_K blocks per row — not enough work to hide memory latency. Solutions: multi-row processing, shared memory caching of the input vector, or a completely different approach (persistent kernels).

3. **LM head dominates compute** (0.66 ms = 14% of total): At 317 GB/s, the 248K-row GEMV uses only 35% of the 896 GB/s bandwidth. The Q6_K dequantization path has complex byte extraction that bottlenecks on ALU, not memory.

## What's Next

Phase 5: Prefill optimization. Currently GWEN processes prompt tokens sequentially — the same single-token decode path. A chunked parallel DeltaNet kernel and Flash Attention would enable processing all prompt tokens at once, dramatically improving time-to-first-token.
