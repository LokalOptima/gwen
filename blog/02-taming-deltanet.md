# Taming DeltaNet + Full Model Assembly

*Blog post #2 in the GWEN series — covering Phases 2 and 3*

## The Challenge

With all individual kernels working (dequant GEMV, RMSNorm, RoPE, etc.), it was time to wire them into a complete forward pass. Qwen3.5-0.8B has 24 layers in a [3x DeltaNet, 1x FullAttn] x 6 pattern. The DeltaNet layers use a linear attention mechanism with a recurrent state — no KV cache needed.

## DeltaNet: The Delta Rule on GPU

The core of DeltaNet is a per-head recurrent state S (128x128 floats) updated via the delta rule:

```
1. S = S * exp(gate)           -- exponential decay
2. sk = S^T @ k                -- project key through state
3. delta = (v - sk) * beta     -- compute correction
4. S += outer(k, delta)        -- rank-1 state update
5. output = S^T @ q            -- query the state
```

For a single decode token, this is O(d_k * d_v) = O(128^2) per head — much cheaper than attention over the full sequence. I implemented this as a CUDA kernel with one thread block per head (128 threads), storing the state rows in registers for the matrix operations.

The state is kept in FP32 for numerical stability — the exponential decay and rank-1 updates can accumulate errors quickly in FP16.

## The 1D Convolution

Before entering DeltaNet, the QKV projection passes through a 1D convolution (kernel_size=4). For autoregressive generation, this needs to cache the last 3 values per channel. The key insight: for the first token with zero state, only `conv_weight[:, 3]` (the newest position) contributes.

## Full Attention Layers

Every 4th layer (3, 7, 11, 15, 19, 23) uses standard GQA attention with:
- 8 query heads, 2 KV heads (4:1 ratio)
- Head dim 256
- Q+gate joint projection with deinterleaved layout
- Per-head Q/K RMSNorm
- imRoPE with sections [11, 11, 10, 0]
- Output gating: `result * sigmoid(gate)`

For single-token decode at position 0, attention reduces to just returning V (single-token self-attention).

## The Bug That Broke Everything

After wiring everything together, GWEN produced garbage. A three-way comparison revealed:

| Comparison | Correlation |
|-----------|------------|
| GWEN vs f32_forward | 0.999994 |
| GWEN vs llama.cpp | 0.373 |
| f32_forward vs llama.cpp | 0.373 |

Both GWEN and my F32 reference computed the *same wrong answer*. The bug was in the forward pass logic, not the CUDA kernels.

### Hunting the Bug

I systematically verified every component:
- Embedding lookup: correct
- RMSNorm: correct
- QKV projection: correct
- Conv1d: correct
- Q/K/V split order: correct
- L2 normalization: correct
- Gate/beta computation: correct
- Full attention: correct

After writing a per-layer state dumper using llama.cpp's eval callback, I found the divergence at the very first layer:

```
attn_output-0 norm:  0.0294 (llama.cpp)
attn_output-0 norm:  0.3325 (GWEN)
```

Ratio: ~11.3 = sqrt(128). The smoking gun.

### The Q Scaling That Mattered

llama.cpp scales Q by 1/sqrt(d_k) before the DeltaNet recurrence:

```cpp
const float scale = 1.0f / sqrtf(S_k);
q = ggml_scale(ctx0, q, scale);
```

I initially dismissed this because RMSNorm theoretically absorbs any scalar scaling:

```
RMSNorm(alpha * x) = RMSNorm(x)  when alpha^2 * mean(x^2) >> eps
```

But on the *first token with zero initial state*, the DeltaNet output is tiny (~0.03 norm). At that scale, `mean(x^2) ~ 4e-7` is comparable to `eps = 1e-6`. The scaling factor leaks through the RMSNorm and corrupts all downstream computation.

The fix was two lines:
```cpp
float q_scale = 1.0f / sqrtf((float)cfg.ssm_state_size);
kernel_scale_half<<<(n + 255) / 256, 256>>>(q, q_scale, n);
```

### After the Fix

| Comparison | Correlation | KL Divergence |
|-----------|------------|---------------|
| GWEN vs llama.cpp | 0.999466 | 0.003 |
| GWEN vs f32_forward | 0.999998 | ~0 |

Top-10 tokens match exactly. Greedy decode produces identical output for 22+ tokens.

## The Tokenizer

I used ggml's BPE tokenizer implementation to match llama.cpp exactly. The Qwen3.5 vocabulary has 248,320 tokens — a massive embedding table (Q6_K quantized).

## Generation Results

```
Prompt: "The quick brown fox"
GWEN:      "The quick brown fox and the lazy dog\n..."
llama.cpp: "The quick brown fox and the lazy dog\n..."

Prompt: "Hello, my name is"
GWEN:      "Hello, my name is John. I am a 20-year-old male..."
```

Coherent, matching output. The model works.

## Performance

Current numbers on RTX 5070 Ti:

| Metric | Value |
|--------|-------|
| Decode speed | 175 tok/s |
| Prompt+30 gen latency | 178ms |
| Model load time | 160ms |
| GPU memory | 566 MB |

This is without any optimization — just correctness-first implementation. Phase 4 will focus on performance.

## Lessons Learned

1. **Never dismiss "theoretically irrelevant" numerical details.** The q scaling was mathematically absorbed by RMSNorm in the general case, but not when values are near the epsilon threshold. In deep networks, these edge cases compound across layers.

2. **Build comparison infrastructure early.** The eval callback dumper, F32 reference forward pass, and logit comparison tools were essential for finding the bug.

3. **Three-way comparison is powerful.** By comparing GWEN, an F32 reference, and llama.cpp, I could isolate whether the bug was in the CUDA kernels (GWEN vs F32) or the forward pass logic (both vs llama.cpp).

## What's Next

Phase 4: profiling and optimization. The current 175 tok/s should be significantly improvable — the RTX 5070 Ti has 896 GB/s bandwidth and the Q4_K_M model is only 500MB, so the theoretical decode limit is ~1700+ tok/s. There's 10x headroom to chase.
