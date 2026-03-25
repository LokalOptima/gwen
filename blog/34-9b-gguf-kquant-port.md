# 34. Scaling to 9B: GGUF K-Quant Decode

*New direction: implementing Qwen3.5-9B inference using Unsloth's UD-Q4_K_XL GGUF weights.*

## Motivation

After months optimizing the 0.8B model with custom FP8/FP4 formats, I wanted to run the 9B instruct model — specifically `unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL` — through GWEN's CUDA kernels. This meant going back to native GGUF K-quant formats (Q4_K, Q5_K, Q6_K, Q8_0) instead of our custom GWFP8/GWFP4 pipelines.

The 9B model is 11× the parameters of 0.8B but fits comfortably in 16 GB VRAM at Q4 quantization (~5.4 GB weights + ~2 GB state). The architecture is identical — same hybrid DeltaNet + Transformer — just bigger in every dimension.

## Architecture Differences: 0.8B → 9B

| Parameter | 0.8B | 9B |
|---|---|---|
| Layers | 24 | 32 |
| Hidden | 1024 | 4096 |
| FFN | 3584 | 12288 |
| Q heads (full attn) | 8 | 16 |
| KV heads (full attn) | 2 | 4 |
| DeltaNet K heads | 16 | 16 |
| **DeltaNet V heads** | **16** | **32** |
| Tie embeddings | yes | **no** |
| Weight size (Q4) | ~0.6 GB | ~5.4 GB |

The critical new challenge: **asymmetric K/V heads in DeltaNet**. The 0.8B has K=V=16 (symmetric), but the 9B has K=16, V=32 — each pair of V-heads shares one K-head, analogous to GQA in standard attention.

## What Needed Changing

### 1. GGML Shape Convention

This one bit me hard. GGML stores 2D tensor shapes as `[ne0, ne1]` where `ne0` is the inner dimension (columns/in_features) and `ne1` is the outer dimension (rows/out_features). GWEN's existing code passed `shape[0]` as `out_features` and `shape[1]` as `in_features` — exactly backwards for GGML.

This worked for FP8/FP4 because those custom formats stored shapes in GWEN's convention. But for GGUF tensors, every GEMV was computing the wrong matrix-vector product.

**Fix**: Swap `shape[0]` and `shape[1]` at load time for all 2D tensors:

```cpp
static WeightRef weight_from_tensor(const GGUFTensor& t) {
    WeightRef w;
    w.host_data = t.data;
    w.type = t.type;
    w.n_elements = t.n_elements;
    w.size_bytes = t.size_bytes;
    w.shape = t.shape;
    if (w.shape.size() == 2) {
        std::swap(w.shape[0], w.shape[1]);
    }
    return w;
}
```

The symptom was NaN after layer 1 — not layer 0 (which happened to have symmetric weight shapes that masked the bug).

### 2. K-Quant GEMV Dispatch

GWEN's `gemv_dispatch()` only handled FP8, FP4, and F16. I added K-quant routing:

```cpp
static inline void gemv_dispatch(const WeightRef& w, const half* x, half* y,
                                  int out_features, int in_features, cudaStream_t s) {
    if (w.type == GGMLType::FP4_E2M1) {
        gwen_gemv_fp4(...);
    } else if (w.type == GGMLType::F16) {
        gwen_gemv_fp16(...);
    } else if (is_kquant_type(w.type)) {
        gwen_gemv(w.device_data, x, y, out_features, in_features, w.type, s);
    } else {
        gwen_gemv_fp8(...);  // default for GWFP8
    }
}
```

The K-quant GEMV kernels (Q4_K, Q5_K, Q6_K, Q8_0) already existed from the original 0.8B GGUF work — they just weren't wired into the dispatch.

The `residual_f32` variant needed a scratch buffer since there's no fused K-quant GEMV+F32 residual kernel. The two-step approach: GEMV → FP16 scratch → `gwen_fp16_to_f32_add()`.

### 3. Separate output.weight (Untied Embeddings)

The 0.8B ties the embedding table to the LM head (`tie_word_embeddings=true`). The 9B has a separate `output.weight` tensor (Q6_K, 788 MB). I added:

- `output_weight` field on Model
- `tie_word_embeddings` flag on ModelConfig
- Auto-detection at load time (check if `output.weight` exists in GGUF)
- `lm_head_weight(model)` helper to select the right tensor

### 4. IQ4_XS → FP16 Conversion

Unsloth's UD-Q4_K_XL quantization uses IQ4_XS (imatrix-based non-linear 4-bit) for 10 ffn_gate/ffn_up tensors on less-sensitive layers. GWEN has no IQ4_XS GEMV kernel.

Rather than implementing a new kernel for 10 tensors, I convert IQ4_XS → FP16 at model load time on CPU. The dequant uses the `kvalues_iq4nl` 16-entry lookup table:

```
value = d * (sub_scale - 32) * kvalues_iq4nl[nibble]
```

Cost: 730 MB extra VRAM (10 tensors × ~73 MB each). Acceptable within the 16 GB budget.

A subtle bug: the converted tensors skipped the GGML shape swap (they were created in the IQ4_XS conversion path, not through `weight_from_tensor`). This caused NaN at layer 1 (which has IQ4_XS ffn_gate). Fix: apply the same 2D shape swap in the conversion code path.

### 5. F16 ssm_alpha/beta

The 0.8B patched GGUF had ssm_alpha/beta as Q8_0. The 9B has them as F16. The DeltaNet fused kernel had a conditional for FP4 (external projection) vs FP8 (inline dot product). I broadened it to:

```cpp
if (w.ssm_alpha.type != GGMLType::FP8_E4M3) {
    // External GEMV projection path (FP4, F16, K-quant)
} else {
    // Inline FP8 dot product in fused kernel
}
```

### 6. Q4_K Embedding Lookup

The 9B's `token_embd` is Q4_K (the 0.8B used Q6_K). I wrote a new `kernel_embed_lookup_q4k` that dequantizes one row of Q4_K on the fly — same pattern as the existing Q6_K kernel but with Q4_K's scale/min/nibble extraction.

### 7. Auto-disable MTP and Prefill

MTP weights are model-specific (trained for 0.8B). Auto-disabled for GGUF models. The FP8 prefill path doesn't work with K-quant weights either — disabled until a dequant-then-GEMM path is implemented.

## The Debugging Journey

### NaN Cascade

First run: `CUDA error: CUTLASS FP8 GEMM initialization failed`. The prefill path was trying FP8 GEMM on K-quant weights. Fixed by disabling prefill for GGUF.

Second run: `illegal memory access`. MTP was auto-loading from the default 0.8B path. Fixed by auto-disabling for GGUF.

Third run: "Hello!" for max_predict=1. But max_predict=5 gave `!!!!`. Logit dump showed: **all NaN**.

### Layer-by-Layer NaN Hunt

With GWEN_DEBUG, I traced NaN through the forward pass:

```
[DBG embed]  0.000323 0.003847 0.007370 -0.006725    ← OK
[DBG L0]     nan nan nan nan                          ← Layer 0 already NaN!
```

After more debug checks:

```
[DBG L0 norm]     0.023 0.266 0.469 -0.451     ← OK
[DBG L0 qkv]      146.4 208.1 427.8 -317.0     ← OK (large but valid)
[DBG L0 ffn_norm]  0.017 0.210 0.440 -0.374     ← OK
[DBG L0 ffn_gate]  -140.1 -56.0 -183.2 189.0    ← OK
[DBG L0 swiglu]    -0.000 0.000 0.000 846.0      ← OK
[DBG L0]           nan nan nan nan                ← NaN after FFN down!
```

The NaN was in `gemv_dispatch_residual_f32(w.ffn_down, ...)`. Root cause: **GGML shape convention** — shape[0] and shape[1] were swapped, so the ffn_down GEMV was computing a 12288→12288 projection instead of 12288→4096. The kernel read out-of-bounds memory.

Fixed with the 2D shape swap at load time (described above).

### Layer 1 NaN: IQ4_XS Shape Bug

After fixing the shape convention, layer 0 was fine but layer 1 had NaN:

```
[DBG L1 ffn_gate] nan nan nan nan
```

Layer 1's ffn_gate is IQ4_XS, converted to F16 at load. The conversion code set `w.shape = t.shape` directly from the GGUF tensor — which was still in GGML convention (not swapped). The F16 GEMV launched with wrong dimensions.

Fix: apply the same 2D swap in the IQ4_XS conversion path.

### 0.8B Works, 9B Doesn't

After fixing both NaN sources, the model ran and produced coherent-ish text. But comparing against llama-simple:

```
llama-simple: "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
gwen:         "1 2 3 4 5 6 7 8 2 2 3 3 3 3 3 3 3 3"
```

Testing with the 0.8B GGUF (same K-quant path): **exact match** with llama-simple. The K-quant GEMV kernels are correct for the 0.8B's symmetric K=V=16 heads.

The 9B's DeltaNet with K=16/V=32 is where the bug lives. Even single-token prompts diverge: GWEN's top prediction for "1" is a Chinese character (token 96287 = 元), while llama-simple predicts "9".

## Current State

**What works:**
- 0.8B GGUF: 195 tok/s, exact match with llama-simple
- 9B loading: all 427 tensors, 5.4 GB → GPU in ~1 sec
- 9B decode: 28 tok/s, no NaN, CUDA graphs working
- All K-quant types: Q4_K, Q5_K, Q6_K, Q8_0, F16

**What's broken:**
- 9B output correctness — DeltaNet K≠V asymmetry handling
- Prefill not yet implemented for K-quant path

**Debugging infrastructure built:**
- `GWEN_DEBUG` build flag: per-layer intermediate value checks
- `GWEN_LAYER_LIMIT=N`: run only first N layers
- `GWEN_LOGITS_BIN=path`: binary logit dumps for comparison
- `GWEN_DUMP_LOGITS=1`: top-5 per position

## Lessons Learned

1. **Convention mismatches are insidious.** The GGML shape swap silently produced wrong-but-not-crashing results for most tensors. Only the asymmetric ones (ffn_down with 12288≠4096) caused NaN. Symmetric tensors worked by accident.

2. **Test on the simplest model first.** Running the 0.8B GGUF through the same K-quant path immediately confirmed the kernels were correct — isolating the 9B-specific bug.

3. **Don't assume equivalent code paths are equivalent.** The IQ4_XS conversion created WeightRefs outside the normal `weight_from_tensor` path, missing the shape swap. Every path that creates a WeightRef needs the same conventions.

4. **NaN is your friend.** It's much easier to debug than "slightly wrong output." The shape bug that caused NaN was found in minutes; the remaining correctness issue (which produces valid-but-wrong logits) is far harder to track down.

## Next Steps

The remaining correctness bug requires per-layer activation comparison against llama.cpp. The `kernel_deltanet_fused` GQA mapping (`k_head = head * n_k_heads / n_v_heads`) looks correct dimensionally, but something about how the S matrix accumulates with shared K-heads is producing subtly wrong results. The DeltaNet recurrence is the prime suspect.

After correctness, the targets are:
- K-quant prefill (dequant-then-GEMM for multi-token prompt processing)
- dp4a decode optimization (quantize input to Q8_1, use integer SIMD)
- CUDA graph optimization and L2 cache tuning for 9B weight reads
