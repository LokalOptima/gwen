# Starting GWEN — GPU-native Wired Execution for Neural Nets

*Phase 0: Infrastructure & Tooling*

## Why?

I want to understand what happens when you type a prompt and get a response. Not at the API level — at the GPU level. How do weights get loaded? How does a quantized 4-bit matrix multiply work on a tensor core? What does a DeltaNet recurrence look like as a CUDA kernel?

The plan: rewrite Qwen3.5-0.8B inference from scratch in CUDA, optimized for a single GPU — the RTX 5070 Ti.

## Why Qwen3.5?

Qwen3.5-0.8B is not a standard transformer. It uses a **hybrid architecture**:
- **18 DeltaNet layers** (linear attention with a recurrent state matrix)
- **6 Full Attention layers** (standard GQA, every 4th layer)

This is more interesting than yet another GPT clone. DeltaNet gives O(1) memory per decoding step — no growing KV cache for 75% of layers. The model is small enough (0.8B params, ~500MB quantized) to fit entirely in GPU memory with room to spare, but complex enough to be a real optimization challenge.

## The Target: RTX 5070 Ti

| Spec | Value |
|------|-------|
| Architecture | Blackwell (SM_120) |
| SMs | 70 |
| Tensor Cores | 280 (5th gen: FP16/BF16/FP8/FP4) |
| VRAM | 16 GB GDDR7 |
| Bandwidth | 896 GB/s |
| L2 Cache | 48 MB |
| L1/Shared per SM | 128 KB |

The key insight: **single-token decode is memory-bandwidth-bound**. The model is ~500MB. At 896 GB/s bandwidth, the theoretical minimum time to load all weights once is ~0.56ms. Every optimization that matters will be about reducing bytes loaded and maximizing bandwidth utilization.

The 48MB L2 cache is interesting — it can hold ~10% of the model weights. If we're clever about access patterns, we can get "free" caching on frequently-accessed small tensors (norms, biases, decay parameters).

## Architecture Deep Dive

```
Input tokens → Embedding (Q6_K, 248K vocab × 1024 dim)
     │
     ├─ × 6 repetitions of:
     │   ├── DeltaNet Layer  (linear attention, recurrent state 128×128)
     │   ├── DeltaNet Layer
     │   ├── DeltaNet Layer
     │   └── Full Attention Layer (GQA: 8Q heads, 2KV heads, dim 256)
     │
     ├─ RMSNorm
     └─ LM Head (tied to embedding)
```

Each DeltaNet layer does: RMSNorm → QKV projection → Conv1D → SiLU → L2-normalize → recurrent state update → gated RMSNorm → output projection → residual → FFN.

Each Full Attention layer does: RMSNorm → Q/K/V projections → QK RMSNorm → imRoPE → GQA attention → gating → output projection → residual → FFN.

The quantization mix from the GGUF file:
- **Q6_K**: embedding (197 MB), some FFN down projections
- **Q5_K**: DeltaNet QKV projections, output projections
- **Q4_K**: attention projections, FFN gate/up, gates
- **Q8_0**: DeltaNet alpha/beta projections (tiny)
- **F32**: norms, biases, decay parameters

Total: 320 tensors, 497 MB.

## Phase 0: What We Built

The infrastructure:

1. **GGUF loader** — Memory-maps the file for zero-copy access to quantized weights. Parses all 46 metadata keys and 320 tensor descriptors. Builds a tensor registry for name-based lookup.

2. **Model config** — Extracts all hyperparameters from GGUF metadata. Knows which layers are DeltaNet vs Full Attention.

3. **Weight structs** — Separate `DeltaNetLayerWeights` and `FullAttnLayerWeights` with all tensor references.

4. **GPU memory management** — RAII `CudaBuffer` wrappers, `CudaAllocator` for tracked allocations.

5. **Weight upload** — Copies all 497 MB from mmap'd host memory to GPU. Takes 188ms.

```
$ ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf --info
Loading model: Qwen3.5-0.8B-Q4_K_M.gguf
GGUF parsed in 13 ms
=== GWEN Model Info ===
Layers: 24 (18 DeltaNet + 6 FullAttn)
Layer pattern: DDDADDDADDDADDDADDDADDDA
Total weight size: 497.4 MB
Tensors: 320
```

## What's Next

Phase 1: the kernel arsenal. Every individual operation as a CUDA kernel:
- Q4_K/Q5_K/Q6_K dequantization
- Fused dequant+GEMV for decode (the critical path)
- RMSNorm, SiLU/SwiGLU, softmax
- imRoPE (the exotic one)

The goal: each kernel at >80% of the theoretical memory bandwidth (>717 GB/s). That's where the real fun begins.
