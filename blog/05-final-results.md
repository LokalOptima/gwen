# Final Results and Lessons Learned

*Blog post #5 in the GWEN series — Phase 6 wrap-up*

## The Journey

GWEN started as a from-scratch CUDA implementation of Qwen3.5-0.8B inference, targeting the RTX 5070 Ti. Over five phases we:

1. Built a GGUF parser, GPU memory allocator, and tokenizer
2. Wrote hand-optimized CUDA kernels for Q4_K/Q5_K/Q6_K dequant+GEMV, RMSNorm, RoPE, softmax
3. Implemented the Gated DeltaNet recurrence (the novel component of this hybrid model)
4. Assembled the full 24-layer model with correct output
5. Optimized decode with kernel fusion and CUDA graphs (184 → 252 tok/s)
6. Added batched prefill with cuBLAS GEMM

## Final Benchmark Numbers

### Decode Throughput (Single Token, Steady State)

| | GWEN | llama.cpp | Ratio |
|------|------|-----------|-------|
| **Decode tok/s** | **252** | **443** | **0.57x** |
| ms per token | 3.98 | 2.26 | 1.76x |
| Bandwidth util | 14.6% | ~25%* | |

*Estimated from llama.cpp's timing.

### Prefill (Time to First Token)

| Prompt Length | GWEN TTFT | llama.cpp TTFT | GWEN tok/s | llama.cpp tok/s |
|---------------|-----------|----------------|------------|-----------------|
| 4 tokens | 30 ms | ~30 ms* | 132 | 172 |
| 42 tokens | 81 ms | 99 ms | 520 | 554 |
| 118 tokens | 177 ms | 117 ms | 669 | 1123 |

*llama.cpp reports 17 tokens for "The quick brown fox" due to different tokenization + thinking tokens.

### End-to-End Generation (prompt + 50 decode tokens)

| Prompt | GWEN | llama.cpp |
|--------|------|-----------|
| 4-token prompt | 231 ms (217 tok/s) | ~162 ms (308 tok/s) |
| 42-token prompt | 284 ms (176 tok/s) | ~169 ms (296 tok/s) |
| 118-token prompt | 404 ms (124 tok/s) | ~187 ms (267 tok/s) |

### Memory Usage

| | GWEN | llama.cpp |
|---|------|-----------|
| Model weights | 497 MB | 497 MB |
| Total GPU alloc | 750 MB | ~515 MB |
| Peak VRAM | ~800 MB | ~560 MB |

GWEN uses more memory due to prefill batch buffers and the dequant scratch for GEMM.

## Where GWEN Wins

1. **Short-prompt TTFT**: For 42-token prompts, GWEN's TTFT (81ms) beats llama.cpp (99ms). This is because GWEN's batched GEMM prefill is efficient for moderate batch sizes, while llama.cpp has more overhead in its general-purpose scheduling.

2. **Simplicity**: GWEN is ~3K lines of CUDA + ~1K lines of infrastructure. llama.cpp's Qwen3.5 backend involves tens of thousands of lines across multiple abstraction layers.

3. **Single-GPU, single-model**: No runtime dispatch, no backend selection, no dynamic shape handling. Everything is statically optimized for this exact model.

## Where llama.cpp Wins

1. **Decode throughput (1.76x faster)**: llama.cpp's GEMV kernels are significantly more optimized:
   - Multi-row processing (multiple output elements per thread block)
   - Better register utilization and memory access patterns
   - Fused dequant kernels with CUDA graph support built-in
   - GWEN's GEMV achieves only 14.6% bandwidth efficiency vs llama.cpp's ~25%

2. **Long-prompt prefill (1.68x faster at 118 tokens)**: llama.cpp has:
   - Flash Attention for the full attention layers
   - More optimized GEMM kernels (cuBLAS vs our dequant+cuBLAS)
   - Better memory management

3. **Memory efficiency**: llama.cpp allocates more precisely and doesn't need large prefill scratch buffers.

## Optimization Timeline

| Phase | Decode tok/s | Key Change |
|-------|-------------|------------|
| Phase 2-3 (baseline) | 175 | First working model |
| Phase 4.1 | 184 | Remove debug overhead, allocator fixes |
| Phase 4.2 | 215 | Kernel fusion (QKV aliasing, batched RMSNorm, Q scaling) |
| Phase 4.3 | 217 | Fused Conv1D+SiLU, sigmoid_mul, async KV cache |
| Phase 4.4 | **252** | CUDA Graph capture for full forward pass |

Prefill: 284ms → 214ms for 42-token prompt + 30 decode tokens (batched GEMM).

## Architecture Insights

### DeltaNet is Decode-Friendly, Prefill-Hostile

The linear attention mechanism in DeltaNet gives O(1) memory per step — no KV cache growing with sequence length. For decode, this is fantastic: the recurrent state update is just a few matrix operations per head.

But for prefill, the sequential nature of the recurrence is a major limitation. While the 6 full attention layers can be parallelized with Flash Attention, the 18 DeltaNet layers must process each token sequentially through the recurrence. The projections can be batched (and we do), but the core state update cannot (without the chunkwise parallel scan algorithm, which we didn't implement).

### CUDA Graphs Were the Biggest Single Win

Going from 217 to 252 tok/s (+16%) with CUDA graphs was the largest single optimization. With ~419 kernel launches per forward pass, the launch overhead was ~2ms out of 4.6ms total — almost half the runtime was just scheduling work, not doing it.

The key trick was device-side indirection: storing position and token ID in device memory so the graph structure stays constant while the data changes.

### The 48MB L2 Cache Is Mostly Wasted

The RTX 5070 Ti's 48MB L2 cache can theoretically hold ~10% of the 497MB model. But with CUDA Graph replay, kernels execute in a fixed order, loading each weight matrix exactly once per forward pass. There's no temporal locality to exploit — by the time we revisit a layer's weights, the previous layer's weights have long been evicted.

A persistent kernel approach (where a single kernel processes multiple layers, keeping frequently-accessed small tensors in L2) could exploit this, but it would require a radically different architecture.

## What I'd Do Differently

1. **Start with GEMV optimization**: The GEMV kernels are the critical path for decode. Getting them to 25%+ bandwidth utilization early would have compounded through all subsequent optimizations.

2. **Multi-row GEMV from the start**: Processing 2-4 output rows per thread block would significantly improve bandwidth utilization by amortizing the input vector load across more computation.

3. **Fused dequant-GEMM for prefill**: The current dequant → FP16 temp → cuBLAS approach doubles memory traffic for weights. A CUTLASS kernel that dequantizes tiles on-the-fly would halve this.

4. **Profile first, optimize second**: I should have profiled with Nsight Compute from Phase 1 instead of waiting until Phase 4. Understanding the roofline model early would have guided better kernel designs.

## Code Statistics

| Component | Files | Lines of Code |
|-----------|-------|---------------|
| Kernels (CUDA) | 7 | ~1100 |
| Inference engine | 1 | ~1150 |
| GGUF parser | 1 | ~350 |
| Model loading | 1 | ~300 |
| Tokenizer | 1 | ~200 |
| Memory allocator | 1 | ~80 |
| Main + tests | 4 | ~450 |
| **Total** | **16** | **~3600** |

## Conclusion

GWEN achieves 252 tok/s decode on an RTX 5070 Ti — 57% of llama.cpp's throughput, in about 3600 lines of purpose-built CUDA code. The gap to llama.cpp is primarily in GEMV kernel optimization (bandwidth utilization), which is the core bottleneck for single-token decode of quantized models.

The hybrid DeltaNet + Transformer architecture of Qwen3.5 is an interesting optimization target. DeltaNet's O(1) state makes decode efficient (no growing KV cache for 18 of 24 layers), but its sequential nature limits prefill parallelism. The full attention layers (with their KV caches) provide long-range context at the cost of O(n) memory.

Building a GPU inference engine from scratch taught me more about GPU programming than any amount of reading could. The gap between "it works" and "it's fast" is enormous, and most of that gap lives in memory access patterns, kernel launch overhead, and the subtle art of keeping the GPU fed with data.
