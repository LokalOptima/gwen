# Prefill: Going Parallel

*Blog post #4 in the GWEN series — covering Phase 5 prefill optimization*

## The Problem

Up through Phase 4, GWEN processed every prompt token through the same single-token decode path. For a 42-token prompt, that's 42 sequential forward passes — each loading the entire 497 MB model from memory. The CUDA graph helped with launch overhead, but we're still doing 42 × 497 MB = ~21 GB of memory reads just for the prompt. With 896 GB/s bandwidth, that's a minimum of 23ms, and in practice much more due to per-token overhead.

The idea behind prefill is simple: batch the prompt tokens so we load each weight matrix once and process all tokens simultaneously.

## The Architecture Challenge

Qwen3.5-0.8B has a hybrid architecture: 18 DeltaNet layers (linear attention with recurrent state) and 6 full attention layers. This creates a fundamental tension:

- **FFN layers** (all 24): Embarrassingly parallel. Gate, up, and down projections are just matrix multiplications — switch from GEMV to GEMM.
- **Full attention layers** (6): The projections can be batched, but attention itself needs causal masking and KV cache management.
- **DeltaNet layers** (18): The recurrent state update S_t = decay * S_{t-1} + beta * outer(k, v) is inherently sequential per token. You can't compute S_42 without knowing S_41.

## What We Batched

### cuBLAS HGEMM for Matrix Multiply

For seq_len > 1, GEMV (matrix-vector) becomes GEMM (matrix-matrix). The approach: dequantize the quantized weight to FP16 in a scratch buffer, then call cuBLAS `cublasHgemm`.

```cpp
void gwen_gemm(const void* W_quant, GGMLType type,
               half* temp_w, const half* x, half* y,
               int out_features, int in_features, int seq_len,
               cudaStream_t stream) {
    // Dequant to FP16
    gwen_dequant(W_quant, temp_w, out_features * in_features, type, stream);
    // cuBLAS: y[seq,out] = x[seq,in] * W^T
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, seq_len, in_features, ...);
}
```

The dequant-then-GEMM approach is not ideal (we load the full weight into a temp buffer, doubling memory traffic), but it's correct and gives us a working baseline. A fused dequant-GEMM with CUTLASS would be the next optimization.

### The Dequant Bug

When testing GEMM against GEMV, the results didn't match at all — diffs of 0.2-0.8. The root cause: **the dequantization kernels used wrong nibble ordering**.

The ggml Q4_K format stores 4-bit values in an interleaved layout: each group of 64 elements uses 32 bytes, with the first 32 elements in low nibbles and the next 32 in high nibbles:

```
// WRONG (sequential pairs):
q_val = (tid % 2 == 0) ? (qs[tid/2] & 0xF) : (qs[tid/2] >> 4);

// CORRECT (ggml interleaved):
int group = tid / 64;
int is_high = (tid % 64) / 32;
int pos = (tid % 64) % 32;
int byte = group * 32 + pos;
q_val = is_high ? (qs[byte] >> 4) : (qs[byte] & 0xF);
```

The fused GEMV kernels had the correct layout (they were tested against llama.cpp), but the standalone dequant kernels were written independently and got it wrong. This affected Q4_K, Q5_K (which also had wrong qh bit indexing), and Q6_K.

After fixing, GEMV vs GEMM matched within FP16 precision (~2e-3 max diff).

### Batched Operations in Prefill

**Per-layer batch operations** (applied to all N tokens at once):
1. RMSNorm: batch kernel normalizes N vectors
2. Embedding lookup: batch kernel for N token IDs
3. SwiGLU: batch element-wise SiLU(gate) * up
4. Residual add: batch element-wise x += residual

**DeltaNet layers** — batch + sequential hybrid:
1. **Batch**: QKV projection GEMM [N, 1024] → [N, 6144]
2. **Batch**: Gate projection GEMM [N, 1024] → [N, 2048]
3. **Sequential per-token**: Conv1D+SiLU, L2 normalize, gate/beta compute, DeltaNet recurrence, gated RMSNorm
4. **Batch**: Output projection GEMM [N, 2048] → [N, 1024]
5. **Batch**: RMSNorm + FFN (gate, up, down GEMMs + SwiGLU)

**Full attention layers** — same strategy:
1. **Batch**: Q, K, V projection GEMMs
2. **Sequential per-token**: Deinterleave Q+gate, RMSNorm, RoPE, KV cache store, attention, sigmoid-mul
3. **Batch**: Output projection GEMM
4. **Batch**: FFN

## Results

### TTFT (Time to First Token)

| Prompt Length | Sequential | Prefill (FFN only) | Prefill (all projections) |
|---------------|------------|---------------------|---------------------------|
| 4 tokens | ~32 ms | ~32 ms | ~30 ms |
| 42 tokens | ~164 ms* | ~134 ms* | ~77 ms* |

*Prefill time only (excluding first decode token).

### End-to-End (42 prompt + 30 decode tokens)

| Approach | Total Time | Throughput |
|----------|-----------|------------|
| Sequential GEMV | 284 ms | 106 tok/s |
| Prefill (FFN batch) | 254 ms | 118 tok/s |
| Prefill (all batched) | 214 ms | 140 tok/s |

Decode speed remains at 252 tok/s (CUDA graph, unchanged).

### Prefill Throughput

For the 42-token prompt, TTFT of 80ms gives us **523 prompt tok/s** — about 2x the prompt processing speed compared to sequential decode.

## What We Didn't Do (Yet)

### Chunked DeltaNet Parallel Scan

The DeltaNet recurrence is mathematically parallelizable using the "chunkwise" algorithm: split the sequence into chunks, compute intra-chunk outputs as a lower-triangular matrix multiply, then scan across chunk states. This would make the 18 DeltaNet layers truly parallel during prefill.

The implementation is complex — it involves chunk-level attention matrices and parallel prefix scans over state matrices. For short prompts (< 100 tokens), the overhead might not be worth it.

### Flash Attention for Prefill

The 6 full attention layers currently process attention sequentially per-token during prefill. A Flash Attention kernel would compute causal self-attention for all N tokens at once, with O(N) memory instead of O(N^2). With head_dim=256, this needs careful tiling.

### Fused Dequant-GEMM

The current approach dequantizes the full weight matrix to FP16, then runs cuBLAS GEMM. This doubles the memory traffic for weights. A CUTLASS kernel that dequantizes tiles on-the-fly during the GEMM would halve the weight-related bandwidth.

## Key Takeaway

Even with DeltaNet's inherently sequential recurrence, batching the linear projections (which dominate compute time) gave a significant speedup. The 42-token prompt went from 164ms to 77ms — a 2.1x improvement — by turning 7 GEMVs per layer into 7 GEMMs per layer. The DeltaNet state ops (conv1d, l2-normalize, recurrence) are cheap in comparison.
