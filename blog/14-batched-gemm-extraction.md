# 14. Batched GEMM Hidden State Extraction

The MTP fine-tuning pipeline needs hidden states from the base model for every token in the training corpus. The original server used a thread pool where each worker ran an independent forward pass — N workers reading the 493 MB model weights independently meant N × 493 MB of DRAM reads. For B=64, that's 31 GB of redundant weight traffic.

The fix is obvious in hindsight: treat all B×L tokens as a single matrix and run one GEMM per weight matrix, reading each weight once.

## Architecture

The batch extraction path (`/batch_extract`) processes B independent sequences of length L:

1. **Batch embedding lookup** for all B×L tokens
2. **24-layer loop**, each layer doing:
   - Batch RMSNorm → CUTLASS GEMM (one weight read for all tokens)
   - Layer-specific: DeltaNet recurrence (B independent states) or causal attention
   - Batch FFN: gate+up GEMM → SwiGLU → down GEMM
3. **D2H copy** of final hidden states [B×L, 1024]

Each sequence gets its own DeltaNet S matrix (128×128 FP32) and conv1d state, allocated contiguously in a batch buffer. The causal attention uses a flash-attention-style kernel with online softmax — one warp per (batch, head) pair, no materialized score matrix, no KV cache.

## New CUDA Kernels

Six batch kernels were added:

- `kernel_batch_conv1d_silu_multi` — B independent conv states, grid (ceil(dim/256), B)
- `kernel_deltanet_prefill_batch` — n_heads×B blocks, each processes L tokens sequentially through 128×128 S
- `kernel_rope_batch` — position = token_idx % L, automatic per-sequence positions
- `kernel_deinterleave_qgate_batch` — split interleaved Q+gate for all B×L tokens
- `kernel_batch_causal_attn` — 32-thread warp per (batch, head), online softmax, O(1) extra memory per block
- `kernel_sigmoid_mul_batch` — element-wise sigmoid gate for attention output

## Correctness

The `/compare_extract` endpoint runs both the GEMV (single-sequence, reference) and GEMM (batched) paths on identical input and reports bit-level mismatches. Result: **zero mismatches** across L=5, 64, 256, 512, and multi-sequence batches. The batched path produces bit-identical output to the reference.

## Profiling

CUDA event timing with `GWEN_PROFILE_BATCH=1` reveals the bottleneck breakdown for B=64, L=512 (32K tokens, 1.6s total):

| Component | Time (ms) | % |
|-----------|-----------|---|
| DeltaNet recurrence | 884 | 54% |
| GEMM (dequant+mm) | 353 | 22% |
| Causal attention | 300 | 18% |
| Elemwise (norm/add/SiLU) | 63 | 4% |
| Conv1d+SiLU | 24 | 1.5% |

The DeltaNet kernel dominates because S (64 KB per head×batch) thrashes L2 cache when 1024 blocks compete for 48 MB L2. The causal attention is O(L²) per head — reads K/V from global memory for every (q,k) pair.

## Optimization Attempts

### Shared memory S (1.23× on DeltaNet)

Loaded S into 64 KB dynamic shared memory, processed all L tokens in-place, wrote back once. Result: 884 ms → 718 ms (19% improvement on DeltaNet). But 65 KB shared limits to 1 block/SM = 4 warps, too low to hide the ~20-cycle shared memory latency.

### Register-tiled S (failed — 1.5× slower)

Each thread keeps its column (128 floats) in a register array. In theory: zero-latency access, no shared memory bottleneck. In practice: 128 registers per thread causes massive spill to local memory. The compiler can't keep all 128 floats in registers simultaneously when the inner loop is unrolled. Result: 884 ms → 1311 ms. Lesson: register arrays > ~32 elements will spill on SM_120.

### What didn't matter

Pre-dequantizing weights to FP16 and caching them would save ~2.7 GB of traffic (dequant writes + reads) per batch, but at 896 GB/s that's only ~3 ms — negligible vs the 1.6s total.

## Performance

Throughput at various batch sizes (B×L = total tokens):

| Config | Tokens | Time (ms) | Throughput |
|--------|--------|-----------|------------|
| 4×128 | 512 | 42 | 12K tok/s |
| 16×256 | 4,096 | 236 | 17K tok/s |
| 64×128 | 8,192 | 350 | 23K tok/s |
| 32×512 | 16,384 | 954 | 17K tok/s |
| 64×512 | 32,768 | 1,633 | 20K tok/s |

The L² scaling of causal attention is visible: same 8K tokens take 350 ms at L=128 but 572 ms at L=512. For training at seq_len=512, the 6 attention layers account for 18% of runtime despite being only 25% of layers.

## Remaining Optimization Targets

1. **DeltaNet S access pattern (54%)**: The 128×128 S matrix per (head, batch) pair is the bottleneck. Need a tiling strategy that balances shared memory capacity with occupancy.
2. **Causal attention (18%)**: Current single-warp kernel loads K/V from global for every query-key pair. Shared-memory K/V tiling (true FlashAttention) would amortize global reads.
3. **Fused FFN gate+up (saves 24 of 150 GEMMs)**: Both have the same input — fusing into one GEMM halves input reads.
