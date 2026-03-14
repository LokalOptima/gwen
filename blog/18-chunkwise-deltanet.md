# 18. Chunkwise DeltaNet: Breaking the Serial Dependency

Post 16 concluded that the DeltaNet bottleneck in `extract_hidden_batch` is **serial token dependency** — each token's S update depends on the previous token. Post 17's nsys profile showed DeltaNet at 62% of batch extraction GPU time. The prescription from the flash-linear-attention (fla) paper: chunkwise decomposition.

## The Algorithm

The DeltaNet recurrence `S_t = decay_t * S_{t-1} + beta_t * k_t * (v_t - S^T k_t)^T` processes tokens one at a time. The chunkwise decomposition (Yang et al. 2024, "Gated Linear Attention Transformers with Hardware-Efficient Training") splits L tokens into chunks of C=64 and restructures the computation:

1. **WY representation**: Convert the within-chunk recurrence into a matrix equation using `T = (I + A)^{-1}` where `A[i,j] = beta[i] * exp(g[i]-g[j]) * K[i]@K[j]` (strictly lower-triangular). Then `W = T @ (beta * exp(g_local) * K)` and `U = T @ (beta * V)` encode all intra-chunk dependencies.
2. **State propagation**: For each chunk sequentially, compute `v_new = U - W @ h` (batch matmul, all tokens at once), then update `h += K^T @ gated_v_new`. Sequential depth drops from L to L/C.
3. **Output**: `o[t] = Q[t] @ h_chunk * exp(g_local) + causal_QK^T @ v_new` — parallel per chunk.

For L=128, C=64: sequential depth goes from 128 to 2. For L=512: 512 to 8.

## Implementation: 4 CUDA Kernels

The old `kernel_deltanet_prefill_batch` (one block per (batch, head), L serial token steps) was replaced by:

| Kernel | Grid | What it does |
|--------|------|-------------|
| `kernel_cumgate_l2norm_batch` | (n_heads, B) | Prefix-sum gate + L2-normalize Q, K in-place |
| `kernel_chunkwise_wy_repr` | (NT*B, n_heads) | Build A matrix, hierarchical triangular solve, compute W and U |
| `kernel_chunkwise_state_propagation` | (4, B*n_heads) | Iterate over chunks: store h, W@h matmul, v_new=U-Wh, decay h, h+=K^T@v_new |
| `kernel_chunkwise_output` | (1, NT*B, n_heads) | QK^T matmul in shared memory, then per-column Q@h + causal_QK^T@v_new |

New InferenceState members: `chunk_gate_cumul`, `chunk_W`, `chunk_U`, `chunk_h_states`, `chunk_v_new` (~221 MB at B=63, L=128).

## Three Algorithm Bugs

The initial implementation compiled and ran without crashes but produced garbage. Finding and fixing three interrelated gating bugs was the hardest part of this work.

### Bug 1: Missing exp(g_local) in W

The ungated WY representation has `W = T @ (beta * K)`. The **gated** variant requires `W = T @ (beta * exp(g_local) * K)`. The exp factor accounts for how much the initial state h has decayed by the time each token "sees" it. U is unchanged because V values are fresh (not accumulated in state).

I missed this because fla's reference code in `chunk.py` passes `g=None` to all kernels — it's the ungated-only orchestrator. The gated variant lives in a separate `gated_delta_rule/` directory.

### Bug 2: Sequential v_new instead of batch

My first state propagation kernel computed v_new token by token, updating h after each:

```cuda
for (int t = 0; t < chunk_len; t++) {
    // compute v_new[t] = U[t] - W[t] @ h
    // update h  <-- WRONG: changes h for next token
}
```

The WY T matrix already encodes all within-chunk sequential dependencies. All tokens in a chunk must use the **same** h state for v_new computation. The correct approach: compute v_new for all tokens first (batch matmul against unchanged h), then update h once.

### Bug 3: Absolute vs chunk-local cumulative gates

State decay used `exp(G_absolute_last)` — total decay from start of sequence. Correct: `exp(G_last - g_offset)` where g_offset tracks the cumulative gate at the end of the previous chunk. Similarly, the output kernel's inter-chunk gating needs `exp(g_t - g_offset)`, not `exp(g_t)`.

All three bugs were caught by writing a numpy reference test (`scratch/test_chunkwise.py`) that compares token-by-token naive recurrence against the chunkwise decomposition at small dimensions (dk=dv=16, C=8). Once the Python matched to machine epsilon, the same fixes applied to CUDA.

## Performance: The Disappointment and the Fix

### First attempt: 40% slower

The initial chunkwise kernels were **491 ms vs 349 ms** for sequential at B=63, L=128. nsys revealed the state propagation kernel at 10.5 ms/layer — nearly as slow as the entire old kernel (11.8 ms/layer).

Root cause: the v_new computation `v_new[t] = U[t] - W[t] @ h` requires a matrix-vector product where W[t] is 128-dim and h is 128x32. My implementation did 32 separate warp reductions per token (one per V-column), with 2 `__syncthreads()` each. For 64 tokens per chunk: **4096 __syncthreads per chunk**. This is a synchronization catastrophe.

### Fix: Shared-memory matmul for W@h

Instead of reducing per-column, treat W@h as a proper [C=64, dk=128] x [dk=128, BV=32] matmul. Load h into shared memory (16 KB), tile W in BK=16 chunks, and have each thread compute a 4x4 tile of the output (TM=4, TN=4, 128 threads = 16x8 thread grid covering the 64x32 output).

Result: state propagation **10.5 ms -> 2.0 ms** per layer (5.2x). Total chunkwise DeltaNet: 19.5 -> 11.1 ms/layer.

### Same fix for the output kernel

The output kernel had the same problem: computing QK^T [64x64] via serial dot product reductions (2016 pairs x 2 syncs = 4032 __syncthreads per chunk). Replaced with a tiled matmul: load Q and K tiles into shared memory, compute the full QK^T matrix in one pass, apply causal mask + gating, then compute output per-column with no reductions.

Output kernel: **5.4 ms -> 3.3 ms** per layer.

## Final Numbers

Per-layer DeltaNet kernel breakdown at B=63, L=128:

| Kernel | Time (ms) | % |
|--------|-----------|---|
| wy_repr (triangular solve + WY) | 3.3 | 37% |
| output (QK^T matmul + Q@h) | 3.3 | 37% |
| state_propagation (W@h + K^T@v_new) | 2.0 | 22% |
| cumgate_l2norm (preprocessing) | 0.3 | 3% |
| **Total** | **8.9** | |

vs old sequential kernel: **11.8 ms/layer -> 8.9 ms/layer (1.33x DeltaNet speedup)**.

Batch extraction throughput:

| Config | Sequential (tok/s) | Chunkwise (tok/s) | Speedup |
|--------|-------------------|-------------------|---------|
| B=2, L=128 | 6,636 | 6,858 | +3% |
| B=10, L=128 | 17,958 | 17,952 | 0% |
| B=32, L=128 | 22,140 | 25,466 | +15% |
| B=63, L=128 | 23,198 | 27,217 | +17% |
| B=63, L=512 | 19,980 | 23,288 | +17% |

Correctness: verified against old sequential kernel at every DeltaNet layer (18 layers). Max error < 2.4e-4, cosine similarity > 0.999999 (7 nines). FP16-level precision.

## Why Only 17%?

The plan estimated 3-6x DeltaNet speedup. We got 1.33x. Three reasons:

1. **L=128 means NT=2 chunks.** The sequential depth drops from 128 to 2, but each "step" is now much heavier — a 64x128x32 matmul instead of a 128-dim dot product. The total FLOP count is similar; we're trading serial-but-cheap for parallel-but-expensive.

2. **The old kernel was already saturating the GPU.** At B=63, the old kernel launches 1008 blocks (63 batches x 16 heads). With 70 SMs, that's 14+ waves — the GPU was already busy. The chunkwise approach adds more parallelism, but the GPU didn't need it.

3. **Intermediate buffer traffic.** The chunkwise path reads/writes W, U, h_states, and v_new through global memory — ~160 MB per layer. The old kernel kept S in L1 cache and only streamed QKV from global. More kernels = more global memory traffic = more bandwidth consumed.

The chunkwise approach wins bigger at longer sequences (more chunks = more sequential depth reduction) and when the DeltaNet kernel is a larger fraction of total time. For the current workload (L=128, DeltaNet at ~55% of pipeline), the 17% end-to-end improvement is the realistic ceiling.

## What I'd Do Differently

**Start with the matmul formulation.** My first instinct was to write the inner loops as explicit token-by-token operations with warp reductions — the same pattern as the old sequential kernel. The chunkwise algorithm is fundamentally about converting recurrences into matrix multiplications. Writing it as anything other than matmuls defeats the purpose. The 4096 __syncthreads per chunk was the direct consequence of not thinking in matmuls.

**Profile the baseline first.** The plan assumed the old kernel was slow because of serial token dependency. It was serial, but it was fast — 1008 blocks, each doing simple L2-cached 128x128 operations. The serial dependency was a latency problem on paper, but at B=63 there was enough batch-level parallelism to hide it. The chunkwise approach helps most when batch parallelism is insufficient (small B, long L).

**The WY kernel is the next bottleneck.** At 3.3 ms/layer (37%), the 64x64 triangular solve dominates. The hierarchical 4x(16x16) block approach should be faster than the current serial forward substitution. But at 37% of an 8.9 ms kernel that's 55% of a 296 ms pipeline, optimizing it further yields diminishing returns — maybe 5% end-to-end at best.

## Runtime Switch

The chunkwise path is the default. Set `GWEN_USE_SEQ_DN=1` to fall back to the old sequential kernel (useful for A/B benchmarking and as a correctness reference).
