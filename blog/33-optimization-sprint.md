# Post 33: Optimization Sprint — Three Wins and a Dead End

Post 32 left us at 584 tok/s average with the 50K restricted head, 670 tok/s peak. The profiling showed the MTP draft was only 0.28ms (GPU compute dominated by the LM head GEMV), CPU overhead was negligible, and CUDA graphs were working for the MTP path. Three obstacles were identified: CUDA graph warmup oscillation, redundant state copies, and the surrounding kernel launch overhead around DeltaNet. This post covers chasing all three — and discovering that some "obstacles" were mischaracterized.

## The Redundant 18MB Memcpy Nobody Noticed

The speculative decode loop called `mtp_intermediate_prefill()` before every 2-token verify batch:

```cpp
// --- Speculative path (2-token verify) ---
llama_mtp_intermediate_prefill(ctx);  // copies ALL S and R state
llama_decode(ctx, batch);             // fused kernel overwrites intermediate state
```

This copied the entire DeltaNet S state (18 layers × 128×128 floats = 18 MB) plus conv R state from live buffers to intermediate snapshot buffers. The purpose: on rejection, restore the state to the point after processing only the accepted token.

But the fused DeltaNet kernel (`gated_delta_net_cuda`) already saves intermediate state after token 0 as part of its execution:

```cuda
// Save intermediate S state after token 0 (for MTP 2-token rollback)
if (t == 0 && istate != nullptr) {
    for (int r = 0; r < rows_per_lane; r++) {
        istate[col * S_v + lane + r * warp_size] = s_shard[r];
    }
}
```

And the compute graph copies this into the intermediate buffers via `ggml_cpy`:

```cpp
// qwen35.cpp — inside the graph, runs as part of llama_decode()
ggml_build_forward_expand(gf, ggml_cpy(ctx0, intermediate_s_view, inter_s_target));
```

So the explicit `mtp_intermediate_prefill()` was copying pre-decode state that got immediately overwritten by the graph's post-token-0 state. 18 MB of wasted GPU memcpy, every cycle.

Removing it: **583 → 613 tok/s average (+5.1%), 665 → 700 tok/s peak.** The biggest single win of the session. TCP/UDP went from 497 to 524 tok/s — finally matching baseline instead of being 5% slower.

The lesson: when you have both explicit state management AND in-graph state extraction, check that they're not redundant.

## CUDA Graph Oscillation — Correct Fix, Minimal Impact

The main model's CUDA graph kept resetting warmup:

```
CUDA_GRAPH: warmup complete key=0x...dce0 nodes=1521
CUDA_GRAPH: warmup reset key=0x...dce0 nodes=1413
```

1521 nodes = 2-token batch, 1413 nodes = 1-token batch. The graph key was just `cgraph->nodes[0]` — the first node pointer, identical for both batch sizes. Every batch size change triggered a property mismatch (`props.size() != n_nodes`), resetting warmup.

The fix: incorporate `n_nodes` into the key via fibonacci hashing:

```cpp
static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    uintptr_t key = reinterpret_cast<uintptr_t>(cgraph->nodes[0]);
    key ^= static_cast<uintptr_t>(cgraph->n_nodes) * UINT64_C(0x9e3779b97f4a7c15);
    return reinterpret_cast<const void *>(key);
}
```

Now 1-token and 2-token graphs get separate cache entries. Both warm up independently and stay stable. Confirmed with diagnostic logging: both graphs reach warmup, no resets.

**Impact: ~1% throughput.** The oscillation was real but the cost per reset (~50μs) was small relative to the 2.7ms decode cycle. At 25% reject rate and 100 cycles, that's 25 × 50μs = 1.25ms out of 270ms — less than 0.5%. Still worth fixing since it's architecturally correct and costs nothing.

## Fusing L2 Norm into the DeltaNet Kernel

The nsys profile of the decode loop showed the time breakdown:

| Kernel | % of GPU time | Notes |
|--------|---------------|-------|
| MMVQ (Q4_K/Q5_K/Q6_K) | ~42% | Weight matmuls, bandwidth-bound |
| quantize_q8_1 | 8% | Input quantization before MMVQ |
| rms_norm_f32 | 5% | Per-layer normalization |
| k_get_rows | 5% | Embedding lookups |
| gated_delta_net_cuda | 2% | DeltaNet recurrence |
| l2_norm_f32 | 2% | Q/K normalization for DeltaNet |
| Other (sigmoid, softplus, etc.) | ~5% | Element-wise ops |

DeltaNet itself was only 2% — already efficient. But `l2_norm_f32` at 108 calls (2 per DeltaNet layer × 18 layers × 3 decode calls) was 2% of GPU time in separate kernel launches.

The L2 normalization of Q and K happens right before the DeltaNet state update. Each vector is 128 elements, exactly one warp's worth of data in the DeltaNet kernel (32 threads × 4 elements each). Perfect for fusion.

Added an `L2_NORM` template parameter to `gated_delta_net_cuda`:

```cuda
if constexpr (L2_NORM) {
    float q_sq = 0.0f, k_sq = 0.0f;
    for (int r = 0; r < rows_per_lane; r++) {
        q_sq += q_reg[r] * q_reg[r];
        k_sq += k_reg[r] * k_reg[r];
    }
    float q_sum = warp_reduce_sum<warp_size>(q_sq);
    float k_sum = warp_reduce_sum<warp_size>(k_sq);
    float q_inv = rsqrtf(fmaxf(q_sum, l2_eps * l2_eps));
    float k_inv = rsqrtf(fmaxf(k_sum, l2_eps * l2_eps));
    for (int r = 0; r < rows_per_lane; r++) {
        q_reg[r] *= q_inv;
        k_reg[r] *= k_inv;
    }
}
```

Two warp reductions, two rsqrtf, and element-wise scaling — all using data already in registers. Zero extra memory traffic.

The plumbing: `ggml_gated_delta_net_l2()` stores `l2_norm_eps` in `op_params`. The CUDA dispatch reads it and selects the `L2_NORM=true` template. The model graph skips separate `ggml_l2_norm()` calls when using the fused path.

**Impact: 613 → 625 tok/s average (+2.0%), 700 → 711 tok/s peak.** The baseline also improved (+1.5%) since the fusion applies to single-token decode too.

## The GEMM False Alarm

The initial nsys profile appeared to show 2-token batch using `mul_mat_q` (GEMM) instead of `mul_mat_vec_q` (MMVQ):

```
8.4%  mul_mat_q<Q4_K, 80>     114 calls
6.7%  mul_mat_q_stream_k_fixup 114 calls
```

This looked like the 2-token speculative verify was falling through to slow GEMM instead of the fast MMVQ path. 15% of GPU time apparently wasted on the wrong dispatch.

After adding debug prints to the dispatch, the truth:

```
MUL_MAT Q4_K: src1=[1024,2,1,1] ne1=2 mmvq=1 mmq=1 bad_pad=0 name=blk.0.ssm_alpha.weight
```

`mmvq=1` — the dispatch IS selecting MMVQ for all Q4_K with ne1=2. The `mul_mat_q` calls in nsys were from **prefill** (ne1=14 tokens for the prompt, well above MMVQ_MAX_BATCH_SIZE=8).

The profile was mixing prefill and decode kernel stats. A decode-only profile (with `--no-warmup` and a 1-token prompt) confirmed: `mul_mat_vec_q<Q4_K, 2>` appears at 9.2% — working correctly.

## Restricted LM Head Size Sweep

With the optimizations in place, ran a comprehensive sweep of restricted head sizes. Built heads at K = 4K, 10K, 20K, 30K, 40K, 50K, 60K, 70K, 80K, 100K. Tested each on all 12 prompts at 200 tokens.

| K | Coverage | Draft (ms) | Accept% | Avg tok/s | Peak tok/s |
|---|----------|------------|---------|-----------|------------|
| full 248K | 100% | 0.59 | 76.3% | 571 | 641 |
| 4,000 | 84.7% | 0.22 | 63.3% | 600 | 727 |
| 10,000 | 91.7% | 0.24 | 67.2% | 605 | 717 |
| 20,000 | 95.9% | 0.25 | 70.2% | 611 | 704 |
| 30,000 | 97.9% | 0.27 | 71.7% | 608 | 706 |
| 40,000 | 99.0% | 0.28 | 72.3% | 618 | 712 |
| **50,000** | **99.1%** | **0.29** | **73.3%** | **619** | **712** |
| 60,000 | 99.8% | 0.30 | 73.6% | 618 | 704 |
| 70,000 | 99.9% | 0.32 | 74.1% | 616 | 705 |
| 80,000 | 99.9% | 0.33 | 74.3% | 615 | 696 |
| 100,000 | 100% | 0.36 | 75.7% | 614 | 691 |

The throughput formula: `E[tokens] / cycle_time` where `E[tokens] = p + 1` and `cycle_time = main_decode + draft_time`. Acceptance rate `p(K)` increases with K but with diminishing returns (84.7% → 99.1% coverage from 4K → 50K), while draft time `d(K)` grows roughly linearly with K.

The peak is at **40K-50K**. Above 50K, every additional token in the vocabulary costs more draft time than the acceptance rate gains justify. Below 30K, acceptance drops faster than draft time savings can compensate. The 4K head is fastest on repetitive text (727 tok/s peak) but 3% slower on average due to 10pp lower acceptance.

50K confirmed as the right default.

## Current State

```
Base model tg64 (llama-bench):     540 ± 4 tok/s
MTP model tg64 (no speculation):   523 ± 4 tok/s
MTP speculative (50K, 12 avg):     625 tok/s (+16% over baseline)
MTP speculative (50K, best):       712 tok/s (Counting, 100% accept)
MTP speculative (50K, worst):      533 tok/s (TCP/UDP, 47% accept)
```

Cumulative improvement this session: 583 → 625 tok/s average (+7.2%), 665 → 711 tok/s peak (+6.9%).

## What's Next

The nsys profile tells us where the remaining time goes: MMVQ weight matmuls at 42% are bandwidth-bound and hard to improve without hardware changes. The path to 1000 tok/s requires multi-token speculation (3 tokens per cycle), which needs extending the DeltaNet S state snapshot for 2 rollback points. The remaining fusible element-wise ops (sigmoid, softplus, add, mul for gate/beta) account for ~4% of GPU time — worth fusing but not game-changing.

## Files Changed

| File | Change |
|------|--------|
| `tools/completion/completion.cpp` | Removed `mtp_intermediate_prefill()` |
| `src/llama-context.cpp` | Default 50K LM head path |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Graph key includes n_nodes |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | L2_NORM template, fused normalization |
| `ggml/src/ggml.c` | `ggml_gated_delta_net_l2()` |
| `ggml/include/ggml.h` | `ggml_gated_delta_net_l2()` declaration |
| `src/models/delta-net-base.cpp` | `l2_norm_eps` parameter threading |
| `src/models/models.h` | Updated function signatures |
| `src/models/qwen35.cpp` | Skip L2 norm when fused GDN active |
| `scripts/bench_lm_head_sizes.sh` | New: comprehensive head size benchmark |
