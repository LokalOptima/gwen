# Post 31: Eliminating CPU Round-Trips — 520 → 590 tok/s

Posts 29 and 30 built the MTP speculative decode infrastructure and KV cache. Everything was correct, acceptance was 85-98% on natural text. But MTP was 4% *slower* than baseline (520 vs 540 tok/s). nsys profiling revealed why: 22ms of GPU kernel time for 50 tokens, but 149ms wall clock. The 127ms gap was CPU overhead — massive data transfers and sync points that had no business existing.

This post documents the hunt for those round-trips and their elimination.

## The Problem: 2MB Per Cycle for Two Integers

The speculation loop does this every accepted cycle:

```
llama_decode(ctx, 2-token batch)           → GPU compute + cudaStreamSync
llama_get_logits_ith(ctx, 0)               → 248K×4B = 1 MB GPU→CPU, CPU argmax
llama_get_logits_ith(ctx, 1)               → 248K×4B = 1 MB GPU→CPU, CPU argmax  
llama_decode_mtp(ctx, token, pos)           → 4 KB GPU→CPU→GPU (hidden state)
                                            → cudaStreamSync (MTP compute)
                                            → 4 B GPU→CPU (MTP argmax)
```

Every cycle: **~2 MB transferred and 4+ sync points**, 100× per generation. All to extract two integers that could be computed on GPU.

CUDA graphs were already captured and replaying (main model: 1389 nodes, ~36μs per launch; MTP: 52 nodes). The kernels were fast. The problem was entirely in the space between kernel launches — the CPU shuffling megabytes of data across the PCIe bus just to find the index of the largest element.

## Step 1: GPU-Side Argmax for the Main Model

The MTP graph already had `ggml_argmax` appended (post 30). The main model's graph did not. For MTP models, the speculation loop only ever needs the argmax of each position's logits — never the full distribution. So:

```cpp
// In qwen35.cpp build_graph(), after setting res->t_logits:
if (hparams.nextn_predict_layers > 0) {
    ggml_tensor * argmax = ggml_argmax(ctx0, cur);  // [n_outputs] I32
    ggml_set_name(argmax, "verify_argmax");
    res->t_verify_argmax = argmax;
    ggml_build_forward_expand(gf, argmax);
}
```

`ggml_argmax` on `[n_vocab, n_outputs]` produces `[n_outputs] I32`. For a 2-token verify batch: 8 bytes instead of 2 MB.

This adds a few nodes to the CUDA graph but the argmax kernel is ~68μs for 248K elements — invisible compared to the GEMV-dominated decode.

## Step 2: Argmax-Only Decode Mode

Computing the argmax on GPU is only half the fix. The other half is *not transferring the logits*. Inside `decode()`, there's an unconditional async transfer:

```cpp
ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0,
    n_outputs * n_vocab * sizeof(float));  // 2 MB async GPU→CPU
```

This happens every decode call, even when no one will read the logits. With 98 accepted cycles, that's 196 MB of wasted PCIe bandwidth.

The fix: an `argmax_only` mode on `llama_context`. When enabled, `decode()` extracts the GPU-computed argmax (8 bytes) instead of the full logits (2 MB):

```cpp
// In decode(), logits extraction:
if (argmax_only && res->t_verify_argmax && n_outputs > 0) {
    // Fast path: 8 bytes instead of 2 MB
    last_argmax.resize(n_outputs);
    ggml_backend_tensor_get_async(backend_res, res->t_verify_argmax,
        last_argmax.data(), 0, n_outputs * sizeof(int32_t));
} else {
    // Normal path: full logits for non-speculation callers
    ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0,
        n_outputs * n_vocab * sizeof(float));
}
```

New public API:
- `llama_set_argmax_only(ctx, true/false)` — toggle the mode
- `llama_get_argmax_ith(ctx, i)` — read GPU-computed argmax for output position `i`

The speculation loop wraps its decode calls:

```cpp
llama_set_argmax_only(ctx, true);
// ... speculation loop: uses llama_get_argmax_ith() instead of greedy_argmax(llama_get_logits_ith())
llama_set_argmax_only(ctx, false);
```

## Challenge: Fusing Hidden State Copy Into the Main Graph

The hidden state copy (4 KB GPU→CPU→GPU) was the next target. The idea: add a `ggml_cpy` node to the main model's graph that copies the last row of `t_hidden_prenorm` directly into `mtp_hidden_state` — part of the same CUDA graph, zero additional overhead.

```cpp
ggml_tensor * last_row = ggml_view_1d(ctx0, cur, n_embd,
    (n_outputs - 1) * n_embd * ggml_type_size(cur->type));
ggml_tensor * cpy = ggml_cpy(ctx0, ggml_reshape_2d(ctx0, last_row, n_embd, 1),
    mtp_hidden_state);
ggml_build_forward_expand(gf, cpy);
```

This broke MTP correctness. The output diverged on one of the five test prompts.

The root cause: `mtp_hidden_state` is an externally-allocated tensor (part of the MTP buffer, allocated by `mtp_kv_ctx`), but the main model's scheduler (`sched`) manages a different set of backends and graph buffers. When `ggml_cpy` targets a tensor outside the scheduler's allocation scope, the backend mapping becomes unreliable — the scheduler may assign the wrong backend or fail to propagate the write correctly.

**Resolution**: kept the 4 KB GPU→CPU→GPU copy in `decode_mtp()`. At 2μs total (after the main scheduler is already synced by the caller), it's 0.1% of cycle time. Not worth fighting the scheduler abstraction for.

The redundant sync was removed though — `decode_mtp()` previously called `ggml_backend_sched_synchronize(sched.get())` explicitly, but the caller (`llama_get_argmax_ith`) already syncs the scheduler. Eliminating this duplicate sync saves ~50μs per cycle.

## Previous Sessions: Infrastructure That Made This Possible

Several earlier changes (spanning posts 29-30 and uncommitted work) were prerequisites for the argmax optimization:

### Q6_K GET_ROWS CUDA Kernel

The token embedding tensor (`tok_embd`, 199 MiB Q6_K) lived on CPU because ggml's CUDA `GET_ROWS` only supported basic quant types (Q4_0/1, Q5_0/1, Q8_0). For MTP, this split the graph across CPU and GPU backends, preventing CUDA graph capture entirely.

The fix: a hand-written `k_get_rows_q6_K` kernel in `getrows.cu` that dequantizes Q6_K blocks on the GPU. The first attempt used `cudaStreamSynchronize` to copy row indices to the host for pointer lookups — this would have killed CUDA graph capture. The final version reads indices directly on-device:

```cuda
template<typename dst_t>
static __global__ void k_get_rows_q6_K(
        const block_q6_K * src0, const int32_t * src1, dst_t * dst, ...) {
    const int row_idx = src1[i_row * s10];  // read index on GPU
    const block_q6_K * x = (const block_q6_K *)((const char *)src0 + row_idx * nb01) + sb;
    // ... Q6_K dequantization (4 outputs per thread, 64 threads per block)
}
```

With `tok_embd` on GPU, CUDA graphs captured successfully: 1389 nodes for the main model, 52 for MTP.

### Fixed-Topology MTP Graph (Graph Reuse)

MTP graph reuse went from 0% to ~100% through three fixes:

1. **Fixed KV size**: the MTP KV cache was sized to `n_ctx_train = 262144`. Iterating 262K masked positions per cycle was wasteful. Capped at 1024 with unused positions masked to `-inf`.

2. **`ggml_set_rows` for KV writes**: replaced `ggml_view_2d` (offset baked into graph topology, changes every call) with `ggml_set_rows` (write position is a runtime I64 tensor, graph stays fixed).

3. **ubatch initialization**: `memset(&ubatch, 0, sizeof(ubatch))` and `ubatch.b_equal_seqs = 0` — uninitialized fields caused `allow_reuse()` to fail because the `equal_seqs()` branch tried to dereference `ubatch.data == nullptr`.

### MTP GPU-Side Argmax

Added `ggml_argmax(logits)` to the MTP graph output. `decode_mtp()` extracts 4 bytes via `ggml_backend_tensor_get_async` instead of transferring the full 248K×4B = 1 MB logit tensor to CPU. This was the prototype that proved the approach before applying it to the main model.

### DeltaNet S State Snapshotting

The speculation rollback originally used `seq_cp`/`seq_rm` which destabilized the recurrent state head, forcing graph rebuilds (~5ms each). Replaced with dedicated GPU snapshot buffers (38.5 MiB for 36 S/R tensors) and async D2D copies. This kept the recurrent state head stable → graph reuse fired (86-97 reuses per 200 tokens) → verify decode dropped from 7ms to 2.5ms.

## Results

| Version | MTP tok/s | vs Baseline | Accept Rate |
|---------|-----------|-------------|-------------|
| Post 30 (KV cache, no graph opt) | ~470 | break-even | 76% |
| + graph reuse + GPU argmax (MTP) | ~520 | -4% | 85-98% |
| **+ GPU argmax (main) + argmax_only** | **~590** | **+10%** | 85-98% |
| Non-MTP baseline | 536 | — | — |
| gwen with MTP | 672 | +25% | ~80% |

The key transitions:
- **470 → 520**: graph reuse + MTP-side GPU argmax (eliminated 1 MB MTP logits transfer)
- **520 → 590**: main model GPU argmax + argmax_only mode (eliminated 2 MB verify logits transfer)

Correctness: all 5 standard prompts produce bit-identical output to greedy baseline, for both non-MTP and MTP models.

## Remaining Sync Points

Two `cudaStreamSynchronize` calls remain per accepted cycle:

1. **After main decode** — read 8 bytes of argmax to decide accept/reject
2. **After MTP decode** — read 4 bytes of draft token for next cycle

These are structural data dependencies: the accept/reject decision requires the main model's prediction, and the next cycle's verify batch requires the draft token. Each sync is ~50μs — a total of ~100μs per cycle, or ~6% of the ~1.7ms cycle time.

Eliminating these requires conditional execution on the GPU (making the accept/reject comparison and branching happen on-device). That's Step 4 in the optimization roadmap — CUDA dynamic parallelism or graph conditionals, both significantly more complex.

## The Remaining Gap

590 tok/s vs gwen's 672 tok/s leaves a 12% gap. This is no longer CPU overhead — it's the raw kernel speed difference:

- llama's MMVQ uses `vec_dot_q4_k_q8_1` with dynamic dispatch and template-based SwiGLU fusion
- gwen's hand-tuned dp4a GEMV uses static warp counts, explicit block-stride loops, and direct Q4_K unpacking

Both produce correct results but gwen's kernels are ~12% faster at decode-time matrix-vector products. Closing this gap would require optimizing llama's MMVQ kernel in-place — possible, but a different class of work than the infrastructure optimizations documented here.

## Files Changed

| File | Change |
|------|--------|
| `src/llama-graph.h` | `t_verify_argmax` in `llm_graph_result` |
| `src/llama-graph.cpp` | Reset/set_outputs for `t_verify_argmax`, `t_hidden_prenorm`, `t_mtp_argmax` |
| `src/models/qwen35.cpp` | `ggml_argmax` in main model graph (MTP models only) |
| `src/llama-context.h` | `argmax_only` flag, `last_argmax` buffer, `set_argmax_only()`, `get_argmax_ith()` |
| `src/llama-context.cpp` | Argmax-only decode path, `get_argmax_ith()`, removed redundant sync in `decode_mtp()` |
| `include/llama.h` | `llama_set_argmax_only()`, `llama_get_argmax_ith()` |
| `tools/completion/completion.cpp` | Speculation loop uses GPU argmax, cleaned up unused vars |
