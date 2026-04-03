# Post 30: MTP KV Cache — 50% to 76% Acceptance

Post 29 built the MTP speculative decode infrastructure in llama-slim: weight loading, graph builder, `llama_decode_mtp()` API, accept/reject/rollback loop. Everything worked correctly, but acceptance was only 50% — too low for the replay rollback approach to break even.

The problem: the MTP attention layer had no KV cache. With a single token input (n_tokens=1), the no-cache attention is a pass-through — Q attending to just its own K. The MTP head couldn't see any history from previous drafts.

This post adds a standalone KV cache for the MTP attention layer, raising acceptance from 50% to 76%.

## Why Separate From llama's KV Cache?

The MTP attention layer (block 24) produces its own K/V tensors. They come from the MTP's internal representation:

```
x = FC(concat(RMSNorm(embed(token)), RMSNorm(hidden)))
x = attn_norm(x)
Q, K, V = projections(x)   ← these K/V are unique to MTP
```

These are completely different from the main model's K/V at the same position. The MTP head builds up its own context over successive evaluations.

We already allocated a KV cache slot for layer 24 in llama's `llama_kv_cache` (via the `n_layer_total()` extension), but *using* it requires going through `init_batch()` → `find_slot()` → `llama_kv_cache_context` — all tightly coupled to the main decode loop. The MTP runs at a different cadence (once between main decodes), and the positions would conflict.

Standalone GPU buffers with a simple incrementing counter are far simpler.

## Implementation

### Allocation

Two persistent F16 tensors on GPU, sized to the full context length:

```cpp
// In llama_context constructor:
mtp_k_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd_k_gqa, n_ctx);  // [512, n_ctx]
mtp_v_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd_v_gqa, n_ctx);  // [512, n_ctx]
```

For Qwen3.5-0.8B with n_ctx=512: 2 x 512 x 512 x 2 bytes = 1 MiB. Negligible.

### Graph Builder

The new `build_layer_attn_mtp()` method replaces `build_layer_attn_no_cache()` in the MTP graph:

1. **Compute Q, K_new, V_new** — same projections as before (Q norm, K norm, RoPE, gate)
2. **Write to cache** — `ggml_cpy` from F32 K/V into F16 cache at `mtp_kv_pos`:
   ```cpp
   ggml_tensor * k_slot = ggml_view_2d(ctx0, mtp_k_cache,
       n_embd_k_gqa, 1, mtp_k_cache->nb[1],
       mtp_kv_pos * mtp_k_cache->nb[1]);
   ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_2d, k_slot));
   ```
3. **Read full cache** — `ggml_view_3d` to reshape [512, n_kv] into [256, 2, n_kv]:
   ```cpp
   ggml_tensor * K_full = ggml_view_3d(ctx0, mtp_k_cache,
       n_embd_head, n_head_kv, n_kv, ...);
   ```
4. **Attend** — `build_attn_mha(Q, K_full, V_full, mask)` with a simple all-zeros mask (current token can attend to all previous MTP tokens)
5. **Gate + output projection** — unchanged

### No Rollback Needed

MTP is only called with accepted tokens. The sequence is:

```
1. Main decode → sample accepted token
2. decode_mtp(accepted) → draft token     ← writes to MTP KV cache
3. Verify [accepted, draft] as 2-token batch
4. Accept or reject
```

If the draft is rejected, we roll back the *main model's* state (KV cache + recurrent state), but the MTP KV entry for `accepted` is still valid — it was the correct token. The next `decode_mtp()` writes to the next position. The MTP KV cache grows monotonically.

### Mask

The mask is trivially simple: `[n_kv, 1, 1, 1]` filled with 0.0f. The current MTP token can attend to all previous entries. No causal masking complexity — every previous MTP evaluation was at an earlier position.

A new `llm_graph_input_mtp_kv` class handles creating and filling this mask via the standard `set_input()` interface. It supports both flash attention (F16 cast) and non-flash paths.

## Results

All 5 standard prompts produce bit-identical output to greedy baseline:

```
=== Prompt: "The quick brown fox" ===
  Accepted: 91  Rejected: 17  Rate: 84.3%  Match: YES

=== Prompt: "In the beginning there was nothing but" ===
  Accepted: 82  Rejected: 35  Rate: 70.1%  Match: YES

=== Prompt: "def fibonacci(n):" ===
  Accepted: 21  Rejected: 13  Rate: 61.8%  Match: YES

=== Prompt: "The capital of France is" ===
  Accepted: 82  Rejected: 24  Rate: 77.4%  Match: YES

=== Prompt: "1 + 1 = 2. 2 + 2 =" ===
  Accepted: 86  Rejected: 26  Rate: 76.8%  Match: YES

=== SUMMARY ===
All outputs match: YES
Total accepted: 362  rejected: 115  rate: 75.9%
```

| Metric | Without KV cache | With KV cache |
|--------|:----------------:|:-------------:|
| Acceptance rate | 50.4% | **75.9%** |
| Best prompt | ~55% | **84.3%** |
| Decode speed | 535 tok/s | 528 tok/s |
| Correctness | all match | all match |

## Break-Even Analysis

From the pp2 benchmark in post 29:
- 2-token batch: 2.43ms (1.3x single-token cost, not 2x — weight reads dominate)
- Single-token decode: 1.87ms
- Accept cost: 2.43ms (verify batch)
- Reject cost: 2.43ms + 1.87ms = 4.30ms (verify + replay)

Expected throughput at 76% acceptance:
- Average cost per main-model token: 0.76 x 2.43 + 0.24 x 4.30 = 2.89ms
- Tokens produced per cycle: 0.76 x 2 + 0.24 x 1 = 1.76
- Effective ms/token: 2.89 / 1.76 = 1.64ms → **~610 tok/s**
- (Plus MTP overhead per cycle — need to measure)

This is a rough estimate. The actual speedup depends on the MTP forward pass cost. Integrating into the completion tool will give us real numbers.

## Test Bug Fix

The accept path in `test_mtp_speculative.cpp` pushes 2 tokens at once (draft + prediction after draft). At higher acceptance rates, this can overshoot `n_gen` by 1, causing length mismatches that look like correctness failures. Fixed with a simple trim before returning.

## What's Next

1. **Integrate into completion tool** — port the speculative loop to `tools/completion/completion.cpp` and measure real tok/s
2. **DeltaNet S matrix snapshotting** — replace replay rollback (1.87ms) with kernel-level snapshot/restore (~0.04ms), dropping break-even to ~25%
