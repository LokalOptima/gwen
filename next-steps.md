# Next Steps: MTP Speculative Decode Optimization

## Current State (after intermediate S state optimization)

MTP speculation is now **+15.7% faster than baseline** at 75% acceptance. Previously it was 20% slower.

### Benchmark (12 diverse prompts × 200 tokens, all bit-identical to baseline)

| Prompt | Accept% | tok/s (MTP) | tok/s (base) |
|--------|---------|-------------|--------------|
| AI history | 66% | 289.0 | 242.7 |
| Narrative | 73% | 282.0 | 257.4 |
| Business | 65% | 292.8 | 262.1 |
| TCP/UDP | 50% | 277.7 | 259.0 |
| Quantum | 71% | 289.8 | 262.1 |
| Transformer | 56% | 289.0 | 254.1 |
| Python sort | 86% | 317.9 | 257.0 |
| Python class | 85% | 306.7 | 257.4 |
| Math word | 85% | 305.8 | 258.0 |
| Fibonacci | 95% | 305.8 | 263.1 |
| Fox repeat | 77% | 299.4 | 255.1 |
| Counting | 100% | 315.4 | 255.1 |
| **Average** | **75%** | **297.1** | **256.8** |

**Every prompt is faster than baseline.** Even TCP/UDP at 50% acceptance is faster (277.7 vs 259.0).

### What was fixed

**Before (reject cost 5.06ms)**:
1. 2-token decode: 2.4ms
2. Snapshot restore: 0.04ms
3. **Re-decode accepted: 1.87ms** ← eliminated
4. MTP draft: 0.75ms

**After (reject cost 3.23ms)**:
1. 2-token decode: 2.4ms
2. Intermediate prefill: 0.04ms
3. Intermediate restore: 0.04ms
4. MTP draft: 0.75ms

### How intermediate state works

The DeltaNet CUDA kernel processes tokens sequentially in a loop. For a 2-token batch, after updating `s_shard[]` with token 0 (the accepted token), the kernel writes the intermediate state to the output buffer BEFORE processing token 1 (the draft). The ggml output tensor was extended to `[attention | final_state | intermediate_state]` when n_tokens >= 2.

The graph builder in `build_layer_attn_linear()` extracts the intermediate S state from the kernel output and the intermediate R (conv) state from the conv_input tensor, copying both to persistent GPU buffers via `ggml_cpy`.

On rejection, instead of restoring a pre-decode snapshot and re-decoding the accepted token (1.87ms), we just copy the intermediate buffers back to the current state (0.04ms).

A key subtlety: the recurrent memory cell's `pos` field stores the LAST token's position. After a 2-token decode at [n, n+1], `cell.pos = n+1`. Calling `seq_rm(0, n+1, n+2)` would match and CLEAR the entire cell. The fix: patch `cell.pos` from n+1 → n before calling seq_rm.

### Restricted LM head (50K vocab) — not helpful

| Config | tok/s | Accept% |
|--------|-------|---------|
| Full vocab (248K) | 297.1 | 75% |
| Restricted (50K) | 292.0 | 72% |

The 50K restricted head reduces acceptance because tokens outside its vocabulary can never be correctly predicted. The acceptance rate drop (75% → 72%) outweighs any speed improvement from the smaller matmul. Full vocab is strictly better.

## Remaining Optimizations

### Priority 1: Reduce MTP draft overhead (0.75ms → <0.3ms)

The MTP draft (running the lightweight MTP head) takes 0.75ms per speculation cycle. This is 23% of the accept-path cost.

**CUDA graph caching**: The MTP graph topology is fixed (always 1 token input, same layer structure). Cache the compiled CUDA graph and replay it, eliminating launch overhead (~2000 launches × ~5μs each).

Expected impact: 0.75ms → ~0.3ms. At 75% acceptance:
- Accept: (2.4 + 0.04 + 0.3)/2 = 1.37ms/tok
- Reject: 2.4 + 0.04 + 0.04 + 0.3 = 2.78ms/tok
- Weighted: 0.75 × 1.37 + 0.25 × 2.78 = **1.72ms/tok** → +15% over current MTP, +30% over baseline

### Priority 2: Async MTP on separate CUDA stream

Overlap the MTP draft prediction with the next main decode. The MTP head runs on a separate CUDA stream while the main model processes the next speculation cycle. This hides the MTP latency entirely.

Expected impact: MTP draft cost → effectively 0ms. At 75% acceptance:
- Accept: (2.4 + 0.04)/2 = 1.22ms/tok
- Reject: 2.4 + 0.04 + 0.04 = 2.48ms/tok
- Weighted: 0.75 × 1.22 + 0.25 × 2.48 = **1.54ms/tok** → +35% over baseline

### Priority 3: Profile absolute decode speed

Our measurements show ~257 tok/s baseline while previous `llama-bench` showed ~534 tok/s. The difference is measurement methodology (end-to-end with prefill vs decode-only). But gwen achieved 672 tok/s with MTP. Understanding the absolute speed gap would identify further optimization opportunities.

## Files

| File | State |
|------|-------|
| `ggml/src/ggml.c` | modified — GDN output extended for intermediate state |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | modified — kernel writes intermediate S |
| `src/llama-graph.h` | modified — intermediate buffer pointers |
| `src/llama-graph.cpp` | modified — initialize intermediate fields |
| `src/llama-context.h` | modified — intermediate_t struct, new methods |
| `src/llama-context.cpp` | modified — allocate buffers, prefill/restore/reject_fixup, decode_mtp hidden_idx |
| `src/models/qwen35.cpp` | modified — graph copies intermediate S/R |
| `include/llama.h` | modified — public API additions |
| `tools/completion/completion.cpp` | modified — rewritten reject path |
