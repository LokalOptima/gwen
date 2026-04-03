# Next Steps: Eliminate CPU Round-Trips in MTP Speculation Loop

## Where We Are

MTP speculative decode is **correct and nearly break-even** with baseline:

| Prompt | MTP tok/s | Baseline | Accept% |
|--------|-----------|----------|---------|
| brown fox | 520 | 541 | 96.1% |
| math | 526 | 542 | 98.0% |

CUDA graphs ARE capturing and replaying for both the main model (1389 nodes) and MTP (52 nodes). Graph reuse works (1 rebuild, rest reused). All tensors are on GPU (tok_embd moved from CPU via Q6_K get_rows kernel we added).

## The Problem: CPU Round-Trips

nsys profiling shows 22ms of GPU kernel time for 50 tokens but 149ms wall clock. The **127ms gap is CPU overhead** — sync points and unnecessary GPU→CPU transfers in the speculation loop.

Per accepted speculation cycle, the current code does:

```
llama_decode(ctx, batch_2tok)     → cudaStreamSync (wait for verify)
llama_get_logits_ith(ctx, 0)      → cudaMemcpy 248K×4B = 1 MB GPU→CPU
greedy_argmax(logits, n_vocab)    → CPU argmax (scans 1MB)
llama_get_logits_ith(ctx, 1)      → cudaMemcpy 248K×4B = 1 MB GPU→CPU  
greedy_argmax(logits, n_vocab)    → CPU argmax (scans 1MB)
llama_decode_mtp(ctx, token, pos) → cudaMemcpy 4KB GPU→CPU→GPU (hidden state)
                                  → cudaStreamSync (wait for MTP compute)
                                  → cudaMemcpy 4B GPU→CPU (argmax result)
```

That's **~2 MB of transfers and 4+ sync points per cycle**, 100× per generation. All to extract two integers.

## The Fix: Keep It On GPU

### Step 1: GPU-side argmax for main model verify logits

The main model already produces logits on GPU. Instead of transferring 2MB to CPU for argmax, add `ggml_argmax` to the graph output.

**For the 2-token verify batch**, the graph should output:
- `argmax(logits[0])` → token predicted for position 0 (the "pred" that verifies draft)
- `argmax(logits[1])` → token predicted for position 1 (the "pred_after_draft")

These are two I32 values — 8 bytes total instead of 2MB.

**Implementation:**
- Add a flag/graph type for "decode with argmax output" (e.g., `LLM_GRAPH_TYPE_VERIFY` or a cparams flag)
- In the graph builder (qwen35.cpp `build_graph`), when this flag is set, append `ggml_argmax(t_logits_0)` and `ggml_argmax(t_logits_1)` to the graph
- Store in `llm_graph_result::t_verify_argmax[2]`
- New API: `llama_decode_verify(ctx, batch_2tok)` that returns the two argmax values directly, or store them like `mtp_last_argmax`

### Step 2: GPU→GPU hidden state copy

Currently `decode_mtp()` does:
```cpp
ggml_backend_tensor_get(t_hidden_prenorm, buf, offset, 4KB);  // GPU→CPU
ggml_backend_tensor_set(mtp_hidden_state, buf, 0, 4KB);       // CPU→GPU
```

Replace with `ggml_backend_tensor_copy_async()` for D2D on the same CUDA stream. But there's a subtlety: `t_hidden_prenorm` is a tensor from the main graph result (could be a view with an offset for the last token in a multi-token batch). Need to handle the sub-tensor copy.

Alternative: make the main model's graph write the hidden state directly to `mtp_hidden_state` as part of its computation. Add a `ggml_cpy(t_hidden_last, mtp_hidden_state)` node at the end of the main graph. Zero overhead since it's part of the same CUDA graph replay.

### Step 3: Single-sync speculation loop

The ideal loop structure (one sync per cycle):

```
// GPU: verify(accepted, draft) → argmax(logits[0]) → argmax(logits[1])
//      → copy hidden state to MTP tensor
// CPU: sync once, read 8 bytes (two token IDs)
// CPU: accept/reject decision (trivial comparison)
// GPU: MTP forward → argmax → draft token
// CPU: sync once, read 4 bytes (draft token ID)
```

Two syncs per cycle instead of four. And 12 bytes transferred instead of 2MB.

With the hidden state copy fused into the main graph, the MTP call doesn't need to sync the main scheduler at all — it's already done.

### Step 4: Fuse accept/reject + MTP into the graph (stretch goal)

The ultimate optimization: make the GPU do the accept/reject comparison too. The graph would:
1. Run verify (2 tokens)
2. Argmax both positions
3. Compare argmax[0] with draft token (stored on GPU)
4. Conditionally run MTP on the appropriate input
5. Output: {accepted_token, pred_after_draft, new_draft, was_accepted}

This eliminates ALL mid-cycle syncs. One graph launch, one sync at the end, 16 bytes back.

This requires conditional execution (CUDA dynamic parallelism or graph conditionals), which is complex. Save for later.

## What's Already Done (This Session)

### Changes committed/ready to commit:

1. **Fixed MTP KV cache size**: Capped at 1024 (was 262144 = model's n_ctx_train). MTP KV buffer: 512 MiB → 2 MiB.

2. **MTP graph reuse**: Fixed-topology attention graph with runtime masking and `ggml_set_rows` for KV write. `can_reuse()` returns true. Fixed uninitialized ubatch fields that prevented `allow_reuse()`.

3. **GPU-side argmax for MTP**: Added `ggml_argmax(logits)` to MTP graph. New `llama_mtp_get_argmax()` API. Extracts 4 bytes instead of 1MB.

4. **Q6_K CUDA get_rows kernel**: Wrote `k_get_rows_q6_K` kernel in `getrows.cu`. Added Q6_K to `supports_op` list. This allows tok_embd (Q6_K, 199 MiB) to live on GPU.

5. **tok_embd on GPU for MTP models**: Changed `dev_input` assignment in `llama-model.cpp` to use GPU buffer when `nextn_predict_layers > 0`. Eliminates CPU-mapped split that prevented CUDA graph capture.

6. **CUDA graphs confirmed working**: nsys shows both main model (1389 nodes) and MTP (52 nodes) graphs captured and replaying at ~36μs per launch.

### Files modified:

| File | Change |
|------|--------|
| `llama-slim/src/llama-context.h` | `mtp_last_argmax`, `get_mtp_argmax()` |
| `llama-slim/src/llama-context.cpp` | Cap n_ctx_mtp=1024, ubatch init fix, mtp_n_kv from cache, argmax extraction, `llama_mtp_get_argmax()` C API |
| `llama-slim/src/llama-graph.h` | Fixed-topology MTP KV input docs, `t_mtp_argmax` in result |
| `llama-slim/src/llama-graph.cpp` | Fixed-size mask + write_idx, `can_reuse()` returns true |
| `llama-slim/src/models/qwen35.cpp` | Fixed n_kv, `ggml_set_rows` for KV write, `ggml_argmax` in MTP graph |
| `llama-slim/include/llama.h` | `llama_mtp_get_argmax()` declaration |
| `llama-slim/tools/completion/completion.cpp` | Uses GPU argmax, wall-clock timing (temporary) |
| `llama-slim/src/llama-model.cpp` | tok_embd on GPU for MTP models |
| `llama-slim/ggml/src/ggml-cuda/getrows.cu` | Q6_K get_rows kernel |
| `llama-slim/ggml/src/ggml-cuda/ggml-cuda.cu` | Q6_K in GET_ROWS supports_op |

### Restricted vocab analysis (for future reference):

Coverage of accepted MTP draft tokens by restricted vocab:
- Top 4,096: 87.3% → effective accept ~83%
- Top 8,192: 92.6% → effective accept ~88%
- Top 50,000: 99.5% → effective accept ~94.5%
- Top 100,000: 100% → full acceptance preserved

Data: `gwen_data_backup/token_counts.bin`, `token_rank.bin`, `restricted_vocab_{2048,4096,8192}.bin`

## Verification

```bash
cd llama-slim/build

# Non-MTP correctness
MODEL=~/.cache/gwen/Qwen3.5-0.8B-Base-Q4_K_M.gguf
for p in "The quick brown fox" "In the beginning there was nothing but" "def fibonacci(n):" "The capital of France is" "1 + 1 = 2. 2 + 2 ="; do
  ./bin/llama-completion --no-conversation -m $MODEL -p "$p" -n 50 --temp 0 2>/dev/null
done > /tmp/llama-slim-check.txt
diff /tmp/llama-slim-baseline.txt /tmp/llama-slim-check.txt

# MTP correctness
MODEL=~/.cache/gwen/Qwen3.5-0.8B-Base-mtp-Q4_K_M.gguf
for p in "The quick brown fox" "In the beginning there was nothing but" "def fibonacci(n):" "The capital of France is" "1 + 1 = 2. 2 + 2 ="; do
  ./bin/llama-completion --no-conversation -m $MODEL -p "$p" -n 50 --temp 0 2>/dev/null
done > /tmp/llama-slim-mtp-check.txt
diff /tmp/llama-slim-baseline.txt /tmp/llama-slim-mtp-check.txt
```

## Performance History

| Version | MTP tok/s (brown fox) | vs baseline |
|---------|----------------------|-------------|
| Before snapshotting | ~470 | -13% |
| + DeltaNet snapshotting | ~470 | break-even |
| + Fixed MTP KV (1024) + graph reuse + GPU argmax | ~520 | **-4%** |
| + tok_embd on GPU + CUDA graphs confirmed | ~520 | **-4%** |
| **+ GPU argmax (main) + argmax_only mode** | **~590** | **+10%** |
| Target: fuse accept/reject on GPU (Step 4) | ~650+ | **+20%** |
