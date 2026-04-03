# Post 29: MTP Infrastructure in llama-slim

gwen's MTP speculative decode delivered +27% (600 → 672 tok/s). Now it's time to port it to llama-slim — the "correctness-first" llama.cpp fork that's already at 535 tok/s. The goal: add MTP without changing a single bit of the main model's output.

This post covers the infrastructure work: getting MTP weights loaded, building the MTP draft graph in ggml, wiring up the speculative decode loop, and discovering that llama.cpp's recurrent memory has a silent failure mode that breaks state rollback.

## The Architecture

Qwen3.5's built-in MTP head predicts the token after the one the main model predicts. Given the main model's hidden state `h` and the embedding of the accepted token `t`:

```
x = FC(concat(RMSNorm(embed(t)), RMSNorm(h)))    # project [2048] → [1024]
x = FullAttentionLayer(x)                          # 1 transformer layer
logits = LMHead(RMSNorm(x))                        # shared with main model
draft = argmax(logits)
```

The attention layer is identical in architecture to the main model's 6 full-attention layers (8Q/2KV heads, 256 head dim, gated attention, RoPE). The FC projection, norms, and the attention layer are the MTP-specific weights (~39 MB in F16).

## Extending llama-slim's Memory for MTP

llama-slim (llama.cpp) treats the model as having exactly `n_layer` layers. All arrays, accessors, and memory allocation loops are bounded to `n_layer`. Adding an MTP layer at index 24 requires extending everything.

The GGUF file has `block_count=25` and `nextn_predict_layers=1`. During loading, we subtract nextn from n_layer (24 main layers), then set per-layer hparams for the MTP layer to match the full-attention layers:

```cpp
// Set hparams for MTP layers (same architecture as full-attention layers)
for (uint32_t i = hparams.n_layer; i < hparams.n_layer + hparams.nextn_predict_layers; ++i) {
    hparams.n_head_arr[i]          = hparams.n_head_arr[0];
    hparams.n_head_kv_arr[i]       = hparams.n_head_kv_arr[0];
    hparams.n_ff_arr[i]            = hparams.n_ff_arr[0];
    hparams.recurrent_layer_arr[i] = false;
}
```

Every accessor function (`n_head(il)`, `n_head_kv(il)`, `is_recurrent(il)`, etc.) got its bounds extended from `n_layer` to `n_layer + nextn_predict_layers`. The KV cache and recurrent memory allocation loops iterate over `n_layer_total()`. The result: layer 24 gets KV cache automatically (it's not recurrent), and the recurrent memory filter correctly skips it.

After these changes, the MTP model loads cleanly:
```
llama_kv_cache: 7 layers    # was 6 — the MTP attention layer gets its own KV cache
llama_memory_recurrent: 25 layers  # loop bound, but filter still allocates only 18 DeltaNet layers
```

Bit-identical output on all 5 test prompts: verified.

## The MTP Graph Builder

The MTP forward pass is a new graph type (`LLM_GRAPH_TYPE_MTP`). When `build_graph()` is called with this type, the qwen35 builder dispatches to `build_mtp()` instead of the main forward pass.

The graph builder creates two inputs:
1. **Token ID** — standard `build_inp_embd(tok_embd)` for the embedding lookup
2. **Hidden state** — a new input tensor `mtp_hidden_state` that gets filled externally from the main graph's `t_hidden_prenorm`

The attention layer initially uses no-cache mode (`build_attn_inp_no_cache`). This means the MTP attention can only see the current token — no history from previous steps. We'll add KV cache later.

## The `llama_decode_mtp()` API

A new public function wraps the MTP evaluation:

```c
int32_t llama_decode_mtp(llama_context * ctx, llama_token token, llama_pos pos);
```

Internally it:
1. Synchronizes the backend (ensures the previous decode finished)
2. Reads the hidden state from `gf_res_prev->t_hidden_prenorm` (GPU → CPU copy)
3. Builds the MTP graph (separate `gf_res_mtp` storage, doesn't overwrite the main graph)
4. Sets the hidden state on the graph's input tensor (CPU → GPU copy)
5. Evaluates the graph
6. Stores draft logits in the output buffer

One critical detail: `decode_mtp()` resets `gf_res_prev` after finishing, forcing the next main `decode()` call to rebuild its graph. The backend scheduler is shared between main and MTP graphs.

Validation: calling `decode_mtp()` between normal decode steps produces output identical to not calling it. The MTP evaluation doesn't corrupt the main decode path.

## The Speculative Decode Loop

The speculative decode loop follows gwen's pattern:

1. Sample `accepted` from main model logits
2. `llama_decode_mtp(ctx, accepted, pos)` → argmax draft logits → `draft`
3. Snapshot state: `seq_cp(0, 1)` (copies both KV cache and recurrent state metadata)
4. Process `[accepted, draft]` as a 2-token batch with logits at both positions
5. Check: does `argmax(logits_after_accepted) == draft`?

**Accept:** Both tokens are valid. Emit `accepted` and `draft`. Clean up snapshot.

**Reject:** Restore snapshot (`seq_rm(0) → seq_cp(1, 0) → seq_rm(1)`), replay `accepted` alone, emit only the model's actual prediction.

## The `n_seq_max` Bug

The first test produced completely wrong output. After hours of debugging:

The recurrent memory allocates `n_seq_max` cells. With the default `n_seq_max=1`, there's exactly 1 cell for 1 sequence. When `seq_cp(0, 1)` tries to create a snapshot on sequence 1:

```cpp
void llama_memory_recurrent::seq_cp(seq_id_src, seq_id_dst, p0, p1) {
    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        // ... do the copy ...
    }
    // else: silently do nothing
}
```

`seq_id_dst=1` but `size=1`, so `1 < 1` is false. **The snapshot silently fails.** No error, no warning — just a no-op. The subsequent rollback restores nothing, and the model's recurrent state is permanently corrupted.

Fix: the speculative decode test creates contexts with `n_seq_max=2`. After this change, rollback produces bit-identical output:

```
Path A (direct):   13 198 760 220 16
Path B (rollback): 13 198 760 220 16
Result: MATCH (rollback works)
```

## Results

**Correctness:** Speculative decode produces tokens **identical** to greedy baseline on all 5 test prompts. The accept/reject/rollback loop is provably correct.

**Acceptance rate:** ~50% without KV cache for MTP attention. The MTP layer can only attend to the current token (no history), limiting prediction quality.

**Performance analysis:** With the replay rollback approach, break-even is at ~80% acceptance. At 50%, the cost of replaying `accepted` on rejection makes MTP net-negative. No point benchmarking yet.

## What's Missing

Two things gwen has that we don't:

1. **KV cache for MTP attention.** gwen's MTP attends over all previous MTP steps (full context). Ours sees only the current token. This is the primary reason acceptance is 50% instead of 80%+. Adding KV cache requires either integrating the MTP layer into the main memory context's init_batch flow, or building a custom memory context for the MTP graph.

2. **Kernel-level DeltaNet snapshotting.** gwen's `kernel_deltanet_fused_2tok` saves S state *between* token A and B inside the kernel — rejection costs 0.04ms. Our replay approach costs 1.87ms (a full decode). This alone drops break-even from 80% to 25%.

| | gwen | llama-slim (current) |
|---|---|---|
| MTP KV cache | yes | no |
| Acceptance rate | 80-95% | ~50% |
| Reject cost | 0.04ms (snapshot) | 1.87ms (replay) |
| Break-even | ~25% | ~80% |
| Net result | +27% speedup | not yet viable |

The infrastructure is complete and correct. The next session adds KV cache for MTP — the single change that should push acceptance past break-even and deliver actual speedup.
