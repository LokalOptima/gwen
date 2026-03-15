# MTP Training Pipeline: Three Bugs and a Diagram

*Blog post #19 in the GWEN series — fixing buffer aliasing, wrong training targets, and discovering a CUDA/PyTorch mismatch*

## The State of Play

[Post #18](18-chunkwise-deltanet.md) got the chunkwise DeltaNet kernel working, giving a 17% batch extraction speedup. The next step was to actually fine-tune the MTP head on spoken English data. Before training, I wanted to do a thorough end-to-end verification of the entire pipeline: from CUDA inference to hidden state extraction to MTP training targets.

Good thing I did. Three bugs were hiding in the pipeline, and a fourth mismatch is still unresolved.

## Bug 1: Buffer Aliasing Corruption in Full Attention Layers

The batch extraction path (`allocate_prefill()`) is aggressive about memory reuse. To save GPU memory, it aliased `prefill_ffn_up` and `prefill_ffn_out` to the same backing storage as `prefill_proj_qkv`. The reasoning: for DeltaNet layers, `proj_qkv` is consumed at step 6 (after the gated DeltaNet recurrence), while `ffn_up` and `ffn_out` aren't needed until step 12 (the FFN block). Different lifetimes, safe to share memory.

Except for full attention layers, the lifetimes overlap:

```cpp
// Full attention layer: V projection writes to prefill_ffn_up...
gwen_gemm(w.attn_v.device_data, w.attn_v.type, prefill_temp_w,
          pf_norm, prefill_ffn_up,    // [N, kv_dim=512]
          kv_dim, w.attn_v.shape[0], N, s);

// ...while prefill_proj_qkv still holds Q+gate, needed by deinterleave:
kernel_deinterleave_qgate_batch<<<grid, 256, 0, s>>>(
    prefill_proj_qkv, prefill_proj_gate, prefill_ffn_out,
    N, cfg.n_head, cfg.head_dim);
```

When `prefill_ffn_up` and `prefill_proj_qkv` point to the same memory, the V projection overwrites the Q+gate data. The corruption is subtle because the buffers are large (`N * 6144` half-elements for QKV vs `N * 512` for V) — only the first `N * 512` elements get clobbered, which may or may not visibly affect the attention output depending on which heads land in that region.

The fix is one comment and two separate allocations:

```cpp
// Cannot alias ffn_up/ffn_out with proj_qkv: full attention layers use ffn_up for V
// projection and ffn_out for deinterleaved gate while proj_qkv still holds Q+gate.
prefill_ffn_up   = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));
prefill_ffn_out  = static_cast<half*>(a(max_tokens * cfg.n_ff * sizeof(half)));
```

Cost: ~28 MB more VRAM at B=64, L=512. Cheap insurance.

## End-to-End Verification (4 Levels)

With the aliasing bug fixed, I ran a systematic verification of the entire pipeline, from raw CUDA inference down to MTP training inputs.

**Level 1 — Greedy decode correctness.** GWEN's greedy decode vs llama.cpp golden data across 8 diverse prompts, 12 test configs. Result: 12/12 pass, all 8 prompts produce identical token sequences.

**Level 2 — GEMV vs GEMM path agreement.** The `extract_hidden` function (F32 GEMV, one token at a time) vs `extract_hidden_batch` (FP16 GEMM, batched) on identical inputs. Result: cosine similarity > 0.9999 per position, max absolute difference < 0.008. The gap is F32-vs-FP16 precision, not data corruption.

**Level 3 — Hidden state sanity check.** Can hidden states predict next tokens? Applied `output_norm + embed.T` to get logits from each hidden state, compared argmax to actual next tokens. Result: 57.5% aggregate match (threshold was 50%). The main model is indeed encoding useful information in its hidden states.

**Level 4 — PyTorch MTP loads and runs.** The MTP head loaded from pre-trained weights processes batch-extracted hidden states without errors and produces sensible logit distributions.

All four levels pass. The extraction pipeline is correct.

## Bug 2: Wrong Training Targets (The Big One)

This was the most conceptually important bug. The training loop was using **ground-truth text tokens** as targets:

```python
# WRONG: target is the next token from the original text
mtp_targets = token_ids[:, 2:]  # ground-truth token t+2
```

This is wrong because of what speculative decoding actually does. During inference, the MTP head drafts a token, and the main model either accepts or rejects it. Acceptance requires the MTP's prediction to match the **main model's** prediction — not what a human originally wrote.

Consider the sentence "The meeting was held yesterday." If the main model predicts "was" at position 3 but the original text says "is", training against "is" teaches the MTP to predict something the main model won't accept. The ground-truth accuracy was only ~17%, which seemed puzzling since the CUDA acceptance rate in [post #11](11-activation-replay-mtp-verdict.md) was ~55%. These are measuring completely different things: ground-truth match measures agreement with the text, while acceptance rate measures agreement with the main model.

### The Fix

The server needs to return what the main model actually predicts, not just hidden states. I added `predict_from_hidden()` to the CUDA side:

```cpp
void InferenceState::predict_from_hidden(
    Model& model, half* hidden_gpu, int N, int32_t* preds_host) {
    // 1. Batch RMSNorm: hidden → normalized
    kernel_rmsnorm_batch_f32w<<<N, 256, 0, s>>>(
        hidden_gpu, model.output_norm.device_data,
        prefill_norm, N, cfg.n_embed, cfg.rms_norm_eps);

    // 2. Per-token: Q8_1 quantize + GEMV (embed_tokens @ normed) + argmax
    for (int t = 0; t < N; t++) {
        gwen_quantize_q8_1(norm_t, x_q8_a, cfg.n_embed, s);
        gwen_gemv_dp4a(model.token_embd.device_data, x_q8_a, logits_h,
                       cfg.n_vocab, cfg.n_embed, model.token_embd.type, s);
        // argmax → preds_host[t]
    }
}
```

The `/batch_extract?preds=1` endpoint now returns both hidden states and main model predictions. The training loop changed to:

```python
# CORRECT: target is what the main model predicts from h[t+1]
hidden, main_preds = gwen.batch_extract_with_preds(token_ids)
mtp_target_ids = main_preds[:, 1:L-1]  # main model prediction from h[i+1]
```

The acceptance metric during training is now computed directly as MTP argmax matching the target — which IS the main model's prediction. No more confusion between ground-truth accuracy and acceptance rate.

### The Corrected Pipeline

Here's the full picture of how training data flows:

```
Sentence: "The cat is on the table"
Tokens:   [The, cat, is, on, the, table]
Positions: [0,   1,   2,  3,   4,    5]


STEP 1: Batch extraction (CUDA server, one-time per batch)
═══════════════════════════════════════════════════════════

Input tokens:    The    cat    is     on     the    table
                  |      |      |      |      |      |
            +-----+------+------+------+------+------+
            |      24-layer main model (Q4_K_M)
            +-----+------+------+------+------+------+
Hidden:          h[0]   h[1]   h[2]   h[3]   h[4]   h[5]
                  |      |      |      |      |      |
                  v      v      v      v      v      v
             norm+lm_head  (same Q6_K embed_tokens used at decode time)
                  |      |      |      |      |      |
                  v      v      v      v      v      v
Main model    pred0   pred1  pred2  pred3  pred4  pred5
predictions: ="cat"  ="is"  ="on"  ="the" ="table" ="."


   Server returns: hidden states [h0..h5] + predictions [pred0..pred5]


STEP 2: MTP fine-tuning (PyTorch, GPU)
═══════════════════════════════════════

MTP input A:     embed[cat]  embed[is]  embed[on]  embed[the]
  (token t+1)        |          |          |          |
                     |          |          |          |
MTP input B:       h[0]       h[1]       h[2]       h[3]
  (hidden t)         |          |          |          |
                     v          v          v          v
                +--------------------------------------------+
                |          MTP Head (21M params)              |
                |  norm+concat -> FC -> attn -> FFN -> head   |
                +----+----------+----------+----------+------+
                     v          v          v          v
MTP output:       mtp0       mtp1       mtp2       mtp3

                     |          |          |          |
     TARGET --->   pred1     pred2     pred3     pred4
     (main model   ="is"     ="on"     ="the"   ="table"
      prediction
      from h[t+1])

                loss = CrossEntropy(mtp_output, target)
```

The key insight: **target[i] = pred_{i+1}** = what the main model predicts from h[i+1], NOT ground-truth text. This way the MTP learns to match the main model exactly, which is what acceptance requires during speculative decoding.

## Bug 3: MTP Weights Never Uploaded to GPU

The GWEN server had a loading order bug: `load_mtp()` was called *after* `upload_weights()`. Since `upload_weights()` does the bulk device memory allocation and H2D copies, the MTP weights were sitting in host memory with null device pointers. Any attempt to run the `/test_mtp` endpoint triggered a CUDA illegal memory access.

One-line fix: move `load_mtp()` before `upload_weights()`. Trivial once found, but the crash message (`CUDA error: an illegal memory access was encountered`) gives zero indication of root cause.

## The Open Issue: CUDA vs PyTorch MTP Mismatch

With all three bugs fixed, I exported the pre-trained MTP weights to GWEN's binary format and ran a head-to-head comparison: CUDA MTP vs PyTorch MTP, same hidden states, same input tokens.

Results:

| Test | CUDA acceptance | PyTorch acceptance | Prediction match |
|------|-----------------|-------------------|-----------------|
| Prompt A | 48.2% | 49.1% | 38% |
| Prompt B | 51.3% | 48.7% | 62% |
| Prompt C | 46.0% | 47.3% | 20% |

The two implementations agree on only 20-62% of their predictions. Both have similar overall acceptance rates (~48%), but they're accepting *different* tokens.

The architecture looks correct on paper: same deinterleave pattern, same RoPE parameters, same attention mechanism. The likely culprit is numerical precision. CUDA uses FP16 throughout (matching the main model's decode path), while PyTorch uses BF16 autocast by default. FP16 and BF16 have different mantissa widths (10 vs 7 bits) and different exponent ranges — the rounding patterns differ everywhere, and those differences compound through the MTP head's attention and FFN layers.

This matters because fine-tuned weights optimized for BF16 arithmetic will produce different token predictions when run through FP16 arithmetic. If the mismatch rate is 40-80%, a fine-tuned MTP head that achieves 70% acceptance in PyTorch might only get 50% in CUDA — which is below break-even and makes the fine-tuning pointless.

This **must** be resolved before fine-tuning starts.

## Infrastructure Added

This session added several new capabilities to the GWEN codebase:

- **`--compare-extract` mode**: Runs Level 2 verification (GEMV vs GEMM extraction) from the main binary
- **`--mtp` server flag**: Enables the `/test_mtp` endpoint for CUDA MTP evaluation
- **`/batch_extract?preds=1`**: Returns main model argmax predictions alongside hidden states
- **`/test_mtp` endpoint**: Runs CUDA MTP sequentially on batch-extracted hidden states, returns predictions
- **`predict_from_hidden()`**: CUDA function computing `norm → Q8_1 → GEMV(embed_tokens) → argmax` per position
- **`pyproject.toml`**: Proper Python project config with all training dependencies
- **`scratch/test_mtp_cuda_vs_pytorch.py`**: Head-to-head MTP comparison harness

## Lessons

**1. Buffer aliasing needs a lifetime analysis, not just a gut check.** The DeltaNet layer lifetime analysis was correct but I failed to consider that full attention layers have a different data flow. When you alias buffers, you need to check *every* code path that uses them, not just the common one.

**2. Training targets must match the verification criterion.** This sounds obvious in retrospect, but it's easy to default to "predict the next token in the text" because that's what language model training always does. Speculative decoding is different — the MTP is trained to match another model, not to match reality.

**3. Precision format mismatches are insidious.** FP16 and BF16 both look like "half precision" but they round differently. If your training loop uses BF16 and your inference uses FP16, the two will diverge unpredictably. The fix is probably to force PyTorch training to FP16 to match CUDA, or to add BF16 support to the CUDA kernels.

**4. Check your loading order.** The `load_mtp()` before `upload_weights()` bug was invisible until the server actually tried to use the MTP weights. Unit tests on the loading functions would have caught this immediately.

## Next Steps

The priority is clear: fix the CUDA/PyTorch MTP mismatch before any fine-tuning. The options are:

1. **Force PyTorch to FP16** — Change autocast dtype and verify the MTP head trains stably in FP16 (it might, given QK-Norm)
2. **Add BF16 to CUDA** — Modify the CUDA MTP kernels to use BF16 arithmetic (RTX 5070 Ti supports BF16 tensor cores natively)
3. **Trace the divergence** — Run both implementations with identical inputs and compare intermediate activations (post-norm, post-attention, post-FFN) to find where they diverge

Option 1 is probably the fastest path. Once the two implementations agree, fine-tuning can begin with confidence that improved PyTorch accuracy will translate to improved CUDA acceptance.
