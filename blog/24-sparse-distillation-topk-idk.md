# Post 24: Sparse Distillation — Top-k + IDK Bucket

We're currently sending 4096 FP16 logits per token from the GWEN server to the training loop. At batch=128, seq=512, that's 512 MB of logits per batch — fine over shared memory on localhost, but a dealbreaker for remote training or any setup without `/dev/shm`. This post explores the literature on sparse teacher logits and arrives at a clean solution: send only the top-k logits and fold the remaining probability mass into the IDK bucket we already have.

## The bandwidth problem

Current `/batch_logits` response per batch (B=128, L=512, K=4096):

| Component | Size |
|-----------|------|
| Hidden states (N × 1024 × FP16) | 128 MB |
| Teacher logits (N × 4096 × FP16) | 512 MB |
| p_idk (N × F32) | 256 KB |
| **Total** | **~640 MB** |

Over shared memory this takes ~50ms (memory bandwidth). Over 10 Gbps ethernet it's ~500ms. Over 1 Gbps it's 5 seconds — nearly 3× the GPU compute time. The logits are 80% of the payload.

## Can we compress the logits?

Short answer: no. FP16 activations are pseudo-random floating point — the mantissa bits have near-maximum entropy. Measured compression ratios:

| Data | LZ4 ratio |
|------|-----------|
| Hidden states (FP16) | ~0.95× (5%) |
| Teacher logits (FP16) | ~0.85× (15%) |
| Token IDs (int32) | ~0.60× (40%) |

General-purpose compression barely helps. We need domain-specific compression.

## Naive top-k: the obvious approach and its problem

The idea: instead of 4096 logits per token, send only the top k=64. That's a 64× reduction in logit bandwidth.

This is well-established in practice. Arcee AI distilled Llama-405B using top-128 logits — full logits for 500M tokens would have been ~2.9 PB, top-k compressed it to ~50 GB. ModelScope's EasyDistill has it as a built-in feature. TinyViT (Microsoft, ECCV 2022) pre-computed sparse teacher logits to disk for vision model distillation.

Common k values in the literature:

| Setting | k | Source |
|---------|---|--------|
| LLM distillation (conservative) | 64–128 | Arcee DistillKit |
| LLM distillation (practical) | 10–20 | EasyDistill, "Don't Ignore the Tail" |
| Vision (ImageNet-1k) | 10 | TinyViT |

But naive top-k has a known bias problem. When you take the top-k logits and renormalize them to sum to 1, you're redistributing the tail mass onto the top tokens:

```
True teacher:      token_382: 0.35,  token_91: 0.15,  ...,  tail: 0.12
After renorm:      token_382: 0.398, token_91: 0.170, ...,  tail: 0.00
```

The student learns "the teacher is 40% sure it's token_382" when the teacher was actually 35% sure. "Distilling in the Dark" (Lund University thesis) showed empirically that **naive top-k with k<25 performs worse than no distillation at all** — the renormalization bias is that harmful.

"Sparse Logit Sampling" (ACL 2025 oral, arXiv:2503.16870) formally proved that top-k provides a **biased gradient estimator**, causing student overconfidence and miscalibration. Their proposed fix: random importance sampling of ~12 tokens from the teacher distribution, which gives unbiased gradients. Effective but more complex to implement.

## Top-k + IDK bucket: the unbiased version

The fix is simple: don't renormalize. Keep the true teacher probabilities for the top-k tokens, and collect everything else into a single IDK bucket:

```
True teacher (k+1 categories):
  token_382: 0.35          ← unchanged
  token_91:  0.15          ← unchanged
  ...
  token_xyz: 0.003         ← (64th token)
  IDK:       0.12          ← = 1 - sum(top_64)
```

This is a **lossless coarsening** of the original distribution. No probability has been moved between tokens. The teacher's uncertainty is preserved in the IDK bucket — which we already have infrastructure for from the p_idk work in post 23.

"Don't Ignore the Tail" (Dasgupta, Cohn & Baldwin, arXiv:2602.20816, Feb 2026) formalizes exactly this decomposition. They prove that full KL divergence decomposes exactly into:

```
KL(teacher ‖ student) = D_KL1 + α_K · D_KL2
```

Where:
- **D_KL1** = KL over k+1 categories (top-k probs + residual bucket)
- **D_KL2** = KL over the normalized tail (the remaining V-k tokens)
- **α_K** = teacher's tail mass (our p_idk)

Because α_K is small (teacher is usually confident), the tail term D_KL2 contributes very little. D_KL1 alone — KL over k+1 categories — captures almost all of the signal. The paper proposes amplifying D_KL2 with a hyperparameter, but shows D_KL1 by itself is already a strong baseline.

This is exactly what we'd implement: KL divergence over k+1 categories, where the (k+1)-th category is IDK.

## What goes over the wire

Per token, the server sends:

```
k × (uint16 token_index + fp16 logit)    = k × 4 bytes
1 × fp32 log_Z                            = 4 bytes
```

The client reconstructs: `p_i = exp(logit_i - log_Z)`, then `p_idk = 1 - sum(p_i)`.

At k=64: **260 bytes/token** vs current **8192 bytes/token**. A **31× reduction**.

| Batch payload (B=128, L=512) | Current (K=4096) | Top-64 + IDK |
|------------------------------|------------------|--------------|
| Logits | 512 MB | 16.3 MB |
| Hidden states | 128 MB | 128 MB |
| p_idk / log_Z | 256 KB | 256 KB |
| **Total** | **640 MB** | **145 MB** |

Over 1 Gbps: 5 seconds → 1.1 seconds. Over 10 Gbps: 500ms → 115ms.

## The training loss

The student side needs care. Even though we only send k token indices, gradients must flow through the entire student vocabulary (so the full lm_head gets trained, not just 64 entries). This works because the student's probability for each token depends on the global normalizer `log_Z = logsumexp(all_logits)`:

```python
# --- Teacher (from server) ---
# topk_logits: [B, L, k] fp16, topk_indices: [B, L, k] uint16, log_Z: [B, L, 1] fp32
teacher_topk_probs = (topk_logits.float() - log_Z).exp()           # [B, L, k]
teacher_idk = 1.0 - teacher_topk_probs.sum(-1, keepdim=True)       # [B, L, 1]
teacher_dist = cat([teacher_topk_probs, teacher_idk], dim=-1)       # [B, L, k+1]

# --- Student ---
student_logits = lm_head(student_hidden)                            # [B, L, vocab]
student_log_Z = student_logits.logsumexp(-1, keepdim=True)          # [B, L, 1]
student_topk = student_logits.gather(-1, topk_indices.long())       # [B, L, k]
student_topk_probs = (student_topk - student_log_Z).exp()           # [B, L, k]
student_idk = 1.0 - student_topk_probs.sum(-1, keepdim=True)       # [B, L, 1]
student_dist = cat([student_topk_probs, student_idk], dim=-1)       # [B, L, k+1]

loss = kl_div(student_dist.log(), teacher_dist, reduction='sum') * T**2 / n_tokens
```

The key insight: `student_log_Z = logsumexp(all_logits)` means every logit in the student's vocabulary contributes to the loss through the normalizer. When the student assigns too much mass to non-top-k tokens (making `student_idk` too large relative to `teacher_idk`), the gradient pushes down those stray logits even though we never explicitly sent them. The full vocabulary head trains properly.

## What changes on the server

We already have most of the pieces:

1. The restricted-vocab GEMM gives us 4096 logits per token (already implemented)
2. The full-vocab chunked GEMM gives us `log_Z` (already implemented for p_idk)
3. **New**: a top-k selection on the 4096 restricted logits — partial sort via `cub::DeviceRadixSort` or a simple threshold kernel
4. **New**: pack the k winners' indices + logits + log_Z into the response

The full-vocab GEMM (the expensive part) is already running for p_idk computation, so the marginal cost is just the top-k selection — negligible.

## Empirical analysis: where does the mass live?

Before picking k, I ran 64 random 512-token sequences from the training corpus through the dev_server with `p_idk=1` (32,768 token positions total) and measured the actual teacher probability distributions over the 4096 restricted vocab.

**The teacher is uncertain — a lot.** 46% of positions have top-1 probability below 0.30. This is an 0.8B model and it genuinely doesn't know what comes next almost half the time.

Cumulative probability mass captured by the top-k tokens (scaled by 1 − p_idk to reflect true mass within restricted vocab):

| k | mean | median | p10 | p25 | p75 | p90 |
|---|------|--------|-----|-----|-----|-----|
| 1 | 0.392 | 0.325 | 0.094 | 0.179 | 0.563 | 0.833 |
| 5 | 0.637 | 0.671 | 0.250 | 0.455 | 0.864 | 0.965 |
| 10 | 0.711 | 0.775 | 0.320 | 0.567 | 0.924 | 0.983 |
| 32 | 0.787 | 0.878 | 0.416 | 0.705 | 0.968 | 0.993 |
| **64** | **0.814** | **0.912** | **0.461** | **0.752** | **0.979** | **0.995** |
| 128 | 0.831 | 0.933 | 0.493 | 0.781 | 0.985 | 0.997 |
| 256 | 0.841 | 0.945 | 0.514 | 0.799 | 0.988 | 0.997 |
| 4096 | 0.850 | 0.955 | 0.532 | 0.819 | 0.991 | 0.998 |

The remaining ~15% gap between k=4096 (0.850) and 1.0 is p_idk — probability mass on the 244K tokens outside the restricted vocab.

How many tokens does it take to reach a given mass threshold?

| Threshold | mean | median | p90 | p99 |
|-----------|------|--------|-----|-----|
| 90% | 1462 | 48 | 4096 | 4096 |
| 95% | 1997 | 406 | 4096 | 4096 |
| 99% | 3034 | 4096 | 4096 | 4096 |

The median position needs just 48 tokens to reach 90% mass — but the mean is 1462, dragged up by the long tail of uncertain positions that spread mass across thousands of tokens.

**Top-1 probability distribution:**

| Range | Fraction |
|-------|----------|
| [0.00, 0.30) | 46.2% |
| [0.30, 0.50) | 23.5% |
| [0.50, 0.70) | 14.0% |
| [0.70, 0.90) | 9.0% |
| [0.90, 1.00] | 7.3% |

### Why k=64

The IDK bucket mass at various k reveals the abstention signal quality:

| k | mean IDK | median IDK | frac > 0.1 | frac > 0.3 | frac > 0.5 |
|---|----------|------------|------------|------------|------------|
| 8 | 0.311 | 0.254 | 73.6% | 43.9% | 22.1% |
| 16 | 0.253 | 0.175 | 63.6% | 32.2% | 16.1% |
| 32 | 0.213 | 0.122 | 54.9% | 24.6% | 13.1% |
| **64** | **0.186** | **0.088** | **46.7%** | **21.2%** | **11.3%** |
| 128 | 0.169 | 0.067 | 40.5% | 19.3% | 10.2% |
| 256 | 0.160 | 0.055 | 37.4% | 18.2% | 9.6% |

k=64 sits at the elbow:

- **Confident positions** (teacher has a clear top choice): cumsum₆₄ ≈ 0.98, IDK ≈ 0. The draft sees a clean distribution and learns to predict normally.
- **Uncertain positions** (teacher spreads mass broadly): cumsum₆₄ ≈ 0.40–0.60, IDK = 0.40–0.60. The draft sees a fat IDK bucket and learns "this is unpredictable."
- **k=32 is too aggressive**: median IDK of 12.2% means even moderately confident positions leak mass past rank 32, making the IDK signal noisy. 55% of positions have IDK > 0.1 — the draft would over-abstain.
- **k=128 adds little**: only 6% fewer positions with IDK > 0.1 compared to k=64, but 2× the bandwidth. The marginal cumulative mass gain is just 2.1% (91.2% → 93.3%).

The cumulative mass curve confirms diminishing returns past k=64: the jump from k=32 → k=64 adds 3.4% median mass, but k=64 → k=128 adds only 2.1%, and k=128 → k=256 only 1.2%.

## IDK as a calibrated abstention signal

There's a deeper consequence of moving from K=4096 to top-64+IDK that goes beyond bandwidth savings: it changes what the IDK token *means*.

With full K=4096 restricted logits, p_idk only captures probability mass that leaked outside the restricted vocabulary — genuinely rare tokens. Since K=4096 covers ~92% of the training corpus, p_idk is almost always small. The draft head learns IDK ≈ "out of vocabulary."

With top-64+IDK, the bucket absorbs everything outside the top 64 predictions — including tokens 65–4096 that the teacher assigned non-trivial probability to. Now p_idk is large in two distinct situations:

1. **OOV**: the correct token is outside the restricted vocabulary (as before)
2. **Teacher uncertainty**: the teacher spread probability across many tokens — none dominant enough to crack the top 64

Case 2 is the interesting one for speculative decoding. An uncertain position is exactly where the draft head is likely to guess wrong, wasting a verify cycle. If the draft learns to output high IDK probability when the teacher was uncertain, it's effectively learning a **calibrated abstention signal**: "I don't know what goes here — don't speculate on my behalf."

At inference time, the speculative decoder could use this: when the draft's IDK probability exceeds a threshold, skip speculation on that position and fall back to single-token verify. This avoids the cost of drafting 2–3 tokens that will get rejected anyway. The draft head becomes not just a predictor but a predictor that *knows when it doesn't know* — and that self-knowledge comes for free from the training objective, no extra loss term needed.

The choice of k controls the tradeoff. Too small (k=16) and the draft abstains too often, defeating the purpose of speculation. Too large (k=256) and IDK reverts to a pure OOV signal. The empirical analysis above shows k=64 is the sweet spot: 47% of positions have IDK > 0.1 (enough to be actionable), but the median is only 8.8% (confident positions are unaffected).

### The throughput case

Current speculative decode runs at ~55% acceptance rate (post 11). Every rejected speculation wastes a batched verify cycle — the expensive GEMM path — for no net gain beyond position 1.

With IDK-based abstention, the decoder can skip speculation on positions where the draft's IDK probability exceeds a threshold. The empirical data suggests ~21% of positions have IDK > 0.3 at k=64 — these are positions where the teacher itself was uncertain and no draft head could reliably predict. Speculating on them burns GEMM cycles for nothing.

By filtering these out, the draft only speculates when it has real confidence. The acceptance rate on speculated positions should increase (you've removed the worst cases from the pool), and the positions you skip fall back to cheap single-token GEMV decode (~1.1μs). The net effect: fewer wasted verify cycles, higher effective tokens per speculation cycle.

## Related work

| Approach | Handles residual mass? | Source |
|----------|----------------------|--------|
| **"Don't Ignore the Tail"** | Yes — exact k+1 decomposition | arXiv:2602.20816 |
| **Sparse Logit Sampling** | Yes, via importance sampling | arXiv:2503.16870, ACL 2025 |
| **DistillKit** (Arcee) | Partial: `missing_probability_handling: uniform` | github.com/arcee-ai/DistillKit |
| **EasyDistill** (ModelScope) | No — renormalizes | github.com/modelscope/easydistill |
| **TRL** (HuggingFace) | No explicit handling | github.com/huggingface/trl |

The combination of top-k sparse logits with an IDK residual bucket, applied to speculative decoding draft model training with a restricted vocabulary, appears to be novel. "Don't Ignore the Tail" formalizes the math but applies it to standard LLM pre-training distillation, not to draft model training with vocab reduction.

## What's next

Implementation plan:
1. Add a top-k selection kernel to the server (extract top-64 from the 4096 restricted logits)
2. New wire format: `?sparse=1` query param returns k × (idx, logit) + log_Z instead of full logits
3. Update `GwenClient` to unpack the sparse format
4. Modify the training loss to use the k+1 bucket formulation
5. A/B test: full-4096 vs top-64+IDK on acceptance rate, training loss, and throughput
