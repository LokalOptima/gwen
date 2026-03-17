# Post 21: MTP v3 — Soft Target Distillation + Vocab Reduction

MTP v2 achieved 50.7% acceptance with hard targets (CrossEntropyLoss) over a 20K restricted vocab. Two problems: massive overfitting (train loss 0.57 vs val loss 3.13 after 5 epochs), and the 20K lm_head dominates the model at 51% of parameters. This post covers the v3 redesign: KL divergence distillation with soft targets, vocab reduction to 4K, and a new dev server for efficient teacher logit extraction. Also: every bug we hit along the way.

## Why v2 Overfitted

The v2 training loop extracted hidden states from the GWEN server, computed hard targets (argmax of the teacher's full-vocab prediction), and trained with CrossEntropyLoss. The problem: hard targets throw away the teacher's entire probability distribution. When the teacher assigns 30% probability to token A and 25% to token B, the student only sees "the answer is A." All the uncertainty information — which is exactly what matters for speculative decoding acceptance — is lost.

This is a textbook case for knowledge distillation. DistillSpec (arXiv:2310.08461) showed +2-10% acceptance rate from using soft targets, and the overfitting gap (5.5× between train and val loss) suggested the model was memorizing training-set-specific patterns rather than learning the teacher's generalizable distribution.

## The v3 Architecture

```
┌───────────────────────────────┐     ┌──────────────────────────────┐
│       gwen_dev_server         │     │       train_mtp.py v3        │
│                               │     │                              │
│  POST /batch_logits           │     │  teacher = softmax(logits/T) │
│    extract_hidden_batch       │     │  student = mtp(embed, hidden)│
│    → RMSNorm(hidden)          │────>│                              │
│    → GEMM × restricted_embed  │     │  loss = KL(student, teacher) │
│                               │     │         × T²                │
│  Returns per token:           │     │                              │
│    hidden  [1024] FP16        │     │  Stage 1: train lm_head only │
│    logits  [K]    FP16        │     │  Stage 2: unfreeze all       │
└───────────────────────────────┘     └──────────────────────────────┘
```

Three changes from v2:

1. **Soft targets via KL divergence** — Instead of argmax → CrossEntropyLoss, the server returns teacher logits over the restricted vocab, and training minimizes KL(teacher ‖ student) × T² with temperature T=2.0.

2. **Vocab reduction** — From K=20K to K=4K. The lm_head drops from 20.5M to 4.1M params (51% → 15% of the model). Draft step latency should drop ~5×.

3. **Two-stage training** — Stage 1 trains only the lm_head at lr=1e-3 (1 epoch), then stage 2 unfreezes all params at lr=1e-4 with early stopping (patience=3).

## The Dev Server

The existing `gwen_server` served hidden states + hard predictions. For v3 we need teacher logits over the restricted vocab, which requires:

1. Dequanting K rows from Q6K embed_tokens → FP16 (one-time at startup)
2. RMSNorm on the hidden states (output_norm)
3. GEMM: normed_hidden × restricted_embed^T → logits [N, K]

I built `gwen_dev_server` as a new binary with a single endpoint: `POST /batch_logits`. Throughput: 22.5K tok/s at B=64, L=512.

### New CUDA code

**`gwen_dequant_rows_q6k`** — Dequants K specific rows from the Q6K embedding table into a contiguous FP16 buffer. Same dequant math as the existing batch embedding lookup, but indexed by a row_ids array instead of sequential token IDs. One CUDA block per row, 256 threads. Runs once at startup.

**`gwen_gemm_fp16`** — CUTLASS GEMM wrapper for FP16 weights (no dequant step). Identical to `gwen_gemm` but skips the dequant-to-temp_w step since the restricted embed is already FP16. Reuses the same CutlassGemm type (Sm80 mma.sync, 128×128×32 threadblock).

## Bug Parade

This implementation hit five bugs before producing correct results. Documenting all of them because every single one would have silently produced wrong training.

### Bug 1: OOM — F32 Reference Path Buffers (CUDA)

First launch: instant OOM at 65536 max_tokens.

```
CUDA error at memory.h:38: out of memory
```

`allocate_prefill(max_tokens)` defaults to `f32_path=true`, allocating ~6 GB of F32 buffers for a reference code path the dev server never uses. Fix: `f32_path=false`. Total memory dropped from ~10.8 GB to ~4.4 GB.

### Bug 2: Buffer Overflow — Logits Larger Than FFN (CUDA)

The initial implementation reused `prefill_ffn_gate` (N×3584 elements) as the GPU logits buffer. But K=4096 > n_ff=3584 — the logits don't fit. Would have silently written past the buffer into adjacent GPU memory. Caught in code review before it ran. Fix: dedicated `d_logits_buf` sized for N×K.

### Bug 3: KL Loss Inflated 500× — PyTorch `batchmean` Trap

First training run showed val_loss=2830 and train_loss=3219 on a pre-initialized model. Those numbers are absurd for KL divergence.

Root cause: `F.kl_div(..., reduction='batchmean')` divides by the **first dimension** (batch size B), not by the total number of tokens. With input shape [B, L-2, K] and B=4, L=512, the sum gets divided by 4 instead of 2040. That's a 510× inflation.

```python
# BROKEN: divides by B=4
F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * T**2

# FIXED: divides by actual token count
F.kl_div(student_log_probs, teacher_probs, reduction='sum') * T**2 / n_tokens
```

This is a well-known PyTorch footgun. The `batchmean` reduction was designed for 2D input [batch, classes] where "batch mean" and "token mean" are the same thing. With 3D input [batch, seq, classes], it silently gives the wrong answer. After the fix: KL loss = 4.6 (sensible).

### Bug 4: Teacher Logits Off-By-One — Wrong Prediction Target

Initial eval showed 6.8% acceptance rate with a pre-trained model that should give ~28%.

The MTP head at position t takes `(embed[t+1], hidden[t])` and predicts **token[t+2]**. The teacher logits from the server are `RMSNorm(hidden[t]) @ embed.T`, which predicts **token[t+1]**. The code was using `teacher_logits[:, :L-2]` (positions 0..L-3, predicting tokens 1..L-2), but the MTP predicts tokens 2..L-1. The teacher target needs to be shifted by one: `teacher_logits[:, 1:L-1]` (positions 1..L-2, predicting tokens 2..L-1).

Same bug in the val cache: it stored `hidden[:, :L-2]` but we need `hidden[:, :L-1]` (one extra position) so eval can slice both `hidden[:, :L-2]` for MTP input and `hidden[:, 1:L-1]` for teacher logits.

After fixing: acceptance went from 6.8% to 27.7%.

### Bug 5: 57-Minute Eval — FP16 Matmul on CPU

The "on-the-fly teacher logit computation" in eval did `RMSNorm(hidden) @ restricted_embed.T` on CPU. FP16 matmul on CPU is catastrophically slow — 57 minutes for 1027 eval batches. The data was already being sent to GPU for the MTP forward pass anyway. Fix: move `restricted_embed` and `output_norm_weight` to GPU, compute teacher logits there. Eval time: ~30 seconds.

## Baseline Acceptance Rates

Before training, I measured the pre-trained MTP head's acceptance rate across all K values to establish proper baselines (something v2 never logged):

| K | vs full-vocab | vs restricted | Coverage |
|---|--------------|---------------|----------|
| 2048 | 27.5% | 29.6% | 88.7% |
| 4096 | 28.0% | 29.3% | 91.9% |
| 8192 | 28.2% | 29.1% | 94.8% |
| 20000 | 28.6% | 29.0% | 97.8% |

Key finding: **K barely matters at baseline**. The pre-trained MTP head starts at ~28% acceptance regardless of vocab size. V2's 50.7% came entirely from training, not from having K=20K. This means vocab reduction from 20K to 4K costs almost nothing in starting accuracy while cutting lm_head parameters by 5×.

The restricted-vocab argmax matches the full-vocab argmax 89-98% of the time across all K values. The vocab restriction itself is not the bottleneck — the MTP body's prediction quality is.

## Distillation Loss

```python
def distillation_loss(student_logits, teacher_logits, T=2.0):
    student_flat = student_logits.reshape(-1, K)
    teacher_flat = teacher_logits.reshape(-1, K)
    n_tokens = student_flat.shape[0]
    teacher_probs = F.softmax(teacher_flat / T, dim=-1)
    student_log_probs = F.log_softmax(student_flat / T, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs,
                    reduction='sum') * (T ** 2) / n_tokens
```

The T² scaling compensates for the 1/T² factor in the softmax gradients, so the loss magnitude stays comparable across temperatures. T=2.0 is the Hinton default — it smooths the teacher distribution enough to expose the relative ranking of non-top-1 tokens without flattening everything to uniform.

## Two-Stage Training

Stage 1 (lm_head warmup, 1 epoch at lr=1e-3): The lm_head is initialized from embed_tokens rows — a reasonable starting point but not trained for the MTP task. Training it alone first prevents noisy gradients from corrupting the pre-trained body. This is the approach from IBM's Recurrent Drafter.

Stage 2 (full fine-tune, 3 epochs at lr=1e-4 with patience=3): Once the lm_head can produce reasonable logits, unfreeze everything and fine-tune end-to-end. Early stopping on val loss prevents the overfitting we saw in v2.

## Memory Budget (K=4096, max_batch=128, max_seq=512)

| Component | Size |
|-----------|------|
| Model weights (Q4_K_M) | 497 MB |
| Decode state | ~100 MB |
| Prefill buffers (FP16 path) | ~2.7 GB |
| Batch DeltaNet states | ~590 MB |
| Restricted embed (4096×1024) | 8 MB |
| Logits buffer (65536×4096) | 512 MB |
| **Total** | **~4.4 GB** |

Fits in 16 GB with room to spare. The F32 reference path would have added another 6 GB.

## Related Work

Our approach combines ideas from several lines of research on training draft models for speculative decoding. Here's what exists and where we fit in.

### Distillation for Draft Models

**DistillSpec** (Zhou et al., arXiv:2310.08461) is the foundational work. They showed that KL divergence distillation with soft teacher targets gives +10-45% speedup over standard speculative decoding, and that on-policy data (generated by the draft model itself) outperforms off-policy. We use their core recipe — KL distillation with temperature T=2.0 — but with offline teacher logits extracted from our CUDA server rather than on-policy generation.

**Recurrent Drafter** (Zhang et al., arXiv:2403.09919, Apple) trains an RNN-based draft head conditioned on the target model's hidden states via KL distillation. Our two-stage lr schedule (lm_head warmup then full fine-tune) comes from their approach. They report ~10% speedup gain from distillation vs hard targets, and up to 2.8× overall inference speedup.

**FastDraft** (arXiv:2411.11055, Intel) is a pre-train + fine-tune pipeline using sequence-level KL distillation. Their key finding for us: **offline distillation consistently outperforms online by 11-25%**. This validates our approach of pre-extracting teacher logits from the GWEN server rather than computing them on-the-fly during training.

**Training Domain Draft Models** (arXiv:2503.07807, IBM, ICLR 2025 workshop) is a best-practices survey confirming offline > online and white-box > black-box distillation. Our setup — hidden states + logits from the full model — is white-box offline, the best category.

### Vocabulary Reduction

This is where our approach is less common in the literature. Most speculative decoding work uses the full vocabulary for the draft model. A few recent papers explore reduction:

**VocabTrim** (arXiv:2506.22694) is a training-free method that prunes the draft model's lm_head to the most frequently sampled tokens. They showed up to 75% vocab reduction with negligible acceptance drop on Llama 3 (128K vocab → 32K). Ours is more aggressive (248K → 4K, 98.4% reduction) but we retrain the lm_head, which should recover more accuracy than training-free pruning.

**SpecVocab** (Williams, arXiv:2602.13836) showed that a speculative vocabulary of k=2048 tokens outperforms a static 32K vocabulary when the vocab is selected to match the target model's output distribution. This supports our finding that K barely matters at baseline (~28% acceptance across all K values) — what matters is training quality, not vocab size.

**Balancing Coverage and Draft Latency in Vocabulary Trimming** (arXiv:2603.05210) is a follow-up to VocabTrim that formalizes the coverage/latency tradeoff. They show that the optimal K depends on the hardware memory bandwidth: on bandwidth-constrained devices, smaller K wins despite lower acceptance because the draft step is so much faster. Our RTX 5070 Ti decode path is firmly bandwidth-bound, so this tradeoff applies directly.

### Beyond KL: Alternative Objectives

**EAGLE** (Li et al., arXiv:2401.15077) takes a different approach — instead of distilling the output distribution, it trains a draft head on the target model's hidden states at the feature level. We borrowed two of their regularization techniques: hidden state noise injection (uniform noise U(-0.05, 0.05) on the hidden input) and gradient clipping at 0.5. EAGLE-3 extends this by fusing hidden states from multiple layers of the target model.

**LK Losses** (arXiv:2602.23881) argues that KL divergence is a proxy for acceptance rate, and that small draft models converge to suboptimal solutions where minimizing KL doesn't maximize acceptance. They propose directly optimizing `-log(acceptance_rate)` as the loss function, reporting +8-10% gains in average acceptance length over KL training. Easy to implement, no overhead — worth trying if our KL training plateaus.

**AdaSPEC** (Hu et al., arXiv:2510.19779) addresses the limited-capacity problem differently: train the draft only on "easy" tokens where a reference model already matches the target, skip the hard ones. The intuition is that a small draft model can't learn everything, so focus its capacity where it can actually match.

### Where We Fit

| Technique | Source | What we use |
|-----------|--------|-------------|
| KL distillation with T=2.0 | DistillSpec | Core loss function |
| Two-stage lr schedule | Recurrent Drafter | Stage 1 lm_head warmup, stage 2 full |
| Offline teacher logits | FastDraft | Pre-extracted from CUDA server |
| White-box hidden states | EAGLE | MTP head conditions on teacher hidden |
| Hidden noise regularization | EAGLE | U(-0.05, 0.05) on hidden input |
| Vocab reduction to K=4096 | VocabTrim, SpecVocab | 98.4% reduction, retrained |
| Early stopping on val loss | Standard | Patience=3, prevents v2's overfitting |

The combination of soft-target distillation *with* aggressive vocab reduction *and* retraining is, as far as I can find, not explored in any of these papers. VocabTrim and SpecVocab study vocab reduction but don't retrain. DistillSpec, FastDraft, and Recurrent Drafter study distillation but use the full vocabulary. Our bet is that the two complement each other: distillation preserves accuracy despite the tiny vocab, while the tiny vocab makes the draft step fast enough to justify the training cost.

## What's Next

Training is running. Expected outcome: acceptance rate climbing from ~28% baseline toward 50%+, with soft targets hopefully reducing the overfitting gap and early stopping catching it if not. After K=4096 finishes, sweep K=2048 and K=8192 to find the throughput-optimal K, then Phase 3: GWMT v3 export + CUDA inference integration.
