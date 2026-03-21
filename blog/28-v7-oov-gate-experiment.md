# Post 28: v7 OOV Gate Experiment — Separating Skip from Predict

Investigation into whether a dedicated OOV binary classifier can replace the IDK neuron for skip decisions in MTP speculative decoding. Short answer: no.

## Motivation

Blog 25 showed the v6 IDK neuron correlates 0.891 with OOV tokens — it's primarily an OOV detector sharing the softmax with 4096 token logits. Two hypotheses motivated v7:

1. A **separate binary OOV classifier** (1025 params) could detect OOV more precisely than a neuron competing in a shared softmax
2. Training the token head **only on non-OOV positions** (masking the ~15% OOV positions from the loss) would give cleaner signal

The v7 design: OOV gate fires first → skip. Otherwise run lm_head → confidence gate → predict or skip.

## Phase 1: Measuring Cycle Costs

Before training, I instrumented `generate_speculative()` with CUDA event timing to get accurate per-operation costs. Previous blog posts used estimates derived from wall-clock math — the measured values differ:

| Operation | Measured (ms) | Blog 11 estimate |
|---|---|---|
| forward_1tok | 1.51 | 1.66 |
| forward_2tok | 1.98 | 2.26 |
| forward_mtp | 0.14 | 0.18 |
| rollback | 0.09 | 0.02 |

Key derived costs per token:

| Outcome | Total ms | Tokens | Per-token cost |
|---|---|---|---|
| Accepted (2-tok verify succeeds) | 2.12 | 2 | **1.06 ms/tok** |
| Rejected (2-tok verify fails) | 2.21 | 1 | **2.21 ms/tok** |
| Skipped (1-tok fallback) | 1.51 | 1 | **1.51 ms/tok** |

Skipping is cheaper than rejection (1.51 vs 2.21) but more expensive than acceptance (1.51 vs 1.06). A skip that avoids a rejection saves 0.70 ms. A skip that misses an acceptance wastes 0.45 ms. **Break-even precision: 0.45 / (0.45 + 0.70) = 39.1%.**

## Pre-Training Analysis: Can Any Skip Strategy Help?

Before training v7, I evaluated all strategies on the v6 val cache (24.9M positions) using measured cycle times. The v6 model's token predictions are the same for all strategies — only the skip decision varies.

| Strategy | Cost (ms/tok) | vs Baseline | Skip% | Skip Precision |
|---|---|---|---|---|
| Baseline (no MTP) | 1.510 | — | — | — |
| No skip (always speculate) | 1.230 | +18.5% | 0% | — |
| IDK neuron (v6) | 1.228 | +18.6% | 25.9% | 47.8% |
| Oracle OOV (perfect classifier) | 1.236 | +18.1% | 15.0% | 41.8% |
| Max confidence (thr=0.10) | 1.227 | +18.7% | 3.5% | 70.2% |

The no-skip strategy is nearly as good as IDK — the acceptance rate (74.2%) is high enough that absorbing the 25.8% rejection penalty is cheaper than aggressively skipping. IDK gains 0.002 ms/tok. Oracle OOV is *worse* than no skip because 58% of OOV positions would have been accepted (teacher and student agree on the same in-vocab token despite the ground truth being OOV).

This analysis predicted that v7 wouldn't help, but I ran the experiment anyway.

## Architecture Changes

**model.py**: Removed IDK neuron. Added `oov_head = Linear(1024, 1, bias=True)` — a 1025-parameter binary classifier sharing the same normalized hidden state as the token head.

**train_mtp.py**:
- Token loss: sparse KL over top-64 teacher probs, masked to non-OOV positions only
- OOV loss: BCE on all positions, ground truth from `vocab.full_to_restricted == -1`
- Combined loss: `loss_token + loss_oov` (no weighting needed)
- Eval metric: cost (ms/tok) from outcome matrix × measured cycle times

**inference.cu**: Added `kernel_oov_gate` (256-thread dot product + bias) and `kernel_oov_override` (conditional -1 write). Both run inside the CUDA graph. The lm_head GEMV always runs regardless of OOV gate output — it's only 0.01ms and can't be conditionally skipped in a graph.

**model.cu**: GWMT v5 format — same as v4 but carries `oov_head.weight` [1, 1024] FP16 + `oov_head.bias` [1] F32.

## Training

Warm-started from v6 best checkpoint (`--drop-idk` copies first K rows of K+1 lm_head). oov_head initialized random. 1 epoch over 473M tokens, ~2000 steps before manual stop.

Training curves at eval checkpoints:

| Step | Token Loss | OOV Loss | Accept% | Cost | OOV Gate% | OOV Acc% |
|---|---|---|---|---|---|---|
| 400 | 0.105 | 0.344 | 71.7% | 1.248 | 8.2% | 87.4% |
| 800 | 0.094 | 0.325 | 72.6% | 1.240 | 7.6% | 87.7% |
| 1200 | 0.089 | 0.313 | 72.4% | 1.241 | 7.0% | 87.7% |
| 1600 | 0.094 | 0.325 | 72.2% | 1.242 | 7.5% | 87.8% |

The OOV gate converged quickly to ~8% skip rate (conservative — ground truth OOV is 15%). OOV classification accuracy 87.8% is only slightly above the 85% you'd get by always predicting non-OOV.

## Benchmark Results

8 diverse prompts, 300 tokens each, greedy decoding, RTX 5070 Ti:

| Config | Avg tok/s | vs Baseline |
|---|---|---|
| Baseline (no MTP) | 648 | — |
| **v6 IDK** | **771** | **+19.0%** |
| v7 OOV gate only | 757 | +16.8% |
| v7 OOV + conf=0.10 | 746 | +15.1% |
| v7 OOV + conf=0.20 | 743 | +14.7% |

Per-prompt breakdown (v6 vs v7 OOV-only):

| Prompt | v6 tok/s | v6 skip% | v7 tok/s | v7 skip% |
|---|---|---|---|---|
| Repetitive | 930 | 4.5% | 924 | 2.6% |
| Narrative | 839 | 12.3% | 843 | 0.0% |
| TCP/UDP | 675 | 61.9% | 659 | 28.9% |
| Quantum | 750 | 38.5% | 708 | 22.1% |
| Code | 660 | 55.3% | 665 | 20.0% |
| Scientific | 806 | 34.4% | 773 | 10.8% |
| Formal | 744 | 41.7% | 724 | 17.1% |
| DevOps | 768 | 56.9% | 761 | 52.7% |

v6 wins on technical prompts (TCP/UDP, Quantum, Scientific, Formal) where IDK fires aggressively (35-62%). v7's OOV gate only fires 10-29% on these — not enough to filter the non-OOV rejections that IDK catches.

Adding confidence gating to v7 makes things *worse* — skipped positions still paid the MTP forward pass cost, and the confidence signal isn't precise enough to recover.

## Why v7 Failed

The core assumption was wrong: **OOV is not the main source of rejections.**

- OOV accounts for 15% of positions but only ~42% of rejections
- The other 58% of rejections are non-OOV positions where the draft predicts the wrong in-vocab token
- A perfect OOV classifier can't catch these
- v6's IDK neuron fires on ~26% of positions, catching both OOV and uncertain-but-in-vocab positions. Its 48% precision is just above the 39% break-even.

The break-even math is also unfavorable. With 74% acceptance rate, the accepted cycle cost (1.06 ms/tok) is so cheap that aggressive speculation is almost always worth it. The penalty for a wrong speculation (2.21 ms) is only 46% worse than skipping (1.51 ms), so you need very high skip precision to justify not trying.

## What I Learned

1. **Measure before optimizing.** The cycle timing instrumentation (`--profile-cycles`) should have been the first thing built. The pre-training cost analysis correctly predicted v7 wouldn't help — I should have listened to the data.

2. **OOV ≠ rejection.** 58% of OOV positions get accepted because teacher and student agree on the same in-vocab prediction even when the ground truth is out-of-vocab. The OOV label is about the data distribution, not about prediction difficulty.

3. **The IDK neuron is hard to beat.** Despite its simplicity (one extra row in the softmax), it learned to encode a useful "uncertainty" signal that correlates with rejection better than any single feature. It fires on both OOV positions and uncertain in-vocab positions — exactly the combination needed.

4. **Skip strategies have diminishing returns at high acceptance rates.** At 74% acceptance, the no-skip strategy already achieves 18.5% speedup. The theoretical maximum from perfect skip decisions is ~19.2%. There's only 0.7% left on the table.

## Current Best: v6 IDK

v6 remains the best MTP configuration at 771 tok/s (+19.0% vs baseline). The v7 code (GWMT v5 loader, OOV gate kernel, training pipeline) is committed and functional but produces worse results.

Next direction: instead of trying to skip better, focus on improving the acceptance rate itself — better token predictions would compound through the 2-for-1 accept cycle.
