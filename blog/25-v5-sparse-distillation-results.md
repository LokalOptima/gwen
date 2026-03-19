# Post 25: MTP v5 Sparse Distillation — Results and Analysis

v5 sparse distillation is implemented and trained. This post covers the end-to-end results: what worked, what the data shows about the IDK neuron, and whether there's room to improve.

## What Changed from v4

v4 sent 4096 FP16 logits + p_idk per token from the CUDA server (512 MB/batch). v5 sends only top-64 logits + log_Z (the full-vocab log-partition function), folding the remaining mass into the IDK bucket. Three changes:

1. **Server**: new `?sparse=64` endpoint. CUB BlockRadixSort kernel selects top-k on GPU. Chunked 248K GEMM + logsumexp produces log_Z. Response: 2308 bytes/token (vs 10244 in v4).
2. **Loss**: 65-category KL divergence (64 teacher probs from `exp(logit - log_Z)` + IDK bucket). Student gathers the same 64 indices from its K+1 softmax + IDK neuron probability. All K+1 logits get gradient through logsumexp.
3. **Cleanup**: removed shared memory transport, `batch_logits_with_p_idk()`, temperature decay. HTTP at 145 MB/batch is fine.

## Training Setup

- Warm-started from v3 best checkpoint (71.1% accept, val_loss=0.52)
- lm_head: 4096 rows from v3, IDK neuron row initialized to zeros
- Stage 1: 1000 steps lm_head only (lr=1e-3), then stage 2: all params (lr=1e-4)
- 389M tokens seen (82% of one epoch), stopped manually
- Val cache: 1027 batches with server-computed p_idk (from v4 extraction)

## Training Curves

| Step | Train Loss | Val Loss | IDK Rate | Non-IDK Accept |
|------|-----------|----------|----------|----------------|
| 1000 | 0.405 | 0.323 | 29.0% | 80.0% |
| 5000 | 0.286 | 0.287 | 29.1% | 81.3% |
| 10000 | 0.260 | 0.272 | 27.1% | 81.4% |
| 15000 | 0.247 | 0.261 | 27.9% | 82.4% |
| 16000 | 0.245 | 0.261 | 27.5% | 82.4% |

Val loss was still declining (best at step 15000), but the rate of improvement was slowing. IDK rate locked in at ~27-29% by step 2000 and barely moved afterward. The model spent the remaining training improving token predictions, not IDK calibration.

## Inference Benchmark

63 diverse prompts, 200 tokens generated each, greedy decoding, RTX 5070 Ti:

| Config | Median tok/s | P10 | P90 | Speedup |
|--------|-------------|-----|-----|---------|
| No MTP | 643 | 640 | 645 | 1.00x |
| MTP v3 (71% accept) | 784 | 705 | 891 | 1.22x |
| **MTP v5 IDK** | **816** | **742** | **887** | **1.27x** |

v5 is 4% faster than v3 in median throughput (816 vs 784). The P10 improvement is larger (742 vs 705, +5%) — the IDK neuron helps most on hard prompts where v3 would have made bad guesses that get rejected.

No-MTP variance is tiny (640-645), confirming the baseline is stable. MTP variance is prompt-dependent (705-891 for v3, 742-887 for v5), as expected since acceptance rate varies by content.

## IDK Neuron Analysis

Analyzed 24.9M tokens from the full val cache to understand what the IDK neuron learned.

### Teacher IDK Decomposition

The IDK bucket combines two sources of "I don't know":
- **OOV mass**: probability on tokens outside the K=4096 restricted vocab (mean 12.7%)
- **Tail mass**: probability on restricted tokens ranked 65-4096, not in the top-64 (mean 4.1%)
- **Total IDK**: OOV + tail (mean 16.8%, median 7.9%)

### Top-64 Coverage

Top-64 captures 94.7% of restricted-vocab probability mass (median 96.9%). The sparse signal loses almost nothing compared to sending all 4096 logits.

### Calibration

| | Teacher OOV | Teacher Tail | Teacher Total IDK |
|-----|------------|-------------|-------------------|
| Student predicts IDK (27.9%) | 0.349 | 0.075 | 0.423 |
| Student predicts token (72.1%) | 0.042 | 0.028 | 0.070 |

The student's IDK decisions are well-calibrated for discrimination: teacher total IDK is 6x higher when the student fires IDK (0.423 vs 0.070).

### OOV vs Uncertainty

| Bucket | Tokens | Student IDK Rate |
|--------|--------|-----------------|
| High OOV, low tail (>0.3 OOV, <0.1 tail) | 2.87M (11.5%) | 90.2% |
| Low OOV, high tail (pure uncertainty) | 181 (0.0%) | 97.2% |
| Both high | 199K (0.8%) | 98.7% |

Correlations with student IDK probability:
- vs OOV mass: **0.891**
- vs tail mass: 0.307
- vs total IDK: 0.918

The IDK neuron is almost entirely an OOV detector (0.891 correlation). Pure tail uncertainty barely exists in the data — only 181 out of 24.9M tokens have high tail + low OOV. When the teacher is confident in the restricted vocab (low OOV), top-64 captures nearly everything (94.7%), leaving almost no tail mass. The "IDK captures teacher uncertainty" hypothesis didn't pan out because tail uncertainty is negligible at k=64 with K=4096.

### Student Over-Abstention

The student fires IDK at 27.9% while teacher total IDK mean is 16.8%. The IDK neuron is the single largest category in the 65-category teacher distribution (16.8% vs ~1.3% per token), so the loss naturally over-emphasizes matching it. This likely costs some throughput — positions where teacher IDK is 0.10-0.20 get classified as IDK when the student could have predicted the correct token.

## Should Training Continue?

Three signals suggest diminishing returns:

1. **Val loss plateau**: 0.261 at step 15K, 0.261 at step 16K. The improvement from step 10K (0.272) to 15K (0.261) was only 4%.
2. **IDK rate frozen**: Locked at ~28% since step 2000, never moved. More training won't change IDK behavior.
3. **Non-IDK accept saturating**: 80.0% at step 1K → 82.4% at step 16K. Gaining 2.4pp over 15K steps.

More training with the same setup would give marginal gains. The next improvement would come from changing the training, not extending it:

- **`--idk-weight 0.5`**: Halve the IDK KL term to reduce over-abstention from 28% toward teacher's 17%. Already implemented.
- **`--stage1-steps 0`**: Skip the aggressive lm_head warmup that overshoots the IDK neuron early. The IDK rate locking at ~28% by step 2000 suggests stage 1's lr=1e-3 overshoots and stage 2 can't recover.

## Implementation Notes

- **Top-k kernel**: CUB `BlockRadixSort` (256 threads, 16 items/thread). Bitonic sort was tried first but had correctness issues with the value-index association.
- **NaN fix**: Eval path reconstructs log_Z from cached p_idk via `log_Z = restricted_log_Z - log1p(-p_idk)`. When p_idk=1.0 (100% OOV), this produces -inf. Fixed by clamping p_idk < 1-1e-6 and teacher probs > 1e-8.
- **OOM fix**: Stage 2 unfreezing all params allocates ~288 MB extra optimizer states. Initial chunk budget of 500 MB was too aggressive. Fixed to 150 MB (chunk=11 seqs), matching v4's proven budget.
- **Bandwidth**: 2308 bytes/token over HTTP loopback (~15ms for a 64×512 batch). Shared memory transport was removed — unnecessary at 4.4x less data.
