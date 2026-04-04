# Next Steps: MTP Speculative Decode — Road to 1000 tok/s

## Current State (after profiling + 50K restricted head)

### Baselines (accurate, instrumented)
```
Base model tg64 (llama-bench, FA off, 5 reps):    529.4 ± 5.3 tok/s
MTP model tg64 (no speculation):                   510.5 ± 5.4 tok/s
MTP speculative (full 248K vocab, 12 prompts avg): 536.8 tok/s
MTP speculative (50K restricted, 12 prompts avg):  584 tok/s
MTP speculative (50K restricted, best):            669.6 tok/s (Counting, 100% accept)
```

### decode_mtp() breakdown (full vocab → 50K restricted)
| Section | Full 248K | Restricted 50K |
|---------|-----------|----------------|
| Hidden state copy | 0.034ms | 0.035ms |
| Setup (graph reuse) | 0.002ms | 0.002ms |
| CUDA graph launch | 0.052ms | 0.052ms |
| GPU compute + sync | 0.489ms | 0.193ms |
| **Total** | **0.58ms** | **0.28ms** |

CPU overhead is 36μs. CUDA graphs are working (56 nodes captured). The bottleneck is pure GPU compute — the LM head GEMV dominates.

### Per-prompt results (50K restricted, 200 tokens)
| Prompt | tok/s | Accept% |
|--------|-------|---------|
| AI history | 569 | 66.7% |
| Narrative | 583 | 73.7% |
| Business | 565 | 65.3% |
| TCP/UDP | 504 | 47.4% |
| Quantum | 558 | 72.6% |
| Transformer | 532 | 55.5% |
| Python sort | 576 | 67.2% |
| Python class | 594 | 73.0% |
| Math word | 628 | 85.0% |
| Fibonacci | 628 | 96.0% |
| Fox repeat | 600 | 77.5% |
| Counting | 670 | 100% |
| **Average** | **584** | **~74%** |

### Correctness
- 50 tokens: 12/12 bit-identical
- 200 tokens: 12/12 bit-identical
- 500 tokens: 8/12 (4 diverge due to FP non-associativity in 2-token batch — known, cosmetic)

## Gap Analysis: 670 → 1000 tok/s

Accept-path cycle at 100% acceptance (50K head):
- 2-token main decode: ~2.71ms
- MTP draft: 0.28ms
- Total: 2.99ms → 2 tokens → 670 tok/s

| Obstacle | Cost | Fix | Expected tok/s |
|----------|------|-----|---------------|
| Main GEMV kernel speed | ~0.35ms/cycle | Port gwen's dp4a GEMV | ~750 |
| MTP draft serial | 0.28ms/cycle | Async on separate stream | ~840 |
| CUDA graph oscillation | ~0.1ms/reject | Dual graph instances | ~870 |
| Single-draft speculation | — | 3-token batch w/ 2 MTP | ~1000+ |

## Priority 1: Default to 50K restricted LM head

Make `lm_head_top50000.bin` the default for MTP. Currently requires `LLAMA_MTP_LM_HEAD` env var. The 2× draft speedup outweighs the ~10% acceptance drop on code.

## Priority 2: Async MTP on separate CUDA stream

Overlap MTP draft with CPU work or next main decode. Launch MTP graph, continue CPU-side token emission and sampler updates, sync only when draft token is needed. Hides the 0.28ms entirely.

## Priority 3: Fix CUDA graph oscillation

Main model graph resets warmup on every 1-token ↔ 2-token batch transition. Options:
- Maintain two separate CUDA graph instances (keyed by batch size)
- Always decode 2-token batches (pad with dummy on reject path)

## Priority 4: Port gwen's GEMV kernels

Close the 12% kernel speed gap. gwen's hand-tuned dp4a GEMV for Q4_K uses static warp counts, block-stride loops, direct Q4_K unpacking. Port as an SM_120 fast path in `ggml/src/ggml-cuda/`.

## Priority 5: Multi-token speculation

For high-acceptance text (>80%), speculate 2 tokens ahead: run MTP twice, batch [accepted, draft1, draft2]. Accept path produces 3 tokens per cycle. With kernel speedup + async overlap: 1000+ tok/s.

## Test/Benchmark Commands

```bash
# Correctness (12 prompts × 3 lengths)
bash scripts/test_correctness.sh

# Quick correctness (12 prompts × 50 tokens)  
bash scripts/test_correctness.sh --quick

# Decode benchmark (llama-bench + instrumented MTP)
bash scripts/bench_decode.sh 200
```
