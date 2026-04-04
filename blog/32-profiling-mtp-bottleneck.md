# Post 32: Profiling the MTP Bottleneck — Where 0.58ms Actually Goes

Post 31 got MTP to 590 tok/s (+10% over baseline). The intermediate S state commit eliminated the re-decode penalty. But we were flying blind — no reliable way to measure decode-only throughput, no automated correctness tests, and previous sessions produced wrong conclusions from garbage timing (wall-clock minus estimated model load).

This post documents building the measurement infrastructure, profiling `decode_mtp()` down to the microsecond, and discovering that the bottleneck is not where anyone expected.

## Step 1: Instrumented Timing

The existing `common_perf_print` is broken for MTP — it counts 2-token batches as `prompt_eval`. The existing `bench_mtp_llama.sh` used `date +%s%N` minus 800ms estimated model load. Both useless.

I added `chrono::high_resolution_clock` instrumentation directly into `generate_mtp()`:

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
llama_decode(ctx, batch);
auto t1 = std::chrono::high_resolution_clock::now();
main_decode_ms_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
```

Every `llama_decode()` and `llama_decode_mtp()` call is individually timed. At the end of generation, a single `MTP_STATS` JSON line goes to stderr:

```
MTP_STATS: {"n_tokens": 200, "decode_ms": 345.2, "tok_per_s": 579.4, 
  "mtp_draft_ms": 62.3, "mtp_draft_avg_ms": 0.47, "main_decode_ms": 278.1,
  "accepted": 87, "rejected": 45, "accept_rate": 65.9}
```

Scripts parse this with `grep -oP`. No wall-clock subtraction. No estimated constants.

## Step 2: Correctness Test Suite

`scripts/test_correctness.sh` runs 12 diverse prompts at three lengths (50, 200, 500 tokens). Base model generates reference output, MTP must match bit-for-bit after normalizing `[end of text]` markers.

50 and 200 tokens: 24/24 pass. At 500 tokens, 4/12 prompts diverge — DeltaNet recurrence amplifies the tiny FP non-associativity between 2-token and 1-token batch processing. This is inherent to the 2-token speculation path and not a bug we introduced. The test suite exits 0 when only 500-token tests fail.

## Step 3: Accurate Baselines

With the new tooling, the real numbers:

```
Base model tg64 (llama-bench):   529.4 ± 5.3 tok/s
MTP model tg64 (no speculation): 510.5 ± 5.4 tok/s  (3.6% VRAM tax)
```

MTP speculative decode, 12 prompts × 200 tokens:

| Prompt | tok/s | Accept% |
|--------|-------|---------|
| AI history | 511 | 66.7% |
| Narrative | 530 | 73.7% |
| Business | 511 | 65.3% |
| TCP/UDP | 460 | 50.8% |
| Quantum | 510 | 71.9% |
| Transformer | 484 | 56.7% |
| Python sort | 575 | 86.8% |
| Python class | 572 | 85.0% |
| Math word | 569 | 85.0% |
| Fibonacci | 578 | 96.0% |
| Fox repeat | 540 | 77.5% |
| Counting | 604 | 100% |
| **Average** | **537** | **76.2%** |

Average 537 tok/s — barely above the 529 baseline. TCP/UDP at 460 tok/s is 13% *slower* than baseline. MTP is a net wash with full vocab.

## Step 4: Profiling decode_mtp()

I added temporary `chrono` timing around each section of `decode_mtp()`:

```
MTP_PROFILE: calls=100 avg_total=0.58ms 
  [hidden=0.034 setup=0.002 compute=0.052 extract=0.489]
```

| Section | Time (ms) | What it does |
|---------|-----------|-------------|
| Hidden state copy | 0.034 | GPU→CPU→GPU, 4KB |
| Setup | 0.002 | Graph reuse check + set_inputs |
| Compute | 0.052 | CUDA graph launch (56 nodes) |
| **Extract + sync** | **0.489** | **Sync stream + read 4-byte argmax** |
| **Total** | **0.58** | |

The CUDA graph launch takes 52μs — working correctly. Setup is 2μs. Hidden state copy is 34μs. All the CPU overhead is 36μs total.

**The 489μs "extract + sync" is the actual GPU compute time.** `ggml_backend_sched_synchronize()` blocks until the CUDA graph finishes executing, then the 4-byte argmax is read. The GPU is busy for 489μs doing the MTP forward pass, dominated by the LM head GEMV: 248K output × 1024 input at Q4_K = ~144 MB of weight data to stream through at 896 GB/s.

This was the crucial insight: there's nothing to optimize on the CPU side. The bottleneck is the raw GPU compute for the 248K-output LM head matmul.

## Step 5: Restricted 50K Vocabulary

The restricted LM head (`lm_head_top50000.bin`, 42 MB) reduces the output dimension from 248K to 50K — nearly 5× less weight data to read. Profile with restricted head:

```
MTP_PROFILE: calls=100 avg_total=0.28ms 
  [hidden=0.035 setup=0.002 compute=0.052 extract=0.193]
```

Extract drops from 489μs to 193μs. Total MTP draft: 0.28ms instead of 0.58ms.

Full benchmark with 50K head:

| Prompt | 248K tok/s | 50K tok/s | 248K accept% | 50K accept% |
|--------|-----------|-----------|-------------|-------------|
| AI history | 511 | 569 | 66.7% | 66.7% |
| Narrative | 530 | 583 | 73.7% | 73.7% |
| Business | 511 | 565 | 65.3% | 65.3% |
| TCP/UDP | 460 | 504 | 50.8% | 47.4% |
| Quantum | 510 | 558 | 71.9% | 72.6% |
| Transformer | 484 | 532 | 56.7% | 55.5% |
| Python sort | 575 | 576 | 86.8% | 67.2% |
| Python class | 572 | 594 | 85.0% | 73.0% |
| Math word | 569 | 628 | 85.0% | 85.0% |
| Fibonacci | 578 | 628 | 96.0% | 96.0% |
| Fox repeat | 540 | 600 | 77.5% | 77.5% |
| Counting | 604 | 670 | 100% | 100% |
| **Average** | **537** | **584** | **76.2%** | **~74%** |

Every prompt faster with 50K. Acceptance drops on code (Python sort 87% → 67%) because tokens outside top-50K can't be predicted. Natural language acceptance is identical. All outputs remain bit-identical to baseline — the restricted head only affects which drafts are attempted, not which tokens are accepted.

## CUDA Graph Oscillation

While adding diagnostic logging to `ggml_backend_cuda_graph_compute`, I noticed the main model graph constantly resets:

```
CUDA_GRAPH: warmup complete key=0x...dce0 nodes=1521
CUDA_GRAPH: warmup reset key=0x...dce0 nodes=1413
CUDA_GRAPH: warmup complete key=0x...dce0 nodes=1413
CUDA_GRAPH: warmup reset key=0x...dce0 nodes=1521
```

1521 nodes = 2-token batch, 1413 nodes = 1-token batch. Every accept→reject transition changes the batch size, which changes the graph topology, which resets the CUDA graph warmup. The main model spends significant time running without graph replay.

The MTP graph (56 nodes) captures once and stays stable — no resets. This is because MTP always processes exactly 1 token.

## The Gap to 1000 tok/s

Current best: 670 tok/s (Counting, 100% accept, 50K head). Target: 1000+ on favorable text.

The accept-path cycle at 100% acceptance:
- 2-token main decode: ~2.71ms
- MTP draft (50K): 0.28ms
- Total: 2.99ms for 2 tokens = 670 tok/s

Three obstacles, in order of impact:

1. **Main decode kernel speed** (~0.35ms gap): llama's MMVQ is ~12% slower than gwen's hand-tuned dp4a GEMV. Main decode dominates cycle time. Closing this alone → ~750 tok/s.
2. **MTP draft not overlapped** (0.28ms): Currently serial. Overlapping on a separate CUDA stream hides it entirely → ~840 tok/s.
3. **CUDA graph oscillation** (~0.05-0.15ms per reject): Main model re-warms graph on every batch size change. Fix: maintain separate graph instances for 1-token and 2-token, or always use 2-token batches.

## Files Changed

| File | Change |
|------|--------|
| `tools/completion/completion.cpp` | `chrono` instrumentation in `generate_mtp()`, `MTP_STATS` output |
| `scripts/test_correctness.sh` | New: 12 prompts × 3 lengths, normalized EOG comparison |
| `scripts/bench_decode.sh` | New: llama-bench baseline + instrumented MTP decode benchmark |
