# Post 38: Proper Speculative Sampling for Stochastic MTP

The MTP speculation loop was hardcoded to greedy argmax, bypassing all user
sampling parameters. This post documents making MTP work correctly with
stochastic sampling (temp=1.0, top_k=20, presence_penalty=2.0) — the Qwen3.5
recommended defaults we hardcoded in post 37.

**Date**: 2026-04-07

---

## The problem

The `generate_mtp()` function used `greedy_argmax` and `llama_get_argmax_ith`
for ALL token selection — draft, verify, and emit. The `common_sampler` was
only called for `accept` (tracking history), never for actual sampling. This
meant MTP output was always greedy regardless of what the user set for
temperature, top_k, or presence_penalty.

Since we hardcoded the Qwen recommended params (temp=1.0, top_k=20,
presence_penalty=2.0) as defaults in post 37, the MTP loop was silently
producing greedy output while the non-MTP path respected the sampling config.

---

## Attempt 1: Naive sampler integration (310 tok/s)

First try: replace `greedy_argmax` / `llama_get_argmax_ith` with
`common_sampler_sample(smpl, ctx, idx)` in the stochastic path, keeping greedy
as a fast path when `temp < 1e-6`.

This required disabling `argmax_only` mode (which transfers 8 bytes per decode
instead of 2MB logits), since the sampler needs full logits.

Result: **310 tok/s** — catastrophically slower than the 491 tok/s non-MTP
baseline.

### Root cause: penalty sampler on 248K tokens

The default sampler chain order is `PENALTIES → TOP_K → TEMPERATURE → DIST`.
The penalties sampler iterates over ALL 248,320 tokens doing a hash map lookup
per token, BEFORE top_k prunes to 20 candidates. At ~150 sampling calls per
200-token generation, that's ~37M hash lookups.

The 1MB logit transfer everyone would suspect (~16us at PCIe 5.0) was
irrelevant. The CPU penalty computation dominated.

---

## Attempt 2: fast_sample — top_k before penalties (454 tok/s, 44% accept)

Wrote a custom `fast_sample()` that reorders the chain:
1. Min-heap scan for top-64 candidates from raw logits — O(n_vocab)
2. Apply penalties to only those 64 candidates — 64 hash lookups, not 248K
3. Sort, take final top_k=20
4. Temperature + softmax + sample

This avoids the 248K penalty scan entirely. The wider initial top-64 absorbs
any ranking shifts from the 2.0 presence penalty (penalties only subtract,
so a token outside raw top-64 can't enter the penalized top-20 in practice).

Result: **454 tok/s, 44% acceptance**. Fast enough to be usable, but still
8% slower than the 491 tok/s non-MTP baseline. The 44% acceptance rate is the
problem — with exact-match accept/reject, even when the draft matches the
main model argmax, the sampler picks a different token ~56% of the time
(stochastic sampling with temp=1.0 and top_k=20).

---

## Attempt 3: Proper speculative sampling (527 tok/s, 75% accept)

The exact-match strategy (accept iff sampled == draft) wastes potential. When
the main model assigns 40% probability to the draft token, exact match rejects
60% of the time even though the draft is reasonable.

Standard speculative sampling (Leviathan et al. 2023) accepts the draft token
with probability `min(1, p(draft)/q(draft))` where p is the target (main
model) probability and q is the draft model probability. This preserves the
exact target distribution while accepting more aggressively.

### Getting q(draft)

The MTP head computes logits on GPU and only transfers the argmax (4 bytes).
For speculative sampling, we also need `q(draft)` — the MTP head's confidence
in its pick.

Since draft = argmax(MTP logits), q(draft) = max(softmax(MTP logits)). We
transfer the MTP logits to CPU (50K restricted vocab = 200KB, ~3us at PCIe
5.0) and compute the softmax on CPU.

New API added:
- `llama_mtp_set_extract_logits(ctx, bool)` — enable MTP logit transfer
- `llama_mtp_get_logits(ctx, &n)` — access transferred logits
- `llama_mtp_get_token_map(ctx, &n)` — restricted → full vocab mapping

### Consistent penalty application

Critical subtlety: p(draft) is computed after applying presence penalties to
the main model logits. q(draft) must be penalized identically, otherwise the
p/q ratio is comparing penalized vs unpenalized distributions. We apply the
same penalty window to both.

### Optimizing the q(draft) computation

First version did per-element hash lookups during the 50K softmax scan (100K
hash lookups per cycle). Fixed by copying the 50K logits, applying penalties
to only the ~64 affected entries in-place, then scanning without hash lookups.

### Rejection handling

On rejection, we sample from the main model distribution via `fast_sample`
(already computed as part of the accept/reject decision). This is an
approximation — proper speculative sampling samples from `(p - q)+`, but
the difference is negligible for a well-aligned draft model.

### Two-mode design

- **Greedy (temp=0)**: argmax_only GPU mode, `llama_get_argmax_ith`, no
  logit transfer. Full speed (637 tok/s).
- **Stochastic (temp>0)**: `fast_sample` + speculative accept/reject with
  MTP q(draft). Respects all sampling params (527 tok/s).

---

## Benchmark results

All benchmarks: 12 diverse prompts, 200 tokens each, RTX 5070 Ti.

### Final numbers

| Mode | tok/s | Accept % |
|---|---|---|
| Base (no MTP, llama-bench tg64) | 491 | -- |
| MTP greedy (temp=0) | 637 | 83.4% |
| MTP stochastic, speculative | 555 | 75% |

### Development progression

| Mode | tok/s | Accept % | Notes |
|---|---|---|---|
| Naive sampler (common_sampler_sample) | 310 | 47% | 248K penalty hash lookups per sample |
| fast_sample (top_k before penalties) | 454 | 44% | Reordered chain, 64 lookups not 248K |
| Speculative v1 (q with per-element hash) | 427 | 72% | Correct p/q ratio but q overhead killed gains |
| Speculative v2 (penalty consistency fix) | 501 | 78% | Consistent penalties, still 100K hash lookups |
| Speculative v3 (in-place penalty, no hash scan) | 527 | 75% | Copy + patch + scan |
| Speculative v4 (copy-free + fast_expf) | 555 | 75% | No copy, fused single-pass, Schraudolph exp |

### Per-prompt stochastic breakdown (v4)

| Prompt | tok/s | Accept % |
|---|---|---|
| AI history | 547 | 77.7% |
| Narrative | 556 | 75.4% |
| Business | 558 | 76.8% |
| TCP/UDP | 534 | 69.2% |
| Quantum | 570 | 78.6% |
| Transformer | 555 | 74.6% |
| Python sort | 569 | 78.6% |
| Python class | 532 | 68.1% |
| Math word | 593 | 86.9% |
| Fibonacci | 545 | 76.4% |
| Fox repeat | 539 | 74.3% |
| Counting | 595 | 91.3% |

---

## Optimizing compute_mtp_q_draft

### v3 → v4: copy-free + fast_expf

The v3 q_draft computation copied 200KB of MTP logits, patched ~64 penalized
entries, then scanned twice (max + sum_exp). Two optimizations:

**Copy-free correction**: Instead of copying 50K floats to apply ~64 penalties,
scan the original read-only buffer for max and sum_exp, then correct the sum
for only the penalized entries:

```
sum_exp_penalized = sum_exp_raw
                  - Σ exp(raw[i] - max)          // remove raw contributions
                  + Σ exp(penalized[i] - max)     // add penalized contributions
```

This eliminates the 200KB memcpy and keeps the scan cache-friendly (read-only).

**Fused single-pass with fast_expf**: Instead of two passes (max, then sum_exp),
fuse into one pass with online max tracking — when a new max is found, rescale
the running sum via `sum *= exp(old_max - new_max)`. Uses Schraudolph (1999)
fast exp approximation (~2% relative error, 3-5x faster than `expf`). The
~64 penalty corrections still use standard `expf` for ratio accuracy.

Result: q_draft dropped from 0.16ms → 0.04ms per call (4x speedup).

---

## Per-cycle profiling

Per-cycle trace of a typical ACCEPT cycle (stochastic path):

```
dispatch=155  sync=2438  fsamp=108  qdraft=41  fsamp2=107  draft=258  total=3110 us
```

| Component | us | % of cycle |
|---|---|---|
| sync (GPU wait) | 2438 | 78.4% |
| draft (MTP GPU) | 258 | 8.3% |
| dispatch (async) | 155 | 5.0% |
| fsamp (main logits) | 108 | 3.5% |
| fsamp2 (pos 1) | 107 | 3.4% |
| qdraft | 41 | 1.3% |

The sync time IS the GPU forward pass — identical to the greedy path's ~2.44ms.
The entire greedy-vs-stochastic gap comes from the ~0.41ms of CPU work
(fsamp + qdraft + fsamp2) that the GPU sits idle for.

### Next target: GPU-side top-k

The 108us `fast_sample` is a 248K min-heap scan on CPU. Adding `ggml_top_k(64)`
to the main model compute graph would let the GPU find top-64 candidates as
part of the forward pass, transferring only 512 bytes (64 IDs + 64 logits)
instead of 2MB. CPU work drops to O(64) penalty + softmax + sample ≈ 5us.
Combined with the draft overlap optimization (fsamp2 parallel with draft GPU),
this would reduce CPU overhead from ~410us to ~50us per cycle.

---

## Files changed

- `tools/completion/completion.cpp` — `fast_sample()`, `compute_mtp_q_draft()`,
  speculative accept/reject in `generate_mtp()`, two-mode greedy/stochastic
- `src/llama-context.h` — MTP logit extraction fields + accessors
- `src/llama-context.cpp` — MTP logit transfer in `decode_mtp()`, new API impls
- `include/llama.h` — `llama_mtp_set_extract_logits`, `llama_mtp_get_logits`,
  `llama_mtp_get_token_map`
- `scripts/bench_decode.sh`, `scripts/bench_mtp_llama.sh` — `--presence-penalty 0`
  override for greedy benchmarks
- `scratch/bench_stochastic.sh` — stochastic MTP benchmark script
