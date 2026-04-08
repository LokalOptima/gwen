# Post 39: AVX2 Top-k Scan — Optimizing fast_sample

The `fast_sample` function scans 248K logits to find the top-64 candidates
before applying penalties and sampling. This post documents replacing the
std::vector min-heap with a stack-allocated sorted array + AVX2 SIMD scan,
cutting per-call time from 108us to 49us.

**Date**: 2026-04-08

---

## The bottleneck

Per-cycle profiling from post 38 showed `fast_sample` at ~108us per call,
called 2× per accept cycle (position 0 verify + position 1 sample). That's
~216us of CPU idle time per cycle where the GPU sits waiting — 7% of the
~3100us cycle.

The hot path: a 248K-element min-heap scan. For each of the 248,320 logits,
compare against the heap minimum, and on the rare insert (~200 times total
before threshold stabilizes), do O(log 64) heap operations.

---

## Failed approach: GPU-side ggml_top_k

First attempt: add `ggml_top_k(logits, 64)` to the compute graph, letting CUB
`DeviceTopK::MaxPairs` find the top-64 on GPU. Transfer 256 bytes instead of
scanning 248K on CPU.

Result: **greedy dropped from 637 → 447 tok/s** (–30%). CUB's dynamic temp
storage allocation (`ggml_cuda_pool_alloc` inside the kernel) breaks CUDA graph
capture. The backend falls back to synchronous per-op dispatch instead of
captured graph replay. The `dispatch` time in per-cycle traces jumped from
~155us to ~3700us — the GPU forward pass was no longer async.

Reverted immediately.

---

## What worked: sorted array + AVX2 threshold scan

Three changes, each independently beneficial:

### 1. Heap → sorted array (no SIMD)

Replace `std::vector<Candidate>` min-heap with a stack-allocated
`top_k_candidate top[64]` sorted descending. Initialize with first 64 logits,
`std::sort` to establish threshold, then scan with binary-search insertion on
the rare hit.

Benefits:
- **No heap allocation**: stack array vs `std::vector` constructor/destructor
- **Better threshold**: sorted initial set has a higher minimum than a heap's
  `top[0]`, so fewer insertions during the scan
- **Cache-friendly insert**: `memmove` on contiguous 504-byte array vs heap
  sift-down with parent/child pointer chasing

### 2. AVX2 8-wide scan

The inner loop compares 8 logits at once against the threshold:

```cpp
__m256 thresh_v = _mm256_set1_ps(threshold);
for (int i = k; i < n_simd_end; i += 8) {
    __m256 v = _mm256_loadu_ps(&logits[i]);
    int mask = _mm256_movemask_ps(_mm256_cmp_ps(v, thresh_v, _CMP_GT_OS));
    if (!mask) continue;  // all 8 below threshold — skip
    // process rare hits
}
```

After the threshold stabilizes (~first 200 elements), 99.97% of 8-element
blocks are skipped with a single compare + movemask + branch. The scan runs
at near memory bandwidth.

### 3. Runtime dispatch

The AVX2 scan lives in its own function with `__attribute__((target("avx2")))`.
A `cpu_has_avx2()` check (one-time `__builtin_cpu_supports`, cached in
`static const bool`) dispatches at runtime. Non-AVX2 CPUs fall back to the
scalar sorted-array scan. The AVX2 tail loop calls the scalar function to
avoid code duplication.

### 4. Stack-allocated probs

`std::vector<float> probs(k)` → `float probs[K_WIDE]`. Eliminates one heap
alloc/free per fast_sample call (~200 calls per generation).

---

## Benchmark results

All benchmarks: 12 diverse prompts, 200 tokens each, RTX 5070 Ti, 3 runs per
configuration averaged. Stochastic mode: temp=1.0, top_k=20,
presence_penalty=2.0.

### Throughput (stochastic MTP)

| Version | Mean tok/s | vs heap |
|---|---|---|
| Heap (old) | 540.6 | — |
| Sorted array, scalar | 557.5 | +3.1% |
| Sorted array + AVX2 | 567.2 | +4.9% |

### Per-call fast_sample timing

| Version | Per-call | Aggregate (200 tok) |
|---|---|---|
| Heap | 108us | ~20ms |
| Sorted array + AVX2 | 49us | ~10ms |

### Greedy MTP (unaffected by fast_sample)

642 tok/s — no regression from the changes.

### Correctness

- 36/36 greedy correctness test (MTP vs base model, 12 prompts × 3 lengths)
- Bit-perfect stochastic output across all three implementations (heap, scalar
  sorted array, AVX2) with fixed seed — same md5 across 15 runs

---

## Per-cycle breakdown (after optimization)

Typical ACCEPT cycle:

| Component | us | % of cycle |
|---|---|---|
| sync (GPU wait) | 2438 | 80.8% |
| draft (MTP GPU) | 262 | 8.7% |
| dispatch (async) | 155 | 5.1% |
| fsamp (main logits) | 49 | 1.6% |
| fsamp2 (pos 1) | 52 | 1.7% |
| qdraft | 36 | 1.2% |

CPU idle time per cycle: ~137us (was ~256us). GPU sync dominates at 81%.

---

## Files changed

- `tools/completion/completion.cpp` — `top_k_candidate` struct, `top_k_insert`,
  `top_k_scan_scalar`, `top_k_scan_avx2` (with runtime dispatch), refactored
  `fast_sample` to use sorted array + stack-allocated probs
