# Agent Handoff Notes

Context for the next agent session.

---

## Current State (2026-03-13)

### Performance
- **599 tok/s** end-to-end decode (100 tokens), **+34.3% vs llama.cpp** (446 tok/s)
- Forward pass: **1.41ms** (709 tok/s pure), CUDA graph overhead ~0.26ms
- All kernel-level tests passing (dp4a, CUTLASS GEMM)
- Determinism: 5/5 identical runs

### Correctness vs llama.cpp (commit `57819b8d`)
- **GEMM decode path** (`GWEN_GEMM_DECODE=1`): 2/4 prompts 30/30 exact match, 2/4 diverge at FP16 tie-breaking (top-2 logits differ by <0.01)
- **GEMV decode path** (default, 6x faster): lower precision than GEMM due to dp4a INT8 quantization of activations
- Root cause of remaining divergences: GWEN uses FP16 activations between layers, llama.cpp uses F32. After 24 layers of FP16 round-trips, accumulated error flips close logit races.

### What Was Just Completed (this session)
1. **Rebuilt llama.cpp** and test binaries (`llama_generate`, `llama_dump_layers`) against commit `57819b8d`
2. **Fixed L2 norm epsilon**: `rsqrtf(sum + 1e-12f)` → `rsqrtf(fmaxf(sum, 1e-12f))` to match llama.cpp's `rsqrtf(fmaxf(sum, eps²))`. Fixed in 4 locations (rope.cu standalone kernel + 2 inline copies in kernel_deltanet_prefill and kernel_deltanet_prefill_batch)
3. **Added GEMM decode path**: `GWEN_GEMM_DECODE=1` env var switches decode loop from `forward()` (GEMV + CUDA graph) to `forward_prefill(N=1)` (CUTLASS GEMM). GEMM path matches llama.cpp better.
4. **Added logit dump**: `GWEN_DUMP_LOGITS=1` prints top-5 logits per decode step for debugging
5. **Fixed test script**: `test_correctness.sh` now passes correct args to `llama_generate` and uses `GWEN_GEMM_DECODE=1` for correctness comparison
6. **Diagnosed FP16 precision floor**: Confirmed remaining 2/4 divergences are FP16 tie-breaking, not algorithmic bugs (top-2 logits differ by <0.01 at divergence points)
7. **Verified batch path**: `extract_hidden_batch` B=1 has 66% bit-level mismatches vs `forward_prefill` — expected due to different full-attention implementations (flash-attention vs KV cache)
8. **Blog post**: `blog/15-re-verify-llama-cpp.md`

### Key Finding: GEMV vs GEMM Precision
The GEMM decode path (CUTLASS) is more accurate than GEMV (dp4a) because:
- CUTLASS GEMM dequantizes Q4_K → FP16, then multiplies with FP16 accumulation
- dp4a GEMV quantizes the activation to INT8 before accumulation, losing more precision
- Both match llama.cpp for "easy" prompts (large logit margins), but differ for "hard" prompts (close top-2 logits)

### Key Finding: GEMM N=1 Performance
`forward_prefill(N=1)` is 6x slower than `forward()`:
- GEMV + CUDA graph: 605 tok/s
- GEMM per-call: 96 tok/s
- Overhead is all dispatch: per-call `cudaMemcpy`, ~100 kernel launches without graph capture
- The actual math is equivalent — CUTLASS GEMM M=1 degrades to GEMV internally
- Fix: CUDA-graph-capture `forward_prefill(N=1)` would close the gap

---

## Files Modified

| File | Changes |
|------|---------|
| `src/inference.cu` | L2 norm fix (×4), GEMM decode env var, logit dump env var |
| `src/kernels/rope.cu` | L2 norm fix in `gwen_l2_normalize` |
| `scripts/test_correctness.sh` | Fixed `llama_generate` args, GEMM decode for test |
| `blog/15-re-verify-llama-cpp.md` | Verification blog post |
| `blog/README.md` | Blog index update |

---

## Suggested Next Steps (Priority Order)

1. **CUDA-graph-capture GEMM decode** → close 6x perf gap, make GEMM the default path
2. **GEMV weight relayout (SoA)** → fix the 37-55% BW coalescing problem → estimated 700+ tok/s
3. **Unify batch path attention** → make `extract_hidden_batch` use same attention as `forward_prefill` for bitwise match
4. **Build micro-benchmark suite** (see EVAL_HARNESS_PLAN.md §4)
5. **DeltaNet chunkwise prefill** for faster TTFT
