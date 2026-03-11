# Agent Handoff Notes

Context for the continuation agent starting outside the Docker sandbox.

---

## Current State (2026-03-11)

### Performance
- **599 tok/s** end-to-end decode (100 tokens), **+34.3% vs llama.cpp** (446 tok/s)
- **30/30 exact greedy token match** vs llama.cpp across all test prompts
- Forward pass: **1.41ms** (709 tok/s pure), CUDA graph overhead ~0.26ms
- All correctness tests passing (dp4a, CUTLASS GEMM, token match, determinism)

### What Was Just Completed (this session)
1. **ncu/nsys profiling** enabled on host — all 4 investigations completed
2. **Multi-block argmax**: 1-block → 256-block parallel reduction (139 μs → 3.5 μs, 40x)
3. **Multi-warp GQA attention**: 32-thread → 256-thread (276 μs → 47 μs, 5.9x)
4. **nsys** working with `--cuda-graph-trace=node` for per-kernel timing in graph replays
5. Blog post `blog/08-profiling-and-kernel-optimization.md`
6. Fixed `test_correctness.sh` — llama_generate now receives prompt argument

### Commits
```
df0b869 Add evaluation harness plan for host-side profiling completion
9de7824 Add PERFORMANCE.md analysis guide + profiling/testing scripts
e1e1bb5 Kernel fusion + CUTLASS GEMM: 452 → 493 tok/s (+10.6% vs llama.cpp)
(pending) Profiling + kernel optimization: 493 → 599 tok/s (+34.3% vs llama.cpp)
```

---

## Completed TODO Items

### 1. ncu profiling — DONE
- Enabled via `sudo ncu` (persistent config written to `/etc/modprobe.d/nvidia-perf.conf`, takes effect after reboot)
- All 4 investigations completed (see EVAL_HARNESS_PLAN.md for results)

### 2. nsys trace export — DONE
- Working on host with nsys v2025.5.2
- Key flag: `--cuda-graph-trace=node` (without this, graph-replayed kernels are invisible)
- Profiles saved in `profiles/gwen_decode_v2.nsys-rep`

### 3. ncu investigations — DONE
- **3A: LM head** — 93% BW confirmed, 48 regs/thread
- **3B: Small GEMV** — 37-55% BW due to poor sector utilization (9.2/32 bytes per sector), Q4_K strided access pattern
- **3C: Fused kernels** — confirmed faster than unfused versions
- **3D: DeltaNet** — only 16 blocks / 70 SMs, but inherently sequential; L1 utilization good

## Immediate TODO

### 1. GEMV coalescing optimization (HIGHEST PRIORITY)
Root cause identified by ncu: `block_q4_k` is 144 bytes, threads access non-contiguous fields with 16-byte stride. Only 9.2 of 32 bytes per DRAM sector are useful.
Fix: structure-of-arrays weight relayout. Estimated gain: small GEMVs from 37-55% to 70-80% BW → forward under 1.0 ms → 700+ tok/s.

### 2. Build micro-benchmark suite
See EVAL_HARNESS_PLAN.md §4. Isolate individual kernels outside CUDA graph for clean measurement.

### 3. Automated regression testing
See EVAL_HARNESS_PLAN.md §5. Baseline: 599 tok/s, 1.41ms forward.

---

## Key Open Questions

### Q1: Why are small GEMVs at 37-55% bandwidth? — ANSWERED
The per-layer GEMVs (1024→3584, 1024→2048) achieve 40-55% of peak BW.
**Root cause (confirmed by ncu)**: Poor DRAM sector utilization.
- `block_q4_k` = 144 bytes (AoS format). Threads in a warp each access a different block.
- Access pattern: `qs[0]` and `qs[4]` with 16-byte stride within each 144-byte block.
- Only **9.2 of 32 bytes** per DRAM sector are useful data.
- Hypothesis #5 (non-coalesced access) was correct.
- Fix: structure-of-arrays weight relayout to group all `qs` contiguously.

### Q2: Is there room for a multi-row GEMV?
For small matrices, processing 2+ output rows per block increases arithmetic intensity and might fill the memory pipeline better. But it also increases shared memory usage. Worth prototyping if ncu confirms hypothesis #1.

### Q3: Persistent kernels — worth the rewrite?
Current overhead breakdown:
- Forward kernels: 1.41ms
- CUDA graph replay + sync: ~0.26ms (16% overhead)
- A single persistent kernel that runs the entire decode step would eliminate graph overhead
- But it's a MAJOR rewrite (manual scheduling, barrier synchronization across SMs)
- Potential gain: 599 → ~700 tok/s if forward stays at 1.41ms

### Q4: Prefill CUTLASS — fused dequant+GEMM?
Current prefill pipeline: dequant Q4_K→FP16 temp buffer → CUTLASS GEMM on FP16.
A fused kernel would halve prefill memory traffic. CUTLASS 4.x supports custom mainloop fusion (EVT), but SM_120 support is unclear. Investigate:
- Does CUTLASS 4.x have SM_120 mainloop examples?
- Can we use `cute::Copy_Atom` with a custom dequant iterator?
- Alternative: just use the 2.x dequant→GEMM pipeline (it works, prefill isn't the bottleneck)

### Q5: DeltaNet chunkwise recurrence for prefill?
Currently DeltaNet layers fall back to sequential per-token recurrence even during prefill. A chunkwise algorithm (chunk size 64 or 128) could parallelize within chunks using matrix operations. This is a significant research+implementation effort but could dramatically improve TTFT.

---

## Theoretical Limits (for reference)

```
Total traffic per decode step: 562.2 MB
  - GEMV weights:     500.7 MB (read)
  - DeltaNet state:    37.7 MB (read + write)
  - Activations:       23.8 MB (estimated)

RTX 5070 Ti peak BW: 896 GB/s

Theoretical minimum forward pass:
  Weights-only: 500.7 MB / 896 GB/s = 0.559 ms → 1790 tok/s
  All-traffic:  562.2 MB / 896 GB/s = 0.627 ms → 1594 tok/s

Current:
  Forward:      1.41 ms → 709 tok/s (44.5% of theoretical)
  E2E decode:   1.67 ms → 599 tok/s (37.6% of theoretical)

Gap breakdown (1.41ms vs 0.627ms theoretical = 0.783ms gap):
  Small GEMV inefficiency:    ~0.50ms (at 37-55% BW vs 100%)
  LM head overhead:           ~0.02ms (already at 93%)
  DeltaNet:                   ~0.15ms (sequential recurrence)
  GQA attention:              ~0.05ms (optimized from 0.28ms)
  Other (norms, acts, quant): ~0.03ms (reduced by fusions + argmax fix)
```

---

## File Map

| File | Purpose |
|------|---------|
| `PERFORMANCE.md` | Full theoretical analysis methodology, roofline, profiling guide |
| `EVAL_HARNESS_PLAN.md` | Detailed TODO list for host-side profiling |
| `scripts/bench.sh` | E2E benchmark vs llama.cpp |
| `scripts/test_correctness.sh` | Full correctness suite (4 test categories) |
| `scripts/nsys_profile.sh` | nsys capture + kernel stats |
| `scripts/ncu_profile.sh` | ncu capture + per-kernel analysis |
| `scripts/dump_weight_sizes.py` | Exact GGUF tensor byte sizes |
| `blog/07-kernel-fusion-cutlass.md` | Documents the fusion + CUTLASS work |
| `blog/08-profiling-and-kernel-optimization.md` | ncu/nsys profiling + argmax/GQA optimization |
| `profiles/gwen_decode_v2.nsys-rep` | nsys trace with node-level CUDA graph tracing |
| `profiles/gemv_q4k.raw.txt` | ncu full profile of Q4_K GEMV |
| `profiles/lm_head.raw.txt` | ncu full profile of Q6_K LM head GEMV |
| `profiles/deltanet.raw.txt` | ncu full profile of DeltaNet kernel |
| `profiles/fused_kernels.raw.txt` | ncu full profile of fused kernels |
| `src/kernels/gemv.cu` | dp4a GEMV (perf-critical, fused residual add) |
| `src/kernels/activation.cu` | SwiGLU + fused SwiGLU+Q8_1 |
| `src/kernels/rmsnorm.cu` | RMSNorm + fused RMSNorm+Q8_1 |
| `src/kernels/gemm_cutlass.cu` | CUTLASS 2.x GEMM (replaced cuBLAS) |
| `src/inference.cu` | Forward pass + gated RMSNorm fusion kernel |

---

## Suggested Next Steps (Priority Order)

1. **GEMV weight relayout (SoA)** → fix the 37-55% BW coalescing problem → estimated 700+ tok/s
2. **Build micro-benchmark suite** (see EVAL_HARNESS_PLAN.md §4)
3. **Set up regression baseline** (see EVAL_HARNESS_PLAN.md §5) — baseline: 599 tok/s
4. **Consider persistent kernel** if graph overhead stays at 0.26ms
5. **DeltaNet chunkwise prefill** for faster TTFT
