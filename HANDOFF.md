# Agent Handoff Notes

Context for the continuation agent starting outside the Docker sandbox.

---

## Current State (2026-03-11)

### Performance
- **493 tok/s** end-to-end decode (100 tokens), **+10.6% vs llama.cpp** (446 tok/s)
- **30/30 exact greedy token match** vs llama.cpp across all test prompts
- Forward pass: **1.70ms** (588 tok/s pure), CUDA graph overhead ~0.33ms
- All correctness tests passing (dp4a, CUTLASS GEMM, token match, determinism)

### What Was Just Completed (this session)
1. **4 kernel fusions** cutting ~120 kernel launches per decode step:
   - GEMV + residual add (4 sites/layer = 96 total)
   - SwiGLU + Q8_1 quantize (1 site/layer = 24 total)
   - RMSNorm + Q8_1 quantize (multiple sites)
   - Gated RMSNorm + Q8_1 quantize (DeltaNet layers)
2. **CUTLASS GEMM** replacing cuBLAS for prefill (cuBLAS dependency fully removed)
3. Blog post `blog/07-kernel-fusion-cutlass.md`
4. Comprehensive docs: `PERFORMANCE.md`, `EVAL_HARNESS_PLAN.md`
5. Scripts: `scripts/test_correctness.sh`, `scripts/nsys_profile.sh`, `scripts/ncu_profile.sh`, `scripts/dump_weight_sizes.py`

### Commits
```
df0b869 Add evaluation harness plan for host-side profiling completion
9de7824 Add PERFORMANCE.md analysis guide + profiling/testing scripts
e1e1bb5 Kernel fusion + CUTLASS GEMM: 452 → 493 tok/s (+10.6% vs llama.cpp)
```

---

## Immediate TODO (Host-Side, Requires Permissions)

### 1. Enable ncu profiling (HIGHEST PRIORITY)
```bash
# One-time host setup:
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-perf.conf
sudo modprobe -r nvidia && sudo modprobe nvidia
# Or for Docker: --cap-add SYS_ADMIN
```

Why: ncu is the ONLY way to understand why small GEMVs are at 37-55% bandwidth efficiency while the LM head achieves 93%. This is the #1 performance gap.

### 2. Fix nsys trace export
The Docker nsys (2025.4.1) generates `.qdstrm` but can't convert to `.nsys-rep`. On host with working nsys, capture a clean trace:
```bash
nsys profile --trace=cuda --output=profiles/gwen_decode \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "The meaning of life is" --n-predict 100 --greedy
```

### 3. Run the ncu investigations
See `EVAL_HARNESS_PLAN.md` Section 3 for exact commands. Priority order:
- **3B: Small GEMV diagnosis** — why 37% BW? Check `smsp__warps_issue_stalled_long_scoreboard`, register count, occupancy
- **3A: LM head verification** — confirm 93% BW claim
- **3C: Fused kernel check** — verify fusions actually helped
- **3D: DeltaNet compute vs memory** — is it hitting L1 well?

---

## Key Open Questions

### Q1: Why are small GEMVs at 37-55% bandwidth?
The per-layer GEMVs (1024→3584, 1024→2048) only achieve 40-55% of peak BW.
Hypotheses (ncu will tell us which):
1. **Insufficient memory requests**: Too few warps to saturate GDDR7 bus (256-bit)
2. **Register pressure**: >42 regs/thread → reduced occupancy → fewer in-flight requests
3. **Block size mismatch**: NW=2 (64 threads) may not be enough for small matrices
4. **L2 cache thrashing**: Multiple small GEMVs contending for cache
5. **DRAM bank conflicts**: Q4_K interleaved layout causing non-coalesced access

Action: Run ncu on `kernel_gemv_q4_k_dp4a<2>` for a 3584×1024 matrix and check the stall breakdown.

### Q2: Is there room for a multi-row GEMV?
For small matrices, processing 2+ output rows per block increases arithmetic intensity and might fill the memory pipeline better. But it also increases shared memory usage. Worth prototyping if ncu confirms hypothesis #1.

### Q3: Persistent kernels — worth the rewrite?
Current overhead breakdown:
- Forward kernels: 1.70ms
- CUDA graph replay + sync: ~0.33ms (16% overhead!)
- A single persistent kernel that runs the entire decode step would eliminate graph overhead
- But it's a MAJOR rewrite (manual scheduling, barrier synchronization across SMs)
- Potential gain: 493 → ~570 tok/s if forward stays at 1.70ms

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
  Forward:      1.70 ms → 588 tok/s (36.9% of theoretical)
  E2E decode:   2.03 ms → 493 tok/s (30.9% of theoretical)

Gap breakdown (1.70ms vs 0.627ms theoretical = 1.073ms gap):
  Small GEMV inefficiency:    ~0.50ms (at 37-55% BW vs 100%)
  LM head overhead:           ~0.02ms (already at 93%)
  DeltaNet:                   ~0.15ms (sequential recurrence)
  Kernel launch/scheduling:   ~0.10ms (even with graph)
  Other (norms, acts, quant): ~0.25ms (reduced by fusions)
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
| `src/kernels/gemv.cu` | dp4a GEMV (perf-critical, fused residual add) |
| `src/kernels/activation.cu` | SwiGLU + fused SwiGLU+Q8_1 |
| `src/kernels/rmsnorm.cu` | RMSNorm + fused RMSNorm+Q8_1 |
| `src/kernels/gemm_cutlass.cu` | CUTLASS 2.x GEMM (replaced cuBLAS) |
| `src/inference.cu` | Forward pass + gated RMSNorm fusion kernel |

---

## Suggested Next Steps (Priority Order)

1. **Enable ncu, profile small GEMVs** → understand the 37% BW problem
2. **Enable nsys, verify kernel launch counts** → confirm fusions reduced launches
3. **Based on ncu findings**: prototype fix (likely multi-row GEMV or occupancy tuning)
4. **Build micro-benchmark suite** (see EVAL_HARNESS_PLAN.md §4)
5. **Set up regression baseline** (see EVAL_HARNESS_PLAN.md §5)
6. **Consider persistent kernel** if graph overhead stays at 0.33ms
7. **Blog post #8** documenting profiling results and next optimization
