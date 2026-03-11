# Evaluation Harness Plan

What exists, what's missing, and how to finish it outside the container.

---

## What's Done

### Scripts (all in `scripts/`)

| Script | Status | Notes |
|--------|--------|-------|
| `bench.sh` | Working | E2E decode benchmark vs llama.cpp |
| `test_correctness.sh` | Working | dp4a + GEMM + token match + determinism |
| `dump_weight_sizes.py` | Working | Exact per-tensor byte sizes from GGUF |
| `nsys_profile.sh` | Working | Use `--cuda-graph-trace=node` for graph kernel visibility |
| `ncu_profile.sh` | Working | Use `sudo ncu` with full path, or set modprobe config |

### C++ test binaries

| Binary | Status | Tests |
|--------|--------|-------|
| `test_dp4a` | Working | dp4a GEMV vs legacy FP16, all quant types |
| `test_gemm` | Working | CUTLASS GEMM vs GEMV, 9 weight matrices |
| `test_kernels` | Working | Individual kernel unit tests |
| `profile_forward` | Working | CUDA event timing for all components |

### Documentation

| File | Status |
|------|--------|
| `PERFORMANCE.md` | Complete — theoretical analysis, nsys/ncu methodology, roofline, bottleneck analysis, profiling cookbook |
| `blog/07-kernel-fusion-cutlass.md` | Complete — documents the fusion work |
| `blog/08-profiling-and-kernel-optimization.md` | Complete — ncu/nsys profiling results, argmax+GQA optimization |

---

## Completed on Host (2026-03-11)

### 1. nsys trace export — DONE

Working on host with nsys v2025.5.2. Key discovery: **must use `--cuda-graph-trace=node`**
to see individual kernels inside graph replays.

```bash
nsys profile --trace=cuda --cuda-graph-trace=node \
    -o profiles/gwen_decode_v2 \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "The meaning of life is" --n-predict 100 --greedy

nsys stats --report cuda_gpu_kern_sum profiles/gwen_decode_v2.nsys-rep
```

**Results:** nsys revealed argmax (139 µs) and GQA attention (276 µs) as hidden bottlenecks,
leading to the optimizations below.

### 2. ncu profiling — DONE

Enabled via `sudo /usr/local/cuda-13.1/bin/ncu` (full path needed — sudo doesn't inherit PATH).
Persistent config written to `/etc/modprobe.d/nvidia-perf.conf` (takes effect after reboot).

For kernel name filtering, use regex: `--kernel-name regex:"kernel_gemv_q4_k_dp4a"`
(template params in kernel names prevent exact matching).

### 3. ncu investigation results

#### A. LM head bandwidth — CONFIRMED 93%
- `dram__throughput`: 93% of peak
- 48 registers/thread (slightly above 42 target)
- Near-optimal, no changes needed

#### B. Small GEMV diagnosis — ROOT CAUSE FOUND
- DRAM throughput: 37-55% of peak
- **Root cause: poor DRAM sector utilization**
  - `block_q4_k` = 144 bytes (Array-of-Structures format)
  - Threads access `qs[0]` and `qs[4]` with 16-byte stride within each 144-byte block
  - Only **9.2 of 32 bytes** per DRAM sector contain useful data
  - This is a fundamental layout issue, not occupancy or register pressure
- **Fix: structure-of-arrays weight relayout** — group all `qs` bytes contiguously
  across blocks so warp threads access consecutive memory
- Estimated improvement: 37-55% → 70-80% BW → ~0.50ms savings → 700+ tok/s

#### C. Fused kernels — CONFIRMED BENEFICIAL
- All fused kernels faster than unfused counterparts
- Primary benefit: fewer CUDA graph nodes → less dispatch overhead
- Individual kernel compute time: 1-3 µs (negligible)

#### D. DeltaNet recurrence — LATENCY-BOUND
- Only 16 thread blocks for 70 SMs (0.23 waves — very poor SM utilization)
- State matrix [128×128] FP32 = 64 KB per head fits well in L1
- Not bandwidth-bound, not compute-bound — **latency-bound by sequential structure**
- Only fix: chunkwise recurrence for prefill (doesn't help decode)

### 4. Build a proper micro-benchmark suite

The current `profile_forward` measures kernel time with CUDA events, which is good
but doesn't isolate kernels perfectly (graph capture changes behavior). Create a
dedicated micro-benchmark that:

1. Allocates realistic-sized tensors
2. Runs each kernel type in isolation (no graph)
3. Measures with CUDA events after warmup
4. Reports: time, achieved BW, BW efficiency, and comparison to theoretical

```
# Proposed output format:
Kernel                    Shape            Type   Time(us)  BW(GB/s)  Eff%   Theoretical
─────────────────────────────────────────────────────────────────────────────────────────
gemv_q4_k_dp4a<2>        3584x1024        Q4_K     5.2      397       44.3%    2.3 us
gemv_q4_k_dp4a<4>        3584x1024        Q4_K     4.8      430       48.0%    2.3 us
gemv_q5_k_dp4a<2>        6144x1024        Q5_K     8.1      534       59.6%    4.8 us
gemv_q6_k_dp4a<4>        248320x1024      Q6_K   249.9      835       93.2%  232.8 us
rmsnorm_quantize_q8_1    1024             F32      2.0       --        --       --
swiglu_quantize_q8_1     3584             FP16     2.0       --        --       --
deltanet_decode          16x128x128       FP32     8.2       --        --       --
```

### 5. Automated regression testing

Create `scripts/regression_test.sh` that:

1. Runs `test_correctness.sh` (must pass 100%)
2. Runs `bench.sh 100` and extracts tok/s
3. Compares against a baseline stored in `benchmarks/baseline.json`
4. Fails if:
   - Any correctness test fails
   - Decode tok/s drops by >3% from baseline
   - Forward pass mean increases by >5% from baseline

```json
// benchmarks/baseline.json
{
    "date": "2026-03-11",
    "commit": "e1e1bb5",
    "gpu": "RTX 5070 Ti",
    "driver": "590.48.01",
    "clocks": "3105/14001",
    "decode_100_tok_per_s": 493,
    "forward_mean_ms": 1.70,
    "forward_min_ms": 1.58,
    "greedy_match": "30/30"
}
```

### 6. Clock management

The scripts try `nvidia-smi -lgc` but it requires root. Create a helper:

```bash
# scripts/lock_clocks.sh
#!/bin/bash
sudo nvidia-smi -lgc 3105,3105
echo "Clocks locked. Run 'sudo nvidia-smi -rgc' when done."
```

All benchmark scripts should warn (not fail) if clocks aren't locked.

---

## Priority Order (Updated)

1. ~~Enable ncu on host~~ — DONE
2. ~~Fix nsys~~ — DONE (use `--cuda-graph-trace=node`)
3. ~~Run ncu investigations~~ — DONE (all 4 completed, root cause found for GEMV)
4. **GEMV weight relayout (SoA)** — the #1 remaining optimization, root cause confirmed by ncu
5. **Build micro-benchmark suite** — isolate kernels outside CUDA graph for clean measurement
6. **Set up regression baseline** — baseline: 599 tok/s, 1.41ms forward
7. **Persistent kernel** — if graph overhead (0.26ms) stays significant after other optimizations

---

## Hardware Numbers for Reference

```
RTX 5070 Ti (SM_120, Blackwell consumer)
  Memory BW:          896 GB/s (GDDR7, 256-bit, 14001 MHz)
  L2 cache:           48 MB
  L1/shared per SM:   128 KB
  SMs:                70
  CUDA cores:         8960
  Max SM clock:       3105 MHz
  FP32 peak:          ~55.6 TFLOPS
  FP16 Tensor peak:   ~222 TFLOPS (estimated, mma.sync)
  dp4a INT8 peak:     ~111 TOPS (estimated)

Model: Qwen3.5-0.8B-Q4_K_M
  Total weights:      521,555,200 bytes (497.4 MB)
  GEMV weights:       500,699,136 bytes (477.5 MB)
  DeltaNet state:     37,748,736 bytes (36.0 MB) R+W per step
  All traffic/step:   562,229,000 bytes (536.2 MB)

Theoretical decode limits:
  Weights-only:       0.582 ms → 1718 tok/s
  All-traffic:        0.627 ms → 1594 tok/s
  Current GWEN:       1.41 ms  →  709 tok/s (forward), 599 tok/s (e2e)
```
