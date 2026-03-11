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
| `nsys_profile.sh` | Needs host | nsys importer broken in this Docker image |
| `ncu_profile.sh` | Needs host | Requires `CAP_SYS_ADMIN` or host `NVreg_RestrictProfilingToAdminUsers=0` |

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
| `PERFORMANCE.md` | Complete — theoretical analysis, nsys/ncu methodology, roofline, bottleneck analysis |
| `blog/07-kernel-fusion-cutlass.md` | Complete — documents the fusion work |

---

## What Needs To Be Done on Host

### 1. Fix nsys trace export

The Docker nsys (2025.4.1) generates `.qdstrm` but can't convert to `.nsys-rep`.
On host with working nsys:

```bash
# Capture
nsys profile --trace=cuda --output=profiles/gwen_decode \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "The meaning of life is" --n-predict 100 --greedy

# Verify .nsys-rep was generated
ls profiles/gwen_decode.nsys-rep

# Get kernel stats
nsys stats --report cuda_gpu_kern_sum profiles/gwen_decode.nsys-rep
nsys stats --report cuda_api_sum profiles/gwen_decode.nsys-rep
```

**Key things to check in nsys:**
1. Count kernel launches per decode step (should be ~250 after fusion)
2. Inter-kernel gaps in the timeline (visible in nsys-ui)
3. `cudaGraphLaunch` time per replay
4. Total CPU API overhead vs GPU kernel time

### 2. Enable ncu profiling

```bash
# On host, one-time setup:
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-perf.conf
sudo modprobe -r nvidia && sudo modprobe nvidia

# Or for Docker:
docker run --gpus all --cap-add SYS_ADMIN ...
```

Then run the ncu script or individual kernels:

```bash
# All kernels (one decode step, ~250 launches)
./scripts/ncu_profile.sh "" all_kernels

# Just the LM head GEMV (the single most important kernel)
./scripts/ncu_profile.sh "kernel_gemv_q6_k_dp4a" lm_head

# DeltaNet recurrence
./scripts/ncu_profile.sh "kernel_deltanet_decode" deltanet

# The fused kernels (verify they're efficient)
./scripts/ncu_profile.sh "kernel_rmsnorm_quantize_q8_1" fused_rmsnorm
./scripts/ncu_profile.sh "kernel_swiglu_quantize_q8_1" fused_swiglu
./scripts/ncu_profile.sh "kernel_gated_rmsnorm_quantize_q8_1" fused_gated_rmsnorm
```

### 3. Specific ncu investigations to run

#### A. Verify LM head bandwidth (should be ~93%)

```bash
ncu --kernel-name "kernel_gemv_q6_k_dp4a" --set full \
    --launch-skip 300 --launch-count 1 \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "test" --n-predict 5 --greedy
```

Check:
- `dram__bytes_read.sum` should be ~208 MB (LM head weight)
- `dram__throughput.avg_pct_of_peak_sustained_elapsed` should be >85%
- `sm__warps_active.avg_pct_of_peak_sustained_active` (occupancy)

#### B. Diagnose why small GEMVs are slow (37% BW efficiency)

The small per-layer GEMVs (1024→3584, 1024→2048) only achieve 40-55% bandwidth.
Profile a few:

```bash
# FFN gate: 1024→3584, Q4_K, NW=2 (64 threads)
ncu --kernel-name "kernel_gemv_q4_k_dp4a<2>" --set full \
    --launch-skip 305 --launch-count 1 \
    ./build/gwen ...
```

Check:
- `smsp__warps_issue_stalled_long_scoreboard.avg` — if high, memory latency limited
- `launch__registers_per_thread` — if >42, reduced occupancy
- `sm__warps_active.avg_pct_of_peak_sustained_active` — target >50%
- `dram__bytes_read.sum` vs expected (are we over-reading?)

#### C. Profile the fused kernels

```bash
# Fused RMSNorm+Q8_1 — should be faster than the 2-kernel version
ncu --kernel-name "kernel_rmsnorm_quantize_q8_1" --set full ...
```

Check:
- Is it actually faster? Compare `gpu__time_duration` vs sum of old unfused kernels
- Is shared memory usage reasonable?
- Any unexpected DRAM traffic?

#### D. DeltaNet recurrence — compute vs memory

```bash
ncu --kernel-name "kernel_deltanet_decode" --set full ...
```

Check:
- `sm__throughput` vs `dram__throughput` — which is the bottleneck?
- State matrix [128,128] FP32 = 64 KB per head — fits in L1?
- `l1tex__throughput` — are we making good use of L1?

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

## Priority Order

1. **Enable ncu on host** — this is the single most important thing for understanding
   why small GEMVs are at 37% BW. It tells us exactly what's wrong.
2. **Fix nsys** — needed for launch overhead analysis and graph replay profiling
3. **Run ncu investigation B** (small GEMV analysis) — this drives the next optimization
4. **Build micro-benchmark suite** — makes regression testing reliable
5. **Set up regression baseline** — prevents performance regressions

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
  Current GWEN:       1.70 ms  →  587 tok/s (forward), 493 tok/s (e2e)
```
