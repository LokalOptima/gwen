# GWEN Performance Analysis Guide

Rigorous methodology for theoretical analysis, profiling, and benchmarking.
Every number must be traceable to either a hardware datasheet or a measurement tool.

---

## 1. Hardware Specifications (RTX 5070 Ti)

Source: nvidia-smi, NVIDIA product specs.

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | Blackwell (SM_120) | `nvidia-smi -q` |
| SMs | 70 | Product spec |
| CUDA cores | 8960 (128/SM) | Product spec |
| Tensor cores | 280 (4/SM, 5th gen) | Product spec |
| Max SM clock | 3105 MHz | `nvidia-smi -q -d CLOCK` |
| Max memory clock | 14001 MHz (GDDR7) | `nvidia-smi -q -d CLOCK` |
| Memory bus | 256-bit | Product spec |
| **Memory bandwidth** | **896 GB/s** | 14001 MHz × 256-bit × 2 (DDR) / 8 = 896 GB/s |
| VRAM | 16 GB GDDR7 | `nvidia-smi` |
| L2 cache | 48 MB | Product spec |
| L1/shared per SM | 128 KB (up to 99 KB shared) | Product spec |
| Registers per SM | 65536 (32-bit) | Product spec |
| Max warps per SM | 48 | Product spec |
| FP16 Tensor TOPS | ~N/A (use measured) | mma.sync throughput |
| INT8 dp4a TOPS | ~N/A (use measured) | dp4a throughput per SM |
| TDP | 300W | `nvidia-smi -q` |

### How to verify bandwidth

```bash
# Run CUDA bandwidthTest (comes with CUDA samples)
# Or measure with a simple kernel:
# Achieved bandwidth = bytes_transferred / kernel_time
# Compare to 896 GB/s theoretical
```

### Clock locking for reproducible measurements

```bash
# Lock to max boost clock (requires root)
sudo nvidia-smi -lgc 3105,3105

# Verify
nvidia-smi -q -d CLOCK | grep -A2 "Clocks$"

# Unlock when done
sudo nvidia-smi -rgc
```

**Always lock clocks before benchmarking.** GPU boost behavior causes 5-15% variance.

---

## 2. Theoretical Analysis: Decode (Bandwidth-Bound)

### Principle

Single-token decode is **memory-bandwidth-bound**: each token reads all model weights
once but performs very little compute per weight byte. The arithmetic intensity is:

```
AI = FLOPs / Bytes = 2 * out_features / bytes_per_element ≈ 0.5-4 FLOP/byte
```

For Q4_K: `AI = 2 * 256 / 144 = 3.6 FLOP/byte` (per super-block).
The RTX 5070 Ti roofline crossover is at ~100-200 FLOP/byte, so decode is firmly BW-bound.

### Computing exact memory traffic

Run `scripts/dump_weight_sizes.py` on the GGUF file to get precise per-tensor byte counts:

```bash
python3 scripts/dump_weight_sizes.py Qwen3.5-0.8B-Q4_K_M.gguf
```

#### Quantization block sizes (from GGML spec)

| Type | Block elements | Block bytes | Bits/element |
|------|---------------|-------------|-------------|
| Q4_K | 256 | 144 | 4.50 |
| Q5_K | 256 | 176 | 5.50 |
| Q6_K | 256 | 210 | 6.56 |
| Q8_0 | 32 | 34 | 8.50 |
| Q8_1 | 32 | 36 | 9.00 |
| F32 | 1 | 4 | 32.00 |
| F16 | 1 | 2 | 16.00 |

#### Weight bytes formula

```
weight_bytes(n_elements, type) = (n_elements / block_elements) * block_bytes
```

#### Per-decode-step traffic breakdown

**Weights (read once per token):**

| Component | Tensors | Total bytes | MB |
|-----------|---------|------------:|---:|
| 18 DeltaNet layers | QKV, gate, ssm_out, ffn×3, norms, conv, α/β | 248,536,320 | 237.0 |
| 6 Full Attention layers | Q, K, V, output, ffn×3, norms | 64,425,984 | 61.4 |
| LM head (=token_embd) | [1024, 248320] Q6_K | 208,588,800 | 198.9 |
| Output norm | [1024] F32 | 4,096 | 0.0 |
| **Total weights** | | **521,555,200** | **497.4** |

**Activations & state (non-weight traffic):**

| Component | Formula | Bytes |
|-----------|---------|------:|
| Q8_1 input vectors | Per GEMV: ceil(in_features/32) × 36 | 270,720 |
| DeltaNet state R+W | 18 × 2 × 16 × 128 × 128 × 4 | 37,748,736 |
| Conv1D state R+W | 18 × 2 × 3 × 6144 × 4 | 2,654,208 |
| FP16 activations | Residuals, norms, etc. | ~160,000 |
| **Total non-weight** | | **~40,834,000** |

**Grand total: ~562 MB per decode step**

#### Theoretical minimum time

```
t_min = total_bytes / bandwidth
      = 562,229,000 / 896,000,000,000
      = 0.627 ms  →  1594 tok/s
```

Weights-only: `521,555,200 / 896e9 = 0.582 ms → 1718 tok/s`

### Per-GEMV theoretical time

For any single GEMV with weight matrix W of shape [out, in]:

```
bytes = weight_bytes(out * in, quant_type) + ceil(in/32) * 36  # weight + Q8_1 input
t_min = bytes / 896e9
achieved_bw = bytes / measured_time
efficiency = achieved_bw / 896e9
```

Example — LM head [248320, 1024] Q6_K:
```
bytes = (248320 * 1024 / 256) * 210 + (1024/32) * 36
      = 208,588,800 + 1,152
      = 208,589,952
t_min = 208,589,952 / 896e9 = 0.2328 ms = 232.8 µs
```

Measured: 249.9 µs → efficiency = 232.8 / 249.9 = **93.2%** ← excellent.

---

## 3. Theoretical Analysis: Prefill (Compute-Bound)

Prefill processes `seq_len` tokens in parallel using GEMM. With `seq_len > ~32`,
most GEMMs become compute-bound.

### Arithmetic intensity of GEMM

```
FLOPs = 2 * M * N * K
Bytes = M * K * bytes_per_elem + K * N * bytes_per_elem + M * N * bytes_per_elem
AI = 2*M*N*K / (M*K + K*N + M*N) / bytes_per_elem
```

For FP16 GEMM with large M,N,K: AI ≈ `min(M, N) / 2` FLOP/byte.
At seq_len=64, K=1024, M=3584: AI ≈ 32 FLOP/byte — in the compute-bound regime.

### FP16 Tensor Core peak

The RTX 5070 Ti with mma.sync (Sm80-compatible path):
- 280 Tensor Cores, each does 128 FP16 FMA/cycle = 256 FP16 ops/cycle
- At 3105 MHz: `280 × 256 × 3.105 GHz = 222 TFLOPS` (theoretical peak, halved for effective)
- Practical peak: measure with a large square GEMM (e.g., 4096×4096×4096)

---

## 4. Profiling with nsys (Timeline Analysis)

nsys captures a timeline of all GPU activity — kernel launches, memcpy, synchronization.
Use it to identify:

- **Launch overhead**: gaps between kernels (host-side launch latency)
- **Serialization**: kernels that could overlap but don't
- **Idle time**: GPU waiting for CPU
- **Memory transfer bottlenecks**: large H2D/D2H copies

### Prerequisites

nsys works without special permissions for CUDA trace capture. However:
- CPU sampling requires `CAP_SYS_ADMIN` or `perf_event_paranoid <= 1`
- GPU metrics (`--gpu-metrics-device`) may not work on all GPU generations
- In Docker: `--cap-add SYS_ADMIN` enables full features

### Capture a profile

```bash
./scripts/nsys_profile.sh [output_name]
```

Or manually:

```bash
nsys profile \
    --trace=cuda \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    -o profiles/my_trace \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "The meaning of life is" --n-predict 100 --greedy
```

### Key nsys reports

```bash
# Kernel time summary (sorted by total GPU time)
nsys stats --report cuda_gpu_kern_sum profiles/my_trace.nsys-rep

# CUDA API call summary (launch overhead)
nsys stats --report cuda_api_sum profiles/my_trace.nsys-rep

# Memory operations
nsys stats --report cuda_mem_size_sum profiles/my_trace.nsys-rep
```

### What to look for

1. **Total kernel time vs wall time**: If wall > kernel, there's CPU overhead or sync stalls
2. **Kernel launch frequency**: Count launches per decode step (target: <300)
3. **Inter-kernel gaps**: In nsys-ui, zoom into a decode step — are there visible gaps?
4. **cudaGraphLaunch time**: Should be <0.5ms per replay
5. **Memory copies during decode**: Should be exactly 1 (the 8-byte token_id+pos copy)

### Interpreting CUDA graph traces

With CUDA graphs, nsys shows the graph capture as individual kernels, then each
`cudaGraphLaunch` as a single event. To see per-kernel breakdown inside the graph:

**Use `--cuda-graph-trace=node`** — this is critical! Without it, graph-replayed
kernels are invisible and you only see prefill/capture kernels.

```bash
nsys profile --trace=cuda --cuda-graph-trace=node \
    -o profiles/gwen_decode \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "The meaning of life is" --n-predict 100 --greedy
```

This flag makes nsys decompose each `cudaGraphLaunch` into its constituent kernel
nodes, giving you per-kernel timing for every decode step — exactly what you need
to find bottlenecks in the graph replay path.

---

## 5. Profiling with ncu (Per-Kernel Analysis)

ncu provides deep per-kernel metrics. It replays kernels multiple times to
collect hardware counters, so it's much slower than nsys but gives definitive answers.

### Prerequisites: GPU performance counter access

ncu requires access to GPU hardware performance counters. This is restricted by default.

**On bare metal (host):**
```bash
# Persistent fix (survives reboot)
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-perf.conf
sudo modprobe -r nvidia && sudo modprobe nvidia
# Or one-time: sudo ncu ...
```

**In Docker:**
```bash
# Run container with --cap-add SYS_ADMIN or --privileged
docker run --gpus all --cap-add SYS_ADMIN ...
```

Without these, ncu will report `ERR_NVGPUCTRPERM`.

### Capture a profile

```bash
./scripts/ncu_profile.sh [kernel_name_filter] [output_name]
```

Or manually:

```bash
sudo /usr/local/cuda-13.1/bin/ncu --set full \
    --launch-skip 15 --launch-count 50 \
    --csv --log-file profiles/kernel.csv \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "test" --n-predict 5 --greedy

# NOTE: Use full path with sudo since PATH is not inherited.
# To filter by kernel name, use regex: --kernel-name regex:"kernel_gemv_q4_k"
# Template params in kernel names require regex matching.
```

### Key metrics and what they mean

#### Memory bandwidth (is this kernel bandwidth-bound?)

| Metric | What it tells you |
|--------|-------------------|
| `dram__bytes_read.sum` | Actual bytes read from VRAM |
| `dram__bytes_write.sum` | Actual bytes written to VRAM |
| `dram__throughput.avg_pct_of_peak_sustained_elapsed` | % of peak DRAM bandwidth used |
| `l2__throughput.avg_pct_of_peak_sustained_elapsed` | % of peak L2 bandwidth used |

**Interpretation:**
- `dram_throughput > 70%` → bandwidth-bound (good for GEMV)
- `dram_throughput < 30%` → likely compute-bound or latency-bound
- `l2_throughput >> dram_throughput` → data fits in L2 cache

**Achieved bandwidth calculation:**
```
achieved_bw = (dram_bytes_read + dram_bytes_write) / kernel_time
efficiency = achieved_bw / 896e9
```

#### Compute utilization (is this kernel compute-bound?)

| Metric | What it tells you |
|--------|-------------------|
| `sm__throughput.avg_pct_of_peak_sustained_elapsed` | % of peak SM throughput |
| `sm__inst_executed.avg_per_cycle_elapsed` | Instructions per cycle across all SMs |

**Interpretation:**
- `sm_throughput > 70%` → compute-bound
- Both `sm` and `dram` throughput low → latency-bound (bad — likely poor occupancy or divergence)

#### Occupancy and stalls

| Metric | What it tells you |
|--------|-------------------|
| `sm__warps_active.avg_pct_of_peak_sustained_active` | Achieved occupancy |
| `smsp__warps_issue_stalled_*` | Why warps are stalled |

**Common stall reasons:**
- `stalled_long_scoreboard` → waiting for memory (DRAM/L2 latency)
- `stalled_short_scoreboard` → waiting for math pipeline
- `stalled_barrier` → waiting at `__syncthreads()`
- `stalled_not_selected` → enough warps, scheduler just picked another

**Target occupancy for GEMV:** 50-75%. Too low = can't hide memory latency.
Too high = register pressure may force spills.

#### Kernel launch configuration

| Metric | What it tells you |
|--------|-------------------|
| `launch__grid_size` | Number of thread blocks |
| `launch__block_size` | Threads per block |
| `launch__registers_per_thread` | Register usage |
| `launch__shared_mem_per_block_allocated` | Shared memory per block |

**Check:** `regs_per_thread × threads_per_block ≤ 65536` (registers per SM).
If violated, blocks can't co-reside → reduced occupancy.

### ncu workflow for a specific kernel

```bash
# 1. Find the kernel name from nsys
nsys stats --report cuda_gpu_kern_sum profiles/trace.nsys-rep | head -20

# 2. Profile just that kernel with full metrics
ncu --kernel-name "kernel_gemv_q4_k_dp4a" --set full \
    --launch-skip 15 --launch-count 10 \
    ./build/gwen ...

# 3. Check the Speed of Light (SOL) section:
#    - SM SOL: compute utilization as % of theoretical peak
#    - Memory SOL: bandwidth utilization as % of 896 GB/s
#    If both are low, you have a latency problem.
```

### Verifying theoretical bandwidth calculations with ncu

For a GEMV kernel:

```
Expected bytes = weight_bytes(out × in, type) + q8_input_bytes + output_bytes
ncu measured   = dram__bytes_read.sum + dram__bytes_write.sum
```

If `ncu measured >> expected`: the kernel is re-reading data (cache misses, poor access patterns).
If `ncu measured ≈ expected`: kernel is accessing memory efficiently.

---

## 6. Correctness Testing

### Test hierarchy

| Level | Tool | What it tests | When |
|-------|------|---------------|------|
| Unit | `./build/test_dp4a` | dp4a GEMV vs legacy FP16 GEMV | After any GEMV change |
| Unit | `./build/test_gemm` | CUTLASS GEMM vs GEMV | After any GEMM change |
| Integration | `./build/test_kernels` | Individual kernel correctness | After any kernel change |
| E2E | `./scripts/test_correctness.sh` | Full pipeline vs llama.cpp | Before any commit |
| E2E | `./scripts/bench.sh 30` | Token match + perf | Before any commit |

### Run the full correctness suite

```bash
./scripts/test_correctness.sh
```

This runs:
1. dp4a kernel unit tests (max element-wise diff < 0.01)
2. CUTLASS GEMM vs GEMV (max diff < 0.1)
3. Greedy token match vs llama.cpp (30 tokens, multiple prompts)
4. Determinism check (5 identical runs)

### Acceptance criteria

| Test | Threshold | Rationale |
|------|-----------|-----------|
| dp4a vs FP16 GEMV | Max diff < 0.01 | Quantization error is bounded |
| CUTLASS GEMM vs GEMV | Max diff < 0.1 | Different accumulation order |
| Greedy token match | 30/30 exact | Any divergence indicates a bug |
| Determinism | Bitwise identical | No random state in greedy decode |

### When tokens diverge

If greedy tokens diverge from llama.cpp:

1. **Check which token first diverges** — run with `--verbose` or dump per-token logits
2. **Bisect layers** — dump intermediate activations after each layer, compare
3. **Check quantization types** — `scripts/dump_gguf.py` shows per-tensor types
4. **Common causes**: wrong scale extraction, nibble interleave bug, off-by-one in block indexing

---

## 7. Benchmarking Methodology

### Macro benchmark (end-to-end decode)

```bash
./scripts/bench.sh 100
```

Reports: tok/s, ms/token, TTFT, VRAM, and comparison vs llama.cpp.

### Micro benchmark (per-kernel)

```bash
./build/profile_forward Qwen3.5-0.8B-Q4_K_M.gguf
```

Reports: per-kernel timing (averaged over 100 runs), achieved bandwidth.

### Rules for reliable benchmarking

1. **Lock GPU clocks**: `sudo nvidia-smi -lgc 3105,3105`
2. **Warmup**: Always run 10+ warmup iterations before measuring
3. **Measure many iterations**: Report mean, min, max over ≥50 runs
4. **Use CUDA events**: `cudaEventRecord` + `cudaEventElapsedTime` for GPU-only timing
5. **Disable display**: Close any GUI using the GPU
6. **Same thermal state**: Wait 5s after warmup for thermals to stabilize
7. **Report full context**: GPU model, driver version, CUDA version, clock settings, model file

### What to report

```
GPU:          RTX 5070 Ti (SM_120)
Driver:       590.48.01
CUDA:         13.1
Clocks:       SM 3105 MHz, Mem 14001 MHz (locked)
Model:        Qwen3.5-0.8B-Q4_K_M.gguf (497.4 MB)
Prompt:       "The meaning of life is"
Decode tokens: 100
Warmup:       10 iterations

Decode:       493 tok/s (mean), 2.03 ms/token
Forward pass: 1.70 ms mean, 1.58 ms min, 1.82 ms max (N=100)
BW efficiency: 34.2% (vs 896 GB/s theoretical)
```

---

## 8. Roofline Model

The roofline model plots achieved performance (FLOP/s or byte/s) against
arithmetic intensity (FLOP/byte). It immediately tells you whether a kernel
is compute-bound or memory-bound.

### Computing arithmetic intensity

For **decode GEMV** (single-token, one row at a time):

```
FLOPs = 2 × out_features × in_features  (1 mul + 1 add per element)
Bytes = weight_bytes + q8_input_bytes + output_bytes

AI = FLOPs / Bytes
```

| GEMV | Out | In | FLOPs | Bytes | AI (FLOP/byte) |
|------|----:|---:|------:|------:|-----------:|
| QKV (Q5_K) | 6144 | 1024 | 12,582,912 | 4,326,528 | 2.91 |
| FFN gate (Q4_K) | 3584 | 1024 | 7,340,032 | 2,065,536 | 3.55 |
| FFN down (Q6_K) | 1024 | 3584 | 7,340,032 | 3,014,592 | 2.44 |
| LM head (Q6_K) | 248320 | 1024 | 508,559,360 | 208,589,952 | 2.44 |

RTX 5070 Ti roofline ridge point (BW-bound ↔ compute-bound crossover):
```
FP32 peak ≈ 2 × 8960 × 3.105 GHz = 55.6 TFLOPS
Ridge = 55.6 TFLOPS / 896 GB/s ≈ 62 FLOP/byte
```

All decode GEMVs have AI < 4 — **firmly bandwidth-bound** (62x below the ridge).
The theoretical ceiling is the bandwidth line: `perf = AI × 896 GB/s`.

### For prefill GEMM (FP16, large batch)

```
FP16 Tensor peak ≈ 222 TFLOPS (estimated)
Ridge = 222 TFLOPS / 896 GB/s ≈ 248 FLOP/byte

FFN gate GEMM, seq=128: M=3584, N=128, K=1024
  FLOPs = 2 × 3584 × 128 × 1024 = 939,524,096
  Bytes = 3584×1024×2 + 1024×128×2 + 3584×128×2 = 7,864,320
  AI = 119 FLOP/byte → below ridge → still memory-bound!

FFN gate GEMM, seq=512: M=3584, N=512, K=1024
  AI = 476 FLOP/byte → above ridge → compute-bound ✓
```

---

## 9. Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/bench.sh [N]` | End-to-end decode benchmark vs llama.cpp |
| `scripts/test_correctness.sh` | Full correctness test suite |
| `scripts/dump_weight_sizes.py MODEL` | Print exact per-tensor byte sizes and theoretical limits |
| `scripts/nsys_profile.sh [name]` | Capture nsys timeline + kernel stats |
| `scripts/ncu_profile.sh [filter] [name]` | Capture ncu per-kernel metrics |
| `build/profile_forward MODEL` | Component-level timing with CUDA events |
| `build/test_dp4a MODEL` | dp4a GEMV unit tests |
| `build/test_gemm MODEL` | CUTLASS GEMM unit tests |

---

## 10. Current Status & Bottleneck Analysis

*Updated 2026-03-11 (post-profiling optimization). Re-run after any change.*

### Measured performance

```
Forward pass:  1.41 ms (709 tok/s pure)
Decode e2e:    1.67 ms (599 tok/s)  [includes CUDA graph overhead]
```

### Where does the time go?

| Component | Measured | Theoretical | Efficiency | % of Forward |
|-----------|---------|-------------|-----------|-------------|
| LM head GEMV | 250 µs | 233 µs | 93.2% | 17.7% |
| 24-layer GEMVs | 928 µs | 347 µs | 37.4% | 65.8% |
| DeltaNet recurrence (×18) | 148 µs | N/A (compute) | — | 10.5% |
| GQA attention (×6) | 47 µs | N/A | — | 3.3% |
| Argmax | 3.5 µs | N/A | — | 0.2% |
| All other (norms, acts, etc.) | 34 µs | ~5 µs (BW) | — | 2.4% |
| **Total forward** | **1410 µs** | **627 µs** | **44.5%** | 100% |
| CUDA graph overhead | ~260 µs | 0 | — | (extra) |

### ncu profiling results (measured)

#### Q4_K GEMV (small per-layer, e.g. FFN gate 3584×1024)
- DRAM throughput: 37-55% of peak
- Root cause: **poor sector utilization** — `block_q4_k` is 144 bytes (AoS).
  Threads access `qs[0]`, `qs[4]` with 16-byte stride within each block.
  Only **9.2 of 32 bytes** per DRAM sector are useful data.
- Register count: varies by template param (NW=2: ~40 regs, NW=4: ~48 regs)
- Occupancy: adequate but not the bottleneck

#### Q6_K GEMV (LM head 248320×1024)
- DRAM throughput: **93%** of peak — confirmed by ncu
- 48 registers/thread (slightly above 42 target, but irrelevant at this BW)
- Near-optimal; no further optimization needed

#### Fused kernels (RMSNorm+Q8_1, SwiGLU+Q8_1, GatedRMSNorm+Q8_1)
- All confirmed faster than unfused counterparts
- Primary benefit: launch overhead reduction (fewer CUDA graph nodes)
- Compute time is negligible (1-3 µs each)

#### DeltaNet recurrence
- Only 16 thread blocks for 70 SMs (0.23 waves)
- Inherently sequential — each head computes S = S + beta*k⊗v - alpha*S
- State matrix [128×128] FP32 = 64 KB per head — fits well in L1 cache
- Not bandwidth-bound, not compute-bound — latency-bound by sequential structure

### Recent optimizations (this session)

| Optimization | Before | After | Speedup |
|-------------|--------|-------|---------|
| Multi-block argmax (1→256 blocks) | 139 µs | 3.5 µs | 40x |
| Multi-warp GQA attention (32→256 threads) | 276 µs | 47 µs | 5.9x |
| Total forward | 1700 µs | 1410 µs | 1.21x |
| E2E decode | 493 tok/s | 599 tok/s | +21.5% |

### Gap analysis: where is the 55% lost bandwidth?

1. **Small GEMV coalescing (~580 µs wasted in layer GEMVs)**:
   The 24-layer GEMVs achieve only 37-55% BW efficiency. Root cause confirmed by
   ncu: Q4_K AoS layout causes scattered DRAM sector access (9.2/32 bytes useful).
   Fix: structure-of-arrays weight relayout.

2. **CUDA graph overhead (~260 µs)**: ~250 kernel nodes in the CUDA graph.
   Each node has ~1 µs dispatch cost even inside a graph.

3. **DeltaNet sequential recurrence (~148 µs)**: Inherently serial.
   Chunkwise recurrence during prefill would help TTFT but not decode.

### What would help (ordered by expected impact)

| Optimization | Expected gain | Difficulty |
|-------------|--------------|-----------|
| SoA weight relayout for small GEMVs | 1.41ms → ~1.0ms (→ ~700 tok/s) | High |
| Persistent decode kernel (eliminate graph overhead) | 1.67ms → ~1.41ms (→ ~710 tok/s) | Very high |
| Both combined | → ~0.85ms (→ ~850 tok/s) | Very high |

### Theoretical limits

| Scenario | Time | Tok/s | How |
|----------|------|-------|-----|
| Current GWEN | 1.67 ms | 599 | Measured |
| Perfect CUDA graph (0 overhead) | 1.41 ms | 709 | Measured forward pass |
| Perfect BW (all kernels at 93%) | 0.67 ms | 1493 | Scale by LM head efficiency |
| Theoretical max (weights only) | 0.58 ms | 1718 | 521 MB / 896 GB/s |
| Theoretical max (all traffic) | 0.63 ms | 1594 | 562 MB / 896 GB/s |

---

## 11. Profiling Cookbook

*Practical recipes for common profiling tasks, based on lessons learned.*

### Recipe 1: Profile decode kernels inside CUDA graph
```bash
# CRITICAL: use --cuda-graph-trace=node to see individual kernels
nsys profile --trace=cuda --cuda-graph-trace=node \
    -o profiles/decode_trace \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "The meaning of life is" --n-predict 100 --greedy

# Then get per-kernel timing:
nsys stats --report cuda_gpu_kern_sum profiles/decode_trace.nsys-rep
```

### Recipe 2: Profile a specific kernel with ncu
```bash
# Use regex: for template kernels, plain --kernel-name won't match
sudo /usr/local/cuda-13.1/bin/ncu \
    --kernel-name regex:"kernel_gemv_q4_k_dp4a" \
    --set full \
    --launch-skip 15 --launch-count 10 \
    ./build/gwen --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --prompt "test" --n-predict 5 --greedy

# Key metrics to check:
#   dram__throughput.avg_pct_of_peak_sustained_elapsed → BW efficiency
#   launch__registers_per_thread → register pressure
#   sm__warps_active.avg_pct_of_peak_sustained_active → occupancy
#   smsp__warps_issue_stalled_long_scoreboard → memory latency stalls
```

### Recipe 3: Verify kernel changes didn't regress performance
```bash
# Before change: capture baseline
./scripts/bench.sh 100 > baseline.txt

# After change: compare
./scripts/bench.sh 100 > after.txt
diff baseline.txt after.txt

# Also verify correctness
./scripts/test_correctness.sh
```

### Recipe 4: Check sector utilization (memory coalescing)
```bash
# In ncu output, look for:
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum → sectors requested
#   l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum → bytes requested
# Divide: bytes_per_sector = bytes / sectors
# Ideal: 32 bytes/sector. Less = uncoalesced access.
```
