# 17. Profiling All Three Code Paths

After three failed optimizations in the last post, I went back to basics: profile first, understand second, plan third. This post covers getting nsys to work properly on all three GPU code paths, the kernel-level breakdown of where time actually goes, and the research into two real optimizations that could help.

## The Problem: Profiling Was Harder Than Expected

I had three code paths to understand:
1. **GEMV decode** (N=1, dp4a) — the fast path, 605 tok/s
2. **GEMM decode** (N=1, CUTLASS) — the slow reference path, ~96 tok/s
3. **Batch extraction** (B×L tokens, CUTLASS) — the training data pipeline, ~20K tok/s

The GEMV and GEMM decode paths are accessible through the main `gwen` binary. But batch extraction only existed in the HTTP server (`gwen_server`), which proved to be a dead end for profiling.

### Why Server Profiling Failed

First attempt: profile the server directly.

```bash
nsys profile ./gwen_server --model model.gguf &
curl -X POST http://localhost:8080/batch_extract ...
```

Result: **0 CUDA kernels captured.** The server uses cpp-httplib, which has an internal thread pool. CUDA work runs in handler threads that nsys wasn't tracing. I tried `--trace-fork-before-exec=true` — still nothing reliable.

### Solution: Add `--batch-extract` to the Main Binary

Instead of fighting nsys+threading, I added a `--batch-extract FILE` flag to the main `gwen` binary. It reads prompts from a file (one per line), tokenizes each, pads/truncates to `--seq-len`, and calls `extract_hidden_batch` directly.

```bash
./gwen --model model.gguf --batch-extract prompts.txt --seq-len 128
```

Same code path as the server, no threading, nsys captures everything cleanly.

### Getting nsys CSV Output Right

Another pitfall: parsing `nsys stats --format csv` output. The command mixes diagnostic text with CSV data:

```
Generating SQLite file /tmp/profile.sqlite from /tmp/profile.nsys-rep
Processing [...] with [...]...
NOTICE: some notice
Time (%),Total Time (ns),Instances,...    ← actual CSV starts here
54.3,12345678,100,...
```

A naive `csv.DictReader` chokes on the preamble. I wrote `parse_nsys.py` that skips everything before the `Time (%)` header line:

```bash
nsys stats --force-export=true --report cuda_gpu_kern_sum \
    profile.nsys-rep --format csv 2>&1 | python parse_nsys.py
```

The `--force-export=true` is essential — without it, nsys uses a cached `.sqlite` that may be stale from a previous capture to the same path.

## The Profiles

I profiled all three paths with `--cuda-graph-trace=node` (critical — without this, CUDA graph-replayed kernels are invisible).

### GEMV Decode: 50 tokens, 422 tok/s — 87.1 ms GPU time

```
Kernel                                                      N  Total(us)   Avg(us)      %
------------------------------------------------------------------------------------------
void kernel_gemv_q4_k_dp4a<                              3920    14041.5      3.58  16.1%
void kernel_gemv_q6_k_dp4a<                               245    12627.7     51.54  14.5%
kernel_deltanet_decode                                    882     8996.4     10.20  10.3%
CUTLASS GEMM (F32 out)                                    132     7701.3     58.34   8.8%
kernel_rmsnorm_quantize_q8_1                             2401     7588.8      3.16   8.7%
void kernel_gemv_q5_k_dp4a<                               882     6086.8      6.90   7.0%
...
TOTAL                                                   19735    87145.8
```

Well-balanced. dp4a GEMV variants are ~49% combined, DeltaNet decode 10%, RMSNorm+quantize 9%, prefill CUTLASS GEMM 9%. No single dominant bottleneck — this path is already well-optimized.

### GEMM Decode: 50 tokens, 77 tok/s — 572.1 ms GPU time

```
Kernel                                                      N  Total(us)   Avg(us)      %
------------------------------------------------------------------------------------------
CUTLASS GEMM (F32 out)                                   6600   295766.2     44.81  51.7%
kernel_dequant_q4_k                                      4900    99599.1     20.33  17.4%
kernel_dequant_q5_k                                      1800    55674.5     30.93   9.7%
kernel_gemv_q6_k                                           50    51235.9   1024.72   9.0%
CUTLASS GEMM (FP16 out)                                   900    22452.3     24.95   3.9%
kernel_dequant_q6_k                                       800    15952.3     19.94   2.8%
...
TOTAL                                                   29486   572051.3
```

**Dequant + GEMM = 79% of total time.** This is the bandwidth amplification problem from blog post 16 — now proven with profiling data. The dequant kernels alone (Q4_K + Q5_K + Q6_K) account for 30% of all GPU time. Each weight matrix gets:
1. Read as Q4_K/Q5_K (~495 MB total)
2. Written as FP16 to temp buffer (~990 MB)
3. Read again by CUTLASS (~990 MB)

Total memory traffic: ~2.5 GB per forward pass vs ~495 MB for GEMV. The 6.6× speed ratio (572/87 ms) matches the ~5× bandwidth ratio (accounting for L2 cache hits on smaller matrices).

### Batch Extract: B=63, L=128, 23K tok/s — 344.6 ms GPU time

```
Kernel                                                      N  Total(us)   Avg(us)      %
------------------------------------------------------------------------------------------
kernel_deltanet_prefill_batch                              18   213822.5  11879.03  62.0%
CUTLASS GEMM (FP16 out)                                   150    83703.6    558.02  24.3%
kernel_batch_causal_attn                                    6    14879.9   2479.99   4.3%
kernel_batch_compute_gate_beta                             18     8082.2    449.01   2.3%
kernel_batch_conv1d_silu_multi                             18     6450.9    358.38   1.9%
kernel_swiglu_batch                                        24     5574.7    232.28   1.6%
...
TOTAL                                                     529   344643.7
```

**DeltaNet dominates at 62%.** The CUTLASS GEMMs are efficient at 24% — they're compute-bound at N=8064 and amortize the weight reads well across the batch. The 6 full-attention layers are only 4.3%. Everything else is noise.

The DeltaNet bottleneck is worse than the 54% previously reported at B=64,L=512 because at L=128 the GEMMs are proportionally cheaper while DeltaNet's serial token dependency doesn't change.

## What The Profiles Tell Us

Two clear targets, each for a different code path:

### 1. Batch Extraction: DeltaNet Is the Bottleneck

DeltaNet at 62% dominates batch extraction. The kernel processes tokens *sequentially within each sequence* — the recurrence `S_t = f(S_{t-1}, token_t)` is inherently serial. At L=128, each of the 63×16=1008 (batch, head) pairs does 128 serial steps. The GPU has parallelism *across* pairs but zero parallelism *within* a sequence.

The fix: **chunkwise DeltaNet**. Split each sequence into chunks of C=64 tokens, convert the within-chunk recurrence into matrix operations (parallel), and propagate state between chunks (L/C sequential steps instead of L). For L=128, C=64: **2 sequential steps instead of 128**.

The algorithm uses a WY representation from numerical linear algebra — the product of C rank-1 updates `(I - beta*k*k^T)` can be expressed as a compact matrix form via a C×C lower-triangular solve. Five phases: compute K*K^T, triangular solve, compute correction terms, sequential inter-chunk state propagation, output. All parallel except the inter-chunk step. The `flash-linear-attention` library implements this in Triton. No CUDA/C++ implementation exists anywhere — we'd be writing the first.

### 2. GEMM Decode: Bandwidth Amplification from Dequant

For GEMM decode (and any small-N GEMM path), the 79% dequant+GEMM cost is architectural. The fix: **fused dequant-GEMM** — dequantize Q4_K/Q5_K weights inside the GEMM kernel's register file, feeding FP16 fragments directly to `mma.sync` tensor cores. No temp buffer. Bandwidth drops from ~2.5 GB to ~495 MB, matching GEMV.

The Marlin kernel does exactly this for uniform INT4 with FP16 scales. Our Q4_K format is more complex (256-element super-blocks, 8 sub-blocks of 32 with 6-bit packed scales and minimums), but the same principle applies: read quantized, dequant in registers, feed to tensor cores.

Three viable approaches:
- **A. Extend dp4a GEMV to N>1** — cheapest, what llama.cpp's MMQ kernels do for batch≤64
- **B. Marlin-style fused dequant with FP16 mma.sync** — most general, single kernel handles N=1-32
- **C. CUTLASS 2.x mixed-input** — only supports uniform INT4/INT8, not Q4_K. Would require offline requantization.

## What's Next

The plan is to tackle both optimizations:
1. **Chunkwise DeltaNet kernel** for batch extraction — first-ever CUDA implementation
2. **Fused dequant-GEMM** (or extended GEMV) to close the GEMM decode gap

Both require substantial kernel development. The chunkwise DeltaNet is the higher-impact target — DeltaNet is 62% of batch extraction time, and the training pipeline depends on batch extraction throughput.
