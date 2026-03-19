# 16. Why GEMV Destroys GEMM at N=1 (and Other Failed Optimizations)

This post documents three optimization attempts that produced zero useful improvement. The plan looked sound on paper. The numbers said otherwise. Here's what I thought would happen, what actually happened, and the root cause analysis for each failure.

## The Starting Point

Two bottlenecks in the GEMM path:

1. **GEMM decode** (N=1): 96 tok/s vs 605 tok/s for GEMV+graph — a 6.3x gap. The plan assumed this was kernel launch overhead (~600 launches × ~2 μs each ≈ 1.2 ms wasted). Solution: capture the GEMM decode path in a CUDA graph, same as the GEMV path.

2. **Batch DeltaNet** (`extract_hidden_batch` at B=64, L=512): 20K tok/s. DeltaNet recurrence was 54% of runtime. Each of the 1024 blocks (64 batches × 16 heads) owns a 64 KB S matrix. 1024 × 64 KB = 64 MB active working set thrashes the 48 MB L2 cache. Solution: wave-limit launches and put S in shared memory.

## Phase 1: CUDA-Graphing the GEMM Decode Path

### What I Built

`forward_prefill_body()` — a graph-capturable version of `forward_prefill(N=1)`. Stripped out all H2D/D2H copies, `cudaStreamSynchronize`, and debug dump code. Inlined the FullAttn per-token loop (trivial for N=1). Wrapped it in `forward_gemm()` using the same capture/replay pattern as the existing GEMV graph.

The graph captured cleanly: 601 nodes. No fallbacks needed.

### What I Expected

96 tok/s → 550–600 tok/s. The reasoning: GEMV+graph achieves 605 tok/s with ~384 kernel launches captured. GEMM has ~600 launches, similar overhead, so graphing should give a similar speedup.

### What Actually Happened

96 tok/s → 102 tok/s. A 6% improvement. The graph works — it does eliminate host-side overhead — but that overhead was never the bottleneck.

### Why: The Dequantization Tax

The GEMV path reads quantized weights directly. One kernel, one memory pass:

```
GEMV: read 495 MB quantized weights → compute → done
Total bandwidth: ~495 MB per forward pass
```

The GEMM path cannot do this. CUTLASS expects dense FP16 input matrices. Every GEMM call first dequantizes the full weight matrix to an FP16 temp buffer, then CUTLASS reads that buffer:

```
GEMM: read 495 MB quantized → write 1503 MB FP16 → read 1503 MB FP16
Total bandwidth: ~3500 MB per forward pass
```

That's **7.1× more memory traffic** for the same computation. On a bandwidth-bound GPU (896 GB/s), this is the entire story:

| Path | Bandwidth | Theoretical Min | Measured | Efficiency |
|------|-----------|-----------------|----------|------------|
| GEMV+graph | 495 MB | 0.55 ms (1810 tok/s) | 1.65 ms (605 tok/s) | 33% |
| GEMM+graph | 3500 MB | 3.91 ms (256 tok/s) | 9.80 ms (102 tok/s) | 40% |

Both paths achieve reasonable bandwidth efficiency. The GEMM path is actually *slightly better* utilized (40% vs 33%). It's just doing 7× more work.

The measured 5.9x ratio (605/102) is close to the theoretical 7.1x bandwidth ratio — the GEMM path gets some benefit from better compute overlap, but it can't overcome reading and writing 3 GB instead of 0.5 GB.

### Why GEMV Is Fundamentally Better for N=1

Three compounding advantages:

1. **No dequantization bounce.** dp4a GEMV reads Q4_K/Q5_K/Q6_K blocks directly and multiplies against Q8_1-quantized input vectors. Zero intermediate buffers. The quantized format *is* the compute format.

2. **Higher arithmetic density.** dp4a computes 4 INT8 multiply-accumulates per instruction on quantized data. CUTLASS FP16 mma.sync (16×8×16) operates on dequantized FP16, which is 2× the bytes per element for less than 2× the precision where it matters (accumulation is FP32 in both paths).

3. **Kernel geometry.** The GEMV kernel launches one block per output row with 2–4 warps, streaming through the weight row sequentially. Perfect for N=1. CUTLASS launches threadblocks with 128×128 output tiles — for N=1, each block computes a 128×1 slice, wasting the entire N-dimension of the tile. The tile traversal, shared memory staging, and warp-level matrix ops are all designed for large N.

### The Lesson

The GEMV-vs-GEMM gap is not launch overhead. It's a **7x bandwidth amplification** from the dequant→temp→CUTLASS pipeline. CUDA graph captures cannot fix bandwidth. The only way to close this gap would be a fused dequant+GEMV CUTLASS kernel that reads quantized weights directly into the mma.sync pipeline — essentially what dp4a GEMV already does, but with tensor cores. That's a research project, not an afternoon optimization.

## Phase 2: Wave-Limited DeltaNet Launches

### What I Built

Split the `kernel_deltanet_prefill_batch` launch from one dispatch of B×n_heads blocks into waves of 32 batches. Each wave puts 32 × 16 heads × 64 KB = 32 MB of S matrices in flight, fitting comfortably in the 48 MB L2 cache.

### What I Expected

DeltaNet: 884 ms → ~600 ms (from better L2 hit rates on S). Total batch extraction: 1633 ms → ~1350 ms (~24K tok/s).

### What Actually Happened

B=64, L=512: 1652 ms (19.8K tok/s). Statistically identical to the pre-wave baseline.

### Why

The DeltaNet kernel processes L=512 tokens *sequentially* within each block. Each token does two passes over the 128×128 S matrix (decay+sk, then update+output). S lives in registers/L1 for the duration of the block's execution — the L2 pressure comes from *other* blocks' S matrices being evicted, but any individual block streams through its own S from L1/registers, not L2.

Wave-limiting reduces the number of concurrent blocks from 1024 to 512 (32 batches × 16 heads), but the 5070 Ti only has 70 SMs. At most 70 blocks run simultaneously. The remaining 954 blocks were never truly concurrent — they were already queued and scheduled in hardware waves. Explicit wave-limiting just adds a CPU-side loop with no effect on the GPU scheduler's natural behavior.

## Phase 3: Shared-Memory S DeltaNet Kernel

### What I Built

`kernel_deltanet_prefill_batch_shmem` — loads the 128×128 S matrix into 64 KB dynamic shared memory at block start, processes all L tokens with S in shared memory (no global traffic during the token loop), and writes S back once at the end.

### What I Expected

DeltaNet: 600 ms → 120–180 ms (eliminating global S reads/writes during the token loop). Total: ~870–930 ms (~35–38K tok/s).

### What Actually Happened

Measured as part of the combined Phase 2+3 result: 1652 ms. No improvement.

### Why

The original kernel's S access pattern was already L1-cache-friendly. Each thread owns one column of S (128 floats = 512 bytes), accessed sequentially across 128 rows per pass. With 128 threads per block and 128 KB L1+shared per SM, the 64 KB S matrix fits entirely in L1 cache on the second access and stays there for all 512 tokens.

Moving S to shared memory changes the memory hierarchy label but not the access latency — L1 cache and shared memory share the same on-chip SRAM on modern NVIDIA GPUs (since Volta). The only difference is that shared memory provides *guaranteed* residency while L1 is subject to eviction. But with `__launch_bounds__(128, 1)` occupancy, only one block occupies each SM, so there's nothing to evict the L1 lines.

Furthermore, the shared-memory kernel requires 66 KB dynamic shared memory per block, which forces `maxBlocksPerMultiprocessor = 1`. This is the same occupancy as the register-hungry original kernel, so there's no scheduling disadvantage — but also no advantage.

The DeltaNet bottleneck is not memory traffic. It's **serial token dependency**: each token's S update depends on the previous token's result. 512 sequentially-dependent iterations × 2 passes × 128 rows = the kernel is latency-bound on the FMA chain, not bandwidth-bound on S.

## What Would Actually Help

For the GEMM decode path:
- **Don't use it for decode.** The GEMV path exists and is 6× faster. The GEMM decode path was only ever needed as a correctness reference against llama.cpp's F32 accumulation. Use it for verification, not production.

For batch DeltaNet:
- **Chunk-parallel recurrence.** Split L=512 into chunks (e.g., 32 tokens each), process chunks in parallel with local S states, then combine via a parallel scan/reduction. This breaks the serial dependency at the cost of more computation. The `fla` (Flash Linear Attention) library implements this for training — adapting it for inference would be the real optimization.
- **Reduce S precision.** FP32 S matrices are 64 KB. BF16 would halve this to 32 KB, doubling effective L1/L2 capacity. Needs stability analysis.
- **Overlap DeltaNet with GEMMs.** The DeltaNet recurrence and the FFN GEMMs are independent after the QKV projection. Multi-stream overlap could hide some of the serial latency.

## The Meta-Lesson

Three optimizations, zero useful gains. The failure pattern:

1. **Misidentified bottleneck.** We assumed the GEMV-GEMM gap was launch overhead. It was bandwidth amplification from dequantization.
2. **Optimized the wrong level.** Wave-limiting targeted L2 cache pressure, but the GPU hardware scheduler already handles wave scheduling. We were "optimizing" something the hardware does automatically.
3. **Wrong memory hierarchy assumption.** Shared memory was supposed to be faster than global memory for S, but S was already in L1 cache. On modern NVIDIA GPUs, shared memory and L1 share the same silicon.

The common thread: **we optimized without profiling.** A 10-minute nsys trace would have shown that:
- GEMM decode time is dominated by CUTLASS kernel execution (not launch overhead)
- DeltaNet S accesses have >95% L1 hit rate already
- The DeltaNet kernel is compute/latency-bound, not memory-bound

Profile first. Hypothesize second. The plan's assumptions about bottleneck sources were plausible but wrong in every case. The code has been reverted.
