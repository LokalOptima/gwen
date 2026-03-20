# Blog 26: Marlin-Style Tensor Core GEMV — Investigation and Dead End

**TL;DR**: Attempted to replace dp4a GEMV with Marlin-style mma.sync tensor core kernels for decode. After implementing weight reshuffling, lop3 dequantization, correct MxNxK formulation, cp.async pipeline, and K-splitting — dp4a still wins by 1.5-4x on all matrix sizes relevant to both the 0.8B and 4B models. The approach is fundamentally wrong for batch=1 GEMV at these dimensions. Code parked on the `marlin` branch.

## Motivation

GWEN's decode GEMV achieves 33-94% of theoretical peak bandwidth depending on matrix size (495 MB model / 896 GB/s = 0.55ms theoretical, 1.65ms measured at 643 tok/s). Blog post 16 identified the potential of "a fused dequant+GEMV kernel that reads quantized weights directly into the mma.sync pipeline." Marlin (IST-DASLab) proved this works for INT4 weights, claiming ~3.87x speedup over FP16.

The question: can we adapt Marlin's approach to Q4_K weights on SM_120 and beat our dp4a GEMV?

## What We Built

### Phase A: Async dp4a (streaming loads)
Added `ld.global.cs` (cache-streaming) hints to the existing dp4a kernel so weight data bypasses L2. Result: 0-3% improvement. The dp4a kernel was already near-optimal.

### Phase B: mma.sync GEMV

Implemented a full Marlin-style tensor core GEMV kernel:

1. **mma.sync.m16n8k16 validation on SM_120** — confirmed tensor cores work on RTX 5070 Ti with our CUDA 13.1 toolchain.

2. **A-fragment register layout discovery** — the PTX spec's register mapping for mma.sync interleaves row groups and K-column groups in a non-obvious way. Registers 1 and 2 are swapped vs the naive `[row_group][col_group]` assumption:
   ```
   a_frag[0] = {A[t/4,   (t%4)*2],   A[t/4,   (t%4)*2+1]}    row 0-7,  K 0-7
   a_frag[1] = {A[t/4+8, (t%4)*2],   A[t/4+8, (t%4)*2+1]}    row 8-15, K 0-7
   a_frag[2] = {A[t/4,   (t%4)*2+8], A[t/4,   (t%4)*2+9]}    row 0-7,  K 8-15
   a_frag[3] = {A[t/4+8, (t%4)*2+8], A[t/4+8, (t%4)*2+9]}    row 8-15, K 8-15
   ```
   This cost a full day to discover empirically (no documentation mentions it).

3. **Q4_K qs byte layout** — Q4_K nibbles are NOT linearly packed. Sub-block pairs share 32 bytes with even sub-blocks in low nibbles and odd in high nibbles:
   ```
   qs_byte = (abs_elem / 64) * 32 + (abs_elem % 32)
   is_high = (abs_elem % 64) >= 32
   ```

4. **First kernel: output rows in M, padded N=1→8** — WRONG formulation. Wasted 7/8 of mma output since only column 0 was non-zero. 3-25x slower than dp4a.

5. **Weight reshuffling** — pre-arranged Q4_K nibbles + pre-computed combined FP16 scales (d×sc, dmin×mn) for coalesced per-thread loads. ~11% memory overhead.

6. **Correct Marlin formulation** — after studying the actual Marlin source code: batch in M (padded 1→16), output features in N. Each mma.sync produces 8 useful output values (one per N-column). This is the correct formulation for GEMV.

7. **lop3 dequantization** — Marlin's trick: embed 4-bit values into FP16 format via bitwise OR with an exponent constant, then subtract bias. 6 instructions for 4 values vs 24 for scalar FP32 dequant.

8. **cp.async pipeline** — 2-stage double-buffered weight loading with 4096 bytes per stage. Made things *worse* because the __syncthreads overhead per 4KB stage dominates compute.

9. **K-splitting** — multiple blocks cooperate on the K reduction via atomicAdd. Helped with parallelism but never closed the gap.

## Why It Doesn't Work

### The fundamental problem: parallelism

ncu profiling comparison for 2048×1024 Q4_K:

| Metric | dp4a | marlin |
|--------|------|--------|
| Blocks | 2048 | 32 |
| Warps launched | 4,096 | 128 |
| Active warps/SM | 34 | 4 |
| Instructions | 573K | 264K |
| Latency/instruction | 27 cycles | 10 cycles |

The marlin kernel uses **fewer instructions** (tensor cores are efficient) and has **lower latency per instruction**. But it has **32x fewer warps**, so the GPU can't hide memory latency through warp switching. dp4a's 1-block-per-output-row gives massive parallelism that saturates all 70 SMs.

### Benchmark results

Tested across both 0.8B and 4B model matrix sizes:

```
Matrix                       | dp4a (µs) | marlin (µs) | ratio
0.8B gate  (2048×1024)       |       3.1 |         5.9 |  1.9x
0.8B ffn_d (1024×3584)       |       4.0 |        16.2 |  4.1x
4B gate    (5120×2560)       |       8.1 |        13.4 |  1.7x
4B ffn_g   (8960×2560)       |      12.2 |        13.7 |  1.1x
4B ffn_d   (2560×8960)       |      10.7 |        38.7 |  3.6x
4B lm_head (151936×2560)     |     259.7 |       273.6 |  1.1x
```

Best case is 1.1x slower (never faster). Deep K dimensions (ffn_down) are worst because each block does hundreds of sequential K-iterations with no latency hiding.

### What the Marlin paper actually says

The paper explicitly states: *"existing kernels achieve relatively close to the optimal 3.87x speedup at batchsize 1."* Marlin's contribution is maintaining that speedup at batch 4-32, where simpler kernels collapse.

Furthermore, Marlin was never benchmarked on matrices smaller than 4096×4096. The smallest model tested was Llama-7B (hidden=4096). At our dimensions (1024-2560), the pipeline startup latency and suboptimal SM partitioning dominate.

## Lessons Learned

1. **Read the paper before implementing.** The Marlin paper says batch=1 is solved. We could have saved days by reading this first.

2. **Profile before optimizing.** The dp4a kernel at 34-94% BW utilization doesn't have a GEMV bottleneck. The decode time gap (0.55ms theoretical vs 1.65ms measured) is from non-GEMV overhead: kernel launches, RMSNorm, attention, DeltaNet recurrence.

3. **PTX fragment layouts are underdocumented.** The register interleaving for mma.sync A-fragments is not in any NVIDIA documentation we could find. It had to be determined empirically.

4. **Parallelism > instruction efficiency at batch=1.** The dp4a approach (1 block per output row, 2048+ blocks) saturates all SMs with simple code. The Marlin approach (64-128 output cols per block, 16-32 blocks) is more instruction-efficient but starves the GPU of parallelism.

5. **cp.async pipeline overhead is non-trivial.** The __syncthreads required between pipeline stages costs more than the latency it hides when data per stage is only 4KB. Marlin uses 8KB+ stages with much more compute between barriers.

## What Would Actually Help Decode Speed

The GEMV kernel is not the bottleneck. At 33% BW utilization for small matrices, the issue is **launch overhead and non-GEMV kernels**. Actual next steps:
- Eliminate Q8_1 quantization launches (the mma path showed this is possible — FP16 input directly)
- Profile and optimize the full pipeline overhead (kernel launches, CUDA graph gaps)
- Optimize DeltaNet recurrence and attention kernels
- Batch-2 GEMV for speculative decode (where tensor cores DO help — N=2 gives 2x utilization)

## Code Location

All implementation code (kernels, reshuffling, tests, benchmarks) is on the `marlin` branch. Not merged to main — the dp4a kernels remain the production path.
