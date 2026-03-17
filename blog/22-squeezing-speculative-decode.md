# Post 22: Squeezing Speculative Decode — From 600 to 928 tok/s

The MTP v3 fine-tuned head (71.2% acceptance, K=4096) was integrated into the CUDA inference engine. First test: 639 tok/s with speculative decoding vs 600 tok/s without — only 6.5% speedup despite 67% acceptance rate. This post covers the profiling-driven optimization that brought it to 752 tok/s sustained (928 peak).

## Why 67% Acceptance Only Gave 6.5% Speedup

The speculative decode cycle is: `forward_2tok(last, draft)` → accept/reject → `forward_mtp(draft)`. The MTP head was already fast at 0.09 ms per call — only 4% of GPU time. The bottleneck was `forward_2tok` taking 1.52x longer than single-token `forward`, not ~1.0x as expected for bandwidth-bound decode.

The batch2 GEMVs read model weights once for both tokens — that part was optimal. But **60% of GPU time was non-GEMV kernels** (RMSNorm, quantize, SwiGLU, DeltaNet state ops, attention) that were launched twice — once per token, as separate kernel calls on the same stream.

## Batch2 Kernel Fusion: 600 → 689 tok/s

Wrote batch2 variants of all independently-batchable kernels:

- `rmsnorm_quantize_q8_1_batch2`: 2 blocks x 256 threads (was 2 separate 1-block launches)
- `swiglu_quantize_q8_1_batch2`: 2D grid with blockIdx.y selecting token
- `gated_rmsnorm_quantize_q8_1_batch2`: 2D grid (x=head, y=token)
- `quantize_q8_1_batch2`: same 2D grid pattern

Eliminated ~96 redundant kernel launches per speculative cycle. Each launch has ~2-3 us overhead, so ~240 us saved per cycle x 119 cycles. DeltaNet state ops (conv1d, deltanet_decode) couldn't be batched because token B's state update depends on token A's.

## DeltaNet Mega-Kernel: 600 → 623 baseline, 689 → 720 MTP

Fused `l2_normalize(Q)` + `l2_normalize(K)` + `compute_gate_beta` + `deltanet_decode` into a single `kernel_deltanet_fused`: 16 blocks x 128 threads (one per head).

Phase 1: L2-normalize Q and K — 128 threads, 1 element each, 4-warp cross-reduction. Phase 2: Gate/beta dot products — 128 threads over 1024 Q8_0 elements (8 per thread). Phase 3: S matrix decay + update + output — 128 threads, 1 column each.

This eliminated 3 kernel launches per DeltaNet layer per token (54 fewer in single-token, 108 in batch2). Improved the baseline too since the single-token `forward_body` also benefits.

## ncu Profiling: The S Matrix Problem

Profiled with `ncu --set full` on a standalone benchmark:

```
Duration:                    10.24 us/call
Achieved Occupancy:          8.1%
Warp Cycles Per Instruction: 18.05
L1TEX Scoreboard Stalls:     76.4% of all stalls
Achieved BW:                 410 GB/s (46% of 896 GB/s peak)
Registers Per Thread:        40
Grid:                        16 blocks (one per head)
```

The problem: **16 blocks on 70 SMs**. 54 SMs completely idle. Each active SM runs 4 warps. When a warp stalls on an L2 load (~200 cycles), the scheduler switches through 4 warps in 4 cycles, then all warps are stalled for 196 cycles.

The S matrix (128x128 floats = 64 KB per head, 1 MB per layer, 18 MB across all DeltaNet layers) doesn't stay in L1 because 18 layers cycle through, evicting each other. Every kernel invocation cold-starts from L2.

### Failed Attempt: More Blocks

Tried splitting each head into 2 blocks of 64 threads (32 blocks total). Each block loads the full 128-element Q/K vectors into shared memory and handles 64 columns of S. Result: **526 tok/s** — much worse. The split forced Q/K/gate/beta results through global memory between prep and decode kernels. The round-trip killed any occupancy gain.

### Failed Attempt: More Threads

Tried 256 threads per block with row tiling (two groups of 128 threads, each handles 64 rows). Result: **549 tok/s** — worse due to `__syncthreads` overhead for inter-group reduction of sk_j and o_j partial sums.

Tried 256 threads with extra threads idle during Phase 3. Result: **636 tok/s** — marginal. Idle warps don't help the scheduler.

## S Matrix in Shared Memory: 623 → 632 baseline, 720 → 726 MTP

Loaded the 64 KB S matrix into dynamic shared memory before the compute passes, operated entirely at ~4 cycle shared memory latency instead of ~200 cycle L2. ncu confirmed: warp stall cycles dropped from 18.05 to 8.17 per instruction — **2.2x improvement in instruction throughput**.

Required `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)` to allow 64 KB dynamic shared.

## Async Pipelining: 632 → 647 baseline, 726 → 743 MTP

The S load from global to shared and Phase 1-2 (L2-norm, gate/beta) are independent — Phase 1-2 reads Q, K, x_norm, alpha/beta weights, none of which is S. Used `cp.async` to start the S load non-blocking, then computed Phase 1-2 while loads were in flight:

```cpp
// Async copy: global → shared (non-blocking)
for (int i = 0; i < 128 * 128; i += 128) {
    uint32_t addr = __cvta_generic_to_shared(&sh_S[i + tid]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(addr), "l"(&S_head[i + tid]));
}
asm volatile("cp.async.commit_group;\n");

// Phase 1-2 runs here while S loads are in flight...

// Wait for S before Phase 3
asm volatile("cp.async.wait_group 0;\n");
__syncthreads();
```

## Two-Stream Overlap: 743 → 752 MTP

In `forward_body_2tok`, the DeltaNet processes token A then token B sequentially because they share the S matrix. But conv1d B only depends on conv1d A (shared conv_state), not on deltanet A. So conv1d B can run on the 54 idle SMs while deltanet A runs on 16:

```
Stream 1: [conv1d A][conv_snap][deltanet A ══10us══][S_snap]──[deltanet B]
Stream 2:                      [conv1d B ─1.5us─]─────────────┘
```

Implemented with a secondary CUDA stream forked via `cudaEventRecord`/`cudaStreamWaitEvent` during graph capture. The captured CUDA graph encodes the parallel execution.

## Final Results

| Prompt | No MTP | Before | After | Speedup |
|--------|--------|--------|-------|---------|
| Narrative (65%) | 647 | 639 | **752** | +16% |
| Quantum (58%) | — | 619 | **~720** | +11% |
| Capital (98%) | — | 770 | **928** | +39% |
| TCP/UDP (46%) | — | 545 | **~660** | +2% |

Baseline (no MTP) improved from 600 to 647 tok/s (+8%) because the mega-kernel and shared memory optimizations benefit the single-token path too.

## What's Left

The DeltaNet kernel is now operating at 8 cycles per instruction with 4 warps — close to the shared memory throughput limit. The 16-head constraint (16 blocks on 70 SMs = 8% occupancy) is architectural. Further improvements would require:

- **FP16 S matrix**: halve the 64 KB to 32 KB, fitting 2 heads per SM's shared memory, doubling throughput
- **IDK token** (plan_v4.md): higher acceptance rate by absorbing OOV probability mass, targeting 80%+ acceptance
- **Algorithmic changes**: low-rank S approximation, or chunked/tiled S update across multiple tokens
