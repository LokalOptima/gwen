# 35. Closing the Gap: dp4a GEMV, Q8_0, and Native IQ4_XS

*From 45.7 tok/s to 118 tok/s on Qwen3.5-9B — matching llama.cpp in one session.*

## Starting Point

Post #34 left the 9B model at 45.7 tok/s decode — correct output, but 2.6× slower than llama.cpp's 117 tok/s. The profiling told a clear story: all GEMV kernels were using the naive dequant path (scalar FP32 multiply-accumulate), while the dp4a integer SIMD kernels that made the 0.8B fast were sitting unused.

This wasn't a kernel problem — it was a wiring problem.

## Phase 1: Wiring dp4a for K-Quant Decode

The dp4a GEMV kernels had been written months ago for the 0.8B's peak optimization phase. They quantize the input vector to Q8_1 format, then use `__dp4a()` to compute 4 int8×int8 products per instruction — reading weights directly as packed integers without dequantization.

The 9B decode path never used them because:
1. `gemv_dispatch()` had no dp4a routing for K-quant types
2. No code quantized the RMSNorm output to Q8_1
3. The Q8_1 scratch buffers were allocated but never passed to GEMV calls

### The Fix

Added an optional `x_q8` parameter to all four dispatch functions (`gemv_dispatch`, `gemv_dispatch_residual_f32`, `gemv_dispatch_batch2`, `gemv_dispatch_batch2_residual_f32`). When a K-quant weight has Q8_1 input available, it calls `gwen_gemv_dp4a` instead of `gwen_gemv`.

In `forward_body`, after each RMSNorm that produces FP16 output, added a `gwen_quantize_q8_1()` call and passed the result to all downstream GEMVs:

```cpp
const bool use_dp4a = is_dp4a_type(model.layers[0].deltanet.attn_qkv.type);

// After RMSNorm:
gwen_rmsnorm_f32_input(buf_a_f32, norm_weight, x_norm, n_embed, eps, s);
if (use_dp4a) gwen_quantize_q8_1(x_norm, x_q8_a, n_embed, s);

// All subsequent GEMVs get the Q8_1 input:
gemv_dispatch(w.attn_qkv, x_norm, qkv, M, K, s, use_dp4a ? x_q8_a : nullptr);
```

Same pattern for the 2-token batch path using `gwen_quantize_q8_1_batch2`.

### The Q8_0 Trap

First run crashed: `Unsupported dp4a GEMV type (Q4_K/Q5_K/Q6_K only)`. The dp4a kernels only handled super-block types (QK_K=256). Q8_0 uses a different block size (QK=32) and was included in `is_kquant_type()` but not `is_dp4a_type()`.

Added `is_dp4a_type()` as a separate check — Q4_K/Q5_K/Q6_K only. The dispatch falls through to naive GEMV for Q8_0 when dp4a isn't available for that type.

### Result: 45.7 → 99.3 tok/s (+117%)

Correct output, 2.17× faster. The Q4_K/Q5_K/Q6_K GEMVs were now running at 80%+ bandwidth efficiency instead of the ~35% of the naive kernels.

## Phase 2: Profiling the Remaining Gap

At 99.3 tok/s vs llama.cpp's 117 tok/s, there was still 15% to find. nsys profiling with `--cuda-graph-trace=node` (critical for profiling CUDA graph nodes) revealed two clear targets:

| Kernel | Time | % | Problem |
|--------|------|---|---------|
| Q8_0 naive GEMV | 64.1ms | 13.1% | ssm_out — no dp4a, 35% efficient |
| F16 GEMV (IQ4_XS exp.) | 64.1ms | 13.0% | 4× memory bloat from expansion |

Together: 26% of total decode time. Everything else was already optimized.

### The Q8_0 Problem

The Unsloth quantizer chose Q8_0 for all 24 `ssm_out` weights (`[4096, 4096]`, 17 MB each). Each per-token GEMV took 54.5µs when the theoretical bandwidth limit was 19µs. The naive kernel does scalar `float × int8 × float` per element — no vectorization, no dp4a.

Q8_0 is structurally different from the super-block K-quants: 32 elements per block (vs 256), simpler scale structure (one `half` per 32 values, no sub-block scales). The existing dp4a template couldn't handle it.

### The IQ4_XS Problem

5 layers have `ffn_gate`/`ffn_up` as IQ4_XS — a non-linear importance-matrix quantization at ~4.25 bits/weight. GWEN had no native IQ4_XS kernel, so at model load these tensors were decompressed from IQ4_XS (25.5 MB each) to F16 (96 MB each) in VRAM. The FP16 GEMV then reads **96 MB per call** instead of the ~27 MB that a native kernel would read. That's 3.7× memory traffic overhead across 10 tensors per token.

## Phase 3: dp4a Q8_0 Kernel

Q8_0 is actually the simplest format for dp4a — the values are already int8, and there's a single `half` scale per 32 elements. The only complexity is alignment: `block_q8_0.qs` sits at offset 2 (after the `half d` field), so it's 2-byte aligned, not 4-byte aligned. Direct `int*` cast would be an unaligned access.

The fix: `get_int_b2()`, a helper that reads two `uint16_t` values and packs them into an `int`:

```cuda
static __device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
```

The kernel structure: NW warps × 32 threads, QI=8 int positions per block, VDR=2 ints per thread. Each thread loads 2 int4 words from Q8_0 weight (via `get_int_b2`) and Q8_1 input (direct cast, 4-byte aligned), does two `__dp4a` calls:

```cuda
int v0 = get_int_b2(blk.qs, iqs + 0);
int v1 = get_int_b2(blk.qs, iqs + 1);
const int* u = reinterpret_cast<const int*>(bq8.qs) + iqs;
int sumi = __dp4a(v0, u[0], __dp4a(v1, u[1], 0));
sumf += __half2float(blk.d) * __low2float(bq8.ds) * (float)sumi;
```

No sub-block scales, no min correction — just `d_weight × d_input × integer_dot`. Much simpler than Q4_K/Q5_K.

ncu result: 78% DRAM throughput (691 GB/s), 52 registers, 63% occupancy. The register pressure is higher than Q4_K (40 regs) because Q8_0 has more blocks per row (128 vs 16-48), requiring more loop state. Still a 2.4× speedup over the naive kernel.

## Phase 4: Native IQ4_XS Kernel

IQ4_XS is the most complex K-quant format. Each 4-bit nibble isn't a linear quantization value — it's an index into a 16-entry lookup table:

```cuda
static __device__ const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
```

The looked-up values are int8, so they can feed directly into dp4a. The challenge is doing the lookup efficiently for 4 packed nibbles in a 32-bit word.

### The `__byte_perm` Trick

Ported from llama.cpp's `get_int_from_table_16`: uses NVIDIA's `__byte_perm` intrinsic to do a 16-entry byte lookup in 6 instructions. The trick: `__byte_perm` selects 4 bytes from two 32-bit registers using 3-bit indices. Since we have 4-bit indices (16 entries), it takes two passes (low 8, high 8) plus a final selection on the 4th bit:

```cuda
static __device__ __forceinline__ int2 get_int_from_table_16(
    const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t sel = (0x32103210 | ((q4 & 0x88888888) >> 1));
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(
        __byte_perm(tmp[0], tmp[1], 0x6420),
        __byte_perm(tmp[0], tmp[1], 0x7531));
}
```

Returns `int2` where `.x` has the looked-up int8 values for even nibbles, `.y` for odd nibbles. These feed directly into dp4a against the Q8_1 input.

### Sub-Block Scales

IQ4_XS has 8 sub-blocks of 32 elements each. Each sub-block has a 6-bit scale packed across `scales_l` (low 4 bits) and `scales_h` (high 2 bits):

```cuda
const int ls = ((blk.scales_l[iqs / 8] >> (iqs & 4)) & 0xF) |
               (((blk.scales_h >> (iqs / 2)) & 3) << 4);
sumi *= (ls - 32);  // centered around 32
```

### Eliminating the F16 Expansion

With the native IQ4_XS GEMV kernel, the load-time F16 conversion is no longer needed. Removed the conversion in `model.cu` — IQ4_XS tensors now upload to GPU in their native 136-byte block format.

VRAM savings: **705 MB** (from 6463 to the previous 7168 MB).

For prefill (multi-token), IQ4_XS doesn't have an MMQ GEMM kernel yet, so the `do_gemm` lambda falls back to sequential dp4a GEMV calls. This is slower for prefill but only affects 10 of ~300 tensors and prefill is not the decode bottleneck.

ncu result: 82% DRAM throughput (720 GB/s), 45 registers, 72% occupancy. The `__byte_perm` lookup adds compute but overlaps well with memory loads in a bandwidth-bound kernel.

## Batch-2 Variants

Both Q8_0 and IQ4_XS got batch-2 kernels for the 2-token speculative decode path. These read weights once and dot against two Q8_1 inputs. For IQ4_XS this is especially valuable — the expensive `get_int_from_table_16` lookup runs once and the results are reused for both tokens.

## Final ncu Analysis

With locked clocks (3105 MHz), all GEMV kernels profiled with `--set full`:

| Kernel | Duration | DRAM BW | Peak% | Regs | Occupancy |
|--------|----------|---------|-------|------|-----------|
| Q4_K dp4a (FFN 12288) | 39.4µs | 720 GB/s | 82% | 40 | 87% |
| Q5_K dp4a (QKV 8192) | 32.7µs | 706 GB/s | 80% | 40 | 87% |
| Q6_K dp4a (ffn_down 4096) | 60.0µs | 697 GB/s | 79% | 48 | 77% |
| Q8_0 dp4a (ssm_out 4096) | 25.8µs | 691 GB/s | 78% | 52 | 63% |
| IQ4_XS dp4a (FFN 12288) | 37.2µs | 720 GB/s | 82% | 45 | 72% |

All kernels at 78-82% of peak DRAM bandwidth. The remaining 18-22% is structural: launch overhead, memory subsystem inefficiency, warp reduction. No kernel has a 2× hiding in it.

The model reads ~5.4 GB of weights per token. At 896 GB/s peak: 6.0ms theoretical minimum. Measured: 8.0ms per token on GPU (8.5ms end-to-end). **75% bandwidth efficiency** overall.

## Results

| Metric | Post #34 | After dp4a | After Q8_0 + IQ4_XS | llama.cpp |
|--------|----------|------------|---------------------|-----------|
| Decode tok/s | 45.7 | 99.3 | **118.1** | 117.8 |
| VRAM | 7168 MB | 7168 MB | **6463 MB** | 5688 MB |
| BW efficiency | ~25% | ~55% | **75%** | ~80% |

**2.58× total speedup. GWEN matches llama.cpp on 9B decode.**

## Remaining Optimization Targets

The nsys profile (100 tokens, locked clocks) shows the time distribution:

| Category | Time | % | Notes |
|----------|------|---|-------|
| dp4a GEMV (all types) | 670ms | 82.8% | At 78-82% BW peak — near ceiling |
| RMSNorm (all) | 39ms | 4.8% | Launch-overhead-bound (~6µs each) |
| DeltaNet fused | 22ms | 2.7% | Sequential recurrence |
| Q8_1 quantize | 13ms | 1.6% | Launch overhead (~1µs each) |
| FP16 GEMV (alpha/beta) | 9ms | 1.2% | Tiny tensors, fast |
| GQA attention | 7ms | 0.9% | 8 FullAttn layers |
| Everything else | 49ms | 6.0% | conv1d, swiglu, norms, etc. |

The GEMV kernels dominate at 83% and are near the bandwidth ceiling. The next wins are structural:

1. **Fuse RMSNorm + Q8_1 quantize**: eliminate ~200 kernel launches/token, save ~50ms
2. **Reduced lm_head**: the Q6_K lm_head reads 816 MB/token (12% of total). Top-K candidate filtering could cut this dramatically.
3. **MTP speculative decode**: process 2 tokens per cycle for ~40% effective throughput gain

## Lessons Learned

1. **Existing kernels are worthless if they're not wired up.** The dp4a kernels existed for months and gave a 2.17× speedup the moment they were connected to the dispatch path. Always check if you're using what you already have before writing new code.

2. **Profile before optimizing, then profile again.** The first profile showed dp4a wiring as the obvious fix. The second profile revealed Q8_0 and IQ4_XS as the next targets. Without the second profile, I might have spent time micro-optimizing Q4_K dp4a (which was already at 82% efficiency) instead of fixing the two kernels that were actually slow.

3. **Format-specific kernels beat generic expansion.** Expanding IQ4_XS to F16 was the easy path — and it cost 705 MB of VRAM and 64ms per 100 tokens. The native kernel with `__byte_perm` lookup is only slightly more code but reads 3.7× less data.

4. **Alignment matters for dp4a.** Q8_0's `qs` at offset 2 (after `half d`) means 2-byte alignment, not 4-byte. Direct `int*` cast would be undefined behavior. The `get_int_b2` helper handles this correctly with two `uint16_t` loads.

5. **ncu captures CUDA graph nodes by default** (ncu 2024.1+, `--graph-profiling node`). No need to disable graphs for profiling — just skip enough launches to get past prefill.
