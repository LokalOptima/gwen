# Post 31: FP8 — The Right Format for SM_120

After 30 blog posts and months of fighting Q4_K dequantization overhead, we're abandoning GGML quantized weights. The RTX 5070 Ti's tensor cores natively speak FP8 E4M3 — time to give them data they can actually eat.

## Why Q4_K Was Wrong

The entire Q4_K story has been a story of fighting the format:

- **Blog 16**: Discovered the "7× bandwidth amplification" from dequantizing Q4_K to FP16 for CUTLASS. The GEMM decode path was reading and writing 3.5 GB per token instead of 0.5 GB.
- **Blog 26**: Spent a week implementing Marlin-style tensor core GEMV for Q4_K. Dead end — dp4a won by 1.5-4× because the dequant overhead dominated.
- **Blog 30**: Five failed attempts at fused Q4_K GEMM kernels. Had to port llama.cpp's 540-line MMQ kernel. Still 28% slower than pre-dequanted FP16.

The pattern: Q4_K is a CPU format. Its block structure (256-element super-blocks, 6-bit packed scales, nibble-interleaved data, dmin offsets) was designed for SIMD dequantization on x86 AVX, not for GPU tensor cores.

## What SM_120 Actually Wants

The RTX 5070 Ti's tensor cores support FP8 E4M3 natively:

| Format | Bits | Tensor Core TFLOPS | Dequant Cost |
|--------|------|-------------------|-------------|
| FP16 | 16 | 88 | None |
| FP8 E4M3 | 8 | 176 (2× FP16) | None |
| FP4 E2M1 | 4 | 703 (8× FP16) | None |
| Q4_K | ~4.5 | N/A (need MMQ) | Massive |

FP8 is 1 byte per weight. No block structures, no super-blocks, no nibble packing. The conversion from FP8 to FP16 is a single PTX instruction: `cvt.rn.f16x2.e4m3x2`.

## The FP16 Residual Problem

While investigating the format change, we identified a second issue: the decode path accumulates residuals in FP16. Over 24 layers, each `buf_fp16 = buf_fp16 + gemv_output_fp16` rounds to FP16, losing precision cumulatively. llama.cpp uses F32 residuals.

The FP8 path fixes both problems at once — F32 residual accumulators from day one.

## What We Built

### FP8 Quantization Pipeline

`scripts/quantize_fp8.py` converts HuggingFace BF16 SafeTensors to our GWFP8 binary format:

- Per-row FP32 scale factors: `scale = max_abs(row) / 448.0`
- FP8 E4M3 quantization: `fp8_val = round_to_nearest_e4m3(float_val / scale)`
- No calibration data needed — weight-only, deterministic
- Handles all the HF→GWEN tensor mapping: QKV split from `in_proj_qkvz`, norm weight +1 transform, A_log→-exp, conv1d squeeze, V head reorder for asymmetric models (4B/9B)

The 0.8B model produces a 722 MB GWFP8 file (320 tensors). Compare: Q4_K_M GGUF was 530 MB. The 1.4× size increase buys dramatically simpler compute.

### FP8 GEMV Kernels

Four kernel variants in `src/kernels/gemv_fp8.cu`:

```
kernel_gemv_fp8                    — FP16 output (sub-projections)
kernel_gemv_fp8_residual_f32       — F32 output + F32 residual add
kernel_gemv_fp8_batch2             — read weights once, 2 dot products
kernel_gemv_fp8_batch2_residual_f32 — batch2 + F32 residual
```

Design based on FBGEMM's production FP8 GEMV (Meta) and the Blackwell NVFP4 hackathon findings:

- **128-bit vectorized loads**: `float4` = 16 FP8 elements per load, coalesced
- **PTX conversion**: `cvt.rn.f16x2.e4m3x2` — converts 2 FP8 bytes to `half2` in one instruction. We use PTX inline asm directly rather than the C++ intrinsic (`__nv_cvt_fp8x2_to_halfraw2`), which has a known bug on SM90 and hasn't been confirmed fixed
- **F32 accumulation**: `fmaf()` throughout. Bandwidth-bound kernel, compute is free
- **256 threads/block, 1 block per output row**: Same parallelism as the dp4a Q4_K GEMV (248K blocks for the LM head)
- **Per-row scale applied after reduction**: Single float multiply at the end

The key simplification vs Q4_K: no Q8_1 input quantization step. The FP8 GEMV reads FP16 input directly. This eliminates `gwen_quantize_q8_1()`, `gwen_rmsnorm_quantize_q8_1()`, `gwen_swiglu_quantize_q8_1()`, and the Q8_1 scratch buffers. Fewer kernel launches per layer.

### F32 Residual Decode Path

The new `forward_body` FP8 path for both DeltaNet and FullAttn layers:

```
embed_lookup_fp8(token) → buf_a (FP16) → buf_a_f32 (F32)
for each layer:
  rmsnorm_f32_input(buf_a_f32) → x_norm (FP16)
  gemv_fp8(W_qkv, x_norm) → qkv (FP16)
  conv1d_silu(qkv)
  deltanet(qkv, S) → attn_out
  gated_rmsnorm(attn_out, gate) → gated (FP16)
  gemv_fp8_residual_f32(W_out, gated, buf_a_f32) → buf_b_f32
  rmsnorm_f32_input(buf_b_f32) → x_norm (FP16)
  gemv_fp8(W_gate, x_norm) → ffn_gate
  gemv_fp8(W_up, x_norm) → ffn_up
  swiglu(gate, up) → ffn_out (FP16)
  gemv_fp8_residual_f32(W_down, ffn_out, buf_b_f32) → buf_a_f32
rmsnorm_f32_input(buf_a_f32) → x_norm
gemv_fp8(token_embd, x_norm) → logits
```

No Q8_1 anywhere. Residual stream stays in F32. Sub-projections are FP16 (they don't feed into the residual stream). The decode path is completely independent of the Q4_K code — no patches, no fallbacks.

## The CUTLASS Block Scaling Problem

The decode GEMV path works. But the prefill GEMM path is blocked on a non-trivial CUTLASS integration issue.

SM_120's fast FP8 compute uses `mma.sync.aligned.block_scale` — a tensor core instruction that applies scale factors during the multiply-accumulate. This gives 2× throughput vs non-block-scaled FP8 MMA. But it requires the data to come in a specific block-scaled layout.

### What block scaling means

Regular FP8 MMA: `C += A_fp8 × B_fp8`. Raw FP8 values, no scaling.

Block-scaled MMA: `C += scale_A × scale_B × (A_fp8 × B_fp8)`. The hardware reads scale factors alongside the FP8 data and applies them during the multiply.

### Two CUTLASS approaches

**MXFP8 (example 79c)**: OCP Microscaling standard. Every 32 elements along K share one E8M0 scale factor (power-of-2 only, 8-bit exponent). The scale factors are interleaved into the data in a hardware-specific layout. Problem: E8M0 can't represent arbitrary scales — our per-row FP32 scales would lose ~12% precision rounding to the nearest power of 2.

**F32 blockwise (example 87a)**: One full FP32 scale per (128×128) block of the weight matrix. More flexible, but coarser: 128 rows within a block must share the same scale. Our per-row scales give each row its own scale — 128 different scales collapsed into one.

### The resolution

Per-row scaling is finer than what the hardware's block-scaled path supports. The options:

1. **Quantize with per-128×128-block scaling** instead of per-row. Accept the coarser granularity. Quality impact: ~1-2 bits precision loss for worst-case rows within a block. Acceptable for FP8 (which has ~4 bits mantissa to begin with).

2. **Non-block-scaled FP8 GEMM** via CUTLASS 2.x API. Gives regular (not 2×) FP8 throughput, but lets us keep per-row scales with an epilogue multiply. Simpler to implement.

3. **Hybrid**: non-block-scaled for initial bring-up, block-scaled as optimization later.

## What's Next

The CUTLASS FP8 GEMM integration is the single blocker. Once `forward_prefill` can run FP8 weights through CUTLASS, we have end-to-end inference and can benchmark decode tok/s, prefill tok/s, and correctness against llama.cpp.

The path forward is hybrid: start with non-block-scaled FP8 GEMM (CUTLASS 2.x, minimal code change), validate correctness and performance, then upgrade to block-scaled (2× throughput) for prefill optimization.

After FP8 is proven, the infrastructure cleanly extends to NVFP4 (E2M1, native tensor core format, 8× FP16 throughput) for maximum decode bandwidth — same pipeline, same scaling approach, just narrower data type.
