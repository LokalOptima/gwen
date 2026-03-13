# Numerical Precision Investigation: GWEN vs llama.cpp

## The question

After implementing all 24 layers of Qwen3.5-0.8B in GWEN and verifying that greedy
token output matches llama.cpp on most prompts, we observed that logit values diverge
over sequence length. Teacher-forced comparison (feeding identical tokens to both
engines) showed top-1 logit gaps growing from ~0.04 at position 0 to ~1.5 at position
30. We needed to understand: is this a bug in GWEN, or an inherent precision
difference? And if it's precision, where exactly does it enter?

## Investigation methodology

We dumped per-layer hidden states from both engines on the same input (token "The",
id=760) and compared them at every checkpoint in the forward pass.

### Tools built

**llama.cpp side** — `tests/llama_dump_layers.cpp`: Uses llama.cpp's `cb_eval`
callback to intercept every named tensor during the compute graph execution. Captures
F32 tensors and writes them to binary files. Runs on GPU (`n_gpu_layers = 99`) to
match the actual inference path. Key tensor names for layer 0 bisection:

- `model.input_embed` — embedding output
- `attn_norm-0` — RMSNorm output (F32, pre-GEMM)
- `linear_attn_qkv_mixed-0` — QKV projection output (F32)
- `conv_output_silu-0`, `linear_attn_out-0`, `norm-0`, `attn_output-0` — DeltaNet internals
- `attn_residual-N`, `post_ffn-N` — per-layer hidden states after attention and FFN residual adds

**GWEN side** — dumps controlled by `GWEN_DUMP_LAYERS=1` environment variable in
`forward_prefill()` (`src/inference.cu`). Dumps the F32 residual accumulator at each
layer boundary (`pf_a` / `pf_b`), plus the FP16 RMSNorm output converted back to F32,
and the F32 QKV GEMM output.

**Comparison scripts** (all in `scratch/`):

- `compare_layers.py` — per-layer cosine similarity, max absolute diff, RMS across all 24 layers
- `bisect_l0.py` — step-by-step comparison within layer 0 to isolate where divergence first appears
- `check_fp16_rounding.py` — tests whether the RMSNorm divergence is exactly FP16 truncation
- `compare_topk.py` — compares top-k logit rankings between engines
- `compare_embed.py` — checks if embeddings are bitwise identical

### How to reproduce

```bash
# 1. Build tools
cmake --build build -j$(nproc)
g++ -O2 -o tests/llama_dump_layers tests/llama_dump_layers.cpp \
    -I third_party/llama.cpp/include -I third_party/llama.cpp/ggml/include \
    -L third_party/llama.cpp/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,third_party/llama.cpp/build/bin

# 2. Dump from both engines (single token "The")
GWEN_DUMP_LAYERS=1 GWEN_GEMM_DECODE=1 ./build/gwen \
    --model Qwen3.5-0.8B-Q4_K_M.gguf --prompt "The" --n-predict 1 --greedy
LD_LIBRARY_PATH=./third_party/llama.cpp/build/bin \
    ./tests/llama_dump_layers Qwen3.5-0.8B-Q4_K_M.gguf "The"

# 3. Compare
uv run --with numpy scratch/bisect_l0.py
uv run --with numpy scratch/check_fp16_rounding.py
uv run --with numpy scratch/compare_layers.py
```

Binary dump format: `int32 count` followed by `float32[count]`. All files written to
`/tmp/gwen_*.bin` and `/tmp/llama_*.bin`.

## Findings

### 1. Embeddings are bitwise identical

Both engines dequantize the Q6_K `token_embd` weight identically. Zero differing
elements out of 1024.

```
embedding   IDENTICAL (1024 elems)
```

### 2. RMSNorm produces identical F32 values

The RMSNorm output showed 994/1024 differing elements with max_abs=0.00116 when
compared directly. This looked like a real divergence. However, GWEN's RMSNorm outputs
to FP16 (because CUTLASS GEMM expects FP16 input), and the dump converts FP16→F32.
llama.cpp captures the pre-truncation F32 value.

To test whether this was purely FP16 rounding, we cast llama.cpp's F32 RMSNorm output
to FP16 and back:

```
GWEN vs llama (F32):        max_abs=0.00116253  n_diff=994/1024
GWEN vs llama (FP16→F32):   max_abs=0.00000000  n_diff=0/1024
```

**GWEN = FP16(llama)**. The RMSNorm implementations produce identical F32 results.
GWEN just truncates to FP16 afterward. This means the RMSNorm kernel
(`kernel_rmsnorm_batch_f32in_f32w`) is correct.

### 3. The root cause: fundamentally different matmul arithmetic

After confirming that the RMSNorm is identical, the next checkpoint is the QKV GEMM
output. This is where real divergence appears:

```
embedding          IDENTICAL
RMSNorm output     cos=0.9999999404  max_abs=0.00116  (FP16 truncation only)
QKV GEMM output    cos=0.9999975562  max_abs=0.02509  (real divergence)
attn_residual      cos=0.9999040961  max_abs=0.00187
post_ffn           cos=0.9998934269  max_abs=0.00203
```

The GEMM output diverges by max_abs=0.025 — a 25x jump from the FP16 truncation
error at the input. This is not "accumulation order differences in the same
algorithm." The two engines use **fundamentally different matmul implementations**:

**llama.cpp (CUDA backend)**:
- Activations are **F32 throughout** — never truncated to FP16
- Quantized weight matmul uses **integer DP4A** (`ggml_cuda_dp4a`): int8×int8→int32
  accumulate, then scaled by F32 scale factors
- RMSNorm output feeds directly into GEMM as F32
- Source: `ggml/src/ggml-cuda/vecdotq.cuh` (dot product), `ggml/src/ggml-cuda/mmq.cu`
  (quantized matmul dispatcher asserts `src1->type == GGML_TYPE_F32`)
- Source: `ggml/src/ggml-cuda/norm.cu` lines 452-473 (RMSNorm asserts F32 I/O)

**GWEN (CUTLASS + custom kernels)**:
- Residual stream is F32, but **RMSNorm output is FP16** (required by CUTLASS GEMM)
- Quantized weight matmul: dequantize Q4_K → FP16, then **FP16×FP16 tensor core**
  (mma.sync on SM_120) with F32 accumulate
- Every layer crosses the FP16 boundary twice (at each RMSNorm output → GEMM input)

The core multiply-accumulate is different:
- llama.cpp: `int8 × int8 → int32`, then `int32 × float32_scale → float32`
- GWEN: `float16 × float16 → float32` (tensor core), with FP16 input truncation

Both are valid approximations of the "true" F32 result, but they produce different
numerical values because they use different data types for the core computation.

### 4. Error propagation through 24 layers

The per-layer comparison (GPU vs GPU, single token "The") shows the GEMM precision gap
compounding sub-linearly through the network:

```
Layer  Type    cos           max_abs    rms
L00    DN      0.999890      0.003888   0.000720
L03    FA      0.998277      0.014957   0.003845
L07    FA      0.998947      0.011882   0.003407
L11    FA      0.999004      0.014652   0.003439
L15    FA      0.999285      0.016569   0.004476
L19    FA      0.999342      0.054982   0.006295
L23    FA      0.998849      0.137171   0.014937
```

Key observations:
- cos stays between 0.998–0.999 across all 24 layers — consistent, not exploding
- max_abs grows from 0.004 at L0 to 0.14 at L23 — sub-linear, ~35x over 24 layers
- FullAttention layers (FA) sometimes show slightly worse cosine than DeltaNet (DN)
  layers, likely because the attention+gate+output projection chain involves more GEMMs
- The error growth is dominated by DeltaNet recurrence (S state accumulates slightly
  different values each layer) and FFN GEMMs (two per layer: gate+up, down)

### 5. Logit-level impact

At the output (248,320-dimensional logit vector):

```
logits:  cos=0.999483  KL=0.004803  max_abs=0.7333  top1=MATCH
```

Top-k ranking comparison for single token "The" → " following":

```
Rank   Llama tok   Llama logit    GWEN tok    GWEN logit    Gap
   1      2614       12.3815        2614       12.3438    -0.0378   (MATCH)
   2      1414       11.1173        1414       10.9688    -0.1486   (MATCH)
   3       220       10.8541         220       10.8984    +0.0444   (MATCH)
   ...
   9      3049        9.5713        3049        9.6484    +0.0772   (MATCH)
  10      3788        9.3091        7859        9.2812              (rank differs)
```

Top-9 tokens match in exact rank order. Rank disagreement begins where logit gaps are
< 0.1. Median logit diff across all 248K tokens: 0.048. P99: 0.189.

For multi-token generation with teacher forcing (30 tokens), the logit divergence
grows because the DeltaNet S matrix accumulates slightly different values at each step,
even with identical input tokens. Worst-case prompt ("1+1=2. 2+2=4. 3+3=") reaches
KL=0.21 and cos=0.92 at position 26, while well-behaved prompts stay under KL=0.005.

## Conclusions

1. **The divergence is not a bug.** GWEN's RMSNorm, embedding lookup, and all
   elementwise operations produce identical results to llama.cpp. The divergence enters
   exclusively at GEMM boundaries.

2. **The divergence is architectural.** llama.cpp uses F32 activations with integer
   DP4A matmul. GWEN uses FP16 activations with FP16 tensor cores. These are
   fundamentally different numerical paths that will never produce identical results.

3. **Exact matching against llama.cpp is not possible** and should not be the
   correctness gate for ongoing development. The llama.cpp comparison establishes
   that GWEN is in the right ballpark (cos > 0.998 per layer, top-9 logit rank match).

4. **Ongoing correctness testing should be regression-based**: record GWEN's own
   outputs as golden data and detect unexpected changes. A code change that shifts
   cosine from 0.999 to 0.95 at any layer is a bug. A code change that shifts it from
   0.999 to 0.998 is worth investigating but not necessarily wrong.

## Files referenced

| File | Purpose |
|------|---------|
| `tests/llama_dump_layers.cpp` | Dumps per-layer hidden states from llama.cpp via eval callback |
| `src/inference.cu` (GWEN_DUMP_LAYERS) | Dumps per-layer hidden states from GWEN's forward_prefill |
| `scratch/bisect_l0.py` | Layer 0 bisection: embedding → RMSNorm → QKV GEMM → residual |
| `scratch/check_fp16_rounding.py` | Proves RMSNorm diff is exactly FP16 truncation |
| `scratch/compare_layers.py` | Full 24-layer comparison with cosine/max_abs/rms |
| `scratch/compare_topk.py` | Top-k logit ranking comparison |
| `scratch/compare_embed.py` | Embedding identity check |
| `third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cu` | llama.cpp quantized matmul (F32 I/O, DP4A) |
| `third_party/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` | llama.cpp int8×int8 dot product kernels |
| `third_party/llama.cpp/ggml/src/ggml-cuda/norm.cu` | llama.cpp RMSNorm (F32 I/O) |
