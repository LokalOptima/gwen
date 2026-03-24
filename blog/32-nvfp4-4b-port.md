# Post 32 — Porting GWEN to 4B with NVFP4

The 0.8B model was the proving ground. Now it's time to scale up: Qwen3.5-4B with NVIDIA's FP4 E2M1 quantization for maximum bandwidth efficiency on the RTX 5070 Ti.

## The 4B Architecture

Qwen3.5-4B is the same hybrid DeltaNet + Transformer architecture, but bigger in every dimension:

| Parameter | 0.8B | 4B |
|-----------|------|-----|
| hidden_size | 1024 | 2560 |
| num_layers | 24 | 32 |
| intermediate_size | 3584 | 9216 |
| n_k_heads | 16 | 16 |
| n_v_heads | 16 | **32** |
| ssm_inner_size | 2048 | 4096 |

The critical difference: **asymmetric V/K heads**. The 0.8B model had `n_k = n_v = 16`. The 4B has `n_k = 16, n_v = 32` — two V heads share each K head. This required changes to every DeltaNet kernel.

## NVFP4 — FP4 E2M1 Quantization

Instead of Q4_K_M (GGML's mixed quantization) or FP8 (what we built for 0.8B), I used NVIDIA's ModelOpt FP4 checkpoint from [AxionML](https://huggingface.co/AxionML/Qwen3.5-4B-NVFP4). The format:

- **4-bit weights**: FP4 E2M1 (16 representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6)
- **Block scales**: FP8 E4M3 per 16 elements
- **Global scale**: FP32 per tensor
- **Dequant formula**: `w = fp4_lut[nibble] × e4m3_block_scale × f32_global_scale`

Model size: **3.4 GB** (vs 2.7 GB for Q4_K_M, vs 8 GB for FP8).

## The GWFP4 Format

I wrote a custom binary format and converter (`scripts/convert_nvfp4.py`) that repacks the ModelOpt safetensors output:

```
Magic "GWF4" (4 bytes)
Version (U32)
N_tensors (U32)
Header_size (U32)
Embedded JSON config
[Per-tensor headers with data/scales offsets]
[64-byte aligned tensor data]
```

FP4 tensors store packed nibbles (2 per byte) + E4M3 block scales + F32 global scale. Non-quantized tensors (norms, embeddings, conv1d) stored as F32 or F16.

## The FP4 GEMV Kernel

Hand-written CUDA kernel with a 16-entry constant memory LUT for FP4→FP16 conversion:

```cuda
__constant__ half c_fp4_lut[16] = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,    // positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  // negative
};
```

Each thread processes 32 FP4 elements per iteration via 128-bit vectorized loads. The inner loop: unpack nibbles → LUT lookup → multiply by E4M3 block scale → multiply by global scale → FMA into accumulator.

## V/K Head Asymmetry

The DeltaNet recurrence maps K heads to V heads: `k_head = v_head * n_k / n_v`. With the 2:1 ratio, V heads 0-1 share K head 0, V heads 2-3 share K head 1, etc. The QKV split also changes — it's no longer `qkv_dim = 3 × n_heads × head_dim` but `qkv_dim = 2 × n_k × dk + n_v × dv = 8192`.

## Bugs Along the Way

### 1. NaN at layer 8 after ~7 tokens

`A_log` has positive values in the 4B model. Without negation: `decay = exp(positive × softplus(...)) > 1` → exponential growth of the S state. Fix: `A_log → -exp(A_log)` in the converter (matching the GGUF convention).

### 2. Conv1d stored as FP16

The converter had a threshold: tensors > 4096 elements → FP16. Conv1d weights (8192 × 1 × 4 = 32768 elements) hit this, but the kernel reads `float*`. Fix: exclude conv1d and 3D tensors from the FP16 path.

### 3. Buffer aliasing in alpha/beta

The FP4 alpha/beta path uses `attn_out` as scratch for half-precision GEMV results, then converts to float decay/beta, then the DeltaNet kernel writes output to `attn_out`. Using the same buffer for input (half*) and output (float*) caused thread races. Fix: use a separate scratch buffer.

### 4. The `(1+weight)` RMSNorm bug

This was the worst one. After fixing everything above, the model generated coherent text initially but degenerated into repetition: *"the knowledge of the knowledge of the knowledge of..."*

I built a 3-way comparison framework:
1. **HF reference**: dequantize FP4→BF16, run through HuggingFace's native Qwen3.5 model
2. **GWEN NVFP4**: our CUDA implementation
3. **llama.cpp Q4_K_M**: independent reference

HF and llama.cpp produced correct output. GWEN degenerated.

The root cause: Qwen3.5 has **two different RMSNorm classes**:
- `Qwen3_5RMSNorm`: uses `output × (1 + weight)` — for input_layernorm, post_attention_layernorm, output_norm
- `Qwen3_5RMSNormGated`: uses plain `output × weight` — for the DeltaNet's per-head gated norm (`linear_attn.norm`)

My converter blindly added `+1.0` to ALL norm weights. This doubled the effective weight of the DeltaNet gated norm in all 24 DeltaNet layers. The error was multiplicative (~2×) and compounded through the recurrent S state, causing gradual degeneration.

Fix: one line — exclude `linear_attn.norm` from the `+1.0` addition.

```python
# Before (wrong):
if "norm" in k and "weight" in k:
    t_f32 = t_f32 + 1.0

# After (correct):
if "norm" in k and "weight" in k and "linear_attn.norm" not in k:
    t_f32 = t_f32 + 1.0
```

## Results

```
--- HF Reference (NVFP4→BF16) ---
The capital of France is **Paris**.

--- GWEN NVFP4 ---
The capital of France is Paris.

--- llama.cpp Q4_K_M ---
The capital of France is **Paris**.
```

All three agree. GWEN runs at ~42 tok/s decode without CUDA graph (conditional FP4/FP8 dispatch prevents capture). The 0.8B FP8 model shows no regression at 313 tok/s.

## SGLang Status

I spent considerable time trying to run SGLang as a reference implementation for the NVFP4 checkpoint. Two blockers:
1. `sgl-kernel` doesn't ship SM120 (Blackwell consumer) binaries
2. Even on SM100, SGLang has [open bugs](https://github.com/sgl-project/sglang/issues/19587) loading Qwen3.5 NVFP4 weights

The dequantize-and-inject-into-HF approach turned out to be more reliable and gave us a ground-truth reference.
