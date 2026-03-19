# GWEN 0.8B

A from-scratch CUDA reimplementation of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) inference.

~11,000 lines of C++17/CUDA. No frameworks. No abstractions between you and the hardware.

---

## What this is

GWEN loads a GGUF-quantized Qwen3.5-0.8B model and runs inference entirely in hand-written CUDA kernels, targeting a single NVIDIA GPU. Every operation — dequantization, matrix-vector products, RMSNorm, RoPE, softmax, the DeltaNet linear attention recurrence, GQA full attention, speculative decoding — is a kernel written for this specific model on this specific architecture.

It is not a general-purpose inference engine. It runs one model on one GPU. That constraint is the point.

### The model

Qwen3.5-0.8B is a hybrid architecture — not a standard transformer:

- **18 DeltaNet layers**: linear attention with a learned recurrent state matrix (O(1) memory per step, no KV cache)
- **6 full attention layers**: grouped-query attention (8Q / 2KV heads, dim 256), every 4th layer
- Pattern: `[DeltaNet, DeltaNet, DeltaNet, FullAttn] x 6` = 24 layers
- Mixed quantization: Q4\_K, Q5\_K, Q6\_K, Q8\_0, F32

### Performance

Measured on RTX 5070 Ti (SM\_120, Blackwell), GDDR7 896 GB/s, clocks locked:

| | tok/s |
|---|---:|
| Baseline decode (no speculation) | 643 |
| With MTP speculative decoding (v5) | **816** |
| Peak speculative decode | 928 |

The theoretical bandwidth limit for this model is ~1,594 tok/s. GWEN reaches 51% of that ceiling in sustained decode and 58% at peak. The gap is well-understood: Q4\_K's struct-of-arrays layout causes 71% sector waste in small GEMVs, and CUDA graph dispatch adds ~260 us per step.

### Speculative decoding

Qwen3.5 ships with a built-in MTP (Multi-Token Prediction) draft head. GWEN fine-tunes this head on spoken English data using knowledge distillation from the base model's own logits, then uses it for speculative decoding during inference. The v5 head uses sparse top-64 distillation with an IDK (I Don't Know) neuron that learns to abstain on out-of-vocabulary tokens rather than guess wrong.

---

## Building

Requires CUDA 13.1+ and CMake 3.24+.

```bash
git clone --recursive https://github.com/LokalOptima/gwen.git
cd gwen
make
```

Requires CUDA 13.1+ and CMake 3.24+. CUTLASS is included as a git submodule.

---

## Usage

Set your model paths once at the top of `Makefile`, then:

```bash
make run PROMPT="The meaning of life is"      # baseline decode
make run-mtp PROMPT="The meaning of life is"  # with speculative decoding
make bench                                     # benchmark (baseline)
make bench-mtp                                 # benchmark (speculative)
make info                                      # print model info
make test                                      # correctness tests vs llama.cpp
```

All flags (`MODEL`, `MTP`, `MTP_HEAD`, `PROMPT`, `N`) can be overridden on the command line:

```bash
make run MODEL=~/models/gguf/Qwen3.5-0.8B-Q8_0.gguf PROMPT="Hello" N=200
```

---

## Project structure

```
src/
  main.cu              CLI entry point
  inference.cu         Forward pass, generation loops, speculative decode
  model.cu             GGUF loading, weight upload, model configuration
  gguf.cu              GGUF file parser
  tokenizer.cu         Tokenizer (from GGUF vocab)
  memory.cu            RAII GPU memory allocator
  server.cpp           HTTP inference server
  dev_server.cpp       Training data extraction server
  kernels/
    gemv.cu            Quantized GEMV (Q4_K, Q5_K, Q6_K, Q8_0 via dp4a)
    gemv_fp16.cu       FP16 GEMV (reference path)
    gemm_cutlass.cu    CUTLASS-based batched GEMM for prefill
    dequant.cu         Dequantization kernels
    rmsnorm.cu         RMSNorm + fused quantize variants
    rope.cu            Rotary position embeddings (interleaved RoPE)
    softmax.cu         Softmax + fused top-k
    activation.cu      SwiGLU + fused quantize
    reduction.cu       Multi-block argmax, reductions
    topk.cu            Top-k selection (CUB radix sort)
include/gwen/
  inference.h          InferenceState, generation API
  model.h              Model, ModelConfig
  kernels.h            Kernel launch wrappers
  gguf.h               GGUF parser
  tokenizer.h          Tokenizer
  memory.h             CudaAllocator, CudaBuffer
  common.h             GWEN_CHECK_CUDA, utilities
train/
  train_mtp.py         MTP head fine-tuning (knowledge distillation)
  model.py             PyTorch MTP model definition
  dataset.py           Training data pipeline
bench/
  profile_forward.cu   Per-kernel timing with CUDA events
  micro_bench.cu       Micro-benchmarks
tests/
  test_kernels.cu      Kernel unit tests
  test_dp4a.cu         dp4a GEMV correctness tests
  test_gemm.cu         CUTLASS GEMM vs GEMV comparison
```

---

## Correctness

All outputs are verified against llama.cpp with equivalent settings. Greedy decoding produces exact token matches across multiple prompts. Logit divergence stays below KL < 0.01, explained entirely by FP16 vs F32 accumulation differences.

```bash
./scripts/test_correctness.sh
```

---

## License

MIT
