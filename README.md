# GWEN

A from-scratch CUDA reimplementation of [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B) inference.

~11,000 lines of C++17/CUDA. No frameworks. No abstractions between you and the hardware.

---

## What this is

GWEN loads a GGUF-quantized Qwen3.5 model and runs inference entirely in hand-written CUDA kernels, targeting a single NVIDIA GPU. Every operation — dequantization, matrix-vector products, RMSNorm, RoPE, softmax, the DeltaNet linear attention recurrence, GQA full attention — is a kernel written for this specific architecture on this specific hardware.

It is not a general-purpose inference engine. It runs one model family on one GPU. That constraint is the point.

### The model

Qwen3.5 is a hybrid architecture — not a standard transformer:

- **DeltaNet layers**: linear attention with a learned recurrent state matrix (O(1) memory per step, no KV cache)
- **Full attention layers**: grouped-query attention, every 4th layer
- Pattern: `[DeltaNet, DeltaNet, DeltaNet, FullAttn] x N`
- Mixed quantization: Q4\_K, Q5\_K, Q6\_K, Q8\_0, F16, F32

### Performance

Measured on RTX 5070 Ti (SM\_120, Blackwell), GDDR7 896 GB/s, clocks locked.

Qwen3.5-9B (UD-Q4\_K\_XL):

| | tok/s |
|---|---:|
| Decode | 122 |

---

## Building

Requires CUDA 13.1+ and CMake 3.24+.

```bash
git clone --recursive https://github.com/LokalOptima/gwen.git
cd gwen
make
```

CUTLASS is included as a git submodule.

---

## Usage

Input is automatically wrapped in the ChatML template for instruct models:

```bash
./build/gwen "Where is Firenze in Italy?"                   # ChatML-wrapped, thinking enabled
./build/gwen --no-reason "Where is Firenze in Italy?"       # disable thinking
./build/gwen --max-predict 200 "Explain quantum computing"  # generate up to 200 tokens
./build/gwen --raw "1 2 3 4 5 6 7 8"                        # raw text completion (no ChatML)
```

Override model path:

```bash
./build/gwen --model /path/to/model.gguf "Hello"
```

Weights are auto-downloaded to `~/.cache/gwen/` on first run if no `--model` is specified.

---

## Project structure

```
src/
  main.cu              CLI entry point
  inference.cu         Forward pass, generation loops
  model.cu             GGUF loading, weight upload, model configuration
  gguf.cu              GGUF file parser
  tokenizer.cu         BPE tokenizer (from GGUF vocab)
  memory.cu            RAII GPU memory allocator
  server.cpp           HTTP inference server
  kernels/
    gemv.cu            Quantized GEMV (Q4_K, Q5_K, Q6_K, Q8_0 via dp4a)
    gemm_cutlass.cu    CUTLASS-based batched GEMM for prefill
    gemm_mmq.cu        Fused K-quant GEMM (vendored from llama.cpp)
    dequant.cu         Dequantization kernels
    rmsnorm.cu         RMSNorm + fused quantize variants
    rope.cu            Rotary position embeddings (interleaved RoPE)
    softmax.cu         Softmax + fused top-k
    activation.cu      SwiGLU + fused quantize
    reduction.cu       Multi-block argmax, reductions
    fattn_mma.cu       Flash attention (vendored from llama.cpp)
include/gwen/
  inference.h          InferenceState, generation API
  model.h              Model, ModelConfig
  options.h            CLI argument parsing
  kernels.h            Kernel launch wrappers
  gguf.h               GGUF parser
  tokenizer.h          Tokenizer
  memory.h             CudaAllocator, CudaBuffer
  common.h             GWEN_CHECK_CUDA, utilities
bench/
  gwen_bench.cu        Benchmark (llama-bench compatible output)
  profile_forward.cu   Per-kernel timing with CUDA events
tests/
  test_kernels.cu      Kernel unit tests
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
