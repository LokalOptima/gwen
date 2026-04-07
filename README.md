# GWEN

Optimized inference for [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), built on a stripped-down fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with MTP speculative decoding.

---

## What this is

GWEN runs Qwen3.5-0.8B inference with MTP (Multi-Token Prediction) speculative decoding on a single NVIDIA GPU. The codebase is a gutted llama.cpp — everything not needed for this model family has been removed.

### The model

Qwen3.5-0.8B is a hybrid architecture:

- **18 DeltaNet layers**: linear attention with a learned recurrent state matrix (O(1) memory per step, no KV cache)
- **6 full attention layers**: grouped-query attention (8Q / 2KV heads, dim 256), every 4th layer
- Pattern: `[DeltaNet, DeltaNet, DeltaNet, FullAttn] × 6` = 24 layers

### Speculative decoding

Qwen3.5 ships with a built-in MTP draft head. GWEN uses it for speculative decoding — the draft head proposes a token, the main model verifies, yielding higher effective throughput than vanilla autoregressive decode.

---

## Building

Requires CUDA 12.8+ and CMake 3.14+.

```bash
git clone --recursive https://github.com/LokalOptima/gwen.git
cd gwen
make completion   # build llama-completion
make bench        # build llama-bench
make server       # build llama-server (OpenAI-compatible HTTP API)
```

Or directly with cmake:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-completion llama-bench llama-server -j$(nproc)
```

---

## Usage

```bash
# Completion (greedy decode)
./build/bin/llama-completion --no-conversation \
    -m ~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf \
    -p "The meaning of life is" -n 100 --temp 0

# MTP speculative decode
./build/bin/llama-completion --no-conversation \
    -m ~/.cache/gwen/Qwen3.5-0.8B-mtp-Q8_0.gguf \
    -p "The meaning of life is" -n 100 --temp 0

# HTTP server (OpenAI-compatible API)
./build/bin/llama-server \
    -m ~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf \
    --port 8080
```

---

## Project structure

```
src/                  llama core library (model loading, context, graph, inference)
  models/             model-specific compute graphs (qwen35.cpp, delta-net-base.cpp)
ggml/                 ggml tensor library + CUDA backend
common/               common utilities (sampling, chat, arg parsing, tokenization)
tools/
  completion/         llama-completion CLI
  cli/                llama-cli (interactive)
  llama-bench/        llama-bench benchmarking tool
  server/             llama-server (OpenAI-compatible HTTP API + web UI)
  mtmd/               multimodal support library
include/              public headers (llama.h)
vendor/               third-party (nlohmann/json, cpp-httplib, stb_image)
cmake/                CMake modules
scripts/              benchmarks, correctness tests, utilities
archive/              original from-scratch CUDA implementation (retired)
train/                MTP head fine-tuning code (PyTorch, inactive)
blog/                 development log
```

---

## Testing

MTP correctness: 12 prompts × 3 lengths (50, 200, 500 tokens), base vs MTP must match:

```bash
make test
# or: ./scripts/test_correctness.sh
```

Decode benchmarks:

```bash
make bench-decode
make bench-mtp
```

---

## Acknowledgments

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** and **[ggml](https://github.com/ggml-org/ggml)** by the ggml authors — this project is a stripped-down fork of llama.cpp (MIT License, Copyright 2023-2026 The ggml authors; see [LICENSE.llama-cpp](LICENSE.llama-cpp))
- **[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)** by the Qwen Team, Alibaba Group — the model architecture and weights (Apache 2.0 License)
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CUDA GEMM templates, used as a git submodule (BSD-3-Clause License, Copyright 2017-2026 NVIDIA Corporation & Affiliates)

## License

MIT. llama.cpp portions licensed separately — see [LICENSE.llama-cpp](LICENSE.llama-cpp).
