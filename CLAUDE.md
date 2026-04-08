# GWEN — Optimized Qwen3.5-0.8B Inference

Stripped-down llama.cpp fork with MTP speculative decoding for Qwen3.5-0.8B.

## Project Rules

### Correctness-First Development
- After any change to source code, run the correctness test: `make test` (or `./scripts/test_correctness.sh`)
- MTP model must produce identical output to base model (greedy, all prompt types, all lengths)
- If outputs diverge, STOP and debug before moving forward

### Benchmarking Discipline
- Use `make bench-decode` or `./scripts/bench_decode.sh` for decode throughput
- Use `make bench-mtp` or `./scripts/bench_mtp_llama.sh` for MTP speculative decode benchmarks
- Use `llama-bench` for controlled tg/pp measurements — never wall-clock minus model-load
- Diverse prompts (12 categories); no single-prompt numbers

### Blog Documentation
- Each implementation phase gets a blog post in `blog/` (markdown)
- Document: what was implemented, decisions, bugs, benchmark results
- Update the blog index (`blog/README.md`) after each new post

### Architecture Reference
- **Model**: Qwen3.5-0.8B (hybrid Gated DeltaNet + Transformer)
- **Layer pattern**: [3× DeltaNet, 1× FullAttn] × 6 = 24 layers
- **Full attention layers**: indices 3, 7, 11, 15, 19, 23
- **DeltaNet layers**: all others (0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22)
- **Reference**: `src/models/qwen35.cpp` (compute graph), GGUF format

### GPU Target: RTX 5070 Ti
- **SM_120** (Blackwell consumer), 70 SMs, 16 GB GDDR7, **896 GB/s** bandwidth
- CUDA 13.1, compile with `-DGGML_CUDA=ON`

### Key Hyperparameters (from GGUF metadata)
```
hidden_size = 1024
num_layers = 24
intermediate_size = 3584
vocab_size = 248320
num_attention_heads = 8 (Q)
num_kv_heads = 2 (KV, GQA 4:1)
head_dim = 256
rope_theta = 10000000.0
rms_norm_eps = 1e-6
ssm_conv_kernel = 4
ssm_state_size = 128
ssm_group_count = 16 (linear attention heads)
ssm_inner_size = 2048
linear_key_head_dim = 128
linear_value_head_dim = 128
full_attention_interval = 4
```

### Building & Running
```bash
make download-models     # download GGUF + MTP sidecar from GitHub release
make completion          # build llama-completion
make bench               # build llama-bench
make server              # build llama-server

# Inference (MTP sidecar auto-discovered from *-mtp.gguf next to model)
./build/bin/llama-completion --no-conversation \
    -m ~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf \
    -p "prompt" -n 100 --temp 0

# Server (OpenAI-compatible API + web UI at http://localhost:8080)
./build/bin/llama-server -m ~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf --port 8080
```

### Project Structure
```
src/                  llama core library
  models/             model-specific compute graphs (qwen35.cpp, delta-net-base.cpp)
ggml/                 ggml tensor library + CUDA backend
common/               utilities (sampling, chat, arg parsing)
tools/                executables (completion, cli, llama-bench, server, mtmd)
include/              public headers (llama.h)
vendor/               third-party (nlohmann/json, cpp-httplib, stb_image)
cmake/                CMake modules
scripts/              benchmarks, correctness tests, utilities
archive/              original from-scratch CUDA implementation (retired)
train/                MTP head fine-tuning code (inactive)
blog/                 development log
```

### Scripts
- `scripts/config.sh` — shared model path defaults (sourced by other scripts)
- `scripts/test_correctness.sh` — MTP correctness regression (12 prompts × 3 lengths)
- `scripts/bench_decode.sh` — decode benchmark (baseline + MTP, 12 diverse prompts)
- `scripts/bench_mtp_llama.sh` — MTP speculative decode with correctness check
- `scripts/bench_lm_head_sizes.sh` — restricted LM head size sweep
- `scripts/test_website.sh` — instruct model website generation quality test

### Agent Discipline
- Before committing source changes: run `make test`. Include results in commit message.
- Before session ends or context compacts: update HANDOFF.md.
- Commit format: `<type>: <summary>` with `Tests:` line.
- New debugging/one-off scripts go in `scratch/` (gitignored), not `tests/` or `scripts/`.

### Dependencies
- CUDA 12.8+ (13.1 recommended)
- CMake 3.14+
- Python 3 with `uv` for utility scripts (`gguf`, `numpy`, `tqdm`)
