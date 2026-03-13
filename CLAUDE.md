# GWEN — GPU-native Wired Execution for Neural nets

Pure CUDA/CUTLASS re-implementation of Qwen3.5-0.8B inference.

## Project Rules

### Correctness-First Development
- **NEVER implement a new component without a corresponding correctness test against llama.cpp**
- After implementing each kernel/module, immediately run the comparison test (`tests/compare_outputs.py`)
- Maximum acceptable divergence: KL-divergence < 0.01 for logits, exact token match for greedy decoding
- If outputs diverge, STOP and debug before moving forward

### Benchmarking Discipline
- After each kernel is implemented and verified correct, run the micro-benchmark (`bench/micro_bench.cu`)
- After each major milestone (full layer working, full model working), run the macro-benchmark (`bench/macro_bench.py`)
- Record all benchmark results in the blog post for the current phase
- Compare against llama.cpp with equivalent settings (same quantization, same prompt, same GPU)

### Blog Documentation
- Each implementation phase gets a blog post in `blog/` (markdown)
- Document: what was implemented, architectural decisions, bugs encountered and how they were fixed, benchmark results
- Include code snippets for key insights
- Write in first person, technical but accessible style
- Update the blog index (`blog/README.md`) after each new post

### Architecture Reference
- **Model**: Qwen3.5-0.8B (hybrid Gated DeltaNet + Transformer)
- **Layer pattern**: [3x DeltaNet, 1x FullAttn] × 6 = 24 layers
- **Full attention layers**: indices 3, 7, 11, 15, 19, 23
- **DeltaNet layers**: all others (0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22)
- **Reference**: llama.cpp's `src/models/qwen35.cpp` (compute graph), GGUF format

### GPU Target: RTX 5070 Ti (Hyper-Optimized)
- **SM_120** (Blackwell consumer) — uses `mma.sync`, NOT `wgmma`/`tcgen05`
- 70 SMs, 8960 CUDA cores, 280 Tensor Cores (5th gen: FP16/BF16/FP8/FP4/INT8)
- 16 GB GDDR7, **896 GB/s** bandwidth — decode is bandwidth-bound
- **48 MB L2 cache** — exploit aggressively (can cache ~10% of model weights)
- 128 KB L1/shared per SM (up to 99 KB shared per block)
- 48 max warps/SM, 65536 regs/SM → aim ~42 regs/thread at full occupancy
- **No thread block clusters** (1x1x1 only), no TMEM, no multicast
- TMA available for async global→shared copies
- CUDA 13.1
- Use CUTLASS 4.x SM_120 path for GEMM (mma.sync based)
- Hand-written CUDA kernels for bandwidth-bound ops (GEMV decode, RMSNorm, RoPE, softmax, DeltaNet)
- Compile with: `-arch=sm_120 -O3`, test with/without `--use_fast_math`
- Use `__launch_bounds__` on every kernel for occupancy control

### Code Style
- C++17 with CUDA
- Header files in `include/`, implementation in `src/`
- Use `gwen_` prefix for all public symbols
- Error checking: always check CUDA errors with `GWEN_CHECK_CUDA()`
- Memory: use RAII wrappers, no raw `cudaMalloc`/`cudaFree` in application code

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
rope_sections = [11, 11, 10, 0]
rms_norm_eps = 1e-6
ssm_conv_kernel = 4
ssm_state_size = 128
ssm_group_count = 16 (linear attention heads)
ssm_inner_size = 2048
linear_key_head_dim = 128
linear_value_head_dim = 128
full_attention_interval = 4
```

### Agent Discipline (see agent-guide/)
- Before committing source changes: run test suite. Include results in commit message.
- Before optimizing: profile first (see agent-guide/rules/profiling.md). Three failed attempts = go read instead.
- Before session ends or context compacts: update HANDOFF.md (template: agent-guide/templates/handoff.md).
- Commit format: `<type>: <summary>` with `Tests:` and `Perf:` lines (see agent-guide/rules/commits.md).
- Training runs: must produce train_setup.json + train_log.csv (see agent-guide/rules/training.md).
- Pre-commit hooks are active: run `agent-guide/hooks/install.sh` if hooks are missing.
- New debugging/one-off scripts go in `scratch/` (gitignored), not `tests/` or `scripts/`.

### Dependencies
- CUDA 13.1
- CUTLASS 4.x (as git submodule in `third_party/cutlass`)
- llama.cpp (for reference/comparison, installed system-wide or in `third_party/`)
- Python 3 with `uv` for test/bench scripts (use `gguf` package for GGUF parsing)
