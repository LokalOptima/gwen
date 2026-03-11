# GWEN Blog — Building a GPU-native Qwen3.5 Inference Engine

A technical blog documenting the journey of reimplementing Qwen3.5-0.8B inference in pure CUDA/CUTLASS, hyper-optimized for the RTX 5070 Ti.

## Posts

0. [Starting GWEN](00-starting-gwen.md) — Motivation, architecture overview, and why this model is interesting
1. [Building the Kernel Arsenal](01-kernel-arsenal.md) — Core CUDA kernels: dequant, GEMV, RMSNorm, RoPE
2. [Taming DeltaNet on the GPU](02-taming-deltanet.md) — Implementing the Gated DeltaNet linear attention recurrence
3. [Squeezing the 5070 Ti](03-squeezing-the-5070ti.md) — Kernel fusion, CUDA graphs, 184 → 251 tok/s decode
4. [Prefill: Going Parallel](04-prefill-going-parallel.md) — Batched GEMM, dequant bug fix, 2x prefill speedup
5. [Final Results and Lessons Learned](05-final-results.md) — Complete benchmarks vs llama.cpp at 252 tok/s, retrospective
6. [Beating llama.cpp](06-beating-llama-cpp.md) — dp4a GEMV, DeltaNet fusion, 252 → 452 tok/s (+79%), surpassing llama.cpp by 4%
7. [Kernel Fusion + CUTLASS](07-kernel-fusion-cutlass.md) — Fused GEMV+residual, SwiGLU+Q8_1, RMSNorm+Q8_1, dropped cuBLAS for CUTLASS, 452 → 493 tok/s (+10.6% vs llama.cpp)
8. [Profiling and Kernel Optimization](08-profiling-and-kernel-optimization.md) — ncu/nsys profiling, multi-block argmax (40x), multi-warp GQA (5.9x), 493 → 599 tok/s (+34.3% vs llama.cpp)
9. [MTP Speculative Decoding](09-mtp-speculative-decoding.md) — Qwen3.5's built-in MTP draft head, Gemma-style RMSNorm bug, Q8_0 quantization, sequential verify bottleneck analysis
