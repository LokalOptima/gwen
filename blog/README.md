# GWEN Blog — Building a GPU-native Qwen3.5 Inference Engine

A technical blog documenting the journey of reimplementing Qwen3.5-0.8B inference in pure CUDA/CUTLASS, hyper-optimized for the RTX 5070 Ti.

## Posts

0. [Starting GWEN](00-starting-gwen.md) — Motivation, architecture overview, and why this model is interesting
1. [Building the Kernel Arsenal](01-kernel-arsenal.md) — Core CUDA kernels: dequant, GEMV, RMSNorm, RoPE
2. [Taming DeltaNet on the GPU](02-taming-deltanet.md) — Implementing the Gated DeltaNet linear attention recurrence
3. [First Words](03-first-words.md) — Full model assembly and first successful generation
4. [Squeezing the 5070 Ti](04-squeezing-5070ti.md) — Nsight profiling, kernel fusion, L2 exploitation
5. [Prefill: Going Parallel](05-prefill-going-parallel.md) — Chunked DeltaNet scan and Flash Attention
6. [Final Results and Lessons Learned](06-final-results.md) — Complete benchmarks and retrospective
