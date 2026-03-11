# GWEN Blog — Building a GPU-native Qwen3.5 Inference Engine

A technical blog documenting the journey of reimplementing Qwen3.5-0.8B inference in pure CUDA/CUTLASS, hyper-optimized for the RTX 5070 Ti.

## Posts

0. [Starting GWEN](00-starting-gwen.md) — Motivation, architecture overview, and why this model is interesting
1. [Building the Kernel Arsenal](01-kernel-arsenal.md) — Core CUDA kernels: dequant, GEMV, RMSNorm, RoPE
2. [Taming DeltaNet on the GPU](02-taming-deltanet.md) — Implementing the Gated DeltaNet linear attention recurrence
3. [Squeezing the 5070 Ti](03-squeezing-the-5070ti.md) — Kernel fusion, CUDA graphs, 184 → 251 tok/s decode
4. [Prefill: Going Parallel](04-prefill-going-parallel.md) — Batched GEMM, dequant bug fix, 2x prefill speedup
5. *Planned: Final Results and Lessons Learned*
