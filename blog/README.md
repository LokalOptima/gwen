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
10. [Token Frequency Distribution](10-token-frequency-distribution.md) — 1.6B-token frequency analysis of Qwen3.5's 248K vocabulary, Zipf's law, optimization opportunities
11. [Activation Replay and the MTP Verdict](11-activation-replay-mtp-verdict.md) — S snapshot rollback (42x faster reject), algebraic undo failures, and why 55% acceptance isn't enough
12. [Fine-Tuning the MTP Head for Spoken English](12-mtp-finetuning-setup.md) — 498M-token spoken English corpus, restricted vocab (20K tokens, 95.9% coverage), training infrastructure for the MTP head
13. [Training the MTP Head](13-mtp-finetuning-training.md) — DeltaNet VRAM catastrophe (13 GB for batch=1), token-budget batching vs padding, BF16 autocast fix, full training run
14. [Batched GEMM Hidden State Extraction](14-batched-gemm-extraction.md) — CUDA server for training data extraction, 20K tok/s at B=64, DeltaNet recurrence bottleneck (54%), optimization attempts and lessons
15. [Re-Verifying Against Latest llama.cpp](15-re-verify-llama-cpp.md) — L2 norm fix, GEMM vs GEMV decode precision gap, FP16 tie-breaking floor, 2/4 exact match + 2/4 within 0.01 logit
16. [Why GEMV Destroys GEMM at N=1](16-failed-gemm-optimization.md) — Three failed optimizations: CUDA-graphed GEMM decode (6% not 6×), wave-limited DeltaNet (0%), shared-memory S (0%). Root cause: 7× bandwidth amplification from dequant pipeline, L1 already caching S, serial token dependency
17. [Profiling All Three Code Paths](17-profiling-all-paths.md) — nsys profiling of GEMV decode, GEMM decode, and batch extraction. DeltaNet at 62% dominates batch path. Dequant+GEMM at 79% dominates GEMM decode. Next targets: chunkwise DeltaNet, fused dequant-GEMM
18. [Chunkwise DeltaNet: Breaking the Serial Dependency](18-chunkwise-deltanet.md) — 4-kernel chunkwise decomposition (WY representation + parallel state propagation), three gating bugs found via numpy reference test, shared-memory matmul optimization for inner loops, 17% batch extraction speedup at B=63
19. [MTP Training Pipeline: Three Bugs and a Diagram](19-mtp-training-pipeline-fixes.md) — Buffer aliasing corruption in full attention layers, wrong training targets (ground-truth vs main model predictions), MTP weight loading order, and discovering the CUDA/PyTorch FP16/BF16 precision mismatch
20. [Aligning CUDA and PyTorch MTP](20-aligning-cuda-pytorch-mtp.md) — Closing the train/inference gap: Q6K-dequantized embeddings, FP16 autocast, reduced lm_head, batch extraction non-determinism, vocab size analysis (20K is the sweet spot), logit-level verification
21. [MTP v3: Soft Target Distillation](21-mtp-v3-soft-distillation.md) — KL divergence distillation replacing hard targets, vocab reduction (20K→4K), two-stage training, gwen_dev_server for teacher logit extraction, OOM and buffer overflow bugs caught in review
22. [Squeezing Speculative Decode](22-squeezing-speculative-decode.md) — Profiling-driven optimization of speculative decode cycle, fused batch2 kernels, 600 → 928 tok/s peak
23. [Server-Side p_idk, Clean Pipeline, and Shared Memory](23-pidk-pipeline-and-shm.md) — Moving p_idk computation to CUDA server (eliminating 62h precompute), numbered pipeline scripts, shared memory transport for 25% training throughput boost
24. [Sparse Distillation: Top-k + IDK Bucket](24-sparse-distillation-topk-idk.md) — Literature survey on sparse teacher logits, the renormalization bias problem, and a 31× bandwidth reduction by sending only top-64 logits with remaining mass folded into the IDK bucket
25. [MTP v5 Sparse Distillation — Results](25-v5-sparse-distillation-results.md) — v5 trained and benchmarked: 816 tok/s (+27% over baseline, +4% over v3), IDK neuron is an OOV detector (0.891 correlation), top-64 captures 94.7% of restricted mass, next steps: reduce IDK over-abstention
26. [Marlin-Style Tensor Core GEMV — Dead End](26-marlin-tensor-core-gemv.md) — Implemented Marlin-style mma.sync GEMV with weight reshuffling, lop3 dequant, cp.async pipeline, K-splitting. Result: dp4a still wins by 1.5-4x at all relevant matrix sizes. Batch=1 GEMV is parallelism-bound, not instruction-bound. Code parked on `marlin` branch.
27. [The Instruct Model Mistake](27-the-instruct-model-mistake.md) — All MTP work used the instruct model with raw text (out-of-distribution), inflating acceptance rates on degenerate output. Switching to the base model for correct train/eval alignment.
28. [v7 OOV Gate Experiment](28-v7-oov-gate-experiment.md) — Separate binary OOV classifier + clean token head. Measured cycle costs, trained with OOV-masked loss, benchmarked against v6 IDK. Result: v6 still wins (+19% vs +17%). OOV accounts for less than half of rejections.
29. [Matching and Beating llama.cpp Prefill](29-matching-and-beating-llama-cpp-prefill.md) — Multi-query flash attention, pre-normalized DeltaNet Q/K, CUTLASS tile tuning, MMA tensor core attention ported from llama.cpp. 22,319 → 29,497 tok/s (+32%), 1.12× llama.cpp across all prompt lengths. Five dead ends documented.
30. [Eliminating the FP16 Weight Copy](30-eliminating-the-fp16-weight-copy.md) — Removed 924 MB FP16 pre-dequant by porting llama.cpp's fused mmq kernel (stream-K + Turing MMA). Five failed custom kernels, per-call dequant as interim fix. 2,129 → 1,341 MB VRAM (-37%), prefill 28% slower but decode unaffected.
31. [FP8 — The Right Format for SM_120](31-fp8-the-right-format.md) — Abandoning Q4_K GGML for FP8 E4M3 native tensor core format. Complete GWFP8 pipeline: quantizer, loader, GEMV kernels, F32 residual decode path. CUTLASS block scaling analysis for prefill GEMM.
32. [Porting GWEN to 4B with NVFP4](32-nvfp4-4b-port.md) — Scaling to Qwen3.5-4B with FP4 E2M1 quantization: V/K head asymmetry, GWFP4 format, FP4 GEMV kernels, and debugging the `(1+weight)` RMSNorm convention bug via 3-way comparison.
33. [FP4 GEMV: From 12 to 139 tok/s](33-fp4-gemv-optimization.md) — Constant memory serialization was the bottleneck: shared memory LUT gave 3.1×, CUDA graph 3.3×, thread utilization fix 6%. 12.4 → 139.1 tok/s, 75.6% of llama.cpp.
