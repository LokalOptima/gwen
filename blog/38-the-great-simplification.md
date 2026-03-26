# 38. The Great Simplification

*Removing 4,877 lines of code that don't serve the mission.*

## Motivation

GWEN had accumulated features for multiple use cases: MTP speculative decoding, batch-of-sequences hidden state extraction, a training data extraction server, FP8/FP4 custom model formats, and a 2-token speculative verify pipeline. The codebase was a web of intertwined code paths — batch2 kernels, activation replay buffers, MTP forward passes, reduced vocabulary LM heads — all branching through the same `InferenceState` struct and `forward_body` function.

The actual mission is simple: **fastest possible single-sequence Qwen3.5-9B inference on RTX 5070 Ti.** One model, one GPU, one sequence at a time. Everything else is dead weight.

## What Was Removed

### MTP Speculative Decoding (the biggest cut)

The MTP (Multi-Token Prediction) head was a draft model used for speculative decoding. It required:

- `forward_mtp_body` / `forward_mtp` — a separate transformer forward pass through an FC layer + 1 attention layer + shared lm_head
- `generate_speculative` — the full speculative decode loop with accept/reject logic
- `forward_body_2tok` / `forward_2tok` — a fused 2-token forward that processes the draft and verify tokens through all 24 layers, reading weight matrices once
- `kernel_deltanet_fused_2tok` — a 260-line fused CUDA kernel for processing two tokens through DeltaNet simultaneously
- Activation replay buffers (`dn_S_snapshot`, `dn_replay_conv_row`) for rolling back DeltaNet state on rejection
- `MTPWeights` struct, `load_mtp()`, `ReducedLMHead` (vocabulary-pruned LM head), GWMT/GWRL binary formats
- 20+ `b2_*` scratch buffers in `InferenceState` for token B's activations

All of this is gone. The speculative decode experiment was documented in posts #9-28 — the conclusion was that 55% acceptance rate on the base model wasn't enough to overcome the verification overhead.

### Batch-2 Kernel Variants

Every major kernel had a `_batch2` variant that read weight matrices once and produced two outputs. These existed solely for the 2-token speculative verify path:

- 5 GEMV batch2 templates (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS) — 490 lines in `gemv.cu`
- SwiGLU batch2, Q8_1 quantize batch2 — 133 lines in `activation.cu`
- RMSNorm+Q8_1 batch2 — 89 lines in `rmsnorm.cu`
- Gated RMSNorm batch2 — in `inference.cu`

### Batch-of-Sequences Infrastructure

Hidden state extraction for training data used a parallel prefill pipeline processing B independent sequences:

- `allocate_batch_prefill()` — separate DeltaNet states per sequence
- `extract_hidden_batch()` — GEMM-batched forward reading weights once for all B×L tokens
- `predict_from_hidden()` — per-token argmax over extracted hidden states
- Chunkwise DeltaNet batch buffers (`batch_dn_S`, `batch_dn_conv`, chunk intermediates)
- Server endpoints: `/batch_extract`, `/compare_extract`, `/test_mtp`

The server keeps `/health`, `/tokenize`, and `/extract` (single-sequence hidden state extraction via the standard prefill path).

### Dead DeltaNet Kernels

Five kernel functions in `inference.cu` that were defined but never launched — superseded by fused/fast variants:

- `kernel_deltanet_prep` — standalone prep, superseded by `kernel_deltanet_fused`
- `kernel_deltanet_decode_v2` — split decode variant, never wired up
- `kernel_compute_gate_beta` — standalone gate/beta, superseded by batched variant
- `kernel_deltanet_prefill` — old prefill, superseded by `kernel_deltanet_prefill_fast`
- `kernel_deltanet_prefill_batch` — multi-sequence batch prefill, dead after batch removal

### Dead Utility Code

- `dequant_iq4xs_to_fp16` + `kvalues_iq4nl` — 65 lines of CPU-side IQ4_XS dequantization that was once used for IQ4_XS→FP16 weight conversion at load time. The `weight_from_tensor_convert` wrapper became a passthrough after dp4a IQ4_XS GEMV was added.
- `is_dp4a_type` — exact duplicate of `is_kquant_type` (same 5 types). Merged into one.
- `mmq_scratch_size` — write-only struct field, never read.

## What Remains

The core is clean and focused:

| Component | Purpose |
|-----------|---------|
| `forward_prefill` | Process prompt tokens via GEMM |
| `forward_body` / `forward` | Single-token decode via GEMV + CUDA graph |
| `generate` | Greedy generation loop |
| `extract_hidden` | Hidden state extraction (used by server) |
| Server | `/health`, `/tokenize`, `/extract` |
| Bench/profile | `gwen_bench`, `profile_forward` |

## By the Numbers

| Metric | Before | After |
|--------|--------|-------|
| `inference.cu` | 5,559 lines | 3,322 lines |
| `gemv.cu` | 1,337 lines | 847 lines |
| `activation.cu` | 339 lines | 206 lines |
| `rmsnorm.cu` | 291 lines | 202 lines |
| `main.cu` | 453 lines | 208 lines |
| `model.cu` | 524 lines | 207 lines |
| `inference.h` fields | 103 | 67 |
| **Total delta** | | **-4,877 lines** |

Performance is unchanged: **128.9 tok/s decode, 5,997 tok/s prefill** — identical to before the cleanup.

## Lessons

The best optimization is often deletion. Every line of code is a liability — it has to be compiled, understood, maintained, and navigated. The batch2 kernels were genuine engineering achievements (reading 5 GB of weights once instead of twice per speculative step), but they served a feature that didn't pan out. Keeping them around "in case we need them later" just makes the hot path harder to understand and the codebase harder to change.

The 0.8B model lives on the `0.8b-peak` branch if we ever need to reference the FP8/FP4 work or the MTP experiments. But `main` is now a clean, single-purpose codebase: one model, one GPU, maximum speed.
