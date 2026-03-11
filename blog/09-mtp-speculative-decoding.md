# Multi-Token Prediction: Speculative Decoding with Qwen3.5's Built-in Draft Head

*Blog post #9 in the GWEN series — when your model predicts two tokens at once*

## The MTP Opportunity

Qwen3.5 ships with a built-in Multi-Token Prediction (MTP) head — a small transformer layer that predicts the *next-next* token from the current hidden state. This is essentially a free draft model: no separate model to load, no distillation, just 15 extra weight tensors (39 MB FP16, 21 MB Q8_0) sitting in the safetensors file that most frameworks ignore entirely.

The idea: use MTP as a speculative decoding draft. If the draft is accepted, we get two tokens per verification cycle instead of one. At ~600 tok/s baseline, even a modest acceptance rate could push throughput significantly higher.

## Architecture

The MTP head takes two inputs:
1. The last layer's hidden state (1024 dims) — carries the full context from 24 layers of processing
2. The embedding of the predicted next token (1024 dims)

These get norm'd and concatenated, then projected through:
```
h' = FC(concat(RMSNorm(embed), RMSNorm(hidden)))   # [2048] → [1024]
```

Then one full transformer layer (same architecture as the main model's attention layers: GQA with 8 Q-heads, 2 KV-heads, head_dim=256, gated attention, SwiGLU FFN), followed by RMSNorm and the shared lm_head.

The MTP doesn't need its own prefill — the 24-layer hidden state already encodes full context. Even single-token MTP decode works because context flows through the main model's state, not MTP's history.

## The Gemma-Style Norm Bug

First attempt: garbage predictions (tokens like 88537, 116349, 225066). Completely meaningless.

After hours of debugging RoPE conventions, attention gating, concat ordering — the root cause was **RMSNorm weights**. Qwen3.5 uses `Qwen3_5RMSNorm`, which computes:
```python
output * (1.0 + self.weight)   # NOT output * self.weight
```

The safetensors stores raw weights (initialized to ~0). GGUF's converter adds 1.0 during conversion. Our MTP weight extractor was using the raw values, so every norm was computing `x * (1 + 0) ≈ x * 1` instead of the correct scale.

Fix: add `+ 1.0` during extraction for all norm tensors:
```python
if is_norm_tensor(name):
    return tensor.astype(np.float32) + 1.0, DTYPE_F32
```

After this fix, MTP predictions immediately became meaningful.

## Weight Extraction and Quantization

We built a custom binary format (GWMT) for MTP weights since they're not in the GGUF file. The extraction script (`scripts/extract_mtp_weights.py`) reads from safetensors and outputs either FP16 or Q8_0 quantized weights.

### FP16 vs Q8_0 MTP weights

| Format | Size | Acceptance Rate | Notes |
|--------|------|----------------|-------|
| FP16 | 39 MB | 43.4% avg | Baseline |
| Q8_0 | 21 MB | 43.5% avg | Identical output in 7/8 cases |

Q8_0 quantization has zero practical impact on draft quality. The acceptance rates are identical across all 8 test prompts (one case actually improved: 64.6% → 65.7%). This makes sense — the MTP head is a single layer, so quantization error doesn't compound.

Both formats support the same inference path: Q8_0 uses the existing `gwen_gemv_q8_0` kernel (FP16 input, Q8_0 weights), while FP16 uses `gwen_gemv_fp16`.

## Speculative Decode Loop

The speculative decode loop follows the standard Leviathan et al. 2023 pattern (greedy variant):

1. Main model produces token A, hidden state h
2. MTP predicts draft D from (embed(A), h)
3. **Verify**: run main model on [A, D] sequentially, getting pred_0 and pred_1
4. If pred_0 == D: accept both D and pred_1 (2 tokens emitted)
5. If pred_0 != D: reject, emit pred_0, restore state, draft again

The DeltaNet recurrent states require checkpointing (19.3 MB per save) for rollback on rejection.

## The Sequential Verify Problem

Here's where things get interesting. vLLM and SGLang verify all draft tokens in **one forward pass** using batched attention. We initially tried this (batch verify with FP16 GEMM for the 2-token batch) but it was **2.7× slower** than sequential verification:

- Sequential verify (2× dp4a GEMV): reads Q4_K weights (0.5 bytes/param) twice
- Batch verify (1× FP16 GEMM for N=2): reads FP16 dequantized weights (2 bytes/param) once
- Net bandwidth: sequential reads 1.0 bytes/param, batch reads 2.0 bytes/param

On a bandwidth-bound GPU (RTX 5070 Ti, 896 GB/s), reading 2× more data per verify cycle dominates. The batch approach would only win with a **batched dp4a kernel** that reads quantized weights once and processes both tokens — that's a future optimization.

## Benchmark Results (200 tokens, 8 prompts)

| Prompt | Baseline | FP16 MTP | Q8_0 MTP | Accept% |
|--------|----------|----------|----------|---------|
| France/Paris | 605 | 377 | 374 | 12.1% |
| Quantum physics | 608 | 435 | 431 | 52.5% |
| Fibonacci code | 615 | 443 | 439 | 55.6% |
| Village story | 606 | 446 | 442 | 60.6% |
| Quick brown fox | 605 | 384 | 381 | 17.2% |
| Climate change | 608 | 434 | 430 | 52.5% |
| ML algorithms | 609 | 415 | 412 | 39.4% |
| Roman Empire | 605 | 453 | 452 | 64.6% |
| **Average** | **608** | **423** | **420** | **43.4%** |

MTP is currently **30% slower** than baseline, even at 60%+ acceptance rates. The math:

- **Accept cycle**: 2 forward passes (verify) + 1 MTP = ~2.15 forward costs → 2 tokens
- **Reject cycle**: 2 forward passes (verify) + 1 forward (state advance) + 1 MTP = ~3.15 forward costs → 2 tokens
- **Effective cost per token**: at 43% acceptance: (0.43 × 2.15 + 0.57 × 3.15) / ((0.43 × 2 + 0.57 × 2)) = ~1.35 forward passes per token
- **Baseline**: 1.0 forward passes per token

So we'd need >75% acceptance rate with sequential verify to break even, or a batched dp4a kernel to cut the verify cost to ~1.2 forward passes.

## What's Next

The path to making MTP faster than baseline:

1. **Batched dp4a GEMV kernel** — read Q4_K weights once, process 2 tokens with dp4a. This would make verify ~1.2× of a single forward instead of 2.0×.
2. **Lighter reject path** — skip the extra `forward()` call on rejection by using the verify's hidden state directly for MTP.
3. **CUDA graph for MTP** — currently using direct kernel launch, could recapture as a graph.

The MTP head itself is correct and the infrastructure is solid. The bottleneck is purely in the verification strategy — a problem shared by all speculative decoding implementations on bandwidth-bound hardware.
