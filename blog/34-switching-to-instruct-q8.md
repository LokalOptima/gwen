# Post 34: Switching to Instruct Q8_0

The previous 33 posts used Qwen3.5-0.8B-Base with Q4_K_M quantization. This post documents switching to the instruct model with Q8_0 quantization, and why.

## Why Instruct

The MTP (Multi-Token Prediction) head in Qwen3.5 was trained as a supplementary pretraining objective, following DeepSeek-V3's design. Both the base and instruct models ship with MTP weights, but they're not identical — the instruct model's MTP head was modified during post-training (SFT/RLHF). Direct comparison of the safetensors shows backbone weights at 0.998-0.999 cosine similarity between base and instruct, while MTP weights diverge to 0.983-0.993. The MTP head was tuned alongside the rest of the model.

Using the instruct model means the MTP head is calibrated for the instruct model's output distribution. This showed up immediately in the acceptance rate: **83.4% average on the instruct model vs 73.3% on the base model** — a 10 percentage point improvement across all 12 test prompts. TCP/UDP went from 47% (the worst prompt, actually slower than baseline) to 85%.

## Why Q8_0

Q8_0 is 8 bits per weight with a single float32 scale factor per block of 32 values. Compared to Q4_K_M (~4.5 bits effective), it's roughly 1.8x more data per weight.

For this model size (772M parameters), Q8_0 fits comfortably in the 5070 Ti's 16 GB VRAM (803 MB vs 504 MB for Q4_K_M). The decode throughput drops by 12-16% due to the extra bandwidth, but the higher MTP acceptance rate of the instruct model almost fully compensates.

No new kernels were needed — Q8_0 is a standard GGML quantization type with existing MMVQ dispatch paths.

## The GGUF Pipeline Problem

Every GGUF on HuggingFace — unsloth, bartowski, everyone — is built with llama.cpp's `convert_hf_to_gguf.py`, which has this line:

```python
if name.startswith("mtp."):
    return  # ignore MTP layers for now
```

No GGUF anywhere includes MTP tensors. There's a WIP PR (#20700) to add MTP support, but it hasn't landed. So the pipeline is:

1. Convert HF safetensors to F16 GGUF (standard converter)
2. Quantize F16 to Q8_0 (`llama-quantize`)
3. Splice MTP tensors back in from the original safetensors (`add_mtp_to_gguf.py`)

Step 3 reads the 15 MTP tensors from the HF checkpoint, applies Qwen's norm+1 convention, and writes them as F16 into block 24 of the GGUF. It also sets `nextn_predict_layers=1` and bumps `block_count` from 24 to 25.

For Q8_0, the MTP head's F16 weights add ~39 MB. The MTP head stays at F16 regardless of the main model's quantization — it's a single transformer layer used only for drafting, so the precision cost is negligible compared to the 24 main layers.

Note: for Q8_0 specifically, self-quantization produces identical results to any community upload (including unsloth). The imatrix-based optimizations that make unsloth quants better only matter at Q4 and below, where bits are scarce. At 8 bits per weight, there's nothing to optimize.

## Script Infrastructure

Extracted shared model paths into `scripts/config.sh`:

```bash
GWEN_CACHE="${GWEN_CACHE:-$HOME/.cache/gwen}"
MODEL_BASE="${MODEL_BASE:-$GWEN_CACHE/Qwen3.5-0.8B-Q8_0.gguf}"
MODEL_MTP="${MODEL_MTP:-$GWEN_CACHE/Qwen3.5-0.8B-mtp-Q8_0.gguf}"
```

All 5 llama-slim benchmark/test scripts source this file. Next time the model changes, it's a single-file edit. Environment variable overrides still work for one-off runs with different models.

## Results

```
Instruct Q8_0 Baseline (tg64):           453 tok/s
Instruct Q8_0 MTP avg (12 prompts):      615 tok/s  (+36% over baseline)
Instruct Q8_0 MTP peak (Fox repeat):     660 tok/s
Instruct Q8_0 MTP worst (Python class):  576 tok/s  (+27% over baseline)
```

Compared to the previous Base Q4_K_M setup:

| Metric | Base Q4_K_M | Instruct Q8_0 | Delta |
|--------|-------------|---------------|-------|
| Baseline tg64 | 540 | 453 | -16% |
| MTP average | 625 | 615 | -2% |
| MTP peak | 711 | 660 | -7% |
| MTP worst | 533 (TCP/UDP, 47%) | 576 (Python class, 72%) | +8% |
| Accept rate avg | 73.3% | 83.4% | +10pp |
| MTP net positive on all prompts? | No (TCP/UDP slower) | Yes | Fixed |

The headline: **MTP is now a net positive on all 12 prompts.** The worst case (Python class, 576 tok/s) is still 27% above the Q8_0 baseline (453 tok/s). The previous worst case (TCP/UDP at 47% acceptance) was actually slower than the Q4_K_M baseline.

The MTP average throughput dropped only 2% despite the main model being 16% slower — the 10pp acceptance rate improvement almost fully compensates.

## Per-Prompt Breakdown

| Prompt | tok/s | Accept% |
|--------|-------|---------|
| AI history | 580 | 70% |
| Narrative | 605 | 78% |
| Business | 606 | 80% |
| TCP/UDP | 616 | 85% |
| Quantum | 601 | 76% |
| Transformer | 635 | 88% |
| Python sort | 605 | 78% |
| Python class | 576 | 72% |
| Math word | 617 | 87% |
| Fibonacci | 623 | 91% |
| Fox repeat | 660 | 98% |
| Counting | 653 | 99% |

## Files Changed

| File | Change |
|------|--------|
| `scripts/config.sh` | New: shared model path defaults |
| `scripts/test_correctness.sh` | Source config.sh, remove inline defaults |
| `scripts/bench_decode.sh` | Source config.sh, add model existence checks |
| `scripts/bench_lm_head_sizes.sh` | Source config.sh |
| `scripts/bench_mtp_llama.sh` | Source config.sh |
| `scripts/bench_prefill.sh` | Source config.sh, add model existence check |
