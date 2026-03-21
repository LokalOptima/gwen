# The Instruct Model Mistake

*Blog post #27 in the GWEN series*

## Discovery

All MTP work from post #9 through #25 used `Qwen3.5-0.8B-Q4_K_M.gguf` — the instruct model (fine-tuned from `Qwen/Qwen3.5-0.8B`). The instruct model expects ChatML template tokens (`<|im_start|>`, `<|im_end|>`). Without them, input is out-of-distribution and the model produces degenerate output: repetitive, incoherent, low-entropy text.

Every step of the MTP pipeline fed raw text — no template.

## The Three-Layer Problem

### Layer 1: Training data extraction was out-of-distribution

The training pipeline (posts #12-14, #19-24):

1. Feed raw English text (subtitles, transcripts, web prose) into the instruct model
2. Extract hidden states from the last layer
3. Extract teacher logits (next-token probabilities)
4. Train the MTP head to predict next tokens from those hidden states

The instruct model expects ChatML-wrapped instructions, not raw text. The hidden states it produced were not representative of normal model behavior — they were the model's internal representation of input it doesn't know how to process. The MTP head learned to predict next tokens from these garbage hidden states.

### Layer 2: Evaluation used the same broken setup

The acceptance rate tests (posts #9, #11, #25) also fed raw prompts without template. The model produced degenerate output — repetitive tokens, low-entropy sequences. Low-entropy text is trivially easy to predict: if the model keeps outputting the same few tokens, even a weak draft head gets most of them right. The reported ~52% stock acceptance rate, and improvements through fine-tuning iterations, were measured on this easy-to-predict degenerate output.

### Layer 3: ChatML reveals the real numbers

Wrapping prompts in ChatML template (which the instruct model needs) dropped acceptance to ~40%. The model started producing varied, coherent text with higher entropy — genuinely harder to predict.

But even 40% is measuring an MTP head trained on confused hidden states, evaluated on a model that needs template tokens we weren't providing during training. The entire measurement is suspect.

## The Cascade

```
Raw text → Instruct model (out-of-distribution input)
         → Confused hidden states
         → MTP head trained on confused states
         → Tested on degenerate output (no template)
         → Artificially high acceptance rate
         → Everything looked fine
         → Actually broken at every level
```

Each layer masked the problems below it. Degenerate output was easy to predict, so acceptance rates looked reasonable. Reasonable acceptance rates meant we kept iterating on the training setup (soft distillation, sparse logits, IDK neurons) instead of questioning the foundation.

## Why the Base Model Fixes This

The base model (`Qwen3.5-0.8B-Base`) was trained on next-token prediction over raw text. No template, no instruction tuning. Raw text in, coherent completions out.

- **Training data is in-distribution**: raw subtitles, transcripts, and web text is what the base model was trained on. Hidden states will be meaningful.
- **No template needed**: raw prompts produce coherent output naturally.
- **Use case alignment**: STT correction is text completion — take partial/noisy text and continue or correct it. That's the base model's native task.
- **No distribution mismatch**: training and inference both see raw text through a model trained on raw text.

## Current State vs Target

**`Qwen3.5-0.8B` instruct (what we had):**
- MTP head trained on instruct model's hidden states with raw text input
- All benchmark numbers (52% stock, 71% v3, 82% v5) measured on degenerate output
- With ChatML template: ~40% acceptance on coherent output

**`Qwen3.5-0.8B-Base` (target):**
- Same architecture, same tensor layout, different weights
- MTP head needs retraining with base model as teacher
- Same raw text training data, now in-distribution
- Acceptance rate measured on naturally coherent output

## The Migration

### Step 1: Delete everything

All instruct artifacts removed:
- `Qwen3.5-0.8B-Q4_K_M.gguf`, `Qwen3.5-0.8B-Q8_0.gguf` (instruct GGUFs)
- `Qwen3.5-0.8B-mtp.bin`, `Qwen3.5-0.8B-mtp-q8_0.bin` (MTP heads trained on instruct)
- `lm_head_top{10k,20k,30k,50k}.bin`, `lm_head_r128.bin` (reduced lm_heads from instruct embeddings)
- `~/models/hf/Qwen3.5-0.8B/` (instruct HF safetensors)
- `data/embed_tokens_q6k.npy` (embeddings extracted from instruct GGUF)

Also deleted pre-existing base GGUFs — downloaded from HuggingFace weeks ago, quantized by unknown party. Can't verify provenance, can't trust.

Old MTP training runs (`mtp_v3_k4096` through `mtp_v5_sparse64_k4096`) preserved as historical records but their outputs are dead.

### Step 2: Single source of truth

Fresh download of `Qwen/Qwen3.5-0.8B-Base` HF safetensors from HuggingFace. All artifacts derived from this one source through `scripts/02_prepare_models.py`:

```
HF safetensors (Qwen/Qwen3.5-0.8B-Base)
    │
    ├── convert_hf_to_gguf.py ──→ F16 GGUF
    │                                │
    │                    llama-quantize Q4_K_M
    │                                │
    │                           Q4_K_M GGUF
    │                                │
    │                     patch ssm_alpha/beta
    │                      (F16 GGUF → Q8_0)
    │                                │
    │                         Patched GGUF ← inference/training server
    │                                │
    │               ┌────────────────┴────────────────┐
    │          extract Q6K                     extract F32
    │         token_embd                      output_norm
    │               │                                │
    │    data/embed_tokens_q6k.npy     data/output_norm.npy
    │               │                                │
    └───────────────┴── cross-validate ──────────────┘
```

Every derived artifact traces back to the HF download. Cross-validation verifies embeddings (cos > 0.999 across sample tokens) and output norm (exact match) against HF safetensors.

### Step 3: The ssm_alpha/beta patch

This bug started the investigation. GWEN's DeltaNet kernels hardcode `block_q8_0` struct access for `ssm_alpha` and `ssm_beta`. The instruct GGUF (Unsloth-quantized, importance matrix) stored these as Q8_0, so it worked. The base GGUF (stock `llama-quantize`) stores them as Q4_K — completely different block layout (144 vs 34 bytes/block).

Reading Q4_K data through Q8_0 struct access produces garbage. Since ssm_alpha and ssm_beta control DeltaNet gating (`gate = ssm_a * softplus(alpha_proj + dt_bias)`, `beta = sigmoid(beta_proj)`), garbage values corrupt the recurrent state at layer 0. The corruption cascades through all 18 DeltaNet layers, producing all-zero logits and token 0 output.

The patch takes F16 weights from the F16 GGUF, quantizes to Q8_0, and splices them into the Q4_K_M GGUF with recomputed data offsets. Size increase: ~300 KB on 503 MB. Proper fix (multi-type kernel dispatch or FP16 dequant at load) is TODO.

### Step 4: Training data unchanged

Training data (`train_speech.bin`, `train_tokens.bin`, etc.) and restricted vocab (`restricted_vocab_4096.bin`, `token_counts.bin`) are derived from the tokenizer, not model weights. Both models share the same tokenizer. These files are clean.

### Step 5: Verification

Base model produces coherent completions on raw text:

```
$ ./build/gwen --model ~/models/gguf/Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf \
    "The capital of France is"

The capital of France is Paris. The capital of England is London.
The capital of the United States is Washington, D.C. ...
```

604 tok/s. With the old instruct MTP head (trained on wrong hidden states), acceptance was ~47% average across 8 prompts — included only for before/after comparison once the base-trained head is ready.

## MTP v6: Base Model Retrain

Changes to `train_mtp.py`:
- Embeddings from `data/embed_tokens_q6k.npy` (base model Q6K, extracted by `02_prepare_models.py`)
- Output norm from `data/output_norm.npy` (base model, multiplicative form — no `1+w` conversion needed)
- `--model-dir` default: `Qwen3.5-0.8B-Base`
- Stage 1 (lm_head-only warmup) default changed from 1000 to 0
- Initial eval skipped when stage1 > 0 (lm_head is untrained, number is meaningless)

### Exact commands

```bash
# 1. Start dev server (max-batch 128 for val cache extraction)
flock --shared /tmp/gpu.lock build/gwen_dev_server \
    --model ~/models/gguf/Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf \
    --restricted-vocab data/restricted_vocab_4096.bin \
    --port 8090 --max-batch 128 --max-seq 512 &

# 2. Extract validation cache
uv run scripts/03_extract_val_cache.py \
    --data data/train_tokens.bin \
    --out-dir train/runs/mtp_v6_base_k4096/val_cache \
    --p-idk

# 3. Restart dev server (max-batch 64 to avoid OOM during training)
kill $(pgrep -f gwen_dev_server)
flock --shared /tmp/gpu.lock build/gwen_dev_server \
    --model ~/models/gguf/Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf \
    --restricted-vocab data/restricted_vocab_4096.bin \
    --port 8090 --max-batch 64 --max-seq 512 &

# 4. Train
uv run train/train_mtp.py train \
    --data data/train_tokens.bin \
    --sparse 64 \
    --max-tokens 32768 \
    --idk-weight 0.5 \
    --stage1-steps 50 \
    --out-dir train/runs/mtp_v6_base_k4096

# 4. Export
uv run train/train_mtp.py export \
    --checkpoint train/runs/mtp_v6_base_k4096/best.pt \
    --output ~/models/gguf/Qwen3.5-0.8B-Base-mtp.bin
```

### Results

| Metric | Final (step 19499) |
|--------|-------------------|
| Val loss | 0.2116 |
| Accept rate | 60.7% |
| IDK rate | 25.9% |
| Non-IDK accept | 81.9% |
| Training time | ~9 hours |
| Training data | 473M tokens, 1 epoch |

IDK over-abstention persists: student IDK 25.9% vs teacher OOV ~15%. `--idk-weight 0.5` did not reduce this — the IDK rate locked early and never moved, same pattern as v5. For v7, increasing `idk-weight` above 1.0 may be the correct direction (penalize the mismatch harder, rather than ignoring it).

## Lessons

The instruct model mistake lasted from post #9 through #25 — seventeen posts, five training iterations (stock, v1, v3, v4, v5), three infrastructure rewrites (activation replay, batched extraction, sparse distillation). All built on a model processing input it was never trained to handle.

The fix is straightforward: switch to base model, verify coherent output, retrain MTP head. The training infrastructure transfers. Only the teacher model changes.

When the numbers look reasonable, you stop questioning the setup. We were focused on improving acceptance rates (52% → 71% → 82%) and never asked whether they were measuring anything real. The progression looked like progress, but it was progress within a broken frame.
