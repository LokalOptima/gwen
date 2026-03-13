# Training the MTP Head: Debugging DeltaNet Memory and Token-Budget Batching

*Blog post #13 in the GWEN series — when your 0.8B frozen model eats 13 GB of VRAM*

## From Setup to Training

[Post #12](12-mtp-finetuning-setup.md) laid out the plan: fine-tune the MTP head on 498M tokens of spoken English with a restricted 20K vocab to push acceptance above 65%. The training code was written, the corpus prepared. Time to actually train.

What followed was a multi-hour debugging session against HuggingFace transformers' DeltaNet implementation that I didn't see coming.

## Challenge 1: Stale Token Counts

The first thing to fix was subtle: `token_counts.bin` had been computed from an earlier 1.6B general-purpose corpus, not from the 498M spoken English training data. Since the training corpus is 60% subtitles, the token frequency distribution is materially different from general web text.

Recomputing with `np.bincount` over the training tokens (excluding sentinels) gave updated coverage numbers:

| K | Old Coverage | New Coverage | New Pair Cov | New Avg Run |
|---|-------------|-------------|--------------|-------------|
| 10K | 88.9% | 91.7% | 84.7% | 12.9 |
| 20K | 94.8% | 95.9% | 92.3% | 25.8 |

Spoken English has a tighter distribution than general text — higher coverage at every K. The average in-vocab run at K=20K improved from 17.9 to 25.8 consecutive tokens before hitting an OOV.

## Challenge 2: FP16 vs BF16 Autocast

The original training code used `torch.amp.autocast("cuda")` without specifying dtype. On CUDA, this defaults to **FP16**, not BF16. FP16 has a narrower dynamic range (max ~65K) which can cause overflow in attention computations, especially with QK-Norm where values cluster near 1.0.

Fix: explicit `torch.amp.autocast("cuda", dtype=torch.bfloat16)`. BF16 has the same 8-bit exponent as FP32 — no overflow risk, no need for `GradScaler`.

## Challenge 3: The DeltaNet VRAM Catastrophe

This was the big one. The training loop runs a frozen forward pass through the full 0.8B Qwen3.5 model to extract hidden states, then trains only the 41M-parameter MTP head. Simple enough — the plan estimated ~2.4 GB total memory at batch=64.

The first attempt OOMed immediately. **13.8 GB allocated** for a single forward pass with batch=1.

### What happened

Qwen3.5 uses a hybrid architecture: 18 DeltaNet layers (linear attention with gated delta rule) and 6 standard GQA layers. HuggingFace transformers includes a Python fallback implementation of `chunk_gated_delta_rule` for when the `flash-linear-attention` library isn't installed.

This fallback materializes full [B, heads, L, L] attention matrices per DeltaNet layer. With 18 layers, 16 heads, seq_len=512, the intermediate tensors accumulate to ~12 GB — even at batch_size=1.

The warning message is easy to miss:
```
The fast path is not available because one of the required library
is not installed. Falling back to torch implementation.
```

### The fix: three things

1. **`use_cache=False`** — Prevents Qwen3.5 from storing DeltaNet recurrent state in the return object, which was keeping references alive.

2. **Clone and delete** — The model return object holds references to intermediate tensors. Cloning just the hidden state and deleting the output object frees them:
```python
out = text_model(input_ids, use_cache=False)
h = out.last_hidden_state.clone()
del out
torch.cuda.empty_cache()
```

3. **Use `AutoModelForCausalLM`, not `AutoModel`** — `AutoModel` loads the full multimodal Qwen3.5 model (473 tensors, includes vision encoder). `AutoModelForCausalLM` loads only the text model (320 tensors), and we extract `.model` to discard the 248K-dim lm_head we don't need.

After these fixes, batch=64 at seq=512 uses only 3.84 GB. The naive DeltaNet fallback is still slow (~0.3s/batch), but it works.

### Why not install flash-linear-attention?

I tried. `flash-linear-attention` (the `fla` package) installed fine — it's Triton-based and PyTorch 2.10 ships Triton 3.6 with SM_120 support. But transformers also wants `causal-conv1d` for the conv1d kernel (size 4) in DeltaNet. `causal-conv1d` requires CUDA compilation, and our system has CUDA 13.1 while PyTorch is built against 12.8. PyTorch's `cpp_extension.py` rejects the mismatch.

The conv1d fallback (PyTorch's built-in) handles kernel=4 just fine, so I proceeded without it. The transformers warning persists but performance is adequate.

## Challenge 4: Token-Budget Batching vs Padding

The plan called for token-budget batching (like rokoko's `TokenBatchSampler`): instead of fixed batch_size, target a constant token budget per batch. Short sequences get large batches, long sequences get small batches.

The first implementation sorted sequences by length and packed them to meet the token budget based on **actual** lengths. But the dataset padded all sequences to `seq_len=512`. Subtitle utterances average ~33 tokens, so the sampler packed ~248 short sequences per batch — each padded to 512. The base model processed 127K positions instead of the budgeted 8K. OOM.

The fix: **per-batch padding**. Instead of padding every sequence to `seq_len` in `__getitem__`, return the raw variable-length sequences. A custom `mtp_collate` function pads only to the max length within each batch:

```python
def mtp_collate(batch):
    max_len = max(item["length"] for item in batch)
    token_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = torch.full((len(batch), max(len(item["targets"]) for item in batch)),
                         -100, dtype=torch.long)
    for i, item in enumerate(batch):
        token_ids[i, :item["length"]] = item["token_ids"]
        targets[i, :len(item["targets"])] = item["targets"]
    return {"token_ids": token_ids, "targets": targets}
```

Now a batch of 248 short (33-token) utterances creates a [248, 33] tensor — 8K tokens of actual compute, matching the budget. A batch of 16 long (512-token) sequences creates [16, 512] — also 8K tokens. Memory and compute are consistent regardless of sequence length distribution.

## Sanity Run

Before the full training, I ran 1 epoch on the 8M-token speech subset (`train_speech.bin`):

```
max_tokens=8192, K=20K, lr=1e-4, from_pretrained
927 batches, 291 seconds

train_loss: 5.22 (down from ~9.9 = ln(20K))
val_loss:   4.71
accept_est: 22.4%
oov_frac:   4.9%
```

Loss dropped substantially in one epoch. The 22.4% acceptance is low because (a) only 1 epoch and (b) only the tiny speech subset. OOV fraction of 4.9% matches the expected ~4.1% at K=20K (95.9% coverage).

## Full Training: Configuration

```bash
uv run --with 'torch>=2.7' --with safetensors --with 'transformers>=4.52' \
    --with numpy --with accelerate --with flash-linear-attention --with einops \
    train/train_mtp.py train \
    --data data/train_tokens.bin \
    --counts data/token_counts.bin \
    --model-dir ~/models/hf/Qwen3.5-0.8B \
    --out-dir train/runs/mtp_spoken_v1 \
    --top-k 20000 \
    --epochs 3 \
    --max-tokens 32768 \
    --seq-len 512 \
    --lr 1e-4 \
    --from-pretrained
```

- **3 epochs** over 498M tokens = ~1.5B token presentations, ~36 tokens/param
- **32K tokens/batch** — ~64 sequences at full length, more for shorter sequences
- **14,458 batches/epoch**, 43K total steps
- **5% warmup** (2,168 steps) + cosine annealing

### Data versioning

Every run records exact SHA256 checksums and the full command in `train_setup.json`. No ambiguity about which data produced which checkpoint. Optimizer and scheduler states are saved in checkpoints for clean resume.

## What to Watch

- **train_loss**: Should start ~9.9 (ln(20K) for random lm_head) and drop to ~3-4
- **val_loss**: Should track train. Divergence = overfitting → stop, use best.pt
- **accept_est**: The key metric. Starts ~55% (pre-trained baseline), target >70%
- **oov_frac**: Should stay ~4-5%. Much higher = counts mismatch

*Training is running. Results in the next update.*

## Code Changes Summary

| File | Changes |
|------|---------|
| `train/train_mtp.py` | Explicit BF16 autocast; removed GradScaler; `use_cache=False` + clone/delete for memory; `AutoModelForCausalLM` instead of `AutoModel`; `--max-tokens` replaces `--batch-size`; SHA256 checksums in setup.json; optimizer/scheduler state in checkpoints |
| `train/dataset.py` | `TokenBatchSampler` (sort by length, pack to budget, shuffle batch order per epoch); `mtp_collate` (pad per-batch, not per-dataset); `lengths` property on datasets |
| `data/token_counts.bin` | Recomputed from 498M training corpus (was from old 1.6B general corpus) |
| `scripts/recompute_counts.py` | New script for count recomputation |
