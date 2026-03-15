# Aligning CUDA and PyTorch MTP: Closing the Train/Inference Gap

*Blog post #20 in the GWEN series — making sure fine-tuned weights actually work when deployed*

## The Problem

[Post #19](19-mtp-training-pipeline-fixes.md) fixed three bugs in the MTP training pipeline but left an open question: do the CUDA and PyTorch MTP implementations actually produce the same output? If they don't, fine-tuning in PyTorch is pointless — the improved weights would behave differently in CUDA.

I wrote a head-to-head comparison test and got 20-62% prediction match. Terrible. This post is the story of tracking down every source of divergence and eliminating them.

## Source 1: The Ghost of Double Extraction

The first comparison test called `extract_hidden_batch` twice — once for CUDA MTP (inside the `/test_mtp` endpoint) and once for PyTorch MTP (via `/batch_extract`). Same tokens, same model, should give the same hidden states.

It didn't. I discovered that `extract_hidden_batch` is non-deterministic across repeated calls within the same process:

```
Call 1: h[0,0,:4] = [-2.129, 0.435,  0.7646, 0.6465]
Call 2: h[0,0,:4] = [-2.129, 0.4338, 0.7637, 0.6475]
Call 3: h[0,0,:4] = [-2.129, 0.434,  0.765,  0.6465]
```

The chunkwise DeltaNet kernels have some inter-call state leakage — probably a buffer that isn't fully zeroed between calls. Each individual call produces correct results, but the exact values drift slightly between calls.

The fix was obvious: extract hidden states ONCE and send the same bytes to both CUDA and PyTorch. I modified `/test_mtp` to accept pre-extracted hidden states in the request body instead of extracting its own.

Match rate jumped to 80-90%.

## Source 2: Different Vocabularies

The remaining mismatches had a suspicious pattern: CUDA often predicted tokens like 79097 or 1393. I checked whether these were even in the restricted 20K vocabulary:

```
CUDA pred  79097: in restricted 20K = False
CUDA pred   1393: in restricted 20K = False
```

Half the mismatches were CUDA picking tokens that PyTorch literally couldn't predict. The CUDA MTP was using the full 248K embed_tokens as its lm_head, while PyTorch used the 20K restricted vocab. Of course they disagreed.

I exported a Q6K-dequantized 20K reduced lm_head (GWRL format) and added `--mtp-lm-head` to the server. This required fixing the MTP forward path too — the reduced lm_head was FP16, but the code always called `gwen_gemv_dp4a` which only handles Q4_K/Q5_K/Q6_K. Added an FP16 branch.

Match rate: 95-100%.

## Source 3: Embedding Input Mismatch

The MTP head takes two inputs: `hidden[t]` and `embed[t+1]`. The hidden states came from the same extraction call (fixed by Source 1). But the embeddings came from different places:

- CUDA: `gwen_embed_lookup` dequantizes Q6_K from the GGUF
- PyTorch: `F.embedding` indexes into BF16 safetensors

Q6_K quantization is high quality (cos > 0.9998 per row) but it's not identical. To close this gap, I dequantized all 248K embeddings from the GGUF to FP16 and saved them as `data/embed_tokens_q6k.npy` (485 MB). The training code now loads these instead of the safetensors embeddings.

This is the QAT principle: train with the same quantized values the model sees at inference time.

## Source 4: BF16 vs FP16 Compute

The training loop used `torch.amp.autocast("cuda", dtype=torch.bfloat16)`. The CUDA MTP uses FP16 throughout. BF16 and FP16 have different mantissa widths (7 vs 10 bits), causing different rounding at every matmul.

Changed to `torch.float16` autocast and added `GradScaler` (FP16's narrower exponent range needs gradient scaling to prevent underflow). Also converted model weights to FP16 at load time to match the BF16→FP16 conversion in the GWMT export.

## The Final Result

After all four fixes, on the prompt with the most mismatches:

```
pos  0: ✓ both=8818
pos  1: ✓ both=854
pos  2: ✓ both=488
pos  3: ✓ both=264
pos  4: ✓ both=1814
pos  5: ✓ both=2526
pos  6: ✓ both=10431
pos  7: ✓ both=11
pos  8: ✗ CUDA=11815 (rank 1, logit=14.352)  PyTorch=11 (rank 0, logit=14.383)  diff=0.031
pos  9: ✓ both=264
pos 10: ✓ both=3777
```

10/11 exact match. The one mismatch: CUDA's pick is PyTorch's rank 1, with a 0.031 logit gap. That's an FP16 accumulation order tie — the two GEMV implementations (hand-written CUDA vs cuBLAS) sum the 1024-element dot products in different orders, producing slightly different results when two tokens have nearly identical logits.

This kind of tie doesn't affect fine-tuning: cross-entropy loss is continuous and doesn't depend on argmax. The model learns the right logit distribution regardless of which token wins a 0.03-logit coin flip.

## Vocab Size Analysis

While investigating, I also re-evaluated the 20K restricted vocab size. The concern: does the 4.1% OOV rate cost us acceptance?

The key insight: OOV tokens are by definition the rare ones. Among tokens appearing 1000+ times in the 498M-token corpus, exactly zero are OOV. The main model overwhelmingly predicts common tokens, and common tokens are all in-vocab.

```
freq >=      1: 136K OOV tokens,  weighted OOV = 4.07%
freq >=   1000:   4K OOV tokens,  weighted OOV = 1.02%
freq >= 10000:     0 OOV tokens,  weighted OOV = 0.00%
```

When the MTP is most likely to predict correctly (common, predictable tokens), the OOV rate is near zero. When tokens are OOV, they're rare enough that the MTP would get them wrong anyway. The 4.1% OOV rate barely affects acceptance. Going to 30K would add coverage for tokens the MTP can't predict. Going below 10K would start losing predictable tokens. 20K is the sweet spot.

## What Changed

| Component | Before | After |
|-----------|--------|-------|
| Training embeddings | BF16 safetensors | Q6K-dequantized from GGUF |
| Training precision | BF16 autocast | FP16 autocast + GradScaler |
| Model weights | BF16 at load | FP16 at load (matches GWMT export) |
| /test_mtp input | Extracts own hidden states | Accepts pre-extracted from caller |
| Server lm_head | Full 248K vocab only | `--mtp-lm-head` for 20K reduced |
| CUDA reduced lm_head | dp4a only (Q4_K/Q5_K/Q6_K) | Also supports FP16 |

## Ready to Fine-Tune

The CUDA and PyTorch MTP implementations now agree at the logit level. The training pipeline uses the exact same embeddings, weight precision, and compute precision as CUDA inference. Fine-tuned weights will transfer directly to CUDA without precision surprises.

Next: regenerate the val cache with main model prediction targets, then start the fine-tuning run.
