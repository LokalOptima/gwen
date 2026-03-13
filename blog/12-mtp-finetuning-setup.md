# Fine-Tuning the MTP Head for Spoken English

*Blog post #12 in the GWEN series — if 55% acceptance isn't enough, train a better draft model*

## The Plan

[Post #11](11-activation-replay-mtp-verdict.md) concluded that activation replay solved the reject-path engineering problem, but the fundamental ~55% acceptance rate keeps MTP speculative decoding below the ~65% break-even threshold. The MTP head, pre-trained on Qwen's general multi-language corpus, struggles with English-specific patterns.

My use case is specific: the model serves as a thin correction layer on speech-to-text output. That means:

- **English only** — no CJK, no code, no math
- **Spoken language patterns** — conversational, subtitles, transcripts
- **Latency-critical** — every millisecond matters in a real-time STT pipeline

If we fine-tune the MTP head on spoken English and restrict the output vocabulary to the most common English tokens, we should be able to push acceptance rates significantly higher.

## Restricted Vocabulary

The full Qwen3.5 vocabulary has 248,320 tokens. Most of the tail is CJK characters, rare Unicode, and tokens that never appear in English text. The LM head GEMV — mapping 1024 hidden dims to 248K logits — is the biggest single operation in the MTP forward pass.

From the [frequency analysis in post #10](10-token-frequency-distribution.md), we already know the distribution is heavily skewed. The question is: how much vocabulary can we cut while maintaining coverage on spoken English?

I ran the analysis on a freshly-prepared spoken-English-heavy corpus (details below). Key results for different K values:

| K | Token Coverage | Pair Coverage | Avg Run | Notes |
|---|---------------|---------------|---------|-------|
| 10,000 | 91.7% | 84.7% | 12.9 | Aggressive — 1 in 12 tokens OOV |
| **20,000** | **95.9%** | **92.3%** | **25.8** | Sweet spot — 26 tokens between OOV hits |
| 30,000 | 97.9% | 96.0% | 48.7 | Diminishing returns vs GEMV cost |
| 50,000 | 99.5% | 99.1% | 190.5 | Nearly full coverage, LM head still 5x smaller |

K=20,000 is the sweet spot: 95.9% of tokens are in-vocab, and the average run of consecutive in-vocab tokens before hitting an OOV is 25.8 — meaning ~26 successful speculation opportunities per interruption. Coverage is higher than the original 1.6B general corpus analysis because spoken English has a much narrower token distribution.

The GEMV shrinks from [1024 × 248K] to [1024 × 20K] — **12.4x smaller**. At inference time, if the ground truth next-next token is OOV, MTP automatically rejects (no computation wasted on impossible predictions).

### Throughput Projections

With K=20K (95.9% coverage) and different acceptance rates within the restricted vocab:

| Accuracy (α) | Effective Accept | Tokens/sec | Speedup |
|-------------|-----------------|------------|---------|
| 60% | 57.6% | 816 | 1.36x |
| 70% | 67.2% | 867 | 1.45x |
| **75%** | **71.9%** | **893** | **1.49x** |
| 80% | 76.7% | 919 | 1.53x |
| 90% | 86.3% | 971 | 1.62x |

The pre-trained MTP head achieves ~55% on general text. If fine-tuning on spoken English pushes within-vocab accuracy from 55% to 70-75%, we cross the break-even threshold and reach **~870-890 tok/s** — a meaningful 45-49% speedup over baseline.

## Training Data: 498M Tokens of Spoken English

I assembled a 498M-token corpus weighted toward spoken English. The data is tokenized with Qwen3.5's tokenizer and stored as flat uint32 arrays with sentinel separators between documents.

### Sources

| Source | Tokens | Share | Description |
|--------|--------|-------|-------------|
| Subscene subtitles | 300M | 60% | English movie/TV subtitles from [refine-ai/subscene](https://huggingface.co/datasets/refine-ai/subscene). 150K documents from 12 compressed JSON files. Heavily cleaned (see below). |
| OpenWebText | 100M | 20% | Open reproduction of GPT-2's WebText. Clean English web text — articles, blog posts, news. Streamed from [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext). |
| FineWeb-Edu | 90M | 18% | Educational English web text from [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) (sample-10BT config). Filtered for educational quality. |
| LibriSpeech LM | 5M | 1% | Text-only language model training data from LibriSpeech. 132K sentences from the [Kaggle archive](https://www.kaggle.com/datasets/saztorralba/librispeechquicklm). |
| LibriTTS | 3M | 1% | Clean speech transcripts. 134K sentences from LibriTTS-R clean subset. |

Total: **498,462,876 tokens** (1.9 GB on disk).

### Subscene Data Cleaning

Raw subtitle data is noisy. The cleaning pipeline removes:

- **Attribution lines**: `Subtitled by...`, `Synced & corrected by...`, `Downloaded from...`, etc.
- **Stage directions**: `[music playing]`, `[door slams]`, `(sighs)`, etc.
- **HTML artifacts**: `<i>`, `<b>`, `<font color=...>`, etc.
- **URLs**: Any line containing `http://`, `www.`, `.com`, `.org`
- **Music markers**: Lines starting/ending with `♪` or `#`
- **Non-English content**: Lines with excessive non-ASCII (>30%)
- **Short/empty lines**: After cleaning, lines under 2 characters are dropped

Of 150K subtitle files processed, 322 were skipped entirely (corrupt or empty after cleaning).

### Data Locations

```
data/
├── train_tokens.bin      # 1.9 GB — concatenated, sentinel-separated
├── train_subscene.bin    # 1.1 GB — subscene only (kept for reprocessing)
├── train_speech.bin      #  32 MB — LibriSpeech + LibriTTS
├── train_openwebtext.bin # 382 MB — OpenWebText
├── train_fineweb.bin     # 345 MB — FineWeb-Edu
├── token_counts.bin      # 1.9 MB — frequency counts from 498M training corpus
└── token_freqs.csv       #         — human-readable frequency ranking
```

Part files are kept so we can adjust the mix or reprocess individual sources without re-downloading.

### Processing Script

`scripts/prepare_training_data.py` handles everything:

```bash
uv run --with datasets --with gguf --with numpy \
    scripts/prepare_training_data.py --target-tokens 500000000
```

All four corpus sources are processed in parallel via `ThreadPoolExecutor`. Each corpus writes to its own part file (memory-bounded via a `TokenWriter` that flushes every 1M tokens), then part files are concatenated. Progress goes to stderr with `flush=True` so you see updates even when piped. Total processing time: ~25 minutes (subscene is the bottleneck).

## MTP Head Architecture

The MTP head matches the pre-trained Qwen3.5-0.8B MTP architecture exactly:

```
Input:  concat(RMSNorm(embed[t+1]), RMSNorm(hidden[t])) → [2048]
        ↓
FC:     Linear(2048 → 1024, no bias)
        ↓
Layer:  RMSNorm → GQA Attention (8Q+gate, 2KV, head_dim=256, QK-Norm) → residual
        RMSNorm → SwiGLU FFN (1024 → 3584 → 1024) → residual
        ↓
Output: RMSNorm → Linear(1024 → K)
```

The attention uses Qwen3.5's output gating: Q projection is 4096-dim (2048 for queries + 2048 for sigmoid gate), and the gate is applied elementwise to the attention output before the output projection.

### Parameters

| Component | Params | Notes |
|-----------|--------|-------|
| Input norms | 2K | 2 × RMSNorm(1024) |
| FC projection | 2.1M | 2048 → 1024 |
| GQA attention | 7.3M | Q+gate, K, V, O projections + QK norms |
| SwiGLU FFN | 11.0M | gate + up + down |
| Output norm | 1K | RMSNorm(1024) |
| LM head (K=20K) | 20.5M | 1024 → 20000 (new, randomly initialized) |
| **Total** | **40.9M** | 20.4M pre-trained + 20.5M new |

We initialize from the pre-trained MTP weights (15 tensors from safetensors) — everything except the LM head, which is randomly initialized for the restricted K=20K vocabulary. This gives us a strong starting point: the pre-trained layers already know how to combine embeddings and hidden states; we just need to learn the new output mapping.

## Training Setup

Online training: each batch runs a frozen BF16 forward pass through the full 0.8B model to get hidden states, then trains the MTP head on those. No pre-extraction needed — the base model forward is fast enough in PyTorch (batch=32, seq=512 processes ~16K tokens in a few ms).

```
train/
├── train_mtp.py   # Main script (train / export subcommands)
├── model.py       # MTP head matching pre-trained architecture
└── dataset.py     # Token dataset + restricted vocab mapping
```

Configuration:
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95))
- **Schedule**: 5% linear warmup → cosine annealing
- **Precision**: BF16 autocast (no GradScaler — BF16 has same exponent range as FP32)
- **Gradient clipping**: max norm 1.0
- **Batching**: Token-budget sampler (`--max-tokens 32768`) — sorts by length, packs similar-length sequences, pads only to max-in-batch
- **Validation**: 5% held-out split
- **Checkpoints**: best.pt (by val loss) + latest.pt, with optimizer/scheduler state for clean resume
- **Reproducibility**: `train_setup.json` records exact command, SHA256 checksums of data files, all hyperparameters

Training tracks OOV fraction and estimated acceptance rate (accuracy on non-masked targets) per epoch. The export subcommand converts the trained checkpoint back to GWMT binary format for GWEN inference.

## What's Next

With 498M tokens prepared, the restricted vocab analysis done, and the training infrastructure built, the next step is actually running training and measuring whether fine-tuning pushes acceptance above the 65% break-even threshold.

The key question: can a 21M-parameter head, initialized from pre-trained weights and fine-tuned on spoken English with a 20K restricted vocab, achieve 70%+ acceptance rate on STT-style text? The frequency analysis suggests the ceiling is there (95.9% coverage, 18-token average runs). It comes down to whether the model can learn the patterns.
