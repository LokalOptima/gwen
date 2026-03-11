# Token Frequency Distribution: Understanding What Qwen3.5 Actually Sees

*Blog post #10 in the GWEN series — building a frequency map of the vocabulary*

## Motivation

The LM head is the single most expensive operation in decode: a 248,320 × 1024 GEMV that dominates every token generation step. Not all of those 248K output rows matter equally — in practice, English text concentrates heavily on a small fraction of the vocabulary. Knowing *which* tokens matter (and how much) opens the door to optimizations: reduced LM heads, approximate top-k, sorted-by-frequency memory layouts, or speculative verification shortcuts.

To build that frequency map, I tokenized **1.625 billion tokens** of English text across four diverse corpora using Qwen3.5-0.8B's tokenizer.

## Corpora

I wanted breadth across domains — encyclopedic, web, educational — without needing authentication tokens or custom downloaders. All four datasets stream from HuggingFace:

| Corpus | Source | Tokens | Character |
|--------|--------|--------|-----------|
| WikiText-103 | Wikipedia excerpts | 124M | Encyclopedic, formal |
| OpenWebText | Reddit-filtered web pages | 500M | Diverse web text, informal |
| English Wikipedia | Full English Wikipedia | 500M | Encyclopedic, structured |
| FineWeb-Edu | Curated educational web text | 500M | Academic, explanatory |
| **Total** | | **1.625B** | |

Tokenization runs at ~2.3M tok/s using HuggingFace's `AutoTokenizer` for `Qwen/Qwen3.5-0.8B`. I verified the HF tokenizer matches the GGUF vocabulary: 4,994/5,000 random tokens match exactly (the 6 mismatches are in the 276 added special tokens that exist in GGUF but not HF's base vocab).

## The Distribution

### Coverage

```
Vocab size:        248,320
Tokens observed:   191,855 (77.3%)
Tokens unseen:      56,465 (22.7%)
```

Nearly a quarter of the vocabulary never appears in 1.6B tokens of English. These are overwhelmingly CJK characters, rare scripts, and code-specific tokens. For an English-only inference engine, they're dead weight in the LM head.

### Zipf's Law in Action

The distribution is extremely heavy-tailed:

```
Top    100 tokens:  45.66%
Top    500 tokens:  58.41%
Top  1,000 tokens:  64.76%
Top  2,000 tokens:  71.78%
Top  5,000 tokens:  81.23%
Top 10,000 tokens:  87.94%
Top 20,000 tokens:  93.85%
Top 50,000 tokens:  99.26%
```

Just 175 tokens cover half of all English text. The top 50K tokens cover 99.26%. The remaining ~200K tokens share less than 1% of total probability.

### Coverage Thresholds

How many tokens do you need for a given coverage level?

| Coverage | Tokens needed | % of vocab |
|----------|---------------|------------|
| 50% | 175 | 0.07% |
| 80% | 4,432 | 1.8% |
| 90% | 12,579 | 5.1% |
| 95% | 23,311 | 9.4% |
| 99% | 46,126 | 18.6% |
| 99.9% | 79,557 | 32.0% |

### Information Theory

```
Shannon entropy:     11.077 bits
Max uniform entropy: 17.550 bits (over 191,855 observed tokens)
Efficiency:          63.1%
```

The effective alphabet is much smaller than the vocabulary suggests. 11 bits of entropy means the distribution behaves like a ~2,100-symbol uniform distribution, not a 248K one.

### Top 30 Tokens

| Rank | Token | Freq (%) | Cumulative |
|------|-------|----------|------------|
| 1 | ` the` | 3.82 | 3.82 |
| 2 | `,` | 3.44 | 7.26 |
| 3 | `.` | 2.97 | 10.23 |
| 4 | ` of` | 2.17 | 12.40 |
| 5 | ` and` | 1.90 | 14.30 |
| 6 | ` ` (space) | 1.74 | 16.04 |
| 7 | ` to` | 1.69 | 17.73 |
| 8 | ` in` | 1.42 | 19.15 |
| 9 | ` a` | 1.39 | 20.54 |
| 10 | `1` | 1.35 | 21.89 |
| 11 | `0` | 1.21 | 23.10 |
| 12 | `\n` | 1.17 | 24.27 |
| 13 | `\n\n` | 0.93 | 25.20 |
| 14 | `2` | 0.89 | 26.09 |
| 15 | ` is` | 0.72 | 26.81 |

No surprises — function words and punctuation dominate. Digits are individually frequent because the tokenizer encodes them as single characters.

## Output Files

The analysis produces three files in `data/`:

- **`token_freqs.bin`** — 248,320 × float32 normalized frequencies, directly memcpy-able into a CUDA buffer. Zero for unseen tokens.
- **`token_counts.bin`** — 248,320 × int64 raw counts from all corpora.
- **`token_freqs.csv`** — Human-readable ranked list with cumulative frequencies.

The script (`scripts/token_frequency.py`) is fully reproducible: it streams from HuggingFace with no local data needed.

## What This Enables

Having per-token frequencies opens several optimization paths:

1. **Reduced LM head**: Only compute the top-N rows of the LM head GEMV. With N=50K, we cover 99.26% of English and cut the GEMV from 248K→50K rows (5× smaller). Rare tokens can be computed on-demand when the top-N result looks uncertain.

2. **Frequency-sorted memory layout**: Reorder the embedding/LM head rows by frequency so the hottest rows are contiguous in memory. This improves L2 cache hit rates during decode — the top 10K rows (88% of accesses) fit in ~20 MB, well within the 5070 Ti's 48 MB L2.

3. **Approximate top-k**: For speculative verification, if the draft token is in the top-1K (65% of the time), we can skip full argmax over 248K logits and just verify against the high-frequency candidates.

4. **Weighted sampling shortcuts**: For temperature sampling, start with the high-frequency bucket and early-exit once cumulative probability exceeds the sample threshold.
