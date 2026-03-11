#!/usr/bin/env python3
"""
Compute token frequency distribution for every token in the Qwen3.5-0.8B vocabulary
by tokenizing multiple large English corpora.

Corpora used:
  1. WikiText-103        (~124M tokens, encyclopedic)
  2. OpenWebText          (~500M tokens, diverse web text)
  3. English Wikipedia    (~500M tokens, encyclopedic)
  4. FineWeb-Edu sample   (~500M tokens, curated educational web text)

Outputs:
  data/token_freqs.bin  — 248320 float32 values (normalized frequencies), memcpy-able into CUDA
  data/token_freqs.csv  — human-readable ranked list
  data/token_counts.bin — 248320 int64 raw counts
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from pathlib import Path
import time

VOCAB_SIZE = 248320
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def tokenize_batch(tokenizer, texts):
    """Tokenize a batch of texts, return flat list of token IDs and count."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return [], 0
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    all_ids = []
    n = 0
    for ids in encoded:
        all_ids.extend(ids)
        n += len(ids)
    return all_ids, n


def process_corpus_full(name, dataset, text_field, tokenizer, counts, batch_size=1000):
    """Process a fully-loaded (non-streaming) dataset."""
    print(f"\n{'='*60}")
    print(f"Corpus: {name}")
    print(f"{'='*60}")
    texts = dataset[text_field]
    total = 0
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        ids, n = tokenize_batch(tokenizer, batch)
        counts.update(ids)
        total += n
        if (i // batch_size) % 100 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  {i:>10,}/{len(texts):,} docs | {total:>12,} tokens | {rate:,.0f} tok/s")
    elapsed = time.time() - t0
    print(f"  Done: {total:,} tokens in {elapsed:.1f}s ({total/elapsed:,.0f} tok/s)")
    return total


def process_corpus_streaming(name, dataset_iter, text_field, tokenizer, counts,
                              target_tokens, batch_size=512):
    """Process a streaming dataset, stopping after target_tokens."""
    print(f"\n{'='*60}")
    print(f"Corpus: {name} (streaming, target {target_tokens:,} tokens)")
    print(f"{'='*60}")
    total = 0
    docs = 0
    batch = []
    t0 = time.time()
    for row in dataset_iter:
        text = row[text_field]
        if not text or not text.strip():
            continue
        batch.append(text)
        docs += 1
        if len(batch) >= batch_size:
            ids, n = tokenize_batch(tokenizer, batch)
            counts.update(ids)
            total += n
            batch = []
            if docs % (batch_size * 50) == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  {docs:>10,} docs | {total:>12,} tokens | {rate:,.0f} tok/s | {len(counts)} unique")
            if total >= target_tokens:
                break
    # Final batch
    if batch:
        ids, n = tokenize_batch(tokenizer, batch)
        counts.update(ids)
        total += n
    elapsed = time.time() - t0
    print(f"  Done: {total:,} tokens from {docs:,} docs in {elapsed:.1f}s ({total/elapsed:,.0f} tok/s)")
    return total


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    assert tokenizer.vocab_size <= VOCAB_SIZE

    counts = Counter()
    grand_total = 0

    # --- Corpus 1: WikiText-103 (small, fast, fully loaded) ---
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    grand_total += process_corpus_full("WikiText-103", ds, "text", tokenizer, counts)
    del ds

    # --- Corpus 2: OpenWebText (streaming, ~500M tokens) ---
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    grand_total += process_corpus_streaming("OpenWebText", ds, "text", tokenizer, counts,
                                             target_tokens=500_000_000)
    del ds

    # --- Corpus 3: English Wikipedia (streaming, ~500M tokens) ---
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    grand_total += process_corpus_streaming("English Wikipedia", ds, "text", tokenizer, counts,
                                             target_tokens=500_000_000)
    del ds

    # --- Corpus 4: FineWeb-Edu sample (streaming, ~500M tokens) ---
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    grand_total += process_corpus_streaming("FineWeb-Edu (sample-10BT)", ds, "text", tokenizer, counts,
                                             target_tokens=500_000_000)
    del ds

    # === Build frequency distribution ===
    print(f"\n{'='*60}")
    print(f"TOTAL: {grand_total:,} tokens, {len(counts)} unique token IDs")
    print(f"{'='*60}")

    # Raw counts array
    raw_counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for token_id, count in counts.items():
        if token_id < VOCAB_SIZE:
            raw_counts[token_id] = count

    # Normalized frequency
    freqs = raw_counts.astype(np.float64)
    freqs /= freqs.sum()
    freqs = freqs.astype(np.float32)

    # Save binary files
    bin_path = OUTPUT_DIR / "token_freqs.bin"
    freqs.tofile(bin_path)
    print(f"Saved: {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    counts_path = OUTPUT_DIR / "token_counts.bin"
    raw_counts.tofile(counts_path)
    print(f"Saved: {counts_path} ({counts_path.stat().st_size / 1024:.1f} KB)")

    # Save CSV
    csv_path = OUTPUT_DIR / "token_freqs.csv"
    sorted_ids = np.argsort(-freqs)
    with open(csv_path, "w") as f:
        f.write("rank,token_id,frequency,count,cumulative_freq,token_text\n")
        cum = 0.0
        for rank, tid in enumerate(sorted_ids):
            if freqs[tid] == 0:
                break
            cum += freqs[tid]
            token_text = tokenizer.decode([int(tid)]).replace(",", "<comma>").replace("\n", "\\n")
            f.write(f"{rank+1},{tid},{freqs[tid]:.10e},{raw_counts[tid]},{cum:.8f},{token_text}\n")
    nonzero_count = rank
    print(f"Saved: {csv_path} ({nonzero_count} tokens with non-zero frequency)")

    # === Statistics ===
    print(f"\n{'='*60}")
    print("TOKEN FREQUENCY DISTRIBUTION")
    print(f"{'='*60}")

    nonzero = (freqs > 0).sum()
    print(f"Vocab size:        {VOCAB_SIZE:,}")
    print(f"Tokens observed:   {nonzero:,} ({100*nonzero/VOCAB_SIZE:.1f}%)")
    print(f"Tokens unseen:     {VOCAB_SIZE - nonzero:,} ({100*(VOCAB_SIZE - nonzero)/VOCAB_SIZE:.1f}%)")
    print(f"Total token count: {grand_total:,}")

    # Cumulative coverage at various thresholds
    print(f"\nCumulative coverage:")
    for k in [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        cov = freqs[sorted_ids[:k]].sum()
        print(f"  Top {k:>6,} tokens: {100*cov:>6.2f}%")

    # Entropy
    p = freqs[freqs > 0].astype(np.float64)
    entropy = -np.sum(p * np.log2(p))
    print(f"\nShannon entropy: {entropy:.3f} bits")
    print(f"Max possible entropy (uniform over {nonzero}): {np.log2(nonzero):.3f} bits")
    print(f"Efficiency: {entropy / np.log2(nonzero) * 100:.1f}%")

    # Percentile thresholds
    print(f"\nTokens needed for coverage thresholds:")
    cum = np.cumsum(freqs[sorted_ids])
    for threshold in [0.50, 0.80, 0.90, 0.95, 0.99, 0.999]:
        idx = np.searchsorted(cum, threshold) + 1
        print(f"  {threshold*100:>5.1f}% coverage: {idx:>6,} tokens")

    # Top 30
    print(f"\nTop 30 tokens:")
    print(f"{'Rank':>5} {'ID':>8} {'Freq':>12} {'Count':>14} {'CumFreq':>10}  Token")
    cum_f = 0.0
    for rank, tid in enumerate(sorted_ids[:30]):
        cum_f += freqs[tid]
        token_text = repr(tokenizer.decode([int(tid)]))
        print(f"{rank+1:>5} {tid:>8} {freqs[tid]:>12.6f} {raw_counts[tid]:>14,} {cum_f:>10.4f}  {token_text}")


if __name__ == "__main__":
    main()
