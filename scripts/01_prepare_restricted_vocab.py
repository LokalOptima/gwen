#!/usr/bin/env python3
"""
Prepare restricted vocabulary binary from training data token counts.

Computes token frequency counts from train_tokens.bin (if needed), then
extracts the top-K most frequent token IDs and writes them as a binary file.

Usage:
    uv run scripts/01_prepare_restricted_vocab.py [--top-k 4096]
    uv run scripts/01_prepare_restricted_vocab.py --top-k 8192
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SENTINEL = 0xFFFFFFFF
VOCAB_SIZE = 248320


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def compute_counts(tokens_path: Path, counts_path: Path) -> np.ndarray:
    """Compute token frequency counts from train_tokens.bin."""
    log(f"Computing token counts from {tokens_path}...")
    tokens = np.fromfile(str(tokens_path), dtype=np.uint32)
    log(f"  Total entries: {len(tokens):,}")

    valid = tokens[tokens != SENTINEL]
    log(f"  Valid tokens (excl sentinels): {len(valid):,}")

    counts = np.bincount(valid.astype(np.int64), minlength=VOCAB_SIZE)
    counts = counts[:VOCAB_SIZE].astype(np.int64)
    log(f"  Unique tokens: {(counts > 0).sum():,}")

    counts.tofile(str(counts_path))
    log(f"  Saved {counts_path} ({counts.nbytes / 1e6:.1f} MB)")
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare restricted vocabulary binary from token frequency counts")
    parser.add_argument("--top-k", type=int, default=4096,
                        help="Number of top tokens to include (default: 4096)")
    parser.add_argument("--tokens", type=Path, default=Path("data/train_tokens.bin"),
                        help="Tokenized training data")
    parser.add_argument("--counts", type=Path, default=Path("data/token_counts.bin"),
                        help="Token frequency counts (computed if missing)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: data/restricted_vocab_{K}.bin)")
    args = parser.parse_args()

    K = args.top_k
    output = args.output or Path(f"data/restricted_vocab_{K}.bin")

    # Step 1: compute counts if missing
    if not args.counts.exists():
        if not args.tokens.exists():
            log(f"ERROR: Neither {args.counts} nor {args.tokens} exist.")
            log(f"  Run scripts/00_prepare_training_data.py first.")
            sys.exit(1)
        counts = compute_counts(args.tokens, args.counts)
    else:
        log(f"Loading counts from {args.counts}...")
        counts = np.fromfile(str(args.counts), dtype=np.int64)
        if len(counts) < VOCAB_SIZE:
            counts = np.pad(counts, (0, VOCAB_SIZE - len(counts)))

    # Step 2: sort by frequency, take top K
    sorted_ids = np.argsort(-counts)
    top_k_ids = sorted_ids[:K].astype(np.int32)

    # Step 3: compute coverage
    total = counts.sum()
    top_k_counts = counts[top_k_ids].sum()
    coverage = top_k_counts / total if total > 0 else 0

    # Step 4: write binary
    output.parent.mkdir(parents=True, exist_ok=True)
    top_k_ids.tofile(str(output))

    log(f"\nRestricted vocab: K={K}")
    log(f"  Coverage: {coverage:.1%} of training tokens")
    log(f"  Top 5 IDs: {top_k_ids[:5].tolist()}")
    log(f"  Output: {output} ({top_k_ids.nbytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
