#!/usr/bin/env python3
"""Recompute token_counts.bin from the 498M training corpus."""
import numpy as np

SENTINEL = 0xFFFFFFFF
tokens = np.fromfile("data/train_tokens.bin", dtype=np.uint32)
print(f"Total entries: {len(tokens):,}")

valid = tokens[tokens != SENTINEL]
print(f"Valid tokens (excl sentinels): {len(valid):,}")

counts = np.bincount(valid.astype(np.int64), minlength=248320)
counts = counts[:248320].astype(np.int64)
print(f"Unique tokens: {(counts > 0).sum():,}")
print(f"Top 5 token IDs by freq: {np.argsort(-counts)[:5]}")
print(f"Sum of counts: {counts.sum():,}")

counts.tofile("data/token_counts.bin")
print(f"Saved token_counts.bin ({counts.nbytes / 1e6:.1f} MB)")
