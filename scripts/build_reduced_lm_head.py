#!/usr/bin/env python3
"""
Build a reduced LM head for MTP speculative decoding.

Extracts the top-K most common token rows from the Q6_K token_embd tensor
in the GGUF file. This reduces the MTP LM head GEMV from 248K rows to K rows,
cutting bandwidth proportionally.

Output format (GWRL):
    Magic: b"GWRL" (4 bytes)
    Version: 1 (uint32 LE)
    K: uint32 LE (number of tokens in reduced set)
    n_embed: uint32 LE (embedding dimension, 1024)
    ggml_type: uint32 LE (14 = Q6_K)
    row_bytes: uint32 LE (bytes per Q6_K row, 840)
    token_ids: int32[K] (mapping: index → real token ID)
    weights: uint8[K × row_bytes] (Q6_K weight data, row-major)

Usage:
    uv run --with gguf --with numpy scripts/build_reduced_lm_head.py
    uv run --with gguf --with numpy scripts/build_reduced_lm_head.py --top-k 50000
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def load_token_counts(counts_path: Path, n_vocab: int) -> np.ndarray:
    """Load pre-computed token counts from binary file (int64[n_vocab])."""
    counts = np.fromfile(str(counts_path), dtype=np.int64)
    if len(counts) != n_vocab:
        print(f"Warning: counts file has {len(counts)} entries, expected {n_vocab}", file=sys.stderr)
        if len(counts) > n_vocab:
            counts = counts[:n_vocab]
        else:
            padded = np.zeros(n_vocab, dtype=np.int64)
            padded[:len(counts)] = counts
            counts = padded
    return counts


def main():
    parser = argparse.ArgumentParser(description="Build reduced LM head for faster MTP")
    parser.add_argument("--gguf", type=Path,
                        default=Path.home() / "models" / "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                        help="Path to GGUF model file")
    parser.add_argument("--counts", type=Path,
                        default=Path("data/token_counts.bin"),
                        help="Path to pre-computed token counts (int64[n_vocab])")
    parser.add_argument("--top-k", type=int, default=50000,
                        help="Number of most common tokens to include (default: 50000)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (default: ~/models/gguf/lm_head_top{K}.bin)")
    args = parser.parse_args()

    gguf_path = args.gguf.expanduser().resolve()
    counts_path = args.counts.expanduser().resolve()
    K = args.top_k
    output = args.output or Path.home() / "models" / "gguf" / f"lm_head_top{K}.bin"
    output = output.expanduser().resolve()

    print(f"GGUF: {gguf_path}")
    print(f"Counts: {counts_path}")
    print(f"Top-K: {K}")
    print(f"Output: {output}")
    print()

    # Read GGUF token_embd tensor
    print("Reading GGUF file...")
    reader = GGUFReader(str(gguf_path))
    embd_tensor = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            embd_tensor = t
            break

    if embd_tensor is None:
        print("Error: token_embd.weight not found in GGUF", file=sys.stderr)
        sys.exit(1)

    n_vocab = embd_tensor.data.shape[0]
    row_bytes = embd_tensor.data.shape[1]
    n_embed = 1024  # Known from model config
    ggml_type = int(embd_tensor.tensor_type)
    print(f"token_embd: {n_vocab} rows × {row_bytes} bytes/row, type={embd_tensor.tensor_type.name}")
    print(f"Total: {embd_tensor.data.nbytes / 1024 / 1024:.1f} MB")
    print()

    if K > n_vocab:
        K = n_vocab
        print(f"Warning: K capped to vocab size {n_vocab}")

    # Load pre-computed token frequencies
    print(f"Loading token counts from {counts_path}...")
    counts = load_token_counts(counts_path, n_vocab)
    used = np.sum(counts > 0)
    total = np.sum(counts)
    print(f"  Total tokens: {total:,}")
    print(f"  Unique tokens: {used:,} / {n_vocab:,}")
    print()

    # Select top-K tokens by frequency
    sorted_indices = np.argsort(-counts)
    top_k_ids = sorted_indices[:K].astype(np.int32)

    # Compute coverage
    top_k_counts = counts[top_k_ids]
    coverage = np.sum(top_k_counts) / total * 100
    print(f"Top-{K} coverage: {coverage:.2f}% of corpus tokens")

    # Sort token IDs for cache-friendly access (optional, helps GPU L2 hits)
    top_k_ids_sorted = np.sort(top_k_ids)

    # Extract Q6_K rows
    print(f"Extracting {K} Q6_K rows...")
    extracted_data = embd_tensor.data[top_k_ids_sorted]
    data_mb = extracted_data.nbytes / 1024 / 1024
    est_us = data_mb / 896 * 1000
    print(f"  Extracted: {data_mb:.1f} MB (est. {est_us:.0f} μs at 896 GB/s)")

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(b"GWRL")                        # Magic
        f.write(struct.pack("<I", 1))            # Version
        f.write(struct.pack("<I", K))            # K
        f.write(struct.pack("<I", n_embed))      # n_embed
        f.write(struct.pack("<I", ggml_type))    # ggml_type (14 = Q6_K)
        f.write(struct.pack("<I", row_bytes))    # row_bytes
        f.write(top_k_ids_sorted.tobytes())      # token_ids: int32[K]
        f.write(extracted_data.tobytes())         # weights: uint8[K × row_bytes]

    file_size = output.stat().st_size
    print(f"\nSaved: {output}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  Token IDs: {K} × 4 bytes = {K * 4 / 1024:.1f} KB")
    print(f"  Weight data: {data_mb:.1f} MB")
    print(f"  Reduction: {n_vocab}→{K} rows = {K/n_vocab*100:.1f}% of original")

    # Show some stats
    print(f"\nCoverage summary:")
    for k_test in [1000, 5000, 10000, 20000, 30000, K, 80000, 100000]:
        if k_test > n_vocab:
            break
        cov = np.sum(counts[sorted_indices[:k_test]]) / total * 100
        sz = k_test * row_bytes / 1024 / 1024
        us = sz / 896 * 1000
        marker = " <<<" if k_test == K else ""
        print(f"  Top {k_test:>6,}: {cov:6.2f}% coverage, {sz:5.1f} MB, ~{us:4.0f} μs{marker}")


if __name__ == "__main__":
    main()
