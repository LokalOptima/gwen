#!/usr/bin/env python3
"""
SVD factorization of the LM head (token embedding) matrix for faster MTP.

Decomposes W (248320×1024) into A (248320×r) × B (r×1024).
Reduces MTP LM head GEMV from ~200 MB → ~63 MB at r=128 (3× less bandwidth).

Reads original BF16 weights from HuggingFace safetensors (pre-quantization).

Output format (GWLR):
    Magic: b"GWLR" (4 bytes)
    Version: 1 (uint32 LE)
    rank: r (uint32 LE)
    n_vocab: uint32 LE
    n_embed: uint32 LE
    B data: r × n_embed × 2 bytes (FP16, row-major)
    A data: n_vocab × r × 2 bytes (FP16, row-major)

Usage:
    uv run --with safetensors --with numpy --with ml-dtypes scripts/svd_lm_head.py
    uv run --with safetensors --with numpy --with ml-dtypes scripts/svd_lm_head.py --rank 256
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy
from safetensors import safe_open


def find_safetensors_files(model_dir: Path) -> list[Path]:
    """Find safetensors files, handling both single and sharded models."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        return [model_dir / f for f in shard_files]

    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        print(f"Error: No safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)
    return files


def load_embed_weight(model_dir: Path) -> np.ndarray:
    """Load token embedding weight from safetensors as float32."""
    # Try both naming conventions (with and without language_model prefix)
    candidates = ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"]
    for path in find_safetensors_files(model_dir):
        with safe_open(str(path), framework="numpy") as f:
            for name in candidates:
                if name in f.keys():
                    tensor = f.get_tensor(name)
                    print(f"Loaded {name}: shape={tensor.shape}, dtype={tensor.dtype}")
                    return tensor.astype(np.float32)

    print("Error: embed_tokens.weight not found in safetensors", file=sys.stderr)
    sys.exit(1)


def analyze_spectrum(S: np.ndarray, W_norm: float, n_vocab: int):
    """Print singular value spectrum analysis."""
    print(f"\nSingular value spectrum:")
    print(f"  σ_0 = {S[0]:.2f} (largest)")
    print(f"  σ_max = {S[-1]:.6f} (smallest, rank {len(S)})")
    print(f"  ||W||_F = {W_norm:.2f}")
    print()

    total_energy = np.sum(S**2)
    ranks = [32, 64, 96, 128, 192, 256, 384, 512, 768, min(1024, len(S))]

    print(f"  {'Rank':<6} {'Rel Error':>10} {'Energy':>10} {'A size':>10} {'Est. μs':>10}")
    print(f"  {'----':<6} {'---------':>10} {'------':>10} {'------':>10} {'-------':>10}")

    for r in ranks:
        if r > len(S):
            break
        retained = np.sum(S[:r]**2)
        dropped = total_energy - retained
        rel_error = np.sqrt(dropped) / W_norm
        energy_pct = retained / total_energy * 100
        a_size_mb = n_vocab * r * 2 / 1024 / 1024  # FP16
        est_us = a_size_mb / 896 * 1000  # 896 GB/s bandwidth
        print(f"  {r:<6} {rel_error:>10.6f} {energy_pct:>9.4f}% {a_size_mb:>8.1f}MB {est_us:>8.0f} μs")


def save_gwlr(path: Path, A: np.ndarray, B: np.ndarray, rank: int):
    """Save low-rank factors as GWLR binary."""
    n_vocab, r = A.shape
    _, n_embed = B.shape
    assert r == rank

    A_fp16 = A.astype(np.float16)
    B_fp16 = B.astype(np.float16)

    with open(path, "wb") as f:
        f.write(b"GWLR")                       # Magic
        f.write(struct.pack("<I", 1))           # Version
        f.write(struct.pack("<I", rank))         # rank
        f.write(struct.pack("<I", n_vocab))      # n_vocab
        f.write(struct.pack("<I", n_embed))      # n_embed
        f.write(B_fp16.tobytes())               # B: (r × n_embed) — small, first
        f.write(A_fp16.tobytes())               # A: (n_vocab × r) — large, second

    file_size = path.stat().st_size
    print(f"\nSaved: {path}")
    print(f"  Total size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  B: {r}×{n_embed} FP16 = {r * n_embed * 2 / 1024:.1f} KB")
    print(f"  A: {n_vocab}×{r} FP16 = {n_vocab * r * 2 / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="SVD factorization of LM head for faster MTP")
    parser.add_argument("--model-dir", type=Path,
                        default=Path.home() / "models" / "hf" / "Qwen3.5-0.8B")
    parser.add_argument("--rank", type=int, default=128,
                        help="Target rank for low-rank approximation (default: 128)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (default: ~/models/gguf/lm_head_r{rank}.bin)")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    rank = args.rank
    output = args.output or Path.home() / "models" / "gguf" / f"lm_head_r{rank}.bin"
    output = output.expanduser().resolve()

    if not model_dir.is_dir():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Model dir: {model_dir}")
    print(f"Target rank: {rank}")
    print(f"Output: {output}")

    # Load original BF16 weights
    W = load_embed_weight(model_dir)
    n_vocab, n_embed = W.shape
    W_norm = np.linalg.norm(W, "fro")
    print(f"Matrix: {n_vocab}×{n_embed}, ||W||_F = {W_norm:.2f}")

    # Full SVD (truncated output: U is n_vocab×n_embed, not n_vocab×n_vocab)
    print(f"\nComputing SVD ({n_vocab}×{n_embed})...")
    t0 = time.time()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    t1 = time.time()
    print(f"SVD completed in {t1 - t0:.1f}s")
    print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

    # Spectrum analysis
    analyze_spectrum(S, W_norm, n_vocab)

    # Construct low-rank factors: W ≈ A @ B where A = U_r * S_r, B = Vt_r
    A = U[:, :rank] * S[:rank]   # (n_vocab, r)
    B = Vt[:rank, :]             # (r, n_embed)

    # Reconstruction quality
    W_approx = A @ B
    error = np.linalg.norm(W - W_approx, "fro") / W_norm
    print(f"\nReconstruction error at rank {rank}: {error:.6f}")

    # Top-1 agreement with random hidden states
    rng = np.random.default_rng(42)
    n_test = 1000
    hidden = rng.standard_normal((n_test, n_embed)).astype(np.float32)
    logits_full = hidden @ W.T
    logits_approx = (hidden @ B.T) @ A.T
    top1_full = np.argmax(logits_full, axis=1)
    top1_approx = np.argmax(logits_approx, axis=1)
    agreement = np.mean(top1_full == top1_approx) * 100
    print(f"Top-1 agreement (random hidden, n={n_test}): {agreement:.1f}%")

    # Top-5 overlap
    top5_full = np.argsort(logits_full, axis=1)[:, -5:]
    top5_approx = np.argsort(logits_approx, axis=1)[:, -5:]
    overlap = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(top5_full, top5_approx)]) * 100
    print(f"Top-5 overlap  (random hidden, n={n_test}): {overlap:.1f}%")

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    save_gwlr(output, A, B, rank)


if __name__ == "__main__":
    main()
