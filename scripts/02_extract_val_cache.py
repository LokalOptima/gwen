#!/usr/bin/env python3
"""
Extract validation cache from dev_server for MTP training.

Calls dev_server's /batch_logits endpoint (with optional p_idk) to extract
hidden states for validation batches. Saves .pt files that run_eval() loads
during training — eliminates live server calls during evaluation.

Usage:
    # v3 format (hidden only):
    uv run scripts/02_extract_val_cache.py \
        --data data/train_tokens.bin \
        --out-dir train/runs/mtp_v3_k4096/val_cache

    # v4 format (hidden + p_idk):
    uv run scripts/02_extract_val_cache.py \
        --data data/train_tokens.bin \
        --out-dir train/runs/mtp_v4_idk_k4096/val_cache \
        --p-idk
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent dirs so we can import from train/
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from dataset import RestrictedVocab, TokenBatchSampler, TokenSequenceDataset, make_splits, mtp_collate
from gwen_client import GwenClient


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract validation cache from dev_server")
    parser.add_argument("--data", type=Path, default=Path("data/train_tokens.bin"),
                        help="Tokenized training data")
    parser.add_argument("--counts", type=Path, default=Path("data/token_counts.bin"),
                        help="Token frequency counts")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8090",
                        help="Dev server URL")
    parser.add_argument("--top-k", type=int, default=4096,
                        help="Restricted vocab size")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for val cache .pt files")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--p-idk", action="store_true", default=False,
                        help="Extract p_idk alongside hidden states (v4 format)")
    args = parser.parse_args()

    # Build dataset + split
    log(f"Building restricted vocab (K={args.top_k})...")
    vocab = RestrictedVocab(args.counts, args.top_k)
    log(f"  Coverage: {vocab.coverage:.2%}")

    log(f"Loading data from {args.data}...")
    _, val_ds = make_splits(
        args.data, vocab, seq_len=args.seq_len, val_fraction=args.val_fraction,
    )
    log(f"  Val: {len(val_ds)} sequences")

    val_sampler = TokenBatchSampler(
        val_ds.lengths, max_tokens=args.max_tokens, shuffle=False, drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=mtp_collate,
        num_workers=4, pin_memory=True,
    )
    log(f"  Val batches: {len(val_sampler)}")

    # Connect to server
    host, port = args.server_url.replace("http://", "").split(":")
    gwen = GwenClient(host, int(port))

    # Extract
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_batches = 0
    t0 = time.time()

    for bi, batch in enumerate(val_dl):
        token_ids = batch["token_ids"]  # [B, L]
        B, L = token_ids.shape

        if args.p_idk:
            hidden, logits, p_idk = gwen.batch_logits_with_p_idk(token_ids)
            # Save v4 format: hidden_input (positions 0..L-2) + p_idk
            # hidden_input = hidden[:, :L-1]  — all positions usable as MTP input
            # But val eval expects hidden_input at positions 0..L-2 (for MTP)
            # and teacher hidden at 1..L-1 (for teacher logits recomputation)
            # Save full hidden[:, :L-1] as in v3, plus p_idk
            cache = {
                "token_ids": token_ids,
                "hidden_input": hidden[:, :L-1],  # [B, L-1, 1024] — positions 0..L-2
                "p_idk": p_idk[:, 1:L-1],         # [B, L-2] — teacher positions
            }
        else:
            hidden, logits = gwen.batch_logits(token_ids)
            cache = {
                "token_ids": token_ids,
                "hidden_input": hidden[:, :L-1],  # [B, L-1, 1024]
            }

        torch.save(cache, str(out_dir / f"val_{bi:04d}.pt"))
        total_tokens += B * L
        total_batches += 1

        if (bi + 1) % 10 == 0 or bi == 0:
            elapsed = time.time() - t0
            rate = total_tokens / elapsed if elapsed > 0 else 0
            log(f"  [{bi+1}/{len(val_sampler)}] {total_tokens:,} tokens, "
                f"{rate:,.0f} tok/s")

    elapsed = time.time() - t0
    size_mb = sum(f.stat().st_size for f in out_dir.glob("*.pt")) / 1024 / 1024
    log(f"\nDone: {total_batches} batches, {total_tokens:,} tokens in {elapsed:.1f}s")
    log(f"  Output: {out_dir} ({size_mb:.1f} MB)")
    log(f"  Format: {'v4 (hidden + p_idk)' if args.p_idk else 'v3 (hidden only)'}")


if __name__ == "__main__":
    main()
