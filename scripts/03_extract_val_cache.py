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
import queue
import sys
import threading
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# Add parent dirs so we can import from train/
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from dataset import RestrictedVocab, TokenBatchSampler, TokenSequenceDataset, make_splits, mtp_collate
from gwen_client import GwenClient


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _save_worker(q: queue.Queue):
    """Drain the write queue, saving each item atomically."""
    while True:
        item = q.get()
        if item is None:
            break
        data, tmp_path, final_path = item
        torch.save(data, str(tmp_path))
        tmp_path.rename(final_path)
        q.task_done()


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
        num_workers=0,
    )
    log(f"  Val batches: {len(val_sampler)}")

    # Connect to server
    host, port = args.server_url.replace("http://", "").split(":")
    gwen = GwenClient(host, int(port))

    # Extract
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure out which batches need extraction (for restartability + accurate ETA)
    n_total = len(val_sampler)
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob("val_*.pt")}
    to_extract = [(bi, batch) for bi, batch in enumerate(val_dl) if bi not in existing]

    if existing:
        log(f"  Skipping {len(existing)}/{n_total} already-extracted batches")

    total_tokens = 0
    total_batches = 0

    # Write queue: GPU thread pushes, background thread saves to disk
    write_q = queue.Queue()
    writer = threading.Thread(target=_save_worker, args=(write_q,), daemon=True)
    writer.start()

    pbar = tqdm(to_extract, desc="extract", unit="batch", file=sys.stderr)
    for bi, batch in pbar:
        token_ids = batch["token_ids"]  # [B, L]
        B, L = token_ids.shape

        if args.p_idk:
            hidden, p_idk = gwen.batch_hidden_with_p_idk(token_ids)
            cache = {
                "token_ids": token_ids,
                "hidden_input": hidden[:, :L-1],  # [B, L-1, 1024]
                "p_idk": p_idk[:, 1:L-1],         # [B, L-2] — teacher positions
            }
        else:
            hidden, logits = gwen.batch_logits(token_ids)
            cache = {
                "token_ids": token_ids,
                "hidden_input": hidden[:, :L-1],  # [B, L-1, 1024]
            }

        tmp_path = out_dir / f".val_{bi:04d}.pt.tmp"
        out_path = out_dir / f"val_{bi:04d}.pt"
        write_q.put((cache, tmp_path, out_path))
        total_tokens += B * L
        total_batches += 1
        pbar.set_postfix_str(f"wq={write_q.qsize()}")

    write_q.put(None)  # sentinel
    writer.join()
    size_mb = sum(f.stat().st_size for f in out_dir.glob("*.pt")) / 1024 / 1024
    log(f"\nDone: {total_batches} batches, {total_tokens:,} tokens")
    log(f"  Output: {out_dir} ({size_mb:.1f} MB)")
    log(f"  Format: {'v4 (hidden + p_idk)' if args.p_idk else 'v3 (hidden only)'}")

    # --- Verify all batches ---
    log(f"\nVerifying {n_total} batches...")
    errors = []
    for bi in trange(n_total, desc="verify", unit="batch", file=sys.stderr):
        path = out_dir / f"val_{bi:04d}.pt"
        if not path.exists():
            errors.append(f"  MISSING: {path.name}")
            continue
        try:
            vc = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as e:
            errors.append(f"  CORRUPT: {path.name}: {e}")
            continue

        if "token_ids" not in vc or "hidden_input" not in vc:
            errors.append(f"  BAD KEYS: {path.name}: {list(vc.keys())}")
            continue

        tids = vc["token_ids"]
        h = vc["hidden_input"]
        B_v, L_v = tids.shape

        if h.shape != (B_v, L_v - 1, 1024):
            errors.append(f"  BAD SHAPE: {path.name}: hidden {h.shape} vs tokens {tids.shape}")
            continue

        if args.p_idk:
            if "p_idk" not in vc:
                errors.append(f"  MISSING p_idk: {path.name}")
                continue
            pi = vc["p_idk"]
            if pi.shape != (B_v, L_v - 2):
                errors.append(f"  BAD SHAPE: {path.name}: p_idk {pi.shape} expected ({B_v}, {L_v - 2})")
                continue
            if pi.min() < 0 or pi.max() > 1:
                errors.append(f"  BAD RANGE: {path.name}: p_idk [{pi.min():.4f}, {pi.max():.4f}]")
                continue

    if errors:
        for e in errors:
            log(e)
        log(f"\nFAILED: {len(errors)} errors")
        sys.exit(1)
    else:
        log(f"  All {n_total} batches OK")


if __name__ == "__main__":
    main()
