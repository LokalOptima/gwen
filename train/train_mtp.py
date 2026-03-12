#!/usr/bin/env python3
"""
Fine-tune the MTP head for restricted-vocab English speculative decoding.

Freezes the main Qwen3.5-0.8B model and trains only the 21M-param MTP head
to predict token[t+2] from embed[t+1] + hidden[t], over a restricted vocab
of the top-K most frequent English tokens.

Online training: each batch runs a frozen forward pass through the base model
to get hidden states, then trains the MTP head on those.

Usage:
    uv run --with 'torch>=2.5' --with safetensors --with transformers --with numpy \
        train/train_mtp.py train \
        --data data/train_tokens.bin \
        --counts data/token_counts.bin \
        --model-dir ~/models/hf/Qwen3.5-0.8B \
        --out-dir train/runs/mtp_v1 \
        --top-k 20000 \
        --epochs 5

    uv run --with 'torch>=2.5' --with safetensors --with numpy \
        train/train_mtp.py export \
        --checkpoint train/runs/mtp_v1/best.pt \
        --output ~/models/gguf/mtp_finetuned.bin
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent to path so we can import from train/
sys.path.insert(0, str(Path(__file__).parent))
from dataset import RestrictedVocab, TokenSequenceDataset, make_splits
from model import MTPHead


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Base model loading
# ---------------------------------------------------------------------------

def load_base_model(model_dir: Path, device: str) -> tuple:
    """Load frozen Qwen3.5-0.8B and return (embed_tokens, forward_fn).

    Returns:
        embed_fn: callable(input_ids) → [B, L, 1024] token embeddings
        hidden_fn: callable(input_ids) → [B, L, 1024] last hidden state
    """
    log(f"Loading base model from {model_dir}...")

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except Exception as e1:
        # Qwen3.5 is a multimodal model; try loading text model directly
        log(f"AutoModel failed ({e1}), trying Qwen3_5 text model...")
        try:
            from transformers import AutoModelForCausalLM
            full_model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model = full_model.model if hasattr(full_model, "model") else full_model
        except Exception as e2:
            log(f"Failed to load model: {e2}")
            log("Make sure you have a compatible transformers version:")
            log("  uv pip install 'transformers>=4.57'")
            sys.exit(1)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Find the text backbone and embedding layer
    # Qwen3_5Model (AutoModel) has .language_model (Qwen3_5TextModel)
    # Qwen3_5TextModel has .embed_tokens and .layers
    text_model = model
    for attr in ("language_model", "model", "text_model"):
        if hasattr(text_model, attr):
            text_model = getattr(text_model, attr)
            break

    embed_layer = text_model.embed_tokens
    n_params = sum(p.numel() for p in text_model.parameters())
    log(f"Base model loaded: {n_params / 1e6:.0f}M params, {device}")

    @torch.no_grad()
    def embed_fn(input_ids: torch.Tensor) -> torch.Tensor:
        return embed_layer(input_ids)

    @torch.no_grad()
    def hidden_fn(input_ids: torch.Tensor) -> torch.Tensor:
        out = text_model(input_ids)
        # Handle different return types
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        elif isinstance(out, tuple):
            return out[0]
        return out

    return embed_fn, hidden_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restricted vocab
    log(f"Building restricted vocab (K={args.top_k})...")
    vocab = RestrictedVocab(Path(args.counts), args.top_k)
    log(f"  Coverage: {vocab.coverage:.2%} of training corpus")

    # Dataset
    log(f"Loading training data from {args.data}...")
    train_ds, val_ds = make_splits(
        Path(args.data), vocab,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
    )
    log(f"  Train: {len(train_ds):,} sequences ({len(train_ds) * args.seq_len / 1e6:.0f}M tokens)")
    log(f"  Val:   {len(val_ds):,} sequences")

    # DataLoaders
    n_workers = min(os.cpu_count() or 4, 8)
    pin = device == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin,
        persistent_workers=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin,
        persistent_workers=True,
    )

    # Base model (frozen)
    embed_fn, hidden_fn = load_base_model(Path(args.model_dir), device)

    # MTP head
    log("Initializing MTP head...")
    if args.from_pretrained:
        mtp = MTPHead.from_pretrained(args.model_dir, vocab_size=args.top_k, device=device)
    else:
        mtp = MTPHead(vocab_size=args.top_k).to(device)
        log("  Initialized from scratch (no pre-trained weights)")

    params = mtp.param_count()
    log(f"  MTP params: {params['total']:,} ({params['total'] * 4 / 1e6:.1f} MB FP32)")
    for k, v in params.items():
        if k != "total" and v > 0:
            log(f"    {k}: {v:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        mtp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler: linear warmup + cosine decay
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        mtp.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"Resumed from epoch {start_epoch} (val_loss={ckpt.get('val_loss', '?')})")

    # Save setup
    setup = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(args.data),
        "counts": str(args.counts),
        "model_dir": str(args.model_dir),
        "mtp": {
            "vocab_size": args.top_k,
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_q_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 256,
            "params": params,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_frac": 0.05,
            "grad_clip": args.grad_clip,
            "amp": use_amp,
            "from_pretrained": args.from_pretrained,
            "val_fraction": args.val_fraction,
        },
        "vocab": {
            "k": args.top_k,
            "coverage": vocab.coverage,
        },
        "data_stats": {
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "seq_len": args.seq_len,
        },
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(out_dir / "train_setup.json", "w") as f:
        json.dump(setup, f, indent=2)
    log(f"Setup saved to {out_dir / 'train_setup.json'}")

    # CSV log
    log_path = out_dir / "train_log.csv"
    log_exists = log_path.exists() and start_epoch > 0
    log_file = open(log_path, "a" if log_exists else "w")
    if not log_exists:
        log_file.write("epoch,train_loss,val_loss,oov_frac,accept_est,lr,secs\n")
        log_file.flush()

    # Training loop
    best_val = float("inf")
    log(f"\nStarting training: {args.epochs} epochs, {len(train_dl)} batches/epoch")
    log(f"Total steps: {total_steps:,}, warmup: {warmup_steps:,}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        mtp.train()
        total_loss = 0.0
        total_oov = 0
        total_targets = 0
        n_batches = 0
        log_interval = max(len(train_dl) // 5, 1)

        for batch in train_dl:
            token_ids = batch["token_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            B, L = token_ids.shape

            # Frozen base model forward
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                embeddings = embed_fn(token_ids)     # [B, L, 1024]
                hidden = hidden_fn(token_ids)        # [B, L, 1024]

            # MTP inputs: embed[1:L-1] and hidden[0:L-2]
            # Target: token[2:L] → restricted vocab
            embed_input = embeddings[:, 1 : L - 1]   # [B, L-2, 1024]
            hidden_input = hidden[:, : L - 2]         # [B, L-2, 1024]

            # MTP forward + loss
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = mtp(embed_input, hidden_input)  # [B, L-2, K]
                loss = loss_fn(logits.reshape(-1, args.top_k), targets.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"  WARNING: {loss.item()} loss at batch {n_batches}, skipping")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            # Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(mtp.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_oov += (targets == -100).sum().item()
            total_targets += targets.numel()
            n_batches += 1

            # Sub-epoch logging
            if n_batches % log_interval == 0 and n_batches < len(train_dl):
                avg = total_loss / n_batches
                pct = n_batches / len(train_dl) * 100
                lr = optimizer.param_groups[0]["lr"]
                log(f"  [{n_batches}/{len(train_dl)} ({pct:.0f}%)] loss={avg:.4f} lr={lr:.2e}")

        avg_train = total_loss / max(n_batches, 1)
        oov_frac = total_oov / max(total_targets, 1)

        # Validation
        mtp.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for batch in val_dl:
                token_ids = batch["token_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

                B, L = token_ids.shape
                embeddings = embed_fn(token_ids)
                hidden = hidden_fn(token_ids)

                embed_input = embeddings[:, 1 : L - 1]
                hidden_input = hidden[:, : L - 2]

                logits = mtp(embed_input, hidden_input)
                loss = loss_fn(logits.reshape(-1, args.top_k), targets.reshape(-1))
                val_loss += loss.item()

                # Acceptance rate estimate (accuracy on non-OOV targets)
                preds = logits.argmax(dim=-1)  # [B, L-2]
                mask = targets != -100
                if mask.any():
                    val_correct += (preds[mask] == targets[mask]).sum().item()
                    val_total += mask.sum().item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        accept_est = val_correct / max(val_total, 1)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        log(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"accept={accept_est:.1%}  oov={oov_frac:.1%}  "
            f"lr={lr:.1e}  {dt:.1f}s"
        )

        # CSV log
        log_file.write(
            f"{epoch + 1},{avg_train:.6f},{avg_val:.6f},"
            f"{oov_frac:.4f},{accept_est:.4f},{lr:.6e},{dt:.1f}\n"
        )
        log_file.flush()

        # Checkpoint
        ckpt = {
            "model": mtp.state_dict(),
            "epoch": epoch,
            "val_loss": avg_val,
            "accept_est": accept_est,
            "vocab_k": args.top_k,
            "setup": setup,
        }
        if avg_val < best_val:
            best_val = avg_val
            torch.save(ckpt, out_dir / "best.pt")
            log(f"  → saved best (val={avg_val:.4f}, accept={accept_est:.1%})")

        # Save latest every epoch
        torch.save(ckpt, out_dir / "latest.pt")

    log_file.close()
    log(f"\nTraining complete. Best val loss: {best_val:.4f}")
    log(f"Checkpoints: {out_dir}/best.pt, {out_dir}/latest.pt")


# ---------------------------------------------------------------------------
# Export: convert trained MTP head → GWMT binary for GWEN inference
# ---------------------------------------------------------------------------

def export(args: argparse.Namespace) -> None:
    """Export trained MTP weights to GWMT format for GWEN inference."""
    import struct

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    vocab_k = ckpt.get("vocab_k", 20000)

    output = Path(args.output) if args.output else ckpt_path.parent / "mtp_finetuned.bin"
    output.parent.mkdir(parents=True, exist_ok=True)

    # GWMT format constants
    DTYPE_F32 = 0
    DTYPE_F16 = 1

    # Map our state dict back to canonical MTP tensor names
    name_map = {
        "fc.weight": "mtp.fc.weight",
        "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
        "self_attn.q_proj.weight": "mtp.layers.0.self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "mtp.layers.0.self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "mtp.layers.0.self_attn.v_proj.weight",
        "self_attn.o_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        "self_attn.q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "self_attn.k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
        "post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "mlp.gate_proj.weight": "mtp.layers.0.mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mtp.layers.0.mlp.up_proj.weight",
        "mlp.down_proj.weight": "mtp.layers.0.mlp.down_proj.weight",
        "norm.weight": "mtp.norm.weight",
    }

    # Build ordered tensor list (15 canonical + lm_head)
    tensors = []
    for our_name, canonical_name in name_map.items():
        tensor = state_dict[our_name]
        is_norm = "norm" in our_name

        if is_norm:
            # Convert back to Qwen3.5 convention: stored = weight - 1.0
            # Then add 1.0 for GWMT (which stores 1+w)
            # Net effect: just keep as-is (our weight IS 1+w already)
            data = tensor.float().numpy()
            dtype_code = DTYPE_F32
        else:
            data = tensor.half().numpy()
            dtype_code = DTYPE_F16

        tensors.append((canonical_name, data, dtype_code))

    # Also include lm_head (the fine-tuned restricted vocab head)
    if "lm_head.weight" in state_dict:
        lm_data = state_dict["lm_head.weight"].half().numpy()
        tensors.append(("mtp.lm_head.weight", lm_data, DTYPE_F16))

    log(f"Exporting {len(tensors)} tensors to {output}")

    with open(output, "wb") as f:
        f.write(b"GWMT")
        f.write(struct.pack("<I", 2))  # Version 2 (fine-tuned, includes lm_head)
        f.write(struct.pack("<I", len(tensors)))

        total_bytes = 0
        for name, data, dtype_code in tensors:
            name_bytes = name.encode("utf-8")
            raw = data.tobytes()

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", dtype_code))
            f.write(struct.pack("<I", data.ndim))
            for d in data.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            total_bytes += len(raw)

            size_str = f"{len(raw) / 1024:.1f} KB" if len(raw) < 1024 * 1024 else f"{len(raw) / 1024 / 1024:.2f} MB"
            log(f"  {name:<55} {str(data.shape):<20} {size_str}")

    file_size = output.stat().st_size
    log(f"\nSaved: {output} ({file_size / 1024 / 1024:.2f} MB)")
    log(f"  Vocab K={vocab_k}, epoch={ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', '?'):.4f}, "
        f"accept_est={ckpt.get('accept_est', '?'):.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MTP head for restricted-vocab English")
    sub = parser.add_subparsers(dest="cmd")

    # --- train ---
    p = sub.add_parser("train", help="Train MTP head")
    p.add_argument("--data", type=Path, required=True, help="Tokenized data (uint32 binary)")
    p.add_argument("--counts", type=Path, default=Path("data/token_counts.bin"),
                    help="Token frequency counts (int64)")
    p.add_argument("--model-dir", type=Path, default=Path.home() / "models" / "hf" / "Qwen3.5-0.8B",
                    help="HuggingFace model directory for base model")
    p.add_argument("--out-dir", type=Path, default=Path("train/runs/mtp_v1"),
                    help="Output directory for checkpoints and logs")
    p.add_argument("--top-k", type=int, default=20000, help="Restricted vocab size")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--from-pretrained", action="store_true", default=True,
                    help="Initialize from pre-trained MTP weights (default: True)")
    p.add_argument("--no-pretrained", dest="from_pretrained", action="store_false",
                    help="Train MTP head from scratch")
    p.add_argument("--resume", type=Path, help="Resume from checkpoint")

    # --- export ---
    p = sub.add_parser("export", help="Export trained MTP to GWMT binary")
    p.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint (.pt)")
    p.add_argument("--output", type=Path, help="Output GWMT file")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "export":
        export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
