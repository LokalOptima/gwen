#!/usr/bin/env python3
"""
Fine-tune the MTP head for restricted-vocab English speculative decoding.

Freezes the main Qwen3.5-0.8B model (served by GWEN server for hidden states)
and trains only the 21M-param MTP head to predict token[t+2] from
embed[t+1] + hidden[t], over a restricted vocab of the top-K most frequent
English tokens.

Requires a running GWEN server (build/gwen_server) for hidden state extraction.
Embeddings are loaded directly from safetensors (no transformers needed).

Usage:
    # Start GWEN server first:
    build/gwen_server --model Qwen3.5-0.8B-Q4_K_M.gguf --port 8090

    # Then train:
    uv run --with 'torch>=2.7' --with safetensors --with numpy --with tqdm \
        train/train_mtp.py train \
        --data data/train_tokens.bin \
        --counts data/token_counts.bin \
        --model-dir ~/models/hf/Qwen3.5-0.8B \
        --server-url http://127.0.0.1:8090 \
        --out-dir train/runs/mtp_spoken_v1 \
        --top-k 20000 \
        --epochs 3 \
        --max-tokens 32768 \
        --lr 1e-4 \
        --from-pretrained

    # Export:
    uv run --with 'torch>=2.7' --with safetensors --with numpy \
        train/train_mtp.py export \
        --checkpoint train/runs/mtp_v1/best.pt \
        --output ~/models/gguf/mtp_finetuned.bin
"""

import argparse
import hashlib
import http.client
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path so we can import from train/
sys.path.insert(0, str(Path(__file__).parent))
from dataset import RestrictedVocab, TokenBatchSampler, TokenSequenceDataset, make_splits, mtp_collate
from model import MTPHead


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Embeddings: load directly from safetensors (no transformers)
# ---------------------------------------------------------------------------

def load_embeddings(model_dir: Path, device: str) -> torch.Tensor:
    """Load token embedding weights from safetensors.

    Returns the [vocab_size, n_embed] embedding matrix on device.
    Only loads this one tensor — no full model, no transformers.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    # Qwen3.5 multimodal uses "model.language_model.embed_tokens.weight"
    # Standard text models use "model.embed_tokens.weight"
    emb_keys = ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"]

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        emb_key = None
        for k in emb_keys:
            if k in index["weight_map"]:
                emb_key = k
                break
        if emb_key is None:
            raise KeyError(f"Cannot find embedding tensor. Available: {list(index['weight_map'].keys())[:10]}")
        sf_path = model_dir / index["weight_map"][emb_key]
    else:
        sf_path = next(model_dir.glob("*.safetensors"))
        emb_key = None  # will try both below

    log(f"Loading embeddings from {sf_path}...")
    with safe_open(str(sf_path), framework="pt") as f:
        if emb_key:
            embed_weight = f.get_tensor(emb_key)
        else:
            for k in emb_keys:
                if k in f.keys():
                    embed_weight = f.get_tensor(k)
                    break
            else:
                raise KeyError(f"Cannot find embedding tensor in {sf_path}")

    # Keep on CPU — F.embedding is just indexing, fast on CPU.
    # Saves ~500 MB GPU memory vs loading to device.
    # Use Q6_K-dequantized embeddings from GGUF (matches CUDA runtime exactly).
    # Falls back to safetensors if Q6K file not available.
    q6k_path = Path("data/embed_tokens_q6k.npy")
    if q6k_path.exists():
        log(f"Loading Q6K-dequantized embeddings from {q6k_path} (matches CUDA runtime)")
        embed_weight = torch.from_numpy(np.load(str(q6k_path))).to(dtype=torch.float16)
    else:
        log(f"WARNING: {q6k_path} not found, using safetensors embeddings (won't match CUDA)")
        embed_weight = embed_weight.to(dtype=torch.float16)
    log(f"  Embedding matrix: {list(embed_weight.shape)}, {embed_weight.dtype} (CPU)")
    return embed_weight


def load_main_model_norm(model_dir: Path) -> torch.Tensor:
    """Load main model's output RMSNorm weight (for acceptance rate computation).

    Returns [n_embed] F32 weight with +1 convention applied.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    norm_keys = ["model.language_model.norm.weight", "model.norm.weight"]

    for sf_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt") as f:
            for k in norm_keys:
                if k in f.keys():
                    w = f.get_tensor(k).float() + 1.0  # Qwen3.5 (1+w) convention
                    log(f"  Main model output norm: {list(w.shape)}, key={k}")
                    return w

    raise KeyError(f"Cannot find output norm weight in {model_dir}")


# ---------------------------------------------------------------------------
# GWEN server client for hidden state extraction
# ---------------------------------------------------------------------------

class GwenClient:
    """Client for GWEN inference server's batch hidden state extraction.

    Uses keep-alive HTTP connection for efficiency.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.conn = http.client.HTTPConnection(host, port, timeout=300)
        self._check_health()

    def _check_health(self):
        """Verify server is running."""
        try:
            self.conn.request("GET", "/health")
            resp = self.conn.getresponse()
            data = json.loads(resp.read().decode())
            log(f"GWEN server: {data.get('model', '?')}, "
                f"n_embed={data.get('n_embed', '?')}, "
                f"max_seq={data.get('max_seq', '?')}")
        except ConnectionRefusedError:
            raise RuntimeError(
                f"Cannot connect to GWEN server at {self.host}:{self.port}. "
                f"Start it first: build/gwen_server --model <path.gguf> --port {self.port}"
            )

    def _reconnect(self):
        """Reconnect if connection dropped."""
        self.conn.close()
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=300)

    def batch_extract(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states for a batch of padded sequences.

        Args:
            token_ids: [B, L] int tensor (padded sequences)

        Returns:
            [B, L, n_embed] float16 tensor of hidden states
        """
        token_np = token_ids.cpu().numpy().astype(np.int32)
        B, L = token_np.shape

        # Binary protocol: [uint32 B][uint32 L][int32 tokens[B*L]]
        body = struct.pack('<II', B, L) + token_np.tobytes()

        try:
            self.conn.request("POST", "/batch_extract", body=body,
                              headers={"Content-Type": "application/octet-stream"})
            resp = self.conn.getresponse()
        except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
            self._reconnect()
            self.conn.request("POST", "/batch_extract", body=body,
                              headers={"Content-Type": "application/octet-stream"})
            resp = self.conn.getresponse()

        if resp.status != 200:
            raise RuntimeError(f"GWEN server error {resp.status}: {resp.read().decode()}")

        data = resp.read()
        B2, L2, d = struct.unpack('<III', data[:12])
        hidden_bytes = B2 * L2 * d * 2
        hidden = np.frombuffer(data[12:12 + hidden_bytes], dtype=np.float16).reshape(B2, L2, d).copy()
        return torch.from_numpy(hidden)

    def batch_extract_with_preds(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract hidden states AND main model predictions.

        Args:
            token_ids: [B, L] int tensor

        Returns:
            (hidden [B, L, n_embed] float16, predictions [B, L] int32)
        """
        token_np = token_ids.cpu().numpy().astype(np.int32)
        B, L = token_np.shape

        body = struct.pack('<II', B, L) + token_np.tobytes()

        try:
            self.conn.request("POST", "/batch_extract?preds=1", body=body,
                              headers={"Content-Type": "application/octet-stream"})
            resp = self.conn.getresponse()
        except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
            self._reconnect()
            self.conn.request("POST", "/batch_extract?preds=1", body=body,
                              headers={"Content-Type": "application/octet-stream"})
            resp = self.conn.getresponse()

        if resp.status != 200:
            raise RuntimeError(f"GWEN server error {resp.status}: {resp.read().decode()}")

        data = resp.read()
        B2, L2, d = struct.unpack('<III', data[:12])
        hidden_bytes = B2 * L2 * d * 2
        hidden = np.frombuffer(data[12:12 + hidden_bytes], dtype=np.float16).reshape(B2, L2, d).copy()
        preds = np.frombuffer(data[12 + hidden_bytes:], dtype=np.int32).reshape(B2, L2).copy()
        return torch.from_numpy(hidden), torch.from_numpy(preds)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_acceptance(
    mtp_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    main_norm_w: torch.Tensor,
    embed_weight: torch.Tensor,
    vocab_top_k_ids: np.ndarray,
    eps: float = 1e-6,
) -> tuple[int, int]:
    """Compute speculative decoding acceptance rate.

    Acceptance = MTP's full-vocab prediction matches main model's prediction.

    Args:
        mtp_logits: [B, L-2, K] MTP output logits over restricted vocab
        hidden_states: [B, L, n_embed] full hidden states from batch extraction
        main_norm_w: [n_embed] main model output RMSNorm weight (with +1)
        embed_weight: [vocab_size, n_embed] embedding matrix (= tied lm_head)
        vocab_top_k_ids: [K] mapping from restricted index to full vocab ID
        eps: RMSNorm epsilon

    Returns:
        (n_accepted, n_total)
    """
    B, Lm2, K = mtp_logits.shape
    L = Lm2 + 2
    device = mtp_logits.device

    # MTP prediction at position i predicts token[i+2].
    # Main model at position i+1: hidden[i+1] -> norm -> embed.T -> predicts token[i+2].
    # So compare MTP argmax at i against main model argmax from hidden[i+1].

    # Main model predictions from hidden[1:L-1] (positions 1..L-2)
    h = hidden_states[:, 1:L-1].float().to(device)  # [B, L-2, n_embed]
    norm_w = main_norm_w.to(device)
    rms = h.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = h * rms * norm_w  # [B, L-2, n_embed]

    # Compute main model predictions in chunks to avoid OOM (248K vocab)
    n_accepted = 0
    n_total = 0
    chunk_size = 32  # sequences per chunk
    embed_t = embed_weight.float().to(device).T  # [n_embed, vocab_size]

    for ci in range(0, B, chunk_size):
        ce = min(ci + chunk_size, B)
        main_logits = normed[ci:ce] @ embed_t  # [chunk, L-2, vocab_size]
        main_preds = main_logits.argmax(dim=-1)  # [chunk, L-2] full vocab IDs

        # MTP predictions mapped to full vocab
        mtp_preds_restricted = mtp_logits[ci:ce].argmax(dim=-1)  # [chunk, L-2]
        top_k_ids_t = torch.from_numpy(vocab_top_k_ids.astype(np.int64)).to(device)
        mtp_preds_full = top_k_ids_t[mtp_preds_restricted]  # [chunk, L-2] full vocab IDs

        n_accepted += (mtp_preds_full == main_preds).sum().item()
        n_total += main_preds.numel()

    return n_accepted, n_total


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

    # DataLoaders with token-budget batching
    n_workers = min(os.cpu_count() or 4, 8)
    pin = device == "cuda"

    train_sampler = TokenBatchSampler(
        train_ds.lengths, max_tokens=args.max_tokens,
        shuffle=True, seed=42, drop_last=True,
    )
    val_sampler = TokenBatchSampler(
        val_ds.lengths, max_tokens=args.max_tokens,
        shuffle=False, drop_last=False,
    )

    log(f"  Token budget: {args.max_tokens:,} tokens/batch")
    log(f"  Train batches: {len(train_sampler):,}, Val batches: {len(val_sampler):,}")

    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=mtp_collate,
        num_workers=n_workers,
        pin_memory=pin,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=mtp_collate,
        num_workers=n_workers,
        pin_memory=pin,
        persistent_workers=True,
    )

    # Embeddings from safetensors (just the lookup table, ~500MB)
    embed_weight = load_embeddings(Path(args.model_dir), device)

    # GWEN server for hidden state extraction
    host, port = args.server_url.replace("http://", "").split(":")
    gwen = GwenClient(host, int(port))

    # MTP head
    log("Initializing MTP head...")
    if args.from_pretrained:
        mtp = MTPHead.from_pretrained(args.model_dir, vocab_size=args.top_k, device=device,
                                       vocab_ids=vocab.top_k_ids.tolist())
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
    total_steps = len(train_sampler) * args.epochs
    warmup_steps = args.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP — FP16 needs GradScaler to prevent underflow in gradients
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        mtp.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"Resumed from epoch {start_epoch} (val_loss={ckpt.get('val_loss', '?')})")

    # Save setup with checksums for reproducibility
    data_path = Path(args.data).resolve()
    counts_path = Path(args.counts).resolve()
    log("Computing file checksums...")
    data_sha = sha256_file(data_path)
    counts_sha = sha256_file(counts_path)

    setup = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": sys.argv,
        "data": {
            "path": str(data_path),
            "sha256": data_sha,
            "size_bytes": data_path.stat().st_size,
        },
        "counts": {
            "path": str(counts_path),
            "sha256": counts_sha,
        },
        "model_dir": str(args.model_dir),
        "server_url": args.server_url,
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
            "max_tokens": args.max_tokens,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": warmup_steps,
            "grad_clip": args.grad_clip,
            "amp_dtype": "float16",
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
            "train_batches": len(train_sampler),
            "val_batches": len(val_sampler),
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

    # CSV log — train_loss logged every batch, val columns filled at eval points
    log_path = out_dir / "train_log.csv"
    log_exists = log_path.exists() and start_epoch > 0
    log_file = open(log_path, "a" if log_exists else "w")
    if not log_exists:
        log_file.write("step,epoch,batch_loss,avg_loss,val_loss,accept_rate,gt_acc,lr\n")
        log_file.flush()
    global_step = start_epoch * len(train_sampler)

    # Load val cache (pre-extracted hidden states from scratch/cache_val.py)
    val_cache_dir = out_dir / "val_cache"
    val_cache_files = sorted(val_cache_dir.glob("*.pt")) if val_cache_dir.exists() else []
    if val_cache_files:
        log(f"Val cache: {len(val_cache_files)} batches in {val_cache_dir}")
    else:
        log(f"WARNING: No val cache found at {val_cache_dir}")
        log(f"  Run: uv run --with 'torch>=2.7' --with safetensors --with numpy --with tqdm --with packaging scratch/cache_val.py")
        log(f"  Falling back to live server eval (slow)")

    # Training loop
    best_val = float("inf")
    log(f"\nStarting training: {args.epochs} epochs, {len(train_dl)} batches/epoch")
    log(f"Total steps: {total_steps:,}, warmup: {warmup_steps:,}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        mtp.train()
        total_loss = 0.0
        total_oov = 0
        total_targets = 0
        n_batches = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}",
                    unit="batch", file=sys.stderr, dynamic_ncols=True)
        for batch in pbar:
            token_ids = batch["token_ids"]
            B, L = token_ids.shape

            # Embeddings: CPU lookup (fast indexing, saves ~500MB GPU)
            with torch.no_grad():
                embeddings = F.embedding(token_ids, embed_weight)  # [B, L, 1024] CPU

            # Hidden states + main model predictions from GWEN server
            with torch.no_grad():
                hidden, main_preds = gwen.batch_extract_with_preds(token_ids)
                # hidden: [B, L, 1024] FP16 CPU
                # main_preds: [B, L] int32 CPU — main model's argmax per position

            # MTP inputs: embed[1:L-1] and hidden[0:L-2]
            embed_input = embeddings[:, 1 : L - 1]   # [B, L-2, 1024]
            hidden_input = hidden[:, : L - 2]         # [B, L-2, 1024]

            # MTP targets: main model predictions from h[1:L-1], mapped to restricted vocab.
            # MTP[i] should predict what main model predicts from h[i+1] = main_preds[i+1].
            mtp_target_ids = main_preds[:, 1:L-1].numpy()  # [B, L-2] full vocab IDs
            mtp_targets = torch.from_numpy(
                vocab.map_targets(mtp_target_ids)
            ).long().to(device, non_blocking=True)

            # Chunked forward/backward to bound GPU memory.
            # With K=20K vocab, logits [chunk, L-2, 20K] in FP32 dominate.
            # Target: keep logits under ~500 MB → chunk * (L-2) * K * 4 < 500M
            seq_logit_bytes = max(L - 2, 1) * args.top_k * 4  # FP32 per sequence
            max_chunk = max(1, int(500_000_000 / seq_logit_bytes))
            max_chunk = min(max_chunk, B)
            n_chunks = (B + max_chunk - 1) // max_chunk
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            skip = False

            for ci in range(0, B, max_chunk):
                ce = min(ci + max_chunk, B)
                e_chunk = embed_input[ci:ce].to(device, non_blocking=True)
                h_chunk = hidden_input[ci:ce].to(device, dtype=torch.float16, non_blocking=True)
                t_chunk = mtp_targets[ci:ce]

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = mtp(e_chunk, h_chunk)  # [chunk, L-2, K]
                    loss = loss_fn(logits.reshape(-1, args.top_k), t_chunk.reshape(-1))
                    loss = loss / n_chunks  # scale for gradient accumulation

                if torch.isnan(loss) or torch.isinf(loss):
                    log(f"  WARNING: {loss.item()} loss at batch {n_batches}, skipping")
                    skip = True
                    break

                scaler.scale(loss).backward()
                batch_loss += loss.item()  # already divided by n_chunks

            if skip:
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(mtp.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += batch_loss
            total_oov += (mtp_targets == -100).sum().item()
            total_targets += mtp_targets.numel()
            n_batches += 1

            # Update progress bar + log
            avg = total_loss / n_batches
            lr_val = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_val:.1e}",
                             B=B, L=L)
            global_step += 1
            frac_epoch = epoch + n_batches / len(train_sampler)
            log_file.write(f"{global_step},{frac_epoch:.4f},{batch_loss:.6f},{avg:.6f},,,,{lr_val:.6e}\n")
            if global_step % 100 == 0:
                log_file.flush()

            # --- Intra-epoch eval + checkpoint every eval_every batches ---
            if n_batches % args.eval_every == 0:
                mtp.eval()
                val_loss = 0.0
                val_accept = 0
                val_total = 0
                val_batches = 0

                with torch.no_grad():
                    if val_cache_files:
                        # Fast path: load pre-cached hidden states + main model preds
                        for vf in val_cache_files:
                            vc = torch.load(str(vf), map_location="cpu", weights_only=True)
                            vh = vc["hidden_input"]  # [B, L-2, n_embed]
                            vtgt = vc["targets"].to(device, non_blocking=True)  # [B, L-2] main model preds (restricted)
                            vtids = vc["token_ids"]
                            vB, vL = vtids.shape
                            ve = F.embedding(vtids, embed_weight)[:, 1:vL-1]

                            seq_logit_bytes_v = max(ve.shape[1], 1) * args.top_k * 4
                            max_chunk_v = max(1, int(500_000_000 / seq_logit_bytes_v))
                            max_chunk_v = min(max_chunk_v, vB)
                            for ci in range(0, vB, max_chunk_v):
                                ce = min(ci + max_chunk_v, vB)
                                ec = ve[ci:ce].to(device, non_blocking=True)
                                hc = vh[ci:ce].to(device, dtype=torch.float16, non_blocking=True)
                                tc = vtgt[ci:ce]

                                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                                    vlogits = mtp(ec, hc)
                                    vloss = loss_fn(vlogits.reshape(-1, args.top_k), tc.reshape(-1))
                                val_loss += vloss.item()

                                # Acceptance = MTP argmax matches main model prediction
                                vpreds = vlogits.argmax(dim=-1)
                                vmask = tc != -100
                                if vmask.any():
                                    val_accept += (vpreds[vmask] == tc[vmask]).sum().item()
                                    val_total += vmask.sum().item()
                                val_batches += 1
                    else:
                        # Slow path: call GWEN server for each val batch
                        for vbatch in val_dl:
                            vtids = vbatch["token_ids"]
                            vB, vL = vtids.shape
                            ve = F.embedding(vtids, embed_weight)[:, 1:vL-1]

                            # Get hidden + main model predictions
                            vh_full, vpreds_full = gwen.batch_extract_with_preds(vtids)
                            vh = vh_full[:, :vL-2]

                            # Build targets from main model predictions
                            vt_ids = vpreds_full[:, 1:vL-1].numpy()
                            vtgt = torch.from_numpy(
                                vocab.map_targets(vt_ids)
                            ).long().to(device, non_blocking=True)

                            ec = ve.to(device)
                            hc = vh.to(device, dtype=torch.float16)
                            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                                vlogits = mtp(ec, hc)
                                vloss = loss_fn(vlogits.reshape(-1, args.top_k), vtgt.reshape(-1))
                            val_loss += vloss.item()

                            vpreds = vlogits.argmax(dim=-1)
                            vmask = vtgt != -100
                            if vmask.any():
                                val_accept += (vpreds[vmask] == vtgt[vmask]).sum().item()
                                val_total += vmask.sum().item()
                            val_batches += 1

                avg_val = val_loss / max(val_batches, 1)
                accept_rate = val_accept / max(val_total, 1)
                avg_train_so_far = total_loss / max(n_batches, 1)
                lr_val = optimizer.param_groups[0]["lr"]
                oov_frac_so_far = total_oov / max(total_targets, 1)

                log(
                    f"  [{epoch+1}:{n_batches}/{len(train_sampler)}]  "
                    f"train={avg_train_so_far:.4f}  val={avg_val:.4f}  "
                    f"accept={accept_rate:.1%}  "
                    f"oov={oov_frac_so_far:.1%}  lr={lr_val:.1e}"
                )

                # CSV log
                frac_epoch = epoch + n_batches / len(train_sampler)
                log_file.write(
                    f"{global_step},{frac_epoch:.4f},{avg_train_so_far:.6f},{avg_train_so_far:.6f},"
                    f"{avg_val:.6f},{accept_rate:.4f},,{lr_val:.6e}\n"
                )
                log_file.flush()

                # Checkpoint
                ckpt = {
                    "model": mtp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "batch": n_batches,
                    "val_loss": avg_val,
                    "accept_rate": accept_rate,
                    "vocab_k": args.top_k,
                    "setup": setup,
                }
                if avg_val < best_val:
                    best_val = avg_val
                    torch.save(ckpt, out_dir / "best.pt")
                    log(f"  -> saved best (val={avg_val:.4f}, accept={accept_rate:.1%})")

                torch.save(ckpt, out_dir / "latest.pt")
                mtp.train()

        avg_train = total_loss / max(n_batches, 1)
        oov_frac = total_oov / max(total_targets, 1)
        log(f"Epoch {epoch+1}/{args.epochs} done: train={avg_train:.4f} oov={oov_frac:.1%}")

    log_file.close()
    log(f"\nTraining complete. Best val loss: {best_val:.4f}")
    log(f"Checkpoints: {out_dir}/best.pt, {out_dir}/latest.pt")


# ---------------------------------------------------------------------------
# Export: convert trained MTP head → GWMT binary for GWEN inference
# ---------------------------------------------------------------------------

def export(args: argparse.Namespace) -> None:
    """Export trained MTP weights to GWMT format for GWEN inference."""

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
    accept = ckpt.get('accept_rate', ckpt.get('accept_est', None))
    accept_str = f"{accept:.1%}" if accept is not None else "?"
    log(f"  Vocab K={vocab_k}, epoch={ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', '?'):.4f}, "
        f"accept_rate={accept_str}")


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
                    help="HuggingFace model dir (for embeddings + pre-trained MTP weights)")
    p.add_argument("--server-url", type=str, default="http://127.0.0.1:8090",
                    help="GWEN server URL for hidden state extraction")
    p.add_argument("--out-dir", type=Path, default=Path("train/runs/mtp_v1"),
                    help="Output directory for checkpoints and logs")
    p.add_argument("--top-k", type=int, default=20000, help="Restricted vocab size")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=32768,
                    help="Token budget per batch (replaces fixed batch_size)")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=0,
                    help="Linear warmup steps (default: 0, not needed for pretrained)")
    p.add_argument("--eval-every", type=int, default=2000,
                    help="Run validation and checkpoint every N training batches")
    p.add_argument("--micro-batch", type=int, default=128,
                    help="Max sequences per MTP forward/backward chunk (bounds GPU memory)")
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
