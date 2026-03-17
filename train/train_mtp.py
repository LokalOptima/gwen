#!/usr/bin/env python3
"""
Fine-tune the MTP head for restricted-vocab English speculative decoding.

Freezes the main Qwen3.5-0.8B model (served by GWEN server for hidden states)
and trains only the 40M-param MTP head to predict what the main model would
output at position t+2, given embed[t+1] + hidden[t], over a restricted vocab
of the top-K most frequent English tokens.

Requires a running GWEN server (build/gwen_server) for hidden state extraction.

Hyperparameter choices based on EAGLE, Medusa, and IBM Recurrent Drafter papers
for speculative decoding draft head fine-tuning.

Usage:
    # Start GWEN server first:
    build/gwen_server --model Qwen3.5-0.8B-Q4_K_M.gguf --port 8090

    # Train:
    uv run train/train_mtp.py train \
        --data data/train_speech.bin \
        --counts data/token_counts.bin \
        --out-dir train/runs/mtp_v3

    # Export:
    uv run train/train_mtp.py export \
        --checkpoint train/runs/mtp_v3/best.pt \
        --output ~/models/gguf/mtp_finetuned.bin
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import warnings
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
from gwen_client import GwenClient
from model import MTPHead


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def load_embeddings(model_dir: Path, device: str) -> torch.Tensor:
    """Load token embedding weights. Prefers Q6K-dequantized (matches CUDA runtime)."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    q6k_path = Path("data/embed_tokens_q6k.npy")

    if q6k_path.exists():
        log(f"Loading Q6K-dequantized embeddings from {q6k_path} (matches CUDA runtime)")
        embed_weight = torch.from_numpy(np.load(str(q6k_path))).to(dtype=torch.float16)
    else:
        emb_keys = ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"]
        index_path = model_dir / "model.safetensors.index.json"

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            emb_key = next((k for k in emb_keys if k in index["weight_map"]), None)
            if not emb_key:
                raise KeyError(f"Cannot find embedding tensor in index")
            sf_path = model_dir / index["weight_map"][emb_key]
        else:
            sf_path = next(model_dir.glob("*.safetensors"))
            emb_key = None

        log(f"WARNING: {q6k_path} not found, using safetensors (won't match CUDA exactly)")
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
        embed_weight = embed_weight.to(dtype=torch.float16)

    log(f"  Embedding matrix: {list(embed_weight.shape)}, {embed_weight.dtype} (CPU)")
    return embed_weight


def _load_output_norm(model_dir: Path) -> torch.Tensor:
    """Load the output RMSNorm weight (model.norm.weight) from safetensors.

    Qwen3.5 convention: stored as additive offset w, applied as x * (1 + w).
    Our convention: x * weight, so we return (1 + w).
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    norm_keys = ["model.norm.weight", "model.language_model.norm.weight"]

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for nk in norm_keys:
            if nk in index["weight_map"]:
                sf_path = model_dir / index["weight_map"][nk]
                with safe_open(str(sf_path), framework="pt") as f:
                    w = f.get_tensor(nk)
                    return (w.float() + 1.0)
    else:
        for sf_path in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt") as f:
                for nk in norm_keys:
                    if nk in f.keys():
                        w = f.get_tensor(nk)
                        return (w.float() + 1.0)

    raise KeyError(f"Cannot find output norm weight in {model_dir}")



# ---------------------------------------------------------------------------
# Distillation loss (v3)
# ---------------------------------------------------------------------------

def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      T: float = 2.0, mask: torch.Tensor | None = None) -> torch.Tensor:
    """KL divergence loss with temperature scaling for knowledge distillation.

    Args:
        student_logits: [*, K] raw logits from MTP head
        teacher_logits: [*, K] raw logits from main model (over restricted vocab)
        T: temperature for softmax smoothing (higher = softer distribution)
        mask: optional [*] bool mask, True where loss should be computed

    Returns:
        Scalar loss = KL(student || teacher) * T^2
    """
    if mask is not None:
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        if student_logits.numel() == 0:
            return student_logits.sum()  # zero loss, preserves grad

    # Flatten to [N_tokens, K] for correct per-token averaging
    shape = student_logits.shape
    student_flat = student_logits.reshape(-1, shape[-1])
    teacher_flat = teacher_logits.reshape(-1, shape[-1])
    n_tokens = student_flat.shape[0]

    teacher_probs = F.softmax(teacher_flat / T, dim=-1)
    student_log_probs = F.log_softmax(student_flat / T, dim=-1)
    # reduction='sum' then divide by n_tokens — 'batchmean' divides by batch dim only
    return F.kl_div(student_log_probs, teacher_probs, reduction='sum') * (T ** 2) / n_tokens



def distillation_loss_idk(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    T: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL divergence loss with split temperature for IDK training.

    Temperature T softens the K restricted token probabilities for gradient flow,
    while IDK probability stays at T=1 (true OOV mass, always calibrated).

    Args:
        student_logits: [*, K+1] raw logits from MTP head
        teacher_probs: [*, K+1] F32 teacher probabilities (from server p_idk + restricted logits)
        T: temperature for restricted tokens (decays T=2→1 over training)
        mask: optional [*] bool mask

    Returns:
        Scalar loss (per-token averaged KL divergence)
    """
    if mask is not None:
        student_logits = student_logits[mask]
        teacher_probs = teacher_probs[mask]
        if student_logits.numel() == 0:
            return student_logits.sum()  # zero loss, preserves grad

    # Flatten to [N, K+1]
    shape = student_logits.shape
    K_plus_1 = shape[-1]
    K = K_plus_1 - 1
    student_flat = student_logits.reshape(-1, K_plus_1)
    teacher_flat = teacher_probs.reshape(-1, K_plus_1).float()
    n_tokens = student_flat.shape[0]

    # Split temperature: soften restricted probs, keep IDK at T=1
    p_restricted = teacher_flat[:, :K]   # [N, K] — sums to (1 - p_idk)
    p_idk = teacher_flat[:, K:]          # [N, 1] — true OOV mass

    if T != 1.0:
        p_soft = p_restricted.pow(1.0 / T)
        p_soft_sum = p_soft.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p_soft = p_soft / p_soft_sum * (1.0 - p_idk)  # re-scale to preserve IDK mass
    else:
        p_soft = p_restricted

    teacher_target = torch.cat([p_soft, p_idk], dim=-1)  # [N, K+1] sums to 1.0
    teacher_target = teacher_target.clamp(min=1e-8)
    teacher_target = teacher_target / teacher_target.sum(dim=-1, keepdim=True)

    student_log_probs = F.log_softmax(student_flat, dim=-1)
    return F.kl_div(student_log_probs, teacher_target, reduction='sum') / n_tokens


# ---------------------------------------------------------------------------
# Evaluation (shared by intra-epoch, end-of-epoch, and initial eval)
# ---------------------------------------------------------------------------

def run_eval(
    mtp: nn.Module,
    val_cache_files: list,
    val_dl: DataLoader | None,
    gwen: GwenClient | None,
    vocab: RestrictedVocab,
    embed_weight: torch.Tensor,
    top_k: int,
    device: str,
    use_amp: bool,
    temperature: float = 2.0,
    restricted_embed: torch.Tensor | None = None,
    output_norm_weight: torch.Tensor | None = None,
    output_norm_eps: float = 1e-6,
    idk: bool = False,
) -> tuple[float, float, int] | tuple[float, float, int, dict]:
    """Run validation with KL distillation loss.

    For v3 val cache: stores (token_ids, hidden_input). Teacher logits are
    recomputed on-the-fly: RMSNorm(hidden) @ restricted_embed.T (~ms in PyTorch).

    For v4 val cache (IDK): stores (token_ids, hidden_input, p_idk).
    Teacher distribution uses server-computed p_idk.

    Args:
        restricted_embed: [K, n_embed] FP16 on CPU (for computing teacher logits)
        output_norm_weight: [n_embed] F32 on CPU (output RMSNorm weight)
        idk: if True, use IDK teacher distribution and metrics

    Returns:
        Without idk: (val_loss, accept_rate, n_tokens_evaluated)
        With idk: (val_loss, accept_rate, n_tokens_evaluated, idk_metrics)
    """
    mtp.eval()
    total_loss = 0.0
    total_accept = 0
    total_tokens = 0
    n_batches = 0

    # IDK metrics accumulators
    total_idk_preds = 0      # student predicted IDK
    total_teacher_oov = 0    # teacher argmax was OOV (IDK)
    total_idk_correct = 0    # student IDK and teacher was OOV (true positive)
    total_non_idk_accept = 0 # non-IDK predictions that match teacher
    total_non_idk_tokens = 0 # total non-IDK predictions

    # Move restricted_embed and output_norm to GPU once for fast teacher logit computation
    d_restricted_embed = restricted_embed.to(device)
    d_output_norm_weight = output_norm_weight.to(device)

    def compute_teacher_logits_gpu(hidden_gpu: torch.Tensor) -> torch.Tensor:
        """RMSNorm(hidden) @ restricted_embed.T on GPU → teacher logits [*, K]."""
        h = hidden_gpu.float()
        rms = h.pow(2).mean(-1, keepdim=True).add(output_norm_eps).rsqrt()
        normed = ((h * rms) * d_output_norm_weight).half()
        return normed @ d_restricted_embed.T  # [*, K]

    def build_teacher_probs_from_p_idk(teacher_logits_gpu, p_idk_gpu, T):
        """Build K+1 teacher distribution from restricted logits + p_idk.

        teacher[:K] = softmax(logits/T) * (1 - p_idk)
        teacher[K]  = p_idk
        """
        p_restricted = F.softmax(teacher_logits_gpu.float() / T, dim=-1)
        p_restricted = p_restricted * (1.0 - p_idk_gpu.unsqueeze(-1))
        return torch.cat([p_restricted, p_idk_gpu.unsqueeze(-1)], dim=-1)

    K = top_k

    with torch.no_grad():
        if val_cache_files:
            for vf in tqdm(val_cache_files, desc="eval", unit="batch",
                           file=sys.stderr, leave=False):
                vc = torch.load(str(vf), map_location="cpu", weights_only=True)
                vh_all = vc["hidden_input"]  # [B, L-1, n_embed] (positions 0..L-2)
                vtids = vc["token_ids"]
                vB, vL = vtids.shape
                ve = F.embedding(vtids, embed_weight)[:, 1:vL-1]  # embed[t+1]
                vh_mtp = vh_all[:, :vL-2]     # hidden[t] for MTP input (positions 0..L-3)
                vh_teacher = vh_all[:, 1:vL-1] # hidden[t+1] for teacher logits (positions 1..L-2)

                # v4 val cache includes p_idk
                has_p_idk = "p_idk" in vc
                if idk and has_p_idk:
                    vc_p_idk = vc["p_idk"]  # [B, L-2]
                elif idk and not has_p_idk:
                    raise RuntimeError(
                        "IDK mode requires v4 val cache with p_idk. "
                        "Re-extract with: uv run scripts/02_extract_val_cache.py --p-idk ..."
                    )

                # Chunk to bound GPU memory
                seq_bytes = max(ve.shape[1], 1) * top_k * 4
                chunk = max(1, min(vB, int(500_000_000 / seq_bytes)))
                for ci in range(0, vB, chunk):
                    ce = min(ci + chunk, vB)
                    vh_mtp_gpu = vh_mtp[ci:ce].to(device, dtype=torch.float16, non_blocking=True)
                    vh_teacher_gpu = vh_teacher[ci:ce].to(device, dtype=torch.float16, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        vlogits = mtp(
                            ve[ci:ce].to(device, non_blocking=True),
                            vh_mtp_gpu,
                        )

                        # Recompute teacher logits from cached hidden states
                        vteacher_logits = compute_teacher_logits_gpu(vh_teacher_gpu)

                        if idk:
                            vp_idk = vc_p_idk[ci:ce].to(device, non_blocking=True)
                            vteacher_probs = build_teacher_probs_from_p_idk(
                                vteacher_logits, vp_idk, temperature)
                            vloss = distillation_loss_idk(vlogits, vteacher_probs, T=temperature)
                        else:
                            vloss = distillation_loss(vlogits, vteacher_logits, T=temperature)

                    total_loss += vloss.item()

                    if idk:
                        vpreds = vlogits.argmax(dim=-1)  # [B, L]
                        vteacher_argmax = vteacher_probs.argmax(dim=-1)  # [B, L]

                        student_idk = (vpreds == K)
                        teacher_oov = (vteacher_argmax == K)
                        total_idk_preds += student_idk.sum().item()
                        total_teacher_oov += teacher_oov.sum().item()
                        total_idk_correct += (student_idk & teacher_oov).sum().item()

                        non_idk = ~student_idk
                        total_non_idk_accept += (vpreds[non_idk] == vteacher_argmax[non_idk]).sum().item()
                        total_non_idk_tokens += non_idk.sum().item()

                        # Overall accept: non-IDK match + correct IDK abstentions
                        total_accept += (vpreds[non_idk] == vteacher_argmax[non_idk]).sum().item()
                        total_accept += (student_idk & teacher_oov).sum().item()
                    else:
                        vpreds = vlogits.argmax(dim=-1)
                        vteacher_argmax = vteacher_logits.argmax(dim=-1)
                        total_accept += (vpreds == vteacher_argmax).sum().item()

                    total_tokens += vpreds.numel()
                    n_batches += 1
        elif val_dl is not None and gwen is not None:
            for vbatch in val_dl:
                vtids = vbatch["token_ids"]
                vB, vL = vtids.shape
                ve = F.embedding(vtids, embed_weight)[:, 1:vL-1]
                vh_full, vteacher_full = gwen.batch_logits(vtids)
                vh = vh_full[:, :vL-2]
                vteacher = vteacher_full[:, 1:vL-1]  # shifted: teacher[t+1] predicts token[t+2]
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    vlogits = mtp(ve.to(device), vh.to(device, dtype=torch.float16))
                    vloss = distillation_loss(
                        vlogits,
                        vteacher.to(device, non_blocking=True),
                        T=temperature,
                    )
                total_loss += vloss.item()
                vpreds = vlogits.argmax(dim=-1).cpu()
                vteacher_argmax = vteacher.argmax(dim=-1)
                total_accept += (vpreds == vteacher_argmax).sum().item()
                total_tokens += vpreds.numel()
                n_batches += 1

    val_loss = total_loss / max(n_batches, 1)
    accept_rate = total_accept / max(total_tokens, 1)

    if idk:
        p_idk = total_idk_preds / max(total_tokens, 1)
        p_accept = total_non_idk_accept / max(total_non_idk_tokens, 1)
        # Estimated ms/tok from cycle timing model:
        #   IDK skip: 1.68ms/tok, accept: 1.125ms/tok, reject: 2.30ms/tok
        ms_per_tok = (p_idk * 1.68 +
                      (1 - p_idk) * (p_accept * 1.125 + (1 - p_accept) * 2.30))
        idk_metrics = {
            "idk_rate": p_idk,
            "idk_precision": total_idk_correct / max(total_idk_preds, 1),
            "idk_recall": total_idk_correct / max(total_teacher_oov, 1),
            "accept_rate_non_idk": p_accept,
            "accept_rate_overall": accept_rate,
            "est_tok_per_sec": 1000.0 / ms_per_tok,
        }
        return val_loss, accept_rate, total_tokens, idk_metrics

    return val_loss, accept_rate, total_tokens


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

    # Compute total tokens for progress tracking
    train_tokens_per_epoch = sum(train_ds.lengths)
    total_train_tokens = train_tokens_per_epoch * args.epochs

    log(f"  Train: {len(train_ds):,} sequences ({train_tokens_per_epoch / 1e6:.0f}M tokens/epoch)")
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
        train_ds, batch_sampler=train_sampler, collate_fn=mtp_collate,
        num_workers=n_workers, pin_memory=pin, persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=mtp_collate,
        num_workers=n_workers, pin_memory=pin, persistent_workers=True,
    )

    # Embeddings
    embed_weight = load_embeddings(Path(args.model_dir), device)

    # GWEN server
    use_idk = args.idk
    host, port = args.server_url.replace("http://", "").split(":")
    gwen = GwenClient(host, int(port), use_shm=use_idk)

    # MTP head
    log("Initializing MTP head..." + (" (IDK mode)" if use_idk else ""))
    if args.from_pretrained:
        mtp = MTPHead.from_pretrained(args.model_dir, vocab_size=args.top_k, device=device,
                                       vocab_ids=vocab.top_k_ids.tolist(), idk=use_idk)
    else:
        mtp = MTPHead(vocab_size=args.top_k, idk=use_idk).to(device)
        log("  Initialized from scratch (no pre-trained weights)")

    # Warm-start from existing checkpoint (e.g. v3 → v4 IDK)
    if args.init_from:
        init_ckpt = torch.load(str(args.init_from), map_location=device, weights_only=False)
        init_sd = init_ckpt["model"]
        current_sd = mtp.state_dict()
        loaded_keys = 0
        for key in init_sd:
            if key in current_sd:
                if init_sd[key].shape == current_sd[key].shape:
                    current_sd[key] = init_sd[key]
                    loaded_keys += 1
                elif key == "lm_head.weight" and use_idk:
                    # v3 [K, 1024] → v4 [K+1, 1024]: copy K rows, leave IDK row as zeros
                    K_src = init_sd[key].shape[0]
                    current_sd[key][:K_src] = init_sd[key]
                    loaded_keys += 1
                    log(f"  lm_head: copied {K_src} rows from init checkpoint, IDK row = zeros")
                else:
                    log(f"  Warning: shape mismatch for {key}: "
                        f"{init_sd[key].shape} vs {current_sd[key].shape}, skipping")
        mtp.load_state_dict(current_sd)
        log(f"  Warm-started from {args.init_from} ({loaded_keys}/{len(init_sd)} keys loaded)")

    params = mtp.param_count()
    log(f"  MTP params: {params['total']:,} ({params['total'] * 4 / 1e6:.1f} MB FP32)")
    for k, v in params.items():
        if k != "total" and v > 0:
            log(f"    {k}: {v:,}")

    # Two-stage training setup
    # Stage 1: train lm_head only (freshly initialized, needs warmup)
    # Stage 2: unfreeze all params
    stage1_steps = args.stage1_steps
    T = args.temperature
    # IDK mode: temperature decay T_start → T_end over training
    T_start = args.temp_start if use_idk else T
    T_end = args.temp_end if use_idk else T

    # Restricted embed + output norm for on-the-fly teacher logit computation
    # (used by val cache eval path — recompute teacher logits from cached hidden states)
    restricted_embed = embed_weight[vocab.top_k_ids.tolist()].clone()  # [K, 1024] FP16
    log(f"  Restricted embed: [{restricted_embed.shape[0]}, {restricted_embed.shape[1]}] FP16")

    # Load output norm weight from safetensors
    output_norm_weight = _load_output_norm(Path(args.model_dir))  # [1024] F32
    output_norm_eps = 1e-6

    # IDK mode: p_idk is computed live by dev_server (no full embed needed on training GPU)
    if use_idk:
        log("  IDK mode: p_idk computed live by dev_server (/batch_logits?p_idk=1)")

    # Build optimizer — stage 1 trains only lm_head
    def make_optimizer(stage: int) -> torch.optim.Optimizer:
        if stage == 1:
            lr = args.stage1_lr
            params_to_train = list(mtp.lm_head.parameters())
            # Freeze everything except lm_head
            for p in mtp.parameters():
                p.requires_grad = False
            for p in mtp.lm_head.parameters():
                p.requires_grad = True
        else:
            lr = args.lr
            # Unfreeze all params
            for p in mtp.parameters():
                p.requires_grad = True
            params_to_train = list(mtp.parameters())
        return torch.optim.AdamW(
            params_to_train, lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
        )

    # Scheduler: linear warmup + cosine decay to min_lr
    total_steps = len(train_sampler) * args.epochs
    warmup_steps = args.warmup_steps
    min_lr_ratio = 0.1  # Floor at 10% of peak lr (avoid wasting training at near-zero lr)

    def make_scheduler(optimizer, total_steps, warmup_steps):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Start in stage 1 (lm_head warmup) or stage 2 directly
    current_stage = 1 if stage1_steps > 0 else 2
    optimizer = make_optimizer(current_stage)
    if current_stage == 1:
        scheduler = make_scheduler(optimizer, stage1_steps, warmup_steps)
        log(f"Stage 1: training lm_head only for {stage1_steps} steps at lr={args.stage1_lr}")
    else:
        scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    # AMP — FP16 autocast + GradScaler (model stays FP32, autocast handles FP16 forward)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def transition_to_stage2():
        nonlocal current_stage, optimizer, scheduler, scaler
        current_stage = 2
        optimizer = make_optimizer(2)
        stage2_total = len(train_sampler) * args.epochs
        scheduler = make_scheduler(optimizer, stage2_total, warmup_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        log(f"\n=== Stage 2: unfreezing all params, lr={args.lr} ===")

    # Resume
    start_epoch = 0
    global_step = 0
    tokens_seen = 0
    best_val = float("inf")
    best_accept = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        mtp.load_state_dict(ckpt["model"])
        global_step = ckpt.get("global_step", 0)
        tokens_seen = ckpt.get("tokens_seen", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        best_accept = ckpt.get("accept_rate", 0.0)
        log(f"Resumed from {args.resume} (step={global_step}, val_loss={best_val:.4f}, accept={best_accept:.1%})")

        # If we've passed stage1_steps, ensure we're in stage 2
        if current_stage == 1 and global_step >= stage1_steps:
            transition_to_stage2()

        # Restore optimizer/scheduler if same stage
        ckpt_stage = ckpt.get("stage", 1)
        if ckpt_stage == current_stage:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

    # Save setup
    data_path = Path(args.data).resolve()
    counts_path = Path(args.counts).resolve()
    log("Computing file checksums...")
    data_sha = sha256_file(data_path)
    counts_sha = sha256_file(counts_path)

    setup = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": sys.argv,
        "data": {"path": str(data_path), "sha256": data_sha, "size_bytes": data_path.stat().st_size},
        "counts": {"path": str(counts_path), "sha256": counts_sha},
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
            "version": 4 if use_idk else 3,
            "loss": "kl_divergence_idk" if use_idk else "kl_divergence",
            "idk": use_idk,
            "temperature": T,
            "temp_start": T_start if use_idk else None,
            "temp_end": T_end if use_idk else None,
            "stage1_steps": stage1_steps,
            "stage1_lr": args.stage1_lr,
            "epochs": args.epochs,
            "max_tokens": args.max_tokens,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "min_lr": args.lr * min_lr_ratio,
            "weight_decay": args.weight_decay,
            "warmup_steps": warmup_steps,
            "grad_clip": args.grad_clip,
            "hidden_noise": args.hidden_noise,
            "patience": args.patience,
            "amp_dtype": "float16",
            "from_pretrained": args.from_pretrained,
            "val_fraction": args.val_fraction,
        },
        "hyperparameter_sources": {
            "lr": "EAGLE uses 3e-5, scaled up for smaller model",
            "stage1_lr": "IBM Recurrent Drafter stage 1",
            "temperature": "Hinton distillation T=2.0",
            "warmup": "~4% of epoch",
            "grad_clip": "EAGLE uses 0.5",
            "hidden_noise": "EAGLE adds noise to input features for regularization",
        },
        "vocab": {"k": args.top_k, "coverage": vocab.coverage},
        "data_stats": {
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "train_batches": len(train_sampler),
            "val_batches": len(val_sampler),
            "train_tokens_per_epoch": train_tokens_per_epoch,
            "total_train_tokens": total_train_tokens,
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
        header = "step,epoch,tokens_seen,batch_loss,avg_loss,val_loss,accept_rate,lr"
        if use_idk:
            header += ",idk_rate,idk_precision,idk_recall,accept_rate_non_idk"
        log_file.write(header + "\n")
        log_file.flush()

    # Load val cache
    val_cache_dir = out_dir / "val_cache"
    val_cache_files = sorted(val_cache_dir.glob("*.pt")) if val_cache_dir.exists() else []
    if val_cache_files and args.eval_batches > 0 and args.eval_batches < len(val_cache_files):
        # Deterministic subsample for faster eval
        step = len(val_cache_files) // args.eval_batches
        val_cache_files = val_cache_files[::step][:args.eval_batches]
    if val_cache_files:
        log(f"Val cache: {len(val_cache_files)} batches in {val_cache_dir}")
    else:
        log(f"WARNING: No val cache found at {val_cache_dir}")
        log(f"  Falling back to live server eval (slow)")

    # --- Initial eval (baseline before any training) ---
    eval_kwargs = dict(
        vocab=vocab, embed_weight=embed_weight, top_k=args.top_k,
        device=device, use_amp=use_amp, temperature=T,
        restricted_embed=restricted_embed, output_norm_weight=output_norm_weight,
        output_norm_eps=output_norm_eps,
        idk=use_idk,
    )

    if (args.init_from and use_idk) or args.resume:
        # Skip initial eval: meaningless when IDK is zeros (init-from), or redundant (resume)
        log("\nSkipping initial eval")
    else:
        log("\nRunning initial eval (baseline)...")
        eval_result = run_eval(
            mtp, val_cache_files, val_dl if not val_cache_files else None,
            gwen if not val_cache_files else None, **eval_kwargs,
        )
        if use_idk:
            val_loss, accept_rate, _, idk_m = eval_result
            log(f"  Baseline: val_loss={val_loss:.4f}, accept={accept_rate:.1%}")
            log(f"  IDK: rate={idk_m['idk_rate']:.1%}, prec={idk_m['idk_precision']:.1%}, "
                f"rec={idk_m['idk_recall']:.1%}, non-idk accept={idk_m['accept_rate_non_idk']:.1%}")
            log_file.write(f"0,0.0000,0,,,{val_loss:.6f},{accept_rate:.4f},{args.lr:.6e},"
                           f"{idk_m['idk_rate']:.4f},{idk_m['idk_precision']:.4f},"
                           f"{idk_m['idk_recall']:.4f},{idk_m['accept_rate_non_idk']:.4f}\n")
        else:
            val_loss, accept_rate, _ = eval_result
            log(f"  Baseline: val_loss={val_loss:.4f}, accept={accept_rate:.1%}")
            log_file.write(f"0,0.0000,0,,,{val_loss:.6f},{accept_rate:.4f},{args.lr:.6e}\n")
        log_file.flush()

    if not ((args.init_from and use_idk) or args.resume):
        if val_loss < best_val:
            best_val = val_loss
        if accept_rate > best_accept:
            best_accept = accept_rate

    # Early stopping
    patience_counter = 0
    patience_limit = args.patience

    # --- Training loop ---
    total_epochs = args.epochs
    stage1_msg = f", stage 1 for first {stage1_steps} steps" if stage1_steps > 0 else ""
    log(f"\nStarting training: {total_epochs} epochs{stage1_msg}")
    log(f"Loss: KL divergence (T={T})")
    if args.hidden_noise > 0:
        log(f"Hidden state noise: U(-{args.hidden_noise}, {args.hidden_noise})")
    if patience_limit > 0:
        log(f"Early stopping: patience={patience_limit} evals")

    t_start = time.time()
    tokens_at_start = tokens_seen

    def save_checkpoint(val_loss, accept_rate, epoch, idk_metrics=None):
        nonlocal best_val, best_accept, patience_counter
        ckpt = {
            "model": mtp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "val_loss": val_loss,
            "accept_rate": accept_rate,
            "vocab_k": args.top_k,
            "temperature": T,
            "stage": current_stage,
            "timestamp": time.time(),
            "has_idk": use_idk,
        }
        torch.save(ckpt, out_dir / "latest.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            log(f"  -> saved best (val={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # For IDK: save best by estimated tok/s; otherwise by accept_rate
        if use_idk and idk_metrics:
            score = idk_metrics["est_tok_per_sec"]
            if score > best_accept:
                best_accept = score
                torch.save(ckpt, out_dir / "best_accept.pt")
                log(f"  -> saved best_accept (est {score:.0f} tok/s)")
        else:
            if accept_rate > best_accept:
                best_accept = accept_rate
                torch.save(ckpt, out_dir / "best_accept.pt")
                log(f"  -> saved best_accept ({accept_rate:.1%})")

    for epoch in range(start_epoch, total_epochs):

        stage_label = f"S{current_stage}"
        train_sampler.set_epoch(epoch)
        mtp.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        n_batches = 0
        steps_since_eval = 0

        pbar = tqdm(
            total=train_tokens_per_epoch / 1e6, desc=f"[{stage_label}] Epoch {epoch+1}/{total_epochs}",
            unit="MTok", bar_format="{l_bar}{bar}| {n:.1f}/{total:.0f} MTok [{elapsed}<{remaining} {postfix}]",
            file=sys.stderr, dynamic_ncols=True,
        )
        batch_iter = iter(train_dl)

        for batch in batch_iter:
            token_ids = batch["token_ids"]
            B, L = token_ids.shape
            batch_tokens = B * L

            # Embeddings: CPU lookup
            with torch.no_grad():
                embeddings = F.embedding(token_ids, embed_weight)

            if use_idk:
                # IDK mode: get hidden + logits + p_idk from dev server in one call
                with torch.no_grad():
                    hidden, teacher_logits, batch_p_idk = gwen.batch_logits_with_p_idk(token_ids)

                embed_input = embeddings[:, 1:L-1]           # [B, L-2, 1024]
                hidden_input = hidden[:, :L-2]                # [B, L-2, 1024]
                teacher_input = teacher_logits[:, 1:L-1]      # [B, L-2, K] — shifted
                teacher_p_idk = batch_p_idk[:, 1:L-1]         # [B, L-2] — shifted
            else:
                # Standard mode: get hidden states + teacher logits from dev server
                with torch.no_grad():
                    hidden, teacher_logits = gwen.batch_logits(token_ids)

                embed_input = embeddings[:, 1:L-1]      # [B, L-2, 1024]
                hidden_input = hidden[:, :L-2]            # [B, L-2, 1024]
                teacher_input = teacher_logits[:, 1:L-1]  # [B, L-2, K] — shifted

            # Hidden state noise injection (EAGLE regularization technique)
            if args.hidden_noise > 0 and mtp.training:
                noise = torch.empty_like(hidden_input).uniform_(-args.hidden_noise, args.hidden_noise)
                hidden_input = hidden_input + noise

            # IDK temperature decay: linear interpolation over training steps
            if use_idk:
                progress = global_step / max(total_steps, 1)
                T_cur = T_start + (T_end - T_start) * progress
            else:
                T_cur = T

            # Chunked forward/backward to bound GPU memory
            # IDK needs more memory: teacher probs (F32) + student logits + KL intermediates
            lm_out_size = args.top_k + 1 if use_idk else args.top_k
            mem_budget = 150_000_000 if use_idk else 500_000_000
            seq_logit_bytes = max(L - 2, 1) * lm_out_size * 4
            max_chunk = max(1, min(B, int(mem_budget / seq_logit_bytes)))
            n_chunks = (B + max_chunk - 1) // max_chunk
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            skip = False

            for ci in range(0, B, max_chunk):
                ce = min(ci + max_chunk, B)
                e_chunk = embed_input[ci:ce].to(device, non_blocking=True)
                h_chunk = hidden_input[ci:ce].to(device, dtype=torch.float16, non_blocking=True)

                if use_idk:
                    # Build K+1 teacher distribution from server logits + p_idk
                    t_logits = teacher_input[ci:ce].to(device, non_blocking=True)
                    p_idk_chunk = teacher_p_idk[ci:ce].to(device, non_blocking=True)
                    with torch.no_grad():
                        p_restricted = F.softmax(t_logits.float() / T_cur, dim=-1)
                        p_restricted = p_restricted * (1.0 - p_idk_chunk.unsqueeze(-1))
                        t_chunk = torch.cat([p_restricted, p_idk_chunk.unsqueeze(-1)], dim=-1)
                else:
                    t_chunk = teacher_input[ci:ce].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = mtp(e_chunk, h_chunk)
                    if use_idk:
                        loss = distillation_loss_idk(logits, t_chunk, T=T_cur)
                    else:
                        loss = distillation_loss(logits, t_chunk, T=T_cur)
                    loss = loss / n_chunks

                if torch.isnan(loss) or torch.isinf(loss):
                    log(f"  WARNING: {loss.item()} loss at step {global_step}, skipping")
                    skip = True
                    break

                scaler.scale(loss).backward()
                batch_loss += loss.item()

                # Track batch IDK rate (cheap — reuses logits already computed)
                if use_idk:
                    with torch.no_grad():
                        batch_idk_count = (logits.detach().argmax(dim=-1) == args.top_k).sum().item()
                        batch_idk_total = logits.shape[0] * logits.shape[1] if logits.dim() == 3 else logits.shape[0]

            if skip:
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(mtp.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()

            epoch_loss += batch_loss
            epoch_tokens += batch_tokens
            tokens_seen += batch_tokens
            steps_since_eval += 1
            n_batches += 1
            global_step += 1

            # Stage transition: step-based
            if current_stage == 1 and global_step >= stage1_steps:
                transition_to_stage2()
                stage_label = f"S{current_stage}"
                pbar.set_description(f"[{stage_label}] Epoch {epoch+1}/{total_epochs}")

            # Progress bar: tokens + steps
            avg = epoch_loss / n_batches
            lr_val = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            tok_per_sec = (tokens_seen - tokens_at_start) / max(elapsed, 1)
            pbar.update(batch_tokens / 1e6)
            if use_idk:
                batch_idk_rate = batch_idk_count / max(batch_idk_total, 1)
                pbar.set_postfix_str(
                    f"step {global_step}, loss={avg:.4f} lr={lr_val:.1e} "
                    f"idk={batch_idk_rate:.1%} ({tok_per_sec/1e3:.1f}K tok/s)"
                )
            else:
                pbar.set_postfix_str(
                    f"step {global_step}, loss={avg:.4f} lr={lr_val:.1e} "
                    f"({tok_per_sec/1e3:.1f}K tok/s)"
                )

            # CSV: every batch
            frac_epoch = epoch + n_batches / len(train_sampler)
            csv_line = (f"{global_step},{frac_epoch:.4f},{tokens_seen},"
                        f"{batch_loss:.6f},{avg:.6f},,,{lr_val:.6e}")
            if use_idk:
                csv_line += f",{batch_idk_rate:.4f},,,"
            log_file.write(csv_line + "\n")
            if global_step % 100 == 0:
                log_file.flush()

            # --- Periodic save (cheap, no eval) ---
            if global_step % 50 == 0:
                torch.save({
                    "model": mtp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "tokens_seen": tokens_seen,
                    "val_loss": best_val,
                    "accept_rate": best_accept,
                    "vocab_k": args.top_k,
                    "temperature": T,
                    "stage": current_stage,
                    "timestamp": time.time(),
                    "has_idk": use_idk,
                }, out_dir / "latest.pt")

            # --- Periodic eval ---
            if steps_since_eval >= args.eval_every:
                # Update eval temperature for IDK mode
                if use_idk:
                    eval_kwargs["temperature"] = T_cur

                eval_result = run_eval(
                    mtp, val_cache_files, val_dl if not val_cache_files else None,
                    gwen if not val_cache_files else None, **eval_kwargs,
                )

                if use_idk:
                    val_loss, accept_rate, _, idk_m = eval_result
                    log(f"  [{stage_label}:{epoch+1}:{n_batches}/{len(train_sampler)}]  "
                        f"train={avg:.4f}  val={val_loss:.4f}  accept={accept_rate:.1%}  "
                        f"lr={lr_val:.1e}  T={T_cur:.2f}  tokens={tokens_seen/1e6:.0f}M")
                    log(f"    IDK: rate={idk_m['idk_rate']:.1%}, prec={idk_m['idk_precision']:.1%}, "
                        f"rec={idk_m['idk_recall']:.1%}, non-idk accept={idk_m['accept_rate_non_idk']:.1%}, "
                        f"est {idk_m['est_tok_per_sec']:.0f} tok/s")
                    log_file.write(f"{global_step},{frac_epoch:.4f},{tokens_seen},"
                                   f"{avg:.6f},{avg:.6f},{val_loss:.6f},{accept_rate:.4f},{lr_val:.6e},"
                                   f"{idk_m['idk_rate']:.4f},{idk_m['idk_precision']:.4f},"
                                   f"{idk_m['idk_recall']:.4f},{idk_m['accept_rate_non_idk']:.4f}\n")
                else:
                    val_loss, accept_rate, _ = eval_result
                    log(f"  [{stage_label}:{epoch+1}:{n_batches}/{len(train_sampler)}]  "
                        f"train={avg:.4f}  val={val_loss:.4f}  accept={accept_rate:.1%}  "
                        f"lr={lr_val:.1e}  tokens={tokens_seen/1e6:.0f}M")
                    log_file.write(f"{global_step},{frac_epoch:.4f},{tokens_seen},"
                                   f"{avg:.6f},{avg:.6f},{val_loss:.6f},{accept_rate:.4f},{lr_val:.6e}\n")
                log_file.flush()

                save_checkpoint(val_loss, accept_rate, epoch,
                                idk_metrics=idk_m if use_idk else None)

                steps_since_eval = 0

                # Early stopping (only in stage 2)
                if current_stage == 2 and patience_limit > 0 and patience_counter >= patience_limit:
                    log(f"\nEarly stopping: val_loss hasn't improved for {patience_limit} evals")
                    break

                mtp.train()

        pbar.close()

        # Check if early stopped mid-epoch
        if current_stage == 2 and patience_limit > 0 and patience_counter >= patience_limit:
            break

        # --- End-of-epoch eval ---
        avg_train = epoch_loss / max(n_batches, 1)

        if use_idk:
            eval_kwargs["temperature"] = T_cur

        eval_result = run_eval(
            mtp, val_cache_files, val_dl if not val_cache_files else None,
            gwen if not val_cache_files else None, **eval_kwargs,
        )

        lr_val = optimizer.param_groups[0]["lr"]

        if use_idk:
            val_loss, accept_rate, _, idk_m = eval_result
            log(f"[{stage_label}] Epoch {epoch+1}/{total_epochs} done: "
                f"train={avg_train:.4f}  val={val_loss:.4f}  accept={accept_rate:.1%}  "
                f"tokens={tokens_seen/1e6:.0f}M")
            log(f"  IDK: rate={idk_m['idk_rate']:.1%}, prec={idk_m['idk_precision']:.1%}, "
                f"rec={idk_m['idk_recall']:.1%}, non-idk accept={idk_m['accept_rate_non_idk']:.1%}, "
                f"est {idk_m['est_tok_per_sec']:.0f} tok/s")
            log_file.write(f"{global_step},{epoch+1:.4f},{tokens_seen},"
                           f"{avg_train:.6f},{avg_train:.6f},{val_loss:.6f},{accept_rate:.4f},{lr_val:.6e},"
                           f"{idk_m['idk_rate']:.4f},{idk_m['idk_precision']:.4f},"
                           f"{idk_m['idk_recall']:.4f},{idk_m['accept_rate_non_idk']:.4f}\n")
        else:
            val_loss, accept_rate, _ = eval_result
            log(f"[{stage_label}] Epoch {epoch+1}/{total_epochs} done: "
                f"train={avg_train:.4f}  val={val_loss:.4f}  accept={accept_rate:.1%}  "
                f"tokens={tokens_seen/1e6:.0f}M")
            log_file.write(f"{global_step},{epoch+1:.4f},{tokens_seen},"
                           f"{avg_train:.6f},{avg_train:.6f},{val_loss:.6f},{accept_rate:.4f},{lr_val:.6e}\n")
        log_file.flush()

        save_checkpoint(val_loss, accept_rate, epoch,
                        idk_metrics=idk_m if use_idk else None)
        mtp.train()

    elapsed = time.time() - t_start
    log_file.close()
    if use_idk:
        log(f"\nTraining complete in {elapsed/60:.1f} min. "
            f"Best val_loss={best_val:.4f}, best est_tok_s={best_accept:.0f}")
    else:
        log(f"\nTraining complete in {elapsed/60:.1f} min. "
            f"Best val_loss={best_val:.4f}, best accept={best_accept:.1%}")
    log(f"Checkpoints: {out_dir}/best.pt, {out_dir}/best_accept.pt, {out_dir}/latest.pt")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args: argparse.Namespace) -> None:
    """Export trained MTP weights to GWMT v3/v4 format for GWEN inference.

    GWMT v3 format:
        Header: "GWMT" [uint32 version=3] [uint32 n_tensors]
        Tensors: for each tensor: [name_len][name][dtype][ndims][shape...][data_size][data]
        Footer: [uint32 K] [int32 restricted_ids[K]]

    GWMT v4 format (IDK):
        Same as v3, but lm_head is [K+1, 1024] and footer has extra [uint8 has_idk=1]
    """

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    vocab_k = ckpt.get("vocab_k", 20000)
    has_idk = ckpt.get("has_idk", False)

    output = Path(args.output) if args.output else ckpt_path.parent / "mtp_finetuned.bin"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load restricted vocab IDs for the footer
    counts_path = Path(args.counts)
    vocab = RestrictedVocab(counts_path, vocab_k)
    restricted_ids = vocab.top_k_ids  # [K] int32, frequency-sorted

    DTYPE_F32 = 0
    DTYPE_F16 = 1

    # Version: 4 if IDK, 3 otherwise
    version = 4 if has_idk else 3

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

    if "lm_head.weight" in state_dict:
        lm_data = state_dict["lm_head.weight"].half().numpy()
        tensors.append(("mtp.lm_head.weight", lm_data, DTYPE_F16))

    log(f"Exporting {len(tensors)} tensors to {output} (GWMT v{version}, K={vocab_k}"
        f"{', IDK' if has_idk else ''})")

    with open(output, "wb") as f:
        f.write(b"GWMT")
        f.write(struct.pack("<I", version))
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

        # Footer: restricted vocab mapping
        f.write(struct.pack("<I", vocab_k))
        f.write(restricted_ids.astype(np.int32).tobytes())
        log(f"  restricted_ids                                        [{vocab_k}]              "
            f"{vocab_k * 4 / 1024:.1f} KB")

        # v4 footer: has_idk flag
        if version >= 4:
            f.write(struct.pack("<B", 1))  # has_idk = true
            log(f"  has_idk                                               1")

    file_size = output.stat().st_size
    log(f"\nSaved: {output} ({file_size / 1024 / 1024:.2f} MB)")
    accept = ckpt.get('accept_rate', None)
    accept_str = f"{accept:.1%}" if accept is not None else "?"
    log(f"  Vocab K={vocab_k}, epoch={ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', '?'):.4f}, accept_rate={accept_str}"
        f"{', IDK=true' if has_idk else ''}")


# ---------------------------------------------------------------------------
# Pre-compute IDK cache
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MTP head for restricted-vocab English")
    sub = parser.add_subparsers(dest="cmd")

    # --- train ---
    p = sub.add_parser("train", help="Train MTP head with KL distillation (v3)")
    p.add_argument("--data", type=Path, required=True, help="Tokenized data (uint32 binary)")
    p.add_argument("--counts", type=Path, default=Path("data/token_counts.bin"))
    p.add_argument("--model-dir", type=Path, default=Path.home() / "models" / "hf" / "Qwen3.5-0.8B")
    p.add_argument("--server-url", type=str, default="http://127.0.0.1:8090",
                    help="Dev server URL (gwen_dev_server with --restricted-vocab)")
    p.add_argument("--out-dir", type=Path, default=Path("train/runs/mtp_v3"))
    p.add_argument("--top-k", type=int, default=4096, help="Restricted vocab size")
    p.add_argument("--epochs", type=int, default=1,
                    help="Training epochs (1 pass over ~500M tokens is sufficient)")
    p.add_argument("--stage1-steps", type=int, default=1000,
                    help="Stage 1 steps (lm_head warmup only, 0 to skip)")
    p.add_argument("--stage1-lr", type=float, default=1e-3,
                    help="Stage 1 learning rate (IBM Recurrent Drafter)")
    p.add_argument("--max-tokens", type=int, default=32768,
                    help="Token budget per batch")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Stage 2 peak learning rate")
    p.add_argument("--temperature", type=float, default=2.0,
                    help="Distillation temperature (Hinton default)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5,
                    help="Gradient clipping (EAGLE uses 0.5)")
    p.add_argument("--warmup-steps", type=int, default=150,
                    help="Linear warmup steps per stage (~4%% of epoch)")
    p.add_argument("--hidden-noise", type=float, default=0.05,
                    help="Uniform noise on hidden state inputs (EAGLE regularization)")
    p.add_argument("--patience", type=int, default=3,
                    help="Early stopping patience (evals without val_loss improvement)")
    p.add_argument("--eval-every", type=int, default=1000,
                    help="Eval + checkpoint every N optimizer steps")
    p.add_argument("--eval-batches", type=int, default=0,
                    help="Limit eval to N val cache batches (0 = use all)")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--from-pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="from_pretrained", action="store_false")
    p.add_argument("--resume", type=Path, help="Resume from checkpoint")
    p.add_argument("--idk", action="store_true", default=False,
                    help="Enable IDK token (K+1 output, absorbs OOV mass)")
    p.add_argument("--temp-start", type=float, default=2.0,
                    help="IDK mode: starting temperature (decays linearly)")
    p.add_argument("--temp-end", type=float, default=1.0,
                    help="IDK mode: ending temperature")
    p.add_argument("--init-from", type=Path, default=None,
                    help="Initialize weights from a checkpoint (e.g. v3 best.pt for IDK warm-start)")

    # --- export ---
    p = sub.add_parser("export", help="Export trained MTP to GWMT v3 binary")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--counts", type=Path, default=Path("data/token_counts.bin"),
                    help="Token frequency counts (for restricted vocab ID mapping)")
    p.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "export":
        export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
