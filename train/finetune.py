#!/usr/bin/env python3
"""Fine-tune Qwen3.5-0.8B for transcript correction.

Full bf16 fine-tune with Muon (2D layers) + AdamW (embeddings/norms).
Token-budget batching: fixed total tokens per batch, not fixed example count.

Usage:
    uv run python finetune.py train
    uv run python finetune.py train --epochs 3 --lr 2e-5
    uv run python finetune.py train --resume runs/finetune_.../best
    uv run python finetune.py eval --checkpoint runs/finetune_.../best
    uv run python finetune.py export --checkpoint runs/finetune_.../best --output model.gguf
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = (
    "You are a speech transcript editor. "
    "Remove filler words, fix repetitions, add punctuation. "
    "Do not change the meaning. Output only the cleaned text."
)

EVAL_SAMPLES = [
    "um so like the thing is uh we need to make sure that the the model is like loading properly and stuff you know",
    "I was going to I was going to say that the the results look pretty good actually they look really good",
    "so basically what happened was the server went down at around 3 am and nobody noticed until the morning and by then we had lost about six hours of data",
    "yeah so uh I think I think what we should probably do is uh maybe look at the the latency numbers again because uh they seemed a bit off to me",
    "the cuda kernel is launching like sixty four threads per block and uh we need at least uh one twenty eight to get full occupancy on the on the sm",
    "uh yeah that sounds good lets do that",
    "we are getting about uh six hundred and thirty tokens per second which is uh roughly uh thirty percent faster than the baseline",
]


# ── Data ─────────────────────────────────────────────────────────────────────


def format_chatml(src: str, tgt: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": src},
        {"role": "assistant", "content": "<think>\n</think>\n\n" + tgt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class TranscriptDataset(Dataset):
    """Pre-tokenizes everything and stores lengths for batching.
    Caches tokenized data to disk as .pt next to the source JSONL."""

    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        cache_path = path.with_suffix(".tok.pt")

        if cache_path.exists():
            print(f"  Loading cached tokens from {cache_path.name}")
            self.items = torch.load(cache_path, weights_only=False)
        else:
            self.items = self._tokenize(path, tokenizer, max_len)
            torch.save(self.items, cache_path)
            print(f"  Saved token cache → {cache_path.name}")

        self.items.sort(key=lambda x: x["length"])
        lengths = [it["length"] for it in self.items]
        print(f"  {len(self.items)} examples, "
              f"len: min={min(lengths)} med={lengths[len(lengths)//2]} max={max(lengths)}")

    @staticmethod
    def _tokenize(path: Path, tokenizer, max_len: int) -> list[dict]:
        items = []
        assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        marker_len = len(assistant_marker)

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Tokenizing", leave=False):
            row = json.loads(line)
            text = format_chatml(row["src"], row["tgt"], tokenizer)
            enc = tokenizer(text, max_length=max_len, truncation=True, padding=False)
            input_ids = enc["input_ids"]

            mask_end = 0
            for i in range(len(input_ids) - marker_len + 1):
                if input_ids[i:i + marker_len] == assistant_marker:
                    mask_end = i + marker_len
                    break

            labels = list(input_ids)
            for i in range(mask_end):
                labels[i] = -100

            items.append({
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
            })

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "input_ids": torch.tensor(it["input_ids"], dtype=torch.long),
            "labels": torch.tensor(it["labels"], dtype=torch.long),
        }


class TokenBudgetSampler(Sampler):
    """Fixed token budget per batch. Shuffle at batch level."""

    def __init__(self, dataset: TranscriptDataset, token_budget: int, shuffle: bool = True, seed: int = 42):
        self.batches = []
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        batch = []
        max_len = 0

        for idx in range(len(dataset)):
            seq_len = dataset.items[idx]["length"]
            new_max = max(max_len, seq_len)
            if batch and (len(batch) + 1) * new_max > token_budget:
                self.batches.append(batch)
                batch = [idx]
                max_len = seq_len
            else:
                batch.append(idx)
                max_len = new_max
        if batch:
            self.batches.append(batch)

        sizes = [len(b) for b in self.batches]
        print(f"  {len(self.batches)} batches, "
              f"examples/batch: min={min(sizes)} med={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch


def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    bs = len(batch)
    input_ids = torch.zeros(bs, max_len, dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ── Model / Optimizer ────────────────────────────────────────────────────────


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
    ).to("cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params/1e6:.0f}M params)")
    return model, tokenizer


def build_optimizers(model, lr: float, muon_lr: float, weight_decay: float = 0.01):
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    muon_n = sum(p.numel() for p in muon_params)
    adam_n = sum(p.numel() for p in adam_params)
    print(f"Muon: {len(muon_params)} tensors ({muon_n/1e6:.0f}M) | "
          f"AdamW: {len(adam_params)} tensors ({adam_n/1e6:.0f}M)")

    muon_opt = torch.optim.Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=5)
    adam_opt = torch.optim.AdamW(adam_params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    return [muon_opt, adam_opt]


def make_schedulers(optimizers, total_steps, warmup_frac=0.05):
    warmup = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in optimizers]


# ── Generate samples ─────────────────────────────────────────────────────────


@torch.no_grad()
def generate_samples(model, tokenizer, samples=EVAL_SAMPLES, max_new=200):
    model.eval()
    print("\n--- Samples ---")
    for src in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": src},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Inject no-think
        prompt += "<think>\n</think>\n\n"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  IN:  {src[:90]}")
        print(f"  OUT: {response[:90]}")
        print()
    model.train()


# ── Train ────────────────────────────────────────────────────────────────────


def cmd_train(args):
    data_dir = Path(__file__).parent / "data"
    train_path = data_dir / "transcript_correction_train.jsonl"
    val_path = data_dir / "transcript_correction_val.jsonl"

    if not train_path.exists():
        print("Run prepare_data.py first!")
        return

    model, tokenizer = load_model(args.model)
    model.gradient_checkpointing_enable()

    # Data
    print("Loading train set...")
    train_ds = TranscriptDataset(train_path, tokenizer, max_len=args.max_len)
    print("Loading val set...")
    val_ds = TranscriptDataset(val_path, tokenizer, max_len=args.max_len)

    print(f"Building batches (budget={args.token_budget})...")
    train_sampler = TokenBudgetSampler(train_ds, args.token_budget, shuffle=True)
    val_sampler = TokenBudgetSampler(val_ds, args.token_budget, shuffle=False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Optimizer + scheduler
    optimizers = build_optimizers(model, lr=args.lr, muon_lr=args.muon_lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    total_steps = len(train_loader) * args.epochs

    # Resume
    if args.resume:
        ckpt_dir = Path(args.resume)
        ckpt_meta = ckpt_dir / "train_state.pt"
        if ckpt_meta.exists():
            state = torch.load(ckpt_meta, map_location="cuda", weights_only=False)
            for i, o in enumerate(optimizers):
                o.load_state_dict(state["optimizers"][i])
            start_epoch = state["epoch"] + 1
            print(f"Resumed from epoch {start_epoch} (val_loss={state.get('val_loss', '?')})")
        # Load model weights
        from transformers import AutoModelForCausalLM as AMCLM
        model = AMCLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16,
                                       attn_implementation="sdpa", trust_remote_code=True).to("cuda")
        model.gradient_checkpointing_enable()
        # Rebuild optimizers with new model params
        optimizers = build_optimizers(model, lr=args.lr, muon_lr=args.muon_lr,
                                      weight_decay=args.weight_decay)
        total_steps = len(train_loader) * (args.epochs - start_epoch)

    schedulers = make_schedulers(optimizers, total_steps)

    # Output dir
    if args.resume:
        out_dir = Path(args.resume).parent
    else:
        out_dir = Path(__file__).parent / "runs" / f"finetune_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save setup
    setup = {
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "muon_lr": args.muon_lr,
        "weight_decay": args.weight_decay,
        "token_budget": args.token_budget,
        "max_len": args.max_len,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "batches_per_epoch": len(train_loader),
        "total_steps": total_steps,
    }
    with open(out_dir / "setup.json", "w") as f:
        json.dump(setup, f, indent=2)

    # CSV log
    log_path = out_dir / "train_log.csv"
    log_exists = log_path.exists() and start_epoch > 0
    log_file = open(log_path, "a" if log_exists else "w")
    if not log_exists:
        log_file.write("epoch,train_loss,val_loss,lr,secs\n")
        log_file.flush()

    print(f"Output: {out_dir}")
    print(f"Steps/epoch: {len(train_loader)}, total: {total_steps}")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        n_batches = 0
        n_tokens = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                for o in optimizers:
                    o.zero_grad(set_to_none=True)
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for o in optimizers:
                o.step()
            for s in schedulers:
                s.step()
            for o in optimizers:
                o.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n_batches += 1
            n_tokens += (labels != -100).sum().item()

            avg_loss = total_loss / n_batches
            lr_now = optimizers[1].param_groups[0]["lr"]
            tps = n_tokens / (time.time() - t0)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.1e}", tps=f"{tps:.0f}")

        pbar.close()

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Val", leave=False, unit="batch"):
                input_ids = batch["input_ids"].to("cuda")
                labels = batch["labels"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                val_loss += outputs.loss.item()
                val_steps += 1

        val_loss /= val_steps
        train_loss = total_loss / n_batches
        elapsed = time.time() - t0
        lr_now = optimizers[1].param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{args.epochs}: train={train_loss:.4f} val={val_loss:.4f} "
              f"lr={lr_now:.1e} time={elapsed:.0f}s")

        log_file.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{lr_now:.6e},{elapsed:.1f}\n")
        log_file.flush()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = out_dir / "best"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "optimizers": [o.state_dict() for o in optimizers],
            }, save_path / "train_state.pt")
            print(f"  Saved best (val={val_loss:.4f}) → {save_path}")

        # Show samples every epoch
        generate_samples(model, tokenizer)

    log_file.close()

    # Save final
    save_path = out_dir / "final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nDone. Final → {save_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")


# ── Eval ─────────────────────────────────────────────────────────────────────


def cmd_eval(args):
    ckpt = Path(args.checkpoint)
    model, tokenizer = load_model(str(ckpt))
    generate_samples(model, tokenizer)


# ── Export to GGUF ───────────────────────────────────────────────────────────


def cmd_export(args):
    import subprocess
    converter = Path(__file__).parent.parent / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"
    if not converter.exists():
        print(f"Converter not found: {converter}")
        return
    cmd = ["uv", "run", str(converter), args.checkpoint,
           "--outfile", args.output, "--outtype", args.quant]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Exported → {args.output}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-0.8B for transcript correction")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p = sub.add_parser("train")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5, help="AdamW learning rate")
    p.add_argument("--muon-lr", type=float, default=2e-6, help="Muon learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--token-budget", type=int, default=3072)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--resume", help="Resume from checkpoint dir")

    # eval
    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")

    # export
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    p.add_argument("--output", default="Qwen3.5-0.8B-transcript-Q8_0.gguf")
    p.add_argument("--quant", default="q8_0", help="Quantization type")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
