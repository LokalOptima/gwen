#!/usr/bin/env python3
"""Unify transcript correction datasets into a single JSONL for fine-tuning.

Datasets:
  - GenSEC Task1: ASR hypothesis → clean transcript (240K)
  - RED-ACE: ASR error detection & correction (503K)
  - CoEdIT (gec): grammar/disfluency correction (20K)
  - Disfl-QA: disfluent → fluent questions (7K)

Output: train/data/transcript_correction_{train,val}.jsonl
  Each line: {"src": "noisy text", "tgt": "clean text"}
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows → {path}")


def load_gensec():
    """GenSEC Task1: take first N-best hypothesis as src."""
    print("Loading GenSEC Task1...")
    ds = load_dataset("GenSEC-LLM/SLT-Task1-Post-ASR-Text-Correction")
    rows = []
    for split in ["train", "test"]:
        for ex in ds[split]:
            hyps = ex["hypothesis"]
            if not hyps:
                continue
            src = hyps[0] if isinstance(hyps, list) else hyps
            tgt = ex["transcription"]
            if src and tgt and src.strip() != tgt.strip():
                rows.append({"src": src.strip(), "tgt": tgt.strip(), "source": "gensec"})
    print(f"  GenSEC: {len(rows)} pairs (skipped identical src/tgt)")
    return rows


def load_redace():
    """RED-ACE: join word tokens, take train split only."""
    print("Loading RED-ACE...")
    ds = load_dataset("google/red_ace_asr_error_detection_and_correction")
    rows = []
    for ex in ds["train"]:
        words = ex["asr_hypothesis"]
        truth = ex["truth"]
        if not words or not truth:
            continue
        src = " ".join(words) if isinstance(words, list) else words
        tgt = truth.strip()
        if src.strip() != tgt:
            rows.append({"src": src.strip(), "tgt": tgt, "source": "redace"})
    print(f"  RED-ACE: {len(rows)} pairs")
    return rows


def load_coedit():
    """CoEdIT gec: strip instruction prefix from src."""
    print("Loading CoEdIT...")
    ds = load_dataset("grammarly/coedit")
    rows = []
    for ex in ds["train"]:
        if ex["task"] != "gec":
            continue
        src_raw = ex["src"]
        tgt = ex["tgt"]
        # Strip instruction prefix like "Fix grammar: " or "Remove all grammatical errors from this text: "
        # The prefix ends with ": " followed by the actual text
        m = re.match(r"^[^:]+:\s*", src_raw)
        src = src_raw[m.end():] if m else src_raw
        if src.strip() and tgt.strip() and src.strip() != tgt.strip():
            rows.append({"src": src.strip(), "tgt": tgt.strip(), "source": "coedit"})
    print(f"  CoEdIT (gec): {len(rows)} pairs")
    return rows


def load_disflqa():
    """Disfl-QA: disfluent → original question."""
    print("Loading Disfl-QA...")
    ds = load_dataset("google-research-datasets/disfl_qa")
    rows = []
    for split in ["train", "validation"]:
        for ex in ds[split]:
            src = ex["disfluent question"]
            tgt = ex["original question"]
            if src and tgt and src.strip() != tgt.strip():
                rows.append({"src": src.strip(), "tgt": tgt.strip(), "source": "disflqa"})
    print(f"  Disfl-QA: {len(rows)} pairs")
    return rows


def main():
    all_rows = []
    all_rows.extend(load_gensec())
    all_rows.extend(load_redace())
    all_rows.extend(load_coedit())
    all_rows.extend(load_disflqa())

    print(f"\nTotal: {len(all_rows)} pairs")
    print(f"  By source: { {s: sum(1 for r in all_rows if r['source'] == s) for s in ['gensec', 'redace', 'coedit', 'disflqa']} }")

    # Shuffle deterministically and split 99/1 train/val
    import random
    random.seed(42)
    random.shuffle(all_rows)

    n_val = max(1000, len(all_rows) // 100)
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    print(f"\nSplit: {len(train_rows)} train, {len(val_rows)} val")

    write_jsonl(OUT / "transcript_correction_train.jsonl", train_rows)
    write_jsonl(OUT / "transcript_correction_val.jsonl", val_rows)

    # Print some examples
    print("\n--- Examples ---")
    for i in range(5):
        r = train_rows[i]
        print(f"\n[{r['source']}]")
        print(f"  src: {r['src'][:100]}")
        print(f"  tgt: {r['tgt'][:100]}")


if __name__ == "__main__":
    main()
