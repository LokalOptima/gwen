#!/usr/bin/env python3
"""
Prepare tokenized training data for MTP fine-tuning.

Tokenizes English text from 5 sources and writes a flat binary file of uint32
token IDs, with sequences separated by a sentinel (0xFFFFFFFF).

=== Data Sources ===

  1. Subscene English Subtitles (~60% of tokens)
     - Source: https://huggingface.co/datasets/refine-ai/subscene
     - Content: Movie/TV dialogue — spoken language patterns
     - Format: .json.gz, one JSON object per line, each with a "transcript"
       array of {id, start_time, end_time, text} subtitle segments.
     - Preprocessing: join all subtitle lines per file into one document,
       skip files with < 5 lines.
     - Download (grab as many files as needed, ~35M tokens each):
         uvx hf download refine-ai/subscene \
             english/english_subscene_0.json.gz \
             english/english_subscene_1.json.gz \
             english/english_subscene_2.json.gz \
             ... \
             --repo-type dataset --local-dir ~/models/data/subscene
       82 files available (~380MB each), ~35M tokens per file, 2.9B total.
     - Location: ~/models/data/subscene/english/*.json.gz

  2. LibriSpeech LM Text + LibriTTS (~2% of tokens)
     - LibriSpeech source: https://www.kaggle.com/datasets/saztorralba/librispeechquicklm
       Download manually from Kaggle, unzip to ~/models/data/librispeech-lm/
       Location: ~/models/data/librispeech-lm/text/librispeech-train.txt
       (132,553 sentences, 4.6M words, ALL CAPS — lowercased during processing)
     - LibriTTS source: already present in ~/git/rokoko/data/sources/libritts/
       Location: ~/git/rokoko/data/sources/libritts/01_clean/sentences.txt
       (134,545 sentences, proper casing)
     - Content: Audiobook narration transcripts — read English speech

  3. OpenWebText (~20% of tokens)
     - Source: https://huggingface.co/datasets/Skylion007/openwebtext
     - Content: Reddit-upvoted web articles — diverse written English
     - Format: Streamed via HuggingFace datasets library.
     - Preprocessing: skip docs < 50 chars.
     - Download: streamed on the fly (no local storage needed).

  4. FineWeb-Edu (~16% of tokens)
     - Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
     - Content: Curated educational web text — textbooks, tutorials, explanations
     - Format: Streamed via HuggingFace datasets library (sample-10BT config).
     - Preprocessing: skip docs < 50 chars.
     - Download: streamed on the fly (no local storage needed).

=== Output Format ===

  File: data/train_tokens.bin
  Format: flat array of uint32 (little-endian)
  Tokenizer: Qwen/Qwen3.5-0.8B (vocab_size=248320)
  Sentinel: 0xFFFFFFFF separates sequences
  Size: ~2 GB for ~500M tokens

=== Usage ===

  # Full run (all sources in parallel, ~15 minutes):
  uv run --with datasets --with transformers --with numpy \
      scripts/prepare_training_data.py

  # Custom token target:
  uv run --with datasets --with transformers --with numpy \
      scripts/prepare_training_data.py --target-tokens 200000000
"""

import argparse
import concurrent.futures
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
SENTINEL = 0xFFFFFFFF

# Local data paths
SUBSCENE_DIR = Path.home() / "models" / "data" / "subscene" / "english"
LIBRISPEECH_TXT = Path.home() / "models" / "data" / "librispeech-lm" / "text" / "librispeech-train.txt"
LIBRITTS_TXT = Path.home() / "git" / "rokoko" / "data" / "sources" / "libritts" / "01_clean" / "sentences.txt"

# Write to disk every N tokens (keeps memory bounded)
FLUSH_INTERVAL = 1_000_000


def log(msg):
    """Print to stderr (unbuffered) so progress is visible even in pipes."""
    print(msg, file=sys.stderr, flush=True)


class TokenWriter:
    """Streams tokenized data directly to disk in chunks."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "wb")
        self.buf = []
        self.total_tokens = 0
        self.n_sequences = 0

    def add_sequence(self, token_ids: list[int]):
        if len(token_ids) < 3:
            return 0
        self.buf.extend(token_ids)
        self.buf.append(SENTINEL)
        n = len(token_ids)
        self.total_tokens += n
        self.n_sequences += 1
        if len(self.buf) >= FLUSH_INTERVAL:
            self._flush()
        return n

    def _flush(self):
        if self.buf:
            np.array(self.buf, dtype=np.uint32).tofile(self.f)
            self.buf = []

    def close(self):
        self._flush()
        self.f.close()


def tokenize_texts(tokenizer, texts, writer, max_seq_len=2048):
    """Tokenize a batch of texts and write to disk. Returns token count."""
    texts = [t for t in texts if t and t.strip() and len(t) > 20]
    if not texts:
        return 0
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True,
                        max_length=max_seq_len)["input_ids"]
    return sum(writer.add_sequence(ids) for ids in encoded)


# ---------------------------------------------------------------------------
# Corpus processors
# ---------------------------------------------------------------------------

# Regex patterns for cleaning subtitle lines
_RE_BRACKET = re.compile(r"^\[.*\]$")          # [camera shutter clicks]
_RE_HTML = re.compile(r"<[^>]+>")               # <i>, </i>, etc.
_RE_ATTRIBUTION = re.compile(
    r"(?i)subtitles?\s+by|opensubtitles|subscene|\.org|\.com|"
    r"sync(?:hronized)?\s+by|ripped\s+by|encoded\s+by|downloaded\s+from")


def _clean_subtitle_line(text: str) -> str:
    """Clean a single subtitle line. Returns empty string if line is junk."""
    text = text.strip()
    if not text:
        return ""
    # Skip pure stage directions: [music playing], [gunshot], etc.
    if _RE_BRACKET.match(text):
        return ""
    # Skip attribution/ad lines
    if _RE_ATTRIBUTION.search(text):
        return ""
    # Strip inline HTML tags
    text = _RE_HTML.sub("", text)
    # Strip leftover ♪ music markers
    text = text.replace("♪", "").strip()
    return text


def process_subscene(tokenizer, writer, target_tokens):
    """Subscene English subtitles — spoken movie/TV dialogue."""
    log(f"\n[subscene] Target: {target_tokens:,} tokens")

    json_files = sorted(SUBSCENE_DIR.glob("*.json.gz"))
    if not json_files:
        log(f"[subscene] ERROR: No files in {SUBSCENE_DIR}")
        log(f"  Download with: uvx hf download refine-ai/subscene "
            f"english/english_subscene_0.json.gz --repo-type dataset "
            f"--local-dir ~/models/data/subscene")
        return 0

    log(f"[subscene] Found {len(json_files)} files")
    total = 0
    docs = 0
    skipped = 0
    t0 = time.time()

    for gz_path in json_files:
        log(f"[subscene] Reading {gz_path.name}...")
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                transcript = entry.get("transcript", [])
                if not transcript:
                    continue

                # Clean each subtitle line
                lines = []
                for seg in transcript:
                    raw = seg.get("text", "")
                    if not isinstance(raw, str):
                        continue
                    cleaned = _clean_subtitle_line(raw)
                    if cleaned:
                        lines.append(cleaned)

                if len(lines) < 5:
                    skipped += 1
                    continue

                total += tokenize_texts(tokenizer, [" ".join(lines)], writer)
                docs += 1

                if docs % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed if elapsed > 0 else 0
                    log(f"[subscene] {docs:>8,} docs | {total:>12,} tokens | "
                        f"{rate:,.0f} tok/s | {skipped:,} skipped")

                if total >= target_tokens:
                    break
        if total >= target_tokens:
            break

    elapsed = time.time() - t0
    log(f"[subscene] Done: {total:,} tokens from {docs:,} files "
        f"({skipped:,} skipped) in {elapsed:.1f}s")
    return total


def process_speech_transcripts(tokenizer, writer, target_tokens):
    """LibriSpeech + LibriTTS — audiobook narration transcripts (local files)."""
    log(f"\n[speech] Target: {target_tokens:,} tokens")

    total = 0
    batch = []

    def flush_batch():
        nonlocal total
        if batch:
            total += tokenize_texts(tokenizer, batch, writer)
            batch.clear()

    t0 = time.time()

    # LibriSpeech LM text (ALL CAPS → lowercase)
    if LIBRISPEECH_TXT.exists():
        docs = 0
        log(f"[speech] Reading LibriSpeech: {LIBRISPEECH_TXT.name}")
        with open(LIBRISPEECH_TXT) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch.append(line.lower())
                docs += 1
                if len(batch) >= 500:
                    flush_batch()
                if total >= target_tokens:
                    break
        flush_batch()
        log(f"[speech] LibriSpeech: {docs:,} sentences, {total:,} tokens so far")
    else:
        log(f"[speech] SKIP: {LIBRISPEECH_TXT} not found")

    # LibriTTS (proper casing, as-is)
    if total < target_tokens and LIBRITTS_TXT.exists():
        docs = 0
        log(f"[speech] Reading LibriTTS: {LIBRITTS_TXT.name}")
        with open(LIBRITTS_TXT) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                docs += 1
                if len(batch) >= 500:
                    flush_batch()
                if total >= target_tokens:
                    break
        flush_batch()
        log(f"[speech] LibriTTS: {docs:,} sentences, {total:,} tokens total")
    elif not LIBRITTS_TXT.exists():
        log(f"[speech] SKIP: {LIBRITTS_TXT} not found")

    elapsed = time.time() - t0
    log(f"[speech] Done: {total:,} tokens in {elapsed:.1f}s")
    return total


def process_openwebtext(tokenizer, writer, target_tokens):
    """OpenWebText — diverse English web text (streamed from HuggingFace)."""
    log(f"\n[openwebtext] Target: {target_tokens:,} tokens")

    total = 0
    docs = 0
    batch = []
    t0 = time.time()

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip() or len(text) < 50:
            continue
        batch.append(text)
        docs += 1
        if len(batch) >= 500:
            total += tokenize_texts(tokenizer, batch, writer)
            batch = []
            if docs % 25000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                log(f"[openwebtext] {docs:>8,} docs | {total:>12,} tokens | {rate:,.0f} tok/s")
        if total >= target_tokens:
            break

    if batch:
        total += tokenize_texts(tokenizer, batch, writer)

    elapsed = time.time() - t0
    log(f"[openwebtext] Done: {total:,} tokens from {docs:,} docs in {elapsed:.1f}s")
    return total


def process_fineweb_edu(tokenizer, writer, target_tokens):
    """FineWeb-Edu — curated educational web text (streamed from HuggingFace)."""
    log(f"\n[fineweb-edu] Target: {target_tokens:,} tokens")

    total = 0
    docs = 0
    batch = []
    t0 = time.time()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip() or len(text) < 50:
            continue
        batch.append(text)
        docs += 1
        if len(batch) >= 500:
            total += tokenize_texts(tokenizer, batch, writer)
            batch = []
            if docs % 25000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                log(f"[fineweb-edu] {docs:>8,} docs | {total:>12,} tokens | {rate:,.0f} tok/s")
        if total >= target_tokens:
            break

    if batch:
        total += tokenize_texts(tokenizer, batch, writer)

    elapsed = time.time() - t0
    log(f"[fineweb-edu] Done: {total:,} tokens from {docs:,} docs in {elapsed:.1f}s")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_corpus(func, name, tokenizer, output_path, target_tokens):
    """Run a single corpus processor, writing to its own part file."""
    writer = TokenWriter(output_path)
    total = func(tokenizer, writer, target_tokens)
    writer.close()
    size_mb = output_path.stat().st_size / 1024 / 1024
    log(f"[{name}] Wrote {output_path.name}: {total:,} tokens, {size_mb:.1f} MB")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Prepare tokenized training data for MTP fine-tuning")
    parser.add_argument("--target-tokens", type=int, default=500_000_000,
                        help="Target total tokens (default: 500M)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (default: data/train_tokens.bin)")
    args = parser.parse_args()

    target = args.target_tokens
    output = args.output or OUTPUT_DIR / "train_tokens.bin"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    log(f"Target: {target:,} tokens")
    log(f"Output: {output}")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Data mix:
    #   60% spoken dialogue (Subscene subtitles)
    #    2% read speech (LibriSpeech + LibriTTS)
    #   20% web text (OpenWebText)
    #   18% educational text (FineWeb-Edu)
    sub_target      = int(target * 0.60)
    speech_target   = int(target * 0.02)
    web_target      = int(target * 0.20)
    edu_target      = target - sub_target - speech_target - web_target

    # Process all corpora in parallel, each writing to its own part file
    part_dir = output.parent
    parts = [
        (process_subscene,             "subscene",    part_dir / "train_subscene.bin",    sub_target),
        (process_speech_transcripts,   "speech",      part_dir / "train_speech.bin",      speech_target),
        (process_openwebtext,          "openwebtext", part_dir / "train_openwebtext.bin", web_target),
        (process_fineweb_edu,          "fineweb-edu", part_dir / "train_fineweb.bin",     edu_target),
    ]

    log(f"\nProcessing {len(parts)} corpora in parallel...")
    for _, name, _, tgt in parts:
        log(f"  {name}: {tgt:,} tokens")

    grand_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = {
            executor.submit(run_corpus, func, name, tokenizer, path, tgt): name
            for func, name, path, tgt in parts
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                grand_total += future.result()
            except Exception as e:
                log(f"[{name}] FAILED: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)

    # Concatenate part files into final output (keep parts for reuse)
    log(f"\nConcatenating into {output.name}...")
    with open(output, "wb") as out_f:
        for _, name, path, _ in parts:
            if path.exists() and path.stat().st_size > 0:
                data = path.read_bytes()
                out_f.write(data)
                log(f"  + {name}: {len(data) / 1024 / 1024:.1f} MB")

    file_size = output.stat().st_size
    log(f"\n{'='*60}")
    log(f"DONE: {grand_total:,} tokens, {file_size / 1024 / 1024:.1f} MB")
    log(f"Output: {output}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
