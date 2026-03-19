#!/usr/bin/env python3
"""Investigate Code prompt acceptance degradation: is the text degenerating, or is MTP broken?"""

import subprocess
import re
import sys

GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "/home/lapo/models/gguf/Qwen3.5-0.8B-mtp-q8_0.bin"
MTP_LM_HEAD = "/home/lapo/models/gguf/lm_head_top50000.bin"

PROMPT = "The following Python function implements a binary search tree with self-balancing:"


def run(prompt, n_tokens, use_mtp=False):
    cmd = [GWEN, "--model", MODEL, "--prompt", prompt, "--n-predict", str(n_tokens)]
    if use_mtp:
        cmd += ["--mtp", MTP, "--mtp-lm-head", MTP_LM_HEAD]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"FAILED: {result.stderr[-200:]}")
        return None, None, None

    tokens = None
    text = None
    ar_seq = None
    for line in result.stdout.split("\n"):
        if line.startswith("Generated token IDs:"):
            ids = line.replace("Generated token IDs:", "").strip().split()
            tokens = [int(t) for t in ids]
        if line.startswith("MTP sequence:"):
            ar_seq = line.replace("MTP sequence:", "").strip()

    # Extract the generated text (after the prompt)
    # Find the line that starts with the prompt
    for line in result.stdout.split("\n"):
        if line.startswith(PROMPT):
            text = line[len(PROMPT):]
            break

    return tokens, text, ar_seq


def main():
    N = 500
    print(f"Investigating Code prompt degradation ({N} tokens)")
    print("=" * 70)

    # Run baseline (no MTP)
    base_tokens, base_text, _ = run(PROMPT, N, use_mtp=False)
    if base_tokens is None:
        print("Baseline failed!")
        return 1

    # Run MTP
    mtp_tokens, mtp_text, ar_seq = run(PROMPT, N, use_mtp=True)
    if mtp_tokens is None:
        print("MTP failed!")
        return 1

    # Check token identity
    if base_tokens == mtp_tokens:
        print("Tokens: IDENTICAL (MTP matches baseline)")
    else:
        min_len = min(len(base_tokens), len(mtp_tokens))
        for i in range(min_len):
            if base_tokens[i] != mtp_tokens[i]:
                print(f"Tokens: DIVERGE at position {i}")
                print(f"  Base: ...{base_tokens[max(0,i-3):i+3]}")
                print(f"  MTP:  ...{mtp_tokens[max(0,i-3):i+3]}")
                break

    if ar_seq:
        n = len(ar_seq)
        q = n // 4

        print(f"\nAcceptance by quarter:")
        for i in range(4):
            start = i * q
            end = (i + 1) * q if i < 3 else n
            chunk = ar_seq[start:end]
            rate = 100.0 * chunk.count("A") / len(chunk)
            tok_start = start  # approximate token position
            tok_end = end
            print(f"  Q{i+1} (decisions {start}-{end}): {rate:.1f}%")

    # Show text by quarters to see if content changes
    if base_text:
        print(f"\n--- Generated text by quarter ---")
        chars_per_q = len(base_text) // 4
        for i in range(4):
            start = i * chars_per_q
            end = (i + 1) * chars_per_q if i < 3 else len(base_text)
            chunk = base_text[start:end]
            # Truncate long chunks
            if len(chunk) > 200:
                chunk = chunk[:200] + "..."
            print(f"\n  Q{i+1}:")
            print(f"    {repr(chunk)}")

    # Now test: run multiple SHORT sequences to see if the acceptance
    # is consistently high early on, ruling out a compounding bug
    print(f"\n--- Fresh starts at different positions ---")
    # Test: if we re-run with a longer prompt (simulating more context),
    # does MTP still get high acceptance?
    test_lengths = [100, 200, 300, 500]
    for n in test_lengths:
        _, _, seq = run(PROMPT, n, use_mtp=True)
        if seq:
            # Look at the last 50 decisions
            last_50 = seq[-50:] if len(seq) > 50 else seq
            first_50 = seq[:50] if len(seq) > 50 else seq
            rate_first = 100.0 * first_50.count("A") / len(first_50)
            rate_last = 100.0 * last_50.count("A") / len(last_50)
            rate_all = 100.0 * seq.count("A") / len(seq)
            print(f"  {n:3d} tokens: first50={rate_first:.0f}% last50={rate_last:.0f}% overall={rate_all:.0f}%")


if __name__ == "__main__":
    sys.exit(main())
