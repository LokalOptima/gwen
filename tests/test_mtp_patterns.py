#!/usr/bin/env python3
"""Analyze MTP accept/reject patterns: streaks, autocorrelation, position effects."""

import subprocess
import re
import sys
from collections import Counter

GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "/home/lapo/models/gguf/Qwen3.5-0.8B-mtp-q8_0.bin"
MTP_LM_HEAD = "/home/lapo/models/gguf/lm_head_top50000.bin"
N = 500

PROMPTS = {
    "Code": "The following Python function implements a binary search tree with self-balancing:",
    "Fiction": "Chapter 1: The old lighthouse keeper had seen many storms, but nothing could have prepared him for",
    "Technical": "The key differences between transformer and state-space model architectures include the following:",
    "DevOps": "To configure a Kubernetes cluster with high availability, you need to follow these steps:",
}


def run_mtp(prompt, n_tokens):
    cmd = [GWEN, "--model", MODEL, "--mtp", MTP, "--mtp-lm-head", MTP_LM_HEAD,
           "--prompt", prompt, "--n-predict", str(n_tokens)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None
    for line in result.stdout.split("\n"):
        if line.startswith("MTP sequence:"):
            return line.replace("MTP sequence:", "").strip()
    return None


def analyze_sequence(seq, label):
    n = len(seq)
    if n == 0:
        return

    acc = seq.count("A")
    rej = seq.count("R")
    rate = 100.0 * acc / n

    # Streak analysis
    streaks_a = []
    streaks_r = []
    current = seq[0]
    length = 1
    for i in range(1, n):
        if seq[i] == current:
            length += 1
        else:
            if current == "A":
                streaks_a.append(length)
            else:
                streaks_r.append(length)
            current = seq[i]
            length = 1
    if current == "A":
        streaks_a.append(length)
    else:
        streaks_r.append(length)

    # Transition probabilities
    aa = ar = ra = rr = 0
    for i in range(n - 1):
        pair = seq[i] + seq[i + 1]
        if pair == "AA": aa += 1
        elif pair == "AR": ar += 1
        elif pair == "RA": ra += 1
        elif pair == "RR": rr += 1

    # Windowed acceptance rate (quarters)
    q = n // 4
    quarters = []
    for i in range(4):
        start = i * q
        end = (i + 1) * q if i < 3 else n
        chunk = seq[start:end]
        quarters.append(100.0 * chunk.count("A") / len(chunk))

    print(f"\n{'=' * 60}")
    print(f"  {label} — {n} decisions, {rate:.1f}% acceptance")
    print(f"{'=' * 60}")

    # Print sequence with visual grouping
    print(f"\nSequence (first 120):")
    line = seq[:120]
    # Color-code: print in blocks of 10
    for i in range(0, len(line), 40):
        chunk = line[i:i+40]
        print(f"  {i:3d}: {chunk}")

    print(f"\nTransition probabilities:")
    if aa + ar > 0:
        print(f"  After A → A: {100*aa/(aa+ar):.1f}%  (A → R: {100*ar/(aa+ar):.1f}%)")
    if ra + rr > 0:
        print(f"  After R → A: {100*ra/(ra+rr):.1f}%  (R → R: {100*rr/(ra+rr):.1f}%)")
    print(f"  (coinflip would be: A→A = R→A = {rate:.1f}%)")

    print(f"\nStreak distribution:")
    if streaks_a:
        a_counts = Counter(streaks_a)
        max_a = max(streaks_a)
        avg_a = sum(streaks_a) / len(streaks_a)
        print(f"  Accept streaks: avg={avg_a:.1f}, max={max_a}, dist={dict(sorted(a_counts.items()))}")
    if streaks_r:
        r_counts = Counter(streaks_r)
        max_r = max(streaks_r)
        avg_r = sum(streaks_r) / len(streaks_r)
        print(f"  Reject streaks: avg={avg_r:.1f}, max={max_r}, dist={dict(sorted(r_counts.items()))}")

    print(f"\nAcceptance by position (quarters):")
    for i, q_rate in enumerate(quarters):
        bar = "#" * int(q_rate / 2)
        print(f"  Q{i+1}: {q_rate:5.1f}% {bar}")


def main():
    for label, prompt in PROMPTS.items():
        seq = run_mtp(prompt, N)
        if seq:
            analyze_sequence(seq, label)


if __name__ == "__main__":
    sys.exit(main())
