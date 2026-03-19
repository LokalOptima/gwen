#!/usr/bin/env python3
"""Analyze MTP acceptance rates: early vs late, and compare with blog 09 prompts."""

import subprocess
import re
import sys

GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "/home/lapo/models/gguf/Qwen3.5-0.8B-mtp-q8_0.bin"
MTP_LM_HEAD = "/home/lapo/models/gguf/lm_head_top50000.bin"

# Current prompts from bench_mtp.sh
CURRENT_PROMPTS = {
    "Scientific": "In a groundbreaking paper published in Nature, researchers demonstrated that",
    "Code": "The following Python function implements a binary search tree with self-balancing:",
    "Fiction": "Chapter 1: The old lighthouse keeper had seen many storms, but nothing could have prepared him for",
    "Economics": "According to the latest economic data, global inflation rates have been influenced by",
    "Formal": "Dear hiring manager, I am writing to express my strong interest in the Senior Software Engineer position at",
    "Technical": "The key differences between transformer and state-space model architectures include the following:",
    "Narrative": "Once upon a time in a small village nestled between two mountains, there lived a young girl who",
    "DevOps": "To configure a Kubernetes cluster with high availability, you need to follow these steps:",
}

# Blog post 09 prompts (from the benchmark table)
OLD_PROMPTS = {
    "France": "The capital of France is",
    "Quantum": "Explain quantum physics in simple terms:",
    "Fibonacci": "Write a Python function that calculates the Fibonacci sequence:",
    "Village": "Once upon a time in a small village, there lived a young girl who loved to",
    "Fox": "The quick brown fox jumps over the lazy dog. This sentence is often used",
    "Climate": "Climate change is one of the most pressing issues facing humanity today because",
    "ML": "The most important machine learning algorithms include:",
    "Roman": "The Roman Empire was one of the most powerful civilizations in history because",
}


def run_mtp(prompt, n_tokens):
    cmd = [GWEN, "--model", MODEL, "--mtp", MTP, "--mtp-lm-head", MTP_LM_HEAD,
           "--prompt", prompt, "--n-predict", str(n_tokens)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None, None, None
    stdout = result.stdout
    # Extract acceptance rate
    m = re.search(r'(\d+) accepted, (\d+) rejected \(([0-9.]+)% acceptance', stdout)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    return None, None, None


def main():
    print("=" * 70)
    print("MTP Acceptance Rate Analysis")
    print("=" * 70)

    # Test 1: Compare 100 vs 500 tokens (does acceptance degrade over time?)
    print("\n--- Test 1: Acceptance rate degradation over sequence length ---")
    prompt = "Once upon a time in a small village nestled between two mountains, there lived a young girl who"
    printf = "%-10s %6s %6s %8s"
    print(printf % ("Tokens", "Accept", "Reject", "Rate"))
    print(printf % ("------", "------", "------", "--------"))
    for n in [50, 100, 200, 500]:
        acc, rej, rate = run_mtp(prompt, n)
        if rate is not None:
            print(printf % (n, acc, rej, f"{rate:.1f}%"))

    # Test 2: Blog post 09 prompts at 200 tokens (same as original benchmark)
    print("\n--- Test 2: Blog post 09 prompts (200 tokens) ---")
    printf2 = "%-12s %6s %6s %8s"
    print(printf2 % ("Prompt", "Accept", "Reject", "Rate"))
    print(printf2 % ("------", "------", "------", "--------"))
    total_acc, total_rej = 0, 0
    for label, prompt in OLD_PROMPTS.items():
        acc, rej, rate = run_mtp(prompt, 200)
        if rate is not None:
            print(printf2 % (label, acc, rej, f"{rate:.1f}%"))
            total_acc += acc
            total_rej += rej
    if total_acc + total_rej > 0:
        print(printf2 % ("AVERAGE", total_acc, total_rej,
              f"{100.0 * total_acc / (total_acc + total_rej):.1f}%"))

    # Test 3: Current prompts at 200 tokens (for apples-to-apples with blog 09)
    print("\n--- Test 3: Current prompts (200 tokens) ---")
    print(printf2 % ("Prompt", "Accept", "Reject", "Rate"))
    print(printf2 % ("------", "------", "------", "--------"))
    total_acc, total_rej = 0, 0
    for label, prompt in CURRENT_PROMPTS.items():
        acc, rej, rate = run_mtp(prompt, 200)
        if rate is not None:
            print(printf2 % (label, acc, rej, f"{rate:.1f}%"))
            total_acc += acc
            total_rej += rej
    if total_acc + total_rej > 0:
        print(printf2 % ("AVERAGE", total_acc, total_rej,
              f"{100.0 * total_acc / (total_acc + total_rej):.1f}%"))


if __name__ == "__main__":
    sys.exit(main())
