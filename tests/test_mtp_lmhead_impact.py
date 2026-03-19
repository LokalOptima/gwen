#!/usr/bin/env python3
"""Compare MTP acceptance rates: full 248K vocab vs reduced 50K LM head."""

import subprocess
import re
import sys

GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "/home/lapo/models/gguf/Qwen3.5-0.8B-mtp-q8_0.bin"
MTP_LM_HEAD = "/home/lapo/models/gguf/lm_head_top50000.bin"
N = 200

PROMPTS = {
    "Scientific": "In a groundbreaking paper published in Nature, researchers demonstrated that",
    "Code": "The following Python function implements a binary search tree with self-balancing:",
    "Fiction": "Chapter 1: The old lighthouse keeper had seen many storms, but nothing could have prepared him for",
    "Technical": "The key differences between transformer and state-space model architectures include the following:",
    "Narrative": "Once upon a time in a small village nestled between two mountains, there lived a young girl who",
    "DevOps": "To configure a Kubernetes cluster with high availability, you need to follow these steps:",
}


def run_mtp(prompt, n_tokens, use_reduced=True):
    cmd = [GWEN, "--model", MODEL, "--mtp", MTP,
           "--prompt", prompt, "--n-predict", str(n_tokens)]
    if use_reduced:
        cmd += ["--mtp-lm-head", MTP_LM_HEAD]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None, None, None
    m = re.search(r'(\d+) accepted, (\d+) rejected \(([0-9.]+)% acceptance', result.stdout)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    return None, None, None


def main():
    fmt = "%-12s %10s %10s %8s"
    print(f"MTP LM Head Impact: Full 248K vs Reduced 50K ({N} tokens)")
    print("=" * 50)
    print(fmt % ("Prompt", "Full 248K", "Top 50K", "Delta"))
    print(fmt % ("------", "---------", "-------", "-----"))

    full_total_acc, full_total_rej = 0, 0
    red_total_acc, red_total_rej = 0, 0

    for label, prompt in PROMPTS.items():
        _, _, full_rate = run_mtp(prompt, N, use_reduced=False)
        _, _, red_rate = run_mtp(prompt, N, use_reduced=True)
        if full_rate is not None and red_rate is not None:
            delta = red_rate - full_rate
            print(fmt % (label, f"{full_rate:.1f}%", f"{red_rate:.1f}%", f"{delta:+.1f}%"))

    print()
    print("If 'Delta' is negative, the reduced LM head is hurting acceptance.")
    print("If ~0, the reduced LM head is fine (top 50K covers model predictions).")


if __name__ == "__main__":
    sys.exit(main())
