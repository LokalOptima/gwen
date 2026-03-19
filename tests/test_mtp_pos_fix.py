#!/usr/bin/env python3
"""Quick acceptance rate check after mtp_pos fix."""

import subprocess
import re
import sys

GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "/home/lapo/models/gguf/Qwen3.5-0.8B-mtp-q8_0.bin"
MTP_LM_HEAD = "/home/lapo/models/gguf/lm_head_top50000.bin"

PROMPTS = {
    "Scientific": "In a groundbreaking paper published in Nature, researchers demonstrated that",
    "Code": "The following Python function implements a binary search tree with self-balancing:",
    "Fiction": "Chapter 1: The old lighthouse keeper had seen many storms, but nothing could have prepared him for",
    "Economics": "According to the latest economic data, global inflation rates have been influenced by",
    "Formal": "Dear hiring manager, I am writing to express my strong interest in the Senior Software Engineer position at",
    "Technical": "The key differences between transformer and state-space model architectures include the following:",
    "Narrative": "Once upon a time in a small village nestled between two mountains, there lived a young girl who",
    "DevOps": "To configure a Kubernetes cluster with high availability, you need to follow these steps:",
}

# Previous results (without mtp_pos fix, 500 tokens)
PREV_RATES = {
    "Scientific": 55.5,
    "Code": 62.5,
    "Fiction": 64.4,
    "Economics": 52.1,
    "Formal": 47.6,
    "Technical": 41.9,
    "Narrative": 59.4,
    "DevOps": 53.2,
}


def run_mtp(prompt, n_tokens):
    cmd = [GWEN, "--model", MODEL, "--mtp", MTP, "--mtp-lm-head", MTP_LM_HEAD,
           "--prompt", prompt, "--n-predict", str(n_tokens)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None
    m = re.search(r'([0-9.]+)% acceptance', result.stdout)
    return float(m.group(1)) if m else None


def main():
    N = 500
    fmt = "%-12s %8s %8s %8s"
    print(f"MTP Acceptance: Before vs After mtp_pos-- fix ({N} tokens)")
    print("=" * 50)
    print(fmt % ("Prompt", "Before", "After", "Delta"))
    print(fmt % ("------", "------", "-----", "-----"))

    total_before, total_after, count = 0, 0, 0
    for label, prompt in PROMPTS.items():
        rate = run_mtp(prompt, N)
        prev = PREV_RATES.get(label)
        if rate is not None and prev is not None:
            delta = rate - prev
            print(fmt % (label, f"{prev:.1f}%", f"{rate:.1f}%", f"{delta:+.1f}%"))
            total_before += prev
            total_after += rate
            count += 1

    if count > 0:
        print(fmt % ("AVERAGE", f"{total_before/count:.1f}%", f"{total_after/count:.1f}%",
              f"{(total_after - total_before)/count:+.1f}%"))


if __name__ == "__main__":
    sys.exit(main())
