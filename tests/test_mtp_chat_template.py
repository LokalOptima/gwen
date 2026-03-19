#!/usr/bin/env python3
"""Compare MTP acceptance rates: raw prompts vs chat-templated prompts."""

import subprocess
import re
import sys

GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "/home/lapo/models/gguf/Qwen3.5-0.8B-mtp-q8_0.bin"
MTP_LM_HEAD = "/home/lapo/models/gguf/lm_head_top50000.bin"
N = 500

RAW_PROMPTS = {
    "Scientific": "In a groundbreaking paper published in Nature, researchers demonstrated that",
    "Code": "The following Python function implements a binary search tree with self-balancing:",
    "Fiction": "Chapter 1: The old lighthouse keeper had seen many storms, but nothing could have prepared him for",
    "Economics": "According to the latest economic data, global inflation rates have been influenced by",
    "Formal": "Dear hiring manager, I am writing to express my strong interest in the Senior Software Engineer position at",
    "Technical": "The key differences between transformer and state-space model architectures include the following:",
    "Narrative": "Once upon a time in a small village nestled between two mountains, there lived a young girl who",
    "DevOps": "To configure a Kubernetes cluster with high availability, you need to follow these steps:",
}

# Previous raw results for reference
PREV_RAW = {
    "Scientific": 55.5, "Code": 62.5, "Fiction": 64.4, "Economics": 52.1,
    "Formal": 47.6, "Technical": 41.9, "Narrative": 59.4, "DevOps": 53.2,
}


def chat_wrap(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def run_mtp(prompt, n_tokens):
    cmd = [GWEN, "--model", MODEL, "--mtp", MTP, "--mtp-lm-head", MTP_LM_HEAD,
           "--prompt", prompt, "--n-predict", str(n_tokens)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None, None
    m = re.search(r'([0-9.]+)% acceptance', result.stdout)
    tps = re.search(r'Tokens/sec:\s+([0-9.]+)', result.stdout)
    rate = float(m.group(1)) if m else None
    speed = float(tps.group(1)) if tps else None
    return rate, speed


def main():
    fmt = "%-12s %8s %8s %8s %8s"
    print(f"MTP: Raw vs Chat Template ({N} tokens)")
    print("=" * 56)
    print(fmt % ("Prompt", "Raw", "Chat", "Delta", "Chat t/s"))
    print(fmt % ("------", "---", "----", "-----", "--------"))

    total_raw, total_chat, count = 0, 0, 0
    total_tps = 0

    for label, prompt in RAW_PROMPTS.items():
        raw_rate = PREV_RAW[label]  # use cached raw results
        chat_rate, chat_tps = run_mtp(chat_wrap(prompt), N)
        if chat_rate is not None:
            delta = chat_rate - raw_rate
            print(fmt % (label, f"{raw_rate:.1f}%", f"{chat_rate:.1f}%",
                        f"{delta:+.1f}%", f"{chat_tps:.0f}"))
            total_raw += raw_rate
            total_chat += chat_rate
            total_tps += chat_tps
            count += 1

    if count > 0:
        print(fmt % ("AVERAGE", f"{total_raw/count:.1f}%", f"{total_chat/count:.1f}%",
              f"{(total_chat-total_raw)/count:+.1f}%", f"{total_tps/count:.0f}"))


if __name__ == "__main__":
    sys.exit(main())
