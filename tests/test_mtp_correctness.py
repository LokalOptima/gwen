#!/usr/bin/env python3
"""Test MTP speculative decode produces identical output to baseline greedy."""

import subprocess
import sys

GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "/home/lapo/models/gguf/Qwen3.5-0.8B-mtp-q8_0.bin"
MTP_LM_HEAD = "/home/lapo/models/gguf/lm_head_top50000.bin"
N_TOKENS = 100

PROMPTS = [
    "In a groundbreaking paper published in Nature, researchers demonstrated that",
    "The following Python function implements a binary search tree with self-balancing:",
    "Chapter 1: The old lighthouse keeper had seen many storms, but nothing could have prepared him for",
    "According to the latest economic data, global inflation rates have been influenced by",
    "Dear hiring manager, I am writing to express my strong interest in the Senior Software Engineer position at",
    "The key differences between transformer and state-space model architectures include the following:",
    "Once upon a time in a small village nestled between two mountains, there lived a young girl who",
    "To configure a Kubernetes cluster with high availability, you need to follow these steps:",
]

LABELS = ["Scientific", "Code", "Fiction", "Economics", "Formal", "Technical", "Narrative", "DevOps"]


def run_gwen(prompt, use_mtp=False):
    cmd = [GWEN, "--model", MODEL, "--prompt", prompt, "--n-predict", str(N_TOKENS)]
    if use_mtp:
        cmd += ["--mtp", MTP, "--mtp-lm-head", MTP_LM_HEAD]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return None
    # Extract token IDs from output
    for line in result.stdout.split("\n"):
        if line.startswith("Generated token IDs:"):
            tokens = line.replace("Generated token IDs:", "").strip().split()
            return [int(t) for t in tokens]
    print("FAILED: could not find 'Generated token IDs' in output")
    return None


def main():
    print(f"MTP Correctness Test: {N_TOKENS} tokens per prompt")
    print(f"Model: {MODEL}")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, (prompt, label) in enumerate(zip(PROMPTS, LABELS)):
        print(f"\n[{i+1}/{len(PROMPTS)}] {label}: ", end="", flush=True)

        base_tokens = run_gwen(prompt, use_mtp=False)
        if base_tokens is None:
            failed += 1
            continue

        mtp_tokens = run_gwen(prompt, use_mtp=True)
        if mtp_tokens is None:
            failed += 1
            continue

        if base_tokens == mtp_tokens:
            print(f"PASS ({len(base_tokens)} tokens match)")
            passed += 1
        else:
            # Find first divergence
            min_len = min(len(base_tokens), len(mtp_tokens))
            diverge_at = min_len
            for j in range(min_len):
                if base_tokens[j] != mtp_tokens[j]:
                    diverge_at = j
                    break
            print(f"FAIL at token {diverge_at}")
            print(f"  Base len={len(base_tokens)}, MTP len={len(mtp_tokens)}")
            start = max(0, diverge_at - 2)
            end = min(min_len, diverge_at + 3)
            print(f"  Base[{start}:{end}]: {base_tokens[start:end]}")
            print(f"  MTP [{start}:{end}]: {mtp_tokens[start:end]}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(PROMPTS)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
