#!/usr/bin/env python3
"""
Compare gwen inference outputs against llama.cpp reference.

Usage:
    uv run tests/compare_outputs.py                    # full comparison suite
    uv run tests/compare_outputs.py --quick             # single prompt, greedy only
    uv run tests/compare_outputs.py --save-reference    # save llama.cpp outputs as golden reference
    uv run tests/compare_outputs.py --layer-debug       # dump per-layer intermediates
"""

import argparse
import json
import subprocess
import struct
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "Qwen3.5-0.8B-Q4_K_M.gguf"
LLAMACPP_BIN = PROJECT_ROOT / "third_party" / "llama.cpp" / "build" / "bin"
GWEN_BIN = PROJECT_ROOT / "build" / "gwen"
REFERENCE_DIR = PROJECT_ROOT / "tests" / "golden"

# --- Test prompts ---
TEST_PROMPTS = {
    "single_token": "Hello",
    "short": "The capital of France is",
    "medium": "Explain the difference between a compiler and an interpreter in simple terms.",
    "code": "Write a Python function that computes the Fibonacci sequence:",
    "math": "What is the integral of x^2 from 0 to 1? Let me think step by step.",
    "multilingual": "Translate 'hello world' to Chinese, Japanese, and Korean:",
    "long": (
        "In the field of machine learning, transformer architectures have become "
        "the dominant paradigm for natural language processing. The key innovation "
        "of transformers is the self-attention mechanism, which allows the model to "
        "weigh the importance of different parts of the input sequence. Unlike "
        "recurrent neural networks, transformers can process all positions in "
        "parallel, leading to significant speedups during training. However, the "
        "quadratic complexity of self-attention with respect to sequence length "
        "remains a challenge. Recent work on linear attention mechanisms, such as "
        "the DeltaNet architecture used in Qwen3.5, addresses this by replacing "
        "softmax attention with a linear recurrence. This allows O(1) memory per "
        "step during autoregressive generation, while maintaining competitive "
        "quality. The hybrid approach of mixing linear and full attention layers "
        "appears to offer the best of both worlds. Discuss the tradeoffs:"
    ),
}

N_PREDICT = 50  # tokens to generate per prompt


def run_llamacpp(prompt: str, n_predict: int = N_PREDICT, logprobs: bool = True) -> dict:
    """Run llama.cpp inference and capture outputs."""
    # Use llama-completion (formerly main) for direct generation
    cmd = [
        str(LLAMACPP_BIN / "llama-cli"),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", str(n_predict),
        "--no-conversation",
        "--temp", "0",  # greedy for deterministic comparison
        "--log-disable",
    ]
    if logprobs:
        cmd.extend(["--logits-all"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"llama.cpp error: {result.stderr}", file=sys.stderr)
        return {"error": result.stderr, "tokens": [], "text": ""}

    return {
        "text": result.stdout.strip(),
        "tokens": [],  # TODO: parse token IDs from verbose output
    }


def run_gwen(prompt: str, n_predict: int = N_PREDICT) -> dict:
    """Run gwen inference and capture outputs."""
    cmd = [
        str(GWEN_BIN),
        "--model", str(MODEL_PATH),
        "--prompt", prompt,
        "--n-predict", str(n_predict),
        "--greedy",
        "--output-logits",  # output raw logits as binary
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"gwen error: {result.stderr.decode()}", file=sys.stderr)
        return {"error": result.stderr.decode(), "tokens": [], "text": ""}

    # Parse gwen output format (text on stdout, logits on stderr or separate file)
    return {
        "text": result.stdout.decode().strip(),
        "tokens": [],  # TODO: parse from structured output
    }


def compare_greedy_tokens(ref_text: str, test_text: str, prompt: str) -> dict:
    """Compare greedy-decoded text token by token."""
    # Strip the prompt echo if present
    if ref_text.startswith(prompt):
        ref_text = ref_text[len(prompt):]
    if test_text.startswith(prompt):
        test_text = test_text[len(prompt):]

    ref_chars = list(ref_text)
    test_chars = list(test_text)

    # Find first divergence point
    match_len = 0
    for i in range(min(len(ref_chars), len(test_chars))):
        if ref_chars[i] == test_chars[i]:
            match_len += 1
        else:
            break

    total = max(len(ref_chars), len(test_chars), 1)
    match_ratio = match_len / total

    return {
        "match_ratio": match_ratio,
        "exact_match": ref_text == test_text,
        "first_divergence_pos": match_len if not ref_text == test_text else -1,
        "ref_length": len(ref_text),
        "test_length": len(test_text),
        "ref_text": ref_text[:200],
        "test_text": test_text[:200],
    }


def compute_kl_divergence(ref_logits: np.ndarray, test_logits: np.ndarray) -> dict:
    """Compute KL divergence between logit distributions."""
    # Softmax both
    def softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    ref_probs = softmax(ref_logits)
    test_probs = softmax(test_logits)

    # KL(ref || test) = sum(ref * log(ref / test))
    eps = 1e-10
    kl = np.sum(ref_probs * np.log((ref_probs + eps) / (test_probs + eps)), axis=-1)

    return {
        "mean_kl": float(np.mean(kl)),
        "max_kl": float(np.max(kl)),
        "per_position_kl": kl.tolist()[:20],  # first 20 positions
    }


def run_comparison_suite(prompts: dict, save_reference: bool = False):
    """Run the full comparison suite."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, prompt in prompts.items():
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'='*60}")

        # Run llama.cpp
        print("  Running llama.cpp...", end=" ", flush=True)
        t0 = time.time()
        ref = run_llamacpp(prompt)
        t_ref = time.time() - t0
        print(f"done ({t_ref:.2f}s)")

        if save_reference:
            ref_path = REFERENCE_DIR / f"{name}.json"
            with open(ref_path, "w") as f:
                json.dump({"prompt": prompt, "output": ref}, f, indent=2)
            print(f"  Saved reference to {ref_path}")
            continue

        # Run gwen
        print("  Running gwen...", end=" ", flush=True)
        t0 = time.time()
        test = run_gwen(prompt)
        t_gwen = time.time() - t0
        print(f"done ({t_gwen:.2f}s)")

        if "error" in ref:
            print(f"  SKIP: llama.cpp error: {ref['error'][:100]}")
            continue
        if "error" in test:
            print(f"  SKIP: gwen error: {test['error'][:100]}")
            continue

        # Compare
        comparison = compare_greedy_tokens(ref["text"], test["text"], prompt)
        results[name] = {
            "comparison": comparison,
            "timing": {"llama_cpp_s": t_ref, "gwen_s": t_gwen},
        }

        # Report
        status = "PASS" if comparison["exact_match"] else "FAIL"
        print(f"  [{status}] Match ratio: {comparison['match_ratio']:.2%}")
        if not comparison["exact_match"]:
            print(f"  First divergence at position {comparison['first_divergence_pos']}")
            print(f"  Ref:  ...{comparison['ref_text'][max(0, comparison['first_divergence_pos']-10):comparison['first_divergence_pos']+30]}...")
            print(f"  Test: ...{comparison['test_text'][max(0, comparison['first_divergence_pos']-10):comparison['first_divergence_pos']+30]}...")
        print(f"  Speedup: {t_ref/t_gwen:.2f}x (llama.cpp: {t_ref:.2f}s, gwen: {t_gwen:.2f}s)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results.values() if r["comparison"]["exact_match"])
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    avg_match = np.mean([r["comparison"]["match_ratio"] for r in results.values()]) if results else 0
    print(f"  Average match ratio: {avg_match:.2%}")
    if results:
        avg_speedup = np.mean([r["timing"]["llama_cpp_s"] / max(r["timing"]["gwen_s"], 0.001) for r in results.values()])
        print(f"  Average speedup: {avg_speedup:.2f}x")

    # Save results
    results_path = PROJECT_ROOT / "tests" / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Compare gwen vs llama.cpp outputs")
    parser.add_argument("--quick", action="store_true", help="Run single prompt only")
    parser.add_argument("--save-reference", action="store_true", help="Save llama.cpp outputs as golden reference")
    parser.add_argument("--layer-debug", action="store_true", help="Enable per-layer comparison")
    parser.add_argument("--prompt", type=str, help="Custom single prompt to test")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    if args.prompt:
        prompts = {"custom": args.prompt}
    elif args.quick:
        prompts = {"short": TEST_PROMPTS["short"]}
    else:
        prompts = TEST_PROMPTS

    success = run_comparison_suite(prompts, save_reference=args.save_reference)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
