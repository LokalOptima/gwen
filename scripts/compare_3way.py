#!/usr/bin/env python3
"""3-way comparison: GWEN NVFP4 vs dequantized HF reference vs llama.cpp Q4_K_M.

All using Qwen3.5-4B with the same prompt.
"""
import subprocess
import sys
import os
import time

PROMPT = "What is the capital of France?"
CHATML = (
    "<|im_start|>user\n"
    f"{PROMPT}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)
MAX_TOKENS = 80

GWEN_BIN = os.path.expanduser("~/git/gwen/build/gwen")
GWEN_MODEL = os.path.expanduser("~/models/Qwen3.5-4B.gwfp4")
GWEN_TEMPLATE = os.path.expanduser("~/git/gwen/prompts/chatml.tmpl")

LLAMA_CLI = os.path.expanduser("~/git/llama.cpp/build/bin/llama-simple")
LLAMA_MODEL = os.path.expanduser("~/models/gguf/Qwen3.5-4B-Q4_K_M.gguf")

NVFP4_DIR = os.path.expanduser("~/models/Qwen3.5-4B-NVFP4")

def run_gwen():
    """Run GWEN NVFP4 inference."""
    cmd = [
        "flock", "--shared", "/tmp/gpu.lock",
        GWEN_BIN, GWEN_MODEL,
        "--max-predict", str(MAX_TOKENS),
        "--greedy",
        CHATML,  # Pass the full ChatML prompt directly (no template)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    # GWEN prints to stdout, logs to stderr
    return result.stdout.strip()


def run_llama_cpp():
    """Run llama.cpp Q4_K_M inference via llama-simple."""
    cmd = [
        "flock", "--shared", "/tmp/gpu.lock",
        LLAMA_CLI,
        "-m", LLAMA_MODEL,
        "-n", str(MAX_TOKENS),
        "-ngl", "99",
        CHATML,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    # llama-simple prints prompt + generation; strip the prompt prefix
    out = result.stdout
    if "</think>\n\n" in out:
        out = out.split("</think>\n\n", 1)[-1]
    return out.strip()


def run_hf_reference():
    """Run dequantized NVFP4→BF16 through HF model (in subprocess for memory isolation)."""
    script = os.path.join(os.path.dirname(__file__), "nvfp4_reference.py")
    cmd = [
        "flock", "--shared", "/tmp/gpu.lock",
        sys.executable, script,
        "--model-path", NVFP4_DIR,
        "--prompt", PROMPT,
        "--max-tokens", str(MAX_TOKENS),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # Extract just the greedy generation from the output
    output = result.stdout
    marker = "=== Generated (greedy) ==="
    if marker in output:
        after = output.split(marker)[1]
        # Get text up to next section marker
        end_marker = "=== Generated (temp="
        if end_marker in after:
            after = after.split(end_marker)[0]
        return after.strip()
    return f"ERROR: {result.stderr[-500:]}" if result.returncode != 0 else "(empty output)"


def main():
    print("=" * 70)
    print(f"3-WAY COMPARISON — Qwen3.5-4B")
    print(f"Prompt: {PROMPT}")
    print(f"Max tokens: {MAX_TOKENS}, greedy decoding")
    print("=" * 70)

    # 1. HF reference (slowest, do first while GPU is cold)
    print("\n[1/3] Running HF reference (dequantized NVFP4→BF16)...")
    t0 = time.time()
    try:
        hf_out = run_hf_reference()
    except Exception as e:
        hf_out = f"ERROR: {e}"
    hf_time = time.time() - t0
    print(f"  Done in {hf_time:.1f}s")

    # 2. GWEN NVFP4
    print("\n[2/3] Running GWEN NVFP4...")
    t0 = time.time()
    try:
        gwen_out = run_gwen()
    except Exception as e:
        gwen_out = f"ERROR: {e}"
    gwen_time = time.time() - t0
    print(f"  Done in {gwen_time:.1f}s")

    # 3. llama.cpp Q4_K_M
    print("\n[3/3] Running llama.cpp Q4_K_M...")
    t0 = time.time()
    try:
        llama_out = run_llama_cpp()
    except Exception as e:
        llama_out = f"ERROR: {e}"
    llama_time = time.time() - t0
    print(f"  Done in {llama_time:.1f}s")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n--- [1] HF Reference (NVFP4→BF16, {hf_time:.1f}s) ---")
    print(hf_out)

    print(f"\n--- [2] GWEN NVFP4 ({gwen_time:.1f}s) ---")
    print(gwen_out)

    print(f"\n--- [3] llama.cpp Q4_K_M ({llama_time:.1f}s) ---")
    print(llama_out)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
