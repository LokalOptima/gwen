#!/usr/bin/env python3
"""GWEN correctness test: compare per-layer hidden states and logits against golden llama.cpp data.

For each prompt:
  1. Run GWEN with GWEN_DUMP_LAYERS=1 to dump per-layer hidden states
  2. Load golden llama.cpp dumps from tests/golden/
  3. Report cosine similarity and max absolute diff at every layer
  4. Compare final logit distributions (KL divergence)

Output: a per-layer distance table that shows exactly where divergence originates.
"""
import struct
import subprocess
import sys
import os
from pathlib import Path

import numpy as np


GOLDEN_DIR = Path("tests/golden")
GWEN = "./build/gwen"
MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf"

PROMPTS = [
    "The quick brown fox",
    "In the beginning",
    "def fibonacci(n):",
    "The capital of France is",
    "1+1=2. 2+2=4. 3+3=",
    "Once upon a time there was",
    "import numpy as np",
    "The meaning of life is",
]

# Layer types for display
LAYER_TYPES = {}
for i in range(24):
    LAYER_TYPES[i] = "FA" if i in [3, 7, 11, 15, 19, 23] else "DN"

# Acceptance: logit cosine > 0.985 catches real bugs (broken kernels drop to <0.95)
# while tolerating worst-case GEMM precision drift on arithmetic prompts (cos≈0.989).
# See blog/07-numerical-precision-investigation.md for analysis.
MIN_LOGIT_COSINE = 0.985


N_EMBED = 1024  # Qwen3.5-0.8B hidden size


def load_bin(path: Path, last_token_only: bool = False) -> np.ndarray:
    """Load binary dump: int32 count, then float32 data.
    If last_token_only and count > N_EMBED, extract the last N_EMBED floats
    (last token's hidden state from a multi-token prompt batch).
    """
    with open(path, "rb") as f:
        count = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype=np.float32).copy()
    if last_token_only and len(data) > N_EMBED and len(data) % N_EMBED == 0:
        data = data[-N_EMBED:]
    return data


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def kl_div(p_logits, q_logits):
    """KL(softmax(p) || softmax(q))."""
    p = p_logits - np.max(p_logits)
    q = q_logits - np.max(q_logits)
    lse_p = np.log(np.sum(np.exp(p)))
    lse_q = np.log(np.sum(np.exp(q)))
    lsp = p - lse_p
    lsq = q - lse_q
    return float(np.sum(np.exp(lsp) * (lsp - lsq)))


def run_gwen_dump(prompt: str):
    """Run GWEN with layer dumps, return True if successful."""
    env = dict(os.environ)
    env["GWEN_DUMP_LAYERS"] = "1"
    env["GWEN_GEMM_DECODE"] = "1"
    result = subprocess.run(
        [GWEN, "--model", MODEL, "--prompt", prompt,
         "--n-predict", "1", "--greedy"],
        capture_output=True, text=True, env=env, timeout=60
    )
    return result.returncode == 0


def test_prompt(idx: int, prompt: str) -> tuple[bool, list[str]]:
    """Test one prompt. Returns (passed, output_lines)."""
    golden = GOLDEN_DIR / f"prompt_{idx}"
    if not golden.exists():
        return True, [f'  SKIP: "{prompt}" (no golden data)']

    # Run GWEN
    if not run_gwen_dump(prompt):
        return False, [f'  FAIL: "{prompt}" (GWEN crashed)']

    lines = [f'  "{prompt}"']
    passed = True

    # Compare embedding (last token of prompt for multi-token)
    gwen_embed = Path("/tmp/gwen_embed.bin")
    gold_embed = golden / "llama_embed.bin"
    if gwen_embed.exists() and gold_embed.exists():
        g = load_bin(gwen_embed, last_token_only=True)
        l = load_bin(gold_embed, last_token_only=True)
        if np.array_equal(g, l):
            lines.append("    embed: EXACT")
        else:
            cos = cosine(g, l)
            lines.append(f"    embed: cos={cos:.8f} max_abs={np.max(np.abs(g-l)):.6f}")
            if cos < 0.9999:
                passed = False

    # Compare per-layer hidden states
    lines.append(f"    {'Layer':<12s} {'cos':>10s} {'max_abs':>10s} {'rms':>10s}")
    lines.append("    " + "-" * 46)

    for layer in range(24):
        lt = LAYER_TYPES[layer]
        for stage in ["attn_res", "post_ffn"]:
            gwen_path = Path(f"/tmp/gwen_{stage}_{layer}.bin")
            gold_path = golden / f"llama_{stage}_{layer}.bin"
            if not gwen_path.exists() or not gold_path.exists():
                continue

            g = load_bin(gwen_path, last_token_only=True)
            l = load_bin(gold_path, last_token_only=True)
            diff = g - l
            cos = cosine(g, l)
            max_abs = float(np.max(np.abs(diff)))
            rms = float(np.sqrt(np.mean(diff**2)))

            tag = f"L{layer:02d}.{stage[:4]} ({lt})"
            lines.append(f"    {tag:<12s} {cos:>10.6f} {max_abs:>10.6f} {rms:>10.6f}")

    # Compare logits (result_output is n_vocab-sized, not n_embed, so no last_token extraction)
    gwen_logits = Path("/tmp/gwen_result_output.bin")
    gold_logits = golden / "llama_result_output.bin"
    if gwen_logits.exists() and gold_logits.exists():
        g = load_bin(gwen_logits)
        l = load_bin(gold_logits)
        if len(g) != len(l):
            lines.append(f"    logits: SIZE MISMATCH gwen={len(g)} llama={len(l)}")
            passed = False
            return passed, lines
        cos = cosine(g, l)
        kl = kl_div(l, g)
        max_abs = float(np.max(np.abs(g - l)))

        # Top-1 agreement
        g_top = int(np.argmax(g))
        l_top = int(np.argmax(l))
        top1 = "MATCH" if g_top == l_top else f"MISMATCH (gwen={g_top} llama={l_top})"

        lines.append("    " + "-" * 46)
        lines.append(f"    logits:  cos={cos:.6f}  KL={kl:.6f}  max_abs={max_abs:.4f}  top1={top1}")

        if cos < MIN_LOGIT_COSINE:
            lines.append(f"    FAIL: logit cosine {cos:.6f} < {MIN_LOGIT_COSINE}")
            passed = False

    status = "PASS" if passed else "FAIL"
    lines[0] = f"  {status}: {lines[0].strip()}"
    return passed, lines


def main():
    if not GOLDEN_DIR.exists():
        print("No golden data. Run: scripts/generate_golden.sh")
        sys.exit(1)

    print("=" * 60)
    print("GWEN Layer Correctness Test")
    print("=" * 60)
    if (GOLDEN_DIR / "README").exists():
        print((GOLDEN_DIR / "README").read_text().strip())
    print()

    total = 0
    n_pass = 0
    n_fail = 0

    for i, prompt in enumerate(PROMPTS):
        golden = GOLDEN_DIR / f"prompt_{i}"
        if not golden.exists():
            continue
        total += 1
        ok, lines = test_prompt(i, prompt)
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        for line in lines:
            print(line)
        print()

    print("=" * 60)
    print(f"  {n_pass} passed, {n_fail} failed, {total} total")
    print(f"  Criteria: logit cosine > {MIN_LOGIT_COSINE}")
    if n_fail == 0 and total > 0:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
