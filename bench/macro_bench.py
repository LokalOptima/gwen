#!/usr/bin/env python3
"""
Macro-level benchmarks comparing gwen vs llama.cpp.

Measures:
  - Time-to-first-token (TTFT) at various prompt lengths
  - Decode tokens/second (steady-state)
  - Peak VRAM usage
  - Per-component latency breakdown (if gwen supports it)

Usage:
    uv run bench/macro_bench.py                     # full benchmark suite
    uv run bench/macro_bench.py --quick              # abbreviated run
    uv run bench/macro_bench.py --gwen-only          # skip llama.cpp
    uv run bench/macro_bench.py --llamacpp-only      # skip gwen
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = Path.home() / "models" / "Qwen3.5-9B-UD-Q4_K_XL.gguf"
LLAMACPP_BIN = PROJECT_ROOT / "third_party" / "llama.cpp" / "build" / "bin"
GWEN_BIN = PROJECT_ROOT / "build" / "gwen"
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"


@dataclass
class BenchResult:
    engine: str
    prompt_tokens: int
    decode_tokens: int
    ttft_ms: float          # time to first token
    decode_tok_per_s: float  # steady-state decode speed
    total_time_ms: float
    peak_vram_mb: float
    # Breakdown (gwen only)
    embed_ms: float = 0.0
    deltanet_ms: float = 0.0
    attention_ms: float = 0.0
    ffn_ms: float = 0.0
    sampling_ms: float = 0.0
    other_ms: float = 0.0


def generate_prompt(n_tokens_approx: int) -> str:
    """Generate a prompt of approximately n tokens."""
    # ~1.3 tokens per word on average
    words_needed = int(n_tokens_approx / 1.3)
    base = (
        "The transformer architecture was introduced in the paper 'Attention Is All "
        "You Need' by Vaswani et al. It uses self-attention mechanisms to process "
        "sequences in parallel rather than sequentially like RNNs. The key components "
        "are multi-head attention query key and value projections feed-forward networks "
        "layer normalization and residual connections. Modern variants include GPT BERT "
        "T5 and more recently hybrid architectures like Mamba and DeltaNet which combine "
        "linear attention with standard attention for efficiency. "
    )
    # Repeat to reach target length
    repeated = (base * (words_needed // len(base.split()) + 1))
    words = repeated.split()[:words_needed]
    return " ".join(words)


def bench_llamacpp(prompt: str, n_predict: int, warmup_runs: int = 1, bench_runs: int = 3) -> BenchResult:
    """Benchmark llama.cpp inference."""
    cmd = [
        str(LLAMACPP_BIN / "llama-cli"),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", str(n_predict),
        "--no-conversation",
        "--temp", "0",
        "-ngl", "999",  # offload all layers to GPU
        "--no-warmup",
    ]

    # Warmup
    for _ in range(warmup_runs):
        subprocess.run(cmd, capture_output=True, timeout=300)

    # Benchmark runs
    times = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if result.returncode != 0:
            print(f"  llama.cpp error: {result.stderr[:200]}", file=sys.stderr)
            return BenchResult(
                engine="llama.cpp", prompt_tokens=0, decode_tokens=0,
                ttft_ms=0, decode_tok_per_s=0, total_time_ms=0, peak_vram_mb=0
            )

    # Parse timing from llama.cpp stderr output
    # llama.cpp prints timing stats like:
    # llama_perf_context_print:        load time =   XXX.XX ms
    # llama_perf_context_print: prompt eval time =   XXX.XX ms /   NNN tokens
    # llama_perf_context_print:        eval time =   XXX.XX ms /   NNN runs
    stderr = result.stderr if result.stderr else ""
    prompt_eval_ms = 0
    eval_ms = 0
    eval_tokens = 0
    prompt_tokens = 0

    for line in stderr.split("\n"):
        if "prompt eval time" in line:
            m = re.search(r"([\d.]+)\s*ms\s*/\s*(\d+)", line)
            if m:
                prompt_eval_ms = float(m.group(1))
                prompt_tokens = int(m.group(2))
        elif "eval time" in line and "prompt" not in line:
            m = re.search(r"([\d.]+)\s*ms\s*/\s*(\d+)", line)
            if m:
                eval_ms = float(m.group(1))
                eval_tokens = int(m.group(2))

    # Fallback to wall time if parsing fails
    median_time = np.median(times) * 1000  # to ms
    if eval_tokens > 0 and eval_ms > 0:
        decode_tok_s = eval_tokens / (eval_ms / 1000)
    else:
        decode_tok_s = n_predict / (median_time / 1000) if median_time > 0 else 0

    ttft = prompt_eval_ms if prompt_eval_ms > 0 else median_time * 0.3  # rough estimate

    # Get VRAM usage
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        vram_mb = float(smi.stdout.strip())
    except Exception:
        vram_mb = 0

    return BenchResult(
        engine="llama.cpp",
        prompt_tokens=prompt_tokens or len(prompt.split()),
        decode_tokens=eval_tokens or n_predict,
        ttft_ms=ttft,
        decode_tok_per_s=decode_tok_s,
        total_time_ms=median_time,
        peak_vram_mb=vram_mb,
    )


def bench_gwen(prompt: str, n_predict: int, warmup_runs: int = 1, bench_runs: int = 3) -> BenchResult:
    """Benchmark gwen inference using gwen_bench."""
    gwen_bench_bin = PROJECT_ROOT / "build" / "gwen_bench"
    prompt_tokens = len(prompt.split())  # approximate

    # Run gwen_bench for decode throughput
    cmd = [
        str(gwen_bench_bin),
        "-m", str(MODEL_PATH),
        "-p", "0",
        "-n", str(n_predict),
        "-r", str(bench_runs),
        "-o", "csv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    decode_tps = 0.0
    if result.returncode == 0:
        # CSV format: model,size_gib,params_b,test,mean_tps,stddev_tps
        for line in result.stdout.strip().split("\n"):
            if line.startswith("qwen"):
                parts = line.split(",")
                decode_tps = float(parts[4])
                break

    # Run gwen_bench for prefill throughput (as proxy for TTFT)
    ttft_ms = 0.0
    if prompt_tokens > 0:
        cmd_pp = [
            str(gwen_bench_bin),
            "-m", str(MODEL_PATH),
            "-p", str(prompt_tokens),
            "-n", "0",
            "-r", str(bench_runs),
            "-o", "csv",
        ]
        result_pp = subprocess.run(cmd_pp, capture_output=True, text=True, timeout=300)
        if result_pp.returncode == 0:
            for line in result_pp.stdout.strip().split("\n"):
                if line.startswith("qwen"):
                    parts = line.split(",")
                    pp_tps = float(parts[4])
                    if pp_tps > 0:
                        ttft_ms = prompt_tokens / pp_tps * 1000
                    break

    total_ms = ttft_ms + (n_predict / decode_tps * 1000 if decode_tps > 0 else 0)

    return BenchResult(
        engine="gwen",
        prompt_tokens=prompt_tokens,
        decode_tokens=n_predict,
        ttft_ms=ttft_ms,
        decode_tok_per_s=decode_tps,
        total_time_ms=total_ms,
        peak_vram_mb=0,
    )


def run_benchmark_suite(args):
    """Run the full benchmark suite."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prompt_lengths = [1, 32, 128, 512, 2048] if not args.quick else [1, 128]
    decode_length = 100 if not args.quick else 20
    bench_runs = 5 if not args.quick else 2

    all_results = []

    print(f"{'='*80}")
    print(f"GWEN vs llama.cpp Benchmark Suite")
    print(f"Model: {MODEL_PATH.name}")
    print(f"GPU: RTX 5070 Ti (SM_120)")
    print(f"Decode tokens: {decode_length}")
    print(f"Runs per config: {bench_runs}")
    print(f"{'='*80}\n")

    for pl in prompt_lengths:
        prompt = generate_prompt(pl)
        print(f"\n--- Prompt length: ~{pl} tokens ---")

        if not args.gwen_only:
            print(f"  Benchmarking llama.cpp...", end=" ", flush=True)
            r_llama = bench_llamacpp(prompt, decode_length, bench_runs=bench_runs)
            all_results.append(asdict(r_llama) | {"prompt_length_target": pl})
            print(f"TTFT={r_llama.ttft_ms:.1f}ms, decode={r_llama.decode_tok_per_s:.1f} tok/s")

        if not args.llamacpp_only:
            print(f"  Benchmarking gwen...", end=" ", flush=True)
            r_gwen = bench_gwen(prompt, decode_length, bench_runs=bench_runs)
            all_results.append(asdict(r_gwen) | {"prompt_length_target": pl})
            print(f"TTFT={r_gwen.ttft_ms:.1f}ms, decode={r_gwen.decode_tok_per_s:.1f} tok/s")

        if not args.gwen_only and not args.llamacpp_only:
            if r_llama.decode_tok_per_s > 0 and r_gwen.decode_tok_per_s > 0:
                speedup = r_gwen.decode_tok_per_s / r_llama.decode_tok_per_s
                print(f"  Speedup: {speedup:.2f}x")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Engine':<12} {'Prompt':<8} {'TTFT(ms)':<10} {'Decode(tok/s)':<15} {'Total(ms)':<12} {'VRAM(MB)':<10}")
    print("-" * 67)
    for r in all_results:
        print(f"{r['engine']:<12} {r['prompt_length_target']:<8} {r['ttft_ms']:<10.1f} {r['decode_tok_per_s']:<15.1f} {r['total_time_ms']:<12.1f} {r['peak_vram_mb']:<10.0f}")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"macro_bench_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Macro-level benchmarks: gwen vs llama.cpp")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gwen-only", action="store_true")
    parser.add_argument("--llamacpp-only", action="store_true")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    run_benchmark_suite(args)


if __name__ == "__main__":
    main()
