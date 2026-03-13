#!/usr/bin/env python3
"""Benchmark batch_extract at training-realistic sizes.

Tests multiple (B, L) configurations and reports throughput.
Runs each config 5 times and reports median to filter warmup.
"""

import struct
import time
import urllib.request
import statistics

SERVER = "http://127.0.0.1:8090"


def bench_batch_extract(B: int, L: int, warmup: int = 1, runs: int = 5):
    """Benchmark /batch_extract with B sequences of length L."""
    N = B * L
    # Use realistic-ish token IDs (range 100..100+L repeated)
    flat_tokens = [100 + (i % L) for i in range(N)]
    body = struct.pack(f"<II{N}i", B, L, *flat_tokens)

    # Warmup
    for _ in range(warmup):
        req = urllib.request.Request(
            f"{SERVER}/batch_extract",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req) as resp:
            resp.read()

    # Timed runs
    times = []
    for _ in range(runs):
        req = urllib.request.Request(
            f"{SERVER}/batch_extract",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med_ms = statistics.median(times) * 1000
    toks_per_sec = N / statistics.median(times)
    out_mb = len(data) / 1024 / 1024
    print(f"  B={B:3d} L={L:3d} → N={N:6d} | {med_ms:7.1f} ms | {toks_per_sec:8.0f} tok/s | {out_mb:.1f} MB out")
    return med_ms


if __name__ == "__main__":
    print("=== Batch Extract Benchmark ===\n")

    # Warmup GPU (first call is always slow)
    print("Warming up...")
    bench_batch_extract(1, 32, warmup=2, runs=1)
    print()

    # Sweep across training-realistic configs
    print("--- Training-realistic configs ---")
    configs = [
        # (B, L) — training will use B=64, L=512 typically
        (1, 64),
        (1, 256),
        (1, 512),
        (4, 128),
        (4, 512),
        (16, 256),
        (16, 512),
        (32, 256),
        (32, 512),
        (64, 128),
        (64, 256),
        (64, 512),  # target training config
    ]

    for B, L in configs:
        try:
            bench_batch_extract(B, L)
        except Exception as e:
            print(f"  B={B:3d} L={L:3d} → FAILED: {e}")

    print("\nDone.")
