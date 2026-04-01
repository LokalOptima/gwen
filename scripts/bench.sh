#!/bin/bash
# GWEN vs llama.cpp — performance benchmark
# Uses gwen_bench and llama-bench with identical parameters.
#
# Usage: ./scripts/bench.sh [--quick]
set -euo pipefail

MODEL="$HOME/.cache/gwen/Qwen3.5-0.8B-Base-Q4_K_M.gguf"
GWEN_BENCH="./build/gwen_bench"
GWEN="./build/gwen"

# Defaults
PP=512; TG=128; REPS=5
if [[ "${1:-}" == "--quick" ]]; then
    PP=128; TG=64; REPS=3
fi

# Verify prerequisites
if [ ! -f "$MODEL" ]; then echo "Error: model not found at $MODEL" >&2; exit 1; fi
if [ ! -x "$GWEN_BENCH" ]; then echo "Error: gwen_bench not built (run make)" >&2; exit 1; fi
if ! command -v llama-bench &>/dev/null; then echo "Error: llama-bench not in PATH" >&2; exit 1; fi

echo "================================================================"
echo "  GWEN vs llama.cpp — Performance Benchmark"
echo "  Model:  $(basename "$MODEL")"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Config: pp${PP} tg${TG} × ${REPS} reps"
echo "================================================================"
echo ""

# ── GWEN ──
echo "── GWEN ──"
flock --exclusive /tmp/gpu.lock "$GWEN_BENCH" -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS"
echo ""

# ── llama.cpp ──
echo "── llama.cpp ──"
flock --exclusive /tmp/gpu.lock llama-bench -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -ngl 99
echo ""

# ── MTP speculative decode ──
echo "── GWEN + MTP (speculative decode, 200 tokens) ──"
MTP_OUT=$(flock --exclusive /tmp/gpu.lock "$GWEN" --model "$MODEL" \
    "In a shocking turn of events, scientists discovered that" \
    --max-predict 200 --greedy --benchmark 2>&1)
MTP_TPS=$(echo "$MTP_OUT" | grep "Tokens/sec:" | grep -oP '[0-9.]+')
MTP_STATS=$(echo "$MTP_OUT" | grep "MTP stats:" | sed 's/^MTP stats: //')
echo "  Effective throughput: ${MTP_TPS} tok/s"
echo "  MTP: ${MTP_STATS}"
echo ""
echo "================================================================"
