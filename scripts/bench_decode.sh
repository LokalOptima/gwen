#!/bin/bash
# Decode-only benchmark: llama-bench baseline vs instrumented MTP
# Uses ONLY valid timing sources: llama-bench for tg1, MTP_STATS for speculation.
# NEVER uses wall-clock minus model-load.
#
# Usage: ./scripts/bench_decode.sh [n_tokens]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../llama-slim/build"
COMPLETION="$BUILD_DIR/bin/llama-completion"
BENCH="$BUILD_DIR/bin/llama-bench"
MODEL_BASE="$HOME/.cache/gwen/Qwen3.5-0.8B-Base-Q4_K_M.gguf"
MODEL_MTP="$HOME/.cache/gwen/Qwen3.5-0.8B-Base-mtp-Q4_K_M.gguf"
N=${1:-200}

for f in "$COMPLETION" "$BENCH"; do
    if [ ! -x "$f" ]; then
        echo "ERROR: $(basename $f) not found at $f" >&2
        exit 1
    fi
done

declare -a PROMPTS=(
    "The history of artificial intelligence begins in the 1950s when Alan Turing proposed"
    "Once upon a time in a small fishing village on the coast of Norway, there lived an old man who"
    "Dear colleagues, I am pleased to announce that our quarterly results have exceeded expectations. Revenue grew by"
    "The key difference between TCP and UDP is that TCP provides reliable, ordered delivery of data between"
    "In quantum mechanics, the uncertainty principle states that it is impossible to simultaneously know both the exact position and"
    "The transformer architecture introduced in 2017 replaced recurrent neural networks by using self-attention mechanisms to process"
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n"
    "class DatabaseConnection:\n    def __init__(self, host, port, database):\n        self.host = host\n"
    "If a train travels at 60 km/h for 2 hours, then accelerates to 90 km/h for 1.5 hours, the total distance"
    "The Fibonacci sequence is defined as F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2). The first 20 terms are"
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy"
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,"
)

declare -a LABELS=(
    "AI history"
    "Narrative"
    "Business"
    "TCP/UDP"
    "Quantum"
    "Transformer"
    "Python sort"
    "Python class"
    "Math word"
    "Fibonacci"
    "Fox repeat"
    "Counting"
)

echo "================================================================"
echo "  llama-slim Decode Benchmark"
echo "  Base: $(basename $MODEL_BASE)"
echo "  MTP:  $(basename $MODEL_MTP)"
echo "  Tokens: $N per prompt"
echo "================================================================"
echo ""

# --- Baseline: llama-bench ---
echo "=== Baseline (llama-bench tg64, FA off, 5 reps) ==="
flock --exclusive /tmp/gpu.lock "$BENCH" -m "$MODEL_BASE" -p 0 -n 64 -r 5 -fa 0 2>&1 | tail -3
echo ""
echo "=== MTP model single-token (llama-bench tg64, FA off, 5 reps) ==="
flock --exclusive /tmp/gpu.lock "$BENCH" -m "$MODEL_MTP" -p 0 -n 64 -r 5 -fa 0 2>&1 | tail -3
echo ""

# --- MTP speculative decode ---
echo "=== MTP Speculative Decode (instrumented, $N tokens) ==="
printf "%-14s %8s %8s %8s %8s %6s\n" "Prompt" "tok/s" "accept%" "draft_ms" "main_ms" "tokens"
printf "%-14s %8s %8s %8s %8s %6s\n" "---------" "------" "------" "--------" "-------" "------"

sum_tps=0
sum_accept=0
n_prompts=0

for i in "${!PROMPTS[@]}"; do
    label="${LABELS[$i]}"

    # Run MTP, capture stderr for stats
    stats=$(flock --exclusive /tmp/gpu.lock "$COMPLETION" --no-conversation \
        -m "$MODEL_MTP" -p "${PROMPTS[$i]}" -n "$N" --temp 0 2>&1 1>/dev/null \
        | grep "MTP_STATS" || echo "")

    if [ -z "$stats" ]; then
        printf "%-14s %8s %8s %8s %8s %6s\n" "$label" "ERR" "-" "-" "-" "-"
        continue
    fi

    tok_s=$(echo "$stats" | grep -oP '"tok_per_s": \K[0-9.]+' || echo "0")
    accept=$(echo "$stats" | grep -oP '"accept_rate": \K[0-9.]+' || echo "0")
    draft=$(echo "$stats" | grep -oP '"mtp_draft_avg_ms": \K[0-9.]+' || echo "0")
    main=$(echo "$stats" | grep -oP '"main_decode_ms": \K[0-9.]+' || echo "0")
    n_tok=$(echo "$stats" | grep -oP '"n_tokens": \K[0-9]+' || echo "0")

    printf "%-14s %8s %7s%% %8s %8s %6s\n" "$label" "$tok_s" "$accept" "$draft" "$main" "$n_tok"

    sum_tps=$(echo "$sum_tps + $tok_s" | bc)
    sum_accept=$(echo "$sum_accept + $accept" | bc)
    n_prompts=$((n_prompts + 1))
done

if [ $n_prompts -gt 0 ]; then
    avg_tps=$(echo "scale=1; $sum_tps / $n_prompts" | bc)
    avg_accept=$(echo "scale=1; $sum_accept / $n_prompts" | bc)
    echo ""
    printf "%-14s %8s %7s%%\n" "AVERAGE" "$avg_tps" "$avg_accept"
fi

echo ""
echo "================================================================"
