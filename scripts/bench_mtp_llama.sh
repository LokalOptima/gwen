#!/bin/bash
# MTP speculative decoding benchmark for llama-slim
# Tests diverse prompts at realistic lengths, measures acceptance and throughput.
#
# Usage: ./scripts/bench_mtp_llama.sh [n_tokens] [--restricted]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
BUILD_DIR="$SCRIPT_DIR/../llama-slim/build"
COMPLETION="$BUILD_DIR/bin/llama-completion"
LM_HEAD="$HOME/.cache/gwen/lm_head_top50000.bin"

N=${1:-200}
RESTRICTED=0
if [[ "${2:-}" == "--restricted" ]] || [[ "${1:-}" == "--restricted" ]]; then
    RESTRICTED=1
    [[ "${1:-}" == "--restricted" ]] && N=200
fi

if [ ! -x "$COMPLETION" ]; then
    echo "Error: llama-completion not found at $COMPLETION" >&2
    echo "Run: cd llama-slim/build && cmake --build . --target llama-completion" >&2
    exit 1
fi

# Diverse prompts covering different content types and acceptance patterns
declare -a PROMPTS=(
    # Natural language (high acceptance expected)
    "The history of artificial intelligence begins in the 1950s when Alan Turing proposed"
    "Once upon a time in a small fishing village on the coast of Norway, there lived an old man who"
    "Dear colleagues, I am pleased to announce that our quarterly results have exceeded expectations. Revenue grew by"

    # Technical/scientific (medium-high acceptance)
    "The key difference between TCP and UDP is that TCP provides reliable, ordered delivery of data between"
    "In quantum mechanics, the uncertainty principle states that it is impossible to simultaneously know both the exact position and"
    "The transformer architecture introduced in 2017 replaced recurrent neural networks by using self-attention mechanisms to process"

    # Code (lower acceptance — more tokens outside common vocab)
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n"
    "class DatabaseConnection:\n    def __init__(self, host, port, database):\n        self.host = host\n"

    # Math/reasoning (variable acceptance)
    "If a train travels at 60 km/h for 2 hours, then accelerates to 90 km/h for 1.5 hours, the total distance"
    "The Fibonacci sequence is defined as F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2). The first 20 terms are"

    # Repetitive/formulaic (very high acceptance)
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
echo "  llama-slim MTP Speculative Decode Benchmark"
echo "  Model: $(basename $MODEL_MTP)"
if [ $RESTRICTED -eq 1 ]; then
    echo "  LM Head: restricted 50K ($(basename $LM_HEAD))"
    export LLAMA_MTP_LM_HEAD="$LM_HEAD"
else
    echo "  LM Head: full vocab (248K)"
    unset LLAMA_MTP_LM_HEAD 2>/dev/null || true
fi
echo "  Tokens: $N per prompt"
echo "================================================================"
echo ""

# Step 1: Generate baseline (non-MTP) for correctness comparison
echo "Generating baseline (non-MTP)..."
BASELINE_DIR=$(mktemp -d)
for i in "${!PROMPTS[@]}"; do
    flock --shared /tmp/gpu.lock "$COMPLETION" --no-conversation \
        -m "$MODEL_BASE" -p "${PROMPTS[$i]}" -n "$N" --temp 0 -fa off 2>/dev/null \
        > "$BASELINE_DIR/$i.txt"
done
echo "Baseline generated."
echo ""

# Step 2: Run MTP and compare
printf "%-14s %7s %7s %6s %s\n" "Prompt" "tok/s" "accept" "reject" "correct"
printf "%-14s %7s %7s %6s %s\n" "--------------" "-------" "-------" "------" "-------"

total_toks=0
total_ms=0
total_accept=0
total_reject=0
n_correct=0
n_prompts=${#PROMPTS[@]}

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    label="${LABELS[$i]}"

    # Run MTP decode, capture stderr for stats and stdout for text
    MTP_OUT=$(flock --exclusive /tmp/gpu.lock "$COMPLETION" --no-conversation \
        -m "$MODEL_MTP" -p "$prompt" -n "$N" --temp 0 2>/dev/null)

    MTP_STATS=$(flock --exclusive /tmp/gpu.lock "$COMPLETION" --no-conversation \
        -m "$MODEL_MTP" -p "$prompt" -n "$N" --temp 0 2>&1 >/dev/null \
        | grep "speculative" || echo "")

    # Parse stats
    accepted=$(echo "$MTP_STATS" | grep -oP 'accepted \K[0-9]+' || echo "0")
    rejected=$(echo "$MTP_STATS" | grep -oP 'rejected \K[0-9]+' || echo "0")
    rate=$(echo "$MTP_STATS" | grep -oP 'rate \K[0-9.]+' || echo "0")

    # Check correctness against baseline
    if diff -q <(echo "$MTP_OUT") "$BASELINE_DIR/$i.txt" > /dev/null 2>&1; then
        correct="YES"
        n_correct=$((n_correct + 1))
    else
        correct="DIVERGE"
    fi

    # Calculate tok/s from acceptance stats
    n_tok=$((1 + 2*accepted + rejected))
    total_accept=$((total_accept + accepted))
    total_reject=$((total_reject + rejected))

    # Use time-based measurement for tok/s
    t_start=$(date +%s%N)
    flock --exclusive /tmp/gpu.lock "$COMPLETION" --no-conversation \
        -m "$MODEL_MTP" -p "$prompt" -n "$N" --temp 0 > /dev/null 2>&1
    t_end=$(date +%s%N)
    ms=$(( (t_end - t_start) / 1000000 ))
    # Subtract ~800ms for model loading
    ms=$((ms - 800))
    [ $ms -lt 50 ] && ms=50

    tps=0
    if [ $ms -gt 0 ]; then
        tps=$(echo "scale=1; $n_tok * 1000 / $ms" | bc)
    fi

    printf "%-14s %7s %5d   %5d   %s\n" "$label" "$tps" "$accepted" "$rejected" "$correct"
done

echo ""
total_rate="0"
total=$((total_accept + total_reject))
if [ $total -gt 0 ]; then
    total_rate=$(echo "scale=1; $total_accept * 100 / $total" | bc)
fi

echo "================================================================"
echo "  Summary"
echo "  Total accepted: $total_accept  rejected: $total_reject  rate: ${total_rate}%"
echo "  Correct: $n_correct / $n_prompts"
if [ $n_correct -eq $n_prompts ]; then
    echo "  ALL OUTPUTS MATCH BASELINE"
else
    echo "  WARNING: SOME OUTPUTS DIVERGED"
fi
echo "================================================================"

# Cleanup
rm -rf "$BASELINE_DIR"
