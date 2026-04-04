#!/bin/bash
# Comprehensive benchmark of different restricted LM head sizes for MTP.
# Tests each head on all 12 prompts at 200 tokens.
# Reports: per-prompt tok/s, acceptance rate, and MTP draft time.
#
# Usage: ./scripts/bench_lm_head_sizes.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../llama-slim/build"
COMPLETION="$BUILD_DIR/bin/llama-completion"
MODEL_MTP="$HOME/.cache/gwen/Qwen3.5-0.8B-Base-mtp-Q4_K_M.gguf"
N=200

if [ ! -x "$COMPLETION" ]; then
    echo "ERROR: llama-completion not found at $COMPLETION" >&2
    exit 1
fi

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
    "AI_history"
    "Narrative"
    "Business"
    "TCP_UDP"
    "Quantum"
    "Transformer"
    "Python_sort"
    "Python_class"
    "Math_word"
    "Fibonacci"
    "Fox_repeat"
    "Counting"
)

declare -a CATEGORIES=(
    "NL" "NL" "NL" "Tech" "Tech" "Tech"
    "Code" "Code" "Math" "Math" "Rep" "Rep"
)

# Head sizes to test (0 = no restricted head / full 248K vocab)
# Edit this array to select which sizes to benchmark
declare -a HEAD_SIZES=(${MTP_HEAD_SIZES:-0 4000 10000 20000 30000 50000})

echo "================================================================"
echo "  MTP Restricted LM Head Size Comparison"
echo "  Model: $(basename $MODEL_MTP)"
echo "  Tokens: $N per prompt, 12 prompts"
echo "================================================================"
echo ""

# CSV output for easy analysis
CSV_FILE="/tmp/mtp_head_comparison.csv"
echo "head_size,prompt,category,tok_s,accept_pct,draft_ms,main_ms,n_tokens" > "$CSV_FILE"

for HEAD_K in "${HEAD_SIZES[@]}"; do
    if [ "$HEAD_K" -eq 0 ]; then
        HEAD_DESC="full_248K"
        # Force no restricted head — set to nonexistent path so default isn't used
        export LLAMA_MTP_LM_HEAD="/nonexistent_disable_restricted_head"
    else
        HEAD_FILE="$HOME/.cache/gwen/lm_head_top${HEAD_K}.bin"
        if [ ! -f "$HEAD_FILE" ]; then
            echo "SKIP: $HEAD_FILE not found"
            continue
        fi
        HEAD_DESC="top_${HEAD_K}"
        export LLAMA_MTP_LM_HEAD="$HEAD_FILE"
    fi

    echo "=== Head: $HEAD_DESC ==="
    printf "%-14s %8s %8s %8s %8s\n" "Prompt" "tok/s" "accept%" "draft_ms" "main_ms"
    printf "%-14s %8s %8s %8s %8s\n" "---------" "------" "------" "--------" "-------"

    sum_tps=0
    sum_accept=0
    n_ok=0

    for i in "${!PROMPTS[@]}"; do
        label="${LABELS[$i]}"
        cat="${CATEGORIES[$i]}"

        stats=$(flock --exclusive /tmp/gpu.lock "$COMPLETION" --no-conversation \
            -m "$MODEL_MTP" -p "${PROMPTS[$i]}" -n "$N" --temp 0 2>&1 1>/dev/null \
            | grep "MTP_STATS" || echo "")

        if [ -z "$stats" ]; then
            printf "%-14s %8s %8s %8s %8s\n" "$label" "ERR" "-" "-" "-"
            echo "${HEAD_K},${label},${cat},0,0,0,0,0" >> "$CSV_FILE"
            continue
        fi

        tok_s=$(echo "$stats" | grep -oP '"tok_per_s": \K[0-9.]+' || echo "0")
        accept=$(echo "$stats" | grep -oP '"accept_rate": \K[0-9.]+' || echo "0")
        draft=$(echo "$stats" | grep -oP '"mtp_draft_avg_ms": \K[0-9.]+' || echo "0")
        main=$(echo "$stats" | grep -oP '"main_decode_ms": \K[0-9.]+' || echo "0")
        n_tok=$(echo "$stats" | grep -oP '"n_tokens": \K[0-9]+' || echo "0")

        printf "%-14s %8s %7s%% %8s %8s\n" "$label" "$tok_s" "$accept" "$draft" "$main"
        echo "${HEAD_K},${label},${cat},${tok_s},${accept},${draft},${main},${n_tok}" >> "$CSV_FILE"

        sum_tps=$(echo "$sum_tps + $tok_s" | bc)
        sum_accept=$(echo "$sum_accept + $accept" | bc)
        n_ok=$((n_ok + 1))
    done

    if [ $n_ok -gt 0 ]; then
        avg_tps=$(echo "scale=1; $sum_tps / $n_ok" | bc)
        avg_accept=$(echo "scale=1; $sum_accept / $n_ok" | bc)
        printf "%-14s %8s %7s%%\n" "AVERAGE" "$avg_tps" "$avg_accept"
    fi
    echo ""
done

echo "================================================================"
echo "  CSV data saved to: $CSV_FILE"
echo "================================================================"
echo ""

# Print summary table
echo "=== SUMMARY ==="
echo ""
printf "%-10s %8s %8s %8s %8s %8s %8s\n" "Head" "Avg_tps" "Avg_acc%" "Best_tps" "Worst_tps" "Draft_ms" "Coverage"
printf "%-10s %8s %8s %8s %8s %8s %8s\n" "--------" "------" "------" "--------" "---------" "--------" "--------"

for HEAD_K in "${HEAD_SIZES[@]}"; do
    if [ "$HEAD_K" -eq 0 ]; then
        HEAD_DESC="full_248K"
        COV="100.0"
    else
        HEAD_DESC="top_${HEAD_K}"
        case $HEAD_K in
            4000)  COV="84.7" ;;
            10000) COV="91.7" ;;
            20000) COV="95.9" ;;
            30000) COV="97.9" ;;
            50000) COV="99.1" ;;
            *)     COV="?" ;;
        esac
    fi

    # Parse from CSV
    if grep -q "^${HEAD_K}," "$CSV_FILE"; then
        avg_tps=$(grep "^${HEAD_K}," "$CSV_FILE" | awk -F, '{sum+=$4; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}')
        avg_acc=$(grep "^${HEAD_K}," "$CSV_FILE" | awk -F, '{sum+=$5; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}')
        best_tps=$(grep "^${HEAD_K}," "$CSV_FILE" | awk -F, 'BEGIN{m=0} {if($4>m) m=$4} END {printf "%.1f", m}')
        worst_tps=$(grep "^${HEAD_K}," "$CSV_FILE" | awk -F, 'BEGIN{m=99999} {if($4<m && $4>0) m=$4} END {printf "%.1f", m}')
        avg_draft=$(grep "^${HEAD_K}," "$CSV_FILE" | awk -F, '{sum+=$6; n++} END {if(n>0) printf "%.2f", sum/n; else print "0"}')
        printf "%-10s %8s %7s%% %8s %9s %8s %7s%%\n" "$HEAD_DESC" "$avg_tps" "$avg_acc" "$best_tps" "$worst_tps" "$avg_draft" "$COV"
    fi
done
echo ""
