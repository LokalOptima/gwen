#!/bin/bash
# MTP speculative decoding benchmark
# Usage: ./scripts/bench_mtp.sh [n_tokens] [runs_per_prompt]
set -euo pipefail

MODEL="${HOME}/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
MTP="train/runs/mtp_v3_k4096/mtp_finetuned.bin"
GWEN="./build/gwen"
N=${1:-200}
RUNS=${2:-3}

declare -a PROMPTS=(
    "The capital of France is"
    "Once upon a time in a small village, there lived a young woman who dreamed of becoming a scientist."
    "Explain the difference between TCP and UDP protocols in networking."
    "The key principles of quantum mechanics include"
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
    "In a groundbreaking paper published in Nature, researchers demonstrated that"
    "Dear hiring manager, I am writing to express my strong interest in the Senior Software Engineer position at"
    "To configure a Kubernetes cluster with high availability, you need to follow these steps:"
)

declare -a LABELS=(
    "Repetitive"
    "Narrative"
    "TCP/UDP"
    "Quantum"
    "Code"
    "Scientific"
    "Formal"
    "DevOps"
)

echo "================================================================"
echo "  GWEN Speculative Decode Benchmark"
echo "  Model: Qwen3.5-0.8B-Q4_K_M | MTP: K=4096 fine-tuned"
echo "  n_predict=$N, runs=$RUNS per prompt"
echo "================================================================"
echo ""

printf "%-14s %8s %8s %8s %8s\n" "Prompt" "No MTP" "MTP" "Accept" "Speedup"
printf "%-14s %8s %8s %8s %8s\n" "--------------" "--------" "--------" "--------" "--------"

total_base=0
total_mtp=0
n_prompts=0

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    label="${LABELS[$i]}"

    # Baseline (average of $RUNS)
    base_sum=0
    for r in $(seq 1 $RUNS); do
        tps=$(flock --shared /tmp/gpu.lock "$GWEN" --model "$MODEL" \
              --prompt "$prompt" --n-predict "$N" --greedy --benchmark 2>&1 \
              | grep -oP '"decode_tok_per_s": \K[0-9.]+')
        base_sum=$(echo "$base_sum + $tps" | bc)
    done
    base_avg=$(echo "scale=1; $base_sum / $RUNS" | bc)

    # MTP (average of $RUNS)
    mtp_sum=0
    accept=""
    for r in $(seq 1 $RUNS); do
        out=$(flock --shared /tmp/gpu.lock "$GWEN" --model "$MODEL" --mtp "$MTP" \
              --prompt "$prompt" --n-predict "$N" --greedy --benchmark 2>&1)
        tps=$(echo "$out" | grep -oP '"decode_tok_per_s": \K[0-9.]+')
        mtp_sum=$(echo "$mtp_sum + $tps" | bc)
        [ -z "$accept" ] && accept=$(echo "$out" | grep -oP '[0-9.]+(?=% acceptance)') || true
    done
    mtp_avg=$(echo "scale=1; $mtp_sum / $RUNS" | bc)

    speedup=$(echo "scale=1; ($mtp_avg / $base_avg - 1) * 100" | bc 2>/dev/null || echo "?")

    printf "%-14s %7.1f  %7.1f  %6.1f%%  %+6.1f%%\n" \
           "$label" "$base_avg" "$mtp_avg" "$accept" "$speedup"

    total_base=$(echo "$total_base + $base_avg" | bc)
    total_mtp=$(echo "$total_mtp + $mtp_avg" | bc)
    n_prompts=$((n_prompts + 1))
done

avg_base=$(echo "scale=1; $total_base / $n_prompts" | bc)
avg_mtp=$(echo "scale=1; $total_mtp / $n_prompts" | bc)
avg_speedup=$(echo "scale=1; ($avg_mtp / $avg_base - 1) * 100" | bc 2>/dev/null || echo "?")

echo ""
printf "%-14s %7.1f  %7.1f           %+6.1f%%\n" "AVERAGE" "$avg_base" "$avg_mtp" "$avg_speedup"
echo ""
echo "================================================================"
