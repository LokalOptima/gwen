#!/bin/bash
# GWEN vs llama.cpp — prefill benchmark
# Runs both at multiple prompt lengths, reports tok/s with statistics.
#
# Usage: ./scripts/bench_prefill.sh [--runs N] [--lengths "128 256 512"]
#        ./scripts/bench_prefill.sh --gwen-only

set -euo pipefail

GWEN="./build/gwen"
LLAMA_BENCH="llama-bench"
LLAMA_MODEL="$HOME/.cache/gwen/Qwen3.5-0.8B-Base-Q4_K_M.gguf"
RUNS=5
LENGTHS="128 256 512"
GWEN_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --lengths) LENGTHS="$2"; shift 2 ;;
        --gwen-only) GWEN_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Generate deterministic prompts of target token count
# "word " tokenizes to 1 token consistently
gen_prompt() {
    local n=$1
    python3 -c "print(' '.join(['word'] * $n))"
}

# Run GWEN prefill benchmark, return tok/s values
bench_gwen() {
    local pp=$1
    local prompt
    prompt=$(gen_prompt "$pp")
    local results=()
    for i in $(seq 1 "$RUNS"); do
        local json
        json=$(flock --exclusive /tmp/gpu.lock "$GWEN" "$prompt" --max-predict 1 --greedy --benchmark 2>&1 | grep '{"prompt_tokens"')
        local tps
        tps=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"prompt_tokens\"] / (d[\"ttft_ms\"]/1000):.1f}')" 2>/dev/null || echo "0")
        results+=("$tps")
    done
    # Print all values space-separated
    echo "${results[*]}"
}

# Run llama.cpp prefill benchmark
bench_llama() {
    local pp=$1
    local out
    out=$(flock --exclusive /tmp/gpu.lock "$LLAMA_BENCH" -m "$LLAMA_MODEL" -p "$pp" -n 0 -r "$RUNS" 2>&1 | grep "pp${pp}")
    # Extract tok/s: "25301.19 ± 2645.48"
    local tps_mean tps_std
    tps_mean=$(echo "$out" | grep -oP '[\d.]+(?= ±)' || echo "0")
    tps_std=$(echo "$out" | grep -oP '(?<=± )[\d.]+' || echo "0")
    echo "$tps_mean $tps_std"
}

# Compute mean and std from space-separated values
compute_stats() {
    python3 -c "
import sys
vals = [float(x) for x in sys.argv[1:] if float(x) > 0]
if not vals:
    print('0.0 0.0')
else:
    import statistics
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f'{m:.1f} {s:.1f}')
" "$@"
}

echo "================================================================"
echo "  GWEN vs llama.cpp — Prefill Benchmark"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Runs: $RUNS per configuration"
echo "================================================================"
echo ""
printf "  %-8s  %20s  %20s  %8s\n" "Prompt" "GWEN (tok/s)" "llama.cpp (tok/s)" "Ratio"
printf "  %-8s  %20s  %20s  %8s\n" "------" "--------------------" "--------------------" "------"

for pp in $LENGTHS; do
    # GWEN
    gwen_vals=$(bench_gwen "$pp")
    read -r gwen_mean gwen_std <<< "$(compute_stats $gwen_vals)"

    if $GWEN_ONLY; then
        printf "  pp%-5s  %10s ± %-7s\n" "$pp" "$gwen_mean" "$gwen_std"
    else
        # llama.cpp
        read -r llama_mean llama_std <<< "$(bench_llama "$pp")"
        ratio=$(python3 -c "
g, l = float('$gwen_mean'), float('$llama_mean')
if l > 0: print(f'{g/l:.2f}x')
else: print('N/A')
" 2>/dev/null || echo "N/A")
        printf "  pp%-5s  %10s ± %-7s  %10s ± %-7s  %8s\n" \
            "$pp" "$gwen_mean" "$gwen_std" "$llama_mean" "$llama_std" "$ratio"
    fi
done

echo ""
echo "  Higher is better. Ratio > 1.0 means GWEN is faster."
echo "================================================================"
