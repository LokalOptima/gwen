#!/bin/bash
# GWEN vs llama.cpp — comprehensive benchmark
# Usage: ./scripts/bench.sh [n_tokens]

MODEL="Qwen3.5-0.8B-Q4_K_M.gguf"
GWEN="./build/gwen"
LLAMA_SIMPLE="./third_party/llama.cpp/build/bin/llama-simple"
LLAMA_GENERATE="./tests/llama_generate"
N=${1:-100}

echo "╔══════════════════════════════════════════════════╗"
echo "║     GWEN vs llama.cpp — Performance Report      ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║ Model:  Qwen3.5-0.8B-Q4_K_M (Q4_K/Q5_K/Q6_K)  ║"
echo "║ GPU:    NVIDIA RTX 5070 Ti (SM_120, 16GB GDDR7) ║"
echo "║ Decode: $N tokens, greedy                        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Correctness ──
echo "── Correctness ──"
if [ -x "$LLAMA_GENERATE" ]; then
    LLAMA_TOKENS=$(LD_LIBRARY_PATH=./third_party/llama.cpp/build/bin \
        timeout 30 "$LLAMA_GENERATE" 2>&1 \
        | grep -P '^\s*\[\d+\]' | head -30 | grep -oP 'token=\K[0-9]+' | tr '\n' ' ' || echo "")
    GWEN_TOKENS=$("${GWEN}" --model "$MODEL" --prompt "The quick brown fox" \
        --n-predict 30 --greedy 2>&1 \
        | grep "Generated token IDs:" | sed 's/Generated token IDs: //' | tr -s ' ' || echo "")

    if [ -n "$LLAMA_TOKENS" ] && [ -n "$GWEN_TOKENS" ]; then
        LLAMA_ARR=($LLAMA_TOKENS)
        GWEN_ARR=($GWEN_TOKENS)
        MATCH=0
        TOTAL=${#GWEN_ARR[@]}
        for i in $(seq 0 $((TOTAL-1))); do
            [ "${LLAMA_ARR[$i]:-}" = "${GWEN_ARR[$i]:-}" ] && MATCH=$((MATCH+1))
        done
        echo "  Greedy token match:  $MATCH/$TOTAL vs llama.cpp"
        [ $MATCH -eq $TOTAL ] && echo "  Status:              EXACT MATCH" || echo "  Status:              DIVERGE"
    else
        echo "  (Could not compare tokens)"
    fi
fi

DP4A_RESULT=$(./build/test_dp4a "$MODEL" 2>&1 | tail -1 || echo "FAILED")
echo "  dp4a kernel tests:   $DP4A_RESULT"
echo ""

# ── 2. Performance ──
echo "── Decode Speed ──"

# GWEN
GWEN_JSON=$("${GWEN}" --model "$MODEL" --prompt "The meaning of life is" \
    --n-predict "$N" --greedy --benchmark 2>&1 | head -1)
GWEN_DECODE=$(echo "$GWEN_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"decode_tok_per_s\"]:.1f}')" 2>/dev/null || echo "??")
GWEN_TTFT=$(echo "$GWEN_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"ttft_ms\"]:.1f}')" 2>/dev/null || echo "??")
GWEN_VRAM=$(echo "$GWEN_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"peak_vram_mb\"]:.0f}')" 2>/dev/null || echo "??")

# llama.cpp
LLAMA_OUT=$(timeout 30 "$LLAMA_SIMPLE" -m "$MODEL" \
    -p "The meaning of life is" -n "$N" -ngl 99 --temp 0 2>&1 || true)
LLAMA_TPS=$(echo "$LLAMA_OUT" | grep "eval time" | grep -v prompt \
    | grep -oP '[0-9.]+ tokens per second' | grep -oP '^[0-9.]+' || echo "")
LLAMA_MS=$(echo "$LLAMA_OUT" | grep "eval time" | grep -v prompt \
    | grep -oP '\(\s*[0-9.]+' | grep -oP '[0-9.]+' || echo "")
LLAMA_PROMPT=$(echo "$LLAMA_OUT" | grep "prompt eval" \
    | grep -oP '[0-9.]+ tokens per second' | grep -oP '^[0-9.]+' || echo "")

# Compute comparison
if [ -n "$LLAMA_TPS" ] && [ "$GWEN_DECODE" != "??" ]; then
    RATIO=$(python3 -c "
g, l = float('$GWEN_DECODE'), float('$LLAMA_TPS')
pct = (g - l) / l * 100
sign = '+' if pct >= 0 else ''
print(f'{sign}{pct:.1f}%')
" 2>/dev/null || echo "??")
    GWEN_MS_VAL=$(python3 -c "print(f'{1000/float(\"$GWEN_DECODE\"):.2f}')" 2>/dev/null || echo "??")
else
    RATIO="N/A"
    GWEN_MS_VAL="??"
fi

printf "  %-22s %10s %12s %8s\n" "" "GWEN" "llama.cpp" "Delta"
printf "  %-22s %10s %12s %8s\n" "────────────────────" "────────" "──────────" "──────"
printf "  %-22s %7s t/s %9s t/s %8s\n" "Decode throughput" "$GWEN_DECODE" "${LLAMA_TPS:-??}" "$RATIO"
printf "  %-22s %7s ms  %9s ms\n" "Per-token latency" "$GWEN_MS_VAL" "${LLAMA_MS:-??}"
printf "  %-22s %7s ms  %9s t/s\n" "TTFT / Prompt eval" "$GWEN_TTFT" "${LLAMA_PROMPT:-??}"
printf "  %-22s %7s MB\n" "Peak VRAM" "$GWEN_VRAM"
echo ""

# ── 3. Breakdown ──
echo "── Forward Pass Profile ──"
PROF_OUT=$(./build/profile_forward "$MODEL" 2>&1)
echo "$PROF_OUT" | grep -E "Average:|Min:|Max:" | head -3 | sed 's/^/  /'
echo ""
echo "$PROF_OUT" | grep -E "dp4a:|legacy:|recurrence:|All 24|Non-GEMV|Theoretical|Actual|Bandwidth" | sed 's/^/  /'
echo ""

# ── 4. Verdict ──
echo "── Verdict ──"
if [ -n "$LLAMA_TPS" ] && [ "$GWEN_DECODE" != "??" ]; then
    if python3 -c "exit(0 if float('$GWEN_DECODE') >= float('$LLAMA_TPS') else 1)" 2>/dev/null; then
        echo "  GWEN is FASTER ($RATIO vs llama.cpp)"
    else
        echo "  GWEN is SLOWER ($RATIO vs llama.cpp)"
    fi
else
    echo "  (llama.cpp benchmark unavailable)"
fi
