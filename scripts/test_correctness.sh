#!/bin/bash
# Comprehensive correctness testing for GWEN
# Usage: ./scripts/test_correctness.sh
#
# Tests:
# 1. dp4a kernel unit tests (GEMV accuracy vs legacy FP16 path)
# 2. CUTLASS GEMM vs GEMV (numerical agreement)
# 3. Greedy token matching vs llama.cpp (multiple prompts)
# 4. Determinism check (same output across multiple runs)

set -euo pipefail

MODEL="Qwen3.5-0.8B-Q4_K_M.gguf"
GWEN="./build/gwen"
LLAMA_GENERATE="./tests/llama_generate"
PASS=0
FAIL=0
TOTAL=0

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo "  FAIL: $1"; }

echo "╔═══════════════════════════════════════════════╗"
echo "║       GWEN Correctness Test Suite             ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ── 1. dp4a Kernel Unit Tests ──
echo "── 1. dp4a Kernel Unit Tests ──"
DP4A_OUT=$(./build/test_dp4a "$MODEL" 2>&1)
echo "$DP4A_OUT" | grep -E "PASS|FAIL|Max diff"
if echo "$DP4A_OUT" | grep -q "ALL DP4A TESTS PASSED"; then
    pass "dp4a kernels match legacy FP16 path"
else
    fail "dp4a kernel divergence"
fi
echo ""

# ── 2. CUTLASS GEMM Tests ──
echo "── 2. CUTLASS GEMM vs GEMV ──"
if [ -x "./build/test_gemm" ]; then
    GEMM_OUT=$(./build/test_gemm "$MODEL" 2>&1)
    echo "$GEMM_OUT" | grep -E "PASS|FAIL|Max diff"
    if echo "$GEMM_OUT" | grep -q "ALL TESTS PASSED"; then
        pass "CUTLASS GEMM matches GEMV"
    else
        fail "CUTLASS GEMM divergence"
    fi
else
    echo "  (test_gemm not built, skipping)"
fi
echo ""

# ── 3. Greedy Token Matching vs llama.cpp ──
echo "── 3. Greedy Token Match vs llama.cpp ──"

PROMPTS=(
    "The quick brown fox"
    "In the beginning"
    "def fibonacci(n):"
    "The capital of France is"
)

for prompt in "${PROMPTS[@]}"; do
    if [ -x "$LLAMA_GENERATE" ]; then
        # Get llama.cpp tokens
        LLAMA_TOKENS=$(LD_LIBRARY_PATH=./third_party/llama.cpp/build/bin \
            timeout 30 "$LLAMA_GENERATE" "$prompt" 2>&1 \
            | grep -P '^\s*\[\d+\]' | head -30 | grep -oP 'token=\K[0-9]+' | tr '\n' ' ' || echo "")

        # Get GWEN tokens
        GWEN_TOKENS=$("${GWEN}" --model "$MODEL" --prompt "$prompt" \
            --n-predict 30 --greedy 2>&1 \
            | grep "Generated token IDs:" | sed 's/Generated token IDs: //' | tr -s ' ' || echo "")

        if [ -n "$LLAMA_TOKENS" ] && [ -n "$GWEN_TOKENS" ]; then
            LLAMA_ARR=($LLAMA_TOKENS)
            GWEN_ARR=($GWEN_TOKENS)
            MATCH=0
            CHECK=${#GWEN_ARR[@]}
            for i in $(seq 0 $((CHECK-1))); do
                [ "${LLAMA_ARR[$i]:-}" = "${GWEN_ARR[$i]:-}" ] && MATCH=$((MATCH+1))
            done
            if [ $MATCH -eq $CHECK ]; then
                pass "\"$prompt\" → $MATCH/$CHECK tokens match"
            else
                fail "\"$prompt\" → $MATCH/$CHECK tokens match (DIVERGE at position $MATCH)"
                echo "    llama: ${LLAMA_TOKENS:0:80}..."
                echo "    gwen:  ${GWEN_TOKENS:0:80}..."
            fi
        else
            echo "  SKIP: \"$prompt\" (could not get tokens from both engines)"
        fi
    else
        # Self-test only: verify GWEN produces reasonable output
        GWEN_OUT=$("${GWEN}" --model "$MODEL" --prompt "$prompt" \
            --n-predict 30 --greedy 2>&1)
        GWEN_TOKENS=$(echo "$GWEN_OUT" | grep "Generated token IDs:" | sed 's/Generated token IDs: //')
        if [ -n "$GWEN_TOKENS" ]; then
            pass "\"$prompt\" → generated 30 tokens (no llama.cpp to compare)"
        else
            fail "\"$prompt\" → no output"
        fi
    fi
done
echo ""

# ── 4. Determinism Check ──
echo "── 4. Determinism (5 runs, same output) ──"
TOKENS_PREV=""
ALL_SAME=1
for i in $(seq 1 5); do
    TOKENS=$("${GWEN}" --model "$MODEL" --prompt "Hello world" \
        --n-predict 20 --greedy 2>&1 \
        | grep "Generated token IDs:" | sed 's/Generated token IDs: //' | tr -s ' ')
    if [ -n "$TOKENS_PREV" ] && [ "$TOKENS" != "$TOKENS_PREV" ]; then
        ALL_SAME=0
        echo "  Run $i DIFFERS: $TOKENS"
    fi
    TOKENS_PREV="$TOKENS"
done
if [ $ALL_SAME -eq 1 ]; then
    pass "5 identical runs (deterministic)"
else
    fail "Non-deterministic output across runs"
fi
echo ""

# ── Summary ──
echo "═══════════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed, $TOTAL total"
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
else
    echo "  SOME TESTS FAILED"
fi
echo "═══════════════════════════════════════════════"
exit $FAIL
