#!/bin/bash
# GWEN Correctness Test Suite
#
# Self-contained regression tests — no live llama.cpp dependency.
#
# Tests:
# 1. dp4a kernel unit tests (GEMV accuracy vs legacy FP16 path)
# 2. CUTLASS GEMM vs GEMV (numerical agreement)
# 3. Per-layer golden check vs llama.cpp reference (tests/golden/)
# 4. Self-regression: greedy tokens match GWEN's own golden data (tests/golden_gwen/)
# 5. Determinism check (same output across multiple runs)
#
# Generate golden data:
#   llama.cpp layers:  scripts/generate_golden.sh
#   GWEN tokens:       scripts/generate_gwen_golden.sh

set -euo pipefail

MODEL="Qwen3.5-0.8B-Q4_K_M.gguf"
GWEN="./build/gwen"
GOLDEN_LLAMA="tests/golden"
GOLDEN_GWEN="tests/golden_gwen"
PASS=0
FAIL=0
SKIP=0
TOTAL=0

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo "  FAIL: $1"; }
skip() { SKIP=$((SKIP+1)); echo "  SKIP: $1"; }

echo "╔═══════════════════════════════════════════════╗"
echo "║       GWEN Correctness Test Suite             ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ── 1. dp4a Kernel Unit Tests ──
echo "── 1. dp4a Kernel Unit Tests ──"
if [ -x "./build/test_dp4a" ]; then
    DP4A_OUT=$(./build/test_dp4a "$MODEL" 2>&1)
    echo "$DP4A_OUT" | grep -E "PASS|FAIL|Max diff"
    if echo "$DP4A_OUT" | grep -q "ALL DP4A TESTS PASSED"; then
        pass "dp4a kernels match legacy FP16 path"
    else
        fail "dp4a kernel divergence"
    fi
else
    skip "test_dp4a not built"
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
    skip "test_gemm not built"
fi
echo ""

# ── 3. Per-Layer Golden Check vs llama.cpp ──
echo "── 3. Per-Layer Golden Check (vs llama.cpp reference) ──"
if [ -d "$GOLDEN_LLAMA" ] && [ -d "$GOLDEN_LLAMA/prompt_0" ]; then
    LOGIT_OUT=$(uv run --with numpy scripts/test_logits.py 2>&1)
    echo "$LOGIT_OUT"

    N_LOGIT_PASS=$(echo "$LOGIT_OUT" | grep -c "PASS:" || true)
    N_LOGIT_FAIL=$(echo "$LOGIT_OUT" | grep -c "FAIL:" || true)

    if [ "$N_LOGIT_FAIL" -eq 0 ] && [ "$N_LOGIT_PASS" -gt 0 ]; then
        pass "per-layer golden check ($N_LOGIT_PASS prompts)"
    else
        fail "per-layer golden check ($N_LOGIT_FAIL failures)"
    fi
else
    skip "no llama.cpp golden data (run scripts/generate_golden.sh)"
fi
echo ""

# ── 4. Self-Regression: GWEN Greedy Token Match ──
echo "── 4. Self-Regression (GWEN greedy tokens vs golden) ──"

PROMPTS=(
    "The quick brown fox"
    "In the beginning"
    "def fibonacci(n):"
    "The capital of France is"
    "1+1=2. 2+2=4. 3+3="
    "Once upon a time there was"
    "import numpy as np"
    "The meaning of life is"
)

if [ -d "$GOLDEN_GWEN" ]; then
    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        golden_file="$GOLDEN_GWEN/prompt_${i}.tokens"

        if [ ! -f "$golden_file" ]; then
            skip "\"$prompt\" (no golden file)"
            continue
        fi

        # Get GWEN's current greedy tokens
        GWEN_TOKENS=$("$GWEN" --model "$MODEL" --prompt "$prompt" \
            --n-predict 30 --greedy 2>&1 \
            | grep "Generated token IDs:" | sed 's/Generated token IDs: //' | tr -s ' ')

        if [ -z "$GWEN_TOKENS" ]; then
            fail "\"$prompt\" (GWEN produced no output)"
            continue
        fi

        # Convert to one-per-line for comparison
        CURRENT=$(echo "$GWEN_TOKENS" | tr ' ' '\n' | grep -v '^$')
        GOLDEN=$(< "$golden_file")

        if [ "$CURRENT" = "$GOLDEN" ]; then
            pass "\"$prompt\" → 30/30 tokens match golden"
        else
            # Count matching prefix
            MATCH=0
            TOTAL_TOKENS=$(echo "$GOLDEN" | wc -l)
            while IFS= read -r gold_tok <&3 && IFS= read -r cur_tok <&4; do
                if [ "$gold_tok" = "$cur_tok" ]; then
                    MATCH=$((MATCH+1))
                else
                    break
                fi
            done 3< <(echo "$GOLDEN") 4< <(echo "$CURRENT")

            fail "\"$prompt\" → $MATCH/$TOTAL_TOKENS tokens match (diverges at position $MATCH)"
        fi
    done
else
    skip "no GWEN golden data (run scripts/generate_gwen_golden.sh)"
fi
echo ""

# ── 5. Determinism Check ──
echo "── 5. Determinism (5 runs, same output) ──"
TOKENS_PREV=""
ALL_SAME=1
for i in $(seq 1 5); do
    TOKENS=$("$GWEN" --model "$MODEL" --prompt "Hello world" \
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
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped, $TOTAL total"
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
else
    echo "  SOME TESTS FAILED"
fi
echo "═══════════════════════════════════════════════"
exit $FAIL
