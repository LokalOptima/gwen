#!/bin/bash
# GWEN Correctness Test Suite
#
# Tests:
#   1. Generation correctness: greedy decode matches golden reference
#   2. Prefill smoke test: gwen_bench -p 32 -n 0 -r 1 completes without error
#   3. Decode smoke test: gwen_bench -p 0 -n 32 -r 1 completes without error
#   4. Determinism: 3 identical runs produce the same output
#
# Usage:
#   scripts/test_correctness.sh                  # run tests (generate golden if missing)
#   scripts/test_correctness.sh --generate-golden # force regenerate golden reference
#
# Golden data is stored in tests/golden_greedy/. Regenerate after intentional
# arithmetic changes (new quantization, kernel rewrites, etc).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${HOME}/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"
GWEN="${ROOT}/build/gwen"
BENCH="${ROOT}/build/gwen_bench"
GOLDEN_DIR="${ROOT}/tests/golden_greedy"
GOLDEN_FILE="${GOLDEN_DIR}/generation.txt"

PROMPT="1 2 3 4 5 6 7 8"
MAX_PREDICT=20

PASS=0
FAIL=0
TOTAL=0

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo "  FAIL: $1"; }

# ── Preflight checks ──
if [ ! -x "$GWEN" ]; then
    echo "ERROR: gwen binary not found at $GWEN"
    echo "Run: cmake --build build"
    exit 1
fi
if [ ! -x "$BENCH" ]; then
    echo "ERROR: gwen_bench binary not found at $BENCH"
    echo "Run: cmake --build build"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL"
    exit 1
fi

# ── Generate golden reference ──
generate_golden() {
    echo "Generating golden reference..."
    mkdir -p "$GOLDEN_DIR"

    local output
    output=$(flock --shared /tmp/gpu.lock \
        "$GWEN" --model "$MODEL" --no-mtp --greedy --max-predict "$MAX_PREDICT" "$PROMPT" 2>/dev/null)

    if [ -z "$output" ]; then
        echo "ERROR: gwen produced no output"
        exit 1
    fi

    echo "$output" > "$GOLDEN_FILE"
    echo "Golden reference saved to $GOLDEN_FILE"
    echo "Content: $(cat "$GOLDEN_FILE")"
}

FORCE_GOLDEN=0
if [ "${1:-}" = "--generate-golden" ]; then
    FORCE_GOLDEN=1
fi

if [ "$FORCE_GOLDEN" -eq 1 ] || [ ! -f "$GOLDEN_FILE" ]; then
    generate_golden
    if [ "$FORCE_GOLDEN" -eq 1 ]; then
        echo "Golden reference regenerated. Run without --generate-golden to test."
        exit 0
    fi
fi

echo ""
echo "========================================"
echo "  GWEN Correctness Test Suite"
echo "========================================"
echo ""

# ── 1. Generation Correctness ──
echo "-- 1. Generation Correctness (greedy decode vs golden) --"

ACTUAL=$(flock --shared /tmp/gpu.lock \
    "$GWEN" --model "$MODEL" --no-mtp --greedy --max-predict "$MAX_PREDICT" "$PROMPT" 2>/dev/null)
EXPECTED=$(cat "$GOLDEN_FILE")

if [ "$ACTUAL" = "$EXPECTED" ]; then
    pass "greedy output matches golden reference"
else
    fail "greedy output differs from golden reference"
    echo "    Expected: $EXPECTED"
    echo "    Actual:   $ACTUAL"
fi
echo ""

# ── 2. Prefill Smoke Test ──
echo "-- 2. Prefill Smoke Test (gwen_bench -p 32 -n 0 -r 1) --"

BENCH_PREFILL_OUT=$(flock --shared /tmp/gpu.lock \
    "$BENCH" -m "$MODEL" -p 32 -n 0 -r 1 2>&1) || true
BENCH_PREFILL_EXIT=$?

if [ $BENCH_PREFILL_EXIT -eq 0 ] && echo "$BENCH_PREFILL_OUT" | grep -qE "tok/s|pp32"; then
    pass "prefill benchmark completed successfully"
else
    fail "prefill benchmark failed (exit=$BENCH_PREFILL_EXIT)"
    echo "    Output: $(echo "$BENCH_PREFILL_OUT" | tail -5)"
fi
echo ""

# ── 3. Decode Smoke Test ──
echo "-- 3. Decode Smoke Test (gwen_bench -p 0 -n 32 -r 1) --"

BENCH_DECODE_OUT=$(flock --shared /tmp/gpu.lock \
    "$BENCH" -m "$MODEL" -p 0 -n 32 -r 1 2>&1) || true
BENCH_DECODE_EXIT=$?

if [ $BENCH_DECODE_EXIT -eq 0 ] && echo "$BENCH_DECODE_OUT" | grep -qE "tok/s|tg32"; then
    pass "decode benchmark completed successfully"
else
    fail "decode benchmark failed (exit=$BENCH_DECODE_EXIT)"
    echo "    Output: $(echo "$BENCH_DECODE_OUT" | tail -5)"
fi
echo ""

# ── 4. Determinism ──
echo "-- 4. Determinism (3 runs, same output) --"

PREV=""
ALL_SAME=1
for i in 1 2 3; do
    RUN_OUT=$(flock --shared /tmp/gpu.lock \
        "$GWEN" --model "$MODEL" --no-mtp --greedy --max-predict "$MAX_PREDICT" "$PROMPT" 2>/dev/null)
    if [ -n "$PREV" ] && [ "$RUN_OUT" != "$PREV" ]; then
        ALL_SAME=0
        echo "    Run $i differs from run $((i-1))"
    fi
    PREV="$RUN_OUT"
done

if [ $ALL_SAME -eq 1 ]; then
    pass "3 identical runs (deterministic)"
else
    fail "non-deterministic output across runs"
fi
echo ""

# ── Summary ──
echo "========================================"
echo "  Results: $PASS passed, $FAIL failed ($TOTAL total)"
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
else
    echo "  SOME TESTS FAILED"
fi
echo "========================================"
exit $FAIL
