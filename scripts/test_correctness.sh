#!/bin/bash
# GWEN Correctness Test Suite
#
# Compares gwen greedy output against llama.cpp reference (live, no golden files).
# Uses llama-completion --no-conversation for clean text comparison.
#
# Usage: ./scripts/test_correctness.sh
set -euo pipefail

MODEL="$HOME/.cache/gwen/Qwen3.5-0.8B-Base-Q4_K_M.gguf"
GWEN="./build/gwen"
LLAMA_COMPLETION="$HOME/git/llama.cpp/build/bin/llama-completion"
N=30

PASS=0; FAIL=0; SKIP=0

pass() { PASS=$((PASS+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "  FAIL: $1"; }
skip() { SKIP=$((SKIP+1)); echo "  SKIP: $1"; }

# Verify prerequisites
if [ ! -f "$MODEL" ]; then echo "Error: model not found at $MODEL" >&2; exit 1; fi
if [ ! -x "$GWEN" ]; then echo "Error: gwen not built (run make)" >&2; exit 1; fi
if [ ! -x "$LLAMA_COMPLETION" ]; then echo "Error: llama-completion not found at $LLAMA_COMPLETION" >&2; exit 1; fi

echo "╔═══════════════════════════════════════════════╗"
echo "║       GWEN Correctness Test Suite             ║"
echo "║  Model: $(basename "$MODEL")"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ── 1. Greedy text match vs llama.cpp ──
echo "── 1. Greedy Text Match (gwen vs llama.cpp, $N tokens) ──"

PROMPTS=(
    "The quick brown fox jumps over"
    "In the beginning there was"
    "The capital of France is"
    "1+1=2. 2+2=4. 3+3="
    "Once upon a time there was a"
)

for prompt in "${PROMPTS[@]}"; do
    # gwen: stdout = prompt + generated text (clean)
    GWEN_TEXT=$(flock --shared /tmp/gpu.lock "$GWEN" --model "$MODEL" --no-mtp \
        "$prompt" --max-predict "$N" --greedy 2>/dev/null)

    # llama-completion --no-conversation: stdout = prompt + generated text (clean)
    LLAMA_TEXT=$(flock --shared /tmp/gpu.lock "$LLAMA_COMPLETION" \
        -m "$MODEL" -p "$prompt" -n "$N" -ngl 99 --temp 0 \
        --no-conversation 2>/dev/null)

    if [ -z "$GWEN_TEXT" ] || [ -z "$LLAMA_TEXT" ]; then
        skip "\"$prompt\" (empty output)"
        continue
    fi

    if [ "$GWEN_TEXT" = "$LLAMA_TEXT" ]; then
        pass "\"$prompt\" → exact match"
    else
        # Find first divergence character position
        MATCH=0
        LEN=${#GWEN_TEXT}
        [ ${#LLAMA_TEXT} -lt "$LEN" ] && LEN=${#LLAMA_TEXT}
        for ((i=0; i<LEN; i++)); do
            [ "${GWEN_TEXT:$i:1}" = "${LLAMA_TEXT:$i:1}" ] && MATCH=$((MATCH+1)) || break
        done
        fail "\"$prompt\" → diverge at char $MATCH"
        # Show the divergent portion
        GWEN_SNIP="${GWEN_TEXT:$MATCH:60}"
        LLAMA_SNIP="${LLAMA_TEXT:$MATCH:60}"
        echo "    gwen:  ...${GWEN_SNIP}"
        echo "    llama: ...${LLAMA_SNIP}"
    fi
done
echo ""

# ── 2. Determinism ──
echo "── 2. Determinism (5 runs, greedy, same output) ──"
TOKENS_PREV=""
ALL_SAME=1
for i in $(seq 1 5); do
    TOKENS=$(flock --shared /tmp/gpu.lock "$GWEN" --model "$MODEL" --no-mtp \
        "Hello world" --max-predict 20 --greedy 2>&1 \
        | grep "Generated token IDs:" | sed 's/Generated token IDs: //' | tr -s ' ')
    if [ -n "$TOKENS_PREV" ] && [ "$TOKENS" != "$TOKENS_PREV" ]; then
        ALL_SAME=0
        echo "  Run $i DIFFERS"
    fi
    TOKENS_PREV="$TOKENS"
done
if [ $ALL_SAME -eq 1 ]; then
    pass "5 identical greedy runs (deterministic)"
else
    fail "non-deterministic output across runs"
fi
echo ""

# ── 3. Sanity check — coherent output ──
echo "── 3. Sanity Check (coherent generation) ──"

SANITY_PROMPTS=(
    "Explain how a computer works in simple terms."
    "The history of the Roman Empire begins with"
    "In a groundbreaking study, researchers found that"
)

for prompt in "${SANITY_PROMPTS[@]}"; do
    OUT=$(flock --shared /tmp/gpu.lock "$GWEN" --model "$MODEL" --no-mtp \
        "$prompt" --max-predict 60 --greedy 2>/dev/null)

    # Strip prompt to get just the continuation
    CONT="${OUT#"$prompt"}"

    # Check: has enough unique words (not degenerate repetition)
    WORDS=$(echo "$CONT" | wc -w)
    UNIQUE=$(echo "$CONT" | tr -s ' ' '\n' | sort -u | wc -l)

    if [ "$WORDS" -gt 5 ] && [ "$UNIQUE" -gt 3 ]; then
        pass "\"${prompt:0:45}...\" → coherent ($WORDS words, $UNIQUE unique)"
    else
        fail "\"${prompt:0:45}...\" → degenerate output ($WORDS words, $UNIQUE unique)"
        echo "    output: ${CONT:0:100}"
    fi
done
echo ""

# ── Summary ──
TOTAL=$((PASS + FAIL))
echo "═══════════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped ($TOTAL total)"
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
else
    echo "  SOME TESTS FAILED"
fi
echo "═══════════════════════════════════════════════"
exit $FAIL
