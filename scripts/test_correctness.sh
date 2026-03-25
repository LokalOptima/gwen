#!/bin/bash
# GWEN Correctness Test Suite
#
# Tests GWEN against llama.cpp golden references using prefill teacher forcing:
# for each prefix length, feed the golden text as prompt, predict 1 token,
# check it matches the golden continuation. No cascade divergence possible.
#
# Tests:
#   1. Teacher-forced prefill: cut golden text at ~20 word boundaries,
#      predict next token at each, compare against llama.cpp ground truth
#   2. Free generation: 6 diverse prompts, full output compared to golden
#   3. Prefill smoke test (gwen_bench pp512)
#   4. Decode smoke test (gwen_bench tg32)
#   5. Determinism (3 identical runs)
#
# Usage:
#   scripts/test_correctness.sh               # run tests
#   scripts/test_correctness.sh --regenerate  # regenerate golden from llama.cpp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${HOME}/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"
GWEN="${ROOT}/build/gwen"
BENCH="${ROOT}/build/gwen_bench"
LLAMA="${HOME}/git/llama.cpp/build/bin/llama-simple"
GOLDEN_DIR="${ROOT}/tests/golden_greedy"
N_PREDICT=30

# Free generation prompts
PROMPTS=(
  "The capital of France is"
  "def fibonacci(n):"
  "If x + 3 = 7, then x ="
  "The HTTP status code 404 means"
  "In the year 1969, humans first"
  "Water boils at 100 degrees Celsius, which is"
)

PASS=0
FAIL=0
TOTAL=0

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo "  FAIL: $1"; }

# ── Preflight ──
for bin in "$GWEN" "$BENCH"; do
  if [ ! -x "$bin" ]; then
    echo "ERROR: $bin not found. Run: cmake --build build"
    exit 1
  fi
done
if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found at $MODEL"
  exit 1
fi

# ── Regenerate golden ──
if [ "${1:-}" = "--regenerate" ]; then
  if [ ! -x "$LLAMA" ]; then
    echo "ERROR: llama-simple not found at $LLAMA"
    exit 1
  fi
  mkdir -p "$GOLDEN_DIR"

  echo "Regenerating golden references from llama.cpp..."

  # Teacher forcing golden: long sequence from short seed
  echo "  teacher_forcing.txt (100 tokens from 'The Eiffel Tower')"
  "$LLAMA" -m "$MODEL" -n 100 "The Eiffel Tower" 2>/dev/null > "${GOLDEN_DIR}/teacher_forcing.txt"

  # Free generation golden: one per prompt
  for i in $(seq 0 $((${#PROMPTS[@]}-1))); do
    echo "  prompt_${i}.txt: '${PROMPTS[$i]}'"
    "$LLAMA" -m "$MODEL" -n "$N_PREDICT" "${PROMPTS[$i]}" 2>/dev/null > "${GOLDEN_DIR}/prompt_${i}.txt"
  done

  echo "Done. Commit tests/golden_greedy/ then run without --regenerate."
  exit 0
fi

# ── Check golden exists ──
if [ ! -f "${GOLDEN_DIR}/teacher_forcing.txt" ]; then
  echo "ERROR: golden reference missing. Run: scripts/test_correctness.sh --regenerate"
  exit 1
fi

echo ""
echo "========================================"
echo "  GWEN Correctness Test Suite"
echo "  Model: $(basename "$MODEL")"
echo "========================================"
echo ""

# ── 1. Teacher-Forced Prefill ──
echo "-- 1. Teacher-Forced Prefill (predict next token at each prefix length) --"

GOLDEN_TEXT=$(cat "${GOLDEN_DIR}/teacher_forcing.txt")
GOLDEN_LEN=${#GOLDEN_TEXT}

# Build word boundary positions (cut at spaces)
POSITIONS=()
for i in $(seq 0 $((GOLDEN_LEN - 1))); do
  if [ "${GOLDEN_TEXT:$i:1}" = " " ]; then
    POSITIONS+=("$i")
  fi
done

# Sample ~20 evenly spaced word boundaries
N_POS=${#POSITIONS[@]}
if [ "$N_POS" -lt 20 ]; then
  STEP=1
else
  STEP=$((N_POS / 20))
fi

TF_PASS=0
TF_FAIL=0
TF_TOTAL=0

for idx in $(seq 0 $STEP $((N_POS - 2))); do
  POS=${POSITIONS[$idx]}
  # Skip very short prefixes (< 20 chars) — too ambiguous for meaningful comparison
  if [ "$POS" -lt 20 ]; then continue; fi
  PREFIX="${GOLDEN_TEXT:0:$POS}"
  EXPECTED_REST="${GOLDEN_TEXT:$POS}"

  # Run GWEN: feed prefix as prompt, predict 1 token
  GWEN_OUT=$("$GWEN" --model "$MODEL" --no-mtp --greedy --max-predict 1 "$PREFIX" 2>/dev/null) || true
  GWEN_CONTINUATION="${GWEN_OUT#"$PREFIX"}"

  # Check: does the golden continuation start with what GWEN predicted?
  if [ -n "$GWEN_CONTINUATION" ] && [ "${EXPECTED_REST#"$GWEN_CONTINUATION"}" != "$EXPECTED_REST" ]; then
    TF_PASS=$((TF_PASS + 1))
  else
    TF_FAIL=$((TF_FAIL + 1))
    # Show context around divergence
    EXPECTED_NEXT="${EXPECTED_REST:0:20}"
    echo "    FAIL at word $idx (char $POS): gwen='${GWEN_CONTINUATION:0:20}' expected='${EXPECTED_NEXT}'"
  fi
  TF_TOTAL=$((TF_TOTAL + 1))
done

if [ $TF_FAIL -eq 0 ]; then
  pass "teacher forcing: ${TF_PASS}/${TF_TOTAL} positions match"
else
  fail "teacher forcing: ${TF_PASS}/${TF_TOTAL} positions match ($TF_FAIL diverged)"
fi
echo ""

# ── 2. Free Generation vs Golden ──
echo "-- 2. Free Generation vs llama.cpp Golden --"

for i in $(seq 0 $((${#PROMPTS[@]}-1))); do
  prompt="${PROMPTS[$i]}"
  golden="${GOLDEN_DIR}/prompt_${i}.txt"
  if [ ! -f "$golden" ]; then
    fail "prompt $i: golden missing"
    continue
  fi
  expected=$(cat "$golden")
  actual=$("$GWEN" --model "$MODEL" --no-mtp --greedy --max-predict "$N_PREDICT" "$prompt" 2>/dev/null) || true

  if [ "$actual" = "$expected" ]; then
    pass "prompt $i: '${prompt:0:40}'"
  else
    fail "prompt $i: '${prompt:0:40}'"
    diff <(echo "$expected") <(echo "$actual") 2>/dev/null | head -4 | sed 's/^/    /' || true
  fi
done
echo ""

# ── 3. Prefill Smoke Test ──
echo "-- 3. Prefill Smoke Test (pp512) --"
BENCH_OUT=$("$BENCH" -m "$MODEL" -p 512 -n 0 -r 1 2>&1) || true
if echo "$BENCH_OUT" | grep -qE "pp512"; then
  pass "prefill pp512 completed"
else
  fail "prefill pp512 failed"
  echo "    $(echo "$BENCH_OUT" | tail -3)"
fi
echo ""

# ── 4. Decode Smoke Test ──
echo "-- 4. Decode Smoke Test (tg32) --"
BENCH_OUT=$("$BENCH" -m "$MODEL" -p 0 -n 32 -r 1 2>&1) || true
if echo "$BENCH_OUT" | grep -qE "tg32"; then
  pass "decode tg32 completed"
else
  fail "decode tg32 failed"
  echo "    $(echo "$BENCH_OUT" | tail -3)"
fi
echo ""

# ── 5. Determinism ──
echo "-- 5. Determinism (3 runs) --"
PREV=""
ALL_SAME=1
for run in 1 2 3; do
  out=$("$GWEN" --model "$MODEL" --no-mtp --greedy --max-predict 20 "The meaning of life is" 2>/dev/null) || true
  if [ -n "$PREV" ] && [ "$out" != "$PREV" ]; then
    ALL_SAME=0
  fi
  PREV="$out"
done
if [ $ALL_SAME -eq 1 ]; then
  pass "3 identical runs (deterministic)"
else
  fail "non-deterministic output"
fi
echo ""

# ── Summary ──
echo "========================================"
echo "  Results: $PASS/$TOTAL passed"
if [ $FAIL -eq 0 ]; then
  echo "  ALL TESTS PASSED"
else
  echo "  $FAIL TESTS FAILED"
fi
echo "========================================"
exit $FAIL
