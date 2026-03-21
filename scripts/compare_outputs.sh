#!/bin/bash
# Compare GWEN vs llama.cpp outputs token-by-token (greedy decoding)
# Usage: ./scripts/compare_outputs.sh [n_tokens] [prompt]

MODEL="/home/lapo/git/gwen/Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf"
GWEN="./build/gwen"
LLAMA="/home/lapo/git/gwen/third_party/llama.cpp/build/bin/llama-simple"
N=${1:-30}
PROMPT="${2:-Once upon a time}"

echo "=== Output Comparison: GWEN vs llama.cpp ==="
echo "Prompt: \"$PROMPT\", Tokens: $N, Greedy decoding"
echo ""

# Get GWEN output
GWEN_OUT=$($GWEN --model "$MODEL" --prompt "$PROMPT" --n-predict "$N" --greedy 2>&1)
GWEN_TOKENS=$(echo "$GWEN_OUT" | grep "^Generated token IDs:" | sed 's/Generated token IDs: //')
GWEN_TEXT=$(echo "$GWEN_OUT" | sed -n '/^Generating/,/^Generated/{ /^Generating/d; /^Generated/d; p; }' | head -1)

echo "GWEN tokens:  $GWEN_TOKENS"
echo "GWEN text:    $(echo "$GWEN_OUT" | grep -A1 "^$PROMPT" | tail -1)"
echo ""

# Get llama.cpp output (capture raw output)
LLAMA_OUT=$(timeout 30 $LLAMA -m "$MODEL" -p "$PROMPT" -n "$N" -ngl 99 --temp 0 2>&1)
# llama-simple prints generated text to stdout mixed with debug
LLAMA_TEXT=$(echo "$LLAMA_OUT" | grep "^main: decoded" | head -1)

echo "llama.cpp:    $LLAMA_TEXT"
echo ""

# The real comparison needs logits or token IDs from both
# For now, visual comparison of the generated text
echo "--- GWEN full output ---"
echo "$GWEN_OUT" | sed -n "/^$PROMPT/,/^$/p" | head -5
echo ""
echo "--- Done ---"
