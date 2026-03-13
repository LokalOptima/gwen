#!/bin/bash
# Record GWEN's own greedy output as golden data for regression testing.
# Run this after verifying correctness, then re-run after intentional arithmetic changes.
# Output: tests/golden_gwen/prompt_N.tokens (one token ID per line)
set -euo pipefail

MODEL="Qwen3.5-0.8B-Q4_K_M.gguf"
GWEN="./build/gwen"
GOLDEN_DIR="tests/golden_gwen"

mkdir -p "$GOLDEN_DIR"

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

echo "Recording GWEN golden outputs..."
echo "Model: $MODEL" > "$GOLDEN_DIR/README"
echo "Generated: $(date -Iseconds)" >> "$GOLDEN_DIR/README"
echo "GWEN build: $(md5sum ./build/gwen | cut -d' ' -f1)" >> "$GOLDEN_DIR/README"

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    out="$GOLDEN_DIR/prompt_${i}.tokens"

    TOKENS=$("$GWEN" --model "$MODEL" --prompt "$prompt" \
        --n-predict 30 --greedy 2>&1 \
        | grep "Generated token IDs:" | sed 's/Generated token IDs: //' | tr -s ' ')

    # Write one token per line
    echo "$TOKENS" | tr ' ' '\n' | grep -v '^$' > "$out"
    N=$(wc -l < "$out")
    echo "  [$((i+1))/${#PROMPTS[@]}] \"$prompt\" → $N tokens"
done

echo ""
echo "Golden data written to $GOLDEN_DIR/"
