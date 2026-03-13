#!/bin/bash
# Generate golden reference data from llama.cpp for correctness testing.
# Run once when llama.cpp version or model changes.
# Output: tests/golden/<prompt_idx>/ — per-layer hidden states + logits
set -euo pipefail

MODEL="Qwen3.5-0.8B-Q4_K_M.gguf"
GOLDEN_DIR="tests/golden"
LLAMA_DUMP="./tests/llama_dump_layers"
LLAMA_GOLDEN="./tests/llama_golden"

# Build tools if needed
if [ ! -x "$LLAMA_DUMP" ]; then
    echo "Building llama_dump_layers..."
    g++ -O2 -o "$LLAMA_DUMP" tests/llama_dump_layers.cpp \
        -I third_party/llama.cpp/include \
        -I third_party/llama.cpp/ggml/include \
        -L third_party/llama.cpp/build/bin \
        -lllama -lggml -lggml-base -lggml-cpu \
        -Wl,-rpath,third_party/llama.cpp/build/bin
fi
if [ ! -x "$LLAMA_GOLDEN" ]; then
    echo "Building llama_golden..."
    g++ -O2 -o "$LLAMA_GOLDEN" tests/llama_golden.cpp \
        -I third_party/llama.cpp/include \
        -I third_party/llama.cpp/ggml/include \
        -L third_party/llama.cpp/build/bin \
        -lllama -lggml -lggml-base -lggml-cpu \
        -Wl,-rpath,third_party/llama.cpp/build/bin
fi

rm -rf "$GOLDEN_DIR"
mkdir -p "$GOLDEN_DIR"

LLAMA_COMMIT=$(cd third_party/llama.cpp && git rev-parse --short HEAD)
cat > "$GOLDEN_DIR/README" <<EOF
llama.cpp commit: $LLAMA_COMMIT
Model: $MODEL
Generated: $(date -Iseconds)
EOF

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

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    dir="$GOLDEN_DIR/prompt_$i"
    mkdir -p "$dir"

    echo "[$((i+1))/${#PROMPTS[@]}] \"$prompt\""

    # 1. Per-layer hidden states (single token = prompt only, dumps to /tmp/llama_*)
    LD_LIBRARY_PATH=./third_party/llama.cpp/build/bin \
        "$LLAMA_DUMP" "$MODEL" "$prompt" 2>/dev/null

    # Move layer dumps to golden dir
    for layer in $(seq 0 23); do
        mv "/tmp/llama_attn_res_${layer}.bin" "$dir/" 2>/dev/null || true
        mv "/tmp/llama_post_ffn_${layer}.bin" "$dir/" 2>/dev/null || true
    done
    mv /tmp/llama_embed.bin "$dir/" 2>/dev/null || true
    mv /tmp/llama_result_norm.bin "$dir/" 2>/dev/null || true
    mv /tmp/llama_result_output.bin "$dir/" 2>/dev/null || true

    # 2. Full logits for 30-token greedy generation (for logit distribution comparison)
    LD_LIBRARY_PATH=./third_party/llama.cpp/build/bin \
        "$LLAMA_GOLDEN" "$MODEL" "$prompt" 30 "$dir/logits.bin" 2>/dev/null

    echo "  → $dir/ ($(ls "$dir/" | wc -l) files)"
done

echo ""
echo "Golden data: $(du -sh "$GOLDEN_DIR" | cut -f1) in $GOLDEN_DIR/"
