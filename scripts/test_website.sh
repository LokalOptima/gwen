#!/bin/bash
# Website generation quality test across quantization levels
# Generates a single-page HTML/CSS website about Qwen3.5 at each precision
# using llama-completion with instruct models.
# Saves outputs to test_website_output/ for manual inspection in a browser.
#
# Usage: ./scripts/test_website.sh [--open]
#   --open: open all generated pages in the default browser after generation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
COMPLETION="$BUILD_DIR/bin/llama-completion"
OUT_DIR="$SCRIPT_DIR/../test_website_output"

# Instruct models at different quantization levels (chat template baked into GGUF)
declare -A MODELS=(
    ["BF16"]="$HOME/models/gguf/Qwen3.5-0.8B-BF16.gguf"
    ["Q8_0"]="$HOME/models/gguf/Qwen3.5-0.8B-Q8_0.gguf"
    ["Q6_K"]="$HOME/models/gguf/Qwen3.5-0.8B-Q6_K.gguf"
    ["Q4_K_M"]="$HOME/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
)

PROMPT="Create a single-page website about the Qwen3.5 language model. \
Write the complete HTML file with inline CSS (no external dependencies). \
Include: a hero section with the model name and tagline, an architecture overview \
section explaining the hybrid DeltaNet + Transformer design (24 layers: \
3 DeltaNet + 1 full attention repeated 6 times), a features section with cards \
for key capabilities, a comparison table of model sizes (0.8B, 4B, 9B, 32B), \
and a footer. Use a dark theme with a color palette based on deep blue (#0a0e27) \
and cyan (#00d4ff) accents. Make it responsive, modern, and visually polished \
with subtle CSS animations. Output ONLY the HTML code, nothing else."

OPEN=0
if [[ "${1:-}" == "--open" ]]; then
    OPEN=1
fi

if [ ! -x "$COMPLETION" ]; then
    echo "ERROR: llama-completion not found at $COMPLETION" >&2
    echo "Run: cd build && cmake --build . --target llama-completion -j\$(nproc)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=== Website Generation Test ==="
echo "Binary: $COMPLETION"
echo "Output: $OUT_DIR"
echo ""

for quant in BF16 Q8_0 Q6_K Q4_K_M; do
    model="${MODELS[$quant]}"
    outfile="$OUT_DIR/website_${quant}.html"

    if [ ! -f "$model" ]; then
        echo "SKIP: $quant — model not found: $model"
        continue
    fi

    echo -n "Generating $quant... "
    t_start=$(date +%s%N)

    "$COMPLETION" --no-conversation \
        -m "$model" -p "$PROMPT" -n 4096 --greedy \
        2>/dev/null > "$outfile"

    t_end=$(date +%s%N)
    elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    size=$(wc -c < "$outfile")

    printf "done (%d.%ds, %d bytes) → %s\n" \
        $((elapsed_ms / 1000)) $(( (elapsed_ms % 1000) / 100 )) "$size" "$outfile"
done

echo ""
echo "=== Done. Open the HTML files in a browser to compare quality. ==="
ls -lh "$OUT_DIR"/website_*.html 2>/dev/null

if [ $OPEN -eq 1 ]; then
    for f in "$OUT_DIR"/website_*.html; do
        xdg-open "$f" 2>/dev/null &
    done
fi
