#!/bin/bash
# Test for thinking loops in the instruct model.
# Runs diverse prompts with long generation, then analyzes <think> blocks
# for repeated n-gram patterns that indicate the model got stuck.
#
# Usage: ./scripts/test_think_loops.sh [--model PATH] [--tokens N]
#
# Requires: llama-completion built, instruct model GGUF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
COMPLETION="$BUILD_DIR/bin/llama-completion"
OUT_DIR="$SCRIPT_DIR/../test_think_loop_output"
ANALYZER="$SCRIPT_DIR/analyze_think_loops.py"

# Defaults — instruct Q8_0 model, 8192 tokens
MODEL="${MODEL:-$HOME/models/gguf/Qwen3.5-0.8B-Q8_0.gguf}"
N_TOKENS=8192
SAMPLING="--greedy"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2"; shift 2 ;;
        --tokens)   N_TOKENS="$2"; shift 2 ;;
        --sampling) SAMPLING="$2"; shift 2 ;;
        --outdir)   OUT_DIR="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -x "$COMPLETION" ]; then
    echo "ERROR: llama-completion not found at $COMPLETION" >&2
    echo "Run: make completion" >&2
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

# Prompts that exercise extended reasoning — likely to trigger loops if the model is prone
declare -a NAMES=(
    "sqrt2_irrational"
    "logic_puzzle"
    "prime_listing"
    "pnp_debate"
    "quintic_impossible"
    "critical_path"
    "chess_middlegame"
    "recursive_proof"
)

declare -a PROMPTS=(
    "Prove that the square root of 2 is irrational."

    "Five people — Alice, Bob, Charlie, Diana, and Eve — each live in a different colored house (red, blue, green, yellow, white) in a row, own a different pet (dog, cat, bird, fish, hamster), and drink a different beverage (coffee, tea, milk, juice, water). Given: Alice lives in the red house. The green house is immediately to the left of the white house. The person in the green house drinks coffee. Bob owns a bird. The person in the yellow house drinks tea. The person in the middle house drinks milk. Eve lives in the first house. Charlie drinks juice. The person who owns a hamster lives next to the person in the blue house. The person in the yellow house lives next to the person who owns a fish. Eve lives next to the blue house. Diana drinks water. Who owns the fish?"

    "List all prime numbers between 1 and 500 and count how many there are."

    "Is P equal to NP? Present the strongest arguments for both P=NP and P≠NP, then explain which side most experts lean toward and why."

    "Find a closed-form solution to the general quintic equation using only radicals. Show your work step by step."

    "You have 8 tasks with dependencies: A->B, A->C, B->D, C->D, D->E, B->F, F->G, E->G, G->H. Each task takes a different amount of time: A=3, B=2, C=4, D=1, E=5, F=3, G=2, H=1. What is the critical path and minimum total time? Show all calculations."

    "In a chess position, White has: Kg1, Qd1, Rf1, Ra1, Bc1, Nf3, pawns a2 b2 c2 d4 e5 f2 g2 h2. Black has: Kg8, Qd8, Rf8, Ra8, Bc8, Nc6, pawns a7 b7 c7 d5 e6 f7 g7 h7. Find the best continuation for White (3 moves deep). Explain each candidate move."

    "Write a formal grammar (in BNF notation) for the language of all balanced parentheses strings. Then prove by structural induction that your grammar generates exactly the set of balanced parentheses strings and nothing else."
)

mkdir -p "$OUT_DIR"

echo "=== Thinking Loop Test ==="
echo "Model:    $MODEL"
echo "Tokens:   $N_TOKENS"
echo "Sampling: $SAMPLING"
echo "Output: $OUT_DIR"
echo ""

for i in "${!PROMPTS[@]}"; do
    name="${NAMES[$i]}"
    prompt="${PROMPTS[$i]}"
    outfile="$OUT_DIR/${name}.txt"

    echo -n "  [$((i+1))/${#PROMPTS[@]}] $name ... "
    t_start=$(date +%s%N)

    flock --shared /tmp/gpu.lock \
        "$COMPLETION" --single-turn --jinja \
            -m "$MODEL" -p "$prompt" -n "$N_TOKENS" $SAMPLING \
            --reasoning on \
            2>/dev/null > "$outfile"

    t_end=$(date +%s%N)
    elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    words=$(wc -w < "$outfile")
    printf "done (%d.%ds, %d words)\n" $((elapsed_ms / 1000)) $(( (elapsed_ms % 1000) / 100 )) "$words"
done

echo ""
echo "--- Analysis ---"
python3 "$ANALYZER" "$OUT_DIR"
