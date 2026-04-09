#!/bin/bash
# Transcription cleanup quality test
# Feeds messy STT output to gwen and checks if cleanup is reasonable.
#
# Usage: ./scripts/test_transcription.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
GWEN="$BUILD_DIR/bin/gwen"

if [ ! -x "$GWEN" ]; then
    echo "ERROR: gwen not found at $GWEN" >&2
    echo "Run: make gwen" >&2
    exit 1
fi

SYSTEM="You are a speech transcript editor. You receive raw speech-to-text output and return a cleaned version. Remove all filler words (um, uh, like, you know, so, right, basically, I mean, ok so). Remove repeated words and false starts. Add proper punctuation and capitalization. Do not change the meaning. Do not add new words. Do not explain anything. Return only the cleaned transcript."

declare -a INPUTS=(
    # Filler words
    "um so like the thing is uh we need to make sure that the the model is like loading properly and stuff you know"

    # Repetitions and false starts
    "I was going to I was going to say that the the results look pretty good actually they look really good"

    # No punctuation, run-on
    "so basically what happened was the server went down at around 3 am and nobody noticed until the morning and by then we had lost about six hours of data"

    # Heavy filler + hedging
    "yeah so uh I think I think what we should probably do is uh maybe look at the the latency numbers again because uh they seemed a bit off to me"

    # Technical speech with errors
    "the cuda kernel is launching like sixty four threads per block and uh we need at least uh one twenty eight to get full occupancy on the on the sm"

    # Conversational with interjections
    "ok so right so the thing about the transformer architecture right is that it uses self attention which is like quadratic in sequence length so thats why we switched to deltanet"

    # Short utterance with filler
    "uh yeah that sounds good lets do that"

    # Numbers and measurements
    "we are getting about uh six hundred and thirty tokens per second which is uh roughly uh thirty percent faster than the baseline"
)

declare -a LABELS=(
    "filler_words"
    "repetitions"
    "no_punctuation"
    "heavy_filler"
    "technical"
    "conversational"
    "short"
    "numbers"
)

echo "================================================================"
echo "  Transcription Cleanup Test"
echo "================================================================"
echo ""

for i in "${!INPUTS[@]}"; do
    label="${LABELS[$i]}"
    input="${INPUTS[$i]}"

    echo "--- $label ---"
    echo "IN:  $input"

    output=$("$GWEN" -s "$SYSTEM" -n 200 "$input" 2>/dev/null)

    echo "OUT: $output"
    echo ""
done
