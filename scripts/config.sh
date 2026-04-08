# scripts/config.sh — shared model path defaults
# Source this at the top of any script that needs model paths.
# All variables respect env var overrides.

GWEN_CACHE="${GWEN_CACHE:-$HOME/.cache/gwen}"
MODEL="${MODEL:-$GWEN_CACHE/Qwen3.5-0.8B-Q8_0.gguf}"

# Upstream llama.cpp for correctness baseline (same GGUF, no MTP sidecar)
# Prefer system-installed, fall back to vendored third_party copy
_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${LLAMA_COMPLETION:-}" ]; then
    if command -v llama-completion &>/dev/null; then
        LLAMA_COMPLETION="$(command -v llama-completion)"
    else
        LLAMA_COMPLETION="$_PROJ_ROOT/third_party/llama.cpp/build/bin/llama-completion"
        export LD_LIBRARY_PATH="${_PROJ_ROOT}/third_party/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi
