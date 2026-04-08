# scripts/config.sh — shared model path defaults
# Source this at the top of any script that needs model paths.
# All variables respect env var overrides.
#
# MODEL_BASE and MODEL_MTP both point to the stock Q8_0 GGUF.
# MTP is enabled by the sidecar (*-mtp.gguf) which is auto-discovered.

GWEN_CACHE="${GWEN_CACHE:-$HOME/.cache/gwen}"
MODEL_BASE="${MODEL_BASE:-$GWEN_CACHE/Qwen3.5-0.8B-Q8_0.gguf}"
MODEL_MTP="${MODEL_MTP:-$GWEN_CACHE/Qwen3.5-0.8B-Q8_0.gguf}"
