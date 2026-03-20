# ── Paths ────────────────────────────────────────────────────────────
CACHE_DIR ?= $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/gwen
MODEL     ?= $(CACHE_DIR)/Qwen3.5-0.8B-Q4_K_M.gguf
MTP       ?= $(CACHE_DIR)/mtp_v5_sparse64.bin
MTP_HEAD  ?= $(CACHE_DIR)/mtp_lm_head_20k.bin
PROMPT    ?= The meaning of life is
N         ?= 100

GH_RELEASE = https://github.com/LokalOptima/gwen/releases/download/v1.0.0
HF_MODEL   = https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf

# ── Build ────────────────────────────────────────────────────────────
BUILD_DIR := build

.PHONY: all clean run run-mtp bench bench-mtp info test weights

all: $(BUILD_DIR)/gwen

$(BUILD_DIR)/gwen: CMakeLists.txt $(wildcard src/*.cu src/*.cpp src/kernels/*.cu include/gwen/*.h)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
	@cmake --build $(BUILD_DIR) -j$$(nproc) 2>&1 | grep -E "error|warning:|^make" || true

clean:
	rm -rf $(BUILD_DIR)

# ── Weights ──────────────────────────────────────────────────────────
$(MODEL):
	@mkdir -p $(CACHE_DIR)
	@echo "Downloading Qwen3.5-0.8B-Q4_K_M.gguf..."
	@wget -q --show-progress -O $@ $(HF_MODEL)

$(MTP):
	@mkdir -p $(CACHE_DIR)
	@echo "Downloading mtp_v5_sparse64.bin..."
	@wget -q --show-progress -O $@ $(GH_RELEASE)/mtp_v5_sparse64.bin

$(MTP_HEAD):
	@mkdir -p $(CACHE_DIR)
	@echo "Downloading mtp_lm_head_20k.bin..."
	@wget -q --show-progress -O $@ $(GH_RELEASE)/mtp_lm_head_20k.bin

weights: $(MODEL) $(MTP) $(MTP_HEAD)

# ── Run ──────────────────────────────────────────────────────────────
run: $(BUILD_DIR)/gwen $(MODEL)
	$(BUILD_DIR)/gwen --model $(MODEL) --prompt "$(PROMPT)" --max-predict $(N) --greedy

run-mtp: $(BUILD_DIR)/gwen $(MODEL) $(MTP) $(MTP_HEAD)
	$(BUILD_DIR)/gwen --model $(MODEL) --mtp $(MTP) --mtp-lm-head $(MTP_HEAD) \
		--prompt "$(PROMPT)" --max-predict $(N)

# ── Benchmark ────────────────────────────────────────────────────────
bench: $(BUILD_DIR)/gwen $(MODEL)
	$(BUILD_DIR)/gwen --model $(MODEL) --prompt "$(PROMPT)" --max-predict $(N) --greedy --benchmark

bench-mtp: $(BUILD_DIR)/gwen $(MODEL) $(MTP) $(MTP_HEAD)
	$(BUILD_DIR)/gwen --model $(MODEL) --mtp $(MTP) --mtp-lm-head $(MTP_HEAD) \
		--prompt "$(PROMPT)" --max-predict $(N) --benchmark

# ── Utility ──────────────────────────────────────────────────────────
info: $(BUILD_DIR)/gwen $(MODEL)
	$(BUILD_DIR)/gwen --model $(MODEL) --info

test: $(BUILD_DIR)/gwen
	./scripts/test_correctness.sh
