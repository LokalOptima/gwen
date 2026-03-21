# ── Paths ────────────────────────────────────────────────────────────
MODEL     ?= $(HOME)/models/gguf/Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf
PROMPT    ?= The meaning of life is
N         ?= 100

# ── Build ────────────────────────────────────────────────────────────
BUILD_DIR := build

.PHONY: all clean run bench info test

all: $(BUILD_DIR)/gwen

$(BUILD_DIR)/gwen: CMakeLists.txt $(wildcard src/*.cu src/*.cpp src/kernels/*.cu include/gwen/*.h)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
	@cmake --build $(BUILD_DIR) -j$$(nproc) 2>&1 | grep -E "error|warning:|^make" || true

clean:
	rm -rf $(BUILD_DIR)

# ── Run ──────────────────────────────────────────────────────────────
run: $(BUILD_DIR)/gwen
	$(BUILD_DIR)/gwen --model $(MODEL) "$(PROMPT)" --max-predict $(N) --greedy

# ── Benchmark ────────────────────────────────────────────────────────
bench: $(BUILD_DIR)/gwen
	$(BUILD_DIR)/gwen --model $(MODEL) "$(PROMPT)" --max-predict $(N) --greedy --benchmark

# ── Utility ──────────────────────────────────────────────────────────
info: $(BUILD_DIR)/gwen
	$(BUILD_DIR)/gwen --model $(MODEL) --info

test: $(BUILD_DIR)/gwen
	./scripts/test_correctness.sh
