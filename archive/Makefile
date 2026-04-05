# ── Paths ────────────────────────────────────────────────────────────
PROMPT    ?= The meaning of life is
N         ?= 100

# ── Build ────────────────────────────────────────────────────────────
BUILD_DIR := build

.PHONY: all clean run bench bench-quick info test

all: $(BUILD_DIR)/gwen

$(BUILD_DIR)/gwen: CMakeLists.txt $(wildcard src/*.cu src/*.cpp src/kernels/*.cu include/gwen/*.h)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
	@cmake --build $(BUILD_DIR) -j$$(nproc) 2>&1 | grep -E "error|warning:|^make" || true

clean:
	rm -rf $(BUILD_DIR)

# ── Run (weights auto-downloaded to ~/.cache/gwen/) ─────────────────
run: $(BUILD_DIR)/gwen
	$(BUILD_DIR)/gwen "$(PROMPT)" --max-predict $(N) --greedy

# ── Benchmark ────────────────────────────────────────────────────────
bench: $(BUILD_DIR)/gwen
	./scripts/bench.sh

bench-quick: $(BUILD_DIR)/gwen
	./scripts/bench.sh --quick

# ── Utility ──────────────────────────────────────────────────────────
info: $(BUILD_DIR)/gwen
	$(BUILD_DIR)/gwen --info

test: $(BUILD_DIR)/gwen
	./scripts/test_correctness.sh
