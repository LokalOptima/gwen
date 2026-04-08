# Convenience targets for the llama.cpp-based build
BUILD_DIR := build
GWEN_CACHE := $(HOME)/.cache/gwen
RELEASE_URL := https://github.com/LokalOptima/gwen/releases/download/v2.0.0

MODEL_BASE := $(GWEN_CACHE)/Qwen3.5-0.8B-Q8_0.gguf
MODEL_MTP  := $(GWEN_CACHE)/Qwen3.5-0.8B-Q8_0-mtp.gguf
MD5_BASE   := d8872c3399f15f172026776e33a0f918
MD5_MTP    := 8e48e160b3c42237a007923a90b8e3e5

.PHONY: all clean completion bench server test bench-decode bench-mtp download-models

all: $(BUILD_DIR)/bin/llama-completion

$(BUILD_DIR)/bin/llama-completion: CMakeLists.txt $(wildcard src/*.cpp src/*.h src/models/*.cpp)
	cmake -S . -B $(BUILD_DIR) -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
	cmake --build $(BUILD_DIR) --target llama-completion -j$$(nproc)

completion: $(BUILD_DIR)/bin/llama-completion

bench: $(BUILD_DIR)/bin/llama-bench
$(BUILD_DIR)/bin/llama-bench: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR) -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
	cmake --build $(BUILD_DIR) --target llama-bench -j$$(nproc)

server: $(BUILD_DIR)/bin/llama-server
$(BUILD_DIR)/bin/llama-server: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR) -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
	cmake --build $(BUILD_DIR) --target llama-server -j$$(nproc)

clean:
	rm -rf $(BUILD_DIR)

# --- Test / Benchmark shortcuts ---
test: $(BUILD_DIR)/bin/llama-completion
	./scripts/test_correctness.sh

bench-decode: $(BUILD_DIR)/bin/llama-completion $(BUILD_DIR)/bin/llama-bench
	./scripts/bench_decode.sh

bench-mtp: $(BUILD_DIR)/bin/llama-completion
	./scripts/bench_mtp_llama.sh

# --- Model download ---
download-models: $(MODEL_BASE) $(MODEL_MTP)

$(MODEL_BASE):
	@mkdir -p $(GWEN_CACHE)
	@echo "Downloading $(notdir $@) (775 MiB)..."
	curl -L -o $@ $(RELEASE_URL)/$(notdir $@)
	@echo "$(MD5_BASE)  $@" | md5sum -c -

$(MODEL_MTP):
	@mkdir -p $(GWEN_CACHE)
	@echo "Downloading $(notdir $@) (80 MiB)..."
	curl -L -o $@ $(RELEASE_URL)/$(notdir $@)
	@echo "$(MD5_MTP)  $@" | md5sum -c -
