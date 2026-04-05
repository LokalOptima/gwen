# Convenience targets for the llama.cpp-based build
BUILD_DIR := build

.PHONY: all clean completion bench server test bench-decode bench-mtp

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
