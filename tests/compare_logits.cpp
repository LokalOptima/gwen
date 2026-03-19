// Compare GWEN logits against llama.cpp for the same token
// Build: g++ -std=c++17 -I../third_party/llama.cpp/include -L../third_party/llama.cpp/build/src -L../third_party/llama.cpp/build/ggml/src -o compare_logits compare_logits.cpp -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -lpthread -ldl
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";

    // Init llama.cpp
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    auto* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 1;
    cparams.no_perf = true;

    auto* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    // Process single token: 9419 = "Hello"
    int token_id = 9419;
    printf("Processing single token: %d\n", token_id);

    llama_batch batch = llama_batch_get_one(&token_id, 1);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    // Get logits
    const float* logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("Vocab size: %d\n", n_vocab);

    // Print first 10 logits
    printf("Logits[0:10]:");
    for (int i = 0; i < 10; i++) printf(" %.4f", logits[i]);
    printf("\n");

    // Print logits around token 9419
    printf("Logits[9415:9425]:");
    for (int i = 9415; i < 9425; i++) printf(" %.4f", logits[i]);
    printf("\n");

    // Top 10
    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < n_vocab; i++) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });

    printf("\nTop-10 logits (llama.cpp):\n");
    for (int i = 0; i < 10; i++) {
        printf("  token=%d logit=%.4f\n", scored[i].second, scored[i].first);
    }

    // Also print the logit for token 236105 (GWEN's top prediction)
    printf("\nLogit for token 236105 (GWEN top): %.4f\n", logits[236105]);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
