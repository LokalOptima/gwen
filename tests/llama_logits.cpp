// Get llama.cpp logits for a single token to compare with GWEN
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";
    int token_id = argc > 2 ? atoi(argv[2]) : 760;  // "The"

    // Init
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // Use GPU

    auto* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 1;
    cparams.no_perf = true;

    auto* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // Process single token
    llama_batch batch = llama_batch_get_one(&token_id, 1);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode\n");
        return 1;
    }

    // Get logits
    float* logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("n_vocab: %d\n", n_vocab);
    printf("Token processed: %d\n", token_id);

    // Print specific logits
    printf("\nLogit[0] = %.4f\n", logits[0]);
    printf("Logit[11] = %.4f\n", logits[11]);
    printf("Logit[220] = %.4f\n", logits[220]);  // GWEN's top (space)
    printf("Logit[198] = %.4f\n", logits[198]);

    // Find top-10
    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < n_vocab; i++) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });

    printf("\nTop-10 logits (llama.cpp):\n");
    for (int i = 0; i < 10; i++) {
        printf("  token=%d logit=%.4f\n", scored[i].second, scored[i].first);
    }

    // Print first 10 logits
    printf("\nLogits[0:10]: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", logits[i]);
    printf("\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
