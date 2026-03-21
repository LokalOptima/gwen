// Greedy generation using llama.cpp C API — for comparing with GWEN
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf";
    const char* prompt = argc > 2 ? argv[2] : "The quick brown fox";
    int n_predict = argc > 3 ? atoi(argv[3]) : 30;

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    auto* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;

    auto* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const auto* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize the prompt
    std::vector<llama_token> tokens(strlen(prompt) + 16);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens.data(), tokens.size(), false, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens.data(), tokens.size(), false, true);
    }
    tokens.resize(n_tokens);

    printf("Prompt: %s\n", prompt);
    printf("Prompt tokens (%d):", n_tokens);
    for (int i = 0; i < n_tokens; i++) printf(" %d", tokens[i]);
    printf("\n\n");

    // Process prompt tokens
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode prompt\n");
        return 1;
    }

    // Print top-10 logits after prompt
    float* logits = llama_get_logits(ctx);
    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < n_vocab; i++) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });

    printf("After prompt, top-10 logits:\n");
    for (int i = 0; i < 10; i++) {
        char buf[256];
        int len = llama_token_to_piece(vocab, scored[i].second, buf, sizeof(buf)-1, 0, true);
        if (len < 0) len = 0;
        buf[len] = '\0';
        printf("  token=%d logit=%.4f '%s'\n", scored[i].second, scored[i].first, buf);
    }
    printf("\n");

    // Greedy generation
    printf("Generated tokens (greedy):\n");
    std::vector<llama_token> generated;
    int pos = n_tokens;

    for (int i = 0; i < n_predict; i++) {
        logits = llama_get_logits(ctx);

        // Find argmax
        int best_id = 0;
        float best_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > best_logit) {
                best_logit = logits[j];
                best_id = j;
            }
        }

        generated.push_back(best_id);

        char buf[256];
        int len = llama_token_to_piece(vocab, best_id, buf, sizeof(buf)-1, 0, true);
        if (len < 0) len = 0;
        buf[len] = '\0';
        printf("  [%d] token=%d logit=%.4f '%s'\n", i, best_id, best_logit, buf);

        // Check EOS
        if (best_id == llama_vocab_eos(vocab) ||
            best_id == llama_vocab_eot(vocab)) {
            printf("  (EOS reached)\n");
            break;
        }

        // Decode next token
        llama_batch next = llama_batch_get_one(&best_id, 1);
        if (llama_decode(ctx, next)) {
            fprintf(stderr, "Failed to decode token %d\n", i);
            return 1;
        }
    }

    // Print full generated text
    printf("\nFull output: %s", prompt);
    for (int id : generated) {
        char buf[256];
        int len = llama_token_to_piece(vocab, id, buf, sizeof(buf)-1, 0, true);
        if (len < 0) len = 0;
        buf[len] = '\0';
        printf("%s", buf);
    }
    printf("\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
