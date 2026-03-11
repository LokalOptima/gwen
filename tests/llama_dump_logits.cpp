// Dump full llama.cpp logits to binary file for comparison
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";
    const char* prompt = argc > 2 ? argv[2] : "The";
    const char* outfile = argc > 3 ? argv[3] : "/tmp/llama_logits.bin";

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

    // Tokenize
    std::vector<llama_token> tokens(strlen(prompt) + 16);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens.data(), tokens.size(), false, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens.data(), tokens.size(), false, true);
    }
    tokens.resize(n_tokens);

    fprintf(stderr, "Tokens (%d):", n_tokens);
    for (int i = 0; i < n_tokens; i++) fprintf(stderr, " %d", tokens[i]);
    fprintf(stderr, "\n");

    // Process all prompt tokens
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode\n");
        return 1;
    }

    // Get logits (for the last token in the batch)
    float* logits = llama_get_logits(ctx);

    // Dump to file
    FILE* fp = fopen(outfile, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }
    fwrite(&n_vocab, sizeof(int), 1, fp);
    fwrite(logits, sizeof(float), n_vocab, fp);
    fclose(fp);
    fprintf(stderr, "Dumped %d logits to %s\n", n_vocab, outfile);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
