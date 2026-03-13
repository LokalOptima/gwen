// Generate golden reference data: greedy tokens + full logits at each position.
// Output binary format:
//   int32 n_positions
//   int32 n_vocab
//   Per position (n_positions times):
//     int32 token_id
//     float32[n_vocab] logits
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model> <prompt> <n_predict> <output.bin>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* prompt = argv[2];
    int n_predict = atoi(argv[3]);
    const char* out_path = argv[4];

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

    fprintf(stderr, "Prompt: %s (%d tokens)\n", prompt, n_tokens);

    // Process prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode prompt\n");
        return 1;
    }

    // Open output file and write header
    FILE* fp = fopen(out_path, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }
    fwrite(&n_predict, sizeof(int), 1, fp);
    fwrite(&n_vocab, sizeof(int), 1, fp);

    // Generate tokens, dump logits at each position
    for (int i = 0; i < n_predict; i++) {
        float* logits = llama_get_logits(ctx);

        // Argmax
        int best_id = 0;
        float best_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > best_logit) {
                best_logit = logits[j];
                best_id = j;
            }
        }

        // Write: token_id + full logit vector
        fwrite(&best_id, sizeof(int), 1, fp);
        fwrite(logits, sizeof(float), n_vocab, fp);

        fprintf(stderr, "  [%d] token=%d logit=%.4f\n", i, best_id, best_logit);

        if (best_id == llama_vocab_eos(vocab) ||
            best_id == llama_vocab_eot(vocab)) {
            // Rewrite header with actual count
            int actual = i + 1;
            fseek(fp, 0, SEEK_SET);
            fwrite(&actual, sizeof(int), 1, fp);
            break;
        }

        // Decode next
        llama_batch next = llama_batch_get_one(&best_id, 1);
        if (llama_decode(ctx, next)) {
            fprintf(stderr, "Failed to decode at step %d\n", i);
            fclose(fp);
            return 1;
        }
    }

    fclose(fp);
    fprintf(stderr, "Written %d positions to %s\n", n_predict, out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
