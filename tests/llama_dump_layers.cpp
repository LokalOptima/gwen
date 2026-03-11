// Dump per-layer hidden states from llama.cpp using eval callback
// Compare against f32_forward to find divergence point
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>

struct DumpState {
    std::map<std::string, std::vector<float>> tensors;
    int target_layer;
};

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true; // always observe

    auto* state = (DumpState*)user_data;
    const char* name = ggml_get_name(t);
    if (!name || name[0] == '\0') return true;

    std::string sname(name);

    int n_elems = ggml_nelements(t);
    size_t nbytes = ggml_nbytes(t);
    enum ggml_type type = t->type;

    // Only capture reasonably sized contiguous F32 tensors
    size_t expected = (size_t)n_elems * sizeof(float);
    if (type == GGML_TYPE_F32 && n_elems > 0 && n_elems <= 1024*1024
        && nbytes == expected && ggml_is_contiguous(t)) {
        std::vector<float> data(n_elems);
        ggml_backend_tensor_get(t, data.data(), 0, expected);
        state->tensors[sname] = std::move(data);
    }

    return true;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";
    const char* prompt = argc > 2 ? argv[2] : "The";

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only for callback to work

    auto* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;

    DumpState dump_state;

    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = &dump_state;

    auto* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const auto* vocab = llama_model_get_vocab(model);

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

    // Process
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode\n");
        return 1;
    }

    // Print all captured tensors
    fprintf(stderr, "\nCaptured %zu tensors:\n", dump_state.tensors.size());
    for (auto& [name, data] : dump_state.tensors) {
        // Compute norm
        float norm = 0;
        for (auto v : data) norm += v*v;
        norm = sqrtf(norm);

        printf("%-40s elems=%-8d norm=%-12.4f first5:", name.c_str(), (int)data.size(), norm);
        int show = std::min((int)data.size(), 5);
        for (int i = 0; i < show; i++) printf(" %.6f", data[i]);
        printf("\n");
    }

    // Dump specific tensors to binary files for comparison
    auto dump_tensor = [&](const std::string& name, const std::string& outfile) {
        auto it = dump_state.tensors.find(name);
        if (it != dump_state.tensors.end()) {
            FILE* fp = fopen(outfile.c_str(), "wb");
            int n = it->second.size();
            fwrite(&n, sizeof(int), 1, fp);
            fwrite(it->second.data(), sizeof(float), n, fp);
            fclose(fp);
            fprintf(stderr, "Dumped '%s' (%d elems) to %s\n", name.c_str(), n, outfile.c_str());
        } else {
            fprintf(stderr, "Tensor '%s' not found!\n", name.c_str());
        }
    };

    // Dump layer outputs
    for (int i = 0; i < 24; i++) {
        std::string name = "l_out-" + std::to_string(i);
        dump_tensor(name, "/tmp/llama_layer_" + std::to_string(i) + ".bin");
    }

    // Dump final output
    dump_tensor("result_norm", "/tmp/llama_result_norm.bin");
    dump_tensor("result_output", "/tmp/llama_result_output.bin");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
