// Dump intermediate tensors from llama.cpp for comparison with GWEN
// Build: g++ -std=c++17 -I../third_party/llama.cpp/include -I../third_party/llama.cpp/ggml/include -L../third_party/llama.cpp/build/src -L../third_party/llama.cpp/build/ggml/src -o dump_llama_tensors dump_llama_tensors.cpp -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -lpthread -ldl
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

struct TensorDump {
    std::string name;
    std::vector<float> data;
    int64_t ne[4];
};

static std::vector<TensorDump> dumped_tensors;

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    const char* name = ggml_get_name(t);
    if (!name || strlen(name) == 0) return false;

    std::string sname(name);

    // Capture tensors for layer 0 and final output
    bool want = false;
    // Layer 0 intermediates
    if (sname.find("-0") != std::string::npos || sname.find("_0") != std::string::npos) want = true;
    // Specific layer 0 tensor names
    if (sname == "model.input_embed") want = true;
    if (sname.find("attn_norm-0") != std::string::npos) want = true;
    if (sname.find("result_norm") != std::string::npos) want = true;
    if (sname.find("result_output") != std::string::npos) want = true;
    // Layer 1-3 outputs
    if (sname.find("post_ffn-1") != std::string::npos) want = true;
    if (sname.find("post_ffn-2") != std::string::npos) want = true;
    if (sname.find("post_ffn-3") != std::string::npos) want = true;

    if (!want) return false;

    if (ask) {
        return true;
    }

    int64_t n = ggml_nelements(t);

    TensorDump dump;
    dump.name = sname;
    for (int i = 0; i < 4; i++) dump.ne[i] = t->ne[i];
    dump.data.resize(n);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dump.data.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> fp16_data(n);
        ggml_backend_tensor_get(t, fp16_data.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) {
            dump.data[i] = ggml_fp16_to_fp32(fp16_data[i]);
        }
    } else {
        printf("[LLAMA] %s: unsupported type %d, skipping\n", name, (int)t->type);
        return false;
    }

    dumped_tensors.push_back(std::move(dump));
    return false;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";

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
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = nullptr;

    auto* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    int token_id = 9419;
    printf("Processing token: %d\n\n", token_id);

    llama_batch batch = llama_batch_get_one(&token_id, 1);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    // Print all dumped tensors
    printf("=== Dumped %zu tensors ===\n\n", dumped_tensors.size());

    for (const auto& dump : dumped_tensors) {
        int64_t n = dump.data.size();
        int64_t show = std::min(n, (int64_t)10);

        printf("[LLAMA] %s: ne=[%lld,%lld,%lld,%lld]\n",
               dump.name.c_str(),
               (long long)dump.ne[0], (long long)dump.ne[1],
               (long long)dump.ne[2], (long long)dump.ne[3]);

        printf("  first %lld:", (long long)show);
        for (int64_t i = 0; i < show; i++) printf(" %.6f", dump.data[i]);
        printf("\n");

        float norm = 0;
        for (int64_t i = 0; i < n; i++) norm += dump.data[i] * dump.data[i];
        printf("  norm=%.6f, n=%lld\n\n", sqrtf(norm), (long long)n);

        // Save binary dump for key tensors
        if (dump.name == "model.input_embed" ||
            dump.name.find("attn_norm-0") != std::string::npos ||
            dump.name.find("post_ffn-0") != std::string::npos ||
            dump.name.find("post_ffn-3") != std::string::npos ||
            dump.name.find("result_norm") != std::string::npos) {
            std::string fname = "/tmp/llama_" + dump.name + ".bin";
            // Replace special chars
            for (auto& c : fname) if (c == ' ' || c == '/') c = '_';
            fname = "/tmp/llama_" + dump.name + ".bin";
            std::replace(fname.begin(), fname.end(), '/', '_');
            std::replace(fname.begin(), fname.end(), '-', '_');
            std::replace(fname.begin(), fname.end(), '.', '_');
            fname = "/tmp/" + fname;
            // Simplify: just use a clean name
        }
    }

    // Get logits
    const float* logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("Logits[0:10]:");
    for (int i = 0; i < 10; i++) printf(" %.4f", logits[i]);
    printf("\n");
    printf("Logit[11]: %.4f\n", logits[11]);

    // Top 10
    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < n_vocab; i++) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    printf("\nTop-10:\n");
    for (int i = 0; i < 10; i++) {
        printf("  token=%d logit=%.4f\n", scored[i].second, scored[i].first);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
