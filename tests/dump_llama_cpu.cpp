// Dump intermediate tensors from llama.cpp using CPU backend
// The eval callback works properly on CPU
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    const char* name = ggml_get_name(t);
    if (!name || strlen(name) == 0) return false;

    std::string sname(name);

    // Capture specific tensors
    bool want = false;
    if (sname == "model.input_embed") want = true;
    if (sname == "attn_norm-0") want = true;
    if (sname == "linear_attn_qkv_mixed-0") want = true;
    if (sname == "z-0") want = true;
    if (sname == "conv_output_raw-0") want = true;
    if (sname == "conv_output_silu-0") want = true;
    if (sname == "q_conv-0") want = true;
    if (sname == "k_conv-0") want = true;
    if (sname == "v_conv-0") want = true;
    if (sname == "q_conv_predelta-0") want = true;
    if (sname == "k_conv_predelta-0") want = true;
    if (sname == "v_conv_predelta-0") want = true;
    if (sname == "alpha-0") want = true;
    if (sname == "beta-0") want = true;
    if (sname == "a_softplus-0") want = true;
    if (sname == "gate-0") want = true;
    if (sname == "attn_output-0") want = true;
    if (sname == "final_output-0") want = true;
    if (sname == "linear_attn_out-0") want = true;
    if (sname == "attn_residual-0") want = true;
    if (sname == "attn_post_norm-0") want = true;
    if (sname == "ffn_out-0") want = true;
    if (sname == "post_ffn-0") want = true;
    if (sname == "post_ffn-1") want = true;
    if (sname == "post_ffn-2") want = true;
    if (sname == "post_ffn-3") want = true;
    if (sname == "dnet_add_ar_state-0") want = true;
    if (sname == "result_norm") want = true;
    if (sname == "result_output") want = true;

    if (!want) return false;

    if (ask) {
        return true;
    }

    int64_t n = ggml_nelements(t);
    int64_t show = std::min(n, (int64_t)10);

    printf("[LLAMA] %s: type=%d, ne=[%lld,%lld,%lld,%lld], n=%lld\n",
           name, (int)t->type,
           (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3], (long long)n);

    std::vector<float> data(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> fp16_data(n);
        ggml_backend_tensor_get(t, fp16_data.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) data[i] = ggml_fp16_to_fp32(fp16_data[i]);
    } else {
        printf("  (unsupported type %d)\n", (int)t->type);
        return false;
    }

    printf("  [:10]:");
    for (int64_t i = 0; i < show; i++) printf(" %.6f", data[i]);
    printf("\n");

    float norm = 0;
    for (int64_t i = 0; i < n; i++) norm += data[i] * data[i];
    printf("  norm=%.6f\n\n", sqrtf(norm));

    // For state tensor, also show more data
    if (sname == "dnet_add_ar_state-0") {
        printf("  [128:138]:");
        for (int64_t i = 128; i < std::min(n, (int64_t)138); i++) printf(" %.6f", data[i]);
        printf("\n\n");
    }

    return false;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU ONLY

    auto* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 1;
    cparams.no_perf = true;
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = nullptr;

    auto* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    int token_id = 9419;
    printf("Processing token: %d (CPU mode)\n\n", token_id);

    llama_batch batch = llama_batch_get_one(&token_id, 1);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }

    const float* logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    printf("=== Final Logits ===\n");
    printf("Logits[0:10]:");
    for (int i = 0; i < 10; i++) printf(" %.4f", logits[i]);
    printf("\nLogit[11]: %.4f\n", logits[11]);

    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < n_vocab; i++) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    printf("Top-5:\n");
    for (int i = 0; i < 5; i++) printf("  token=%d logit=%.4f\n", scored[i].second, scored[i].first);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
