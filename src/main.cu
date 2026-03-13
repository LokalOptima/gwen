#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/tokenizer.h"

#include <chrono>
#include <getopt.h>
#include <sstream>

using namespace gwen;

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model PATH       Path to GGUF model file (required)\n");
    printf("  --mtp PATH         Path to MTP weights file (enables speculative decoding)\n");
    printf("  --mtp-lm-head PATH Path to reduced LM head for faster MTP (GWRL format)\n");
    printf("  --prompt TEXT       Prompt text\n");
    printf("  --n-predict N      Number of tokens to generate (default: 50)\n");
    printf("  --greedy           Use greedy decoding\n");
    printf("  --benchmark        Output benchmark timing as JSON\n");
    printf("  --output-logits    Output raw logits\n");
    printf("  --teacher-tokens T Comma-separated reference token IDs for teacher-forced comparison\n");
    printf("  --info             Print model info and exit\n");
    printf("  --help             Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string mtp_path;
    std::string mtp_lm_head_path;
    std::string prompt;
    std::string teacher_tokens_str;
    int n_predict = 50;
    bool greedy = true;  // default greedy for now
    bool benchmark = false;
    bool output_logits = false;
    bool info_only = false;

    static struct option long_options[] = {
        {"model",        required_argument, nullptr, 'm'},
        {"mtp",          required_argument, nullptr, 'M'},
        {"mtp-lm-head",  required_argument, nullptr, 'L'},
        {"prompt",       required_argument, nullptr, 'p'},
        {"n-predict",    required_argument, nullptr, 'n'},
        {"greedy",       no_argument,       nullptr, 'g'},
        {"benchmark",    no_argument,       nullptr, 'b'},
        {"output-logits",no_argument,       nullptr, 'l'},
        {"teacher-tokens",required_argument, nullptr, 't'},
        {"info",         no_argument,       nullptr, 'i'},
        {"help",         no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:M:L:p:n:t:gblih", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'M': mtp_path = optarg; break;
            case 'L': mtp_lm_head_path = optarg; break;
            case 'p': prompt = optarg; break;
            case 'n': n_predict = atoi(optarg); break;
            case 't': teacher_tokens_str = optarg; break;
            case 'g': greedy = true; break;
            case 'b': benchmark = true; break;
            case 'l': output_logits = true; break;
            case 'i': info_only = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    (void)output_logits;

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    auto t0 = std::chrono::high_resolution_clock::now();
    printf("Loading model: %s\n", model_path.c_str());

    auto model = Model::load(model_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("GGUF parsed in %.1f ms\n", load_ms);

    // Load MTP weights if provided
    if (!mtp_path.empty()) {
        model->load_mtp(mtp_path);
    }

    // Load reduced LM head if provided
    if (!mtp_lm_head_path.empty()) {
        model->load_reduced_lm_head(mtp_lm_head_path);
    }

    model->print_info();

    if (info_only) {
        return 0;
    }

    // Build tokenizer
    printf("\nBuilding tokenizer...\n");
    auto tokenizer = Tokenizer::from_gguf(*model->gguf);
    printf("Vocab size: %d\n", tokenizer->vocab_size());

    // Upload weights to GPU
    printf("\nUploading weights to GPU...\n");
    auto t2 = std::chrono::high_resolution_clock::now();

    CudaAllocator allocator;
    model->upload_weights(allocator);

    auto t3 = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("Weights uploaded in %.1f ms (%.1f MB, %zu allocations)\n",
           upload_ms,
           allocator.total_allocated() / 1024.0 / 1024.0,
           allocator.n_allocations());

    if (prompt.empty()) {
        printf("\nNo prompt specified. Use --prompt to generate text.\n");
        return 0;
    }

    // Tokenize prompt
    auto prompt_tokens = tokenizer->encode(prompt);
    printf("\nPrompt: %s\n", prompt.c_str());
    printf("Prompt tokens (%zu): ", prompt_tokens.size());
    for (int i = 0; i < (int)prompt_tokens.size() && i < 20; i++) {
        printf("%d ", prompt_tokens[i]);
    }
    if (prompt_tokens.size() > 20) printf("...");
    printf("\n");

    // Allocate inference state
    printf("\nAllocating inference state...\n");
    InferenceState state;
    state.allocate(model->config, allocator);
    state.allocate_prefill(model->config, allocator, 4096);
    if (model->has_mtp) {
        state.allocate_mtp(model->config, allocator, 4096);
        state.allocate_batch2(model->config, allocator);
    }
    printf("Total GPU memory: %.1f MB\n", allocator.total_allocated() / 1024.0 / 1024.0);

    // Parse teacher tokens if provided
    std::vector<int> teacher_tokens;
    if (!teacher_tokens_str.empty()) {
        std::istringstream iss(teacher_tokens_str);
        std::string tok;
        while (std::getline(iss, tok, ',')) {
            if (!tok.empty()) teacher_tokens.push_back(std::stoi(tok));
        }
        printf("\nTeacher-forcing with %zu reference tokens\n", teacher_tokens.size());
    }

    // Generate
    printf("\nGenerating %d tokens (greedy=%s)...\n", n_predict, greedy ? "true" : "false");
    auto t4 = std::chrono::high_resolution_clock::now();

    std::vector<int> output_tokens;
    if (!teacher_tokens.empty()) {
        output_tokens = state.generate(*model, prompt_tokens, n_predict, greedy, 1.0f, teacher_tokens);
    } else if (model->has_mtp) {
        output_tokens = state.generate_speculative(*model, prompt_tokens, n_predict);
    } else {
        output_tokens = state.generate(*model, prompt_tokens, n_predict, greedy);
    }

    GWEN_CHECK_CUDA(cudaDeviceSynchronize());
    auto t5 = std::chrono::high_resolution_clock::now();

    // Decode and print output
    std::string output_text = tokenizer->decode(output_tokens);
    printf("\n%s%s\n", prompt.c_str(), output_text.c_str());

    // Print token IDs for debugging
    printf("\nGenerated token IDs: ");
    for (int id : output_tokens) printf("%d ", id);
    printf("\n");
    printf("Decoded per-token: ");
    for (int id : output_tokens) printf("[%s]", tokenizer->decode(id).c_str());
    printf("\n");

    // Timing stats
    double gen_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    int n_prompt = (int)prompt_tokens.size();
    int n_gen = (int)output_tokens.size();
    double tok_per_s = n_gen / (gen_ms / 1000.0);

    printf("\n--- Timing ---\n");
    printf("Prompt tokens: %d\n", n_prompt);
    printf("Generated tokens: %d\n", n_gen);
    printf("Total time: %.1f ms\n", gen_ms);
    printf("Tokens/sec: %.1f\n", tok_per_s);

    if (benchmark) {
        // JSON output for benchmark scripts
        fprintf(stderr,
            "{\"prompt_tokens\": %d, \"decode_tokens\": %d, "
            "\"total_ms\": %.2f, \"decode_tok_per_s\": %.2f, "
            "\"ttft_ms\": %.2f, \"peak_vram_mb\": %.1f}\n",
            n_prompt, n_gen, gen_ms, tok_per_s,
            gen_ms / (n_prompt + n_gen) * n_prompt,  // rough TTFT estimate
            allocator.total_allocated() / 1024.0 / 1024.0);
    }

    return 0;
}
