#include "gwen/model.h"

#include <chrono>
#include <getopt.h>

using namespace gwen;

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model PATH       Path to GGUF model file (required)\n");
    printf("  --prompt TEXT       Prompt text\n");
    printf("  --n-predict N      Number of tokens to generate (default: 50)\n");
    printf("  --greedy           Use greedy decoding\n");
    printf("  --benchmark        Output benchmark timing as JSON\n");
    printf("  --output-logits    Output raw logits\n");
    printf("  --info             Print model info and exit\n");
    printf("  --help             Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    int n_predict = 50;
    bool greedy = false;
    bool benchmark = false;
    bool output_logits = false;
    bool info_only = false;

    static struct option long_options[] = {
        {"model",        required_argument, nullptr, 'm'},
        {"prompt",       required_argument, nullptr, 'p'},
        {"n-predict",    required_argument, nullptr, 'n'},
        {"greedy",       no_argument,       nullptr, 'g'},
        {"benchmark",    no_argument,       nullptr, 'b'},
        {"output-logits",no_argument,       nullptr, 'l'},
        {"info",         no_argument,       nullptr, 'i'},
        {"help",         no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:n:gblih", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': prompt = optarg; break;
            case 'n': n_predict = atoi(optarg); break;
            case 'g': greedy = true; break;
            case 'b': benchmark = true; break;
            case 'l': output_logits = true; break;
            case 'i': info_only = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

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

    model->print_info();

    if (info_only) {
        return 0;
    }

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
        printf("(Inference kernels not yet implemented — Phase 1+)\n");
        return 0;
    }

    // TODO: Phase 1+ — tokenize, prefill, decode
    printf("\nPrompt: %s\n", prompt.c_str());
    printf("Inference not yet implemented — Phase 1+\n");

    return 0;
}
