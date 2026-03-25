// gwen-bench: GWEN benchmark — llama-bench compatible methodology
//
// Measures prompt processing (pp) and text generation (tg) throughput.
// Methodology: warmup → N timed repetitions → mean ± stddev.
//
// Usage:
//   gwen-bench -m ~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf -p 0 -n 128 -r 5
//   gwen-bench -m ~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf -p 512 -n 128 -r 3
//
// Timing methodology matches llama-bench exactly:
//   - Warmup: full prefill + 1 decode token (discarded)
//   - Each repetition: reset state → prefill (if pp>0) → N decode steps → wall clock
//   - Decode uses random tokens (not real generation) — pure throughput measurement
//   - Each forward() call is fully GPU-synchronized before timing the next

#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/paths.h"
#include "gwen/gguf.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

using namespace gwen;

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options] [model.gguf]\n\n", prog);
    fprintf(stderr, "GWEN benchmark — llama-bench compatible methodology.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model PATH     Model file (default: %s)\n", gwen::DEFAULT_MODEL);
    fprintf(stderr, "  -p, --n-prompt N     Prompt tokens for pp test (default: 512, 0 = skip)\n");
    fprintf(stderr, "  -n, --n-gen N        Decode tokens for tg test (default: 128, 0 = skip)\n");
    fprintf(stderr, "  -r, --repetitions N  Repetitions per test (default: 5)\n");
    fprintf(stderr, "  --no-warmup          Skip warmup run\n");
    fprintf(stderr, "  -o, --output FMT     Output format: md, csv (default: md)\n");
    fprintf(stderr, "  -h, --help           Show this help\n");
}

static uint64_t get_time_ns() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

// Compute model size (sum of all weight tensor bytes) and parameter count
static void compute_model_stats(const Model& model, double& size_gib, double& n_params_b) {
    size_t total_bytes = 0;
    size_t total_elements = 0;

    auto add = [&](const WeightRef& w) {
        if (w.size_bytes > 0) {
            total_bytes += w.size_bytes;
            total_elements += w.n_elements;
        }
    };

    add(model.token_embd);
    add(model.output_weight);
    add(model.output_norm);

    for (size_t i = 0; i < model.layers.size(); i++) {
        const auto& layer = model.layers[i];
        if (layer.is_full_attention) {
            const auto& fa = layer.full_attn;
            add(fa.attn_norm); add(fa.attn_q); add(fa.attn_k); add(fa.attn_v);
            add(fa.attn_output); add(fa.attn_q_norm); add(fa.attn_k_norm);
            add(fa.post_attn_norm);
            add(fa.ffn_gate); add(fa.ffn_up); add(fa.ffn_down);
        } else {
            const auto& dn = layer.deltanet;
            add(dn.attn_norm); add(dn.attn_qkv); add(dn.attn_gate);
            add(dn.ssm_conv1d); add(dn.ssm_a); add(dn.ssm_alpha); add(dn.ssm_beta);
            add(dn.ssm_dt_bias); add(dn.ssm_norm); add(dn.ssm_out);
            add(dn.post_attn_norm);
            add(dn.ffn_gate); add(dn.ffn_up); add(dn.ffn_down);
        }
    }

    size_gib = total_bytes / (1024.0 * 1024.0 * 1024.0);
    n_params_b = total_elements / 1e9;
}

// Extract a short model description from the file path
static std::string model_description(const std::string& path, const ModelConfig& cfg) {
    // Extract filename
    std::string fname = path;
    auto pos = fname.rfind('/');
    if (pos != std::string::npos) fname = fname.substr(pos + 1);

    // Try to build a readable name like "qwen35 9B Q4_K_XL"
    std::string desc;

    // Architecture
    desc = "qwen35";

    // Size class — rough estimate from hidden dim
    if (cfg.n_embed >= 4096) desc += " 9B";
    else if (cfg.n_embed >= 2048) desc += " 4B";
    else desc += " 0.8B";

    // Quantization from filename
    for (const char* q : {"Q4_K_XL", "Q4_K_M", "Q4_K_S", "Q4_K", "Q5_K_M", "Q5_K_S", "Q5_K",
                          "Q6_K", "Q8_0", "IQ4_XS", "F16", "FP8", "FP4"}) {
        if (fname.find(q) != std::string::npos) {
            desc += " ";
            desc += q;
            break;
        }
    }

    return desc;
}

struct BenchTest {
    int n_prompt;
    int n_gen;
    std::vector<double> samples_ns;

    std::string test_label() const {
        std::string label;
        if (n_prompt > 0) label += "pp" + std::to_string(n_prompt);
        if (n_prompt > 0 && n_gen > 0) label += "+";
        if (n_gen > 0) label += "tg" + std::to_string(n_gen);
        return label;
    }

    int measured_tokens() const {
        // pp measures prompt throughput, tg measures decode throughput
        // When both are set, we report combined but that's unusual
        if (n_prompt > 0 && n_gen == 0) return n_prompt;
        if (n_prompt == 0 && n_gen > 0) return n_gen;
        return n_prompt + n_gen;  // combined
    }

    double mean_tps() const {
        int n = measured_tokens();
        double sum = 0;
        for (auto ns : samples_ns) sum += n / (ns / 1e9);
        return sum / samples_ns.size();
    }

    double stddev_tps() const {
        double m = mean_tps();
        int n = measured_tokens();
        double sum_sq = 0;
        for (auto ns : samples_ns) {
            double tps = n / (ns / 1e9);
            sum_sq += (tps - m) * (tps - m);
        }
        return std::sqrt(sum_sq / samples_ns.size());
    }
};

int main(int argc, char** argv) {
    std::string model_path;
    int n_prompt = 512;
    int n_gen = 128;
    int reps = 5;
    bool no_warmup = false;
    std::string output_fmt = "md";

    static struct option long_options[] = {
        {"model",       required_argument, nullptr, 'm'},
        {"n-prompt",    required_argument, nullptr, 'p'},
        {"n-gen",       required_argument, nullptr, 'n'},
        {"repetitions", required_argument, nullptr, 'r'},
        {"no-warmup",   no_argument,       nullptr, 'W'},
        {"output",      required_argument, nullptr, 'o'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:n:r:o:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': n_prompt = atoi(optarg); break;
            case 'n': n_gen = atoi(optarg); break;
            case 'r': reps = atoi(optarg); break;
            case 'W': no_warmup = true; break;
            case 'o': output_fmt = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    // Positional model path
    for (int i = optind; i < argc; i++) {
        std::string arg = argv[i];
        if (model_path.empty() && (ends_with(arg, ".gguf") || ends_with(arg, ".gwfp8") || ends_with(arg, ".gwfp4"))) {
            model_path = arg;
        }
    }

    if (model_path.empty()) model_path = gwen::default_model_path();
    gwen::ensure_file(model_path, (std::string(gwen::RELEASE_BASE) + "/" + gwen::DEFAULT_MODEL).c_str());

    if (n_prompt == 0 && n_gen == 0) {
        fprintf(stderr, "gwen-bench: nothing to benchmark (both -p and -n are 0)\n");
        return 1;
    }

    // ---- Load model ----
    fprintf(stderr, "gwen-bench: loading %s\n", model_path.c_str());

    std::unique_ptr<Model> model;
    bool is_gguf = ends_with(model_path, ".gguf");
    bool is_fp4 = ends_with(model_path, ".gwfp4");

    if (is_fp4) {
        model = Model::load_fp4(model_path);
    } else if (ends_with(model_path, ".gwfp8")) {
        model = Model::load_fp8(model_path);
    } else {
        model = Model::load(model_path);
    }

    CudaAllocator allocator;
    model->upload_weights(allocator);

    // ---- Model info ----
    double size_gib, n_params_b;
    compute_model_stats(*model, size_gib, n_params_b);
    std::string desc = model_description(model_path, model->config);

    fprintf(stderr, "gwen-bench: %s | %.2f GiB | %.2fB params\n", desc.c_str(), size_gib, n_params_b);

    // ---- Allocate inference state ----
    int max_seq = std::max(n_prompt + n_gen, 4096);
    InferenceState state;
    state.allocate(model->config, allocator, max_seq);
    if (n_prompt > 0 && !is_fp4) {
        state.allocate_prefill(model->config, allocator, max_seq,
                               /*f32_path=*/!is_gguf, /*gguf_mode=*/is_gguf);
    }

    fprintf(stderr, "gwen-bench: VRAM %.0f MB | tests: %s | reps: %d\n",
            allocator.total_allocated() / 1024.0 / 1024.0,
            BenchTest{n_prompt, n_gen, {}}.test_label().c_str(), reps);

    // ---- Build test list ----
    // Like llama-bench: separate pp and tg tests
    std::vector<BenchTest> tests;
    if (n_prompt > 0 && n_gen > 0) {
        // Separate tests for pp and tg (like llama-bench default)
        tests.push_back({n_prompt, 0, {}});
        tests.push_back({0, n_gen, {}});
    } else {
        tests.push_back({n_prompt, n_gen, {}});
    }

    // ---- Run benchmarks ----
    int n_vocab = model->config.n_vocab;
    std::srand(42);

    for (auto& test : tests) {
        fprintf(stderr, "\ngwen-bench: running %s", test.test_label().c_str());

        // Warmup
        if (!no_warmup) {
            fprintf(stderr, " (warmup");
            state.reset_state();

            if (test.n_prompt > 0) {
                // Prefill with random tokens
                std::vector<int> prompt_tokens(test.n_prompt);
                prompt_tokens[0] = 151643;  // BOS token
                for (int i = 1; i < test.n_prompt; i++)
                    prompt_tokens[i] = std::rand() % n_vocab;
                state.forward_prefill(*model, prompt_tokens);
            }

            if (test.n_gen > 0) {
                // 1 warmup decode step
                int token = 151643;  // BOS
                state.forward(*model, token);
            }

            fprintf(stderr, ")");
        }

        // Timed repetitions
        for (int rep = 0; rep < reps; rep++) {
            state.reset_state();
            fprintf(stderr, " %d/%d", rep + 1, reps);

            uint64_t t_start = get_time_ns();

            if (test.n_prompt > 0) {
                std::vector<int> prompt_tokens(test.n_prompt);
                prompt_tokens[0] = 151643;
                for (int i = 1; i < test.n_prompt; i++)
                    prompt_tokens[i] = std::rand() % n_vocab;
                state.forward_prefill(*model, prompt_tokens);
                GWEN_CHECK_CUDA(cudaDeviceSynchronize());
            }

            if (test.n_gen > 0) {
                int token = 151643;
                for (int i = 0; i < test.n_gen; i++) {
                    state.forward(*model, token);
                    token = std::rand() % n_vocab;
                }
                // forward() already synchronizes, but be explicit
                GWEN_CHECK_CUDA(cudaDeviceSynchronize());
            }

            uint64_t t_end = get_time_ns();
            test.samples_ns.push_back(static_cast<double>(t_end - t_start));
        }

        fprintf(stderr, " done\n");
    }

    // ---- Output results ----
    fprintf(stderr, "\n");

    if (output_fmt == "csv") {
        printf("model,size_gib,params_b,test,mean_tps,stddev_tps\n");
        for (const auto& t : tests) {
            printf("%s,%.2f,%.2f,%s,%.2f,%.2f\n",
                   desc.c_str(), size_gib, n_params_b,
                   t.test_label().c_str(), t.mean_tps(), t.stddev_tps());
        }
    } else {
        // Markdown table (llama-bench style)
        printf("| %-30s | %10s | %10s | %10s | %15s | %20s |\n",
               "model", "size", "params", "backend", "test", "t/s");
        printf("| %-30s | %10s | %10s | %10s | %15s | %20s |\n",
               "------------------------------", "---------:", "---------:",
               "----------", "--------------:", "-------------------:");
        for (const auto& t : tests) {
            char size_str[32], params_str[32], tps_str[32];
            snprintf(size_str, sizeof(size_str), "%.2f GiB", size_gib);
            snprintf(params_str, sizeof(params_str), "%.2f B", n_params_b);
            snprintf(tps_str, sizeof(tps_str), "%.2f ± %.2f", t.mean_tps(), t.stddev_tps());
            printf("| %-30s | %10s | %10s | %10s | %15s | %20s |\n",
                   desc.c_str(), size_str, params_str, "CUDA",
                   t.test_label().c_str(), tps_str);
        }
    }

    return 0;
}
