#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/tokenizer.h"

#include <chrono>
#include <fstream>
#include <getopt.h>
#include <sstream>
#include <vector>

using namespace gwen;

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model PATH         Path to GGUF model file (required)\n");
    printf("  --mtp PATH           Path to MTP weights file (enables speculative decoding)\n");
    printf("  --mtp-lm-head PATH   Path to reduced LM head for faster MTP (GWRL format)\n");
    printf("  --prompt TEXT         Prompt text\n");
    printf("  --n-predict N        Number of tokens to generate (default: 50)\n");
    printf("  --greedy             Use greedy decoding\n");
    printf("  --mtp-threshold F    Confidence threshold for MTP (skip if softmax prob < F)\n");
    printf("  --benchmark          Output benchmark timing as JSON\n");
    printf("  --output-logits      Output raw logits\n");
    printf("  --teacher-tokens T   Comma-separated reference token IDs for teacher-forced comparison\n");
    printf("  --batch-extract FILE Batch GEMM extract mode: read prompts from FILE (one per line)\n");
    printf("  --seq-len N          Sequence length for batch-extract (pad/truncate, default: 512)\n");
    printf("  --compare-extract    Compare extract_hidden (F32 GEMV) vs extract_hidden_batch (B=1 GEMM)\n");
    printf("  --info               Print model info and exit\n");
    printf("  --help               Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string mtp_path;
    std::string mtp_lm_head_path;
    std::string prompt;
    std::string teacher_tokens_str;
    std::string batch_extract_file;
    int n_predict = 50;
    float mtp_threshold = 0.0f;
    bool greedy = true;  // default greedy for now
    bool benchmark = false;
    bool output_logits = false;
    bool info_only = false;
    bool compare_extract = false;
    int seq_len = 512;

    static struct option long_options[] = {
        {"model",        required_argument, nullptr, 'm'},
        {"mtp",          required_argument, nullptr, 'M'},
        {"mtp-lm-head",  required_argument, nullptr, 'L'},
        {"mtp-threshold",required_argument, nullptr, 'T'},
        {"prompt",       required_argument, nullptr, 'p'},
        {"n-predict",    required_argument, nullptr, 'n'},
        {"greedy",       no_argument,       nullptr, 'g'},
        {"benchmark",    no_argument,       nullptr, 'b'},
        {"output-logits",no_argument,       nullptr, 'l'},
        {"teacher-tokens",required_argument, nullptr, 't'},
        {"batch-extract",required_argument, nullptr, 'B'},
        {"seq-len",      required_argument, nullptr, 'S'},
        {"compare-extract", no_argument,    nullptr, 'C'},
        {"info",         no_argument,       nullptr, 'i'},
        {"help",         no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:M:L:T:p:n:t:B:S:Cgblih", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'M': mtp_path = optarg; break;
            case 'L': mtp_lm_head_path = optarg; break;
            case 'T': mtp_threshold = atof(optarg); break;
            case 'p': prompt = optarg; break;
            case 'n': n_predict = atoi(optarg); break;
            case 't': teacher_tokens_str = optarg; break;
            case 'g': greedy = true; break;
            case 'b': benchmark = true; break;
            case 'l': output_logits = true; break;
            case 'B': batch_extract_file = optarg; break;
            case 'S': seq_len = atoi(optarg); break;
            case 'C': compare_extract = true; break;
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

    // ========== Compare extract mode ==========
    if (compare_extract) {
        // Compare single-seq extract_hidden (F32 GEMV) vs batch extract_hidden_batch (B=1 GEMM)
        const char* test_prompts[] = {
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was nothing but silence.",
            "def fibonacci(n):\n    if n <= 1:\n        return n",
        };
        int n_prompts = 3;
        int L = seq_len;
        int n_embed = model->config.n_embed;

        printf("\n=== Compare Extract: F32 GEMV (extract_hidden) vs FP16 GEMM (extract_hidden_batch, B=1) ===\n");
        printf("Seq length: %d, n_embed: %d\n\n", L, n_embed);

        // Allocate with F32 path enabled + batch buffers
        InferenceState state;
        state.allocate(model->config, allocator);
        state.allocate_batch_prefill(model->config, allocator, L, 1, /*f32_path=*/true);
        printf("Total GPU memory: %.1f MB\n\n", allocator.total_allocated() / 1024.0 / 1024.0);

        int all_pass = 1;
        for (int p = 0; p < n_prompts; p++) {
            auto tokens = tokenizer->encode(test_prompts[p]);

            // Pad/truncate to L
            std::vector<int> tok_padded(L, 0);
            int copy_len = std::min((int)tokens.size(), L);
            for (int j = 0; j < copy_len; j++) tok_padded[j] = tokens[j];
            int actual_len = copy_len;

            printf("[%d] \"%s\" (%d tokens, padded to %d)\n", p, test_prompts[p], (int)tokens.size(), L);

            // Run F32 GEMV path (extract_hidden)
            std::vector<uint16_t> gemv_out((size_t)L * n_embed);
            state.extract_hidden(*model, tok_padded, gemv_out.data());

            // Run FP16 GEMM batch path (extract_hidden_batch, B=1)
            std::vector<int32_t> tok_i32(tok_padded.begin(), tok_padded.end());
            std::vector<uint16_t> gemm_out((size_t)L * n_embed);
            state.extract_hidden_batch(*model, tok_i32.data(), 1, L, gemm_out.data());

            // Compare per position
            int bit_mismatches = 0;
            float max_abs = 0.0f;
            double sum_abs = 0.0;
            for (int t = 0; t < actual_len; t++) {
                float cos_num = 0, cos_den_a = 0, cos_den_b = 0;
                float pos_max_abs = 0;
                for (int d = 0; d < n_embed; d++) {
                    size_t idx = (size_t)t * n_embed + d;
                    uint16_t a_bits = gemv_out[idx], b_bits = gemm_out[idx];
                    if (a_bits != b_bits) bit_mismatches++;
                    // Convert to float for comparison
                    half a_h, b_h;
                    memcpy(&a_h, &a_bits, 2);
                    memcpy(&b_h, &b_bits, 2);
                    float a = __half2float(a_h), b = __half2float(b_h);
                    float diff = fabsf(a - b);
                    if (diff > pos_max_abs) pos_max_abs = diff;
                    sum_abs += diff;
                    cos_num += a * b;
                    cos_den_a += a * a;
                    cos_den_b += b * b;
                }
                float cos_sim = cos_num / (sqrtf(cos_den_a) * sqrtf(cos_den_b) + 1e-12f);
                if (pos_max_abs > max_abs) max_abs = pos_max_abs;

                if (t < 5 || t == actual_len - 1) {
                    printf("  pos[%3d]: cos=%.6f  max_abs=%.6f\n", t, cos_sim, pos_max_abs);
                } else if (t == 5) {
                    printf("  ...\n");
                }

                if (cos_sim < 0.9999f) {
                    printf("  WARN: low cosine at pos %d: %.6f\n", t, cos_sim);
                    all_pass = 0;
                }
            }
            size_t total_elems = (size_t)actual_len * n_embed;
            float mean_abs = (float)(sum_abs / total_elems);
            printf("  Summary: bit_mismatches=%d/%zu  max_abs=%.6f  mean_abs=%.6f\n",
                   bit_mismatches, total_elems, max_abs, mean_abs);

            // F32 vs FP16 paths: max_abs up to ~0.01 is expected (FP16 precision)
            // Cosine < 0.9999 would indicate actual corruption
            if (max_abs > 0.05f) {
                printf("  FAIL: max_abs > 0.05 (likely corruption)\n");
                all_pass = 0;
            } else {
                printf("  PASS\n");
            }
            printf("\n");
        }

        printf("=== %s ===\n", all_pass ? "ALL PASSED" : "SOME FAILED");
        return all_pass ? 0 : 1;
    }

    // ========== Batch extract mode ==========
    if (!batch_extract_file.empty()) {
        // Read prompts from file (one per line)
        std::ifstream infile(batch_extract_file);
        if (!infile.is_open()) {
            fprintf(stderr, "Error: cannot open %s\n", batch_extract_file.c_str());
            return 1;
        }
        std::vector<std::string> prompts;
        std::string line;
        while (std::getline(infile, line)) {
            if (!line.empty()) prompts.push_back(line);
        }
        if (prompts.empty()) {
            fprintf(stderr, "Error: no prompts in %s\n", batch_extract_file.c_str());
            return 1;
        }

        int B = (int)prompts.size();
        int L = seq_len;
        printf("\nBatch extract mode: B=%d, L=%d (from %s)\n", B, L, batch_extract_file.c_str());

        // Tokenize each prompt, pad/truncate to L
        std::vector<int32_t> all_tokens(B * L, 0);
        for (int b = 0; b < B; b++) {
            auto tokens = tokenizer->encode(prompts[b]);
            int copy_len = std::min((int)tokens.size(), L);
            for (int j = 0; j < copy_len; j++) {
                all_tokens[b * L + j] = tokens[j];
            }
            printf("  [%d] %zu tok%s — %.40s%s\n", b, tokens.size(),
                   (int)tokens.size() > L ? " (truncated)" :
                   (int)tokens.size() < L ? " (padded)" : "",
                   prompts[b].c_str(),
                   prompts[b].size() > 40 ? "..." : "");
        }

        // Allocate batch state
        printf("\nAllocating batch state...\n");
        InferenceState state;
        state.allocate(model->config, allocator);
        state.allocate_batch_prefill(model->config, allocator, B * L, B);
        printf("Total GPU memory: %.1f MB\n", allocator.total_allocated() / 1024.0 / 1024.0);

        // Output buffer
        int n_embed = model->config.n_embed;
        std::vector<uint16_t> output((size_t)B * L * n_embed);

        // Run extraction
        printf("\nRunning batch extraction...\n");
        auto t4 = std::chrono::high_resolution_clock::now();

        state.extract_hidden_batch(*model, all_tokens.data(), B, L, output.data());

        GWEN_CHECK_CUDA(cudaDeviceSynchronize());
        auto t5 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
        int total_tokens = B * L;
        double tok_per_s = total_tokens / (ms / 1000.0);

        printf("\n--- Batch Extract Timing ---\n");
        printf("Batch size: %d\n", B);
        printf("Seq length: %d\n", L);
        printf("Total tokens: %d\n", total_tokens);
        printf("Time: %.1f ms\n", ms);
        printf("Throughput: %.0f tok/s\n", tok_per_s);

        if (benchmark) {
            fprintf(stderr,
                "{\"batch\": %d, \"seq_len\": %d, \"total_tokens\": %d, "
                "\"total_ms\": %.2f, \"tok_per_s\": %.2f, "
                "\"peak_vram_mb\": %.1f}\n",
                B, L, total_tokens, ms, tok_per_s,
                allocator.total_allocated() / 1024.0 / 1024.0);
        }

        return 0;
    }

    // ========== Normal generation mode ==========
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
        state.mtp_confidence_threshold = mtp_threshold;
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
