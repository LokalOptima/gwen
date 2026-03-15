// GWEN Inference Server
// Loads the GWEN model and serves hidden state extraction via HTTP
//
// Endpoints:
//   GET  /health          - Health check
//   POST /tokenize        - Tokenize text → token IDs
//   POST /extract         - Extract per-token hidden states (FP16)
//   POST /batch_extract   - Batched GEMM extraction (B sequences × L tokens)
//   POST /compare_extract - Compare GEMV vs GEMM paths for correctness

#include "httplib.h"

#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <getopt.h>
#include <sstream>
#include <chrono>

using namespace gwen;

// ============================================================
// Minimal JSON helpers (only what we need)
// ============================================================

static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.length();
    // Skip whitespace and colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ':')) pos++;
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++; // skip opening quote
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case '/':  result += '/';  break;
                default:   result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

static std::vector<int> json_get_int_array(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos = json.find('[', pos);
    if (pos == std::string::npos) return {};
    auto end = json.find(']', pos);
    if (end == std::string::npos) return {};
    std::string arr = json.substr(pos + 1, end - pos - 1);
    std::vector<int> result;
    std::istringstream iss(arr);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // Skip whitespace
        size_t start = token.find_first_not_of(" \t\n");
        if (start == std::string::npos) continue;
        result.push_back(std::stoi(token.substr(start)));
    }
    return result;
}

// ============================================================
// Main
// ============================================================

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model PATH       Path to GGUF model file (required)\n");
    printf("  --host HOST        Listen address (default: 127.0.0.1)\n");
    printf("  --port PORT        Listen port (default: 8090)\n");
    printf("  --max-seq N        Max sequence length (default: 4096)\n");
    printf("  --max-batch N      Max sequences per batch (default: 128)\n");
    printf("  --help             Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string mtp_path;
    std::string mtp_lm_head_path;
    std::string host = "127.0.0.1";
    int port = 8090;
    int max_seq = 512;
    int max_batch = 128;

    static struct option long_options[] = {
        {"model",     required_argument, nullptr, 'm'},
        {"mtp",       required_argument, nullptr, 'M'},
        {"mtp-lm-head", required_argument, nullptr, 'L'},
        {"host",      required_argument, nullptr, 'H'},
        {"port",      required_argument, nullptr, 'p'},
        {"max-seq",   required_argument, nullptr, 's'},
        {"max-batch", required_argument, nullptr, 'b'},
        {"help",      no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:M:L:H:p:s:b:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'M': mtp_path = optarg; break;
            case 'L': mtp_lm_head_path = optarg; break;
            case 'H': host = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 's': max_seq = atoi(optarg); break;
            case 'b': max_batch = atoi(optarg); break;
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
    printf("Loading model: %s\n", model_path.c_str());
    auto t0 = std::chrono::high_resolution_clock::now();
    auto model = Model::load(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("GGUF parsed in %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());
    model->print_info();

    // Load MTP weights if provided (must be before upload_weights)
    if (!mtp_path.empty()) {
        model->load_mtp(mtp_path);
    }
    if (!mtp_lm_head_path.empty()) {
        model->load_reduced_lm_head(mtp_lm_head_path);
    }

    // Upload weights to GPU
    printf("\nUploading weights to GPU...\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    CudaAllocator allocator;
    model->upload_weights(allocator);
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("Weights uploaded in %.1f ms (%.1f MB)\n",
           std::chrono::duration<double, std::milli>(t3 - t2).count(),
           allocator.total_allocated() / 1024.0 / 1024.0);

    // Build tokenizer
    printf("Building tokenizer...\n");
    auto tokenizer = Tokenizer::from_gguf(*model->gguf);
    printf("Vocab size: %d\n", tokenizer->vocab_size());

    // Allocate single InferenceState with batch prefill support
    int max_total_tokens = max_batch * max_seq;
    printf("\nAllocating inference state (max_seq=%d, max_batch=%d, max_tokens=%d)...\n",
           max_seq, max_batch, max_total_tokens);
    InferenceState state;
    state.allocate(model->config, allocator, max_seq);
    state.allocate_batch_prefill(model->config, allocator, max_total_tokens, max_batch);
    if (model->has_mtp) {
        state.allocate_mtp(model->config, allocator, max_seq);
    }
    printf("Total GPU memory: %.1f MB\n\n",
           allocator.total_allocated() / 1024.0 / 1024.0);

    std::mutex gpu_mtx;
    long request_count = 0;

    // HTTP server
    httplib::Server svr;

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "{\"status\":\"ok\",\"model\":\"%s\",\"n_embed\":%d,\"max_seq\":%d,\"max_batch\":%d,\"requests\":%ld}",
                 model_path.c_str(), model->config.n_embed, max_seq, max_batch, request_count);
        res.set_content(buf, "application/json");
    });

    svr.Post("/tokenize", [&](const httplib::Request& req, httplib::Response& res) {
        auto text = json_get_string(req.body, "text");
        if (text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing text field\"}", "application/json");
            return;
        }
        auto tokens = tokenizer->encode(text);
        std::string json = "{\"tokens\":[";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) json += ",";
            json += std::to_string(tokens[i]);
        }
        json += "],\"n_tokens\":" + std::to_string(tokens.size()) + "}";
        res.set_content(json, "application/json");
    });

    // /extract: single sequence via GEMV path (for interactive use)
    svr.Post("/extract", [&](const httplib::Request& req, httplib::Response& res) {
        std::vector<int> tokens;

        if (req.get_header_value("Content-Type").find("application/octet-stream") != std::string::npos) {
            int n = (int)(req.body.size() / sizeof(int32_t));
            tokens.resize(n);
            memcpy(tokens.data(), req.body.data(), n * sizeof(int32_t));
        } else {
            auto text = json_get_string(req.body, "text");
            if (!text.empty()) {
                tokens = tokenizer->encode(text);
            } else {
                tokens = json_get_int_array(req.body, "tokens");
            }
        }

        if (tokens.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"no tokens provided\"}", "application/json");
            return;
        }
        if ((int)tokens.size() > max_seq) {
            res.status = 400;
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "{\"error\":\"sequence too long (%zu > %d)\"}",
                     tokens.size(), max_seq);
            res.set_content(buf, "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(gpu_mtx);
        auto t_start = std::chrono::high_resolution_clock::now();

        int N = (int)tokens.size();
        int n_embed = model->config.n_embed;

        size_t data_bytes = (size_t)N * n_embed * sizeof(uint16_t);
        std::string body(8 + data_bytes, '\0');

        uint32_t header[2] = {(uint32_t)N, (uint32_t)n_embed};
        memcpy(&body[0], header, 8);

        state.extract_hidden(*model, tokens, &body[8]);

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        request_count++;

        res.set_header("X-Tokens", std::to_string(N));
        res.set_header("X-Time-Ms", std::to_string((int)ms));
        res.set_content(body, "application/octet-stream");
    });

    // /batch_extract: batched GEMM extraction (reads weights once for all B*L tokens)
    // Binary input:  [uint32 B][uint32 L][int32 tokens[B*L]]
    // Binary output: [uint32 B][uint32 L][uint32 n_embed][fp16 hidden[B*L*n_embed]]
    svr.Post("/batch_extract", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.body.size() < 8) {
            res.status = 400;
            res.set_content("{\"error\":\"body too short\"}", "application/json");
            return;
        }

        uint32_t B, L;
        memcpy(&B, req.body.data(), 4);
        memcpy(&L, req.body.data() + 4, 4);

        size_t expected = 8 + (size_t)B * L * sizeof(int32_t);
        if (req.body.size() < expected) {
            res.status = 400;
            res.set_content("{\"error\":\"body size mismatch\"}", "application/json");
            return;
        }
        if ((int)L > max_seq) {
            res.status = 400;
            char buf[128];
            snprintf(buf, sizeof(buf), "{\"error\":\"seq_len %u > max_seq %d\"}", L, max_seq);
            res.set_content(buf, "application/json");
            return;
        }
        if ((int)B > max_batch) {
            res.status = 400;
            char buf[128];
            snprintf(buf, sizeof(buf), "{\"error\":\"batch %u > max_batch %d\"}", B, max_batch);
            res.set_content(buf, "application/json");
            return;
        }
        if ((int)(B * L) > max_total_tokens) {
            res.status = 400;
            char buf[128];
            snprintf(buf, sizeof(buf), "{\"error\":\"total tokens %u > max %d\"}", B * L, max_total_tokens);
            res.set_content(buf, "application/json");
            return;
        }

        const int32_t* all_tokens = (const int32_t*)(req.body.data() + 8);
        int n_embed = model->config.n_embed;
        int N = B * L;

        // Check if predictions requested (query param ?preds=1)
        bool want_preds = (req.get_param_value("preds") == "1");

        size_t hidden_bytes = (size_t)N * n_embed * sizeof(uint16_t);
        size_t preds_bytes = want_preds ? (size_t)N * sizeof(int32_t) : 0;
        std::string body(12 + hidden_bytes + preds_bytes, '\0');
        uint32_t header[3] = {B, L, (uint32_t)n_embed};
        memcpy(&body[0], header, 12);

        std::lock_guard<std::mutex> lock(gpu_mtx);
        auto t_start = std::chrono::high_resolution_clock::now();

        // Batch extraction: GEMM-batched forward, reads each weight matrix once
        state.extract_hidden_batch(*model, all_tokens, B, L, &body[12]);

        // Compute main model predictions from hidden states (per-token GEMV + argmax)
        if (want_preds) {
            // pf_a still holds hidden states on GPU after extract_hidden_batch
            state.predict_from_hidden(*model, state.prefill_x, N,
                                       (int32_t*)&body[12 + hidden_bytes]);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        request_count += B;

        printf("[%ld] batch_extract%s: %u×%u=%u tok → %.1f ms (%.0f tok/s)\n",
               request_count, want_preds ? "+preds" : "",
               B, L, B * L, ms, (double)(B * L) / (ms / 1000.0));
        fflush(stdout);

        res.set_header("X-Batch", std::to_string(B));
        res.set_header("X-SeqLen", std::to_string(L));
        res.set_header("X-Time-Ms", std::to_string((int)ms));
        res.set_content(body, "application/octet-stream");
    });

    // /compare_extract: run GEMV (single-seq) and GEMM (batch) paths, compare outputs
    // Binary input:  [uint32 L][int32 tokens[L]]
    // JSON output: max_abs_diff, mean_abs_diff, match status
    svr.Post("/compare_extract", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.body.size() < 4) {
            res.status = 400;
            res.set_content("{\"error\":\"body too short\"}", "application/json");
            return;
        }

        uint32_t L;
        memcpy(&L, req.body.data(), 4);
        if (req.body.size() < 4 + L * sizeof(int32_t)) {
            res.status = 400;
            res.set_content("{\"error\":\"body size mismatch\"}", "application/json");
            return;
        }

        const int32_t* token_data = (const int32_t*)(req.body.data() + 4);
        std::vector<int> tokens(token_data, token_data + L);
        int n_embed = model->config.n_embed;

        std::lock_guard<std::mutex> lock(gpu_mtx);

        // Run GEMV path (single sequence)
        std::vector<uint16_t> gemv_output(L * n_embed);
        state.extract_hidden(*model, tokens, gemv_output.data());

        // Run GEMM path (batch of 1)
        std::vector<uint16_t> gemm_output(L * n_embed);
        state.extract_hidden_batch(*model, token_data, 1, L, gemm_output.data());

        // Compare (bit-level — both outputs are FP16 binary)
        int mismatches = 0;
        for (size_t i = 0; i < gemv_output.size(); i++) {
            if (gemv_output[i] != gemm_output[i]) mismatches++;
        }
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "{\"n_tokens\":%u,\"n_embed\":%d,\"total_elements\":%zu,"
                 "\"bit_mismatches\":%d,\"match\":%s}",
                 L, n_embed, gemv_output.size(),
                 mismatches, (mismatches == 0) ? "true" : "false");
        res.set_content(buf, "application/json");
        printf("[compare] L=%u: bit_mismatches=%d / %zu\n",
               L, mismatches, gemv_output.size());
        fflush(stdout);
    });

    // /test_mtp: Run CUDA MTP on pre-extracted hidden states.
    // Binary input: [uint32 L][int32 tokens[L]][half hidden[L * n_embed]]
    //   tokens: the original token sequence (needed for embed lookup of t+1)
    //   hidden: pre-extracted hidden states (from /batch_extract, same call used by PyTorch)
    // Binary output: [uint32 n_preds][int32 preds[n_preds]]
    if (model->has_mtp) {
        svr.Post("/test_mtp", [&](const httplib::Request& req, httplib::Response& res) {
            if (req.body.size() < 4) {
                res.status = 400;
                res.set_content("{\"error\":\"body too short\"}", "application/json");
                return;
            }

            uint32_t L;
            memcpy(&L, req.body.data(), 4);
            int n_embed = model->config.n_embed;
            size_t expected = 4 + L * sizeof(int32_t) + L * n_embed * sizeof(uint16_t);
            if (req.body.size() < expected || L < 3) {
                res.status = 400;
                char buf[256];
                snprintf(buf, sizeof(buf),
                    "{\"error\":\"need [L][tokens[L]][hidden[L*%d]], got %zu bytes, expected %zu\"}",
                    n_embed, req.body.size(), expected);
                res.set_content(buf, "application/json");
                return;
            }

            const int32_t* token_data = (const int32_t*)(req.body.data() + 4);
            const uint16_t* hidden_data = (const uint16_t*)(req.body.data() + 4 + L * sizeof(int32_t));
            int n_preds = L - 2;

            std::lock_guard<std::mutex> lock(gpu_mtx);

            // Reset MTP state
            state.mtp_pos = 0;
            auto& cache = state.mtp_kv_cache;
            size_t kv_bytes = (size_t)cache.max_seq * cache.n_kv_heads * cache.head_dim * sizeof(uint16_t);
            GWEN_CHECK_CUDA(cudaMemset(cache.k_cache, 0, kv_bytes));
            GWEN_CHECK_CUDA(cudaMemset(cache.v_cache, 0, kv_bytes));
            if (state.mtp_graph_captured) {
                cudaGraphExecDestroy(state.mtp_graph_exec);
                state.mtp_graph_captured = false;
            }

            cudaStream_t s = state.compute_stream ? state.compute_stream : 0;
            std::vector<int32_t> preds(n_preds);

            for (int t = 0; t < n_preds; t++) {
                // Copy hidden[t] from the caller's pre-extracted data
                GWEN_CHECK_CUDA(cudaMemcpy(
                    state.mtp_hidden,
                    hidden_data + (size_t)t * n_embed,
                    n_embed * sizeof(uint16_t),
                    cudaMemcpyHostToDevice));

                int mtp_params[2] = {token_data[t + 1], state.mtp_pos};
                GWEN_CHECK_CUDA(cudaMemcpy(state.d_mtp_token, mtp_params,
                                            2 * sizeof(int), cudaMemcpyHostToDevice));

                state.forward_mtp_body(*model, s);
                GWEN_CHECK_CUDA(cudaStreamSynchronize(s));

                int pred;
                GWEN_CHECK_CUDA(cudaMemcpy(&pred, state.d_argmax_token,
                                            sizeof(int), cudaMemcpyDeviceToHost));
                state.mtp_pos++;
                preds[t] = pred;
            }

            std::string body(4 + n_preds * 4, '\0');
            uint32_t np = n_preds;
            memcpy(&body[0], &np, 4);
            memcpy(&body[4], preds.data(), n_preds * 4);

            printf("[test_mtp] L=%u, %d preds\n", L, n_preds);
            fflush(stdout);

            res.set_content(body, "application/octet-stream");
        });
    }

    printf("GWEN server starting on %s:%d\n", host.c_str(), port);
    printf("Endpoints:\n");
    printf("  GET  /health           Health check\n");
    printf("  POST /tokenize         Tokenize text → {tokens: [...]}\n");
    printf("  POST /extract          Text/tokens → FP16 hidden states (GEMV)\n");
    printf("  POST /batch_extract    Binary batch → FP16 hidden states (GEMM batched)\n");
    printf("  POST /compare_extract  Compare GEMV vs GEMM paths\n");
    printf("\nReady.\n");
    fflush(stdout);

    svr.listen(host, port);
    return 0;
}
