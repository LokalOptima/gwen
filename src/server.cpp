// GWEN Inference Server
// Loads the GWEN model and serves inference via HTTP
//
// Endpoints:
//   GET  /health          - Health check
//   POST /tokenize        - Tokenize text → token IDs
//   POST /extract         - Extract per-token hidden states (FP16)

#include "httplib.h"

#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/paths.h"
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
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ':')) pos++;
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;
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
    printf("  --model PATH       Path to GGUF model file\n");
    printf("  --host HOST        Listen address (default: 127.0.0.1)\n");
    printf("  --port PORT        Listen port (default: 8090)\n");
    printf("  --max-seq N        Max sequence length (default: 4096)\n");
    printf("  --help             Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8090;
    int max_seq = 4096;

    static struct option long_options[] = {
        {"model",     required_argument, nullptr, 'm'},
        {"host",      required_argument, nullptr, 'H'},
        {"port",      required_argument, nullptr, 'p'},
        {"max-seq",   required_argument, nullptr, 's'},
        {"help",      no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:H:p:s:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'H': host = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 's': max_seq = atoi(optarg); break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    if (model_path.empty()) model_path = gwen::default_model_path();
    gwen::ensure_file(model_path, (std::string(gwen::RELEASE_BASE) + "/" + gwen::DEFAULT_MODEL).c_str());

    // Load model
    printf("Loading model: %s\n", model_path.c_str());
    auto t0 = std::chrono::high_resolution_clock::now();
    auto model = Model::load(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("GGUF parsed in %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());
    model->print_info();

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

    // Allocate inference state
    printf("\nAllocating inference state (max_seq=%d)...\n", max_seq);
    InferenceState state;
    state.allocate(model->config, allocator, max_seq);
    state.allocate_prefill(model->config, allocator, max_seq);
    printf("Total GPU memory: %.1f MB\n\n",
           allocator.total_allocated() / 1024.0 / 1024.0);

    std::mutex gpu_mtx;
    long request_count = 0;

    // HTTP server
    httplib::Server svr;

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "{\"status\":\"ok\",\"model\":\"%s\",\"n_embed\":%d,\"max_seq\":%d,\"requests\":%ld}",
                 model_path.c_str(), model->config.n_embed, max_seq, request_count);
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

    // /extract: single sequence hidden state extraction
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

        printf("[%ld] extract: %d tok → %.1f ms (%.0f tok/s)\n",
               request_count, N, ms, (double)N / (ms / 1000.0));
        fflush(stdout);

        res.set_header("X-Tokens", std::to_string(N));
        res.set_header("X-Time-Ms", std::to_string((int)ms));
        res.set_content(body, "application/octet-stream");
    });

    printf("GWEN server starting on %s:%d\n", host.c_str(), port);
    printf("Endpoints:\n");
    printf("  GET  /health           Health check\n");
    printf("  POST /tokenize         Tokenize text → {tokens: [...]}\n");
    printf("  POST /extract          Text/tokens → FP16 hidden states\n");
    printf("\nReady.\n");
    fflush(stdout);

    svr.listen(host, port);
    return 0;
}
