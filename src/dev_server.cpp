// GWEN Dev Server — serves hidden states + teacher logits for MTP v3 training
//
// Single endpoint optimized for offline distillation:
//   POST /batch_logits  — returns hidden states + teacher logits over restricted vocab
//
// Usage:
//   build/gwen_dev_server --model Qwen3.5-0.8B-Q4_K_M.gguf \
//       --restricted-vocab data/restricted_vocab_4096.bin --port 8090

#include "httplib.h"

#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <getopt.h>
#include <chrono>
#include <fstream>

using namespace gwen;

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model PATH             Path to GGUF model file (required)\n");
    printf("  --restricted-vocab PATH  Binary file of K int32 token IDs (required)\n");
    printf("  --host HOST              Listen address (default: 127.0.0.1)\n");
    printf("  --port PORT              Listen port (default: 8090)\n");
    printf("  --max-seq N              Max sequence length (default: 512)\n");
    printf("  --max-batch N            Max sequences per batch (default: 128)\n");
    printf("  --help                   Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string vocab_path;
    std::string host = "127.0.0.1";
    int port = 8090;
    int max_seq = 512;
    int max_batch = 128;

    static struct option long_options[] = {
        {"model",            required_argument, nullptr, 'm'},
        {"restricted-vocab", required_argument, nullptr, 'v'},
        {"host",             required_argument, nullptr, 'H'},
        {"port",             required_argument, nullptr, 'p'},
        {"max-seq",          required_argument, nullptr, 's'},
        {"max-batch",        required_argument, nullptr, 'b'},
        {"help",             no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:v:H:p:s:b:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'v': vocab_path = optarg; break;
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
    if (vocab_path.empty()) {
        fprintf(stderr, "Error: --restricted-vocab is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // --- Load restricted vocab ---
    std::vector<int32_t> restricted_ids;
    {
        std::ifstream vf(vocab_path, std::ios::binary | std::ios::ate);
        if (!vf.is_open()) {
            fprintf(stderr, "Error: cannot open restricted vocab file: %s\n", vocab_path.c_str());
            return 1;
        }
        size_t file_size = vf.tellg();
        vf.seekg(0);
        int K = file_size / sizeof(int32_t);
        restricted_ids.resize(K);
        vf.read(reinterpret_cast<char*>(restricted_ids.data()), K * sizeof(int32_t));
        printf("Restricted vocab: %d token IDs from %s\n", K, vocab_path.c_str());
    }
    int K = (int)restricted_ids.size();

    // --- Load model ---
    printf("Loading model: %s\n", model_path.c_str());
    auto t0 = std::chrono::high_resolution_clock::now();
    auto model = Model::load(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("GGUF parsed in %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());
    model->print_info();

    // --- Upload weights to GPU ---
    printf("\nUploading weights to GPU...\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    CudaAllocator allocator;
    model->upload_weights(allocator);
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("Weights uploaded in %.1f ms (%.1f MB)\n",
           std::chrono::duration<double, std::milli>(t3 - t2).count(),
           allocator.total_allocated() / 1024.0 / 1024.0);

    // --- Allocate inference state ---
    int max_total_tokens = max_batch * max_seq;
    printf("\nAllocating inference state (max_seq=%d, max_batch=%d, max_tokens=%d)...\n",
           max_seq, max_batch, max_total_tokens);
    InferenceState state;
    state.allocate(model->config, allocator, max_seq);
    // f32_path=false: skip ~6 GB of F32 reference-path buffers (not needed for training server)
    state.allocate_prefill(model->config, allocator, max_total_tokens, /*f32_path=*/false);
    state.allocate_batch_prefill(model->config, allocator, max_total_tokens, max_batch, /*f32_path=*/false);
    printf("Inference state: %.1f MB\n", allocator.total_allocated() / 1024.0 / 1024.0);

    // --- Dequant restricted embed rows → FP16 on GPU ---
    int n_embed = model->config.n_embed;
    printf("\nDequantizing %d restricted vocab rows (%d × %d = %.1f MB FP16)...\n",
           K, K, n_embed, (double)K * n_embed * 2 / 1024 / 1024);

    // Upload restricted IDs to GPU
    int* d_restricted_ids = static_cast<int*>(allocator.alloc(K * sizeof(int)));
    GWEN_CHECK_CUDA(cudaMemcpy(d_restricted_ids, restricted_ids.data(),
                                K * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate restricted_embed_fp16 on GPU
    state.restricted_embed_fp16 = static_cast<half*>(
        allocator.alloc((size_t)K * n_embed * sizeof(half)));
    state.restricted_vocab_K = K;

    // Allocate dedicated logits GPU buffer: [max_total_tokens, K] FP16
    // Can't reuse prefill_ffn_gate because K (4096) may exceed n_ff (3584)
    half* d_logits_buf = static_cast<half*>(
        allocator.alloc((size_t)max_total_tokens * K * sizeof(half)));
    printf("Logits buffer: %d × %d = %.1f MB FP16\n",
           max_total_tokens, K, (double)max_total_tokens * K * 2 / 1024 / 1024);

    // Dequant at startup (one-time cost)
    auto t4 = std::chrono::high_resolution_clock::now();
    gwen_dequant_rows_q6k(model->token_embd.device_data, d_restricted_ids,
                           state.restricted_embed_fp16, K, n_embed);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());
    auto t5 = std::chrono::high_resolution_clock::now();
    printf("Dequantized in %.1f ms\n",
           std::chrono::duration<double, std::milli>(t5 - t4).count());

    // --- Dequant full token_embd → FP16 on GPU (for p_idk computation) ---
    int n_vocab = model->config.n_vocab;  // 248320
    half* d_full_embed_fp16 = static_cast<half*>(
        allocator.alloc((size_t)n_vocab * n_embed * sizeof(half)));
    printf("Full token_embd: %d × %d = %.1f MB FP16 (for p_idk)\n",
           n_vocab, n_embed, (double)n_vocab * n_embed * 2 / 1024 / 1024);
    gwen_dequant_q6_k(model->token_embd.device_data, d_full_embed_fp16,
                       n_vocab * n_embed);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    // --- Allocate p_idk scratch buffers ---
    int pidk_chunk = 512;  // tokens per chunk for the 248K GEMM
    half* d_full_logits_chunk = static_cast<half*>(
        allocator.alloc((size_t)pidk_chunk * n_vocab * sizeof(half)));
    float* d_log_Z = static_cast<float*>(
        allocator.alloc((size_t)max_total_tokens * sizeof(float)));
    float* d_p_idk = static_cast<float*>(
        allocator.alloc((size_t)max_total_tokens * sizeof(float)));
    printf("p_idk buffers: chunk=%d (%.1f MB), log_Z+p_idk (%.1f KB each)\n",
           pidk_chunk, (double)pidk_chunk * n_vocab * 2 / 1024 / 1024,
           (double)max_total_tokens * 4 / 1024);

    printf("Total GPU memory: %.1f MB\n\n",
           allocator.total_allocated() / 1024.0 / 1024.0);

    std::mutex gpu_mtx;
    long request_count = 0;

    // --- HTTP server ---
    httplib::Server svr;

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "{\"status\":\"ok\",\"model\":\"%s\",\"n_embed\":%d,"
                 "\"max_seq\":%d,\"max_batch\":%d,\"restricted_vocab_k\":%d,"
                 "\"requests\":%ld}",
                 model_path.c_str(), n_embed, max_seq, max_batch, K, request_count);
        res.set_content(buf, "application/json");
    });

    // POST /batch_logits
    // Binary input:  [uint32 B][uint32 L][int32 tokens[B*L]]
    // Binary output: [uint32 B][uint32 L][uint32 K]
    //                [fp16 hidden[N*n_embed]]
    //                [fp16 teacher_logits[N*K]]
    // where N = B * L
    svr.Post("/batch_logits", [&](const httplib::Request& req, httplib::Response& res) {
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
        int N = B * L;
        if (N > max_total_tokens) {
            res.status = 400;
            char buf[128];
            snprintf(buf, sizeof(buf), "{\"error\":\"total tokens %d > max %d\"}", N, max_total_tokens);
            res.set_content(buf, "application/json");
            return;
        }

        const int32_t* all_tokens = (const int32_t*)(req.body.data() + 8);

        bool want_p_idk = (req.get_param_value("p_idk") == "1");

        // Prepare output buffer: header + hidden + logits [+ p_idk]
        size_t hidden_bytes = (size_t)N * n_embed * sizeof(uint16_t);
        size_t logits_bytes = (size_t)N * K * sizeof(uint16_t);
        size_t pidk_bytes = want_p_idk ? (size_t)N * sizeof(float) : 0;
        size_t header_bytes = 12;
        std::string body(header_bytes + hidden_bytes + logits_bytes + pidk_bytes, '\0');

        uint32_t header[3] = {B, L, (uint32_t)K};
        memcpy(&body[0], header, 12);

        std::lock_guard<std::mutex> lock(gpu_mtx);
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. Extract hidden states: tokens → all layers → prefill_x (GPU)
        //    Copies hidden to host (body[12..]) and leaves data on GPU.
        state.extract_hidden_batch(*model, all_tokens, B, L, &body[header_bytes]);

        // 2. Compute teacher logits from hidden states on GPU (prefill_x).
        //    Inline RMSNorm + GEMM to avoid double-copy of hidden states.
        {
            cudaStream_t s = 0;

            // Batch RMSNorm: prefill_x → prefill_norm [N, n_embed]
            gwen_rmsnorm_batched_f32w(
                state.prefill_x,
                static_cast<const float*>(model->output_norm.device_data),
                state.prefill_norm, N, n_embed, model->config.rms_norm_eps, s);

            // GEMM: prefill_norm × restricted_embed^T → logits [N, K]
            gwen_gemm_fp16(state.restricted_embed_fp16, state.prefill_norm,
                           d_logits_buf, K, n_embed, N, s);

            // Copy logits to host
            GWEN_CHECK_CUDA(cudaMemcpy(&body[header_bytes + hidden_bytes],
                                        d_logits_buf, logits_bytes,
                                        cudaMemcpyDeviceToHost));

            // Compute p_idk if requested
            if (want_p_idk) {
                // Chunked 248K GEMM + logsumexp + p_idk
                for (int ci = 0; ci < N; ci += pidk_chunk) {
                    int chunk = std::min(pidk_chunk, N - ci);
                    // Full vocab logits: prefill_norm[ci..] × full_embed^T → [chunk, 248K]
                    gwen_gemm_fp16(d_full_embed_fp16,
                                   &state.prefill_norm[ci * n_embed],
                                   d_full_logits_chunk,
                                   n_vocab, n_embed, chunk, s);
                    // logsumexp over 248K → log_Z[chunk]
                    gwen_logsumexp_rows(d_full_logits_chunk, &d_log_Z[ci],
                                         chunk, n_vocab, s);
                    // p_idk from restricted logits (already in d_logits_buf) + log_Z
                    gwen_p_idk_from_logits(&d_logits_buf[(size_t)ci * K], &d_log_Z[ci],
                                            &d_p_idk[ci], chunk, K, s);
                }
                // Copy p_idk to host
                GWEN_CHECK_CUDA(cudaMemcpy(&body[header_bytes + hidden_bytes + logits_bytes],
                                            d_p_idk, pidk_bytes,
                                            cudaMemcpyDeviceToHost));
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        request_count++;

        printf("[%ld] batch_logits%s: %u×%u=%d tok, K=%d → %.1f ms (%.0f tok/s)\n",
               request_count, want_p_idk ? "+p_idk" : "",
               B, L, N, K, ms, (double)N / (ms / 1000.0));
        fflush(stdout);

        res.set_header("X-Batch", std::to_string(B));
        res.set_header("X-SeqLen", std::to_string(L));
        res.set_header("X-Time-Ms", std::to_string((int)ms));
        res.set_content(body, "application/octet-stream");
    });

    printf("GWEN dev server starting on %s:%d\n", host.c_str(), port);
    printf("Restricted vocab: K=%d\n", K);
    printf("Endpoints:\n");
    printf("  GET  /health         Health check\n");
    printf("  POST /batch_logits   Hidden states + teacher logits (binary)\n");
    printf("\nReady.\n");
    fflush(stdout);

    svr.listen(host, port);
    return 0;
}
