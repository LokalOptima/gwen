// Micro-benchmark: compare GEMM approaches for prefill weight handling.
//
// Tests at every representative matrix size from Qwen3.5-0.8B prefill:
//   Approach A: Pre-dequanted FP16 weights + CUTLASS (current baseline)
//   Approach B: Per-call dequant Q4_K → temp FP16 + CUTLASS (saves 950 MB VRAM)
//   Approach C: Dequant-only (measures raw dequant overhead)
//
// Usage: ./bench_gemm_approaches [--seq 128,256,512] [--iters 20] [--warmup 5]

#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// Representative GEMM sizes from Qwen3.5-0.8B
// CUTLASS convention: C[M,N] = A[M,K] * B[K,N]
//   M = out_features (weight rows)
//   K = in_features  (weight cols / activation dim)
//   N = seq_len      (varies)
struct GemmSize {
    const char* name;
    int M, K;
    int count;  // how many times this appears per forward pass
};

static const GemmSize SIZES[] = {
    {"attn_qkv (DN×18)", 6144, 1024, 18},
    {"attn_gate (DN×18)", 2048, 1024, 18},
    {"ssm_out   (DN×18)", 1024, 2048, 18},
    {"ffn_gate  (all×24)", 3584, 1024, 24},
    {"ffn_up    (all×24)", 3584, 1024, 24},
    {"ffn_down  (all×24)", 1024, 3584, 24},
    {"attn_q    (FA×6)",  2048, 1024,  6},
    {"attn_k    (FA×6)",   512, 1024,  6},
    {"attn_v    (FA×6)",   512, 1024,  6},
    {"attn_out  (FA×6)",  1024, 2048,  6},
};
static constexpr int N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

struct TimingResult {
    float median_us;
    float stddev_us;
};

static TimingResult benchmark(const std::function<void()>& fn, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    std::vector<float> times(iters);
    for (int i = 0; i < iters; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms * 1000.0f;  // → microseconds
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::sort(times.begin(), times.end());
    float median = times[iters / 2];

    // Stddev
    float mean = std::accumulate(times.begin(), times.end(), 0.0f) / iters;
    float var = 0;
    for (float t : times) var += (t - mean) * (t - mean);
    float stddev = std::sqrt(var / iters);

    return {median, stddev};
}

// Fill buffer with non-zero pattern (avoids NaN/denorm weirdness in timing)
static void fill_pattern(void* d_ptr, size_t bytes) {
    std::vector<uint8_t> h_buf(bytes);
    // Repeating pattern that produces valid FP16 and valid Q4_K blocks
    for (size_t i = 0; i < bytes; i++)
        h_buf[i] = (uint8_t)((i * 37 + 13) & 0xFF);
    cudaMemcpy(d_ptr, h_buf.data(), bytes, cudaMemcpyHostToDevice);
}

int main(int argc, char** argv) {
    // Parse args
    std::vector<int> seq_lens = {128, 256, 512};
    int iters = 20;
    int warmup = 5;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            seq_lens.clear();
            char* tok = strtok(argv[++i], ",");
            while (tok) {
                seq_lens.push_back(atoi(tok));
                tok = strtok(nullptr, ",");
            }
        }
    }

    printf("GEMM Approaches Benchmark\n");
    printf("=========================\n");
    printf("Warmup: %d, Iterations: %d\n", warmup, iters);
    printf("Seq lengths: ");
    for (int n : seq_lens) printf("%d ", n);
    printf("\n\n");

    // For each seq_len
    for (int N : seq_lens) {
        printf("━━━ N = %d (seq_len) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", N);
        printf("%-22s  %5s %5s %5s  │ %12s │ %12s │ %12s │ %12s\n",
               "Kernel", "M", "K", "N",
               "FP16 (µs)", "Dq+GEMM (µs)", "Fused Q4K", "Dq only");
        printf("──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n");

        float total_fp16 = 0, total_dqgemm = 0, total_fused = 0, total_dq = 0;
        float weighted_fp16 = 0, weighted_dqgemm = 0, weighted_fused = 0;

        for (int s = 0; s < N_SIZES; s++) {
            const auto& sz = SIZES[s];
            int M = sz.M, K = sz.K;

            // Weight: Q4_K format
            int n_elements = M * K;
            int n_blocks_q4k = n_elements / 256;
            size_t q4k_bytes = n_blocks_q4k * sizeof(gwen::block_q4_k);

            // Allocate
            void*  d_w_q4k   = nullptr;
            half*  d_w_fp16  = nullptr;
            half*  d_x       = nullptr;
            half*  d_y_a     = nullptr;
            half*  d_y_b     = nullptr;
            half*  d_y_c     = nullptr;
            half*  d_temp    = nullptr;
            void*  d_scratch  = nullptr;
            // Scratch for mmq: F32 activations + Q8_1_mmq blocks + F32 output
            int K_padded = ((K + 511) / 512) * 512;  // MATRIX_ROW_PADDING = 512
            size_t scratch_size = (size_t)K * N * sizeof(float)       // F32 activations
                                + (size_t)(K_padded / 128) * N * 144  // Q8_1_mmq blocks (144 bytes each)
                                + (size_t)M * N * sizeof(float)       // F32 output
                                + 70 * 128 * 128 * sizeof(float)      // stream-K fixup buffer (nsm * mmq_x * mmq_y)
                                + 1024 * 1024;                        // extra padding

            cudaMalloc(&d_w_q4k,  q4k_bytes);
            cudaMalloc(&d_w_fp16, n_elements * sizeof(half));
            cudaMalloc(&d_x,      K * N * sizeof(half));
            cudaMalloc(&d_y_a,    M * N * sizeof(half));
            cudaMalloc(&d_y_b,    M * N * sizeof(half));
            cudaMalloc(&d_y_c,    M * N * sizeof(half));
            cudaMalloc(&d_temp,   n_elements * sizeof(half));
            cudaMalloc(&d_scratch, scratch_size);

            // Fill with non-zero data
            fill_pattern(d_w_q4k, q4k_bytes);
            fill_pattern(d_x, K * N * sizeof(half));

            // Pre-dequant for baseline
            gwen::gwen_dequant(d_w_q4k, d_w_fp16, n_elements, gwen::GGMLType::Q4_K, 0);
            cudaDeviceSynchronize();

            // Approach A: Pre-dequanted FP16 + CUTLASS
            auto res_a = benchmark([&]() {
                gwen::gwen_gemm_fp16(d_w_fp16, d_x, d_y_a, M, K, N, 0);
            }, warmup, iters);

            // Approach B: Per-call dequant + CUTLASS
            auto res_b = benchmark([&]() {
                gwen::gwen_dequant(d_w_q4k, d_temp, n_elements, gwen::GGMLType::Q4_K, 0);
                gwen::gwen_gemm_fp16(d_temp, d_x, d_y_b, M, K, N, 0);
            }, warmup, iters);

            // Approach C: Fused mmq GEMM (no temp buffer, no pre-dequant)
            auto res_c = benchmark([&]() {
                gwen::gwen_gemm_mmq(d_w_q4k, gwen::GGMLType::Q4_K, d_x, d_y_c, d_scratch, M, K, N, 0);
            }, warmup, iters);

            // Dequant only (measure raw overhead)
            auto res_dq = benchmark([&]() {
                gwen::gwen_dequant(d_w_q4k, d_temp, n_elements, gwen::GGMLType::Q4_K, 0);
            }, warmup, iters);

            printf("%-22s  %5d %5d %5d  │ %6.0f ± %4.0f │ %6.0f ± %4.0f │ %6.0f ± %4.0f │ %6.0f ± %4.0f\n",
                   sz.name, M, K, N,
                   res_a.median_us, res_a.stddev_us,
                   res_b.median_us, res_b.stddev_us,
                   res_c.median_us, res_c.stddev_us,
                   res_dq.median_us, res_dq.stddev_us);

            total_fp16   += res_a.median_us;
            total_dqgemm += res_b.median_us;
            total_fused  += res_c.median_us;
            total_dq     += res_dq.median_us;
            weighted_fp16   += res_a.median_us * sz.count;
            weighted_dqgemm += res_b.median_us * sz.count;
            weighted_fused  += res_c.median_us * sz.count;

            cudaFree(d_w_q4k);
            cudaFree(d_w_fp16);
            cudaFree(d_x);
            cudaFree(d_y_a);
            cudaFree(d_y_b);
            cudaFree(d_y_c);
            cudaFree(d_temp);
            cudaFree(d_scratch);
        }

        printf("──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n");
        printf("%-22s  %17s  │ %12.0f │ %12.0f │ %12.0f │ %12.0f\n",
               "SUM (1 each)", "", total_fp16, total_dqgemm, total_fused, total_dq);
        printf("%-22s  %17s  │ %12.0f │ %12.0f │ %12.0f │\n",
               "WEIGHTED (full fwd)", "", weighted_fp16, weighted_dqgemm, weighted_fused);
        printf("\n  vs FP16 baseline:  Dq+GEMM %+.1f%%   Fused Q4K %+.1f%%\n",
               (weighted_dqgemm / weighted_fp16 - 1.0f) * 100.0f,
               (weighted_fused / weighted_fp16 - 1.0f) * 100.0f);

        float vram_saved_mb = 0;
        for (int s = 0; s < N_SIZES; s++) {
            vram_saved_mb += (float)SIZES[s].M * SIZES[s].K * sizeof(half) / (1024.0f * 1024.0f) * SIZES[s].count;
        }
        printf("  VRAM saved: %.0f MB\n\n", vram_saved_mb);
    }

    // ======================================================================
    // Pipeline test: simulate a full forward pass with double-buffered dequant
    // ======================================================================
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("PIPELINE TEST: Full forward pass simulation (double-buffered dequant)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    // Layer structure: [3×DeltaNet + 1×FullAttn] × 6
    // DeltaNet GEMMs: attn_qkv, attn_gate, ssm_out, ffn_gate, ffn_up, ffn_down
    // FullAttn GEMMs: attn_q, attn_k, attn_v, attn_out, ffn_gate, ffn_up, ffn_down
    struct LayerGemm { int M, K; };

    LayerGemm deltanet_gemms[] = {
        {6144, 1024}, {2048, 1024}, {1024, 2048},  // attn_qkv, attn_gate, ssm_out
        {3584, 1024}, {3584, 1024}, {1024, 3584},  // ffn_gate, ffn_up, ffn_down
    };
    LayerGemm fullattn_gemms[] = {
        {2048, 1024}, {512, 1024}, {512, 1024}, {1024, 2048},  // attn_q/k/v/out
        {3584, 1024}, {3584, 1024}, {1024, 3584},               // ffn_gate/up/down
    };

    for (int N : seq_lens) {
        // Find max weight size for temp buffers
        int max_elements = 0;
        for (auto& g : deltanet_gemms) max_elements = std::max(max_elements, g.M * g.K);
        for (auto& g : fullattn_gemms) max_elements = std::max(max_elements, g.M * g.K);

        // Collect all GEMMs in forward order: [3×DN + 1×FA] × 6
        struct GemmOp { int M, K; void* w_q4k; half* w_fp16; };
        std::vector<GemmOp> all_gemms;

        // Allocate Q4_K weights and pre-dequanted FP16 for ALL unique weight matrices
        // In reality weights are shared across layers of same type, but for timing
        // we allocate per-layer to match real memory access patterns
        std::vector<void*> allocs;  // track for cleanup

        auto alloc_gemm = [&](int M, int K) -> GemmOp {
            int n_elem = M * K;
            int n_blocks = n_elem / 256;
            size_t q4k_bytes = n_blocks * sizeof(gwen::block_q4_k);
            void* w_q4k;
            half* w_fp16;
            cudaMalloc(&w_q4k, q4k_bytes);
            cudaMalloc(&w_fp16, n_elem * sizeof(half));
            fill_pattern(w_q4k, q4k_bytes);
            gwen::gwen_dequant(w_q4k, w_fp16, n_elem, gwen::GGMLType::Q4_K, 0);
            allocs.push_back(w_q4k);
            allocs.push_back(w_fp16);
            return {M, K, w_q4k, w_fp16};
        };

        for (int rep = 0; rep < 6; rep++) {
            // 3 DeltaNet layers
            for (int d = 0; d < 3; d++)
                for (auto& g : deltanet_gemms)
                    all_gemms.push_back(alloc_gemm(g.M, g.K));
            // 1 FullAttn layer
            for (auto& g : fullattn_gemms)
                all_gemms.push_back(alloc_gemm(g.M, g.K));
        }
        cudaDeviceSynchronize();

        int n_gemms = (int)all_gemms.size();

        // Allocate activation and output buffers
        half* d_x;
        half* d_y;
        cudaMalloc(&d_x, 4096 * N * sizeof(half));  // max K
        cudaMalloc(&d_y, 6144 * N * sizeof(half));   // max M
        fill_pattern(d_x, 4096 * N * sizeof(half));

        // Two temp buffers for double-buffering
        half* d_temp[2];
        cudaMalloc(&d_temp[0], max_elements * sizeof(half));
        cudaMalloc(&d_temp[1], max_elements * sizeof(half));

        // Create streams and events
        cudaStream_t stream_dq, stream_compute;
        cudaStreamCreate(&stream_dq);
        cudaStreamCreate(&stream_compute);

        // --- Approach A: Pre-dequanted FP16 (baseline) ---
        auto res_fp16 = benchmark([&]() {
            for (int i = 0; i < n_gemms; i++) {
                auto& g = all_gemms[i];
                gwen::gwen_gemm_fp16(g.w_fp16, d_x, d_y, g.M, g.K, N, 0);
            }
        }, warmup, iters);

        // --- Approach B: Serial dequant + GEMM (single stream) ---
        auto res_serial = benchmark([&]() {
            for (int i = 0; i < n_gemms; i++) {
                auto& g = all_gemms[i];
                int n_elem = g.M * g.K;
                gwen::gwen_dequant(g.w_q4k, d_temp[0], n_elem, gwen::GGMLType::Q4_K, 0);
                gwen::gwen_gemm_fp16(d_temp[0], d_x, d_y, g.M, g.K, N, 0);
            }
        }, warmup, iters);

        // --- Approach C: Double-buffered dequant (two streams) ---
        auto res_pipeline = benchmark([&]() {
            // Dequant first weight on dq stream
            {
                auto& g0 = all_gemms[0];
                gwen::gwen_dequant(g0.w_q4k, d_temp[0], g0.M * g0.K,
                                   gwen::GGMLType::Q4_K, stream_dq);
            }

            for (int i = 0; i < n_gemms; i++) {
                auto& g = all_gemms[i];
                int cur = i % 2;
                int nxt = 1 - cur;

                // Wait for current dequant to finish before GEMM
                cudaEvent_t dq_done;
                cudaEventCreate(&dq_done);
                cudaEventRecord(dq_done, stream_dq);
                cudaStreamWaitEvent(stream_compute, dq_done);

                // Start GEMM on compute stream
                gwen::gwen_gemm_fp16(d_temp[cur], d_x, d_y, g.M, g.K, N, stream_compute);

                // Start dequanting NEXT weight on dq stream (overlaps with GEMM)
                if (i + 1 < n_gemms) {
                    // Wait for previous GEMM to finish using the nxt buffer
                    cudaEvent_t gemm_done;
                    cudaEventCreate(&gemm_done);
                    cudaEventRecord(gemm_done, stream_compute);
                    cudaStreamWaitEvent(stream_dq, gemm_done);

                    auto& g_next = all_gemms[i + 1];
                    gwen::gwen_dequant(g_next.w_q4k, d_temp[nxt], g_next.M * g_next.K,
                                       gwen::GGMLType::Q4_K, stream_dq);
                    cudaEventDestroy(gemm_done);
                }

                cudaEventDestroy(dq_done);
            }
            // Sync both streams
            cudaStreamSynchronize(stream_compute);
            cudaStreamSynchronize(stream_dq);
        }, warmup, iters);

        printf("N=%d (%d GEMMs per forward):\n", N, n_gemms);
        printf("  Pre-dequant FP16:     %8.0f µs  (%.2f ms)  — baseline\n",
               res_fp16.median_us, res_fp16.median_us / 1000.0f);
        printf("  Serial dq+GEMM:       %8.0f µs  (%.2f ms)  — %+.1f%%\n",
               res_serial.median_us, res_serial.median_us / 1000.0f,
               (res_serial.median_us / res_fp16.median_us - 1.0f) * 100.0f);
        printf("  Double-buffer pipeline:%7.0f µs  (%.2f ms)  — %+.1f%%\n",
               res_pipeline.median_us, res_pipeline.median_us / 1000.0f,
               (res_pipeline.median_us / res_fp16.median_us - 1.0f) * 100.0f);
        printf("  VRAM saved: 924 MB\n\n");

        // Cleanup
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_temp[0]);
        cudaFree(d_temp[1]);
        cudaStreamDestroy(stream_dq);
        cudaStreamDestroy(stream_compute);
        for (void* p : allocs) cudaFree(p);
    }

    return 0;
}
