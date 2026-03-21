#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"
#include "gwen/model.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace gwen;

// ============================================================
// CPU reference dequantization (for verification)
// ============================================================

static void cpu_dequant_q8_0(const void* src, float* dst, int n) {
    const block_q8_0* blocks = static_cast<const block_q8_0*>(src);
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        float d;
        memcpy(&d, &blocks[b].d, sizeof(half));
        // Convert half to float manually
        d = __half2float(blocks[b].d);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = d * blocks[b].qs[j];
        }
    }
}

static void cpu_dequant_q4_k(const void* src, float* dst, int n) {
    const block_q4_k* blocks = static_cast<const block_q4_k*>(src);
    int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        const auto& blk = blocks[b];
        float d = __half2float(blk.d);
        float dmin = __half2float(blk.dmin);

        for (int tid = 0; tid < 256; tid++) {
            int sub_block = tid / 32;

            uint8_t sc_lo, m_lo;
            if (sub_block < 4) {
                sc_lo = blk.scales[sub_block] & 0x3F;
                m_lo  = blk.scales[sub_block + 4] & 0x3F;
            } else {
                sc_lo = (blk.scales[sub_block + 4] & 0xF) | ((blk.scales[sub_block - 4] >> 6) << 4);
                m_lo  = (blk.scales[sub_block + 4] >> 4) | ((blk.scales[sub_block] >> 6) << 4);
            }

            float scale = d * sc_lo;
            float min = dmin * m_lo;

            uint8_t q_byte;
            int q_val;
            if (tid < 128) {
                q_byte = blk.qs[tid / 2];
                q_val = (tid % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
            } else {
                q_byte = blk.qs[(tid - 128) / 2 + 64];
                q_val = ((tid - 128) % 2 == 0) ? (q_byte & 0xF) : (q_byte >> 4);
            }

            dst[b * 256 + tid] = scale * q_val - min;
        }
    }
}

static void cpu_dequant_q6_k(const void* src, float* dst, int n) {
    const block_q6_k* blocks = static_cast<const block_q6_k*>(src);
    int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        const auto& blk = blocks[b];
        float d = __half2float(blk.d);

        for (int tid = 0; tid < 256; tid++) {
            int sub_group = tid / 16;
            int8_t scale = blk.scales[sub_group];

            int ql_idx = tid / 2;
            int ql_nibble;
            if (tid % 2 == 0) {
                ql_nibble = blk.ql[ql_idx] & 0xF;
            } else {
                ql_nibble = blk.ql[ql_idx] >> 4;
            }

            int qh_idx = tid / 4;
            int qh_shift = (tid % 4) * 2;
            int qh_bits = (blk.qh[qh_idx] >> qh_shift) & 0x3;

            int q_val = ql_nibble | (qh_bits << 4);
            dst[b * 256 + tid] = d * scale * (q_val - 32);
        }
    }
}

// ============================================================
// Test utilities
// ============================================================

struct TestResult {
    const char* name;
    bool passed;
    float max_abs_err;
    float mean_abs_err;
    int n_elements;
};

static void print_result(const TestResult& r) {
    printf("  [%s] %s — max_err=%.6e, mean_err=%.6e (%d elements)\n",
           r.passed ? "PASS" : "FAIL", r.name, r.max_abs_err, r.mean_abs_err, r.n_elements);
}

static TestResult compare_arrays(const char* name, const float* ref, const half* gpu_out,
                                  int n, float tolerance) {
    // Download GPU result
    std::vector<half> h_out(n);
    GWEN_CHECK_CUDA(cudaMemcpy(h_out.data(), gpu_out, n * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0;
    double sum_err = 0;
    for (int i = 0; i < n; i++) {
        float gpu_val = __half2float(h_out[i]);
        float err = fabsf(ref[i] - gpu_val);
        max_err = fmaxf(max_err, err);
        sum_err += err;
    }
    float mean_err = (float)(sum_err / n);

    TestResult r;
    r.name = name;
    r.passed = max_err < tolerance;
    r.max_abs_err = max_err;
    r.mean_abs_err = mean_err;
    r.n_elements = n;
    return r;
}

// ============================================================
// Test: Dequantization
// ============================================================

static TestResult test_dequant(const char* name, const void* host_quant_data,
                                int n_elements, GGMLType type,
                                void (*cpu_dequant)(const void*, float*, int)) {
    size_t quant_bytes = ggml_type_size(type, n_elements);

    // Upload quantized data to GPU
    void* d_quant;
    GWEN_CHECK_CUDA(cudaMalloc(&d_quant, quant_bytes));
    GWEN_CHECK_CUDA(cudaMemcpy(d_quant, host_quant_data, quant_bytes, cudaMemcpyHostToDevice));

    // GPU dequant
    half* d_out;
    GWEN_CHECK_CUDA(cudaMalloc(&d_out, n_elements * sizeof(half)));
    gwen_dequant(d_quant, d_out, n_elements, type);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    // CPU reference
    std::vector<float> cpu_out(n_elements);
    cpu_dequant(host_quant_data, cpu_out.data(), n_elements);

    auto result = compare_arrays(name, cpu_out.data(), d_out, n_elements, 0.001f);

    cudaFree(d_quant);
    cudaFree(d_out);
    return result;
}

// ============================================================
// Test: GEMV
// ============================================================

static TestResult test_gemv(const char* name, const void* host_W, const float* host_x,
                             int out_features, int in_features, GGMLType type,
                             void (*cpu_dequant)(const void*, float*, int)) {
    size_t W_bytes = ggml_type_size(type, (size_t)out_features * in_features);

    // Dequantize full weight matrix on CPU for reference
    std::vector<float> W_f32((size_t)out_features * in_features);
    for (int row = 0; row < out_features; row++) {
        size_t row_bytes = ggml_type_size(type, in_features);
        const uint8_t* row_data = static_cast<const uint8_t*>(host_W) + row * row_bytes;
        cpu_dequant(row_data, W_f32.data() + (size_t)row * in_features, in_features);
    }

    // CPU GEMV reference
    std::vector<float> ref_y(out_features, 0.0f);
    for (int i = 0; i < out_features; i++) {
        double acc = 0;
        for (int j = 0; j < in_features; j++) {
            acc += (double)W_f32[(size_t)i * in_features + j] * host_x[j];
        }
        ref_y[i] = (float)acc;
    }

    // Upload to GPU
    void* d_W;
    GWEN_CHECK_CUDA(cudaMalloc(&d_W, W_bytes));
    GWEN_CHECK_CUDA(cudaMemcpy(d_W, host_W, W_bytes, cudaMemcpyHostToDevice));

    half* d_x;
    GWEN_CHECK_CUDA(cudaMalloc(&d_x, in_features * sizeof(half)));
    std::vector<half> h_x(in_features);
    for (int i = 0; i < in_features; i++) h_x[i] = __float2half(host_x[i]);
    GWEN_CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), in_features * sizeof(half), cudaMemcpyHostToDevice));

    half* d_y;
    GWEN_CHECK_CUDA(cudaMalloc(&d_y, out_features * sizeof(half)));

    gwen_gemv(d_W, d_x, d_y, out_features, in_features, type);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = compare_arrays(name, ref_y.data(), d_y, out_features, 1.0f);

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y);
    return result;
}

// ============================================================
// Test: RMSNorm
// ============================================================

static TestResult test_rmsnorm() {
    const int dim = 1024;
    const float eps = 1e-6f;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x_f32(dim), w_f32(dim), ref_y(dim);
    for (int i = 0; i < dim; i++) {
        x_f32[i] = dist(rng);
        w_f32[i] = dist(rng) * 0.1f + 1.0f;
    }

    // CPU reference
    double sum_sq = 0;
    for (int i = 0; i < dim; i++) sum_sq += (double)x_f32[i] * x_f32[i];
    float rms_inv = 1.0f / sqrtf((float)(sum_sq / dim) + eps);
    for (int i = 0; i < dim; i++) ref_y[i] = x_f32[i] * rms_inv * w_f32[i];

    // GPU
    std::vector<half> h_x(dim), h_y(dim);
    for (int i = 0; i < dim; i++) h_x[i] = __float2half(x_f32[i]);

    half *d_x, *d_y;
    float *d_w;
    GWEN_CHECK_CUDA(cudaMalloc(&d_x, dim * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y, dim * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_w, dim * sizeof(float)));
    GWEN_CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), dim * sizeof(half), cudaMemcpyHostToDevice));
    GWEN_CHECK_CUDA(cudaMemcpy(d_w, w_f32.data(), dim * sizeof(float), cudaMemcpyHostToDevice));

    gwen_rmsnorm_f32w(d_x, d_w, d_y, dim, eps);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = compare_arrays("RMSNorm (dim=1024)", ref_y.data(), d_y, dim, 0.01f);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_w);
    return result;
}

// ============================================================
// Test: SwiGLU
// ============================================================

static TestResult test_swiglu() {
    const int n = 3584;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> gate_f32(n), up_f32(n), ref_y(n);
    for (int i = 0; i < n; i++) {
        gate_f32[i] = dist(rng);
        up_f32[i] = dist(rng);
    }

    // CPU reference
    for (int i = 0; i < n; i++) {
        float g = gate_f32[i];
        float sig = 1.0f / (1.0f + expf(-g));
        ref_y[i] = g * sig * up_f32[i];
    }

    // GPU
    std::vector<half> h_gate(n), h_up(n);
    for (int i = 0; i < n; i++) {
        h_gate[i] = __float2half(gate_f32[i]);
        h_up[i] = __float2half(up_f32[i]);
    }

    half *d_gate, *d_up, *d_y;
    GWEN_CHECK_CUDA(cudaMalloc(&d_gate, n * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_up, n * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    GWEN_CHECK_CUDA(cudaMemcpy(d_up, h_up.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    gwen_swiglu(d_gate, d_up, d_y, n);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = compare_arrays("SwiGLU (dim=3584)", ref_y.data(), d_y, n, 0.01f);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_y);
    return result;
}

// ============================================================
// Test: Softmax
// ============================================================

static TestResult test_softmax() {
    const int cols = 512;
    const int rows = 8;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x_f32(rows * cols), ref_y(rows * cols);
    for (int i = 0; i < rows * cols; i++) x_f32[i] = dist(rng);

    // CPU reference
    for (int r = 0; r < rows; r++) {
        float max_val = *std::max_element(x_f32.begin() + r * cols, x_f32.begin() + (r + 1) * cols);
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            ref_y[r * cols + c] = expf(x_f32[r * cols + c] - max_val);
            sum += ref_y[r * cols + c];
        }
        for (int c = 0; c < cols; c++) ref_y[r * cols + c] /= sum;
    }

    // GPU
    std::vector<half> h_x(rows * cols);
    for (int i = 0; i < rows * cols; i++) h_x[i] = __float2half(x_f32[i]);

    half *d_x, *d_y;
    GWEN_CHECK_CUDA(cudaMalloc(&d_x, rows * cols * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y, rows * cols * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), rows * cols * sizeof(half), cudaMemcpyHostToDevice));

    gwen_softmax(d_x, d_y, rows, cols);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = compare_arrays("Softmax (8x512)", ref_y.data(), d_y, rows * cols, 0.01f);

    cudaFree(d_x);
    cudaFree(d_y);
    return result;
}

// ============================================================
// Test with real model weights
// ============================================================

static void test_with_model(const std::string& model_path) {
    printf("\n=== Testing with real model weights ===\n");

    auto model = Model::load(model_path);
    const auto& cfg = model->config;

    // Test Q6_K dequant on embedding (first 1024 elements = 4 Q6_K blocks)
    {
        const auto& embd = model->token_embd;
        int n = 1024;  // one row

        std::vector<float> cpu_out(n);
        cpu_dequant_q6_k(embd.host_data, cpu_out.data(), n);

        void* d_quant;
        GWEN_CHECK_CUDA(cudaMalloc(&d_quant, ggml_type_size(GGMLType::Q6_K, n)));
        GWEN_CHECK_CUDA(cudaMemcpy(d_quant, embd.host_data,
                        ggml_type_size(GGMLType::Q6_K, n), cudaMemcpyHostToDevice));

        half* d_out;
        GWEN_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(half)));
        gwen_dequant_q6_k(d_quant, d_out, n);
        GWEN_CHECK_CUDA(cudaDeviceSynchronize());

        auto r = compare_arrays("Q6_K dequant (embedding row 0)", cpu_out.data(), d_out, n, 0.001f);
        print_result(r);

        cudaFree(d_quant);
        cudaFree(d_out);
    }

    // Test Q4_K GEMV on a real FFN gate weight [1024, 3584] (but stored as [3584, 1024] in row-major)
    // Actually GGUF stores weights as [out_features, in_features] in row-major order
    {
        const auto& w = model->layers[0].deltanet.ffn_gate;
        // ffn_gate: [1024, 3584] means in_features=1024, out_features=3584
        int out_f = w.shape[1];  // 3584
        int in_f = w.shape[0];   // 1024

        printf("  FFN gate shape: [%lu, %lu] %s\n", w.shape[0], w.shape[1], ggml_type_name(w.type));

        // Random input
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::vector<float> x_f32(in_f);
        for (int i = 0; i < in_f; i++) x_f32[i] = dist(rng);

        auto r = test_gemv("Q4_K GEMV (ffn_gate 1024->3584)", w.host_data, x_f32.data(),
                           out_f, in_f, w.type, cpu_dequant_q4_k);
        print_result(r);
    }

    // Test embedding lookup
    {
        CudaAllocator alloc;
        void* d_embd = alloc.upload(model->token_embd.host_data, model->token_embd.size_bytes);

        half* d_out;
        GWEN_CHECK_CUDA(cudaMalloc(&d_out, cfg.n_embed * sizeof(half)));

        gwen_embed_lookup(d_embd, model->token_embd.type, 0, d_out, cfg.n_embed);
        GWEN_CHECK_CUDA(cudaDeviceSynchronize());

        // Compare against CPU dequant of first row
        std::vector<float> cpu_out(cfg.n_embed);
        cpu_dequant_q6_k(model->token_embd.host_data, cpu_out.data(), cfg.n_embed);

        auto r = compare_arrays("Embedding lookup (token 0)", cpu_out.data(), d_out, cfg.n_embed, 0.001f);
        print_result(r);

        cudaFree(d_out);
    }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    printf("=== GWEN Kernel Tests ===\n\n");

    std::vector<TestResult> results;

    // Test RMSNorm
    results.push_back(test_rmsnorm());
    print_result(results.back());

    // Test SwiGLU
    results.push_back(test_swiglu());
    print_result(results.back());

    // Test Softmax
    results.push_back(test_softmax());
    print_result(results.back());

    // Test with real model weights if available
    std::string model_path = "Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf";
    if (argc > 1) model_path = argv[1];

    FILE* f = fopen(model_path.c_str(), "rb");
    if (f) {
        fclose(f);
        test_with_model(model_path);
    } else {
        printf("\nModel file not found at %s — skipping model weight tests\n", model_path.c_str());
    }

    // Summary
    printf("\n=== Summary ===\n");
    int passed = 0;
    for (auto& r : results) {
        if (r.passed) passed++;
    }
    printf("  %d/%zu tests passed\n", passed, results.size());

    return (passed == (int)results.size()) ? 0 : 1;
}
