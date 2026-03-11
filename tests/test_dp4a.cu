#include "gwen/model.h"
#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"
#include "gwen/memory.h"
#include <cstdio>
#include <cmath>

using namespace gwen;

static bool test_dp4a_weight(const char* name, const WeightRef& w,
                              half* d_x, void* d_x_q8,
                              half* d_y_legacy, half* d_y_dp4a) {
    int out_features = w.shape[1];
    int in_features = w.shape[0];
    printf("\n=== %s: [%d, %d] type=%d ===\n", name, (int)w.shape[0], (int)w.shape[1], (int)w.type);

    // Legacy GEMV (FP16 input)
    gwen_gemv(w.device_data, d_x, d_y_legacy, out_features, in_features, w.type);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    // Quantize x to Q8_1
    gwen_quantize_q8_1(d_x, d_x_q8, in_features);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    // dp4a GEMV (Q8_1 input)
    gwen_gemv_dp4a(w.device_data, d_x_q8, d_y_dp4a, out_features, in_features, w.type);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<half> h_legacy(out_features), h_dp4a(out_features);
    GWEN_CHECK_CUDA(cudaMemcpy(h_legacy.data(), d_y_legacy, out_features * sizeof(half), cudaMemcpyDeviceToHost));
    GWEN_CHECK_CUDA(cudaMemcpy(h_dp4a.data(), d_y_dp4a, out_features * sizeof(half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    int worst_idx = 0;
    double sum_legacy = 0, sum_dp4a = 0;
    for (int i = 0; i < out_features; i++) {
        float vl = __half2float(h_legacy[i]);
        float vd = __half2float(h_dp4a[i]);
        sum_legacy += vl;
        sum_dp4a += vd;
        float diff = fabsf(vl - vd);
        if (diff > max_diff) { max_diff = diff; worst_idx = i; }
    }

    // dp4a quantizes input to int8, so there's more error than FP16 vs FP16
    bool pass = max_diff < 0.5f;
    printf("  Max diff: %e at [%d] (legacy=%f, dp4a=%f)\n",
           max_diff, worst_idx,
           __half2float(h_legacy[worst_idx]), __half2float(h_dp4a[worst_idx]));
    printf("  Sum legacy=%f, dp4a=%f, ratio=%f → %s\n",
           sum_legacy, sum_dp4a, sum_dp4a / (sum_legacy + 1e-10),
           pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    auto model = Model::load(argv[1]);
    CudaAllocator alloc;
    model->upload_weights(alloc);

    int max_in = 3584, max_out = 6144;
    half *d_x, *d_y_legacy, *d_y_dp4a;
    void* d_x_q8;
    GWEN_CHECK_CUDA(cudaMalloc(&d_x, max_in * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y_legacy, max_out * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y_dp4a, max_out * sizeof(half)));
    // Q8_1: max_in/32 blocks × 36 bytes
    GWEN_CHECK_CUDA(cudaMalloc(&d_x_q8, (max_in / 32) * sizeof(block_q8_1)));

    // Fill x with varied values
    std::vector<half> h_x(max_in);
    for (int i = 0; i < max_in; i++) h_x[i] = __float2half(0.01f * (i % 100) - 0.5f);
    GWEN_CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), max_in * sizeof(half), cudaMemcpyHostToDevice));

    const auto& dn = model->layers[0].deltanet;
    const auto& fa = model->layers[3].full_attn;

    bool all_pass = true;

    // Q4_K weights
    all_pass &= test_dp4a_weight("FFN gate (Q4_K)", dn.ffn_gate, d_x, d_x_q8, d_y_legacy, d_y_dp4a);
    all_pass &= test_dp4a_weight("FFN up (Q4_K)", dn.ffn_up, d_x, d_x_q8, d_y_legacy, d_y_dp4a);

    // Q5_K weights
    all_pass &= test_dp4a_weight("DeltaNet QKV (Q5_K)", dn.attn_qkv, d_x, d_x_q8, d_y_legacy, d_y_dp4a);
    all_pass &= test_dp4a_weight("DeltaNet gate (Q5_K)", dn.attn_gate, d_x, d_x_q8, d_y_legacy, d_y_dp4a);
    all_pass &= test_dp4a_weight("DeltaNet out (Q5_K)", dn.ssm_out, d_x, d_x_q8, d_y_legacy, d_y_dp4a);

    // Q6_K weight (FA V proj)
    all_pass &= test_dp4a_weight("FA V proj (Q6_K)", fa.attn_v, d_x, d_x_q8, d_y_legacy, d_y_dp4a);

    // Q4_K attention weights
    all_pass &= test_dp4a_weight("FA Q proj", fa.attn_q, d_x, d_x_q8, d_y_legacy, d_y_dp4a);
    all_pass &= test_dp4a_weight("FA output", fa.attn_output, d_x, d_x_q8, d_y_legacy, d_y_dp4a);

    printf("\n%s\n", all_pass ? "ALL DP4A TESTS PASSED" : "SOME DP4A TESTS FAILED");

    cudaFree(d_x); cudaFree(d_y_legacy); cudaFree(d_y_dp4a); cudaFree(d_x_q8);
    return all_pass ? 0 : 1;
}
