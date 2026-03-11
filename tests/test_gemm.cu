#include "gwen/model.h"
#include "gwen/kernels.h"
#include "gwen/memory.h"
#include <cstdio>
#include <cmath>

using namespace gwen;

static bool test_weight(const char* name, const WeightRef& w,
                        half* d_x, half* d_y_gemv, half* d_y_gemm, half* d_temp_w) {
    int out_features = w.shape[1];
    int in_features = w.shape[0];
    printf("\n=== %s: [%d, %d] type=%d ===\n", name, (int)w.shape[0], (int)w.shape[1], (int)w.type);

    gwen_gemv(w.device_data, d_x, d_y_gemv, out_features, in_features, w.type);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    gwen_gemm(w.device_data, w.type, d_temp_w, d_x, d_y_gemm, out_features, in_features, 1);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<half> h_gemv(out_features), h_gemm(out_features);
    GWEN_CHECK_CUDA(cudaMemcpy(h_gemv.data(), d_y_gemv, out_features * sizeof(half), cudaMemcpyDeviceToHost));
    GWEN_CHECK_CUDA(cudaMemcpy(h_gemm.data(), d_y_gemm, out_features * sizeof(half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < out_features; i++) {
        float diff = fabsf(__half2float(h_gemv[i]) - __half2float(h_gemm[i]));
        if (diff > max_diff) { max_diff = diff; worst_idx = i; }
    }

    bool pass = max_diff < 0.1f;
    printf("  Max diff: %e at [%d] (GEMV=%f, GEMM=%f) → %s\n",
           max_diff, worst_idx,
           __half2float(h_gemv[worst_idx]), __half2float(h_gemm[worst_idx]),
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

    // Allocate max-sized buffers
    int max_in = 3584, max_out = 6144;
    half *d_x, *d_y_gemv, *d_y_gemm, *d_temp_w;
    GWEN_CHECK_CUDA(cudaMalloc(&d_x, max_in * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y_gemv, max_out * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y_gemm, max_out * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_temp_w, (size_t)max_out * max_in * sizeof(half)));

    // Fill x
    std::vector<half> h_x(max_in);
    for (int i = 0; i < max_in; i++) h_x[i] = __float2half(0.01f * (i % 100));
    GWEN_CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), max_in * sizeof(half), cudaMemcpyHostToDevice));

    const auto& dn = model->layers[0].deltanet;
    const auto& fa = model->layers[3].full_attn;

    bool all_pass = true;
    // Q4_K weights (FFN)
    all_pass &= test_weight("FFN gate (Q4_K)", dn.ffn_gate, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    all_pass &= test_weight("FFN down (Q4_K/Q6_K)", dn.ffn_down, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    // Q5_K weights (DeltaNet QKV, gate)
    all_pass &= test_weight("DeltaNet QKV (Q5_K)", dn.attn_qkv, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    all_pass &= test_weight("DeltaNet gate (Q5_K)", dn.attn_gate, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    all_pass &= test_weight("DeltaNet out (Q5_K)", dn.ssm_out, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    // Full attention Q/K/V
    all_pass &= test_weight("FA Q proj", fa.attn_q, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    all_pass &= test_weight("FA K proj", fa.attn_k, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    all_pass &= test_weight("FA V proj", fa.attn_v, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    all_pass &= test_weight("FA output", fa.attn_output, d_x, d_y_gemv, d_y_gemm, d_temp_w);
    // Skip embedding — too large (248320 outputs) for test buffers

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    cudaFree(d_x); cudaFree(d_y_gemv); cudaFree(d_y_gemm); cudaFree(d_temp_w);
    return all_pass ? 0 : 1;
}
