#include "gwen/model.h"
#include "gwen/kernels.h"
#include "gwen/memory.h"
#include <cstdio>
#include <cmath>

using namespace gwen;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    auto model = Model::load(argv[1]);
    CudaAllocator alloc;
    model->upload_weights(alloc);

    // Use first DeltaNet layer's ffn_gate weight for testing
    const auto& w = model->layers[0].deltanet.ffn_gate;
    int out_features = w.shape[1];
    int in_features = w.shape[0];
    printf("FFN gate weight: shape[0]=%d, shape[1]=%d, type=%d\n",
           w.shape[0], w.shape[1], (int)w.type);
    printf("  → out_features=%d, in_features=%d\n", out_features, in_features);

    // Allocate test vectors
    half *d_x, *d_y_gemv, *d_y_gemm, *d_temp_w;
    GWEN_CHECK_CUDA(cudaMalloc(&d_x, in_features * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y_gemv, out_features * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_y_gemm, out_features * sizeof(half)));
    GWEN_CHECK_CUDA(cudaMalloc(&d_temp_w, (size_t)out_features * in_features * sizeof(half)));

    // Fill x with a simple pattern
    std::vector<half> h_x(in_features);
    for (int i = 0; i < in_features; i++) {
        h_x[i] = __float2half(0.01f * (i % 100));
    }
    GWEN_CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), in_features * sizeof(half), cudaMemcpyHostToDevice));

    // Run GEMV
    gwen_gemv(w.device_data, d_x, d_y_gemv, out_features, in_features, w.type);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    // Run GEMM with seq_len=1
    gwen_gemm(w.device_data, w.type, d_temp_w, d_x, d_y_gemm, out_features, in_features, 1);
    GWEN_CHECK_CUDA(cudaDeviceSynchronize());

    // Compare results
    std::vector<half> h_gemv(out_features), h_gemm(out_features);
    GWEN_CHECK_CUDA(cudaMemcpy(h_gemv.data(), d_y_gemv, out_features * sizeof(half), cudaMemcpyDeviceToHost));
    GWEN_CHECK_CUDA(cudaMemcpy(h_gemm.data(), d_y_gemm, out_features * sizeof(half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < out_features; i++) {
        float v_gemv = __half2float(h_gemv[i]);
        float v_gemm = __half2float(h_gemm[i]);
        float diff = fabsf(v_gemv - v_gemm);
        float rel = diff / (fabsf(v_gemv) + 1e-6f);
        if (diff > max_diff) {
            max_diff = diff;
            worst_idx = i;
        }
        if (rel > max_rel_diff) max_rel_diff = rel;
    }

    printf("\nGEMV vs GEMM comparison (seq_len=1):\n");
    printf("  Max absolute diff: %e (at idx %d)\n", max_diff, worst_idx);
    printf("  Max relative diff: %e\n", max_rel_diff);
    printf("  GEMV[%d] = %f, GEMM[%d] = %f\n",
           worst_idx, __half2float(h_gemv[worst_idx]),
           worst_idx, __half2float(h_gemm[worst_idx]));

    // Print first 10 values
    printf("\nFirst 10 values:\n");
    for (int i = 0; i < 10 && i < out_features; i++) {
        printf("  [%d] GEMV=%f  GEMM=%f  diff=%e\n",
               i, __half2float(h_gemv[i]), __half2float(h_gemm[i]),
               fabsf(__half2float(h_gemv[i]) - __half2float(h_gemm[i])));
    }

    bool pass = max_diff < 0.1f;
    printf("\n%s (threshold 0.1)\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y_gemv); cudaFree(d_y_gemm); cudaFree(d_temp_w);
    gwen_cublas_destroy();
    return pass ? 0 : 1;
}
