// Test GWEN's GEMV against F32 reference using ggml dequantization
// This isolates whether the GEMV kernels are correct
#include "gwen/model.h"
#include "gwen/kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace gwen;

int main() {
    // Load model
    auto model = Model::load("Qwen3.5-9B-UD-Q4_K_XL.gguf");

    CudaAllocator allocator;
    model->upload_weights(allocator);

    const auto& cfg = model->config;

    // Use the embedding of token 760 as input vector
    half* d_x;
    cudaMalloc(&d_x, cfg.n_embed * sizeof(half));
    gwen_embed_lookup(model->token_embd.device_data, model->token_embd.type,
                      760, d_x, cfg.n_embed);

    // Apply RMSNorm (layer 0)
    half* d_x_norm;
    cudaMalloc(&d_x_norm, cfg.n_embed * sizeof(half));
    const auto& w = model->layers[0].deltanet;
    gwen_rmsnorm_f32w(d_x, static_cast<const float*>(w.attn_norm.device_data),
                      d_x_norm, cfg.n_embed, cfg.rms_norm_eps);

    // Compute QKV via GWEN's GEMV
    half* d_qkv;
    cudaMalloc(&d_qkv, cfg.ssm_inner_size * 3 * sizeof(half));
    gwen_gemv(w.attn_qkv.device_data, d_x_norm, d_qkv,
              w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type);

    cudaDeviceSynchronize();

    // Download results
    std::vector<half> h_x_norm(cfg.n_embed);
    std::vector<half> h_qkv(cfg.ssm_inner_size * 3);
    cudaMemcpy(h_x_norm.data(), d_x_norm, cfg.n_embed * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_qkv.data(), d_qkv, cfg.ssm_inner_size * 3 * sizeof(half), cudaMemcpyDeviceToHost);

    printf("x_norm[:5]:");
    for (int i = 0; i < 5; i++) printf(" %.6f", __half2float(h_x_norm[i]));
    printf("\n");

    printf("qkv[:5] (GWEN):");
    for (int i = 0; i < 5; i++) printf(" %.6f", __half2float(h_qkv[i]));
    printf("\n");

    // Now compute the same with F32 reference
    // 1. Get embedding row for token 760 via GWEN's embedding
    std::vector<float> x_f32(cfg.n_embed);
    for (int i = 0; i < (int)cfg.n_embed; i++) x_f32[i] = __half2float(h_x_norm[i]);

    // 2. Get QKV weight (dequantize to F32 on CPU)
    // We need to access the raw GGUF data and dequantize
    const auto& qkv_tensor = w.attn_qkv;
    printf("\nQKV weight: shape=[%d, %d], type=%d\n",
           qkv_tensor.shape[0], qkv_tensor.shape[1], (int)qkv_tensor.type);

    // Dump GWEN's GEMV output to file
    FILE* fp = fopen("/tmp/gwen_qkv.bin", "wb");
    if (fp) {
        std::vector<float> qkv_f32(cfg.ssm_inner_size * 3);
        for (int i = 0; i < (int)(cfg.ssm_inner_size * 3); i++)
            qkv_f32[i] = __half2float(h_qkv[i]);
        fwrite(qkv_f32.data(), sizeof(float), cfg.ssm_inner_size * 3, fp);
        fclose(fp);
        printf("Dumped GWEN QKV to /tmp/gwen_qkv.bin\n");
    }

    // Dump x_norm to file
    fp = fopen("/tmp/gwen_xnorm.bin", "wb");
    if (fp) {
        fwrite(x_f32.data(), sizeof(float), cfg.n_embed, fp);
        fclose(fp);
        printf("Dumped x_norm to /tmp/gwen_xnorm.bin\n");
    }

    // Also test the final LM head GEMV
    // Use the GWEN final x_norm from the previous run (if available)
    FILE* fp2 = fopen("/tmp/gwen_x_norm.bin", "rb");
    if (fp2) {
        std::vector<float> final_xnorm(cfg.n_embed);
        fread(final_xnorm.data(), sizeof(float), cfg.n_embed, fp2);
        fclose(fp2);

        // Upload to GPU as half
        std::vector<half> final_xnorm_h(cfg.n_embed);
        for (int i = 0; i < (int)cfg.n_embed; i++)
            final_xnorm_h[i] = __float2half(final_xnorm[i]);

        half* d_final_xnorm;
        cudaMalloc(&d_final_xnorm, cfg.n_embed * sizeof(half));
        cudaMemcpy(d_final_xnorm, final_xnorm_h.data(), cfg.n_embed * sizeof(half), cudaMemcpyHostToDevice);

        // Compute logits via GWEN's GEMV
        half* d_logits;
        cudaMalloc(&d_logits, cfg.n_vocab * sizeof(half));
        gwen_gemv(model->token_embd.device_data, d_final_xnorm, d_logits,
                  cfg.n_vocab, cfg.n_embed, model->token_embd.type);
        cudaDeviceSynchronize();

        // Download first 20 logits
        std::vector<half> h_logits(20);
        cudaMemcpy(h_logits.data(), d_logits, 20 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("\nLM head logits from GWEN x_norm (first 20):\n");
        for (int i = 0; i < 20; i++) printf("  logit[%d] = %.4f\n", i, __half2float(h_logits[i]));

        cudaFree(d_final_xnorm);
        cudaFree(d_logits);
    }

    cudaFree(d_x);
    cudaFree(d_x_norm);
    cudaFree(d_qkv);

    return 0;
}
