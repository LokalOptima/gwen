// Step-by-step forward pass using llama.cpp's own weight access
// Compare intermediate values with GWEN's output
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

// Helper: get tensor data as float vector
std::vector<float> get_tensor_f32(const llama_model* model, const char* name) {
    ggml_tensor* t = llama_get_model_tensor(model, name);
    if (!t) { fprintf(stderr, "Tensor not found: %s\n", name); exit(1); }
    int64_t n = ggml_nelements(t);

    if (t->type == GGML_TYPE_F32) {
        std::vector<float> data(n);
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
        return data;
    }

    // For quantized types, dequantize
    // Get raw bytes and use ggml dequantization
    size_t raw_size = ggml_nbytes(t);
    std::vector<uint8_t> raw(raw_size);
    ggml_backend_tensor_get(t, raw.data(), 0, raw_size);

    std::vector<float> data(n);
    // Use ggml's type traits to dequantize
    const auto* tt = ggml_get_type_traits(t->type);
    if (tt->to_float) {
        tt->to_float(raw.data(), data.data(), n);
    } else {
        fprintf(stderr, "No dequant for type %d\n", t->type);
        exit(1);
    }
    return data;
}

// Get shape
void print_tensor_info(const llama_model* model, const char* name) {
    ggml_tensor* t = llama_get_model_tensor(model, name);
    if (!t) { printf("  %s: NOT FOUND\n", name); return; }
    printf("  %s: type=%d ne=[%lld,%lld,%lld,%lld] n=%lld\n", name,
           (int)t->type, (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3], (long long)ggml_nelements(t));
}

void print_first_n(const char* label, const float* data, int n) {
    printf("  %s:", label);
    for (int i = 0; i < n; i++) printf(" %.6f", data[i]);
    printf("\n");
}

float compute_norm(const float* data, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += data[i] * data[i];
    return sqrtf(sum);
}

void rmsnorm(float* out, const float* x, const float* w, int n, float eps = 1e-6f) {
    float sum_sq = 0;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] / rms * w[i];
}

float silu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float softplus_scalar(float x) {
    return logf(1.0f + expf(x));
}

// Manual GEMV: y[r] = dot(W_row[r], x) for r in [0, out_features)
// W is stored row-major: W[r][c] = w_flat[r * in_features + c]
void gemv(float* y, const float* W, const float* x, int out_features, int in_features) {
    for (int r = 0; r < out_features; r++) {
        float sum = 0;
        for (int c = 0; c < in_features; c++) {
            sum += W[r * in_features + c] * x[c];
        }
        y[r] = sum;
    }
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU only for weight access

    auto* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const int n_embed = 1024;
    const int ssm_inner = 2048;
    const int ssm_n_heads = 16;
    const int ssm_state_size = 128;
    const int token_id = 9419;

    // Print tensor info
    printf("=== Tensor info ===\n");
    print_tensor_info(model, "token_embd.weight");
    print_tensor_info(model, "blk.0.attn_norm.weight");
    print_tensor_info(model, "blk.0.attn_qkv.weight");
    print_tensor_info(model, "blk.0.attn_gate.weight");
    print_tensor_info(model, "blk.0.ssm_conv1d.weight");
    print_tensor_info(model, "blk.0.ssm_a");
    print_tensor_info(model, "blk.0.ssm_dt.bias");
    print_tensor_info(model, "blk.0.ssm_alpha.weight");
    print_tensor_info(model, "blk.0.ssm_beta.weight");
    print_tensor_info(model, "blk.0.ssm_norm.weight");
    print_tensor_info(model, "blk.0.ssm_out.weight");

    // Step 1: Embedding
    printf("\n=== Step 1: Embedding ===\n");
    auto embd_full = get_tensor_f32(model, "token_embd.weight");
    // embd_full is [n_embed * vocab_size], row-major with n_embed as inner dim
    std::vector<float> x(n_embed);
    memcpy(x.data(), &embd_full[token_id * n_embed], n_embed * sizeof(float));
    print_first_n("embed[:10]", x.data(), 10);
    printf("  embed norm: %.6f\n", compute_norm(x.data(), n_embed));

    // Step 2: RMSNorm
    printf("\n=== Step 2: RMSNorm ===\n");
    auto norm_w = get_tensor_f32(model, "blk.0.attn_norm.weight");
    print_first_n("norm_w[:10]", norm_w.data(), 10);

    std::vector<float> x_norm(n_embed);
    rmsnorm(x_norm.data(), x.data(), norm_w.data(), n_embed);
    print_first_n("x_norm[:10]", x_norm.data(), 10);
    printf("  x_norm norm: %.6f\n", compute_norm(x_norm.data(), n_embed));

    // Step 3: QKV projection
    printf("\n=== Step 3: QKV projection ===\n");
    auto qkv_w = get_tensor_f32(model, "blk.0.attn_qkv.weight");
    // qkv_w has n_elems = 1024 * 6144, stored as 6144 rows of 1024
    printf("  qkv_w size: %zu (expect %d)\n", qkv_w.size(), n_embed * ssm_inner * 3);

    std::vector<float> qkv(ssm_inner * 3);
    gemv(qkv.data(), qkv_w.data(), x_norm.data(), ssm_inner * 3, n_embed);
    print_first_n("qkv[:10]", qkv.data(), 10);
    printf("  qkv norm: %.6f\n", compute_norm(qkv.data(), ssm_inner * 3));

    // Step 4: Conv1d (first token, state=0)
    printf("\n=== Step 4: Conv1d ===\n");
    auto conv_w = get_tensor_f32(model, "blk.0.ssm_conv1d.weight");
    printf("  conv_w size: %zu (expect %d)\n", conv_w.size(), 4 * 6144);
    // conv_w is [4, 6144] in GGUF = ne[0]=4, ne[1]=6144
    // Physical: 6144 groups of 4 elements. conv_w[c*4 + k] for channel c, position k.
    // For first token with zero state: output[c] = input[c] * kernel[3, c]
    for (int c = 0; c < 6144; c++) {
        qkv[c] = qkv[c] * conv_w[c * 4 + 3];
    }
    print_first_n("after conv[:10]", qkv.data(), 10);

    // Step 5: SiLU
    printf("\n=== Step 5: SiLU ===\n");
    for (int i = 0; i < 6144; i++) {
        qkv[i] = silu_scalar(qkv[i]);
    }
    print_first_n("after silu[:10]", qkv.data(), 10);

    // Step 6: Split Q/K/V
    printf("\n=== Step 6: Q/K/V split ===\n");
    float* Q = &qkv[0];
    float* K = &qkv[ssm_inner];
    float* V = &qkv[2 * ssm_inner];
    print_first_n("Q[:5]", Q, 5);
    print_first_n("K[:5]", K, 5);
    print_first_n("V[:5]", V, 5);

    // Step 7: L2 normalize Q and K per head
    printf("\n=== Step 7: L2 normalize ===\n");
    for (int h = 0; h < ssm_n_heads; h++) {
        float qnorm = compute_norm(&Q[h * ssm_state_size], ssm_state_size);
        float knorm = compute_norm(&K[h * ssm_state_size], ssm_state_size);
        for (int i = 0; i < ssm_state_size; i++) {
            Q[h * ssm_state_size + i] /= fmaxf(qnorm, 1e-6f);
            K[h * ssm_state_size + i] /= fmaxf(knorm, 1e-6f);
        }
    }
    print_first_n("Q_norm[:5]", Q, 5);
    print_first_n("K_norm[:5]", K, 5);

    // Step 8: Gate and beta computation
    printf("\n=== Step 8: Gate/Beta ===\n");
    auto ssm_a = get_tensor_f32(model, "blk.0.ssm_a");
    auto dt_bias = get_tensor_f32(model, "blk.0.ssm_dt.bias");
    auto alpha_w = get_tensor_f32(model, "blk.0.ssm_alpha.weight");
    auto beta_w = get_tensor_f32(model, "blk.0.ssm_beta.weight");

    print_first_n("ssm_a[:4]", ssm_a.data(), 4);
    print_first_n("dt_bias[:4]", dt_bias.data(), 4);

    std::vector<float> gates(ssm_n_heads), betas(ssm_n_heads);
    for (int h = 0; h < ssm_n_heads; h++) {
        // alpha projection: dot(alpha_w[h], x_norm)
        float alpha_proj = 0;
        for (int c = 0; c < n_embed; c++) {
            alpha_proj += alpha_w[h * n_embed + c] * x_norm[c];
        }
        // beta projection: dot(beta_w[h], x_norm)
        float beta_proj = 0;
        for (int c = 0; c < n_embed; c++) {
            beta_proj += beta_w[h * n_embed + c] * x_norm[c];
        }
        float sp = softplus_scalar(alpha_proj + dt_bias[h]);
        gates[h] = ssm_a[h] * sp;
        betas[h] = sigmoid_scalar(beta_proj);
    }
    print_first_n("gates[:4]", gates.data(), 4);
    print_first_n("betas[:4]", betas.data(), 4);

    // Step 9: DeltaNet (first token, state=0)
    printf("\n=== Step 9: DeltaNet ===\n");
    std::vector<float> attn_out(ssm_inner, 0.0f);
    for (int h = 0; h < ssm_n_heads; h++) {
        int s = h * ssm_state_size;
        // With zero state: output = beta * dot(k, q) * v
        float dot_kq = 0;
        for (int i = 0; i < ssm_state_size; i++) {
            dot_kq += K[s + i] * Q[s + i];
        }
        for (int i = 0; i < ssm_state_size; i++) {
            attn_out[s + i] = dot_kq * betas[h] * V[s + i];
        }
    }
    print_first_n("attn_out[:10]", attn_out.data(), 10);

    // Step 10: Gated RMSNorm (output * SiLU(z))
    printf("\n=== Step 10: Gated RMSNorm ===\n");
    auto gate_w_full = get_tensor_f32(model, "blk.0.attn_gate.weight");
    std::vector<float> gate_z(ssm_inner);
    gemv(gate_z.data(), gate_w_full.data(), x_norm.data(), ssm_inner, n_embed);

    auto ssm_norm_w = get_tensor_f32(model, "blk.0.ssm_norm.weight");
    std::vector<float> gated_out(ssm_inner);
    for (int h = 0; h < ssm_n_heads; h++) {
        int s = h * ssm_state_size;
        // Per-head RMSNorm
        float rms_sq = 0;
        for (int i = 0; i < ssm_state_size; i++) rms_sq += attn_out[s+i] * attn_out[s+i];
        float rms = sqrtf(rms_sq / ssm_state_size + 1e-6f);
        for (int i = 0; i < ssm_state_size; i++) {
            float normed = attn_out[s+i] / rms * ssm_norm_w[i];
            float silu_g = silu_scalar(gate_z[s + i]);
            gated_out[s + i] = normed * silu_g;
        }
    }
    print_first_n("gated_out[:10]", gated_out.data(), 10);

    // Step 11: Output projection
    printf("\n=== Step 11: Output projection ===\n");
    auto out_w = get_tensor_f32(model, "blk.0.ssm_out.weight");
    std::vector<float> out_proj(n_embed);
    gemv(out_proj.data(), out_w.data(), gated_out.data(), n_embed, ssm_inner);
    print_first_n("out_proj[:10]", out_proj.data(), 10);

    // Step 12: Residual
    printf("\n=== Step 12: Residual ===\n");
    for (int i = 0; i < n_embed; i++) x[i] += out_proj[i];
    print_first_n("x_residual[:10]", x.data(), 10);
    printf("  x_residual norm: %.6f\n", compute_norm(x.data(), n_embed));

    // Quick logit check after layer 0
    printf("\n=== Quick logit check ===\n");
    auto output_norm_w = get_tensor_f32(model, "output_norm.weight");
    std::vector<float> final_norm(n_embed);
    rmsnorm(final_norm.data(), x.data(), output_norm_w.data(), n_embed);

    // Compute logit for token 0 and token 11
    float logit0 = 0, logit11 = 0;
    for (int c = 0; c < n_embed; c++) {
        logit0  += embd_full[0 * n_embed + c] * final_norm[c];
        logit11 += embd_full[11 * n_embed + c] * final_norm[c];
    }
    printf("  Early logit[0] after layer 0: %.4f\n", logit0);
    printf("  Early logit[11] after layer 0: %.4f\n", logit11);

    llama_model_free(model);
    return 0;
}
