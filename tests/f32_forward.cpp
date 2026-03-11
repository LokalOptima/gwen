// Full forward pass in F32 using ggml's dequantization, then compare final logits
// This will tell us if the model interpretation is correct (F32 should match llama.cpp)
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>

// ============================================================
// GGUF Parser (minimal, for reading tensor data)
// ============================================================
struct TensorInfo {
    std::vector<uint64_t> shape;
    int type;
    uint64_t offset;
    uint64_t n_elems;
};

struct GGUFReader {
    FILE* fp;
    std::map<std::string, TensorInfo> tensors;
    uint64_t data_start;
    const uint8_t* mmap_data;
    size_t file_size;

    static std::string read_string(FILE* f) {
        uint64_t len; fread(&len, 8, 1, f);
        std::string s(len, 0);
        fread(&s[0], 1, len, f);
        return s;
    }

    static void skip_value(FILE* f, uint32_t vtype) {
        if (vtype == 0) { fseek(f, 1, SEEK_CUR); }
        else if (vtype == 1) { fseek(f, 1, SEEK_CUR); }
        else if (vtype == 4 || vtype == 5 || vtype == 6) { fseek(f, 4, SEEK_CUR); }
        else if (vtype == 7) { fseek(f, 1, SEEK_CUR); }
        else if (vtype == 8) { read_string(f); }
        else if (vtype == 9) {
            uint32_t atype; fread(&atype, 4, 1, f);
            uint64_t alen; fread(&alen, 8, 1, f);
            for (uint64_t i = 0; i < alen; i++) skip_value(f, atype);
        }
        else if (vtype == 10 || vtype == 11 || vtype == 12) { fseek(f, 8, SEEK_CUR); }
    }

    void open(const char* path) {
        fp = fopen(path, "rb");
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        uint32_t magic; fread(&magic, 4, 1, fp);
        uint32_t version; fread(&version, 4, 1, fp);
        uint64_t n_tensors; fread(&n_tensors, 8, 1, fp);
        uint64_t n_kv; fread(&n_kv, 8, 1, fp);

        for (uint64_t i = 0; i < n_kv; i++) {
            read_string(fp);
            uint32_t vtype; fread(&vtype, 4, 1, fp);
            skip_value(fp, vtype);
        }

        for (uint64_t i = 0; i < n_tensors; i++) {
            std::string name = read_string(fp);
            uint32_t n_dims; fread(&n_dims, 4, 1, fp);
            TensorInfo t;
            t.shape.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; d++) {
                fread(&t.shape[d], 8, 1, fp);
            }
            fread(&t.type, 4, 1, fp);
            fread(&t.offset, 8, 1, fp);
            t.n_elems = 1;
            for (auto d : t.shape) t.n_elems *= d;
            tensors[name] = t;
        }

        data_start = ((ftell(fp) + 31) / 32) * 32;

        // mmap
        fseek(fp, 0, SEEK_SET);
        mmap_data = (const uint8_t*)malloc(file_size);
        fread((void*)mmap_data, 1, file_size, fp);
    }

    const uint8_t* tensor_data(const std::string& name) {
        auto& t = tensors.at(name);
        return mmap_data + data_start + t.offset;
    }

    // Dequantize entire tensor to F32
    std::vector<float> dequant(const std::string& name) {
        auto& t = tensors.at(name);
        std::vector<float> result(t.n_elems);
        const uint8_t* raw = tensor_data(name);

        if (t.type == 0) { // F32
            memcpy(result.data(), raw, t.n_elems * 4);
        } else {
            ggml_type gtype = (ggml_type)t.type;
            auto traits = ggml_get_type_traits(gtype);
            if (traits->to_float) {
                traits->to_float(raw, result.data(), t.n_elems);
            } else {
                fprintf(stderr, "No dequant for type %d (tensor %s)\n", t.type, name.c_str());
                exit(1);
            }
        }
        return result;
    }

    // Get just one row (for embedding lookup etc.)
    std::vector<float> dequant_row(const std::string& name, int row) {
        auto& t = tensors.at(name);
        int in_dim = t.shape[0];
        int block_size = ggml_blck_size((ggml_type)t.type);
        int type_size = ggml_type_size((ggml_type)t.type);
        int blocks_per_row = in_dim / block_size;
        int row_bytes = blocks_per_row * type_size;

        std::vector<float> result(in_dim);
        const uint8_t* raw = tensor_data(name) + (size_t)row * row_bytes;

        if (t.type == 0) {
            memcpy(result.data(), raw, in_dim * 4);
        } else {
            ggml_type gtype = (ggml_type)t.type;
            auto traits = ggml_get_type_traits(gtype);
            traits->to_float(raw, result.data(), in_dim);
        }
        return result;
    }
};

// ============================================================
// Math helpers
// ============================================================
void rmsnorm(float* out, const float* x, const float* w, int n, float eps = 1e-6f) {
    float sum_sq = 0;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] / rms * w[i];
}

void gemv(float* y, const float* W, const float* x, int out_features, int in_features) {
    for (int r = 0; r < out_features; r++) {
        float acc = 0;
        for (int c = 0; c < in_features; c++) {
            acc += W[(size_t)r * in_features + c] * x[c];
        }
        y[r] = acc;
    }
}

float silu_f(float x) { return x / (1.0f + expf(-x)); }
float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
float softplus_f(float x) { return logf(1.0f + expf(x)); }

float vec_norm(const float* x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i]*x[i];
    return sqrtf(s);
}

void print10(const char* label, const float* x) {
    printf("  %s:", label);
    for (int i = 0; i < 10; i++) printf(" %.4f", x[i]);
    printf("\n");
}

// ============================================================

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";

    printf("Loading GGUF...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    GGUFReader gguf;
    gguf.open(model_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Loaded in %.1f ms\n", std::chrono::duration<double, std::milli>(t1-t0).count());

    const int N = 1024;    // n_embed
    const int FF = 3584;   // n_ff
    const int INNER = 2048; // ssm_inner
    const int HEADS = 16;   // ssm_n_heads
    const int DK = 128;     // ssm_state_size
    const int N_HEAD = 8;   // full attn heads
    const int N_KV = 2;     // KV heads
    const int HD = 256;     // head_dim
    const int TOKEN = argc > 2 ? atoi(argv[2]) : 760;  // "The" = 760 in Qwen3.5 tokenizer
    const float EPS = 1e-6f;
    const int FULL_ATTN[] = {3, 7, 11, 15, 19, 23};
    auto is_full_attn = [&](int l) {
        for (int f : FULL_ATTN) if (l == f) return true;
        return false;
    };

    // Embedding
    auto x_vec = gguf.dequant_row("token_embd.weight", TOKEN);
    std::vector<float> x(x_vec.begin(), x_vec.end());
    printf("\nEmbed norm: %.4f first5:", vec_norm(x.data(), N));
    for (int i = 0; i < 5; i++) printf(" %.6f", x[i]);
    printf("\n");

    // Preload output_norm and embd rows for early logits
    auto output_norm_w = gguf.dequant("output_norm.weight");
    auto embd_row0 = gguf.dequant_row("token_embd.weight", 0);
    auto embd_row11 = gguf.dequant_row("token_embd.weight", 11);

    // Buffers
    std::vector<float> x_norm(N), residual(N);
    std::vector<float> qkv(INNER*3), gate_z(INNER);
    std::vector<float> q(INNER), k(INNER), v(INNER);
    std::vector<float> attn_out(INNER), gated_out(INNER);
    std::vector<float> ffn_gate(FF), ffn_up(FF), ffn_out(FF);
    std::vector<float> out_proj(N);
    float gates[HEADS], betas[HEADS];

    // DeltaNet states (zero-initialized for first token)
    // (not needed for first token with zero state)

    // FullAttn: for pos=0, attention output = V (single token)
    std::vector<float> fa_q(N_HEAD*HD*2), fa_q_only(N_HEAD*HD), fa_gate(N_HEAD*HD);
    std::vector<float> fa_k(N_KV*HD), fa_v(N_KV*HD);

    for (int layer = 0; layer < 24; layer++) {
        auto lt0 = std::chrono::high_resolution_clock::now();
        std::string prefix = "blk." + std::to_string(layer) + ".";

        auto norm_w = gguf.dequant(prefix + "attn_norm.weight");
        rmsnorm(x_norm.data(), x.data(), norm_w.data(), N, EPS);
        memcpy(residual.data(), x.data(), N * sizeof(float));

        if (layer == 0) {
            printf("  [L0] x_norm: norm=%.4f first5:", vec_norm(x_norm.data(), N));
            for (int i = 0; i < 5; i++) printf(" %.6f", x_norm[i]);
            printf("\n");
        }

        if (!is_full_attn(layer)) {
            // ===== DeltaNet Layer =====
            auto W_qkv = gguf.dequant(prefix + "attn_qkv.weight");
            gemv(qkv.data(), W_qkv.data(), x_norm.data(), INNER*3, N);

            auto W_gate = gguf.dequant(prefix + "attn_gate.weight");
            gemv(gate_z.data(), W_gate.data(), x_norm.data(), INNER, N);

            if (layer == 0) {
                printf("  [L0] qkv_proj: norm=%.4f first5:", vec_norm(qkv.data(), INNER*3));
                for (int i = 0; i < 5; i++) printf(" %.6f", qkv[i]);
                printf("\n");
                printf("  [L0] gate_z: norm=%.4f first5:", vec_norm(gate_z.data(), INNER));
                for (int i = 0; i < 5; i++) printf(" %.6f", gate_z[i]);
                printf("\n");
            }

            // Conv1d (first token, state=0)
            auto conv_w = gguf.dequant(prefix + "ssm_conv1d.weight");
            // conv_w[channel*4+k] for channel c, kernel k
            for (int c = 0; c < INNER*3; c++) {
                qkv[c] = qkv[c] * conv_w[c * 4 + 3];
            }

            if (layer == 0) {
                printf("  [L0] after_conv: norm=%.4f first5:", vec_norm(qkv.data(), INNER*3));
                for (int i = 0; i < 5; i++) printf(" %.6f", qkv[i]);
                printf("\n");
            }

            // SiLU
            for (int i = 0; i < INNER*3; i++) qkv[i] = silu_f(qkv[i]);

            if (layer == 0) {
                printf("  [L0] after_silu: norm=%.4f first5:", vec_norm(qkv.data(), INNER*3));
                for (int i = 0; i < 5; i++) printf(" %.6f", qkv[i]);
                printf("\n");
            }

            // Split Q/K/V
            memcpy(q.data(), &qkv[0], INNER*sizeof(float));
            memcpy(k.data(), &qkv[INNER], INNER*sizeof(float));
            memcpy(v.data(), &qkv[2*INNER], INNER*sizeof(float));

            // L2 normalize Q/K per head
            for (int h = 0; h < HEADS; h++) {
                float qn = vec_norm(&q[h*DK], DK);
                float kn = vec_norm(&k[h*DK], DK);
                for (int i = 0; i < DK; i++) q[h*DK+i] /= fmaxf(qn, EPS);
                for (int i = 0; i < DK; i++) k[h*DK+i] /= fmaxf(kn, EPS);
            }

            if (layer == 0) {
                printf("  [L0] q_l2norm: first5:", vec_norm(q.data(), DK));
                for (int i = 0; i < 5; i++) printf(" %.6f", q[i]);
                printf("\n");
                printf("  [L0] v: norm=%.4f first5:", vec_norm(v.data(), INNER));
                for (int i = 0; i < 5; i++) printf(" %.6f", v[i]);
                printf("\n");
            }

            // Gate/beta
            auto ssm_a = gguf.dequant(prefix + "ssm_a");
            auto dt_bias = gguf.dequant(prefix + "ssm_dt.bias");
            auto alpha_w = gguf.dequant(prefix + "ssm_alpha.weight");
            auto beta_w = gguf.dequant(prefix + "ssm_beta.weight");

            for (int h = 0; h < HEADS; h++) {
                float alpha_proj = 0, beta_proj = 0;
                for (int c = 0; c < N; c++) {
                    alpha_proj += alpha_w[h*N+c] * x_norm[c];
                    beta_proj  += beta_w[h*N+c] * x_norm[c];
                }
                float sp = softplus_f(alpha_proj + dt_bias[h]);
                gates[h] = ssm_a[h] * sp;
                betas[h] = sigmoid_f(beta_proj);
            }

            if (layer == 0) {
                printf("  [L0] gates:");
                for (int h = 0; h < HEADS; h++) printf(" %.6f", gates[h]);
                printf("\n  [L0] betas:");
                for (int h = 0; h < HEADS; h++) printf(" %.6f", betas[h]);
                printf("\n");
            }

            // Scale Q by 1/sqrt(S_k) to match llama.cpp
            float q_scale = 1.0f / sqrtf((float)DK);
            for (int i = 0; i < INNER; i++) q[i] *= q_scale;

            // DeltaNet (state=0, first token)
            memset(attn_out.data(), 0, INNER*sizeof(float));
            for (int h = 0; h < HEADS; h++) {
                float dot_kq = 0;
                for (int i = 0; i < DK; i++) dot_kq += k[h*DK+i] * q[h*DK+i];
                for (int i = 0; i < DK; i++) {
                    attn_out[h*DK+i] = dot_kq * betas[h] * v[h*DK+i];
                }
            }

            if (layer == 0) {
                printf("  [L0] attn_out: norm=%.4f first5:", vec_norm(attn_out.data(), INNER));
                for (int i = 0; i < 5; i++) printf(" %.6f", attn_out[i]);
                printf("\n");
            }

            // Gated RMSNorm
            auto ssm_norm_w = gguf.dequant(prefix + "ssm_norm.weight");
            for (int h = 0; h < HEADS; h++) {
                float rms_sq = 0;
                for (int i = 0; i < DK; i++) rms_sq += attn_out[h*DK+i]*attn_out[h*DK+i];
                float rms = sqrtf(rms_sq/DK + EPS);
                for (int i = 0; i < DK; i++) {
                    float normed = attn_out[h*DK+i] / rms * ssm_norm_w[i];
                    float sg = silu_f(gate_z[h*DK+i]);
                    gated_out[h*DK+i] = normed * sg;
                }
            }

            if (layer == 0) {
                printf("  [L0] gated_out: norm=%.4f first5:", vec_norm(gated_out.data(), INNER));
                for (int i = 0; i < 5; i++) printf(" %.6f", gated_out[i]);
                printf("\n");
            }

            // Output projection
            auto W_out = gguf.dequant(prefix + "ssm_out.weight");
            gemv(x.data(), W_out.data(), gated_out.data(), N, INNER);

            if (layer == 0) {
                printf("  [L0] out_proj: norm=%.4f first5:", vec_norm(x.data(), N));
                for (int i = 0; i < 5; i++) printf(" %.6f", x[i]);
                printf("\n");
            }

            // Residual
            for (int i = 0; i < N; i++) x[i] += residual[i];

            if (layer == 0) {
                printf("  [L0] attn_residual: norm=%.4f first5:", vec_norm(x.data(), N));
                for (int i = 0; i < 5; i++) printf(" %.6f", x[i]);
                printf("\n");
            }
        } else {
            // ===== Full Attention Layer =====
            auto W_q = gguf.dequant(prefix + "attn_q.weight");
            gemv(fa_q.data(), W_q.data(), x_norm.data(), N_HEAD*HD*2, N);

            // Deinterleave Q+gate
            for (int h = 0; h < N_HEAD; h++) {
                for (int d = 0; d < HD; d++) {
                    fa_q_only[h*HD+d] = fa_q[h*HD*2+d];
                    fa_gate[h*HD+d] = fa_q[h*HD*2+HD+d];
                }
            }

            auto W_k = gguf.dequant(prefix + "attn_k.weight");
            gemv(fa_k.data(), W_k.data(), x_norm.data(), N_KV*HD, N);

            auto W_v = gguf.dequant(prefix + "attn_v.weight");
            gemv(fa_v.data(), W_v.data(), x_norm.data(), N_KV*HD, N);

            // Q/K RMSNorm
            auto q_norm_w = gguf.dequant(prefix + "attn_q_norm.weight");
            auto k_norm_w = gguf.dequant(prefix + "attn_k_norm.weight");
            for (int h = 0; h < N_HEAD; h++)
                rmsnorm(&fa_q_only[h*HD], &fa_q_only[h*HD], q_norm_w.data(), HD, EPS);
            for (int h = 0; h < N_KV; h++)
                rmsnorm(&fa_k[h*HD], &fa_k[h*HD], k_norm_w.data(), HD, EPS);

            // RoPE at pos=0 → identity (cos=1, sin=0), skip

            // Attention at pos=0, seq_len=1: output = V for each head (single-token attention)
            // GQA: Q heads 0-3→KV0, 4-7→KV1
            for (int qh = 0; qh < N_HEAD; qh++) {
                int kvh = qh / (N_HEAD / N_KV);
                for (int d = 0; d < HD; d++) {
                    attn_out[qh*HD+d] = fa_v[kvh*HD+d];
                }
            }

            // Gate: output = attn_out * sigmoid(gate)
            for (int i = 0; i < N_HEAD*HD; i++) {
                gated_out[i] = attn_out[i] * sigmoid_f(fa_gate[i]);
            }

            // Output projection
            auto W_out = gguf.dequant(prefix + "attn_output.weight");
            gemv(x.data(), W_out.data(), gated_out.data(), N, N_HEAD*HD);

            // Residual
            for (int i = 0; i < N; i++) x[i] += residual[i];
        }

        // Save residual for FFN
        memcpy(residual.data(), x.data(), N * sizeof(float));

        // Post-attention norm + FFN
        auto post_norm_w = gguf.dequant(prefix + "post_attention_norm.weight");
        rmsnorm(x_norm.data(), x.data(), post_norm_w.data(), N, EPS);

        auto W_ffn_gate = gguf.dequant(prefix + "ffn_gate.weight");
        gemv(ffn_gate.data(), W_ffn_gate.data(), x_norm.data(), FF, N);

        auto W_ffn_up = gguf.dequant(prefix + "ffn_up.weight");
        gemv(ffn_up.data(), W_ffn_up.data(), x_norm.data(), FF, N);

        // SwiGLU: silu(gate) * up
        for (int i = 0; i < FF; i++) ffn_out[i] = silu_f(ffn_gate[i]) * ffn_up[i];

        auto W_ffn_down = gguf.dequant(prefix + "ffn_down.weight");
        gemv(x.data(), W_ffn_down.data(), ffn_out.data(), N, FF);

        // Residual
        for (int i = 0; i < N; i++) x[i] += residual[i];

        if (layer == 0) {
            printf("  [L0] post_ffn: norm=%.4f first5:", vec_norm(x.data(), N));
            for (int i = 0; i < 5; i++) printf(" %.6f", x[i]);
            printf("\n");
        }

        // Early logit probe
        std::vector<float> xn(N);
        rmsnorm(xn.data(), x.data(), output_norm_w.data(), N, EPS);
        float l0 = 0, l11 = 0;
        for (int c = 0; c < N; c++) { l0 += embd_row0[c]*xn[c]; l11 += embd_row11[c]*xn[c]; }

        auto lt1 = std::chrono::high_resolution_clock::now();
        printf("[F32] Layer %2d (%s): norm=%.4f, logit[0]=%.2f, logit[11]=%.2f (%.1fs)\n",
               layer, is_full_attn(layer) ? "FullAttn" : "DeltaNet",
               vec_norm(x.data(), N), l0, l11,
               std::chrono::duration<double>(lt1-lt0).count());
    }

    // Final norm + logits
    printf("\n=== Final logits ===\n");
    std::vector<float> xn(N);
    rmsnorm(xn.data(), x.data(), output_norm_w.data(), N, EPS);
    print10("x_norm", xn.data());

    // Compute full logits and dump to file
    const int VOCAB = 248320;
    std::vector<float> logits(VOCAB);
    auto& embd_info = gguf.tensors.at("token_embd.weight");
    for (int tok = 0; tok < VOCAB; tok++) {
        auto row = gguf.dequant_row("token_embd.weight", tok);
        float logit = 0;
        for (int c = 0; c < N; c++) logit += row[c] * xn[c];
        logits[tok] = logit;
    }

    // Dump to file
    FILE* fp = fopen("/tmp/f32_forward_logits.bin", "wb");
    if (fp) {
        int nv = VOCAB;
        fwrite(&nv, sizeof(int), 1, fp);
        fwrite(logits.data(), sizeof(float), VOCAB, fp);
        fclose(fp);
        printf("Dumped %d logits to /tmp/f32_forward_logits.bin\n", VOCAB);
    }

    // Find and print top-10
    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < VOCAB; i++) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    printf("\nTop-10 logits (f32_forward):\n");
    for (int i = 0; i < 10; i++) {
        printf("  token=%d logit=%.4f\n", scored[i].second, scored[i].first);
    }
    printf("\nLogit[11] = %.4f\n", logits[11]);

    free((void*)gguf.mmap_data);
    fclose(gguf.fp);
    return 0;
}
