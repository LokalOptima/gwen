// Profile a single forward pass with CUDA events
#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/tokenizer.h"
#include "gwen/kernels.h"
#include <cstdio>
#include <cmath>

using namespace gwen;

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-0.8B-Q4_K_M.gguf";
    int n_warmup = 10;
    int n_measure = 50;

    printf("=== GWEN Forward Pass Profiler ===\n\n");

    auto model = Model::load(model_path);
    CudaAllocator allocator;
    model->upload_weights(allocator);
    auto tokenizer = Tokenizer::from_gguf(*model->gguf);

    InferenceState state;
    state.allocate(model->config, allocator);

    printf("Model loaded. VRAM: %.1f MB\n\n", allocator.total_allocated() / 1024.0 / 1024.0);

    // Seed with prompt token
    state.forward(*model, 760);  // "The"

    // Warmup
    int tok = 321;  // " and"
    for (int i = 0; i < n_warmup; i++) {
        tok = state.forward(*model, tok);
    }
    cudaDeviceSynchronize();

    // Measure full forward pass
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    float total_ms = 0, min_ms = 1e9f, max_ms = 0;

    for (int i = 0; i < n_measure; i++) {
        cudaEventRecord(t0);
        tok = state.forward(*model, tok);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        total_ms += ms;
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
    }

    float avg_ms = total_ms / n_measure;

    printf("--- Full Forward Pass ---\n");
    printf("  Average: %.3f ms (%.1f tok/s)\n", avg_ms, 1000.0f / avg_ms);
    printf("  Min:     %.3f ms (%.1f tok/s)\n", min_ms, 1000.0f / min_ms);
    printf("  Max:     %.3f ms (%.1f tok/s)\n", max_ms, 1000.0f / max_ms);

    // --- Component profiling ---
    const auto& cfg = model->config;
    printf("\n--- Component Timing (avg of 100 runs) ---\n");

    auto bench = [&](const char* name, int reps, auto fn) {
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < reps; i++) fn();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("  %-25s %.4f ms\n", name, ms / reps);
        return ms / reps;
    };

    // Embedding
    bench("Embedding lookup:", 1000, [&]() {
        gwen_embed_lookup(model->token_embd.device_data, model->token_embd.type,
                          321, state.x, cfg.n_embed);
    });

    // RMSNorm (1024 dim)
    bench("RMSNorm (1024):", 1000, [&]() {
        gwen_rmsnorm_f32w(state.x, static_cast<const float*>(model->output_norm.device_data),
                          state.x_norm, cfg.n_embed, cfg.rms_norm_eps);
    });

    // QKV GEMV (1024 → 6144, Q5_K)
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = (double)w.attn_qkv.shape[0] * w.attn_qkv.shape[1] / 256 * 146.0;
        float ms = bench("QKV GEMV (1024→6144):", 100, [&]() {
            gwen_gemv(w.attn_qkv.device_data, state.x_norm, state.qkv,
                      w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // Gate GEMV (1024 → 2048, Q5_K)
    {
        const auto& w = model->layers[0].deltanet;
        bench("Gate GEMV (1024→2048):", 100, [&]() {
            gwen_gemv(w.attn_gate.device_data, state.x_norm, state.gate_z,
                      w.attn_gate.shape[1], w.attn_gate.shape[0], w.attn_gate.type);
        });
    }

    // FFN gate+up GEMVs (1024 → 3584, Q4_K)
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = (double)w.ffn_gate.shape[0] * w.ffn_gate.shape[1] / 256 * 146.0 * 2; // two GEMVs
        float ms = bench("FFN gate+up (2x GEMV):", 100, [&]() {
            gwen_gemv(w.ffn_gate.device_data, state.x_norm, state.ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
            gwen_gemv(w.ffn_up.device_data, state.x_norm, state.ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
        });
        printf("    → %.1f GB/s total\n", bytes / ms / 1e6);
    }

    // FFN down GEMV (3584 → 1024, Q4_K)
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = (double)w.ffn_down.shape[0] * w.ffn_down.shape[1] / 256 * 146.0;
        float ms = bench("FFN down GEMV:", 100, [&]() {
            gwen_gemv(w.ffn_down.device_data, state.ffn_out, state.x,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // SSM output GEMV (2048 → 1024, Q5_K)
    {
        const auto& w = model->layers[0].deltanet;
        bench("SSM out GEMV (2048→1024):", 100, [&]() {
            gwen_gemv(w.ssm_out.device_data, state.gated_out, state.x,
                      w.ssm_out.shape[1], w.ssm_out.shape[0], w.ssm_out.type);
        });
    }

    // LM head GEMV (1024 → 248320, Q6_K)
    {
        double bytes = (double)cfg.n_vocab * cfg.n_embed / 256 * 210.0;
        float ms = bench("LM head GEMV:", 20, [&]() {
            gwen_gemv(model->token_embd.device_data, state.x_norm, state.logits_h,
                      cfg.n_vocab, cfg.n_embed, model->token_embd.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // DeltaNet recurrence — timing estimated from full forward minus other components

    // SwiGLU
    bench("SwiGLU (3584):", 1000, [&]() {
        gwen_swiglu(state.ffn_gate, state.ffn_up, state.ffn_out, cfg.n_ff);
    });

    // Add inplace
    bench("Add inplace (1024):", 1000, [&]() {
        gwen_add_inplace(state.x, state.residual, cfg.n_embed);
    });

    // SiLU inplace
    bench("SiLU inplace (6144):", 1000, [&]() {
        gwen_silu_inplace(state.qkv, cfg.ssm_inner_size * 3);
    });

    // L2 normalize
    bench("L2 normalize (16x128):", 1000, [&]() {
        gwen_l2_normalize(state.qkv, state.qkv, cfg.ssm_n_heads, cfg.ssm_state_size);
    });

    // Theoretical
    double weight_bytes = 497.4 * 1024 * 1024;
    double theoretical_ms = weight_bytes / (896.0 * 1e9) * 1000.0;
    printf("\n--- Summary ---\n");
    printf("  Theoretical min:   %.3f ms (%.0f tok/s) at 896 GB/s\n", theoretical_ms, 1000.0 / theoretical_ms);
    printf("  Actual average:    %.3f ms (%.0f tok/s)\n", avg_ms, 1000.0 / avg_ms);
    printf("  Bandwidth efficiency: %.1f%%\n", theoretical_ms / avg_ms * 100);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
