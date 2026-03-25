// Profile a single forward pass with CUDA events
#include "gwen/model.h"
#include "gwen/inference.h"
#include "gwen/tokenizer.h"
#include "gwen/kernels.h"
#include <cstdio>
#include <cmath>

using namespace gwen;

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "Qwen3.5-9B-UD-Q4_K_XL.gguf";
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
                          state.d_token_id, state.x, cfg.n_embed);
    });

    // RMSNorm (1024 dim)
    bench("RMSNorm (1024):", 1000, [&]() {
        gwen_rmsnorm_f32w(state.x, static_cast<const float*>(model->output_norm.device_data),
                          state.x_norm, cfg.n_embed, cfg.rms_norm_eps);
    });

    // Helper: compute weight bytes for a given type and element count
    auto weight_bytes = [](int nelem, GGMLType type) -> double {
        int blk_size = 256;
        double bytes_per_blk;
        switch (type) {
            case GGMLType::Q4_K: bytes_per_blk = 146.0; break;
            case GGMLType::Q5_K: bytes_per_blk = 146.0; break; // actually 176 for Q5_K
            case GGMLType::Q6_K: bytes_per_blk = 210.0; break;
            case GGMLType::Q8_0: bytes_per_blk = 34.0; blk_size = 32; break;
            default: bytes_per_blk = 146.0; break;
        }
        return (double)nelem / blk_size * bytes_per_blk;
    };

    // QKV GEMV (1024 → 6144, Q5_K) — dp4a path
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = weight_bytes(w.attn_qkv.shape[0] * w.attn_qkv.shape[1], w.attn_qkv.type);
        float ms = bench("QKV GEMV dp4a:", 100, [&]() {
            gwen_quantize_q8_1(state.x_norm, state.x_q8_a, w.attn_qkv.shape[0]);
            gwen_gemv_dp4a(w.attn_qkv.device_data, state.x_q8_a, state.qkv,
                      w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // Gate GEMV (1024 → 2048, Q5_K) — dp4a, reuses x_q8_a
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = weight_bytes(w.attn_gate.shape[0] * w.attn_gate.shape[1], w.attn_gate.type);
        float ms = bench("Gate GEMV dp4a:", 100, [&]() {
            gwen_gemv_dp4a(w.attn_gate.device_data, state.x_q8_a, state.gate_z,
                      w.attn_gate.shape[1], w.attn_gate.shape[0], w.attn_gate.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // FFN gate+up GEMVs (1024 → 3584, Q4_K) — dp4a
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = weight_bytes(w.ffn_gate.shape[0] * w.ffn_gate.shape[1], w.ffn_gate.type) * 2;
        float ms = bench("FFN gate+up dp4a:", 100, [&]() {
            gwen_quantize_q8_1(state.x_norm, state.x_q8_a, w.ffn_gate.shape[0]);
            gwen_gemv_dp4a(w.ffn_gate.device_data, state.x_q8_a, state.ffn_gate,
                      w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
            gwen_gemv_dp4a(w.ffn_up.device_data, state.x_q8_a, state.ffn_up,
                      w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
        });
        printf("    → %.1f GB/s total\n", bytes / ms / 1e6);
    }

    // FFN down GEMV (3584 → 1024, Q4_K) — dp4a
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = weight_bytes(w.ffn_down.shape[0] * w.ffn_down.shape[1], w.ffn_down.type);
        float ms = bench("FFN down dp4a:", 100, [&]() {
            gwen_quantize_q8_1(state.ffn_out, state.x_q8_b, w.ffn_down.shape[0]);
            gwen_gemv_dp4a(w.ffn_down.device_data, state.x_q8_b, state.x,
                      w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // SSM output GEMV (2048 → 1024, Q5_K) — dp4a
    {
        const auto& w = model->layers[0].deltanet;
        double bytes = weight_bytes(w.ssm_out.shape[0] * w.ssm_out.shape[1], w.ssm_out.type);
        float ms = bench("SSM out dp4a:", 100, [&]() {
            gwen_quantize_q8_1(state.gated_out, state.x_q8_b, w.ssm_out.shape[0]);
            gwen_gemv_dp4a(w.ssm_out.device_data, state.x_q8_b, state.x,
                      w.ssm_out.shape[1], w.ssm_out.shape[0], w.ssm_out.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // LM head GEMV (1024 → 248320, Q6_K) — dp4a
    {
        double bytes = weight_bytes(cfg.n_vocab * cfg.n_embed, model->token_embd.type);
        float ms = bench("LM head dp4a:", 20, [&]() {
            gwen_quantize_q8_1(state.x_norm, state.x_q8_a, cfg.n_embed);
            gwen_gemv_dp4a(model->token_embd.device_data, state.x_q8_a, state.logits_h,
                      cfg.n_vocab, cfg.n_embed, model->token_embd.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // Also benchmark legacy LM head for comparison
    {
        double bytes = weight_bytes(cfg.n_vocab * cfg.n_embed, model->token_embd.type);
        float ms = bench("LM head legacy:", 20, [&]() {
            gwen_gemv(model->token_embd.device_data, state.x_norm, state.logits_h,
                      cfg.n_vocab, cfg.n_embed, model->token_embd.type);
        });
        printf("    → %.1f GB/s (%.1f%% of 896 GB/s)\n", bytes / ms / 1e6, bytes / ms / 1e6 / 896 * 100);
    }

    // --- All-layers GEMV-only timing (to measure non-GEMV overhead) ---
    {
        float ms = bench("All 24 layers GEMV:", 20, [&]() {
            for (uint32_t li = 0; li < cfg.n_layers; li++) {
                const auto& layer = model->layers[li];
                if (!layer.is_full_attention) {
                    const auto& w = layer.deltanet;
                    gwen_quantize_q8_1(state.x_norm, state.x_q8_a, cfg.n_embed);
                    gwen_gemv_dp4a(w.attn_qkv.device_data, state.x_q8_a, state.qkv,
                              w.attn_qkv.shape[1], w.attn_qkv.shape[0], w.attn_qkv.type);
                    gwen_gemv_dp4a(w.attn_gate.device_data, state.x_q8_a, state.gate_z,
                              w.attn_gate.shape[1], w.attn_gate.shape[0], w.attn_gate.type);
                    gwen_quantize_q8_1(state.gated_out, state.x_q8_b, cfg.ssm_inner_size);
                    gwen_gemv_dp4a(w.ssm_out.device_data, state.x_q8_b, state.x,
                              w.ssm_out.shape[1], w.ssm_out.shape[0], w.ssm_out.type);
                    gwen_quantize_q8_1(state.x_norm, state.x_q8_a, cfg.n_embed);
                    gwen_gemv_dp4a(w.ffn_gate.device_data, state.x_q8_a, state.ffn_gate,
                              w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
                    gwen_gemv_dp4a(w.ffn_up.device_data, state.x_q8_a, state.ffn_up,
                              w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
                    gwen_quantize_q8_1(state.ffn_out, state.x_q8_b, cfg.n_ff);
                    gwen_gemv_dp4a(w.ffn_down.device_data, state.x_q8_b, state.x,
                              w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);
                } else {
                    const auto& w = layer.full_attn;
                    gwen_quantize_q8_1(state.x_norm, state.x_q8_a, cfg.n_embed);
                    gwen_gemv_dp4a(w.attn_q.device_data, state.x_q8_a, state.qkv,
                              w.attn_q.shape[1], w.attn_q.shape[0], w.attn_q.type);
                    gwen_gemv_dp4a(w.attn_k.device_data, state.x_q8_a, state.fa_k,
                              w.attn_k.shape[1], w.attn_k.shape[0], w.attn_k.type);
                    gwen_gemv_dp4a(w.attn_v.device_data, state.x_q8_a, state.fa_v,
                              w.attn_v.shape[1], w.attn_v.shape[0], w.attn_v.type);
                    gwen_quantize_q8_1(state.gated_out, state.x_q8_b, cfg.n_head * cfg.head_dim);
                    gwen_gemv_dp4a(w.attn_output.device_data, state.x_q8_b, state.x,
                              w.attn_output.shape[1], w.attn_output.shape[0], w.attn_output.type);
                    gwen_quantize_q8_1(state.x_norm, state.x_q8_a, cfg.n_embed);
                    gwen_gemv_dp4a(w.ffn_gate.device_data, state.x_q8_a, state.ffn_gate,
                              w.ffn_gate.shape[1], w.ffn_gate.shape[0], w.ffn_gate.type);
                    gwen_gemv_dp4a(w.ffn_up.device_data, state.x_q8_a, state.ffn_up,
                              w.ffn_up.shape[1], w.ffn_up.shape[0], w.ffn_up.type);
                    gwen_quantize_q8_1(state.ffn_out, state.x_q8_b, cfg.n_ff);
                    gwen_gemv_dp4a(w.ffn_down.device_data, state.x_q8_b, state.x,
                              w.ffn_down.shape[1], w.ffn_down.shape[0], w.ffn_down.type);
                }
            }
            // LM head
            gwen_quantize_q8_1(state.x_norm, state.x_q8_a, cfg.n_embed);
            gwen_gemv_dp4a(model->token_embd.device_data, state.x_q8_a, state.logits_h,
                      cfg.n_vocab, cfg.n_embed, model->token_embd.type);
        });
        printf("    Non-GEMV overhead: %.3f ms (%.1f%%)\n", avg_ms - ms, (avg_ms - ms) / avg_ms * 100);
    }

    // DeltaNet recurrence (18 calls × this)
    {
        auto& dn = state.deltanet_states[0];
        int q_width = cfg.ssm_n_k_heads * cfg.ssm_state_size;
        float ms = bench("DeltaNet recurrence:", 100, [&]() {
            gwen_deltanet_decode(dn.S, state.qkv, state.qkv + q_width,
                                 state.qkv + 2 * q_width,
                                 state.d_alpha, state.d_beta, state.attn_out,
                                 cfg.ssm_n_v_heads, cfg.ssm_state_size, cfg.ssm_state_size);
        });
        int n_dn = cfg.n_layers - cfg.n_layers / cfg.full_attn_interval;
        printf("    × %d layers = %.3f ms\n", n_dn, ms * n_dn);
    }

    // SwiGLU
    bench("SwiGLU:", 1000, [&]() {
        gwen_swiglu(state.ffn_gate, state.ffn_up, state.ffn_out, cfg.n_ff);
    });

    // Add inplace
    bench("Add inplace:", 1000, [&]() {
        gwen_add_inplace(state.x, state.residual, cfg.n_embed);
    });

    // SiLU inplace
    bench("SiLU inplace:", 1000, [&]() {
        gwen_silu_inplace(state.qkv, cfg.ssm_qkv_dim());
    });

    // L2 normalize
    bench("L2 normalize:", 1000, [&]() {
        gwen_l2_normalize(state.qkv, state.qkv, cfg.ssm_n_v_heads, cfg.ssm_state_size);
    });

    // Theoretical
    double total_weight_bytes = 497.4 * 1024 * 1024;
    double theoretical_ms = total_weight_bytes / (896.0 * 1e9) * 1000.0;
    printf("\n--- Summary ---\n");
    printf("  Theoretical min:   %.3f ms (%.0f tok/s) at 896 GB/s\n", theoretical_ms, 1000.0 / theoretical_ms);
    printf("  Actual average:    %.3f ms (%.0f tok/s)\n", avg_ms, 1000.0 / avg_ms);
    printf("  Bandwidth efficiency: %.1f%%\n", theoretical_ms / avg_ms * 100);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
