// gwen.cpp — Implementation of gwen LLM inference library
//
// Wraps llama.cpp model loading + MTP speculative decode into a simple API.
// All sampling helpers and the generate_mtp loop are self-contained here.

#include "gwen.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Sampling helpers (GPU-side argmax, fast top-k, speculative acceptance)
// ---------------------------------------------------------------------------

namespace {

struct top_k_candidate { llama_token id; float logit; };

static inline void top_k_insert(top_k_candidate * top, int k, int idx, float val, float & threshold) {
    int lo = 0, hi = k - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (top[mid].logit > val) lo = mid + 1; else hi = mid;
    }
    memmove(&top[lo + 1], &top[lo], (k - 1 - lo) * sizeof(top_k_candidate));
    top[lo] = {(llama_token)idx, val};
    threshold = top[k - 1].logit;
}

static void top_k_scan_scalar(const float * logits, int n_vocab, int k,
                               top_k_candidate * top, float & threshold, int start = -1) {
    if (start < 0) start = k;
    for (int i = start; i < n_vocab; i++) {
        if (logits[i] > threshold) {
            top_k_insert(top, k, i, logits[i], threshold);
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>

__attribute__((target("avx2")))
static void top_k_scan_avx2(const float * logits, int n_vocab, int k,
                              top_k_candidate * top, float & threshold) {
    __m256 thresh_v = _mm256_set1_ps(threshold);
    const int n_simd_end = k + ((n_vocab - k) & ~7);
    for (int i = k; i < n_simd_end; i += 8) {
        __m256 v = _mm256_loadu_ps(&logits[i]);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(v, thresh_v, _CMP_GT_OS));
        if (!mask) continue;
        while (mask) {
            int bit = __builtin_ctz(mask);
            top_k_insert(top, k, i + bit, logits[i + bit], threshold);
            thresh_v = _mm256_set1_ps(threshold);
            mask &= mask - 1;
        }
    }
    top_k_scan_scalar(logits, n_vocab, k, top, threshold, n_simd_end);
}

static bool cpu_has_avx2() {
    static const bool v = __builtin_cpu_supports("avx2");
    return v;
}
#define HAS_AVX2_DISPATCH
#endif

static llama_token greedy_argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

static llama_token fast_sample(
        const float * logits, int n_vocab,
        const std::vector<llama_token> & output_tokens,
        const common_params_sampling & sparams,
        std::mt19937 & rng,
        llama_token probe_token = -1,
        float * probe_p = nullptr) {

    constexpr int K_WIDE = 64;
    const int top_k  = std::max(1, sparams.top_k);
    const float temp = sparams.temp;

    top_k_candidate top[K_WIDE];
    const int k_init = std::min(K_WIDE, n_vocab);

    for (int i = 0; i < k_init; i++) {
        top[i] = {(llama_token)i, logits[i]};
    }
    std::sort(top, top + k_init, [](const top_k_candidate & a, const top_k_candidate & b) {
        return a.logit > b.logit;
    });
    float threshold = top[k_init - 1].logit;

#ifdef HAS_AVX2_DISPATCH
    if (cpu_has_avx2()) {
        top_k_scan_avx2(logits, n_vocab, k_init, top, threshold);
    } else
#endif
    {
        top_k_scan_scalar(logits, n_vocab, k_init, top, threshold);
    }

    const bool has_penalty = (sparams.penalty_last_n != 0) &&
        (sparams.penalty_repeat != 1.0f || sparams.penalty_freq != 0.0f || sparams.penalty_present != 0.0f);

    if (has_penalty) {
        const int window = (sparams.penalty_last_n < 0)
            ? (int)output_tokens.size()
            : std::min((int)output_tokens.size(), sparams.penalty_last_n);
        std::unordered_map<llama_token, int> token_count;
        for (int j = (int)output_tokens.size() - window; j < (int)output_tokens.size(); j++) {
            token_count[output_tokens[j]]++;
        }

        for (int ci = 0; ci < k_init; ci++) {
            auto & c = top[ci];
            auto it = token_count.find(c.id);
            if (it == token_count.end()) continue;
            const int count = it->second;
            if (c.logit <= 0) {
                c.logit *= sparams.penalty_repeat;
            } else {
                c.logit /= sparams.penalty_repeat;
            }
            c.logit -= float(count) * sparams.penalty_freq + float(count > 0) * sparams.penalty_present;
        }
    }

    std::sort(top, top + k_init, [](const top_k_candidate & a, const top_k_candidate & b) { return a.logit > b.logit; });
    const int k = std::min(top_k, k_init);

    const float max_l = top[0].logit;
    float sum = 0.0f;
    float probs[K_WIDE];
    for (int i = 0; i < k; i++) {
        probs[i] = expf((top[i].logit - max_l) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < k; i++) {
        probs[i] /= sum;
    }

    if (probe_p) {
        *probe_p = 0.0f;
        if (probe_token >= 0) {
            for (int i = 0; i < k; i++) {
                if (top[i].id == probe_token) {
                    *probe_p = probs[i];
                    break;
                }
            }
        }
    }

    std::uniform_real_distribution<float> udist(0.0f, 1.0f);
    float r = udist(rng);
    float cum = 0.0f;
    for (int i = 0; i < k; i++) {
        cum += probs[i];
        if (r <= cum) return top[i].id;
    }
    return top[k - 1].id;
}

static inline float fast_expf(float x) {
    if (x < -87.0f) return 0.0f;
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x) + 1065353216;
    return v.f;
}

static float compute_mtp_q_draft(
        llama_context * ctx,
        llama_token draft,
        const std::vector<llama_token> & output_tokens,
        const common_params_sampling & sparams,
        const std::unordered_map<int32_t, int> & reverse_map) {

    int mtp_n = 0;
    const float * logits = llama_mtp_get_logits(ctx, &mtp_n);
    if (!logits || mtp_n == 0) return 1.0f;

    int draft_idx = -1;
    auto it_draft = reverse_map.find(draft);
    if (it_draft != reverse_map.end()) {
        draft_idx = it_draft->second;
    } else if (reverse_map.empty() && draft >= 0 && draft < mtp_n) {
        draft_idx = draft;
    }
    if (draft_idx < 0) return 1.0f;

    struct PenCorr { int idx; float raw; float pen; };
    thread_local std::vector<PenCorr> corrections;
    corrections.clear();

    const bool has_penalty = (sparams.penalty_last_n != 0) &&
        (sparams.penalty_repeat != 1.0f || sparams.penalty_freq != 0.0f || sparams.penalty_present != 0.0f);
    if (has_penalty) {
        const int window = (sparams.penalty_last_n < 0)
            ? (int)output_tokens.size()
            : std::min((int)output_tokens.size(), sparams.penalty_last_n);
        std::unordered_map<int, int> pen_count;
        for (int j = (int)output_tokens.size() - window; j < (int)output_tokens.size(); j++) {
            auto it = reverse_map.find(output_tokens[j]);
            if (it != reverse_map.end()) pen_count[it->second]++;
        }
        corrections.reserve(pen_count.size());
        for (auto & [idx, cnt] : pen_count) {
            float raw = logits[idx];
            float pen = raw;
            if (pen <= 0) pen *= sparams.penalty_repeat;
            else pen /= sparams.penalty_repeat;
            pen -= float(cnt) * sparams.penalty_freq + sparams.penalty_present;
            corrections.push_back({idx, raw, pen});
        }
    }

    float max_l = logits[0];
    float sum_exp = 1.0f;
    for (int j = 1; j < mtp_n; j++) {
        const float l = logits[j];
        if (l > max_l) {
            sum_exp *= fast_expf(max_l - l);
            sum_exp += 1.0f;
            max_l = l;
        } else {
            sum_exp += fast_expf(l - max_l);
        }
    }

    for (const auto & c : corrections) {
        sum_exp -= expf(c.raw - max_l);
        sum_exp += expf(c.pen - max_l);
    }

    float draft_logit = logits[draft_idx];
    for (const auto & c : corrections) {
        if (c.idx == draft_idx) { draft_logit = c.pen; break; }
    }

    return expf(draft_logit - max_l) / sum_exp;
}

// ---------------------------------------------------------------------------
// generate_mtp — core MTP speculative decode loop
// ---------------------------------------------------------------------------

// Emit a token: appends to output_tokens, converts to text, calls callback.
// Returns false if callback requests stop.
using EmitFn = std::function<bool(llama_token token)>;

static void generate_mtp(
        llama_context * ctx,
        const llama_vocab * vocab,
        common_sampler * smpl,
        const common_params_sampling & sparams,
        int & n_past,
        int & n_remain,
        int & mtp_accepted,
        int & mtp_rejected,
        std::vector<llama_token> & output_tokens,
        EmitFn emit,
        double & out_decode_ms) {

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int initial_output_size = (int)output_tokens.size();
    double mtp_draft_ms_total = 0.0;
    double main_decode_ms_total = 0.0;
    const bool greedy = (sparams.temp < 1e-6f);
    std::mt19937 rng(common_sampler_get_seed(smpl));
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    std::unordered_map<int32_t, int> mtp_reverse_map;
    if (!greedy) {
        llama_mtp_set_extract_logits(ctx, true);
        int map_n = 0;
        const int32_t * token_map = llama_mtp_get_token_map(ctx, &map_n);
        if (token_map && map_n > 0) {
            mtp_reverse_map.reserve(map_n);
            for (int i = 0; i < map_n; i++) {
                mtp_reverse_map[token_map[i]] = i;
            }
        }
    }

    // Sample first token from prefill logits
    llama_token accepted;
    if (greedy) {
        accepted = greedy_argmax(llama_get_logits(ctx), n_vocab);
    } else {
        llama_synchronize(ctx);
        accepted = fast_sample(llama_get_logits(ctx), n_vocab, output_tokens, sparams, rng);
    }
    common_sampler_accept(smpl, accepted, true);
    output_tokens.push_back(accepted);
    if (!emit(accepted)) { n_remain = 0; goto done; }
    --n_remain;

    if (llama_vocab_is_eog(vocab, accepted) || n_remain <= 0) goto done;

    {
        // Get first MTP draft
        llama_token draft = -1;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (llama_decode_mtp(ctx, accepted, n_past) == 0) {
                draft = llama_mtp_get_argmax(ctx);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            mtp_draft_ms_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        if (greedy) {
            llama_set_argmax_only(ctx, true);
        }

        auto t_decode_start = std::chrono::high_resolution_clock::now();

        while (n_remain > 0) {
            if (llama_vocab_is_eog(vocab, accepted)) break;

            if (draft < 0 || draft == accepted) {
                // No valid draft — single token decode
                llama_batch batch = llama_batch_init(1, 0, 1);
                common_batch_add(batch, accepted, n_past++, {0}, true);

                auto t0 = std::chrono::high_resolution_clock::now();
                llama_decode(ctx, batch);
                auto t1 = std::chrono::high_resolution_clock::now();
                main_decode_ms_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
                llama_batch_free(batch);

                if (greedy) {
                    accepted = llama_get_argmax_ith(ctx, 0);
                } else {
                    llama_synchronize(ctx);
                    accepted = fast_sample(llama_get_logits_ith(ctx, 0), n_vocab, output_tokens, sparams, rng);
                }
                common_sampler_accept(smpl, accepted, true);
                output_tokens.push_back(accepted);
                if (!emit(accepted)) { n_remain = 0; break; }
                --n_remain;

                if (llama_decode_mtp(ctx, accepted, n_past) == 0) {
                    draft = llama_mtp_get_argmax(ctx);
                } else {
                    draft = -1;
                }
                mtp_draft_ms_total += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t1).count();
                continue;
            }

            // --- Speculative path (2-token verify) ---
            llama_batch batch = llama_batch_init(2, 0, 1);
            common_batch_add(batch, accepted, n_past, {0}, true);
            common_batch_add(batch, draft, n_past + 1, {0}, true);

            auto t_dispatch0 = std::chrono::high_resolution_clock::now();
            llama_decode(ctx, batch);
            auto t_dispatch1 = std::chrono::high_resolution_clock::now();
            main_decode_ms_total += std::chrono::duration<double, std::milli>(t_dispatch1 - t_dispatch0).count();
            llama_batch_free(batch);

            bool do_accept;
            llama_token fallback = -1;

            if (greedy) {
                do_accept = (llama_get_argmax_ith(ctx, 0) == draft);
            } else {
                llama_synchronize(ctx);

                float p_draft = 0.0f;
                fallback = fast_sample(llama_get_logits_ith(ctx, 0), n_vocab, output_tokens, sparams, rng, draft, &p_draft);
                float q_draft = compute_mtp_q_draft(ctx, draft, output_tokens, sparams, mtp_reverse_map);
                float accept_prob = std::min(1.0f, p_draft / std::max(q_draft, 1e-10f));
                do_accept = (coin(rng) < accept_prob);
            }

            if (do_accept) {
                mtp_accepted++;
                n_past += 2;

                common_sampler_accept(smpl, accepted, true);
                common_sampler_accept(smpl, draft, true);
                output_tokens.push_back(draft);
                if (!emit(draft)) { n_remain = 0; break; }
                --n_remain;

                llama_token pred_after_draft;
                if (greedy) {
                    pred_after_draft = llama_get_argmax_ith(ctx, 1);
                } else {
                    pred_after_draft = fast_sample(llama_get_logits_ith(ctx, 1), n_vocab, output_tokens, sparams, rng);
                }

                if (n_remain > 0) {
                    common_sampler_accept(smpl, pred_after_draft, true);
                    output_tokens.push_back(pred_after_draft);
                    if (!emit(pred_after_draft)) { n_remain = 0; break; }
                    --n_remain;
                }

                accepted = pred_after_draft;
                auto td0 = std::chrono::high_resolution_clock::now();
                if (llama_decode_mtp(ctx, accepted, n_past) == 0) {
                    draft = llama_mtp_get_argmax(ctx);
                } else {
                    draft = -1;
                }
                mtp_draft_ms_total += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - td0).count();
            } else {
                mtp_rejected++;

                llama_mtp_reject_fixup(ctx, n_past);
                n_past++;

                accepted = greedy ? llama_get_argmax_ith(ctx, 0) : fallback;
                common_sampler_accept(smpl, accepted, true);
                output_tokens.push_back(accepted);
                if (!emit(accepted)) { n_remain = 0; break; }
                --n_remain;

                auto td0 = std::chrono::high_resolution_clock::now();
                if (llama_decode_mtp(ctx, accepted, n_past, 0) == 0) {
                    draft = llama_mtp_get_argmax(ctx);
                } else {
                    draft = -1;
                }
                mtp_draft_ms_total += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - td0).count();
            }
        }

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        out_decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    }

done:
    if (greedy) {
        llama_set_argmax_only(ctx, false);
    } else {
        llama_mtp_set_extract_logits(ctx, false);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// gwen::Context implementation
// ---------------------------------------------------------------------------

namespace gwen {

struct Context::Impl {
    common_params                params;
    common_init_result_ptr       init_result;
    llama_model                * model   = nullptr;
    llama_context              * ctx     = nullptr;
    common_sampler             * smpl    = nullptr;
    const llama_vocab          * vocab   = nullptr;
    Stats                        stats;

    ~Impl() { destroy(); }

    bool init(const std::string & model_path) {
        params.model.path = model_path;
        params.n_gpu_layers = 999;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // required for 2-token MTP verify
        params.fit_params = false;  // skip probe load — model always fits on 16GB GPU
        params.warmup = false;     // skip warmup — first real decode warms up anyway
        params.verbosity = -1;

        // Suppress all log output from llama.cpp and common
        llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
        common_log_set_verbosity_thold(-1);

        llama_backend_init();
        llama_numa_init(params.numa);

        init_result = common_init_from_params(params);

        if (!init_result) return false;

        model = init_result->model();
        ctx   = init_result->context();
        smpl  = init_result->sampler(0);

        if (!model || !ctx) return false;

        if (!llama_model_has_mtp(model)) {
            fprintf(stderr, "gwen::Context::init: model has no MTP sidecar\n");
            return false;
        }

        vocab = llama_model_get_vocab(model);
        return true;
    }

    std::string generate(const std::string & prompt, int n_predict, bool greedy,
                         TokenCallback on_token) {
        stats = {};

        // Configure sampling
        auto & sparams = params.sampling;
        if (greedy) {
            sparams.temp            = 0.0f;
            sparams.penalty_present = 0.0f;
            sparams.penalty_freq    = 0.0f;
            sparams.penalty_repeat  = 1.0f;
            sparams.penalty_last_n  = 0;
        }
        // else: use Qwen3.5 defaults from common.h

        // Tokenize
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        auto tokens = common_tokenize(ctx, prompt, add_bos, true);
        stats.prompt_tokens = (int)tokens.size();

        if (tokens.empty()) {
            tokens.push_back(llama_vocab_bos(vocab));
        }

        // Reset KV cache and sampler for clean generation
        llama_memory_clear(llama_get_memory(ctx), false);
        common_sampler_reset(smpl);
        llama_perf_context_reset(ctx);

        // Prefill
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        int n_past = 0;
        const int n_batch = params.n_batch > 0 ? params.n_batch : 512;

        for (int i = 0; i < (int)tokens.size(); i += n_batch) {
            int n = std::min(n_batch, (int)tokens.size() - i);
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; j++) {
                bool last = (i + j == (int)tokens.size() - 1);
                common_batch_add(batch, tokens[i + j], n_past + j, {0}, last);
            }
            llama_decode(ctx, batch);
            llama_batch_free(batch);
            n_past += n;

            // Feed tokens to sampler for penalty tracking
            for (int j = 0; j < n; j++) {
                common_sampler_accept(smpl, tokens[i + j], false);
            }
        }
        auto t_prefill_end = std::chrono::high_resolution_clock::now();
        stats.prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

        // Generate via MTP
        int n_remain = n_predict;
        std::ostringstream text_ss;
        std::vector<llama_token> output_tokens;

        auto emit = [&](llama_token token) -> bool {
            std::string piece = common_token_to_piece(ctx, token, false);
            text_ss << piece;
            if (on_token) {
                return on_token(piece.c_str(), token);
            }
            return true;
        };

        generate_mtp(ctx, vocab, smpl, sparams, n_past, n_remain,
                     stats.mtp_accepted, stats.mtp_rejected,
                     output_tokens, emit, stats.decode_ms);

        stats.decode_tokens = (int)output_tokens.size();
        stats.tok_per_s = stats.decode_ms > 0
            ? stats.decode_tokens * 1000.0 / stats.decode_ms : 0;

        return text_ss.str();
    }

    void destroy() {
        if (init_result) {
            init_result.reset();
            model = nullptr;
            ctx   = nullptr;
            smpl  = nullptr;
            vocab = nullptr;
            llama_backend_free();
        }
    }
};

Context::Context() : impl(std::make_unique<Impl>()) {}
Context::~Context() = default;

bool Context::init(const std::string & model_path) {
    return impl->init(model_path);
}

std::string Context::generate(const std::string & prompt, int n_predict,
                               bool greedy, TokenCallback on_token) {
    return impl->generate(prompt, n_predict, greedy, on_token);
}

Stats Context::last_stats() const {
    return impl->stats;
}

void Context::destroy() {
    impl->destroy();
}

} // namespace gwen
