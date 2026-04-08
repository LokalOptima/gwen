// gwen.h — Public API for gwen LLM inference library
//
// Usage:
//   #include "gwen.h"
//   gwen::Context ctx;
//   ctx.init("~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf");
//   std::string text = ctx.generate("The meaning of life is", 100);

#pragma once

#include <functional>
#include <memory>
#include <string>

namespace gwen {

// Called per generated token. Return false to stop generation.
using TokenCallback = std::function<bool(const char* text, int token_id)>;

struct Stats {
    int prompt_tokens  = 0;
    int decode_tokens  = 0;
    int mtp_accepted   = 0;
    int mtp_rejected   = 0;
    double prefill_ms  = 0;
    double decode_ms   = 0;
    double tok_per_s   = 0;
};

struct Context {
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Load model + MTP sidecar (auto-discovered from *-mtp.gguf next to model).
    // Returns false on failure.
    bool init(const std::string& model_path);

    // Generate text from prompt.
    // greedy=true: deterministic argmax, no penalties (fast path).
    // greedy=false: stochastic with Qwen3.5 recommended params.
    // on_token: optional callback per token, return false to stop early.
    // Returns the full generated text.
    std::string generate(const std::string& prompt, int n_predict = 100,
                         bool greedy = true, TokenCallback on_token = nullptr);

    // Stats from last generate() call.
    Stats last_stats() const;

    void destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace gwen
