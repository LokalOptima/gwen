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

struct ToolCall {
    std::string name;       // e.g. "brave_search", "spotify_play"
    std::string arguments;  // raw JSON object string
};

struct ChatResult {
    std::string text;       // raw model output (after thinking)
    ToolCall    tool_call;  // populated if model emitted <tool_call>

    bool has_tool_call() const { return !tool_call.name.empty(); }
};

struct Context {
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Load model + MTP sidecar (auto-discovered from *-mtp.gguf next to model).
    // Returns false on failure.
    bool init(const std::string& model_path);

    // Generate text from raw prompt string.
    // greedy=true: deterministic argmax, no penalties (fast path).
    // greedy=false: stochastic with Qwen3.5 recommended params.
    // on_token: optional callback per token, return false to stop early.
    // Returns the full generated text.
    std::string generate(const std::string& prompt, int n_predict = 100,
                         bool greedy = true, TokenCallback on_token = nullptr);

    // Chat with ChatML formatting. Handles system/user roles and optional
    // Qwen3 tool definitions. Parses <tool_call> from output if present.
    // tools_json: tool definitions (JSON objects, one per line) or empty.
    ChatResult chat(const std::string& system, const std::string& user,
                    const std::string& tools_json = "",
                    int n_predict = 500, bool greedy = true,
                    TokenCallback on_token = nullptr);

    // Stats from last generate()/chat() call.
    Stats last_stats() const;

    void destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace gwen
