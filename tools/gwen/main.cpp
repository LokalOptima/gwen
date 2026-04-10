// gwen — CLI for Qwen3.5-0.8B inference via gwen_lib
//
// Usage:
//   gwen "What is the capital of France?"
//   gwen -s "Clean up this transcribed speech." "um so like the thing is..."
//   gwen -n 200 --greedy "1 + 1 ="
//   gwen --agent "Who won the 2026 NCAA championship?"
//   gwen --agent "Play Bohemian Rhapsody by Queen"

#include "gwen.h"
#include "brave_search.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const char * DEFAULT_SYSTEM = "You are a helpful assistant. No emojis. Short answers.";

static std::string find_model() {
    const char * env = getenv("GWEN_MODEL");
    if (env && env[0]) return env;

    const char * home = getenv("HOME");
    if (home) return std::string(home) + "/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf";

    return "";
}

// ---------------------------------------------------------------------------
// Agent mode: dispatcher + tool execution
// ---------------------------------------------------------------------------

static const char * DISPATCH_SYSTEM =
    "You are a voice assistant router. Route the user's request to the correct tool. "
    "Do not answer questions directly — always use a tool.";

static const char * DISPATCH_TOOLS =
    R"({"type":"function","function":{"name":"brave_search","description":"Search the web for information, facts, current events, weather, or any question","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"]}}}
{"type":"function","function":{"name":"spotify_play","description":"Play a song or music on Spotify","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Song and/or artist to play"}},"required":["query"]}}})";

static const char * ANSWER_SYSTEM =
    "You are a voice assistant. Answer the user's question using ONLY the search results below. "
    "Respond in plain spoken English — one or two sentences, no markdown, no bullet points, "
    "no URLs. Be concise and natural, as if speaking aloud.";

static std::string extract_query(const std::string & args_json) {
    try {
        return nlohmann::json::parse(args_json).at("query").get<std::string>();
    } catch (...) {
        return "";
    }
}

static int run_agent(gwen::Context & ctx, const std::string & input, int n_predict) {
    // Step 1: dispatch — classify input via tool calling
    fprintf(stderr, "[agent] dispatching: %s\n", input.c_str());
    auto dispatch = ctx.chat(DISPATCH_SYSTEM, input, DISPATCH_TOOLS, 200, true);

    if (!dispatch.has_tool_call()) {
        fprintf(stderr, "[agent] no tool call detected, unknown request\n");
        printf("Sorry, I'm not sure how to help with that.\n");
        return 0;
    }

    const auto & tc = dispatch.tool_call;
    fprintf(stderr, "[agent] tool: %s, args: %s\n", tc.name.c_str(), tc.arguments.c_str());

    if (tc.name == "brave_search") {
        std::string query = extract_query(tc.arguments);
        if (query.empty()) {
            printf("Sorry, I couldn't understand your question.\n");
            return 1;
        }

        // Step 2: search Brave
        fprintf(stderr, "[agent] searching: %s\n", query.c_str());
        auto results = brave::search(query);
        if (results.empty()) {
            printf("Sorry, I couldn't find anything about that.\n");
            return 1;
        }
        std::string context = brave::format_results(results);
        fprintf(stderr, "[agent] got %zu results, extracting answer...\n", results.size());

        // Step 3: extract answer from search results
        std::string user_msg = "Question: " + input + "\n\nSearch results:\n" + context;
        ctx.chat(ANSWER_SYSTEM, user_msg, "", n_predict, true,
            [](const char * piece, int) -> bool {
                printf("%s", piece);
                fflush(stdout);
                return true;
            });
        printf("\n");

    } else if (tc.name == "spotify_play") {
        std::string query = extract_query(tc.arguments);
        if (query.empty()) {
            printf("Sorry, I couldn't understand what to play.\n");
            return 1;
        }
        printf("Now playing: %s\n", query.c_str());

    } else {
        fprintf(stderr, "[agent] unknown tool: %s\n", tc.name.c_str());
        printf("Sorry, I'm not sure how to help with that.\n");
    }

    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage() {
    fprintf(stderr,
        "Usage: gwen [-m model.gguf] [-s system] [-n tokens] [--greedy] \"input\"\n"
        "       gwen --agent [-m model.gguf] [-n tokens] \"input\"\n");
}

int main(int argc, char ** argv) {
    int n_predict = 500;
    bool greedy = false;
    bool agent_mode = false;
    std::string model_path;
    std::string system_prompt = DEFAULT_SYSTEM;
    std::string question;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            greedy = true;
        } else if (strcmp(argv[i], "--agent") == 0) {
            agent_mode = true;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (argv[i][0] != '-') {
            question = argv[i];
        } else {
            print_usage();
            return 1;
        }
    }

    if (question.empty()) {
        print_usage();
        return 1;
    }

    if (model_path.empty()) model_path = find_model();
    if (model_path.empty()) {
        fprintf(stderr, "No model found. Set GWEN_MODEL or run: make download-models\n");
        return 1;
    }

    gwen::Context ctx;
    if (!ctx.init(model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    if (agent_mode) {
        return run_agent(ctx, question, n_predict);
    }

    // Direct chat mode
    ctx.chat(system_prompt, question, "", n_predict, greedy,
        [](const char * piece, int) -> bool {
            printf("%s", piece);
            fflush(stdout);
            return true;
        });
    printf("\n");
    return 0;
}
