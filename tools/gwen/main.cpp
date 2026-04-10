// gwen — CLI for Qwen3.5-0.8B inference via gwen_lib
//
// Usage:
//   gwen "What is the capital of France?"
//   gwen -s "Clean up this transcribed speech." "um so like the thing is..."
//   gwen -n 200 --greedy "1 + 1 ="
//   gwen --agent "Who won the 2026 NCAA championship?"
//   gwen --agent "Play Bohemian Rhapsody by Queen"

#include "gwen.h"
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
        auto result = ctx.agent(question, n_predict);
        if (!result.tool_used.empty())
            fprintf(stderr, "[agent] tool: %s\n", result.tool_used.c_str());
        printf("%s\n", result.text.c_str());
        return 0;
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
