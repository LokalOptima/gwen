// gwen — simple CLI for Qwen3.5-0.8B inference via gwen_lib
//
// Usage:
//   gwen "What is the capital of France?"
//   gwen -n 200 "Write a haiku about CUDA"
//   gwen --greedy "1 + 1 ="

#include "gwen.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static std::string find_model() {
    // Check env var first
    const char * env = getenv("GWEN_MODEL");
    if (env && env[0]) return env;

    // Default: ~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf
    const char * home = getenv("HOME");
    if (home) return std::string(home) + "/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf";

    return "";
}

static std::string format_prompt(const std::string & question) {
    return "<|im_start|>system\nYou are a helpful assistant. No emojis. Short answers.<|im_end|>\n"
           "<|im_start|>user\n" + question + "<|im_end|>\n"
           "<|im_start|>assistant\n<think>\n</think>\n\n";
}

int main(int argc, char ** argv) {
    int n_predict = 500;
    bool greedy = false;
    std::string model_path;
    std::string question;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            greedy = true;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argv[i][0] != '-') {
            question = argv[i];
        } else {
            fprintf(stderr, "Usage: gwen [-m model.gguf] [-n tokens] [--greedy] \"question\"\n");
            return 1;
        }
    }

    if (question.empty()) {
        fprintf(stderr, "Usage: gwen [-m model.gguf] [-n tokens] [--greedy] \"question\"\n");
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

    std::string prompt = format_prompt(question);

    ctx.generate(prompt, n_predict, greedy,
        [](const char * piece, int) -> bool {
            printf("%s", piece);
            fflush(stdout);
            return true;
        });

    printf("\n");
    return 0;
}
