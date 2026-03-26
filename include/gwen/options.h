#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <sstream>
#include <string>
#include <vector>

namespace gwen {

struct Options {
    std::string model_path;
    std::string template_path;
    int n_predict = -1;  // -1 = unlimited (until EOS)
    bool greedy = false;
    bool info_only = false;
    bool raw = false;       // skip ChatML wrapping
    bool reason = false;    // enable thinking (omit empty <think> block)
    bool debug = false;     // show thinking + full prompt
    std::string input_text;
    std::vector<int> teacher_tokens;

    // Apply template to input text, returning the final prompt.
    // Default: wraps in ChatML format for instruct models.
    // --template: loads file or inline string, replaces {input}.
    // --raw: no wrapping at all.
    std::string build_prompt() const {
        if (raw) return input_text;

        if (!template_path.empty()) {
            std::string tmpl;
            std::ifstream tf(template_path);
            if (tf.is_open()) {
                tmpl.assign(std::istreambuf_iterator<char>(tf),
                            std::istreambuf_iterator<char>());
            } else {
                tmpl = template_path;
            }
            auto pos = tmpl.find("{input}");
            if (pos != std::string::npos) {
                tmpl.replace(pos, 7, input_text);
            } else {
                tmpl += input_text;
            }
            return tmpl;
        }

        // Default: ChatML wrapping
        std::string prompt = "<|im_start|>user\n" + input_text + "<|im_end|>\n<|im_start|>assistant\n";
        if (!reason) prompt += "<think>\n\n</think>\n\n";
        return prompt;
    }

    static void print_usage(const char* prog) {
        fprintf(stderr, "Usage: %s [options] [model.gguf] [input text...]\n", prog);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --model PATH         Path to GGUF model file\n");
        fprintf(stderr, "  --template STR/FILE  Template (file or string); {input} replaced with input text\n");
        fprintf(stderr, "  --max-predict N      Number of tokens to generate (default: 50)\n");
        fprintf(stderr, "  --greedy             Use greedy decoding (default: temp=0.7, top_k=20, top_p=0.8)\n");
        fprintf(stderr, "  --raw                Send input text as-is (no ChatML wrapping)\n");
        fprintf(stderr, "  --reason             Enable thinking (disabled by default)\n");
        fprintf(stderr, "  --debug              Show prompt, token IDs, and thinking\n");
        fprintf(stderr, "  --teacher-tokens T   Comma-separated reference token IDs for teacher-forced comparison\n");
        fprintf(stderr, "  --info               Print model info and exit\n");
        fprintf(stderr, "  --help               Show this help\n");
    }

    static Options parse(int argc, char** argv) {
        Options opts;
        std::string teacher_str;

        static struct option long_options[] = {
            {"model",          required_argument, nullptr, 'm'},
            {"template",       required_argument, nullptr, 'p'},
            {"max-predict",    required_argument, nullptr, 'n'},
            {"greedy",         no_argument,       nullptr, 'g'},
            {"raw",            no_argument,       nullptr, 'r'},
            {"reason",         no_argument,       nullptr, 'R'},
            {"debug",          no_argument,       nullptr, 'd'},
            {"teacher-tokens", required_argument, nullptr, 't'},
            {"info",           no_argument,       nullptr, 'i'},
            {"help",           no_argument,       nullptr, 'h'},
            {nullptr, 0, nullptr, 0},
        };

        optind = 1;
        int opt;
        while ((opt = getopt_long(argc, argv, "m:p:n:t:grRdih", long_options, nullptr)) != -1) {
            switch (opt) {
                case 'm': opts.model_path = optarg; break;
                case 'p': opts.template_path = optarg; break;
                case 'n': opts.n_predict = atoi(optarg); break;
                case 't': teacher_str = optarg; break;
                case 'g': opts.greedy = true; break;
                case 'r': opts.raw = true; break;
                case 'R': opts.reason = true; break;
                case 'd': opts.debug = true; break;
                case 'i': opts.info_only = true; break;
                case 'h': print_usage(argv[0]); exit(0);
                default:  print_usage(argv[0]); exit(1);
            }
        }

        // Positional arguments: [model.gguf] [input text...]
        for (int i = optind; i < argc; i++) {
            std::string arg = argv[i];
            if (opts.model_path.empty() && arg.size() > 5 &&
                arg.compare(arg.size() - 5, 5, ".gguf") == 0) {
                opts.model_path = arg;
            } else {
                if (!opts.input_text.empty()) opts.input_text += " ";
                opts.input_text += arg;
            }
        }

        // Parse comma-separated teacher tokens
        if (!teacher_str.empty()) {
            std::istringstream iss(teacher_str);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                if (!tok.empty()) opts.teacher_tokens.push_back(std::stoi(tok));
            }
        }

        return opts;
    }
};

} // namespace gwen
