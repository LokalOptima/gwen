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
    int n_predict = 50;
    bool greedy = true;
    bool info_only = false;
    std::string input_text;
    std::vector<int> teacher_tokens;

    // Apply template to input text, returning the final prompt.
    // If template_path is a readable file, loads it; otherwise treats it as an inline template string.
    // {input} in the template is replaced with input_text.
    std::string build_prompt() const {
        if (template_path.empty()) return input_text;

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

    static void print_usage(const char* prog) {
        fprintf(stderr, "Usage: %s [options] [model.gguf] [input text...]\n", prog);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --model PATH         Path to GGUF model file\n");
        fprintf(stderr, "  --template STR/FILE  Template (file or string); {input} replaced with input text\n");
        fprintf(stderr, "  --max-predict N      Number of tokens to generate (default: 50)\n");
        fprintf(stderr, "  --greedy             Use greedy decoding\n");
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
            {"teacher-tokens", required_argument, nullptr, 't'},
            {"info",           no_argument,       nullptr, 'i'},
            {"help",           no_argument,       nullptr, 'h'},
            {nullptr, 0, nullptr, 0},
        };

        optind = 1;
        int opt;
        while ((opt = getopt_long(argc, argv, "m:p:n:t:gih", long_options, nullptr)) != -1) {
            switch (opt) {
                case 'm': opts.model_path = optarg; break;
                case 'p': opts.template_path = optarg; break;
                case 'n': opts.n_predict = atoi(optarg); break;
                case 't': teacher_str = optarg; break;
                case 'g': opts.greedy = true; break;
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
