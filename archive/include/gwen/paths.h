#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace gwen {

// GitHub release URL for auto-downloading weights
static constexpr const char* RELEASE_BASE =
    "https://github.com/LokalOptima/gwen/releases/download/v1.0.0";

// Default filenames
static constexpr const char* DEFAULT_MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf";
static constexpr const char* DEFAULT_MTP   = "mtp_v6_sparse64.bin";

inline std::string cache_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/gwen";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.cache/gwen";
    return ".";
}

inline void mkdirs(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/')
            mkdir(cur.c_str(), 0755);
    }
    if (!cur.empty() && cur.back() != '/')
        mkdir(cur.c_str(), 0755);
}

inline void ensure_file(const std::string& path, const char* url) {
    if (access(path.c_str(), F_OK) == 0) return;
    std::string dir = path.substr(0, path.rfind('/'));
    if (!dir.empty()) mkdirs(dir);
    fprintf(stderr, "Downloading %s\n", path.c_str());
    pid_t pid = fork();
    if (pid == 0) {
        execlp("curl", "curl", "-#", "-fL", "-o", path.c_str(), url, nullptr);
        _exit(127);
    }
    int status;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        unlink(path.c_str());
        fprintf(stderr, "Download failed. Get weights from %s\n", RELEASE_BASE);
        exit(1);
    }
}

inline std::string default_model_path() {
    return cache_dir() + "/" + DEFAULT_MODEL;
}

inline std::string default_mtp_path() {
    return cache_dir() + "/" + DEFAULT_MTP;
}

// Ensure default weights exist, downloading if needed.
inline void ensure_default_weights(const std::string& model_path,
                                   const std::string& mtp_path) {
    std::string model_url = std::string(RELEASE_BASE) + "/" + DEFAULT_MODEL;
    std::string mtp_url   = std::string(RELEASE_BASE) + "/" + DEFAULT_MTP;
    ensure_file(model_path, model_url.c_str());
    ensure_file(mtp_path, mtp_url.c_str());
}

}  // namespace gwen
