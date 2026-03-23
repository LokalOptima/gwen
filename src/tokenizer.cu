#include "gwen/tokenizer.h"
#include <algorithm>
#include <sstream>
#include <fstream>

namespace gwen {

// ============================================================
// Load from HuggingFace model directory
// ============================================================

// Minimal JSON string parser — extracts vocab from tokenizer.json
// Format: { "model": { "vocab": { "token": id, ... } }, "added_tokens": [...] }
static std::unordered_map<std::string, int> parse_vocab_from_tokenizer_json(const std::string& path) {
    std::ifstream f(path);
    GWEN_CHECK(f.is_open(), ("Failed to open " + path).c_str());
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    std::unordered_map<std::string, int> vocab;

    // Find "vocab": { ... } section
    size_t vocab_start = content.find("\"vocab\"");
    if (vocab_start == std::string::npos) return vocab;
    vocab_start = content.find('{', vocab_start);
    if (vocab_start == std::string::npos) return vocab;
    vocab_start++;

    // Parse key-value pairs: "token": id
    size_t pos = vocab_start;
    int brace_depth = 1;
    while (pos < content.size() && brace_depth > 0) {
        // Skip whitespace
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' ||
               content[pos] == '\r' || content[pos] == '\t' || content[pos] == ',')) pos++;
        if (pos >= content.size() || content[pos] == '}') { brace_depth--; pos++; continue; }
        if (content[pos] == '{') { brace_depth++; pos++; continue; }

        // Parse key (quoted string)
        if (content[pos] != '"') { pos++; continue; }
        pos++; // skip opening quote
        std::string key;
        while (pos < content.size() && content[pos] != '"') {
            if (content[pos] == '\\' && pos + 1 < content.size()) {
                pos++;
                switch (content[pos]) {
                    case 'n': key += '\n'; break;
                    case 't': key += '\t'; break;
                    case '\\': key += '\\'; break;
                    case '"': key += '"'; break;
                    case '/': key += '/'; break;
                    case 'u': {
                        // Parse \uXXXX
                        if (pos + 4 < content.size()) {
                            int cp = std::stoi(content.substr(pos + 1, 4), nullptr, 16);
                            pos += 4;
                            if (cp < 0x80) {
                                key += (char)cp;
                            } else if (cp < 0x800) {
                                key += (char)(0xC0 | (cp >> 6));
                                key += (char)(0x80 | (cp & 0x3F));
                            } else {
                                key += (char)(0xE0 | (cp >> 12));
                                key += (char)(0x80 | ((cp >> 6) & 0x3F));
                                key += (char)(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: key += content[pos]; break;
                }
            } else {
                key += content[pos];
            }
            pos++;
        }
        if (pos < content.size()) pos++; // skip closing quote

        // Skip colon
        while (pos < content.size() && content[pos] != ':') pos++;
        if (pos < content.size()) pos++;

        // Skip whitespace
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' ||
               content[pos] == '\r' || content[pos] == '\t')) pos++;

        // Parse integer value
        if (pos < content.size() && (content[pos] == '-' || (content[pos] >= '0' && content[pos] <= '9'))) {
            size_t end;
            int val = std::stoi(content.substr(pos), &end);
            pos += end;
            vocab[key] = val;
        }
    }

    // Also parse added_tokens: [{ "id": N, "content": "...", ... }, ...]
    size_t at_start = content.find("\"added_tokens\"");
    if (at_start != std::string::npos) {
        at_start = content.find('[', at_start);
        if (at_start != std::string::npos) {
            pos = at_start + 1;
            while (pos < content.size() && content[pos] != ']') {
                size_t obj_start = content.find('{', pos);
                if (obj_start == std::string::npos) break;
                size_t obj_end = content.find('}', obj_start);
                if (obj_end == std::string::npos) break;
                std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

                // Extract "id" and "content"
                size_t id_pos = obj.find("\"id\"");
                size_t ct_pos = obj.find("\"content\"");
                if (id_pos != std::string::npos && ct_pos != std::string::npos) {
                    id_pos = obj.find(':', id_pos) + 1;
                    while (id_pos < obj.size() && obj[id_pos] == ' ') id_pos++;
                    int id = std::stoi(obj.substr(id_pos));

                    ct_pos = obj.find(':', ct_pos) + 1;
                    while (ct_pos < obj.size() && obj[ct_pos] == ' ') ct_pos++;
                    if (ct_pos < obj.size() && obj[ct_pos] == '"') {
                        ct_pos++;
                        size_t ct_end = obj.find('"', ct_pos);
                        if (ct_end != std::string::npos) {
                            std::string token = obj.substr(ct_pos, ct_end - ct_pos);
                            vocab[token] = id;
                        }
                    }
                }
                pos = obj_end + 1;
            }
        }
    }

    return vocab;
}

std::unique_ptr<Tokenizer> Tokenizer::from_hf_dir(const std::string& dir) {
    auto tok = std::make_unique<Tokenizer>();

    // Load vocab from vocab.json (simpler format: { "token": id, ... })
    auto vocab_map = parse_vocab_from_tokenizer_json(dir + "/vocab.json");
    GWEN_CHECK(!vocab_map.empty(), "Failed to parse vocab from vocab.json");

    // Build id_to_token (find max id)
    int max_id = 0;
    for (const auto& [token, id] : vocab_map) {
        if (id > max_id) max_id = id;
    }
    tok->id_to_token_.resize(max_id + 1);
    tok->token_to_id_.reserve(vocab_map.size());
    for (const auto& [token, id] : vocab_map) {
        tok->id_to_token_[id] = token;
        tok->token_to_id_[token] = id;
    }

    // Load BPE merges from merges.txt
    std::ifstream mf(dir + "/merges.txt");
    if (mf.is_open()) {
        std::string line;
        int rank = 0;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;
            tok->merge_rank_[line] = rank++;
        }
    }

    // Special tokens (Qwen3.5 defaults)
    auto find_special = [&](const std::string& name, int default_id) -> int {
        auto it = tok->token_to_id_.find(name);
        return (it != tok->token_to_id_.end()) ? it->second : default_id;
    };
    tok->eos_id_ = find_special("<|endoftext|>", 248046);
    tok->pad_id_ = find_special("<|endoftext|>", 248055);

    fprintf(stderr, "Tokenizer: %d tokens, %d merges (from %s)\n",
            (int)tok->id_to_token_.size(), (int)tok->merge_rank_.size(), dir.c_str());
    // Debug: verify basic tokens
    auto check = [&](const std::string& s) {
        auto it = tok->token_to_id_.find(s);
        fprintf(stderr, "  '%s' → %s\n", s.c_str(),
                it != tok->token_to_id_.end() ? std::to_string(it->second).c_str() : "NOT FOUND");
    };
    check("A"); check("The"); check("!");
    return tok;
}

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const GGUFFile& gguf) {
    auto tok = std::make_unique<Tokenizer>();

    // Load vocabulary
    auto it = gguf.metadata().find("tokenizer.ggml.tokens");
    if (it != gguf.metadata().end()) {
        const auto& tokens = std::get<std::vector<std::string>>(it->second);
        tok->id_to_token_ = tokens;
        tok->token_to_id_.reserve(tokens.size());
        for (int i = 0; i < (int)tokens.size(); i++) {
            tok->token_to_id_[tokens[i]] = i;
        }
    }

    // Load BPE merges
    auto mit = gguf.metadata().find("tokenizer.ggml.merges");
    if (mit != gguf.metadata().end()) {
        const auto& merge_strs = std::get<std::vector<std::string>>(mit->second);
        tok->merge_rank_.reserve(merge_strs.size());
        for (int i = 0; i < (int)merge_strs.size(); i++) {
            tok->merge_rank_[merge_strs[i]] = i;
        }
    }

    // Special tokens
    tok->eos_id_ = gguf.get_u32("tokenizer.ggml.eos_token_id", 248046);
    tok->pad_id_ = gguf.get_u32("tokenizer.ggml.padding_token_id", 248055);

    return tok;
}

// GPT-2 style byte encoding: bytes 0-255 map to unicode characters
// This is the standard mapping used by Qwen/GPT-2/etc tokenizers
static std::string byte_to_unicode(uint8_t b) {
    // GPT-2 uses a specific mapping for bytes
    // Characters 33-126, 161-172, 174-255 are identity-mapped
    // Others are mapped to 256+ unicode codepoints
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        return std::string(1, (char)b);
    }
    // Map to higher unicode range: 0→Ā(256), 1→ā(257), etc.
    // UTF-8 encoding of codepoint 256+b
    int cp = 256 + b;
    // Codepoints 256-511 are 2-byte UTF-8
    char buf[3];
    buf[0] = (char)(0xC0 | (cp >> 6));
    buf[1] = (char)(0x80 | (cp & 0x3F));
    buf[2] = 0;
    return std::string(buf);
}

static std::unordered_map<std::string, uint8_t> build_unicode_to_byte() {
    std::unordered_map<std::string, uint8_t> map;
    for (int b = 0; b < 256; b++) {
        map[byte_to_unicode((uint8_t)b)] = (uint8_t)b;
    }
    return map;
}

// BPE-encode a plain text segment (no special tokens)
std::vector<int> Tokenizer::encode_bpe(const std::string& text) const {
    if (text.empty()) return {};

    std::vector<std::string> tokens;
    for (uint8_t b : text) {
        tokens.push_back(byte_to_unicode(b));
    }

    while (tokens.size() > 1) {
        int best_rank = INT_MAX;
        int best_pos = -1;

        for (int i = 0; i < (int)tokens.size() - 1; i++) {
            std::string key = tokens[i] + " " + tokens[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_pos < 0) break;

        tokens[best_pos] = tokens[best_pos] + tokens[best_pos + 1];
        tokens.erase(tokens.begin() + best_pos + 1);
    }

    std::vector<int> ids;
    for (const auto& t : tokens) {
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            for (uint8_t b : t) {
                auto bit = token_to_id_.find(byte_to_unicode(b));
                if (bit != token_to_id_.end()) {
                    ids.push_back(bit->second);
                }
            }
        }
    }

    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    enum { NORMAL, SAW_LT, IN_SPECIAL } state = NORMAL;

    std::vector<int> ids;
    std::string buf;
    std::string special;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        switch (state) {
        case NORMAL:
            if (c == '<') state = SAW_LT;
            else buf += c;
            break;

        case SAW_LT:
            if (c == '|') {
                special = "<|";
                state = IN_SPECIAL;
            } else {
                buf += '<';
                buf += c;
                state = NORMAL;
            }
            break;

        case IN_SPECIAL:
            special += c;
            if (c == '|' && i + 1 < text.size() && text[i + 1] == '>') {
                special += '>';
                i++;
                // Flush plain text, then emit special token
                auto it = token_to_id_.find(special);
                if (it == token_to_id_.end()) {
                    fprintf(stderr, "tokenizer: unknown special token: %s\n", special.c_str());
                    exit(1);
                }
                auto bpe = encode_bpe(buf);
                ids.insert(ids.end(), bpe.begin(), bpe.end());
                buf.clear();
                ids.push_back(it->second);
                special.clear();
                state = NORMAL;
            }
            break;
        }
    }

    if (state != NORMAL) {
        fprintf(stderr, "tokenizer: unterminated special token: %s\n",
                state == SAW_LT ? "<" : special.c_str());
        exit(1);
    }

    auto bpe = encode_bpe(buf);
    ids.insert(ids.end(), bpe.begin(), bpe.end());

    return ids;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= (int)id_to_token_.size()) return "";

    const std::string& token = id_to_token_[token_id];

    // Convert from unicode representation back to bytes
    static auto u2b = build_unicode_to_byte();

    std::string result;
    // Walk through the token string, matching unicode chars to bytes
    size_t i = 0;
    while (i < token.size()) {
        // Try 2-byte UTF-8 first
        if (i + 1 < token.size()) {
            std::string two(token.substr(i, 2));
            auto it = u2b.find(two);
            if (it != u2b.end()) {
                result += (char)it->second;
                i += 2;
                continue;
            }
        }
        // Single byte
        std::string one(1, token[i]);
        auto it = u2b.find(one);
        if (it != u2b.end()) {
            result += (char)it->second;
        } else {
            result += token[i];  // pass through
        }
        i++;
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int id : tokens) {
        result += decode(id);
    }
    return result;
}

} // namespace gwen
