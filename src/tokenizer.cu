#include "gwen/tokenizer.h"
#include <algorithm>
#include <sstream>

namespace gwen {

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const GGUFFile& gguf) {
    auto tok = std::make_unique<Tokenizer>();

    // Load vocabulary
    auto it = gguf.metadata().find("tokenizer.ggml.tokens");
    if (it != gguf.metadata().end()) {
        const auto& tokens = std::get<std::vector<std::string>>(it->second);
        tok->id_to_token_ = tokens;
        for (int i = 0; i < (int)tokens.size(); i++) {
            tok->token_to_id_[tokens[i]] = i;
        }
    }

    // Load BPE merges
    auto mit = gguf.metadata().find("tokenizer.ggml.merges");
    if (mit != gguf.metadata().end()) {
        const auto& merge_strs = std::get<std::vector<std::string>>(mit->second);
        tok->merges_.reserve(merge_strs.size());
        for (int i = 0; i < (int)merge_strs.size(); i++) {
            // Each merge is "first second"
            auto space_pos = merge_strs[i].find(' ');
            if (space_pos != std::string::npos) {
                BPEMerge m;
                m.first = merge_strs[i].substr(0, space_pos);
                m.second = merge_strs[i].substr(space_pos + 1);
                m.rank = i;
                tok->merge_rank_[merge_strs[i]] = i;
                tok->merges_.push_back(std::move(m));
            }
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
