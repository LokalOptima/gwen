#pragma once

#include "gwen/gguf.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace gwen {

// Minimal tokenizer from GGUF metadata
// Uses the token vocabulary and BPE merges from the model file
class Tokenizer {
public:
    static std::unique_ptr<Tokenizer> from_gguf(const GGUFFile& gguf);

    // Encode text to token IDs (handles <|special|> tokens, then BPE)
    std::vector<int> encode(const std::string& text) const;

    // BPE-encode a plain text segment (no special tokens)
    std::vector<int> encode_bpe(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode(int token_id) const;

    int vocab_size() const { return (int)id_to_token_.size(); }
    int eos_token_id() const { return eos_id_; }
    int pad_token_id() const { return pad_id_; }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    // BPE merge rank: "first second" → priority rank
    std::unordered_map<std::string, int> merge_rank_;

    int eos_id_ = 248046;
    int pad_id_ = 248055;
};

} // namespace gwen
