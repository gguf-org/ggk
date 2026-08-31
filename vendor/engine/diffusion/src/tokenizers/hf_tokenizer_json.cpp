#include "hf_tokenizer_json.h"

#include "json.hpp"

bool hf_tokenizer_json_extract_merges(const std::string& tokenizer_json, std::string& merges_out) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(tokenizer_json);
    } catch (const nlohmann::json::parse_error&) {
        return false;
    }
    if (!root.contains("model") || !root["model"].contains("merges") || !root["model"]["merges"].is_array()) {
        return false;
    }
    std::string result;
    for (const auto& merge : root["model"]["merges"]) {
        if (merge.is_string()) {
            result += merge.get<std::string>();
        } else if (merge.is_array() && merge.size() == 2 && merge[0].is_string() && merge[1].is_string()) {
            result += merge[0].get<std::string>() + " " + merge[1].get<std::string>();
        } else {
            return false;
        }
        result += "\n";
    }
    if (result.empty()) {
        return false;
    }
    result.pop_back();  // parsers split on '\n' and expect no trailing empty line
    merges_out = std::move(result);
    return true;
}

bool hf_tokenizer_json_extract_vocab(const std::string& tokenizer_json, std::string& vocab_json_out) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(tokenizer_json);
    } catch (const nlohmann::json::parse_error&) {
        return false;
    }
    if (!root.contains("model") || !root["model"].contains("vocab") || !root["model"]["vocab"].is_object()) {
        return false;
    }
    nlohmann::json vocab = root["model"]["vocab"];
    if (root.contains("added_tokens") && root["added_tokens"].is_array()) {
        for (const auto& added : root["added_tokens"]) {
            if (added.contains("content") && added.contains("id")) {
                vocab[added["content"].get<std::string>()] = added["id"];
            }
        }
    }
    vocab_json_out = vocab.dump();
    return true;
}
