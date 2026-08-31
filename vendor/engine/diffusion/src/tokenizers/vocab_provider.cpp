#include "vocab_provider.h"

#include <fstream>
#include <sstream>
#include <vector>

#include "core/util.h"
#include "hf_tokenizer_json.h"

namespace {

struct VocabKeyInfo {
    const char* key;
    const char* family;
    const char* file;  // canonical file name inside the family dir
};

const VocabKeyInfo VOCAB_KEYS[] = {
    {"clip.merges", "clip", "merges.txt"},
    {"qwen2.merges", "qwen2", "merges.txt"},
    {"mistral.merges", "mistral", "merges.txt"},
    {"mistral.vocab", "mistral", "vocab.json"},
    {"gemma.merges", "gemma", "merges.txt"},
    {"gemma.vocab", "gemma", "vocab.json"},
    {"gemma2.merges", "gemma2", "merges.txt"},
    {"gemma2.vocab", "gemma2", "vocab.json"},
    {"gpt_oss.merges", "gpt_oss", "merges.txt"},
    {"gpt_oss.vocab", "gpt_oss", "vocab.json"},
    {"t5.tokenizer_json", "t5", "tokenizer.json"},
    {"umt5.tokenizer_json", "umt5", "tokenizer.json"},
};

const VocabKeyInfo* find_key_info(const std::string& key) {
    for (const auto& info : VOCAB_KEYS) {
        if (key == info.key) {
            return &info;
        }
    }
    return nullptr;
}

bool read_whole_file(const std::string& path, std::string& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    out = ss.str();
    return !file.bad();
}

// Loads <family_dir>/<file>, synthesizing merges/vocab from a sibling
// tokenizer.json when the canonical file is absent.
bool load_from_family_dir(const std::string& family_dir,
                          const VocabKeyInfo& info,
                          const std::string& key,
                          std::string& out) {
    std::string path = family_dir + "/" + info.file;
    if (file_exists(path)) {
        if (read_whole_file(path, out)) {
            LOG_INFO("tokenizer data '%s' injected from '%s'", key.c_str(), path.c_str());
            return true;
        }
        LOG_WARN("failed to read tokenizer file '%s'", path.c_str());
    }
    std::string tokenizer_json_path = family_dir + "/tokenizer.json";
    if (std::string(info.file) == "tokenizer.json" || !file_exists(tokenizer_json_path)) {
        return false;
    }
    std::string tokenizer_json;
    if (!read_whole_file(tokenizer_json_path, tokenizer_json)) {
        return false;
    }
    bool ok = false;
    if (std::string(info.file) == "merges.txt") {
        ok = hf_tokenizer_json_extract_merges(tokenizer_json, out);
        if (ok && key == "clip.merges" && out.compare(0, 9, "#version:") != 0) {
            // CLIPTokenizer::load_from_merges skips a "#version:" header line
            // and asserts the exact line count.
            out = "#version: 0.2\n" + out;
        }
    } else {
        ok = hf_tokenizer_json_extract_vocab(tokenizer_json, out);
    }
    if (ok) {
        LOG_INFO("tokenizer data '%s' synthesized from '%s'", key.c_str(), tokenizer_json_path.c_str());
    }
    return ok;
}

}  // namespace

VocabProvider& VocabProvider::instance() {
    static VocabProvider provider;
    return provider;
}

void VocabProvider::set_pack_dir(const std::string& dir) {
    pack_dir_ = dir;
    if (!pack_dir_.empty()) {
        LOG_INFO("using tokenizer pack dir '%s'", pack_dir_.c_str());
    }
}

void VocabProvider::set_override(const std::string& key, std::string bytes) {
    LOG_INFO("tokenizer data for '%s' provided by model file (%zu bytes)", key.c_str(), bytes.size());
    overrides_[key] = std::move(bytes);
}

bool VocabProvider::has_override(const std::string& key) const {
    return overrides_.find(key) != overrides_.end();
}

void VocabProvider::clear_overrides() {
    overrides_.clear();
}

bool VocabProvider::try_load(const std::string& key, std::string& out) {
    const VocabKeyInfo* info = find_key_info(key);
    if (info != nullptr && !pack_dir_.empty()) {
        if (load_from_family_dir(pack_dir_ + "/" + info->family, *info, key, out)) {
            return true;
        }
        if (load_from_family_dir(pack_dir_ + "/tokenizers/" + info->family, *info, key, out)) {
            return true;
        }
    }
    auto it = overrides_.find(key);
    if (it != overrides_.end()) {
        out = it->second;
        return true;
    }
    return false;
}
