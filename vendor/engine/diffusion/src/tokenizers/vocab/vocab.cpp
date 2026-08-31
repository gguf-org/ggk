#include "vocab.h"

#include "tokenizers/vocab_provider.h"

#include "clip_merges.hpp"
#include "gemma2_merges.hpp"
#include "gemma2_vocab.hpp"
#include "gemma_merges.hpp"
#include "gemma_vocab.hpp"
#include "gpt_oss_merges.hpp"
#include "gpt_oss_vocab.hpp"
#include "mistral_merges.hpp"
#include "mistral_vocab.hpp"
#include "qwen_merges.hpp"
#include "t5.hpp"
#include "umt5.hpp"

// Each accessor serves the compiled-in data unless the VocabProvider has an
// injected replacement (tokenizer pack dir or model-file override) for the key.

std::string load_clip_merges() {
    std::string data;
    if (VocabProvider::instance().try_load("clip.merges", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(clip_merges_utf8_c_str), sizeof(clip_merges_utf8_c_str));
}

std::string load_qwen2_merges() {
    std::string data;
    if (VocabProvider::instance().try_load("qwen2.merges", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(qwen2_merges_utf8_c_str), sizeof(qwen2_merges_utf8_c_str));
}

std::string load_mistral_merges() {
    std::string data;
    if (VocabProvider::instance().try_load("mistral.merges", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(mistral_merges_utf8_c_str), sizeof(mistral_merges_utf8_c_str));
}

std::string load_mistral_vocab_json() {
    std::string data;
    if (VocabProvider::instance().try_load("mistral.vocab", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(mistral_vocab_json_utf8_c_str), sizeof(mistral_vocab_json_utf8_c_str));
}

std::string load_t5_tokenizer_json() {
    std::string data;
    if (VocabProvider::instance().try_load("t5.tokenizer_json", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
}

std::string load_umt5_tokenizer_json() {
    std::string data;
    if (VocabProvider::instance().try_load("umt5.tokenizer_json", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(umt5_tokenizer_json_str), sizeof(umt5_tokenizer_json_str));
}

std::string load_gemma_merges() {
    std::string data;
    if (VocabProvider::instance().try_load("gemma.merges", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(gemma_merges_utf8_c_str), sizeof(gemma_merges_utf8_c_str));
}

std::string load_gemma_vocab_json() {
    std::string data;
    if (VocabProvider::instance().try_load("gemma.vocab", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(gemma_vocab_json_utf8_c_str), sizeof(gemma_vocab_json_utf8_c_str));
}

std::string load_gemma2_merges() {
    std::string data;
    if (VocabProvider::instance().try_load("gemma2.merges", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(gemma2_merges_utf8_c_str), sizeof(gemma2_merges_utf8_c_str));
}

std::string load_gemma2_vocab_json() {
    std::string data;
    if (VocabProvider::instance().try_load("gemma2.vocab", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(gemma2_vocab_json_utf8_c_str), sizeof(gemma2_vocab_json_utf8_c_str));
}

std::string load_gpt_oss_merges() {
    std::string data;
    if (VocabProvider::instance().try_load("gpt_oss.merges", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(gpt_oss_merges_utf8_c_str), sizeof(gpt_oss_merges_utf8_c_str));
}

std::string load_gpt_oss_vocab_json() {
    std::string data;
    if (VocabProvider::instance().try_load("gpt_oss.vocab", data)) {
        return data;
    }
    return std::string(reinterpret_cast<const char*>(gpt_oss_vocab_json_utf8_c_str), sizeof(gpt_oss_vocab_json_utf8_c_str));
}
