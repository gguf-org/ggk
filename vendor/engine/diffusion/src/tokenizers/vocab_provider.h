#ifndef __SD_TOKENIZERS_VOCAB_PROVIDER_H__
#define __SD_TOKENIZERS_VOCAB_PROVIDER_H__

#include <map>
#include <string>

// Optional runtime tokenizer injection layered over the compiled-in vocab
// data. Keys are "<family>.<kind>", e.g. "qwen2.merges", "mistral.vocab",
// "umt5.tokenizer_json".
//
// try_load() search order per key:
//   1. explicit tokenizer pack dir set via set_pack_dir() -- probes
//      <pack>/<family>/<file> then <pack>/tokenizers/<family>/<file>; if the
//      exact file is missing but the family dir has a HuggingFace
//      tokenizer.json, merges/vocab are synthesized from it
//   2. in-memory overrides (e.g. tokenizer_json embedded in a GGUF model file)
// If neither hits, try_load() returns false and the caller (vocab.cpp) falls
// back to the compiled-in data, so a pack only needs to carry the families it
// wants to replace.
//
// The provider is process-global: tokenizers are constructed with default
// arguments deep inside the conditioners, so configuration must happen before
// new_sd_ctx() builds them. Multiple contexts in one process share the pack dir.
class VocabProvider {
public:
    static VocabProvider& instance();

    void set_pack_dir(const std::string& dir);
    const std::string& pack_dir() const { return pack_dir_; }

    void set_override(const std::string& key, std::string bytes);
    bool has_override(const std::string& key) const;
    void clear_overrides();

    // Fills `out` with injected bytes for the key if any source provides them.
    // Returns false when the caller should use the compiled-in data.
    bool try_load(const std::string& key, std::string& out);

private:
    VocabProvider() = default;

    std::string pack_dir_;
    std::map<std::string, std::string> overrides_;
};

#endif  // __SD_TOKENIZERS_VOCAB_PROVIDER_H__
