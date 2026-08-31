#ifndef __SD_TOKENIZERS_HF_TOKENIZER_JSON_H__
#define __SD_TOKENIZERS_HF_TOKENIZER_JSON_H__

#include <string>

// Extractors for HuggingFace tokenizer.json files, so packs (and GGUF-embedded
// blobs) can ship the single file HF actually distributes instead of
// pre-split merges.txt / vocab.json.

// model.merges -> merges.txt content (one "a b" pair per line). Handles both
// the legacy ["a b", ...] and the newer [["a","b"], ...] representations.
// Returns false if the JSON has no BPE merges.
bool hf_tokenizer_json_extract_merges(const std::string& tokenizer_json, std::string& merges_out);

// model.vocab (token -> id object) -> plain vocab.json string, plus
// added_tokens merged in (they carry ids outside model.vocab for some models).
// Returns false if the JSON has no vocab object.
bool hf_tokenizer_json_extract_vocab(const std::string& tokenizer_json, std::string& vocab_json_out);

#endif  // __SD_TOKENIZERS_HF_TOKENIZER_JSON_H__
