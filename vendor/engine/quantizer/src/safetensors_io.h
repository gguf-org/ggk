#pragma once

// Minimal standalone safetensors reader - no external JSON library.
//
// Parses the safetensors header (https://huggingface.co/docs/safetensors)
// with a small built-in JSON parser and maps every tensor onto the same
// qz::gguf_file representation the quantizer uses for GGUF inputs:
//   - shapes are reversed into ne[] order (innermost dim first)
//   - dtypes with a tensor-type equivalent (F32/F16/BF16/I8/I16/I32) map directly
//   - dtypes without one are tagged for load-time conversion, following
//     stable-diffusion.cpp: F64 -> f32, I64 -> i32, F8_E4M3/F8_E5M2 -> f16
//   - U8 tensors are skipped (reported via `warnings`)
// safetensors has no KV metadata section, so the resulting gguf_file has
// n_kv = 0 and the default alignment.
//
// Multi-part models are merged automatically: when the path matches the
// "<prefix>-00001-of-00003.safetensors" naming pattern (any part may be
// passed, any digit width), all parts are enumerated and their tensors are
// combined into one gguf_file, with gguf_tensor_info::file_idx pointing at
// the owning entry of gguf_file::data_paths.

#include "gguf_io.h"

#include <string>
#include <vector>

namespace qz {

// true if the file starts with a plausible safetensors header
// (used for input-format detection together with the GGUF magic)
bool safetensors_sniff(const char * path);

// parse the safetensors header(s) of `path` - plus its sibling parts when the
// name follows the multi-part pattern - into `out`.
// non-fatal issues (skipped tensors) are appended to `warnings`.
// returns false and sets `error` on failure.
bool safetensors_read_meta(const char * path, gguf_file & out, std::vector<std::string> & warnings,
                           std::string & error);

// convert `n_elements` raw on-disk elements to the in-memory tensor type,
// in place (the buffer must be at least nbytes() large, see storage_nbytes()).
// no-op for storage_kind::native.
void convert_storage_in_place(storage_kind kind, void * data, int64_t n_elements);

} // namespace qz
