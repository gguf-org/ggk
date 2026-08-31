#pragma once

// Minimal standalone GGUF reader/writer.
//
// The quantizer only ever changes tensor types, sizes and offsets; all
// key/value metadata is passed through untouched. So the reader parses the
// KV section just far enough to skip over it (and to pick up
// "general.alignment"), keeps the raw KV bytes for verbatim copy into the
// output file, and fully parses only the tensor-info section.

#include "kernels/qz_quant.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace qz {

constexpr uint32_t GGUF_MAGIC             = 0x46554747; // "GGUF" little-endian
constexpr size_t   GGUF_DEFAULT_ALIGNMENT = 32;

// on-disk encoding of the tensor data when it differs from the in-memory
// tensor type. GGUF data is always native; safetensors dtypes without a
// equivalent are converted right after reading (see convert_storage_in_place
// in safetensors_io.h).
enum class storage_kind : uint8_t {
    native = 0, // bytes on disk are already in `type`
    f8_e4m3,    // 1 byte/element on disk -> f16 in memory
    f8_e5m2,    // 1 byte/element on disk -> f16 in memory
    f64,        // 8 bytes/element on disk -> f32 in memory
    i64,        // 8 bytes/element on disk -> i32 in memory
};

struct gguf_tensor_info {
    std::string  name;
    uint32_t     n_dims = 1;
    int64_t      ne[QZ_MAX_DIMS] = { 1, 1, 1, 1 };
    qz_type    type     = QZ_TYPE_F32;
    uint64_t     offset   = 0; // relative to the start of the data section
    uint32_t     file_idx = 0; // which gguf_file::data_paths entry holds the data
    storage_kind storage  = storage_kind::native;

    int64_t n_elements() const {
        return ne[0] * ne[1] * ne[2] * ne[3];
    }
    int64_t nrows() const {
        return ne[1] * ne[2] * ne[3];
    }
    // size of the tensor data in bytes (in memory, i.e. as `type`);
    // 0 if the type is unknown/removed
    size_t nbytes() const;
    // size of the tensor data as stored on disk (== nbytes() for native)
    size_t storage_nbytes() const;
};

struct gguf_file {
    uint32_t version   = 3;
    uint64_t n_kv      = 0;
    size_t   alignment = GGUF_DEFAULT_ALIGNMENT;

    std::vector<uint8_t>          kv_bytes; // raw KV section, copied verbatim on write
    std::vector<gguf_tensor_info> tensors;

    // files holding the tensor data, indexed by gguf_tensor_info::file_idx.
    // a single file for GGUF; multi-part safetensors inputs have one entry
    // per part.
    std::vector<std::string> data_paths;

    uint64_t data_offset = 0; // file offset of the data section, added to
                              // every tensor offset (0 for safetensors, where
                              // tensor offsets are absolute per part)
};

// parse the header/KV/tensor-info sections of a GGUF file.
// returns false and sets `error` on failure.
bool gguf_read_meta(const char * path, gguf_file & out, std::string & error);

// write header + verbatim KV bytes + tensor infos + padding, leaving the file
// positioned at the start of the data section. `tensors` must carry the final
// (output) types and offsets. returns false and sets `error` on failure.
bool gguf_write_meta(FILE * f, const gguf_file & in, const std::vector<gguf_tensor_info> & tensors,
                     std::string & error);

// 64-bit-safe seek + short-transfer-safe read/write helpers (shared with the
// tensor-data streaming loop in quantize.cpp)
bool seek64(FILE * f, int64_t offset);
int64_t tell64(FILE * f);
bool read_exact(FILE * f, void * buf, size_t n);
bool write_exact(FILE * f, const void * buf, size_t n);
bool write_zero_padding(FILE * f, size_t nbytes, size_t alignment);

inline size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

} // namespace qz
