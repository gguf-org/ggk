#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#  define _FILE_OFFSET_BITS 64
#endif

#include "gguf_io.h"

#include <cinttypes>
#include <cstring>

#if !defined(_WIN32)
#  include <sys/types.h>
#endif

namespace qz {

static_assert(sizeof(float) == 4, "gguf requires 32-bit floats");

// GGUF metadata value types (part of the file format)
enum gguf_kv_type : uint32_t {
    GGUF_KV_UINT8   = 0,
    GGUF_KV_INT8    = 1,
    GGUF_KV_UINT16  = 2,
    GGUF_KV_INT16   = 3,
    GGUF_KV_UINT32  = 4,
    GGUF_KV_INT32   = 5,
    GGUF_KV_FLOAT32 = 6,
    GGUF_KV_BOOL    = 7,
    GGUF_KV_STRING  = 8,
    GGUF_KV_ARRAY   = 9,
    GGUF_KV_UINT64  = 10,
    GGUF_KV_INT64   = 11,
    GGUF_KV_FLOAT64 = 12,
};

bool seek64(FILE * f, int64_t offset) {
#if defined(_WIN32)
    return _fseeki64(f, offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) offset, SEEK_SET) == 0;
#endif
}

int64_t tell64(FILE * f) {
#if defined(_WIN32)
    return _ftelli64(f);
#else
    return (int64_t) ftello(f);
#endif
}

// fread()/fwrite() may legally transfer fewer bytes than requested in a single
// call for reasons other than EOF/error (observed on Windows for large single
// transfers - antivirus/cloud-filter drivers, network/exFAT volumes, etc.).
bool read_exact(FILE * f, void * buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t chunk = fread((char *) buf + done, 1, n - done, f);
        if (chunk == 0) {
            return false; // real EOF or error
        }
        done += chunk;
    }
    return true;
}

bool write_exact(FILE * f, const void * buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t chunk = fwrite((const char *) buf + done, 1, n - done, f);
        if (chunk == 0) {
            return false;
        }
        done += chunk;
    }
    return true;
}

bool write_zero_padding(FILE * f, size_t nbytes, size_t alignment) {
    static const char zeros[256] = { 0 };
    size_t pad = align_up(nbytes, alignment) - nbytes;
    while (pad > 0) {
        size_t chunk = pad < sizeof(zeros) ? pad : sizeof(zeros);
        if (!write_exact(f, zeros, chunk)) {
            return false;
        }
        pad -= chunk;
    }
    return true;
}

size_t gguf_tensor_info::nbytes() const {
    if (qz_type_name(type) == nullptr) {
        return 0; // removed/unknown type id
    }
    const int64_t blck = qz_blck_size(type);
    if (blck <= 0 || ne[0] % blck != 0) {
        return 0;
    }
    return qz_row_size(type, ne[0]) * (size_t) nrows();
}

size_t gguf_tensor_info::storage_nbytes() const {
    switch (storage) {
        case storage_kind::f8_e4m3:
        case storage_kind::f8_e5m2: return (size_t) n_elements();     // 1 byte/element
        case storage_kind::f64:
        case storage_kind::i64:     return (size_t) n_elements() * 8; // 8 bytes/element
        case storage_kind::native:  break;
    }
    return nbytes();
}

namespace {

struct reader {
    FILE *        f;
    std::string & error;
    bool          ok = true;

    bool fail(const std::string & msg) {
        if (ok) { // keep the first error
            error = msg;
            ok    = false;
        }
        return false;
    }

    template <typename T>
    bool read(T & out) {
        if (!ok) {
            return false;
        }
        if (!read_exact(f, &out, sizeof(out))) {
            return fail("unexpected end of file");
        }
        return true;
    }

    bool read_str(std::string & out, const char * what) {
        uint64_t len = 0;
        if (!read(len)) {
            return false;
        }
        if (len > (uint64_t) 1 << 31) {
            return fail(std::string("implausible ") + what + " length (corrupt file?)");
        }
        out.resize((size_t) len);
        if (len > 0 && !read_exact(f, &out[0], (size_t) len)) {
            return fail("unexpected end of file");
        }
        return true;
    }

    bool skip(uint64_t n) {
        if (!ok) {
            return false;
        }
        if (!seek64(f, tell64(f) + (int64_t) n)) {
            return fail("unexpected end of file");
        }
        return true;
    }
};

size_t kv_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_KV_UINT8:
        case GGUF_KV_INT8:
        case GGUF_KV_BOOL:    return 1;
        case GGUF_KV_UINT16:
        case GGUF_KV_INT16:   return 2;
        case GGUF_KV_UINT32:
        case GGUF_KV_INT32:
        case GGUF_KV_FLOAT32: return 4;
        case GGUF_KV_UINT64:
        case GGUF_KV_INT64:
        case GGUF_KV_FLOAT64: return 8;
        default:              return 0; // string/array/invalid
    }
}

} // namespace

bool gguf_read_meta(const char * path, gguf_file & out, std::string & error) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        error = "failed to open file";
        return false;
    }

    reader r{ f, error };

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t n_tensors = 0;

    r.read(magic);
    if (r.ok && magic != GGUF_MAGIC) {
        r.fail("not a GGUF file (bad magic)");
    }
    r.read(version);
    if (r.ok && (version < 2 || version > 3)) {
        r.fail("unsupported GGUF version " + std::to_string(version) +
               (version == 1 ? " (version 1 files are obsolete)" : ""));
    }
    r.read(n_tensors);
    r.read(out.n_kv);
    if (r.ok && (n_tensors > (uint64_t) 1 << 24 || out.n_kv > (uint64_t) 1 << 24)) {
        r.fail("implausible tensor/kv count (corrupt file?)");
    }
    out.version = version;

    // ------------------------------------------------------------------
    // KV section: skip through it, extracting only general.alignment
    // ------------------------------------------------------------------
    const int64_t kv_start = r.ok ? tell64(f) : 0;

    for (uint64_t i = 0; r.ok && i < out.n_kv; ++i) {
        std::string key;
        uint32_t    vtype = 0;
        if (!r.read_str(key, "key") || !r.read(vtype)) {
            break;
        }

        uint64_t alignment_value  = 0;
        bool     alignment_valid  = false;

        if (vtype == GGUF_KV_STRING) {
            std::string dummy;
            r.read_str(dummy, "string value");
        } else if (vtype == GGUF_KV_ARRAY) {
            uint32_t etype = 0;
            uint64_t n     = 0;
            if (!r.read(etype) || !r.read(n)) {
                break;
            }
            if (n > (uint64_t) 1 << 32) {
                r.fail("implausible array length for key '" + key + "' (corrupt file?)");
            } else if (etype == GGUF_KV_STRING) {
                for (uint64_t j = 0; r.ok && j < n; ++j) {
                    std::string dummy;
                    r.read_str(dummy, "string array element");
                }
            } else if (etype == GGUF_KV_ARRAY) {
                r.fail("arrays of arrays are not supported");
            } else {
                size_t esize = kv_scalar_size(etype);
                if (esize == 0) {
                    r.fail("invalid array element type " + std::to_string(etype) + " for key '" + key + "'");
                } else {
                    r.skip(n * esize);
                }
            }
        } else {
            size_t vsize = kv_scalar_size(vtype);
            if (vsize == 0) {
                r.fail("invalid value type " + std::to_string(vtype) + " for key '" + key + "'");
            } else {
                uint64_t raw = 0;
                if (read_exact(f, &raw, vsize)) { // little-endian: low bytes first
                    alignment_value = raw;
                    alignment_valid = vtype == GGUF_KV_UINT32 || vtype == GGUF_KV_UINT64 ||
                                      vtype == GGUF_KV_INT32  || vtype == GGUF_KV_INT64;
                } else {
                    r.fail("unexpected end of file");
                }
            }
        }

        if (r.ok && key == "general.alignment") {
            if (alignment_valid && alignment_value > 0 && (alignment_value & (alignment_value - 1)) == 0) {
                out.alignment = (size_t) alignment_value;
            } else {
                r.fail("invalid general.alignment (must be an integer power of 2)");
            }
        }
    }

    // stash the raw KV bytes for verbatim copy into the output file
    const int64_t kv_end = r.ok ? tell64(f) : 0;
    if (r.ok) {
        out.kv_bytes.resize((size_t) (kv_end - kv_start));
        if (!seek64(f, kv_start) || !read_exact(f, out.kv_bytes.data(), out.kv_bytes.size()) ||
            !seek64(f, kv_end)) {
            r.fail("failed to re-read KV section");
        }
    }

    // ------------------------------------------------------------------
    // tensor infos
    // ------------------------------------------------------------------
    out.tensors.clear();
    out.tensors.reserve(r.ok ? (size_t) n_tensors : 0);

    for (uint64_t i = 0; r.ok && i < n_tensors; ++i) {
        gguf_tensor_info info;
        if (!r.read_str(info.name, "tensor name")) {
            break;
        }
        if (!r.read(info.n_dims)) {
            break;
        }
        if (info.n_dims < 1 || info.n_dims > QZ_MAX_DIMS) {
            r.fail("tensor '" + info.name + "' has unsupported number of dimensions (" +
                   std::to_string(info.n_dims) + ")");
            break;
        }
        for (uint32_t d = 0; r.ok && d < info.n_dims; ++d) {
            uint64_t ne = 0;
            r.read(ne);
            info.ne[d] = (int64_t) ne;
        }
        uint32_t type = 0;
        r.read(type);
        r.read(info.offset);
        if (!r.ok) {
            break;
        }
        if (type >= QZ_TYPE_COUNT || qz_type_name((qz_type) type) == nullptr) {
            r.fail("tensor '" + info.name + "' has unknown/removed type id " + std::to_string(type));
            break;
        }
        info.type = (qz_type) type;
        if (info.nbytes() == 0 && info.n_elements() != 0) {
            r.fail("tensor '" + info.name + "' has a row size that is not a multiple of its type's block size");
            break;
        }
        out.tensors.push_back(std::move(info));
    }

    if (r.ok) {
        out.data_offset = align_up((size_t) tell64(f), out.alignment);
        out.data_paths  = { path };
    }

    fclose(f);
    return r.ok;
}

bool gguf_write_meta(FILE * f, const gguf_file & in, const std::vector<gguf_tensor_info> & tensors,
                     std::string & error) {
    bool ok = true;

    auto w = [&](const void * buf, size_t n) {
        if (ok && !write_exact(f, buf, n)) {
            error = "write failed";
            ok    = false;
        }
    };
    auto w32 = [&](uint32_t v) { w(&v, sizeof(v)); };
    auto w64 = [&](uint64_t v) { w(&v, sizeof(v)); };

    w32(GGUF_MAGIC);
    w32(in.version);
    w64((uint64_t) tensors.size());
    w64(in.n_kv);
    w(in.kv_bytes.data(), in.kv_bytes.size());

    for (const gguf_tensor_info & t : tensors) {
        w64((uint64_t) t.name.size());
        w(t.name.data(), t.name.size());
        w32(t.n_dims);
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            w64((uint64_t) t.ne[d]);
        }
        w32((uint32_t) t.type);
        w64(t.offset);
    }

    if (ok) {
        const int64_t pos = tell64(f);
        if (pos < 0 || !write_zero_padding(f, (size_t) pos, in.alignment)) {
            error = "write failed";
            ok    = false;
        }
    }

    return ok;
}

} // namespace qz
