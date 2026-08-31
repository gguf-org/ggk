#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#  define _FILE_OFFSET_BITS 64
#endif

#include "safetensors_io.h"

#include <cmath>
#include <cstring>
#include <set>

#if !defined(_WIN32)
#  include <sys/types.h>
#endif

namespace qz {

namespace {

constexpr uint64_t ST_MAX_HEADER_SIZE = (uint64_t) 512 * 1024 * 1024; // same cap as the reference impl

bool seek_end64(FILE * f) {
#if defined(_WIN32)
    return _fseeki64(f, 0, SEEK_END) == 0;
#else
    return fseeko(f, 0, SEEK_END) == 0;
#endif
}

// ----------------------------------------------------------------------
// minimal JSON parser - just enough for safetensors headers: one object of
// objects whose values are strings, arrays of non-negative integers, and
// (inside __metadata__ or unknown keys) arbitrary values that only need to
// be skipped.
// ----------------------------------------------------------------------
struct json_reader {
    const char *  p;
    const char *  end;
    std::string & error;
    bool          ok = true;

    json_reader(const char * begin, const char * end_, std::string & err) : p(begin), end(end_), error(err) {}

    bool fail(const std::string & msg) {
        if (ok) { // keep the first error
            error = "invalid safetensors header: " + msg;
            ok    = false;
        }
        return false;
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
            ++p;
        }
    }

    // consume `c` if it is the next non-whitespace character; never fails
    bool consume_if(char c) {
        skip_ws();
        if (ok && p < end && *p == c) {
            ++p;
            return true;
        }
        return false;
    }

    bool consume(char c) {
        if (!consume_if(c)) {
            return fail(std::string("expected '") + c + "'");
        }
        return true;
    }

    bool parse_hex4(uint32_t & v) {
        if (end - p < 4) {
            return fail("truncated \\u escape");
        }
        v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') {
                v |= (uint32_t) (c - '0');
            } else if (c >= 'a' && c <= 'f') {
                v |= (uint32_t) (c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                v |= (uint32_t) (c - 'A' + 10);
            } else {
                return fail("bad \\u escape");
            }
        }
        return true;
    }

    void append_utf8(std::string & out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }

    bool parse_string(std::string & out) {
        out.clear();
        if (!consume('"')) {
            return false;
        }
        while (ok) {
            if (p >= end) {
                return fail("unterminated string");
            }
            char c = *p++;
            if (c == '"') {
                return true;
            }
            if ((unsigned char) c < 0x20) {
                return fail("unescaped control character in string");
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (p >= end) {
                return fail("unterminated string escape");
            }
            char e = *p++;
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!parse_hex4(cp)) {
                        return false;
                    }
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        uint32_t lo = 0;
                        if (end - p < 2 || p[0] != '\\' || p[1] != 'u') {
                            return fail("unpaired UTF-16 surrogate");
                        }
                        p += 2;
                        if (!parse_hex4(lo)) {
                            return false;
                        }
                        if (lo < 0xDC00 || lo > 0xDFFF) {
                            return fail("invalid UTF-16 surrogate pair");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("unpaired UTF-16 surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return fail("invalid string escape");
            }
        }
        return false;
    }

    bool parse_u64(uint64_t & out) {
        skip_ws();
        if (p >= end || *p < '0' || *p > '9') {
            return fail("expected a non-negative integer");
        }
        out = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            uint64_t digit = (uint64_t) (*p - '0');
            if (out > (UINT64_MAX - digit) / 10) {
                return fail("integer out of range");
            }
            out = out * 10 + digit;
            ++p;
        }
        return true;
    }

    bool skip_literal(const char * lit) {
        size_t n = strlen(lit);
        if ((size_t) (end - p) < n || memcmp(p, lit, n) != 0) {
            return fail("invalid value");
        }
        p += n;
        return true;
    }

    bool skip_number() {
        if (p < end && *p == '-') {
            ++p;
        }
        const char * digits_start = p;
        while (p < end && *p >= '0' && *p <= '9') {
            ++p;
        }
        if (p == digits_start) {
            return fail("invalid number");
        }
        if (p < end && *p == '.') {
            ++p;
            while (p < end && *p >= '0' && *p <= '9') {
                ++p;
            }
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            ++p;
            if (p < end && (*p == '+' || *p == '-')) {
                ++p;
            }
            while (p < end && *p >= '0' && *p <= '9') {
                ++p;
            }
        }
        return true;
    }

    // skip any JSON value (used for __metadata__ and unknown keys)
    bool skip_value(int depth = 0) {
        if (depth > 32) {
            return fail("value nesting too deep");
        }
        skip_ws();
        if (!ok) {
            return false;
        }
        if (p >= end) {
            return fail("unexpected end of header");
        }
        char c = *p;
        if (c == '"') {
            std::string dummy;
            return parse_string(dummy);
        }
        if (c == '{' || c == '[') {
            const char open  = c;
            const char close = c == '{' ? '}' : ']';
            ++p;
            if (consume_if(close)) {
                return ok;
            }
            do {
                if (open == '{') {
                    std::string key;
                    if (!parse_string(key) || !consume(':')) {
                        return false;
                    }
                }
                if (!skip_value(depth + 1)) {
                    return false;
                }
            } while (consume_if(','));
            return consume(close);
        }
        if (c == 't') {
            return skip_literal("true");
        }
        if (c == 'f') {
            return skip_literal("false");
        }
        if (c == 'n') {
            return skip_literal("null");
        }
        return skip_number();
    }
};

// map a safetensors dtype string onto (tensor type, on-disk storage kind);
// returns false for dtypes the quantizer does not support
bool map_st_dtype(const std::string & dtype, qz_type & type, storage_kind & storage) {
    storage = storage_kind::native;
    if (dtype == "F32") {
        type = QZ_TYPE_F32;
    } else if (dtype == "F16") {
        type = QZ_TYPE_F16;
    } else if (dtype == "BF16") {
        type = QZ_TYPE_BF16;
    } else if (dtype == "F64") {
        type    = QZ_TYPE_F32;
        storage = storage_kind::f64;
    } else if (dtype == "F8_E4M3") {
        type    = QZ_TYPE_F16;
        storage = storage_kind::f8_e4m3;
    } else if (dtype == "F8_E5M2") {
        type    = QZ_TYPE_F16;
        storage = storage_kind::f8_e5m2;
    } else if (dtype == "I64") {
        type    = QZ_TYPE_I32;
        storage = storage_kind::i64;
    } else if (dtype == "I32") {
        type = QZ_TYPE_I32;
    } else if (dtype == "I16") {
        type = QZ_TYPE_I16;
    } else if (dtype == "I8") {
        type = QZ_TYPE_I8;
    } else {
        return false;
    }
    return true;
}

// read the 8-byte header length + header JSON; returns false on any problem
bool read_st_header(FILE * f, std::vector<char> & header, int64_t & file_size, std::string & error) {
    // determine the file size
    if (!seek_end64(f)) {
        error = "seek failed";
        return false;
    }
    file_size = tell64(f);
    if (file_size < 0 || !seek64(f, 0)) {
        error = "seek failed";
        return false;
    }

    uint64_t header_size = 0;
    if (file_size < 8 || !read_exact(f, &header_size, sizeof(header_size))) {
        error = "file too small to be a safetensors file";
        return false;
    }
    if (header_size < 2 || header_size > ST_MAX_HEADER_SIZE || (int64_t) header_size > file_size - 8) {
        error = "implausible safetensors header size (corrupt file?)";
        return false;
    }

    header.resize((size_t) header_size);
    if (!read_exact(f, header.data(), header.size())) {
        error = "unexpected end of file while reading the safetensors header";
        return false;
    }
    return true;
}

} // namespace

bool safetensors_sniff(const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    std::vector<char> header;
    int64_t           file_size = 0;
    std::string       error;
    bool              ok = read_st_header(f, header, file_size, error);
    fclose(f);
    if (!ok) {
        return false;
    }
    // the header must at least start as a JSON object
    json_reader r(header.data(), header.data() + header.size(), error);
    return r.consume_if('{');
}

namespace {

// parse one safetensors file, appending its tensors to `out` with absolute
// file offsets and file_idx set; `seen_names` is shared across parts so
// duplicate tensor names are rejected globally
bool parse_one_part(const char * path, uint32_t file_idx, gguf_file & out, std::set<std::string> & seen_names,
                    std::vector<std::string> & warnings, std::string & error) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        error = "failed to open file";
        return false;
    }

    std::vector<char> header;
    int64_t           file_size = 0;
    if (!read_st_header(f, header, file_size, error)) {
        fclose(f);
        return false;
    }
    fclose(f);

    const uint64_t data_start = 8 + header.size(); // tensor offsets are relative to this
    const uint64_t data_size  = (uint64_t) file_size - data_start;

    json_reader r(header.data(), header.data() + header.size(), error);

    r.consume('{');
    if (!r.consume_if('}')) {
        do {
            std::string name;
            if (!r.parse_string(name) || !r.consume(':')) {
                break;
            }
            if (name == "__metadata__") {
                r.skip_value();
                continue;
            }

            // tensor entry: {"dtype": "...", "shape": [...], "data_offsets": [begin, end]}
            std::string           dtype;
            std::vector<uint64_t> shape;
            uint64_t              begin = 0, end_off = 0;
            bool                  have_dtype = false, have_shape = false, have_offsets = false;

            if (!r.consume('{')) {
                break;
            }
            if (!r.consume_if('}')) {
                do {
                    std::string key;
                    if (!r.parse_string(key) || !r.consume(':')) {
                        break;
                    }
                    if (key == "dtype") {
                        have_dtype = r.parse_string(dtype);
                    } else if (key == "shape") {
                        if (!r.consume('[')) {
                            break;
                        }
                        if (!r.consume_if(']')) {
                            do {
                                uint64_t v = 0;
                                if (!r.parse_u64(v)) {
                                    break;
                                }
                                shape.push_back(v);
                            } while (r.consume_if(','));
                            if (!r.consume(']')) {
                                break;
                            }
                        }
                        have_shape = r.ok;
                    } else if (key == "data_offsets") {
                        have_offsets = r.consume('[') && r.parse_u64(begin) && r.consume(',') &&
                                       r.parse_u64(end_off) && r.consume(']');
                    } else {
                        r.skip_value();
                    }
                } while (r.consume_if(','));
                if (!r.consume('}')) {
                    break;
                }
            }
            if (!r.ok) {
                break;
            }

            if (!have_dtype || !have_shape || !have_offsets) {
                r.fail("tensor '" + name + "' is missing dtype/shape/data_offsets");
                break;
            }

            if (dtype == "U8" || dtype == "BOOL") {
                // opaque byte tensors (matches stable-diffusion.cpp, which skips U8)
                warnings.push_back("skipping tensor '" + name + "' with unsupported dtype " + dtype);
                continue;
            }

            gguf_tensor_info info;
            if (!map_st_dtype(dtype, info.type, info.storage)) {
                r.fail("tensor '" + name + "' has unsupported dtype '" + dtype + "'");
                break;
            }

            // a 5D shape is collapsed by merging the two outermost dims,
            // like other GGUF producers do (a tensor carries at most 4 dims)
            if (shape.size() == 5) {
                if (shape[0] != 0 && shape[1] > UINT64_MAX / shape[0]) {
                    r.fail("tensor '" + name + "' has an implausibly large shape");
                    break;
                }
                shape[1] *= shape[0];
                shape.erase(shape.begin());
            }
            if (shape.size() > QZ_MAX_DIMS) {
                r.fail("tensor '" + name + "' has unsupported number of dimensions (" +
                       std::to_string(shape.size()) + ")");
                break;
            }

            info.name   = name;
            info.n_dims = shape.empty() ? 1 : (uint32_t) shape.size(); // scalars become 1D [1]
            // safetensors shapes are outermost-first; ne[] is innermost-first
            for (size_t i = 0; i < shape.size(); ++i) {
                info.ne[i] = (int64_t) shape[shape.size() - 1 - i];
            }

            uint64_t n_elems = 1;
            for (uint64_t d : shape) {
                if (d != 0 && n_elems > UINT64_MAX / d) {
                    n_elems = UINT64_MAX;
                    break;
                }
                n_elems *= d;
            }
            if (n_elems > (uint64_t) 1 << 40) {
                r.fail("tensor '" + name + "' has an implausible element count (corrupt file?)");
                break;
            }

            if (end_off < begin || end_off > data_size) {
                r.fail("tensor '" + name + "' has data_offsets outside the file");
                break;
            }
            info.offset   = data_start + begin; // absolute offset within this part
            info.file_idx = file_idx;

            if (info.storage_nbytes() != end_off - begin) {
                r.fail("size mismatch for tensor '" + name + "' (" + dtype + ": expected " +
                       std::to_string(info.storage_nbytes()) + " bytes, data_offsets span " +
                       std::to_string(end_off - begin) + ")");
                break;
            }

            if (!seen_names.insert(name).second) {
                r.fail("duplicate tensor name '" + name + "'");
                break;
            }

            out.tensors.push_back(std::move(info));
        } while (r.consume_if(','));

        if (r.ok) {
            r.consume('}');
        }
    }

    if (r.ok) {
        r.skip_ws(); // the header may be padded with trailing whitespace
        if (r.p != r.end) {
            r.fail("trailing data after the top-level object");
        }
    }

    return r.ok;
}

// expand "<prefix>-00001-of-00003.safetensors" (any digit width) into the
// full list of part paths; returns false if `path` does not follow the
// multi-part naming pattern
bool enumerate_parts(const std::string & path, std::vector<std::string> & parts) {
    const std::string ext = ".safetensors";
    if (path.size() <= ext.size() || path.compare(path.size() - ext.size(), ext.size(), ext) != 0) {
        return false;
    }
    const std::string stem = path.substr(0, path.size() - ext.size());

    const size_t of_pos = stem.rfind("-of-");
    if (of_pos == std::string::npos) {
        return false;
    }

    const std::string count_str = stem.substr(of_pos + 4);
    if (count_str.empty() || count_str.size() > 9) {
        return false;
    }
    for (char c : count_str) {
        if (c < '0' || c > '9') {
            return false;
        }
    }

    // digits (plus a leading '-') immediately before "-of-"
    size_t idx_start = of_pos;
    while (idx_start > 0 && stem[idx_start - 1] >= '0' && stem[idx_start - 1] <= '9') {
        --idx_start;
    }
    if (idx_start == of_pos || idx_start < 2 || stem[idx_start - 1] != '-') {
        return false;
    }
    const int width = (int) (of_pos - idx_start);
    if (width > 9) {
        return false;
    }

    const unsigned long count = strtoul(count_str.c_str(), nullptr, 10);
    if (count < 1 || count > 99999) {
        return false;
    }

    const std::string prefix = stem.substr(0, idx_start); // includes the '-'
    parts.clear();
    parts.reserve(count);
    for (unsigned long i = 1; i <= count; ++i) {
        char idx_str[16];
        snprintf(idx_str, sizeof(idx_str), "%0*lu", width, i);
        parts.push_back(prefix + idx_str + "-of-" + count_str + ext);
    }
    return true;
}

} // namespace

bool safetensors_read_meta(const char * path, gguf_file & out, std::vector<std::string> & warnings,
                           std::string & error) {
    out.version     = 3;
    out.n_kv        = 0; // safetensors carries no GGUF KV metadata
    out.alignment   = GGUF_DEFAULT_ALIGNMENT;
    out.data_offset = 0; // safetensors tensor offsets are absolute per part
    out.kv_bytes.clear();
    out.tensors.clear();

    if (!enumerate_parts(path, out.data_paths)) {
        out.data_paths = { path }; // plain single-file input
    }

    std::set<std::string> seen_names;
    for (uint32_t i = 0; i < (uint32_t) out.data_paths.size(); ++i) {
        if (!parse_one_part(out.data_paths[i].c_str(), i, out, seen_names, warnings, error)) {
            if (out.data_paths.size() > 1) {
                error = "part '" + out.data_paths[i] + "': " + error;
            }
            return false;
        }
    }

    if (out.tensors.empty()) {
        error = "no tensors found";
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------
// load-time conversions for dtypes with no tensor-type equivalent
// ----------------------------------------------------------------------

namespace {

// ported from stable-diffusion.cpp's f8_e4m3_to_f16
qz_fp16_t f8_e4m3_to_f16(uint8_t f8) {
    const uint32_t exponent_bias = 7;
    if (f8 == 0xff) {
        return qz_fp32_to_fp16(-NAN);
    }
    if (f8 == 0x7f) {
        return qz_fp32_to_fp16(NAN);
    }

    uint32_t sign     = f8 & 0x80;
    uint32_t exponent = (f8 & 0x78) >> 3;
    uint32_t mantissa = f8 & 0x07;
    uint32_t result   = sign << 24;
    if (exponent == 0) {
        if (mantissa > 0) {
            exponent = 0x7f - exponent_bias;

            // yes, 2 times
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }

            result |= (mantissa & 0x03) << 21;
            result |= exponent << 23;
        }
    } else {
        result |= mantissa << 20;
        exponent += 0x7f - exponent_bias;
        result |= exponent << 23;
    }

    float value;
    memcpy(&value, &result, sizeof(value));
    return qz_fp32_to_fp16(value);
}

} // namespace

void convert_storage_in_place(storage_kind kind, void * data, int64_t n_elements) {
    switch (kind) {
        case storage_kind::native:
            break;
        case storage_kind::f8_e4m3: {
            // widening: walk backwards so the in-place expansion is safe
            const uint8_t * src = (const uint8_t *) data;
            qz_fp16_t *   dst = (qz_fp16_t *) data;
            for (int64_t i = n_elements - 1; i >= 0; --i) {
                dst[i] = f8_e4m3_to_f16(src[i]);
            }
            break;
        }
        case storage_kind::f8_e5m2: {
            // f8_e5m2 is a truncated f16: widening, walk backwards
            const uint8_t * src = (const uint8_t *) data;
            qz_fp16_t *   dst = (qz_fp16_t *) data;
            for (int64_t i = n_elements - 1; i >= 0; --i) {
                dst[i] = (qz_fp16_t) ((uint16_t) src[i] << 8);
            }
            break;
        }
        case storage_kind::f64: {
            // narrowing: forward walk is safe in place
            const double * src = (const double *) data;
            float *        dst = (float *) data;
            for (int64_t i = 0; i < n_elements; ++i) {
                dst[i] = (float) src[i];
            }
            break;
        }
        case storage_kind::i64: {
            const int64_t * src = (const int64_t *) data;
            int32_t *       dst = (int32_t *) data;
            for (int64_t i = 0; i < n_elements; ++i) {
                dst[i] = (int32_t) src[i];
            }
            break;
        }
    }
}

} // namespace qz
