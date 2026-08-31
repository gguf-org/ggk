#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#  define _FILE_OFFSET_BITS 64
#endif

#include "quantize.h"

#include "device.h"
#include "gguf_io.h"
#include "kernels/qz_quant.h"
#include "safetensors_io.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

gqz_log_callback g_log_cb        = nullptr;
void *           g_log_user_data = nullptr;

void log_msg(gqz_log_level level, const char * fmt, ...) {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (g_log_cb) {
        g_log_cb(level, buf, g_log_user_data);
        return;
    }
    const char * prefix = level == GQZ_LOG_ERROR ? "error:" : level == GQZ_LOG_WARN ? "warning:" : " ";
    if (strlen(buf) >= 7 && buf[0] == '[' && buf[4] == '%' && buf[5] == ']' && buf[6] == ' ') {
        fprintf(stderr, "%.6s %s %s\n", buf, prefix, buf + 7);
    } else {
        fprintf(stderr, "%s %s\n", prefix, buf);
    }
}

#define LOG_INFO(...)  log_msg(GQZ_LOG_INFO, __VA_ARGS__)
#define LOG_WARN(...)  log_msg(GQZ_LOG_WARN, __VA_ARGS__)
#define LOG_ERROR(...) log_msg(GQZ_LOG_ERROR, __VA_ARGS__)

std::string to_lower(const std::string & s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
    return r;
}

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t pos = s.find(sep, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

qz_type parse_type(const std::string & name_in) {
    std::string name = to_lower(trim(name_in));
    if (name.empty()) {
        return QZ_TYPE_COUNT;
    }
    for (int i = 0; i < QZ_TYPE_COUNT; ++i) {
        qz_type    t  = (qz_type) i;
        const char * tn = qz_type_name(t);
        if (tn != nullptr && to_lower(tn) == name) {
            return t;
        }
    }
    return QZ_TYPE_COUNT;
}

// best-effort: can we dequantize (to_float) this source type at all?
bool can_dequantize(qz_type t) {
    float dummy_out[1];
    if (t == QZ_TYPE_F32) {
        return true;
    }
    // probe with a zero-length run: qz_dequantize only dispatches on the type
    return qz_dequantize(t, dummy_out, dummy_out, 0);
}

struct TensorRule {
    std::string pattern_str;
    std::regex  pattern;
    qz_type   type;
};

std::vector<TensorRule> parse_tensor_type_rules(const std::string & rules_str) {
    std::vector<TensorRule> rules;
    for (const std::string & item : split(rules_str, ',')) {
        std::string trimmed = trim(item);
        if (trimmed.empty()) {
            continue;
        }
        size_t eq = trimmed.find('=');
        if (eq == std::string::npos || eq == 0 || eq == trimmed.size() - 1) {
            LOG_WARN("ignoring malformed --tensor-type-rules entry '%s'", trimmed.c_str());
            continue;
        }
        std::string pattern_str = trimmed.substr(0, eq);
        std::string type_str    = trimmed.substr(eq + 1);
        qz_type   type        = parse_type(type_str);
        if (type == QZ_TYPE_COUNT || !qz_is_quantize_target(type)) {
            LOG_WARN("ignoring --tensor-type-rules entry with unknown type '%s'", type_str.c_str());
            continue;
        }
        try {
            TensorRule rule;
            rule.pattern_str = pattern_str;
            rule.pattern     = std::regex(pattern_str);
            rule.type        = type;
            rules.push_back(std::move(rule));
        } catch (const std::regex_error & e) {
            LOG_WARN("ignoring --tensor-type-rules entry with invalid regex '%s': %s", pattern_str.c_str(), e.what());
        }
    }
    return rules;
}

// decide the target type for one tensor; QZ_TYPE_COUNT means "keep original".
// `default_type` may itself be QZ_TYPE_COUNT when --type was not given, in
// which case only rule-matched tensors are converted.
qz_type decide_target_type(const std::string & name, int n_dims, int64_t ne0, qz_type default_type,
                             const std::vector<TensorRule> & rules, bool & explicit_match) {
    explicit_match = false;
    qz_type target = default_type;
    for (const TensorRule & rule : rules) {
        if (std::regex_search(name, rule.pattern)) {
            target         = rule.type;
            explicit_match = true;
            break;
        }
    }

    if (explicit_match) {
        return target;
    }

    if (target == QZ_TYPE_COUNT) {
        return QZ_TYPE_COUNT; // no default type: keep original
    }

    // default (--type) path: only touch >=2D tensors when quantizing to a
    // block-quantized type, mirroring llama.cpp/stable-diffusion.cpp convention
    // of leaving 1D norm/bias vectors at full precision unless explicitly overridden.
    if (qz_is_quantized(target) && (n_dims < 2 || ne0 % qz_blck_size(target) != 0)) {
        return QZ_TYPE_COUNT; // signal: keep original type
    }
    return target;
}

class ConverterDispatch {
public:
    ConverterDispatch(std::unique_ptr<qz::device_converter> converter) : converter_(std::move(converter)) {}

    bool enabled() const {
        return converter_ != nullptr;
    }

    const std::string & name() const {
        static const std::string cpu = "cpu";
        return converter_ ? converter_->name() : cpu;
    }

    // returns true if the conversion was done on the accelerator
    bool convert(const float * src, qz_type dst_type, void * dst, int64_t nrows, int64_t n_per_row) {
        if (!converter_ || dst_type == QZ_TYPE_F32) {
            return false;
        }
        if (!converter_->supports(dst_type)) {
            if (unsupported_types_.insert(dst_type).second) {
                LOG_WARN("device %s has no f32 -> %s conversion kernel; using CPU for that type",
                         converter_->name().c_str(), qz_type_name(dst_type));
            }
            return false;
        }
        std::string error;
        if (!converter_->convert(src, dst_type, dst, nrows, n_per_row, error)) {
            if (failed_types_.insert(dst_type).second) {
                LOG_WARN("device %s failed f32 -> %s conversion (%s); using CPU for that type",
                         converter_->name().c_str(), qz_type_name(dst_type), error.c_str());
            }
            return false;
        }
        return true;
    }

private:
    std::unique_ptr<qz::device_converter> converter_;
    std::set<qz_type>                   unsupported_types_;
    std::set<qz_type>                   failed_types_;
};

// convert a single tensor's raw bytes from src_type to dst_type.
void convert_tensor_data(qz_type src_type, const void * src, qz_type dst_type, void * dst, int64_t nrows,
                         int64_t n_per_row, int n_threads, ConverterDispatch & device, bool & used_device) {
    used_device = false;
    const size_t dst_row_size = qz_row_size(dst_type, n_per_row);
    const size_t dst_bytes    = dst_row_size * nrows;

    if (src_type == dst_type) {
        memcpy(dst, src, dst_bytes);
        return;
    }

    n_threads = std::max(1, n_threads);

    auto parallel_rows = [&](int64_t total_rows, const std::function<void(int64_t, int64_t)> & fn) {
        int nt = n_threads;
        if ((int64_t) nt > total_rows) {
            nt = (int) std::max<int64_t>(1, total_rows);
        }
        const int64_t rows_per_thread = (total_rows + nt - 1) / nt;
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; ++t) {
            const int64_t row0 = t * rows_per_thread;
            const int64_t row1 = std::min(total_rows, row0 + rows_per_thread);
            if (row0 >= row1) {
                break;
            }
            pool.emplace_back(fn, row0, row1);
        }
        for (auto & th : pool) {
            th.join();
        }
    };

    // 1) dequantize the source to f32 (multi-threaded by rows)
    std::vector<float> f32_buf;
    const float *      f32_src;
    if (src_type == QZ_TYPE_F32) {
        f32_src = (const float *) src;
    } else {
        f32_buf.resize(nrows * n_per_row);
        const size_t src_row_size = qz_row_size(src_type, n_per_row);
        parallel_rows(nrows, [&](int64_t row0, int64_t row1) {
            qz_dequantize(src_type, (const char *) src + row0 * src_row_size, f32_buf.data() + row0 * n_per_row,
                          (row1 - row0) * n_per_row);
        });
        f32_src = f32_buf.data();
    }

    // 2) quantize f32 -> dst, on the accelerator when possible
    if (device.convert(f32_src, dst_type, dst, nrows, n_per_row)) {
        used_device = true;
        return;
    }

    // dummy (uniform) importance matrix: needed by a few quant kernels that
    // require a non-null imatrix, harmless no-op weighting for the rest.
    std::vector<float> imatrix(n_per_row, 1.0f);

    parallel_rows(nrows, [&](int64_t row0, int64_t row1) {
        qz_quantize_chunk(dst_type, f32_src, dst, row0 * n_per_row, row1 - row0, n_per_row, imatrix.data());
    });
}

struct FileCloser {
    FILE * f;
    ~FileCloser() {
        if (f) {
            fclose(f);
        }
    }
};

// true if the file starts with the GGUF magic
bool file_has_gguf_magic(const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    uint32_t magic = 0;
    bool     is_gguf = fread(&magic, 1, sizeof(magic), f) == sizeof(magic) && magic == qz::GGUF_MAGIC;
    fclose(f);
    return is_gguf;
}

// resolve --device: "cpu" (default), "auto", exact device name, or an alias
// like "cuda"/"rocm"/"hip"/"gpu" for the first matching accelerator.
bool resolve_device(const char * requested_in, std::string & resolved) {
    std::string requested = requested_in == nullptr || requested_in[0] == '\0' ? "cpu" : to_lower(trim(requested_in));
    if (requested == "hip" || requested == "hipblas" || requested == "rocm") {
        requested = "rocm";
    }

    const auto & devices = qz::device_list();

    if (requested == "cpu") {
        resolved = "cpu";
        return true;
    }
    if (requested == "auto") {
        // first non-cpu device, if any
        resolved = devices.size() > 1 ? devices[1].name : "cpu";
        return true;
    }

    for (const auto & d : devices) {
        if (to_lower(d.name) == requested) {
            resolved = d.name;
            return true;
        }
    }
    // alias without an index ("cuda" -> "cuda0", "gpu" -> first accelerator)
    for (const auto & d : devices) {
        if (d.name == "cpu") {
            continue;
        }
        if (requested == "gpu" || to_lower(d.name).rfind(requested, 0) == 0) {
            resolved = d.name;
            return true;
        }
    }

    LOG_ERROR("device '%s' is not available in this build", requested_in);
    for (const auto & d : devices) {
        LOG_INFO("available device: %-8s %s", d.name.c_str(), d.description.c_str());
    }
    return false;
}

} // namespace

extern "C" struct gqz_params gqz_default_params(void) {
    gqz_params p;
    memset(&p, 0, sizeof(p));
    p.n_threads = 0;
    p.device    = "cpu";
    return p;
}

extern "C" int gqz_get_device_count(void) {
    return (int) qz::device_list().size();
}

extern "C" const char * gqz_get_device_name(int index) {
    const auto & devices = qz::device_list();
    if (index < 0 || (size_t) index >= devices.size()) {
        return "";
    }
    return devices[index].name.c_str();
}

extern "C" const char * gqz_get_device_description(int index) {
    const auto & devices = qz::device_list();
    if (index < 0 || (size_t) index >= devices.size()) {
        return "";
    }
    return devices[index].description.c_str();
}

extern "C" int gqz_quantize(const gqz_params * params) {
    if (params == nullptr || params->input_path == nullptr || params->output_path == nullptr) {
        LOG_ERROR("missing required parameter (input/output)");
        return 1;
    }

    if (strcmp(params->input_path, params->output_path) == 0) {
        LOG_ERROR("input and output paths must be different ('%s')", params->input_path);
        return 1;
    }

    g_log_cb        = params->log_cb;
    g_log_user_data = params->log_user_data;

    const bool has_type  = params->default_type != nullptr && params->default_type[0] != '\0';
    const bool has_rules = params->tensor_type_rules != nullptr && params->tensor_type_rules[0] != '\0';

    if (!has_type && !has_rules) {
        LOG_ERROR("nothing to do: provide --type and/or --tensor-type-rules");
        return 1;
    }

    qz_type default_type = QZ_TYPE_COUNT; // COUNT = no default, keep original types
    if (has_type) {
        default_type = parse_type(params->default_type);
        if (default_type == QZ_TYPE_COUNT || !qz_is_quantize_target(default_type)) {
            LOG_ERROR("invalid --type '%s'", params->default_type);
            return 1;
        }
    }

    std::vector<TensorRule> rules;
    if (has_rules) {
        rules = parse_tensor_type_rules(params->tensor_type_rules);
        if (rules.empty() && !has_type) {
            LOG_ERROR("no usable --tensor-type-rules entries and no --type given");
            return 1;
        }
    }

    std::string device_name;
    if (!resolve_device(params->device, device_name)) {
        return 1;
    }

    std::string       device_error;
    ConverterDispatch device(qz::device_open(device_name, device_error));
    if (!device_error.empty()) {
        LOG_ERROR("failed to initialize device '%s': %s", device_name.c_str(), device_error.c_str());
        return 1;
    }
    if (device.enabled()) {
        LOG_INFO("Using accelerator device %s", device_name.c_str());
    }

    int n_threads = params->n_threads;
    if (n_threads <= 0) {
        n_threads = (int) std::thread::hardware_concurrency();
        if (n_threads <= 0) {
            n_threads = 4;
        }
    }

    qz::gguf_file in;
    std::string   error;
    if (file_has_gguf_magic(params->input_path)) {
        if (!qz::gguf_read_meta(params->input_path, in, error)) {
            LOG_ERROR("failed to load gguf file '%s': %s", params->input_path, error.c_str());
            return 1;
        }
    } else if (qz::safetensors_sniff(params->input_path)) {
        std::vector<std::string> warnings;
        if (!qz::safetensors_read_meta(params->input_path, in, warnings, error)) {
            LOG_ERROR("failed to load safetensors file '%s': %s", params->input_path, error.c_str());
            return 1;
        }
        for (const std::string & w : warnings) {
            LOG_WARN("%s", w.c_str());
        }
        if (in.data_paths.size() > 1) {
            LOG_INFO("Merging multi-part safetensors input (%zu parts, %zu tensors)", in.data_paths.size(),
                     in.tensors.size());
        }
        LOG_INFO("Input is a safetensors file; the output GGUF will contain tensor data only (no KV metadata)");
    } else {
        FILE * probe = fopen(params->input_path, "rb");
        if (probe == nullptr) {
            LOG_ERROR("failed to open input file '%s'", params->input_path);
        } else {
            fclose(probe);
            LOG_ERROR("input file '%s' is neither a GGUF nor a safetensors file", params->input_path);
        }
        return 1;
    }

    // ------------------------------------------------------------------
    // plan: decide the target type of every tensor
    // ------------------------------------------------------------------
    struct TensorPlan {
        const qz::gguf_tensor_info * src;
        qz_type                    dst_type;
    };

    std::vector<TensorPlan> plans;
    plans.reserve(in.tensors.size());

    for (const qz::gguf_tensor_info & t : in.tensors) {
        bool      explicit_match = false;
        qz_type target = decide_target_type(t.name, (int) t.n_dims, t.ne[0], default_type, rules, explicit_match);

        if (target == QZ_TYPE_COUNT) {
            target = t.type; // keep original type
        } else if (explicit_match && qz_is_quantized(target) && t.ne[0] % qz_blck_size(target) != 0) {
            LOG_WARN("tensor '%s': row size %lld not divisible by block size of %s, keeping original type %s",
                     t.name.c_str(), (long long) t.ne[0], qz_type_name(target), qz_type_name(t.type));
            target = t.type;
        } else if (target != t.type && !can_dequantize(t.type)) {
            LOG_WARN("tensor '%s': cannot dequantize source type %s, keeping original type", t.name.c_str(),
                     qz_type_name(t.type));
            target = t.type;
        }

        plans.push_back({ &t, target });
    }

    // ------------------------------------------------------------------
    // output metadata: same KVs, tensors with target types and new offsets
    // ------------------------------------------------------------------
    std::vector<qz::gguf_tensor_info> out_tensors;
    out_tensors.reserve(plans.size());

    uint64_t running_offset = 0;
    for (const TensorPlan & plan : plans) {
        qz::gguf_tensor_info info = *plan.src;
        info.type    = plan.dst_type;
        info.offset  = running_offset;
        info.storage = qz::storage_kind::native; // output data is always written as `type`
        out_tensors.push_back(info);
        running_offset += qz::align_up(out_tensors.back().nbytes(), in.alignment);
    }

    FILE * fout = fopen(params->output_path, "wb");
    if (fout == nullptr) {
        LOG_ERROR("failed to create output file '%s'", params->output_path);
        return 1;
    }
    FileCloser fout_closer{ fout };

    if (!qz::gguf_write_meta(fout, in, out_tensors, error)) {
        LOG_ERROR("failed to write gguf header to '%s': %s", params->output_path, error.c_str());
        return 1;
    }

    std::vector<FILE *> fins(in.data_paths.size(), nullptr);
    struct FinsCloser {
        std::vector<FILE *> & fs;
        ~FinsCloser() {
            for (FILE * f : fs) {
                if (f) {
                    fclose(f);
                }
            }
        }
    } fins_closer{ fins };

    for (size_t i = 0; i < in.data_paths.size(); ++i) {
        if (in.data_paths[i] == params->output_path) {
            LOG_ERROR("output path '%s' is also an input part", params->output_path);
            return 1;
        }
        fins[i] = fopen(in.data_paths[i].c_str(), "rb");
        if (fins[i] == nullptr) {
            LOG_ERROR("failed to reopen input file '%s'", in.data_paths[i].c_str());
            return 1;
        }
    }

    // ------------------------------------------------------------------
    // stream the tensor data
    // ------------------------------------------------------------------
    size_t total_src_bytes = 0;
    size_t total_dst_bytes = 0;
    size_t n_converted     = 0;
    size_t n_copied        = 0;
    size_t n_accelerated   = 0;
    size_t n_processed     = 0;

    std::vector<char> src_buf;
    std::vector<char> dst_buf;

    for (const TensorPlan & plan : plans) {
        if (params->cancel_cb != nullptr && params->cancel_cb(params->cancel_user_data) != 0) {
            fclose(fout);
            fout_closer.f = nullptr;
            remove(params->output_path);
            LOG_INFO("Cancelled after %zu of %zu tensors; removed partial output '%s'", n_processed, plans.size(),
                     params->output_path);
            return 2;
        }

        const qz::gguf_tensor_info & t           = *plan.src;
        const size_t                 src_size    = t.nbytes();          // in memory, as t.type
        const size_t                 stored_size = t.storage_nbytes();  // on disk
        const int64_t                src_offset  = (int64_t) (in.data_offset + t.offset);
        FILE *                       fin         = fins[t.file_idx];

        src_buf.resize(std::max(src_size, stored_size));
        if (!qz::seek64(fin, src_offset)) {
            LOG_ERROR("failed to seek to tensor '%s' (offset %lld) in input file", t.name.c_str(),
                      (long long) src_offset);
            return 1;
        }
        if (!qz::read_exact(fin, src_buf.data(), stored_size)) {
            LOG_ERROR("failed to read tensor '%s' from input file (expected %llu bytes at offset %lld)",
                      t.name.c_str(), (unsigned long long) stored_size, (long long) src_offset);
            return 1;
        }

        // safetensors dtypes with no tensor-type equivalent (f8/f64/i64) are
        // stored in their original encoding; widen/narrow to t.type in place
        qz::convert_storage_in_place(t.storage, src_buf.data(), t.n_elements());

        const int64_t n_per_row = t.ne[0];
        const int64_t nrows     = t.nrows();

        const size_t dst_size = qz_row_size(plan.dst_type, n_per_row) * nrows;
        dst_buf.resize(dst_size);

        bool used_device = false;
        convert_tensor_data(t.type, src_buf.data(), plan.dst_type, dst_buf.data(), nrows, n_per_row, n_threads,
                            device, used_device);
        if (used_device) {
            n_accelerated++;
        }

        if (!qz::write_exact(fout, dst_buf.data(), dst_size)) {
            LOG_ERROR("failed to write tensor '%s' to output file", t.name.c_str());
            return 1;
        }
        if (!qz::write_zero_padding(fout, dst_size, in.alignment)) {
            LOG_ERROR("failed to write tensor padding to output file");
            return 1;
        }

        total_src_bytes += stored_size;
        total_dst_bytes += dst_size;
        if (plan.dst_type != t.type) {
            n_converted++;
        } else {
            n_copied++;
        }

        n_processed++;
        const int progress = (int) (100.0 * n_processed / plans.size());
        // LOG_INFO("[%3d%%] %-60s %10s -> %-10s (%8.2f MB -> %8.2f MB)", progress, plan.name.c_str(),
        // LOG_INFO("[%3d%%] %-35s %5s -> %-5s (%5.2f MB -> %5.2f MB)", progress, t.name.c_str(),
        LOG_INFO("[%3d%%] %-50s %8s -> %-8s (%7.2f MB -> %7.2f MB)", progress, t.name.c_str(),
                 qz_type_name(t.type), qz_type_name(plan.dst_type), stored_size / 1024.0 / 1024.0,
                 dst_size / 1024.0 / 1024.0);
    }

    fflush(fout);

    LOG_INFO("Done: %lld tensors quantized, %lld copied unchanged", (long long) n_converted, (long long) n_copied);
    if (device.enabled()) {
        LOG_INFO("Accelerator: %lld tensors converted on %s; remaining conversions used CPU fallback",
                 (long long) n_accelerated, device.name().c_str());
    }
    LOG_INFO("Total size: %.2f MB -> %.2f MB", total_src_bytes / 1024.0 / 1024.0, total_dst_bytes / 1024.0 / 1024.0);

    return 0;
}
