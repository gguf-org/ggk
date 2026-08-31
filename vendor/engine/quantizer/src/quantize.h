#pragma once

#include <stdint.h>

#if defined(_WIN32) && defined(GGUF_QUANTIZER_SHARED)
#  ifdef GGUF_QUANTIZER_BUILD
#    define GQZ_API __declspec(dllexport)
#  else
#    define GQZ_API __declspec(dllimport)
#  endif
#elif defined(GGUF_QUANTIZER_SHARED)
#  define GQZ_API __attribute__((visibility("default")))
#else
#  define GQZ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// log levels passed to the optional log callback
enum gqz_log_level {
    GQZ_LOG_INFO = 0,
    GQZ_LOG_WARN = 1,
    GQZ_LOG_ERROR = 2,
};

typedef void (*gqz_log_callback)(enum gqz_log_level level, const char * message, void * user_data);

struct gqz_params {
    const char * input_path;          // -m, GGUF or safetensors (auto-detected by content)
    const char * output_path;         // -o, always GGUF
    const char * default_type;        // --type, e.g. "q4_k", "iq4_xs", "mxfp4" (case-insensitive).
                                      // may be NULL: then only tensors matched by tensor_type_rules
                                      // are converted and everything else is copied unchanged
    const char * tensor_type_rules;   // --tensor-type-rules "pattern=type,pattern=type,..."
                                      // (regex, first match wins); works with or without default_type
                                      // and overrides it for matching tensors.
                                      // at least one of default_type/tensor_type_rules is required
    int          n_threads;           // 0 = auto-detect
    const char * device;              // NULL/"cpu" = CPU, "auto" = best available accelerator,
                                      // or a name shown by gqz_get_device_name()

    gqz_log_callback log_cb;          // optional, NULL = log to stderr
    void *            log_user_data;

    // optional cooperative cancellation: polled between tensors; a non-zero
    // return aborts the run, the partial output file is removed and
    // gqz_quantize() returns 2
    int (*cancel_cb)(void * user_data);
    void * cancel_user_data;
};

GQZ_API struct gqz_params gqz_default_params(void);

// devices compiled into this build; index 0 is always "cpu"
GQZ_API int          gqz_get_device_count(void);
GQZ_API const char * gqz_get_device_name(int index);
GQZ_API const char * gqz_get_device_description(int index);

// Returns 0 on success, 2 when aborted via cancel_cb, 1 on failure.
GQZ_API int gqz_quantize(const struct gqz_params * params);

#ifdef __cplusplus
}
#endif
