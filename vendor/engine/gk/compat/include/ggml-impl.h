// The slice of ggml's old internal header that engine-side helpers actually
// used: the failure macros and the size constants. Everything routes through
// the public API; nothing here exposes gk internals.

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ggml_abort is declared by ggml.h

#ifndef GGML_ABORT
#define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#endif

#ifndef GGML_ASSERT
#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            ggml_abort(__FILE__, __LINE__, "GGML_ASSERT(%s) failed", #x); \
        } \
    } while (0)
#endif

#ifndef GGML_UNUSED
#define GGML_UNUSED(x) (void) (x)
#endif

#ifndef GGML_PAD
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#endif

// Computes nodes [i0, i1) of a graph on `backend` - the piecewise evaluation
// engine-side debug callbacks are built on. The graph struct is opaque out
// here, so the slicing lives with the implementation.
GGML_API enum ggml_status ggml_compat_graph_compute_range(
        struct ggml_backend * backend, struct ggml_cgraph * cgraph, int i0, int i1);

// The leaf list, which the old struct exposed as fields.
GGML_API int                  ggml_compat_graph_n_leafs (struct ggml_cgraph * cgraph);
GGML_API struct ggml_tensor * ggml_compat_graph_leaf    (struct ggml_cgraph * cgraph, int i);
GGML_API void                 ggml_compat_graph_add_leaf(struct ggml_cgraph * cgraph, struct ggml_tensor * t);

#define GGML_LOG_ERROR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define GGML_LOG_WARN(...)  do { fprintf(stderr, __VA_ARGS__); } while (0)
#define GGML_LOG_INFO(...)  do { fprintf(stderr, __VA_ARGS__); } while (0)
#define GGML_LOG_DEBUG(...) do { } while (0)

#ifdef __cplusplus
}
#endif
