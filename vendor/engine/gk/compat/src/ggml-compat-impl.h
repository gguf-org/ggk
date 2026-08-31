// Internal glue for the ggml compatibility layer.
//
// The layer keeps the historical ggml API as its public face and implements
// every function on gk. The two sides share struct layouts by construction -
// ggml_tensor and gk_tensor are field-for-field identical, and the op enums
// carry the same values - so the implementation is casts plus forwarding.
// The static asserts below are what make "by construction" a checked claim
// instead of a hope.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// the private header on purpose: this layer is part of the gk family and
// reaches the arena internals, the narrow-float helpers and the buffer structs
#include "gk_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// logging and failure
// ---------------------------------------------------------------------------

void ggml_compat_log(enum ggml_log_level level, const char * fmt, ...);

#define GGML_LOG_ERROR(...) ggml_compat_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_compat_log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_compat_log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_compat_log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)

// ggml.h defines the public spellings of these in terms of ggml_abort(); inside
// the layer they route to ggml_compat_abort() instead, which carries the
// printf format attribute and the noreturn hint. Undefine first so including
// ggml.h before this header is not a redefinition warning.

#ifdef __GNUC__
__attribute__((format(printf, 3, 4), noreturn))
#endif
void ggml_compat_abort(const char * file, int line, const char * fmt, ...);

#undef GGML_ABORT
#define GGML_ABORT(...) ggml_compat_abort(__FILE__, __LINE__, __VA_ARGS__)

#undef GGML_ASSERT
#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            ggml_compat_abort(__FILE__, __LINE__, "GGML_ASSERT(%s) failed", #x); \
        } \
    } while (0)

#undef GGML_UNUSED
#define GGML_UNUSED(x) (void) (x)

#undef GGML_PAD
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

// ---------------------------------------------------------------------------
// the casts
//
// One macro pair per opaque type. Tensor and graph casts are pointer
// reinterpretation of layout-identical structs; the rest are opaque handles
// that only this layer ever dereferences.
// ---------------------------------------------------------------------------

#define GKT(t)  ((struct gk_tensor  *) (t))
#define GGT(t)  ((struct ggml_tensor *) (t))
#define GKC(c)  ((struct gk_ctx     *) (c))
#define GKG(g)  ((struct gk_cgraph  *) (g))
#define GGG(g)  ((struct ggml_cgraph *) (g))

#define GK_CONST_T(t) ((const struct gk_tensor *) (t))

// gk_status and ggml_status differ by design; map, never cast
enum ggml_status ggml_compat_status(enum gk_status st);

// ---------------------------------------------------------------------------
// the layout contract, checked
// ---------------------------------------------------------------------------

#ifdef __cplusplus
#define GGML_COMPAT_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define GGML_COMPAT_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

GGML_COMPAT_STATIC_ASSERT(sizeof(struct ggml_tensor) == sizeof(struct gk_tensor),
                          "ggml_tensor and gk_tensor must be layout-identical");
GGML_COMPAT_STATIC_ASSERT(offsetof(struct ggml_tensor, op) == offsetof(struct gk_tensor, op),
                          "op field must line up");
GGML_COMPAT_STATIC_ASSERT(offsetof(struct ggml_tensor, src) == offsetof(struct gk_tensor, src),
                          "src array must line up");
GGML_COMPAT_STATIC_ASSERT(offsetof(struct ggml_tensor, name) == offsetof(struct gk_tensor, name),
                          "name field must line up");
GGML_COMPAT_STATIC_ASSERT(GGML_MAX_DIMS == GK_MAX_DIMS && GGML_MAX_SRC == GK_MAX_SRC,
                          "dimension limits must agree");

// the enums travel inside shared structs, so their values are the contract
GGML_COMPAT_STATIC_ASSERT((int) GGML_TYPE_COUNT == (int) GK_TYPE_COUNT &&
                          (int) GGML_TYPE_Q4_K == (int) GK_TYPE_Q4_K &&
                          (int) GGML_TYPE_NVFP4 == (int) GK_TYPE_NVFP4,
                          "type enums must agree");
GGML_COMPAT_STATIC_ASSERT((int) GGML_OP_COUNT == (int) GK_OP_COUNT &&
                          (int) GGML_OP_MUL_MAT == (int) GK_OP_MUL_MAT &&
                          (int) GGML_OP_FLASH_ATTN_EXT == (int) GK_OP_FLASH_ATTN_EXT &&
                          (int) GGML_OP_GLU == (int) GK_OP_GLU &&
                          (int) GGML_OP_DSV4_HC_POST == (int) GK_OP_DSV4_HC_POST,
                          "op enums must agree");
GGML_COMPAT_STATIC_ASSERT((int) GGML_UNARY_OP_COUNT == (int) GK_UNARY_OP_COUNT &&
                          (int) GGML_UNARY_OP_SILU == (int) GK_UNARY_OP_SILU &&
                          (int) GGML_UNARY_OP_XIELU == (int) GK_UNARY_OP_XIELU,
                          "unary enums must agree");
GGML_COMPAT_STATIC_ASSERT((int) GGML_GLU_OP_COUNT == (int) GK_GLU_OP_COUNT &&
                          (int) GGML_GLU_OP_SWIGLU_OAI == (int) GK_GLU_OP_SWIGLU_OAI,
                          "glu enums must agree");
GGML_COMPAT_STATIC_ASSERT((int) GGML_STATUS_SUCCESS == (int) GK_STATUS_SUCCESS,
                          "status enums must agree");

#ifdef __cplusplus
}
#endif
