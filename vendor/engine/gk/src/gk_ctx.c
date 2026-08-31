// The context arena, tensor construction and the shape/stride queries.
//
// A context is a bump allocator and nothing more. Tensors are carved out of it
// in creation order, each preceded by a small object header so the arena can
// be walked without a side index, and the whole arena is dropped at once. No
// tensor is ever individually freed: graph building is a phase with a clear
// end, and tying the lifetime of every struct to that phase removes a whole
// class of ownership questions from the rest of the library.
//
// Two allocations happen per tensor at most - the struct, and optionally the
// data. When a context is created with `no_alloc`, only structs are carved and
// `data` is left null; that is how a graph's memory requirement is measured
// before anything is committed, and it is the mode every backend uses, because
// backends place tensor data in their own buffers.

#include "gk_impl.h"

#include <stdarg.h>

// --------------------------------------------------------------------------
// diagnostics
// --------------------------------------------------------------------------

void gk_logf(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

GK_NORETURN void gk_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
    abort();
}

// --------------------------------------------------------------------------
// context
// --------------------------------------------------------------------------

struct gk_ctx * gk_init(struct gk_init_params params) {
    struct gk_ctx * ctx = (struct gk_ctx *) malloc(sizeof(struct gk_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    // A context with no explicit size still has to hold the object headers of
    // whatever is built in it; round anything smaller up to one alignment unit
    // so the arithmetic below never sees zero.
    const size_t mem_size = params.mem_buffer
        ? params.mem_size
        : GK_MAX(gk_pad_size(params.mem_size, GK_MEM_ALIGN), GK_MEM_ALIGN);

    void * mem_buffer = params.mem_buffer;
    if (mem_buffer == NULL) {
        mem_buffer = malloc(mem_size);
        if (mem_buffer == NULL) {
            free(ctx);
            return NULL;
        }
    }

    ctx->mem_size         = mem_size;
    ctx->mem_buffer       = mem_buffer;
    ctx->mem_buffer_owned = params.mem_buffer == NULL;
    ctx->no_alloc         = params.no_alloc;
    ctx->n_objects        = 0;
    ctx->objects_begin    = NULL;
    ctx->objects_end      = NULL;

    return ctx;
}

void gk_free(struct gk_ctx * ctx) {
    if (ctx == NULL) {
        return;
    }
    if (ctx->mem_buffer_owned) {
        free(ctx->mem_buffer);
    }
    free(ctx);
}

size_t gk_used_mem(const struct gk_ctx * ctx) {
    return ctx->objects_end == NULL
        ? 0
        : ctx->objects_end->offs + ctx->objects_end->size;
}

void gk_set_no_alloc(struct gk_ctx * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

bool gk_get_no_alloc(const struct gk_ctx * ctx) {
    return ctx->no_alloc;
}

void * gk_mem_buffer(const struct gk_ctx * ctx) {
    return ctx->mem_buffer;
}

size_t gk_mem_size(const struct gk_ctx * ctx) {
    return ctx->mem_size;
}

size_t gk_tensor_overhead(void) {
    return GK_OBJECT_SIZE + GK_TENSOR_SIZE;
}

// The arena's object chain doubles as a tensor index: walk it, skipping the
// graphs and work buffers interleaved with the tensors.
struct gk_tensor * gk_get_first_tensor(const struct gk_ctx * ctx) {
    for (struct gk_object * obj = ctx->objects_begin; obj != NULL; obj = obj->next) {
        if (obj->type == GK_OBJECT_TYPE_TENSOR) {
            return (struct gk_tensor *) ((char *) ctx->mem_buffer + obj->offs);
        }
    }
    return NULL;
}

struct gk_tensor * gk_get_next_tensor(const struct gk_ctx * ctx, struct gk_tensor * t) {
    // every payload sits right after its header, so the header is found by
    // stepping back rather than searching the chain
    struct gk_object * obj = (struct gk_object *) ((char *) t - GK_OBJECT_SIZE);

    for (obj = obj->next; obj != NULL; obj = obj->next) {
        if (obj->type == GK_OBJECT_TYPE_TENSOR) {
            return (struct gk_tensor *) ((char *) ctx->mem_buffer + obj->offs);
        }
    }
    return NULL;
}

struct gk_tensor * gk_get_tensor(struct gk_ctx * ctx, const char * name) {
    for (struct gk_tensor * t = gk_get_first_tensor(ctx); t != NULL;
         t = gk_get_next_tensor(ctx, t)) {
        if (strcmp(t->name, name) == 0) {
            return t;
        }
    }
    return NULL;
}

// Carves `size` bytes plus a header off the end of the arena.
struct gk_object * gk_new_object(struct gk_ctx * ctx, enum gk_object_type type, size_t size) {
    struct gk_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;

    const size_t size_needed = gk_pad_size(size, GK_MEM_ALIGN);

    char * const mem_buffer = (char *) ctx->mem_buffer;
    struct gk_object * const obj_new =
        (struct gk_object *) (mem_buffer + cur_offs + cur_size);

    if (cur_offs + cur_size + GK_OBJECT_SIZE + size_needed > ctx->mem_size) {
        gk_logf("gk: context is out of space: needed %zu more bytes, have %zu of %zu used\n",
               GK_OBJECT_SIZE + size_needed, cur_offs + cur_size, ctx->mem_size);
        return NULL;
    }

    *obj_new = (struct gk_object) {
        .offs = cur_offs + cur_size + GK_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;
    ctx->n_objects++;

    return obj_new;
}

// --------------------------------------------------------------------------
// tensor construction
// --------------------------------------------------------------------------

// The one place a tensor struct is produced. `view_src` non-null means the
// data lives in another tensor and no storage is carved; otherwise storage is
// carved unless the context is in no_alloc mode.
static struct gk_tensor * gk_new_tensor_impl(
        struct gk_ctx    * ctx,
        enum gk_type       type,
        int                n_dims,
        const int64_t    * ne,
        struct gk_tensor * view_src,
        size_t             view_offs) {

    GK_ASSERT(type >= 0 && type < GK_TYPE_COUNT);
    GK_ASSERT(n_dims >= 1 && n_dims <= GK_MAX_DIMS);

    // A view of a view resolves to the tensor that actually owns the memory,
    // so view chains never get longer than one link and every kernel can reach
    // the owning buffer in a single hop.
    if (view_src != NULL && view_src->view_src != NULL) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    size_t data_size = gk_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; ++i) {
        data_size *= ne[i];
    }

    GK_ASSERT(view_src == NULL || data_size == 0 ||
              data_size + view_offs <= gk_nbytes(view_src));

    void * data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char *) data + view_offs;
    }

    const size_t obj_alloc_size =
        (view_src == NULL && !ctx->no_alloc) ? data_size : 0;

    struct gk_object * const obj_new =
        gk_new_object(ctx, GK_OBJECT_TYPE_TENSOR, GK_TENSOR_SIZE + obj_alloc_size);
    if (obj_new == NULL) {
        return NULL;
    }

    struct gk_tensor * const result =
        (struct gk_tensor *) ((char *) ctx->mem_buffer + obj_new->offs);

    *result = (struct gk_tensor) {
        .type      = type,
        .buffer    = NULL,
        .ne        = { 1, 1, 1, 1 },
        .nb        = { 0, 0, 0, 0 },
        .op        = GK_OP_NONE,
        .op_params = { 0 },
        .flags     = 0,
        .src       = { NULL },
        .view_src  = view_src,
        .view_offs = view_offs,
        .data      = obj_alloc_size > 0 ? (void *) (result + 1) : data,
        .name      = { 0 },
        .extra     = NULL,
        .padding   = { 0 },
    };

    for (int i = 0; i < n_dims; ++i) {
        result->ne[i] = ne[i];
    }

    // Default strides describe a contiguous tensor. nb[0] is the size of one
    // block, so for a quantized type a single "element" step is meaningless
    // and only whole rows are addressable - which is exactly the constraint
    // the formats impose anyway.
    result->nb[0] = gk_type_size(type);
    result->nb[1] = result->nb[0] * (result->ne[0] / gk_blck_size(type));
    for (int i = 2; i < GK_MAX_DIMS; ++i) {
        result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
    }

    return result;
}

struct gk_tensor * gk_new_tensor(
        struct gk_ctx * ctx, enum gk_type type, int n_dims, const int64_t * ne) {
    return gk_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

struct gk_tensor * gk_new_tensor_1d(struct gk_ctx * ctx, enum gk_type type, int64_t ne0) {
    return gk_new_tensor(ctx, type, 1, &ne0);
}

struct gk_tensor * gk_new_tensor_2d(struct gk_ctx * ctx, enum gk_type type,
                                    int64_t ne0, int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    return gk_new_tensor(ctx, type, 2, ne);
}

struct gk_tensor * gk_new_tensor_3d(struct gk_ctx * ctx, enum gk_type type,
                                    int64_t ne0, int64_t ne1, int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    return gk_new_tensor(ctx, type, 3, ne);
}

struct gk_tensor * gk_new_tensor_4d(struct gk_ctx * ctx, enum gk_type type,
                                    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return gk_new_tensor(ctx, type, 4, ne);
}

struct gk_tensor * gk_new_tensor_view(
        struct gk_ctx * ctx, enum gk_type type, int n_dims, const int64_t * ne,
        struct gk_tensor * src, size_t offset) {
    return gk_new_tensor_impl(ctx, type, n_dims, ne, src, offset);
}

struct gk_tensor * gk_dup_tensor(struct gk_ctx * ctx, const struct gk_tensor * src) {
    return gk_new_tensor(ctx, src->type, GK_MAX_DIMS, src->ne);
}

struct gk_tensor * gk_view_tensor(struct gk_ctx * ctx, struct gk_tensor * src) {
    struct gk_tensor * result =
        gk_new_tensor_impl(ctx, src->type, GK_MAX_DIMS, src->ne, src, 0);
    gk_format_name(result, "%s (view)", src->name);

    // a plain view keeps the parent's layout, including any permutation
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        result->nb[i] = src->nb[i];
    }

    return result;
}

struct gk_tensor * gk_new_i32(struct gk_ctx * ctx, int32_t value) {
    GK_ASSERT(!ctx->no_alloc);
    struct gk_tensor * result = gk_new_tensor_1d(ctx, GK_TYPE_I32, 1);
    *(int32_t *) result->data = value;
    return result;
}

struct gk_tensor * gk_new_f32(struct gk_ctx * ctx, float value) {
    GK_ASSERT(!ctx->no_alloc);
    struct gk_tensor * result = gk_new_tensor_1d(ctx, GK_TYPE_F32, 1);
    *(float *) result->data = value;
    return result;
}

// --------------------------------------------------------------------------
// shape queries
// --------------------------------------------------------------------------

int64_t gk_nelements(const struct gk_tensor * t) {
    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

int64_t gk_nrows(const struct gk_tensor * t) {
    return t->ne[1] * t->ne[2] * t->ne[3];
}

// Walks the strides rather than assuming a contiguous layout, so a view's
// footprint is reported as the span it actually covers.
size_t gk_nbytes(const struct gk_tensor * t) {
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        if (t->ne[i] <= 0) {
            return 0;
        }
    }

    const int64_t blck = gk_blck_size(t->type);

    if (blck == 1) {
        size_t nbytes = gk_type_size(t->type);
        for (int i = 0; i < GK_MAX_DIMS; ++i) {
            nbytes += (size_t) (t->ne[i] - 1) * t->nb[i];
        }
        return nbytes;
    }

    // For a block type the first dimension is counted in whole blocks.
    size_t nbytes = (size_t) (t->ne[0] / blck) * t->nb[0];
    for (int i = 1; i < GK_MAX_DIMS; ++i) {
        nbytes += (size_t) (t->ne[i] - 1) * t->nb[i];
    }
    return nbytes;
}

int gk_n_dims(const struct gk_tensor * t) {
    for (int i = GK_MAX_DIMS - 1; i >= 1; --i) {
        if (t->ne[i] > 1) {
            return i + 1;
        }
    }
    return 1;
}

size_t gk_element_size(const struct gk_tensor * t) {
    return gk_type_size(t->type);
}

bool gk_is_scalar(const struct gk_tensor * t) {
    return t->ne[0] == 1 && t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1;
}

bool gk_is_vector(const struct gk_tensor * t) {
    return t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1;
}

bool gk_is_matrix(const struct gk_tensor * t) {
    return t->ne[2] == 1 && t->ne[3] == 1;
}

bool gk_is_empty(const struct gk_tensor * t) {
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        if (t->ne[i] == 0) {
            return true;
        }
    }
    return false;
}

// `n` is the first dimension whose stride is allowed to be loose; dimensions
// above it must still pack tightly. n=0 is the fully contiguous case.
static bool gk_is_contiguous_n(const struct gk_tensor * t, int n) {
    const int64_t blck = gk_blck_size(t->type);

    if (t->ne[0] % blck != 0) {
        return false;
    }

    size_t next = gk_type_size(t->type);
    if (t->nb[0] != next) {
        return false;
    }

    next *= (size_t) (t->ne[0] / blck);
    for (int i = 1; i < GK_MAX_DIMS; ++i) {
        if (t->ne[i] != 1) {
            if (i > n) {
                if (t->nb[i] != next) {
                    return false;
                }
            } else if (t->nb[i] < next) {
                // below the cut a stride may be padded out, but never
                // overlapped
                return false;
            }
        }
        next = t->nb[i] * (size_t) t->ne[i];
    }

    return true;
}

bool gk_is_contiguous(const struct gk_tensor * t) {
    return gk_is_contiguous_n(t, 0);
}

bool gk_is_contiguous_1(const struct gk_tensor * t) {
    return gk_is_contiguous_n(t, 1);
}

bool gk_is_contiguous_2(const struct gk_tensor * t) {
    return gk_is_contiguous_n(t, 2);
}

// Elements within a row are packed and the higher dimensions are packed, but
// nb[1] is free - so a row may be followed by padding. This is the weakest
// layout an elementwise kernel can still walk with a single stride per row.
bool gk_is_padded_1(const struct gk_tensor * t) {
    return t->nb[0] == gk_type_size(t->type)
        && t->nb[2] == t->nb[1] * (size_t) t->ne[1]
        && t->nb[3] == t->nb[2] * (size_t) t->ne[2];
}

bool gk_is_transposed(const struct gk_tensor * t) {
    return t->nb[0] > t->nb[1];
}

bool gk_is_permuted(const struct gk_tensor * t) {
    return t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

bool gk_are_same_shape(const struct gk_tensor * a, const struct gk_tensor * b) {
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }
    return true;
}

bool gk_are_same_stride(const struct gk_tensor * a, const struct gk_tensor * b) {
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

bool gk_can_repeat(const struct gk_tensor * a, const struct gk_tensor * b) {
    // An empty operand repeats onto an empty result and onto nothing else. It
    // has to be answered before the loop, which would divide by a zero extent.
    // Graphs really do carry empty tensors - a vision encoder run with no
    // images, a batch whose sequence contributes no tokens - and the ops that
    // consume them are expected to become no-ops rather than to fail.
    if (gk_is_empty(a)) {
        return gk_is_empty(b);
    }
    // `b` may itself be empty here: a zero extent divides by anything, so a
    // normal weight broadcasts onto an empty result and the op drops out at
    // compute time. Only an empty *source* is special, hence the test above.
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        if (b->ne[i] % a->ne[i] != 0) {
            return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------------
// names, flags and op params
// --------------------------------------------------------------------------

const char * gk_get_name(const struct gk_tensor * t) {
    return t->name;
}

struct gk_tensor * gk_set_name(struct gk_tensor * t, const char * name) {
    size_t i;
    for (i = 0; i < sizeof(t->name) - 1 && name[i] != '\0'; ++i) {
        t->name[i] = name[i];
    }
    t->name[i] = '\0';
    return t;
}

struct gk_tensor * gk_format_name(struct gk_tensor * t, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(t->name, sizeof(t->name), fmt, args);
    va_end(args);
    return t;
}

void gk_set_input(struct gk_tensor * t) {
    t->flags |= GK_TENSOR_FLAG_INPUT;
}

void gk_set_output(struct gk_tensor * t) {
    t->flags |= GK_TENSOR_FLAG_OUTPUT;
}

void gk_set_op_params(struct gk_tensor * t, const void * params, size_t size) {
    GK_ASSERT(size <= GK_MAX_OP_PARAMS);
    memcpy(t->op_params, params, size);
}

int32_t gk_get_op_params_i32(const struct gk_tensor * t, int i) {
    GK_ASSERT((size_t) i < GK_MAX_OP_PARAMS / sizeof(int32_t));
    return t->op_params[i];
}

float gk_get_op_params_f32(const struct gk_tensor * t, int i) {
    GK_ASSERT((size_t) i < GK_MAX_OP_PARAMS / sizeof(float));
    float v;
    memcpy(&v, &t->op_params[i], sizeof(float));
    return v;
}

void gk_set_op_params_i32(struct gk_tensor * t, int i, int32_t value) {
    GK_ASSERT((size_t) i < GK_MAX_OP_PARAMS / sizeof(int32_t));
    t->op_params[i] = value;
}

void gk_set_op_params_f32(struct gk_tensor * t, int i, float value) {
    GK_ASSERT((size_t) i < GK_MAX_OP_PARAMS / sizeof(float));
    memcpy(&t->op_params[i], &value, sizeof(float));
}

enum gk_unary_op gk_get_unary_op(const struct gk_tensor * t) {
    GK_ASSERT(t->op == GK_OP_UNARY);
    return (enum gk_unary_op) gk_get_op_params_i32(t, 0);
}

enum gk_glu_op gk_get_glu_op(const struct gk_tensor * t) {
    GK_ASSERT(t->op == GK_OP_GLU);
    return (enum gk_glu_op) gk_get_op_params_i32(t, 0);
}
