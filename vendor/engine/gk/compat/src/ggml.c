// The ggml compatibility layer: the historical API, implemented on gk.
//
// Almost everything here forwards. The functions that do more than cast fall
// into three groups: queries ggml exposed that gk keeps internal (contiguity
// variants, index unravelling), behaviours tied to ggml's context layout
// (ggml_new_buffer, max tensor size), and the quantisation entry points,
// which route to the shared qz codec so this layer and the quantizer can
// never disagree about bytes.

#include "ggml-compat-impl.h"

#include "qz_quant.h"

#include <math.h>
#include <stdarg.h>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <time.h>
#endif

// ---------------------------------------------------------------------------
// logging, failure, time
// ---------------------------------------------------------------------------

static ggml_log_callback g_log_callback = NULL;
static void *            g_log_userdata = NULL;
#if defined(_WIN32)
static LARGE_INTEGER      g_qpc_frequency = { 0 };
#endif

void ggml_log_set(ggml_log_callback cb, void * user_data) {
    g_log_callback = cb;
    g_log_userdata = user_data;
}

void ggml_log_get(ggml_log_callback * cb, void ** user_data) {
    if (cb != NULL) {
        *cb = g_log_callback;
    }
    if (user_data != NULL) {
        *user_data = g_log_userdata;
    }
}

void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    fputs(text, stderr);
    fflush(stderr);
}

void ggml_compat_log(enum ggml_log_level level, const char * fmt, ...) {
    char buf[1024];

    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (g_log_callback != NULL) {
        g_log_callback(level, buf, g_log_userdata);
    } else {
        fputs(buf, stderr);
        fflush(stderr);
    }
}

// the public entry the engine's own GGML_ABORT macro expands to
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

void ggml_compat_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

const char * ggml_version(void) { return "gk-compat"; }
const char * ggml_commit (void) { return "unknown"; }

void ggml_time_init(void) {
#if defined(_WIN32)
    // QueryPerformanceCounter is the Windows equivalent of the POSIX
    // CLOCK_MONOTONIC clock.  The frequency is fixed for the lifetime of the
    // process and is cached so the hot timing path only reads the counter.
    QueryPerformanceFrequency(&g_qpc_frequency);
#else
    // clock_gettime needs no setup; the call stays for API compatibility
#endif
}

int64_t ggml_time_us(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;

    if (g_qpc_frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&g_qpc_frequency);
    }
    QueryPerformanceCounter(&counter);

    const int64_t seconds = counter.QuadPart / g_qpc_frequency.QuadPart;
    const int64_t remainder = counter.QuadPart % g_qpc_frequency.QuadPart;
    return seconds * 1000000 + remainder * 1000000 / g_qpc_frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

int64_t ggml_time_ms(void) {
    return ggml_time_us() / 1000;
}

int64_t ggml_cycles(void) {
    return ggml_time_us();
}

int64_t ggml_cycles_per_ms(void) {
    return 1000;
}

FILE * ggml_fopen(const char * fname, const char * mode) {
    // on POSIX a path is already UTF-8; the Windows build would need the
    // wide-char conversion ggml carried
    return fopen(fname, mode);
}

bool ggml_guid_matches(ggml_guid_t a, ggml_guid_t b) {
    return memcmp(a, b, sizeof(ggml_guid)) == 0;
}

// the status enums deliberately differ (gk's are its own); map, never cast
enum ggml_status ggml_compat_status(enum gk_status st);
enum ggml_status ggml_compat_status(enum gk_status st) {
    switch (st) {
        case GK_STATUS_SUCCESS:      return GGML_STATUS_SUCCESS;
        case GK_STATUS_ALLOC_FAILED: return GGML_STATUS_ALLOC_FAILED;
        default:                     return GGML_STATUS_FAILED;
    }
}

const char * ggml_status_to_string(enum ggml_status status) {
    switch (status) {
        case GGML_STATUS_ALLOC_FAILED: return "GGML status: error (failed to allocate memory)";
        case GGML_STATUS_FAILED:       return "GGML status: error (operation failed)";
        case GGML_STATUS_SUCCESS:      return "GGML status: success";
        case GGML_STATUS_ABORTED:      return "GGML status: warning (operation aborted)";
    }
    return "GGML status: unknown";
}

// ---------------------------------------------------------------------------
// type and tensor queries
// ---------------------------------------------------------------------------

int64_t ggml_blck_size(enum ggml_type type)              { return gk_blck_size((enum gk_type) type); }
size_t  ggml_type_size(enum ggml_type type)              { return gk_type_size((enum gk_type) type); }
size_t  ggml_row_size (enum ggml_type type, int64_t ne)  { return gk_row_size((enum gk_type) type, ne); }
bool    ggml_is_quantized(enum ggml_type type)           { return gk_is_quantized((enum gk_type) type); }
const char * ggml_type_name(enum ggml_type type)         { return gk_type_name((enum gk_type) type); }

double ggml_type_sizef(enum ggml_type type) {
    return ((double) gk_type_size((enum gk_type) type)) / gk_blck_size((enum gk_type) type);
}

const char * ggml_op_name  (enum ggml_op op) { return gk_op_name  ((enum gk_op) op); }
const char * ggml_op_symbol(enum ggml_op op) { return gk_op_symbol((enum gk_op) op); }
const char * ggml_unary_op_name(enum ggml_unary_op op) { return gk_unary_op_name((enum gk_unary_op) op); }
const char * ggml_glu_op_name  (enum ggml_glu_op op)   { return gk_glu_op_name((enum gk_glu_op) op); }

const char * ggml_op_desc(const struct ggml_tensor * t) {
    if (t->op == GGML_OP_UNARY) {
        return gk_unary_op_name(gk_get_unary_op(GK_CONST_T(t)));
    }
    if (t->op == GGML_OP_GLU) {
        return gk_glu_op_name(gk_get_glu_op(GK_CONST_T(t)));
    }
    return ggml_op_name(t->op);
}

int64_t ggml_nelements(const struct ggml_tensor * t) { return gk_nelements(GK_CONST_T(t)); }
int64_t ggml_nrows    (const struct ggml_tensor * t) { return gk_nrows(GK_CONST_T(t)); }
size_t  ggml_nbytes   (const struct ggml_tensor * t) { return gk_nbytes(GK_CONST_T(t)); }
int     ggml_n_dims   (const struct ggml_tensor * t) { return gk_n_dims(GK_CONST_T(t)); }
size_t  ggml_element_size(const struct ggml_tensor * t) { return gk_element_size(GK_CONST_T(t)); }

size_t ggml_nbytes_pad(const struct ggml_tensor * t) {
    return GGML_PAD(ggml_nbytes(t), GGML_MEM_ALIGN);
}

bool ggml_is_transposed(const struct ggml_tensor * t) { return gk_is_transposed(GK_CONST_T(t)); }
bool ggml_is_permuted  (const struct ggml_tensor * t) { return gk_is_permuted(GK_CONST_T(t)); }
bool ggml_is_empty     (const struct ggml_tensor * t) { return gk_is_empty(GK_CONST_T(t)); }
bool ggml_is_scalar    (const struct ggml_tensor * t) { return gk_is_scalar(GK_CONST_T(t)); }
bool ggml_is_vector    (const struct ggml_tensor * t) { return gk_is_vector(GK_CONST_T(t)); }
bool ggml_is_matrix    (const struct ggml_tensor * t) { return gk_is_matrix(GK_CONST_T(t)); }

bool ggml_is_3d(const struct ggml_tensor * t) { return t->ne[3] == 1; }

bool ggml_is_view(const struct ggml_tensor * t) { return t->view_src != NULL; }

bool ggml_is_contiguous  (const struct ggml_tensor * t) { return gk_is_contiguous(GK_CONST_T(t)); }
bool ggml_is_contiguous_0(const struct ggml_tensor * t) { return gk_is_contiguous(GK_CONST_T(t)); }
bool ggml_is_contiguous_1(const struct ggml_tensor * t) { return gk_is_contiguous_1(GK_CONST_T(t)); }
bool ggml_is_contiguous_2(const struct ggml_tensor * t) { return gk_is_contiguous_2(GK_CONST_T(t)); }

// contiguous below dimension d: the strides up to d are the packed ones
static bool ggml_compat_contig_below(const struct ggml_tensor * t, int d) {
    size_t expect = ggml_type_size(t->type);
    for (int i = 0; i < d; ++i) {
        if (i == 0) {
            if (t->nb[0] != expect) {
                return false;
            }
            expect = expect * (t->ne[0] / ggml_blck_size(t->type));
        } else {
            if (t->nb[i] != expect) {
                return false;
            }
            expect *= t->ne[i];
        }
    }
    return true;
}

bool ggml_is_contiguous_to_1(const struct ggml_tensor * t) { return ggml_compat_contig_below(t, 1); }
bool ggml_is_contiguous_to_2(const struct ggml_tensor * t) { return ggml_compat_contig_below(t, 2); }
bool ggml_is_contiguous_to_3(const struct ggml_tensor * t) { return ggml_compat_contig_below(t, 3); }

bool ggml_is_contiguously_allocated(const struct ggml_tensor * t) {
    return ggml_nbytes(t) == (size_t) (ggml_nelements(t) * ggml_type_size(t->type)
                                        / ggml_blck_size(t->type));
}

bool ggml_is_contiguous_channels(const struct ggml_tensor * t) {
    return t->nb[0] > t->nb[2] && t->nb[1] > t->nb[0] &&
           t->nb[2] == ggml_type_size(t->type);
}

bool ggml_is_contiguous_rows(const struct ggml_tensor * t) {
    return t->ne[0] == 0 || t->nb[0] == ggml_type_size(t->type);
}

bool ggml_are_same_shape(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return gk_are_same_shape(GK_CONST_T(a), GK_CONST_T(b));
}

bool ggml_are_same_stride(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return gk_are_same_stride(GK_CONST_T(a), GK_CONST_T(b));
}

bool ggml_can_repeat(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return gk_can_repeat(GK_CONST_T(a), GK_CONST_T(b));
}

void ggml_unravel_index(const struct ggml_tensor * t, int64_t i,
                        int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2];

    if (i3 != NULL) { *i3 = i / (ne2 * ne1 * ne0); }
    if (i2 != NULL) { *i2 = (i / (ne1 * ne0)) % ne2; }
    if (i1 != NULL) { *i1 = (i / ne0) % ne1; }
    if (i0 != NULL) { *i0 = i % ne0; }
}

enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype) {
    switch (ftype) {
        case GGML_FTYPE_ALL_F32:        return GGML_TYPE_F32;
        case GGML_FTYPE_MOSTLY_F16:     return GGML_TYPE_F16;
        case GGML_FTYPE_MOSTLY_BF16:    return GGML_TYPE_BF16;
        case GGML_FTYPE_MOSTLY_Q4_0:    return GGML_TYPE_Q4_0;
        case GGML_FTYPE_MOSTLY_Q4_1:    return GGML_TYPE_Q4_1;
        case GGML_FTYPE_MOSTLY_Q8_0:    return GGML_TYPE_Q8_0;
        case GGML_FTYPE_MOSTLY_Q5_0:    return GGML_TYPE_Q5_0;
        case GGML_FTYPE_MOSTLY_Q5_1:    return GGML_TYPE_Q5_1;
        case GGML_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case GGML_FTYPE_MOSTLY_Q3_K:    return GGML_TYPE_Q3_K;
        case GGML_FTYPE_MOSTLY_Q4_K:    return GGML_TYPE_Q4_K;
        case GGML_FTYPE_MOSTLY_Q5_K:    return GGML_TYPE_Q5_K;
        case GGML_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case GGML_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case GGML_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case GGML_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case GGML_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case GGML_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case GGML_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case GGML_FTYPE_MOSTLY_IQ3_S:   return GGML_TYPE_IQ3_S;
        case GGML_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_S;
        case GGML_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case GGML_FTYPE_MOSTLY_MXFP4:   return GGML_TYPE_MXFP4;
        case GGML_FTYPE_MOSTLY_NVFP4:   return GGML_TYPE_NVFP4;
        case GGML_FTYPE_MOSTLY_Q1_0:    return GGML_TYPE_Q1_0;
        case GGML_FTYPE_MOSTLY_Q2_0:    return GGML_TYPE_Q2_0;
        default: GGML_ABORT("unknown ftype %d", (int) ftype);
    }
}

size_t ggml_tensor_overhead(void) {
    return gk_tensor_overhead();
}

// ---------------------------------------------------------------------------
// narrow floats
// ---------------------------------------------------------------------------

float       ggml_fp16_to_fp32(ggml_fp16_t x) { return gk_fp16_to_fp32(x); }
ggml_fp16_t ggml_fp32_to_fp16(float x)       { return gk_fp32_to_fp16(x); }

void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n) {
    gk_fp16_to_fp32_row(x, y, n);
}

void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
    gk_fp32_to_fp16_row(x, y, n);
}

float ggml_bf16_to_fp32(ggml_bf16_t x) {
    gk_bf16_t b = { x.bits };
    return gk_bf16_to_fp32(b);
}

ggml_bf16_t ggml_fp32_to_bf16(float x) {
    const gk_bf16_t b = gk_fp32_to_bf16(x);
    ggml_bf16_t r = { b.bits };
    return r;
}

void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n) {
    gk_bf16_to_fp32_row((const gk_bf16_t *) x, y, n);
}

void ggml_fp32_to_bf16_row(const float * x, ggml_bf16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = ggml_fp32_to_bf16(x[i]);
    }
}

void ggml_fp32_to_bf16_row_ref(const float * x, ggml_bf16_t * y, int64_t n) {
    ggml_fp32_to_bf16_row(x, y, n);
}

// ---------------------------------------------------------------------------
// type traits
// ---------------------------------------------------------------------------

const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type) {
    GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT);

    // built lazily from gk's traits; the shapes of the two structs differ
    static struct ggml_type_traits cache[GGML_TYPE_COUNT];
    static bool filled[GGML_TYPE_COUNT];

    if (!filled[type]) {
        const struct gk_type_traits * tr = gk_get_type_traits((enum gk_type) type);
        cache[type].type_name             = tr->name;
        cache[type].blck_size             = tr->blck_size;
        cache[type].blck_size_interleave  = 0;
        cache[type].type_size             = tr->type_size;
        cache[type].is_quantized          = tr->is_quantized;
        cache[type].to_float              = (ggml_to_float_t) tr->to_float;
        cache[type].from_float_ref        = (ggml_from_float_t) tr->from_float;
        filled[type] = true;
    }

    return &cache[type];
}

// ---------------------------------------------------------------------------
// context
// ---------------------------------------------------------------------------

struct ggml_context * ggml_init(struct ggml_init_params params) {
    struct gk_init_params p = {
        .mem_size   = params.mem_size,
        .mem_buffer = params.mem_buffer,
        .no_alloc   = params.no_alloc,
    };
    return (struct ggml_context *) gk_init(p);
}

void ggml_free(struct ggml_context * ctx) {
    gk_free(GKC(ctx));
}

size_t ggml_used_mem(const struct ggml_context * ctx) {
    return gk_used_mem((const struct gk_ctx *) ctx);
}

bool ggml_get_no_alloc(struct ggml_context * ctx) {
    return gk_get_no_alloc(GKC(ctx));
}

void ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc) {
    gk_set_no_alloc(GKC(ctx), no_alloc);
}

void * ggml_get_mem_buffer(const struct ggml_context * ctx) {
    return gk_mem_buffer((const struct gk_ctx *) ctx);
}

size_t ggml_get_mem_size(const struct ggml_context * ctx) {
    return gk_mem_size((const struct gk_ctx *) ctx);
}

size_t ggml_get_max_tensor_size(const struct ggml_context * ctx) {
    size_t max = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
         t = ggml_get_next_tensor(ctx, t)) {
        const size_t n = ggml_nbytes(t);
        if (n > max) {
            max = n;
        }
    }
    return max;
}

struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx) {
    return GGT(gk_get_first_tensor((const struct gk_ctx *) ctx));
}

struct ggml_tensor * ggml_get_next_tensor(const struct ggml_context * ctx, struct ggml_tensor * t) {
    return GGT(gk_get_next_tensor((const struct gk_ctx *) ctx, GKT(t)));
}

struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name) {
    return GGT(gk_get_tensor(GKC(ctx), name));
}

// ---------------------------------------------------------------------------
// tensor construction
// ---------------------------------------------------------------------------

struct ggml_tensor * ggml_new_tensor(struct ggml_context * ctx, enum ggml_type type,
                                     int n_dims, const int64_t * ne) {
    return GGT(gk_new_tensor(GKC(ctx), (enum gk_type) type, n_dims, ne));
}

struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0) {
    return GGT(gk_new_tensor_1d(GKC(ctx), (enum gk_type) type, ne0));
}

struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context * ctx, enum ggml_type type,
                                        int64_t ne0, int64_t ne1) {
    return GGT(gk_new_tensor_2d(GKC(ctx), (enum gk_type) type, ne0, ne1));
}

struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context * ctx, enum ggml_type type,
                                        int64_t ne0, int64_t ne1, int64_t ne2) {
    return GGT(gk_new_tensor_3d(GKC(ctx), (enum gk_type) type, ne0, ne1, ne2));
}

struct ggml_tensor * ggml_new_tensor_4d(struct ggml_context * ctx, enum ggml_type type,
                                        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return GGT(gk_new_tensor_4d(GKC(ctx), (enum gk_type) type, ne0, ne1, ne2, ne3));
}

void * ggml_new_buffer(struct ggml_context * ctx, size_t nbytes) {
    struct gk_object * obj = gk_new_object(GKC(ctx), GK_OBJECT_TYPE_WORK_BUFFER, nbytes);
    GGML_ASSERT(obj != NULL);
    return (char *) gk_mem_buffer(GKC(ctx)) + obj->offs;
}

struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src) {
    return GGT(gk_dup_tensor(GKC(ctx), GK_CONST_T(src)));
}

struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src) {
    return GGT(gk_view_tensor(GKC(ctx), GKT(src)));
}

const char * ggml_get_name(const struct ggml_tensor * t) {
    return t->name;
}

struct ggml_tensor * ggml_set_name(struct ggml_tensor * t, const char * name) {
    return GGT(gk_set_name(GKT(t), name));
}

struct ggml_tensor * ggml_format_name(struct ggml_tensor * t, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(t->name, sizeof(t->name), fmt, args);
    va_end(args);
    return t;
}

void ggml_set_input (struct ggml_tensor * t) { gk_set_input(GKT(t)); }
void ggml_set_output(struct ggml_tensor * t) { gk_set_output(GKT(t)); }

void ggml_set_param(struct ggml_tensor * t) {
    GGML_ASSERT(t->op == GGML_OP_NONE);
    t->flags |= GGML_TENSOR_FLAG_PARAM;
}

void ggml_set_loss(struct ggml_tensor * t) {
    t->flags |= GGML_TENSOR_FLAG_LOSS;
}

void * ggml_get_data(const struct ggml_tensor * t) {
    return t->data;
}

float * ggml_get_data_f32(const struct ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    return (float *) t->data;
}

struct ggml_tensor * ggml_set_zero(struct ggml_tensor * t) {
    if (ggml_is_empty(t)) {
        return t;
    }
    if (t->buffer != NULL) {
        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    } else {
        GGML_ASSERT(t->data != NULL);
        memset(t->data, 0, ggml_nbytes(t));
    }
    return t;
}

enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * t) {
    return (enum ggml_unary_op) gk_get_unary_op(GK_CONST_T(t));
}

enum ggml_glu_op ggml_get_glu_op(const struct ggml_tensor * t) {
    return (enum ggml_glu_op) gk_get_glu_op(GK_CONST_T(t));
}

// ---------------------------------------------------------------------------
// op builders
//
// From here down the file is forwarding. Each wrapper is named for the ggml
// entry point it keeps alive; the behaviour lives in gk_ops.c.
// ---------------------------------------------------------------------------

#define CTX GKC(ctx)

struct ggml_tensor * ggml_dup(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_dup(CTX, GKT(a)));
}

struct ggml_tensor * ggml_dup_inplace(struct ggml_context * ctx, struct ggml_tensor * a) {
    // a dup over its own storage moves nothing
    struct gk_tensor * r = gk_view_tensor(CTX, GKT(a));
    r->op = GK_OP_DUP;
    r->src[0] = GKT(a);
    return GGT(r);
}

#define FWD2(name) \
    struct ggml_tensor * ggml_##name(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) { \
        return GGT(gk_##name(CTX, GKT(a), GKT(b))); \
    } \
    struct ggml_tensor * ggml_##name##_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) { \
        return GGT(gk_##name##_inplace(CTX, GKT(a), GKT(b))); \
    }

FWD2(add)
FWD2(sub)
FWD2(mul)
FWD2(div)

struct ggml_tensor * ggml_add_cast(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, enum ggml_type type) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(type);
    GGML_ABORT("ggml_add_cast is a training-path op and is not supported");
}

struct ggml_tensor * ggml_add_id(struct ggml_context * ctx, struct ggml_tensor * a,
                                 struct ggml_tensor * b, struct ggml_tensor * ids) {
    return GGT(gk_add_id(CTX, GKT(a), GKT(b), GKT(ids)));
}

struct ggml_tensor * ggml_add1(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    // deprecated in the interface; keep the semantics via broadcast add
    return GGT(gk_add(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_add1_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_add_inplace(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_acc(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                              size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return GGT(gk_acc(CTX, GKT(a), GKT(b), nb1, nb2, nb3, offset));
}

struct ggml_tensor * ggml_acc_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                              size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return GGT(gk_acc_inplace(CTX, GKT(a), GKT(b), nb1, nb2, nb3, offset));
}

#define FWD1(name) \
    struct ggml_tensor * ggml_##name(struct ggml_context * ctx, struct ggml_tensor * a) { \
        return GGT(gk_##name(CTX, GKT(a))); \
    } \
    struct ggml_tensor * ggml_##name##_inplace(struct ggml_context * ctx, struct ggml_tensor * a) { \
        return GGT(gk_##name##_inplace(CTX, GKT(a))); \
    }

FWD1(sqr)
FWD1(sqrt)
FWD1(log)
FWD1(expm1)
FWD1(softplus)
FWD1(sin)
FWD1(cos)
FWD1(abs)
FWD1(sgn)
FWD1(neg)
FWD1(step)
FWD1(tanh)
FWD1(elu)
FWD1(relu)
FWD1(sigmoid)
FWD1(gelu)
FWD1(gelu_erf)
FWD1(gelu_quick)
FWD1(silu)
FWD1(exp)
FWD1(floor)
FWD1(ceil)
FWD1(round)
FWD1(trunc)

struct ggml_tensor * ggml_leaky_relu(struct ggml_context * ctx, struct ggml_tensor * a,
                                     float negative_slope, bool inplace) {
    return GGT(gk_leaky_relu(CTX, GKT(a), negative_slope, inplace));
}

struct ggml_tensor * ggml_hardswish(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_hardswish(CTX, GKT(a)));
}

struct ggml_tensor * ggml_hardsigmoid(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_hardsigmoid(CTX, GKT(a)));
}

struct ggml_tensor * ggml_xielu(struct ggml_context * ctx, struct ggml_tensor * a,
                                float alpha_n, float alpha_p, float beta, float eps) {
    return GGT(gk_xielu(CTX, GKT(a), alpha_n, alpha_p, beta, eps));
}

struct ggml_tensor * ggml_silu_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b);
    GGML_ABORT("gradient ops are not supported");
}

struct ggml_tensor * ggml_unary(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op) {
    return GGT(gk_unary(CTX, GKT(a), (enum gk_unary_op) op));
}

struct ggml_tensor * ggml_unary_inplace(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op) {
    return GGT(gk_unary_inplace(CTX, GKT(a), (enum gk_unary_op) op));
}

// glu family
struct ggml_tensor * ggml_glu(struct ggml_context * ctx, struct ggml_tensor * a,
                              enum ggml_glu_op op, bool swapped) {
    return GGT(gk_glu(CTX, GKT(a), (enum gk_glu_op) op, swapped));
}

struct ggml_tensor * ggml_glu_split(struct ggml_context * ctx, struct ggml_tensor * a,
                                    struct ggml_tensor * b, enum ggml_glu_op op) {
    return GGT(gk_glu_split(CTX, GKT(a), GKT(b), (enum gk_glu_op) op));
}

#define FWD_GLU(name, op) \
    struct ggml_tensor * ggml_##name(struct ggml_context * ctx, struct ggml_tensor * a) { \
        return GGT(gk_glu(CTX, GKT(a), op, false)); \
    } \
    struct ggml_tensor * ggml_##name##_swapped(struct ggml_context * ctx, struct ggml_tensor * a) { \
        return GGT(gk_glu(CTX, GKT(a), op, true)); \
    } \
    struct ggml_tensor * ggml_##name##_split(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) { \
        return GGT(gk_glu_split(CTX, GKT(a), GKT(b), op)); \
    }

FWD_GLU(reglu,       GK_GLU_OP_REGLU)
FWD_GLU(geglu,       GK_GLU_OP_GEGLU)
FWD_GLU(swiglu,      GK_GLU_OP_SWIGLU)
FWD_GLU(geglu_erf,   GK_GLU_OP_GEGLU_ERF)
FWD_GLU(geglu_quick, GK_GLU_OP_GEGLU_QUICK)

struct ggml_tensor * ggml_swiglu_oai(struct ggml_context * ctx, struct ggml_tensor * a,
                                     struct ggml_tensor * b, float alpha, float limit) {
    return GGT(gk_swiglu_oai(CTX, GKT(a), GKT(b), alpha, limit));
}

// reductions
struct ggml_tensor * ggml_sum(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_sum(CTX, GKT(a)));
}

struct ggml_tensor * ggml_sum_rows(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_sum_rows(CTX, GKT(a)));
}

struct ggml_tensor * ggml_cumsum(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_cumsum(CTX, GKT(a)));
}

struct ggml_tensor * ggml_mean(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_mean(CTX, GKT(a)));
}

struct ggml_tensor * ggml_argmax(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_argmax(CTX, GKT(a)));
}

struct ggml_tensor * ggml_count_equal(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_count_equal(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_repeat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_repeat(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_repeat_4d(struct ggml_context * ctx, struct ggml_tensor * a,
                                    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return GGT(gk_repeat_4d(CTX, GKT(a), ne0, ne1, ne2, ne3));
}

struct ggml_tensor * ggml_repeat_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b);
    GGML_ABORT("gradient ops are not supported");
}

struct ggml_tensor * ggml_concat(struct ggml_context * ctx, struct ggml_tensor * a,
                                 struct ggml_tensor * b, int dim) {
    return GGT(gk_concat(CTX, GKT(a), GKT(b), dim));
}

// norms
struct ggml_tensor * ggml_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps) {
    return GGT(gk_norm(CTX, GKT(a), eps));
}

struct ggml_tensor * ggml_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps) {
    return GGT(gk_norm_inplace(CTX, GKT(a), eps));
}

struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps) {
    return GGT(gk_rms_norm(CTX, GKT(a), eps));
}

struct ggml_tensor * ggml_rms_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps) {
    return GGT(gk_rms_norm_inplace(CTX, GKT(a), eps));
}

struct ggml_tensor * ggml_rms_norm_back(struct ggml_context * ctx, struct ggml_tensor * a,
                                        struct ggml_tensor * b, float eps) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(eps);
    GGML_ABORT("gradient ops are not supported");
}

struct ggml_tensor * ggml_group_norm(struct ggml_context * ctx, struct ggml_tensor * a,
                                     int n_groups, float eps) {
    return GGT(gk_group_norm(CTX, GKT(a), n_groups, eps));
}

struct ggml_tensor * ggml_group_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                     int n_groups, float eps) {
    // gk's group_norm always writes fresh storage; matching in-place
    // behaviour is not worth a second kernel for a cosmetic saving
    return GGT(gk_group_norm(CTX, GKT(a), n_groups, eps));
}

struct ggml_tensor * ggml_l2_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps) {
    return GGT(gk_l2_norm(CTX, GKT(a), eps));
}

struct ggml_tensor * ggml_l2_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps) {
    return GGT(gk_l2_norm(CTX, GKT(a), eps));
}

// matmul
struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_mul_mat(CTX, GKT(a), GKT(b)));
}

void ggml_mul_mat_set_prec(struct ggml_tensor * a, enum ggml_prec prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);
    gk_set_op_params_i32(GKT(a), 0, (int32_t) prec);
}

void ggml_mul_mat_set_hint(struct ggml_tensor * a, enum ggml_op_hint hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);
    gk_set_op_params_i32(GKT(a), 1, (int32_t) hint);
}

struct ggml_tensor * ggml_mul_mat_id(struct ggml_context * ctx, struct ggml_tensor * as,
                                     struct ggml_tensor * b, struct ggml_tensor * ids) {
    return GGT(gk_mul_mat_id(CTX, GKT(as), GKT(b), GKT(ids)));
}

struct ggml_tensor * ggml_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_out_prod(CTX, GKT(a), GKT(b)));
}

// scale / clamp / placement
struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s) {
    return GGT(gk_scale(CTX, GKT(a), s));
}

struct ggml_tensor * ggml_scale_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s) {
    return GGT(gk_scale_inplace(CTX, GKT(a), s));
}

struct ggml_tensor * ggml_scale_bias(struct ggml_context * ctx, struct ggml_tensor * a, float s, float b) {
    return GGT(gk_scale_bias(CTX, GKT(a), s, b));
}

struct ggml_tensor * ggml_scale_bias_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s, float b) {
    struct gk_tensor * r = gk_view_tensor(CTX, GKT(a));
    float params[2] = { s, b };
    gk_set_op_params(r, params, sizeof(params));
    r->op = GK_OP_SCALE;
    r->src[0] = GKT(a);
    return GGT(r);
}

struct ggml_tensor * ggml_clamp(struct ggml_context * ctx, struct ggml_tensor * a, float min, float max) {
    return GGT(gk_clamp(CTX, GKT(a), min, max));
}

struct ggml_tensor * ggml_set(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                              size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return GGT(gk_set(CTX, GKT(a), GKT(b), nb1, nb2, nb3, offset));
}

struct ggml_tensor * ggml_set_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                              size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return GGT(gk_set_inplace(CTX, GKT(a), GKT(b), nb1, nb2, nb3, offset));
}

struct ggml_tensor * ggml_set_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                 size_t offset) {
    return GGT(gk_set_1d(CTX, GKT(a), GKT(b), offset));
}

struct ggml_tensor * ggml_set_1d_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                 size_t offset) {
    return GGT(gk_set_1d_inplace(CTX, GKT(a), GKT(b), offset));
}

struct ggml_tensor * ggml_set_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                 size_t nb1, size_t offset) {
    return GGT(gk_set(CTX, GKT(a), GKT(b), nb1, GKT(a)->nb[2], GKT(a)->nb[3], offset));
}

struct ggml_tensor * ggml_set_2d_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                 size_t nb1, size_t offset) {
    return GGT(gk_set_inplace(CTX, GKT(a), GKT(b), nb1, GKT(a)->nb[2], GKT(a)->nb[3], offset));
}

// copies and shape
struct ggml_tensor * ggml_cpy(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_cpy(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_cast(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type) {
    return GGT(gk_cast(CTX, GKT(a), (enum gk_type) type));
}

struct ggml_tensor * ggml_cont(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_cont(CTX, GKT(a)));
}

struct ggml_tensor * ggml_cont_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0) {
    return GGT(gk_cont_2d(CTX, GKT(a), ne0, 1));
}

struct ggml_tensor * ggml_cont_2d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, int64_t ne1) {
    return GGT(gk_cont_2d(CTX, GKT(a), ne0, ne1));
}

struct ggml_tensor * ggml_cont_3d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, int64_t ne1, int64_t ne2) {
    return GGT(gk_cont_3d(CTX, GKT(a), ne0, ne1, ne2));
}

struct ggml_tensor * ggml_cont_4d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return GGT(gk_cont_4d(CTX, GKT(a), ne0, ne1, ne2, ne3));
}

struct ggml_tensor * ggml_reshape(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_reshape(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_reshape_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0) {
    return GGT(gk_reshape_1d(CTX, GKT(a), ne0));
}

struct ggml_tensor * ggml_reshape_2d(struct ggml_context * ctx, struct ggml_tensor * a,
                                     int64_t ne0, int64_t ne1) {
    return GGT(gk_reshape_2d(CTX, GKT(a), ne0, ne1));
}

struct ggml_tensor * ggml_reshape_3d(struct ggml_context * ctx, struct ggml_tensor * a,
                                     int64_t ne0, int64_t ne1, int64_t ne2) {
    return GGT(gk_reshape_3d(CTX, GKT(a), ne0, ne1, ne2));
}

struct ggml_tensor * ggml_reshape_4d(struct ggml_context * ctx, struct ggml_tensor * a,
                                     int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return GGT(gk_reshape_4d(CTX, GKT(a), ne0, ne1, ne2, ne3));
}

struct ggml_tensor * ggml_view_1d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, size_t offset) {
    return GGT(gk_view_1d(CTX, GKT(a), ne0, offset));
}

struct ggml_tensor * ggml_view_2d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
    return GGT(gk_view_2d(CTX, GKT(a), ne0, ne1, nb1, offset));
}

struct ggml_tensor * ggml_view_3d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, int64_t ne1, int64_t ne2,
                                  size_t nb1, size_t nb2, size_t offset) {
    return GGT(gk_view_3d(CTX, GKT(a), ne0, ne1, ne2, nb1, nb2, offset));
}

struct ggml_tensor * ggml_view_4d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                  size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return GGT(gk_view_4d(CTX, GKT(a), ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset));
}

struct ggml_tensor * ggml_permute(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int axis0, int axis1, int axis2, int axis3) {
    return GGT(gk_permute(CTX, GKT(a), axis0, axis1, axis2, axis3));
}

struct ggml_tensor * ggml_transpose(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_transpose(CTX, GKT(a)));
}

struct ggml_tensor * ggml_get_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_get_rows(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_get_rows_back(struct ggml_context * ctx, struct ggml_tensor * a,
                                        struct ggml_tensor * b, struct ggml_tensor * c) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(c);
    GGML_ABORT("gradient ops are not supported");
}

struct ggml_tensor * ggml_set_rows(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, struct ggml_tensor * c) {
    return GGT(gk_set_rows(CTX, GKT(a), GKT(b), GKT(c)));
}

struct ggml_tensor * ggml_diag(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_diag(CTX, GKT(a)));
}

struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx, struct ggml_tensor * a, int n_past) {
    return GGT(gk_diag_mask_inf(CTX, GKT(a), n_past));
}

struct ggml_tensor * ggml_diag_mask_inf_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_past) {
    return GGT(gk_diag_mask_inf_inplace(CTX, GKT(a), n_past));
}

struct ggml_tensor * ggml_diag_mask_zero(struct ggml_context * ctx, struct ggml_tensor * a, int n_past) {
    return GGT(gk_diag_mask_zero(CTX, GKT(a), n_past));
}

struct ggml_tensor * ggml_diag_mask_zero_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_past) {
    // gk's diag_mask_zero always allocates; the in-place saving is cosmetic
    return GGT(gk_diag_mask_zero(CTX, GKT(a), n_past));
}

// softmax
struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_soft_max(CTX, GKT(a)));
}

struct ggml_tensor * ggml_soft_max_inplace(struct ggml_context * ctx, struct ggml_tensor * a) {
    return GGT(gk_soft_max(CTX, GKT(a)));
}

struct ggml_tensor * ggml_soft_max_ext(struct ggml_context * ctx, struct ggml_tensor * a,
                                       struct ggml_tensor * mask, float scale, float max_bias) {
    return GGT(gk_soft_max_ext(CTX, GKT(a), GKT(mask), scale, max_bias));
}

struct ggml_tensor * ggml_soft_max_ext_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                       struct ggml_tensor * mask, float scale, float max_bias) {
    return GGT(gk_soft_max_ext(CTX, GKT(a), GKT(mask), scale, max_bias));
}

void ggml_soft_max_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks) {
    gk_soft_max_add_sinks(GKT(a), GKT(sinks));
}

struct ggml_tensor * ggml_soft_max_ext_back(struct ggml_context * ctx, struct ggml_tensor * a,
                                            struct ggml_tensor * b, float scale, float max_bias) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(scale); GGML_UNUSED(max_bias);
    GGML_ABORT("gradient ops are not supported");
}

struct ggml_tensor * ggml_soft_max_ext_back_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                            struct ggml_tensor * b, float scale, float max_bias) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(scale); GGML_UNUSED(max_bias);
    GGML_ABORT("gradient ops are not supported");
}

// rope
struct ggml_tensor * ggml_rope(struct ggml_context * ctx, struct ggml_tensor * a,
                               struct ggml_tensor * b, int n_dims, int mode) {
    return GGT(gk_rope(CTX, GKT(a), GKT(b), n_dims, mode));
}

struct ggml_tensor * ggml_rope_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                               struct ggml_tensor * b, int n_dims, int mode) {
    return GGT(gk_rope(CTX, GKT(a), GKT(b), n_dims, mode));
}

struct ggml_tensor * ggml_rope_ext(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, struct ggml_tensor * c,
                                   int n_dims, int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    return GGT(gk_rope_ext(CTX, GKT(a), GKT(b), GKT(c), n_dims, mode, n_ctx_orig,
                           freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
}

struct ggml_tensor * ggml_rope_ext_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, struct ggml_tensor * c,
                                   int n_dims, int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    return GGT(gk_rope_ext(CTX, GKT(a), GKT(b), GKT(c), n_dims, mode, n_ctx_orig,
                           freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
}

struct ggml_tensor * ggml_rope_multi(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, struct ggml_tensor * c,
                                   int n_dims, int sections[GGML_MROPE_SECTIONS], int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    return GGT(gk_rope_multi(CTX, GKT(a), GKT(b), GKT(c), n_dims, sections, mode, n_ctx_orig,
                             freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
}

struct ggml_tensor * ggml_rope_multi_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, struct ggml_tensor * c,
                                   int n_dims, int sections[GGML_MROPE_SECTIONS], int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    return GGT(gk_rope_multi(CTX, GKT(a), GKT(b), GKT(c), n_dims, sections, mode, n_ctx_orig,
                             freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
}

struct ggml_tensor * ggml_rope_custom(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b,
                                   int n_dims, int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    return GGT(gk_rope_ext(CTX, GKT(a), GKT(b), NULL, n_dims, mode, n_ctx_orig,
                           freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
}

struct ggml_tensor * ggml_rope_custom_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b,
                                   int n_dims, int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    return GGT(gk_rope_ext(CTX, GKT(a), GKT(b), NULL, n_dims, mode, n_ctx_orig,
                           freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
}

void ggml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                              float beta_fast, float beta_slow, float dims[2]) {
    gk_rope_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, dims);
}

struct ggml_tensor * ggml_rope_ext_back(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, struct ggml_tensor * c,
                                   int n_dims, int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(c);
    GGML_UNUSED(n_dims); GGML_UNUSED(mode); GGML_UNUSED(n_ctx_orig);
    GGML_UNUSED(freq_base); GGML_UNUSED(freq_scale); GGML_UNUSED(ext_factor);
    GGML_UNUSED(attn_factor); GGML_UNUSED(beta_fast); GGML_UNUSED(beta_slow);
    GGML_ABORT("gradient ops are not supported");
}

struct ggml_tensor * ggml_rope_multi_back(struct ggml_context * ctx, struct ggml_tensor * a,
                                   struct ggml_tensor * b, struct ggml_tensor * c,
                                   int n_dims, int sections[4], int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(c);
    GGML_UNUSED(n_dims); GGML_UNUSED(sections); GGML_UNUSED(mode); GGML_UNUSED(n_ctx_orig);
    GGML_UNUSED(freq_base); GGML_UNUSED(freq_scale); GGML_UNUSED(ext_factor);
    GGML_UNUSED(attn_factor); GGML_UNUSED(beta_fast); GGML_UNUSED(beta_slow);
    GGML_ABORT("gradient ops are not supported");
}

// convolution
struct ggml_tensor * ggml_im2col(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                 int s0, int s1, int p0, int p1, int d0, int d1,
                                 bool is_2D, enum ggml_type dst_type) {
    return GGT(gk_im2col(CTX, GKT(a), GKT(b), s0, s1, p0, p1, d0, d1, is_2D,
                         (enum gk_type) dst_type));
}

struct ggml_tensor * ggml_im2col_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                 int64_t * ne, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(ne);
    GGML_UNUSED(s0); GGML_UNUSED(s1); GGML_UNUSED(p0); GGML_UNUSED(p1);
    GGML_UNUSED(d0); GGML_UNUSED(d1); GGML_UNUSED(is_2D);
    GGML_ABORT("gradient ops are not supported");
}

struct ggml_tensor * ggml_im2col_3d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                 int64_t IC, int s0, int s1, int s2, int p0, int p1, int p2,
                                 int d0, int d1, int d2, enum ggml_type dst_type) {
    return GGT(gk_im2col_3d(CTX, GKT(a), GKT(b), IC, s0, s1, s2, p0, p1, p2, d0, d1, d2,
                            (enum gk_type) dst_type));
}

struct ggml_tensor * ggml_col2im_1d(struct ggml_context * ctx, struct ggml_tensor * a,
                                    int s0, int oc, int p0) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(s0); GGML_UNUSED(oc); GGML_UNUSED(p0);
    GGML_ABORT("col2im_1d has no consumer in this engine and no gk kernel");
}

struct ggml_tensor * ggml_conv_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                  int s0, int p0, int d0) {
    return GGT(gk_conv_1d(CTX, GKT(a), GKT(b), s0, p0, d0));
}

struct ggml_tensor * ggml_conv_1d_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s, int d) {
    return GGT(gk_conv_1d_ph(CTX, GKT(a), GKT(b), s, d));
}

struct ggml_tensor * ggml_conv_1d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s0, int p0, int d0) {
    return GGT(gk_conv_1d_dw(CTX, GKT(a), GKT(b), s0, p0, d0));
}

struct ggml_tensor * ggml_conv_1d_dw_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                        int s0, int d0) {
    return GGT(gk_conv_1d_dw_ph(CTX, GKT(a), GKT(b), s0, d0));
}

struct ggml_tensor * ggml_conv_transpose_1d(struct ggml_context * ctx, struct ggml_tensor * a,
                                            struct ggml_tensor * b, int s0, int p0, int d0) {
    return GGT(gk_conv_transpose_1d(CTX, GKT(a), GKT(b), s0, p0, d0));
}

struct ggml_tensor * ggml_conv_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                  int s0, int s1, int p0, int p1, int d0, int d1) {
    return GGT(gk_conv_2d(CTX, GKT(a), GKT(b), s0, s1, p0, p1, d0, d1));
}

struct ggml_tensor * ggml_conv_3d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                  int64_t IC, int s0, int s1, int s2, int p0, int p1, int p2,
                                  int d0, int d1, int d2) {
    return GGT(gk_conv_3d(CTX, GKT(a), GKT(b), IC, s0, s1, s2, p0, p1, p2, d0, d1, d2));
}

struct ggml_tensor * ggml_conv_2d_sk_p0(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_conv_2d_sk_p0(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_conv_2d_s1_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    return GGT(gk_conv_2d_s1_ph(CTX, GKT(a), GKT(b)));
}

struct ggml_tensor * ggml_conv_2d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s0, int s1, int p0, int p1, int d0, int d1) {
    return GGT(gk_conv_2d_dw(CTX, GKT(a), GKT(b), s0, s1, p0, p1, d0, d1));
}

struct ggml_tensor * ggml_conv_2d_dw_direct(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s0, int s1, int p0, int p1, int d0, int d1) {
    return GGT(gk_conv_2d_dw_direct(CTX, GKT(a), GKT(b), s0, s1, p0, p1, d0, d1));
}

struct ggml_tensor * ggml_conv_2d_direct(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s0, int s1, int p0, int p1, int d0, int d1) {
    return GGT(gk_conv_2d_direct(CTX, GKT(a), GKT(b), s0, s1, p0, p1, d0, d1));
}

struct ggml_tensor * ggml_conv_3d_direct(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s0, int s1, int s2, int p0, int p1, int p2,
                                     int d0, int d1, int d2, int c, int n, int oc) {
    return GGT(gk_conv_3d_direct(CTX, GKT(a), GKT(b), s0, s1, s2, p0, p1, p2, d0, d1, d2, c, n, oc));
}

struct ggml_tensor * ggml_conv_transpose_2d_p0(struct ggml_context * ctx, struct ggml_tensor * a,
                                     struct ggml_tensor * b, int stride) {
    return GGT(gk_conv_transpose_2d_p0(CTX, GKT(a), GKT(b), stride));
}

struct ggml_tensor * ggml_pool_1d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  enum ggml_op_pool op, int k0, int s0, int p0) {
    return GGT(gk_pool_1d(CTX, GKT(a), (enum gk_op_pool) op, k0, s0, p0));
}

struct ggml_tensor * ggml_pool_2d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  enum ggml_op_pool op, int k0, int k1, int s0, int s1,
                                  float p0, float p1) {
    return GGT(gk_pool_2d(CTX, GKT(a), (enum gk_op_pool) op, k0, k1, s0, s1, p0, p1));
}

struct ggml_tensor * ggml_pool_2d_back(struct ggml_context * ctx, struct ggml_tensor * a,
                                  struct ggml_tensor * af, enum ggml_op_pool op,
                                  int k0, int k1, int s0, int s1, float p0, float p1) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(af); GGML_UNUSED(op);
    GGML_UNUSED(k0); GGML_UNUSED(k1); GGML_UNUSED(s0); GGML_UNUSED(s1);
    GGML_UNUSED(p0); GGML_UNUSED(p1);
    GGML_ABORT("gradient ops are not supported");
}

// resampling and padding
struct ggml_tensor * ggml_upscale(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int scale_factor, enum ggml_scale_mode mode) {
    return GGT(gk_upscale(CTX, GKT(a), scale_factor, (enum gk_scale_mode) mode));
}

struct ggml_tensor * ggml_upscale_ext(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int ne0, int ne1, int ne2, int ne3, enum ggml_scale_mode mode) {
    return GGT(gk_interpolate(CTX, GKT(a), ne0, ne1, ne2, ne3, (uint32_t) mode));
}

struct ggml_tensor * ggml_interpolate(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode) {
    return GGT(gk_interpolate(CTX, GKT(a), ne0, ne1, ne2, ne3, mode));
}

struct ggml_tensor * ggml_pad(struct ggml_context * ctx, struct ggml_tensor * a,
                              int p0, int p1, int p2, int p3) {
    return GGT(gk_pad(CTX, GKT(a), p0, p1, p2, p3));
}

struct ggml_tensor * ggml_pad_circular(struct ggml_context * ctx, struct ggml_tensor * a,
                              int p0, int p1, int p2, int p3) {
    return GGT(gk_pad_circular(CTX, GKT(a), p0, p1, p2, p3));
}

struct ggml_tensor * ggml_pad_ext(struct ggml_context * ctx, struct ggml_tensor * a,
                              int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3) {
    return GGT(gk_pad_ext(CTX, GKT(a), lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3));
}

struct ggml_tensor * ggml_pad_ext_circular(struct ggml_context * ctx, struct ggml_tensor * a,
                              int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3) {
    return GGT(gk_pad_ext_circular(CTX, GKT(a), lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3));
}

struct ggml_tensor * ggml_pad_reflect_1d(struct ggml_context * ctx, struct ggml_tensor * a,
                              int p0, int p1) {
    return GGT(gk_pad_reflect_1d(CTX, GKT(a), p0, p1));
}

struct ggml_tensor * ggml_roll(struct ggml_context * ctx, struct ggml_tensor * a,
                               int shift0, int shift1, int shift2, int shift3) {
    return GGT(gk_roll(CTX, GKT(a), shift0, shift1, shift2, shift3));
}

struct ggml_tensor * ggml_tri(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_tri_type type) {
    return GGT(gk_tri(CTX, GKT(a), (enum gk_tri_type) type));
}

struct ggml_tensor * ggml_fill(struct ggml_context * ctx, struct ggml_tensor * a, float c) {
    return GGT(gk_fill(CTX, GKT(a), c));
}

struct ggml_tensor * ggml_fill_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float c) {
    return GGT(gk_fill_inplace(CTX, GKT(a), c));
}

struct ggml_tensor * ggml_timestep_embedding(struct ggml_context * ctx, struct ggml_tensor * timesteps,
                                             int dim, int max_period) {
    return GGT(gk_timestep_embedding(CTX, GKT(timesteps), dim, max_period));
}

struct ggml_tensor * ggml_argsort(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_sort_order order) {
    return GGT(gk_argsort(CTX, GKT(a), (enum gk_sort_order) order));
}

struct ggml_tensor * ggml_argsort_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k) {
    return GGT(gk_argsort_top_k(CTX, GKT(a), k));
}

struct ggml_tensor * ggml_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k) {
    return GGT(gk_top_k(CTX, GKT(a), k));
}

struct ggml_tensor * ggml_arange(struct ggml_context * ctx, float start, float stop, float step) {
    return GGT(gk_arange(CTX, start, stop, step));
}

// attention
struct ggml_tensor * ggml_flash_attn_ext(struct ggml_context * ctx,
                                         struct ggml_tensor * q, struct ggml_tensor * k,
                                         struct ggml_tensor * v, struct ggml_tensor * mask,
                                         float scale, float max_bias, float logit_softcap) {
    return GGT(gk_flash_attn_ext(CTX, GKT(q), GKT(k), GKT(v), GKT(mask),
                                 scale, max_bias, logit_softcap));
}

void ggml_flash_attn_ext_set_prec(struct ggml_tensor * a, enum ggml_prec prec) {
    gk_flash_attn_ext_set_prec(GKT(a), (enum gk_prec) prec);
}

enum ggml_prec ggml_flash_attn_ext_get_prec(const struct ggml_tensor * a) {
    return (enum ggml_prec) gk_flash_attn_ext_get_prec(GK_CONST_T(a));
}

void ggml_flash_attn_ext_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks) {
    gk_flash_attn_ext_add_sinks(GKT(a), GKT(sinks));
}

struct ggml_tensor * ggml_flash_attn_back(struct ggml_context * ctx,
                                          struct ggml_tensor * q, struct ggml_tensor * k,
                                          struct ggml_tensor * v, struct ggml_tensor * d, bool masked) {
    GGML_UNUSED(ctx); GGML_UNUSED(q); GGML_UNUSED(k); GGML_UNUSED(v);
    GGML_UNUSED(d); GGML_UNUSED(masked);
    GGML_ABORT("gradient ops are not supported");
}

// recurrent
struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx, struct ggml_tensor * sx, struct ggml_tensor * c) {
    return GGT(gk_ssm_conv(CTX, GKT(sx), GKT(c)));
}

struct ggml_tensor * ggml_ssm_scan(struct ggml_context * ctx, struct ggml_tensor * s, struct ggml_tensor * x,
                                   struct ggml_tensor * dt, struct ggml_tensor * A,
                                   struct ggml_tensor * B, struct ggml_tensor * C, struct ggml_tensor * ids) {
    return GGT(gk_ssm_scan(CTX, GKT(s), GKT(x), GKT(dt), GKT(A), GKT(B), GKT(C), GKT(ids)));
}

struct ggml_tensor * ggml_win_part(struct ggml_context * ctx, struct ggml_tensor * a, int w) {
    return GGT(gk_win_part(CTX, GKT(a), w));
}

struct ggml_tensor * ggml_win_unpart(struct ggml_context * ctx, struct ggml_tensor * a, int w0, int h0, int w) {
    return GGT(gk_win_unpart(CTX, GKT(a), w0, h0, w));
}

struct ggml_tensor * ggml_get_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a, int qh, int kh) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(qh); GGML_UNUSED(kh);
    GGML_ABORT("get_rel_pos has no consumer in this engine and no gk kernel");
}

struct ggml_tensor * ggml_add_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a,
                                      struct ggml_tensor * pw, struct ggml_tensor * ph) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(pw); GGML_UNUSED(ph);
    GGML_ABORT("add_rel_pos has no consumer in this engine and no gk kernel");
}

struct ggml_tensor * ggml_add_rel_pos_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                      struct ggml_tensor * pw, struct ggml_tensor * ph) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(pw); GGML_UNUSED(ph);
    GGML_ABORT("add_rel_pos has no consumer in this engine and no gk kernel");
}

struct ggml_tensor * ggml_rwkv_wkv6(struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v,
                                    struct ggml_tensor * r, struct ggml_tensor * tf,
                                    struct ggml_tensor * td, struct ggml_tensor * state) {
    return GGT(gk_rwkv_wkv6(CTX, GKT(k), GKT(v), GKT(r), GKT(tf), GKT(td), GKT(state)));
}

struct ggml_tensor * ggml_gated_linear_attn(struct ggml_context * ctx, struct ggml_tensor * k,
                                    struct ggml_tensor * v, struct ggml_tensor * q,
                                    struct ggml_tensor * g, struct ggml_tensor * state, float scale) {
    return GGT(gk_gated_linear_attn(CTX, GKT(k), GKT(v), GKT(q), GKT(g), GKT(state), scale));
}

struct ggml_tensor * ggml_rwkv_wkv7(struct ggml_context * ctx, struct ggml_tensor * r, struct ggml_tensor * w,
                                    struct ggml_tensor * k, struct ggml_tensor * v,
                                    struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * state) {
    return GGT(gk_rwkv_wkv7(CTX, GKT(r), GKT(w), GKT(k), GKT(v), GKT(a), GKT(b), GKT(state)));
}

struct ggml_tensor * ggml_solve_tri(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                    bool left, bool lower, bool uni) {
    return GGT(gk_solve_tri(CTX, GKT(a), GKT(b), left, lower, uni));
}

struct ggml_tensor * ggml_gated_delta_net(struct ggml_context * ctx, struct ggml_tensor * q,
                                    struct ggml_tensor * k, struct ggml_tensor * v,
                                    struct ggml_tensor * g, struct ggml_tensor * beta,
                                    struct ggml_tensor * state, int64_t K) {
    return GGT(gk_gated_delta_net(CTX, GKT(q), GKT(k), GKT(v), GKT(g), GKT(beta), GKT(state), K));
}

struct ggml_tensor * ggml_lightning_indexer(struct ggml_context * ctx, struct ggml_tensor * q,
                                    struct ggml_tensor * k, struct ggml_tensor * weights,
                                    struct ggml_tensor * mask) {
    return GGT(gk_lightning_indexer(CTX, GKT(q), GKT(k), GKT(weights), GKT(mask)));
}

struct ggml_tensor * ggml_dsv4_hc_comb(struct ggml_context * ctx, struct ggml_tensor * mixes,
                                    struct ggml_tensor * scale, struct ggml_tensor * base,
                                    float eps, int32_t n_iter) {
    return GGT(gk_dsv4_hc_comb(CTX, GKT(mixes), GKT(scale), GKT(base), eps, n_iter));
}

struct ggml_tensor * ggml_dsv4_hc_pre(struct ggml_context * ctx, struct ggml_tensor * x,
                                    struct ggml_tensor * weights) {
    return GGT(gk_dsv4_hc_pre(CTX, GKT(x), GKT(weights)));
}

struct ggml_tensor * ggml_dsv4_hc_post(struct ggml_context * ctx, struct ggml_tensor * x,
                                    struct ggml_tensor * residual, struct ggml_tensor * post,
                                    struct ggml_tensor * comb) {
    return GGT(gk_dsv4_hc_post(CTX, GKT(x), GKT(residual), GKT(post), GKT(comb)));
}

// custom ops
struct ggml_tensor * ggml_map_custom1(struct ggml_context * ctx, struct ggml_tensor * a,
                                      ggml_custom1_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_map_custom1(CTX, GKT(a), (gk_custom1_op_t) fun, n_tasks, userdata));
}

struct ggml_tensor * ggml_map_custom1_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                      ggml_custom1_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_map_custom1_inplace(CTX, GKT(a), (gk_custom1_op_t) fun, n_tasks, userdata));
}

struct ggml_tensor * ggml_map_custom2(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                      ggml_custom2_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_map_custom2(CTX, GKT(a), GKT(b), (gk_custom2_op_t) fun, n_tasks, userdata));
}

struct ggml_tensor * ggml_map_custom2_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                      ggml_custom2_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_map_custom2_inplace(CTX, GKT(a), GKT(b), (gk_custom2_op_t) fun, n_tasks, userdata));
}

struct ggml_tensor * ggml_map_custom3(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                      struct ggml_tensor * c,
                                      ggml_custom3_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_map_custom3(CTX, GKT(a), GKT(b), GKT(c), (gk_custom3_op_t) fun, n_tasks, userdata));
}

struct ggml_tensor * ggml_map_custom3_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                      struct ggml_tensor * c,
                                      ggml_custom3_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_map_custom3_inplace(CTX, GKT(a), GKT(b), GKT(c), (gk_custom3_op_t) fun, n_tasks, userdata));
}

struct ggml_tensor * ggml_custom_4d(struct ggml_context * ctx, enum ggml_type type,
                                    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                    struct ggml_tensor ** args, int n_args,
                                    ggml_custom_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_custom_4d(CTX, (enum gk_type) type, ne0, ne1, ne2, ne3,
                            (struct gk_tensor **) args, n_args, (gk_custom_op_t) fun,
                            n_tasks, userdata));
}

struct ggml_tensor * ggml_custom_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                    struct ggml_tensor ** args, int n_args,
                                    ggml_custom_op_t fun, int n_tasks, void * userdata) {
    return GGT(gk_custom_inplace(CTX, GKT(a), (struct gk_tensor **) args, n_args,
                                 (gk_custom_op_t) fun, n_tasks, userdata));
}

// loss and training steps: declared for the interface, never runnable here
struct ggml_tensor * ggml_cross_entropy_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b);
    GGML_ABORT("training is not supported");
}

struct ggml_tensor * ggml_cross_entropy_loss_back(struct ggml_context * ctx, struct ggml_tensor * a,
                                    struct ggml_tensor * b, struct ggml_tensor * c) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(c);
    GGML_ABORT("training is not supported");
}

struct ggml_tensor * ggml_opt_step_adamw(struct ggml_context * ctx, struct ggml_tensor * a,
                                    struct ggml_tensor * grad, struct ggml_tensor * m,
                                    struct ggml_tensor * v, struct ggml_tensor * adamw_params) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(grad); GGML_UNUSED(m);
    GGML_UNUSED(v); GGML_UNUSED(adamw_params);
    GGML_ABORT("training is not supported");
}

struct ggml_tensor * ggml_opt_step_sgd(struct ggml_context * ctx, struct ggml_tensor * a,
                                    struct ggml_tensor * grad, struct ggml_tensor * sgd_params) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(grad); GGML_UNUSED(sgd_params);
    GGML_ABORT("training is not supported");
}

// ---------------------------------------------------------------------------
// graphs
// ---------------------------------------------------------------------------

struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx) {
    return GGG(gk_new_graph(GKC(ctx)));
}

struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads) {
    return GGG(gk_new_graph_custom(GKC(ctx), size, grads));
}

size_t ggml_graph_overhead(void) {
    return gk_graph_overhead();
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    return gk_graph_overhead_custom(size, grads);
}

void ggml_build_forward_expand(struct ggml_cgraph * graph, struct ggml_tensor * tensor) {
    gk_build_forward_expand(GKG(graph), GKT(tensor));
}

struct ggml_tensor * ggml_build_forward_select(struct ggml_cgraph * graph,
                                               struct ggml_tensor ** tensors,
                                               int n_tensors, int idx) {
    return GGT(gk_build_forward_select(GKG(graph), (struct gk_tensor **) tensors, n_tensors, idx));
}

void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * cgraph,
                                struct ggml_tensor ** grad_accs) {
    GGML_UNUSED(ctx); GGML_UNUSED(cgraph); GGML_UNUSED(grad_accs);
    GGML_ABORT("training is not supported");
}

int ggml_graph_size(struct ggml_cgraph * g) {
    // the capacity, not the count
    return gk_graph_size(GKG(g));
}

struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * g, int i) {
    return GGT(gk_graph_node(GKG(g), i));
}

struct ggml_tensor ** ggml_graph_nodes(struct ggml_cgraph * g) {
    return (struct ggml_tensor **) gk_graph_nodes(GKG(g));
}

int ggml_graph_n_nodes(struct ggml_cgraph * g) {
    return gk_graph_n_nodes(GKG(g));
}

void ggml_graph_add_node(struct ggml_cgraph * g, struct ggml_tensor * t) {
    gk_graph_add_node(GKG(g), GKT(t));
}

void ggml_graph_clear(struct ggml_cgraph * g) {
    gk_graph_clear(GKG(g));
}

void ggml_graph_cpy(struct ggml_cgraph * src, struct ggml_cgraph * dst) {
    gk_graph_cpy(GKG(src), GKG(dst));
}

struct ggml_cgraph * ggml_graph_dup(struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads) {
    GGML_ASSERT(!force_grads);
    struct gk_cgraph * copy = gk_new_graph_custom(GKC(ctx), gk_graph_size(GKG(cgraph)), false);
    gk_graph_cpy(GKG(cgraph), copy);
    return GGG(copy);
}

void ggml_graph_reset(struct ggml_cgraph * cgraph) {
    GGML_UNUSED(cgraph); // gradients only; inference graphs carry none
}

struct ggml_tensor * ggml_graph_get_tensor(const struct ggml_cgraph * cgraph, const char * name) {
    return GGT(gk_graph_get_tensor((const struct gk_cgraph *) cgraph, name));
}

struct ggml_tensor * ggml_graph_get_grad(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node) {
    GGML_UNUSED(cgraph); GGML_UNUSED(node);
    return NULL; // no gradients exist in this engine
}

struct ggml_tensor * ggml_graph_get_grad_acc(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node) {
    GGML_UNUSED(cgraph); GGML_UNUSED(node);
    return NULL;
}

void ggml_graph_print(const struct ggml_cgraph * cgraph) {
    const struct gk_cgraph * g = (const struct gk_cgraph *) cgraph;
    GGML_LOG_INFO("=== GRAPH: %d nodes ===\n", gk_graph_n_nodes((struct gk_cgraph *) g));
}

void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * cgraph, const char * filename) {
    GGML_UNUSED(gb); GGML_UNUSED(cgraph); GGML_UNUSED(filename);
    GGML_LOG_WARN("ggml_graph_dump_dot is not implemented in the gk compatibility layer\n");
}

void ggml_print_object(const struct ggml_object * obj) {
    GGML_UNUSED(obj);
}

void ggml_print_objects(const struct ggml_context * ctx) {
    GGML_UNUSED(ctx);
}

// ---------------------------------------------------------------------------
// quantisation - routed to the shared codec
// ---------------------------------------------------------------------------

void ggml_quantize_init(enum ggml_type type) {
    GGML_UNUSED(type); // the codec initialises its tables lazily and thread-safely
}

void ggml_quantize_free(void) {
}

bool ggml_quantize_requires_imatrix(enum ggml_type type) {
    return qz_quantize_requires_imatrix((qz_type) type);
}

size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst,
                           int64_t start, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return qz_quantize_chunk((qz_type) type, src, dst, start, nrows, n_per_row, imatrix);
}

bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes) {
    return qz_validate_row_data((qz_type) type, data, nbytes);
}

// ---------------------------------------------------------------------------
// threadpool parameter helpers (the pool itself lives in the cpu layer)
// ---------------------------------------------------------------------------

struct ggml_threadpool_params ggml_threadpool_params_default(int n_threads) {
    struct ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    return p;
}

void ggml_threadpool_params_init(struct ggml_threadpool_params * p, int n_threads) {
    p->n_threads  = n_threads;
    p->prio       = GGML_SCHED_PRIO_NORMAL;
    p->poll       = 50;
    p->strict_cpu = false;
    p->paused     = false;
    memset(p->cpumask, 0, sizeof(p->cpumask));
}

bool ggml_threadpool_params_match(const struct ggml_threadpool_params * p0,
                                  const struct ggml_threadpool_params * p1) {
    if (p0->n_threads != p1->n_threads) {
        return false;
    }
    if (p0->prio != p1->prio || p0->poll != p1->poll || p0->strict_cpu != p1->strict_cpu) {
        return false;
    }
    return memcmp(p0->cpumask, p1->cpumask, sizeof(p0->cpumask)) == 0;
}
