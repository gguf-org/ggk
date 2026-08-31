// The ggml-cpu.h surface: element accessors, feature queries, the threadpool
// handle, and the direct graph-compute entry points the tests and tools use.

#include "ggml-compat-impl.h"

#include "ggml-cpu.h"

#include <stdlib.h>

// ---------------------------------------------------------------------------
// init and system queries
// ---------------------------------------------------------------------------

void ggml_cpu_init(void) {
    // gk and the qz codec initialise their tables lazily; nothing to do here
}

void ggml_numa_init(enum ggml_numa_strategy numa) {
    GGML_UNUSED(numa);
}

bool ggml_is_numa(void) {
    return false;
}

int ggml_cpu_has_sse3(void)        { return 0; }
int ggml_cpu_has_ssse3(void)       { return 0; }
int ggml_cpu_has_avx(void)         {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}
int ggml_cpu_has_avx_vnni(void)    { return 0; }
int ggml_cpu_has_avx2(void)        {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}
int ggml_cpu_has_bmi2(void)        { return 0; }
int ggml_cpu_has_f16c(void)        { return 0; }
int ggml_cpu_has_fma(void)         {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}
int ggml_cpu_has_avx512(void)      {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}
int ggml_cpu_has_avx512_vbmi(void) { return 0; }
int ggml_cpu_has_avx512_vnni(void) { return 0; }
int ggml_cpu_has_avx512_bf16(void) { return 0; }
int ggml_cpu_has_amx_int8(void)    { return 0; }
int ggml_cpu_has_neon(void)        {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}
int ggml_cpu_has_arm_fma(void)     {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}
int ggml_cpu_has_fp16_va(void)     { return 0; }
int ggml_cpu_has_dotprod(void)     {
#if defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}
int ggml_cpu_has_matmul_int8(void) { return 0; }
int ggml_cpu_has_sve(void)         { return 0; }
int ggml_cpu_get_sve_cnt(void)     { return 0; }
int ggml_cpu_has_sme(void)         { return 0; }
int ggml_cpu_has_sme2(void)        { return 0; }
int ggml_cpu_has_riscv_v(void)     { return 0; }
int ggml_cpu_get_rvv_vlen(void)    { return 0; }
int ggml_cpu_has_vsx(void)         { return 0; }
int ggml_cpu_has_vxe(void)         { return 0; }
int ggml_cpu_has_wasm_simd(void)   { return 0; }
int ggml_cpu_has_llamafile(void)   { return 0; }

// ---------------------------------------------------------------------------
// element accessors
// ---------------------------------------------------------------------------

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    return GGT(gk_new_i32(GKC(ctx), value));
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    return GGT(gk_new_f32(GKC(ctx), value));
}

static float ggml_compat_read_f32(const struct ggml_tensor * t, size_t off) {
    const char * p = (const char *) t->data + off;
    switch (t->type) {
        case GGML_TYPE_F32: return *(const float *) p;
        case GGML_TYPE_F16: return gk_fp16_to_fp32(*(const gk_fp16_t *) p);
        case GGML_TYPE_BF16: {
            gk_bf16_t b = { *(const uint16_t *) p };
            return gk_bf16_to_fp32(b);
        }
        case GGML_TYPE_I8:  return (float) *(const int8_t  *) p;
        case GGML_TYPE_I16: return (float) *(const int16_t *) p;
        case GGML_TYPE_I32: return (float) *(const int32_t *) p;
        case GGML_TYPE_I64: return (float) *(const int64_t *) p;
        default: GGML_ABORT("cannot read element of type %s", ggml_type_name(t->type));
    }
}

static void ggml_compat_write_f32(const struct ggml_tensor * t, size_t off, float v) {
    char * p = (char *) t->data + off;
    switch (t->type) {
        case GGML_TYPE_F32: *(float *) p = v; break;
        case GGML_TYPE_F16: *(gk_fp16_t *) p = gk_fp32_to_fp16(v); break;
        case GGML_TYPE_BF16: {
            const gk_bf16_t b = gk_fp32_to_bf16(v);
            *(uint16_t *) p = b.bits;
            break;
        }
        case GGML_TYPE_I8:  *(int8_t  *) p = (int8_t)  v; break;
        case GGML_TYPE_I16: *(int16_t *) p = (int16_t) v; break;
        case GGML_TYPE_I32: *(int32_t *) p = (int32_t) v; break;
        case GGML_TYPE_I64: *(int64_t *) p = (int64_t) v; break;
        default: GGML_ABORT("cannot write element of type %s", ggml_type_name(t->type));
    }
}

static size_t ggml_compat_elem_offset_1d(const struct ggml_tensor * t, int i) {
    GGML_ASSERT(ggml_is_contiguous(t) || ggml_n_dims(t) == 1);
    return (size_t) i * t->nb[0];
}

static size_t ggml_compat_elem_offset_nd(const struct ggml_tensor * t,
                                         int i0, int i1, int i2, int i3) {
    return i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}

struct ggml_tensor * ggml_set_i32(struct ggml_tensor * t, int32_t value) {
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        ggml_compat_write_f32(t, (size_t) i * t->nb[0], (float) value);
    }
    return t;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * t, float value) {
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        ggml_compat_write_f32(t, (size_t) i * t->nb[0], value);
    }
    return t;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * t, int i) {
    return (int32_t) ggml_compat_read_f32(t, ggml_compat_elem_offset_1d(t, i));
}

void ggml_set_i32_1d(const struct ggml_tensor * t, int i, int32_t value) {
    ggml_compat_write_f32(t, ggml_compat_elem_offset_1d(t, i), (float) value);
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * t, int i0, int i1, int i2, int i3) {
    return (int32_t) ggml_compat_read_f32(t, ggml_compat_elem_offset_nd(t, i0, i1, i2, i3));
}

void ggml_set_i32_nd(const struct ggml_tensor * t, int i0, int i1, int i2, int i3, int32_t value) {
    ggml_compat_write_f32(t, ggml_compat_elem_offset_nd(t, i0, i1, i2, i3), (float) value);
}

float ggml_get_f32_1d(const struct ggml_tensor * t, int i) {
    return ggml_compat_read_f32(t, ggml_compat_elem_offset_1d(t, i));
}

void ggml_set_f32_1d(const struct ggml_tensor * t, int i, float value) {
    ggml_compat_write_f32(t, ggml_compat_elem_offset_1d(t, i), value);
}

float ggml_get_f32_nd(const struct ggml_tensor * t, int i0, int i1, int i2, int i3) {
    return ggml_compat_read_f32(t, ggml_compat_elem_offset_nd(t, i0, i1, i2, i3));
}

void ggml_set_f32_nd(const struct ggml_tensor * t, int i0, int i1, int i2, int i3, float value) {
    ggml_compat_write_f32(t, ggml_compat_elem_offset_nd(t, i0, i1, i2, i3), value);
}

// ---------------------------------------------------------------------------
// threadpool handle
//
// The engine creates a threadpool and hands it to the backend; gk's backend
// owns its own pool, so this handle only remembers the requested width.
// ---------------------------------------------------------------------------

struct ggml_threadpool {
    int n_threads;
};

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * params) {
    struct ggml_threadpool * tp = malloc(sizeof(*tp));
    GGML_ASSERT(tp != NULL);
    tp->n_threads = params != NULL ? params->n_threads : 0;
    return tp;
}

void ggml_threadpool_free(struct ggml_threadpool * threadpool) {
    free(threadpool);
}

int ggml_threadpool_get_n_threads(struct ggml_threadpool * threadpool) {
    return threadpool != NULL ? threadpool->n_threads : 0;
}

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
    GGML_UNUSED(threadpool);
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
    GGML_UNUSED(threadpool);
}

// ---------------------------------------------------------------------------
// direct graph compute
// ---------------------------------------------------------------------------

struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph * cgraph, int n_threads,
                                  struct ggml_threadpool * threadpool) {
    GGML_UNUSED(cgraph);

    struct ggml_cplan plan;
    memset(&plan, 0, sizeof(plan));
    plan.n_threads  = threadpool != NULL ? ggml_threadpool_get_n_threads(threadpool) : n_threads;
    plan.threadpool = threadpool;
    return plan;
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    const int n_threads = cplan != NULL && cplan->n_threads > 0 ? cplan->n_threads : 0;
    return ggml_compat_status(gk_graph_compute(GKG(cgraph), n_threads));
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph,
                                             int n_threads) {
    GGML_UNUSED(ctx); // gk needs no work buffer carved from the context
    return ggml_compat_status(gk_graph_compute(GKG(cgraph), n_threads));
}

// ---------------------------------------------------------------------------
// cpu type traits
// ---------------------------------------------------------------------------

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    static struct ggml_type_traits_cpu cache[GGML_TYPE_COUNT];
    static bool filled[GGML_TYPE_COUNT];

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT);

    if (!filled[type]) {
        const struct gk_type_traits * tr = gk_get_type_traits((enum gk_type) type);
        cache[type].from_float   = (ggml_from_float_t) tr->from_float;
        cache[type].vec_dot      = (ggml_vec_dot_t) tr->vec_dot;
        cache[type].vec_dot_type = (enum ggml_type) tr->vec_dot_type;
        cache[type].nrows        = (int64_t) tr->nrows;
        filled[type] = true;
    }

    return &cache[type];
}
