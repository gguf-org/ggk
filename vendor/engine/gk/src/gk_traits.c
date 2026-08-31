// Type traits, scalar conversions and the row/dot entry points per type.
//
// The block formats themselves are not defined here. They live in
// ../../quantizer/src/kernels, which is the one implementation of the GGUF
// on-disk layouts in this tree; this file adapts that codec to what a compute
// engine asks of a type. Keeping a single codec means a tensor written by the
// quantizer and a tensor read by the engine can never drift apart, and it
// keeps the format work - the part that has to match published files bit for
// bit - in one reviewed place.

#include "gk_impl.h"
#include "gk_simd.h"

#include "qz_quant.h"

#include <math.h>

// The two enums carry the GGUF type ids, so they agree by construction. The
// checks below are here to fail loudly if either side is ever renumbered.
_Static_assert((int) GK_TYPE_COUNT == (int) QZ_TYPE_COUNT, "gk/qz type id drift");
_Static_assert((int) GK_TYPE_Q4_K == (int) QZ_TYPE_Q4_K,   "gk/qz type id drift");
_Static_assert((int) GK_TYPE_BF16 == (int) QZ_TYPE_BF16,   "gk/qz type id drift");
_Static_assert((int) GK_TYPE_NVFP4 == (int) QZ_TYPE_NVFP4, "gk/qz type id drift");

// --------------------------------------------------------------------------
// scalar conversions
// --------------------------------------------------------------------------

float gk_fp16_to_fp32(gk_fp16_t x) {
    return qz_fp16_to_fp32((qz_fp16_t) x);
}

gk_fp16_t gk_fp32_to_fp16(float x) {
    return (gk_fp16_t) qz_fp32_to_fp16(x);
}

// bf16 is the top 16 bits of an f32, so widening is a shift and narrowing is a
// round-to-nearest-even on the discarded low half.
float gk_bf16_to_fp32(gk_bf16_t x) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t) x.bits << 16;
    return v.f;
}

gk_bf16_t gk_fp32_to_bf16(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;

    gk_bf16_t r;
    if ((v.u & 0x7fffffffu) > 0x7f800000u) {
        r.bits = (uint16_t) ((v.u >> 16) | 64); // nan, forced quiet
        return r;
    }
    r.bits = (uint16_t) ((v.u + 0x7fffu + ((v.u >> 16) & 1)) >> 16);
    return r;
}

void gk_fp16_to_fp32_row(const gk_fp16_t * x, float * y, int64_t n) {
    qz_fp16_to_fp32_row((const qz_fp16_t *) x, y, n);
}

void gk_fp32_to_fp16_row(const float * x, gk_fp16_t * y, int64_t n) {
    qz_fp32_to_fp16_row(x, (qz_fp16_t *) y, n);
}

void gk_bf16_to_fp32_row(const gk_bf16_t * x, float * y, int64_t n) {
    qz_bf16_to_fp32_row((const qz_bf16_t *) x, y, n);
}

void gk_fp32_to_bf16_row(const float * x, gk_bf16_t * y, int64_t n) {
    qz_fp32_to_bf16_row(x, (qz_bf16_t *) y, n);
}

// --------------------------------------------------------------------------
// row conversions
//
// One pair per storage class. The quantized types all route through the
// codec, so a new format needs no code here - only a traits entry.
// --------------------------------------------------------------------------

static void to_float_f32(const void * src, float * dst, int64_t k) {
    memcpy(dst, src, (size_t) k * sizeof(float));
}

static void from_float_f32(const float * src, void * dst, int64_t k) {
    memcpy(dst, src, (size_t) k * sizeof(float));
}

static void to_float_f16(const void * src, float * dst, int64_t k) {
    gk_fp16_to_fp32_row((const gk_fp16_t *) src, dst, k);
}

static void from_float_f16(const float * src, void * dst, int64_t k) {
    gk_fp32_to_fp16_row(src, (gk_fp16_t *) dst, k);
}

static void to_float_bf16(const void * src, float * dst, int64_t k) {
    gk_bf16_to_fp32_row((const gk_bf16_t *) src, dst, k);
}

static void from_float_bf16(const float * src, void * dst, int64_t k) {
    gk_fp32_to_bf16_row(src, (gk_bf16_t *) dst, k);
}

static void to_float_f64(const void * src, float * dst, int64_t k) {
    const double * s = (const double *) src;
    for (int64_t i = 0; i < k; ++i) {
        dst[i] = (float) s[i];
    }
}

static void from_float_f64(const float * src, void * dst, int64_t k) {
    double * d = (double *) dst;
    for (int64_t i = 0; i < k; ++i) {
        d[i] = (double) src[i];
    }
}

// The integer types are storage, not arithmetic - nothing multiplies them, so
// they only need to be readable as floats for casts and for debug dumps.
#define GK_INT_ROW(name, ctype) \
    static void to_float_##name(const void * src, float * dst, int64_t k) { \
        const ctype * s = (const ctype *) src; \
        for (int64_t i = 0; i < k; ++i) { \
            dst[i] = (float) s[i]; \
        } \
    } \
    static void from_float_##name(const float * src, void * dst, int64_t k) { \
        ctype * d = (ctype *) dst; \
        for (int64_t i = 0; i < k; ++i) { \
            d[i] = (ctype) src[i]; \
        } \
    }

GK_INT_ROW(i8,  int8_t)
GK_INT_ROW(i16, int16_t)
GK_INT_ROW(i32, int32_t)
GK_INT_ROW(i64, int64_t)

// Quantized rows. `gk_quant_type` is threaded through by the table entry so
// one pair of functions serves every format the codec knows.
#define GK_QUANT_ROW(name, type_id) \
    static void to_float_##name(const void * src, float * dst, int64_t k) { \
        const bool ok = qz_dequantize((qz_type) (type_id), src, dst, k); \
        GK_ASSERT(ok); \
    } \
    static void from_float_##name(const float * src, void * dst, int64_t k) { \
        qz_quantize_init((qz_type) (type_id)); \
        qz_quantize_chunk((qz_type) (type_id), src, dst, 0, 1, k, NULL); \
    }

GK_QUANT_ROW(q1_0,    GK_TYPE_Q1_0)
GK_QUANT_ROW(q2_0,    GK_TYPE_Q2_0)
GK_QUANT_ROW(q4_0,    GK_TYPE_Q4_0)
GK_QUANT_ROW(q4_1,    GK_TYPE_Q4_1)
GK_QUANT_ROW(q5_0,    GK_TYPE_Q5_0)
GK_QUANT_ROW(q5_1,    GK_TYPE_Q5_1)
GK_QUANT_ROW(q8_0,    GK_TYPE_Q8_0)
GK_QUANT_ROW(q2_k,    GK_TYPE_Q2_K)
GK_QUANT_ROW(q3_k,    GK_TYPE_Q3_K)
GK_QUANT_ROW(q4_k,    GK_TYPE_Q4_K)
GK_QUANT_ROW(q5_k,    GK_TYPE_Q5_K)
GK_QUANT_ROW(q6_k,    GK_TYPE_Q6_K)
GK_QUANT_ROW(iq2_xxs, GK_TYPE_IQ2_XXS)
GK_QUANT_ROW(iq2_xs,  GK_TYPE_IQ2_XS)
GK_QUANT_ROW(iq2_s,   GK_TYPE_IQ2_S)
GK_QUANT_ROW(iq3_xxs, GK_TYPE_IQ3_XXS)
GK_QUANT_ROW(iq3_s,   GK_TYPE_IQ3_S)
GK_QUANT_ROW(iq1_s,   GK_TYPE_IQ1_S)
GK_QUANT_ROW(iq1_m,   GK_TYPE_IQ1_M)
GK_QUANT_ROW(iq4_nl,  GK_TYPE_IQ4_NL)
GK_QUANT_ROW(iq4_xs,  GK_TYPE_IQ4_XS)
GK_QUANT_ROW(tq1_0,   GK_TYPE_TQ1_0)
GK_QUANT_ROW(tq2_0,   GK_TYPE_TQ2_0)
GK_QUANT_ROW(mxfp4,   GK_TYPE_MXFP4)
GK_QUANT_ROW(nvfp4,   GK_TYPE_NVFP4)

// --------------------------------------------------------------------------
// dot products
//
// `vec_dot_type` says what the activation side of a matmul is converted to
// before the dot runs. Every quantized format currently reports F32 and uses
// the generic dot below, which widens the weight row and multiplies in float.
// That is the reference: it is correct for every format the codec can decode,
// and it is what the integer fast paths are checked against as they land, one
// format at a time, by changing a single traits entry.
// --------------------------------------------------------------------------

static void vec_dot_f32(int n, float * s, size_t bs,
                        const void * vx, size_t bx,
                        const void * vy, size_t by, int nrc) {
    GK_ASSERT(nrc == 1);
    GK_UNUSED(bs); GK_UNUSED(bx); GK_UNUSED(by);

    *s = gk_dot_f32(n, (const float *) vx, (const float *) vy);
}

// f16 weights against f32 activations. Where the hardware converts a whole
// vector of halves in one instruction, that is the difference between f16
// weights being free to read and being the bottleneck; where it does not, this
// falls back to converting one at a time.
static void vec_dot_f16(int n, float * s, size_t bs,
                        const void * vx, size_t bx,
                        const void * vy, size_t by, int nrc) {
    GK_ASSERT(nrc == 1);
    GK_UNUSED(bs); GK_UNUSED(bx); GK_UNUSED(by);

    const gk_fp16_t * x = (const gk_fp16_t *) vx;
    const float     * y = (const float *) vy;

    int64_t i = 0;
    float sum = 0.0f;

#if GK_SIMD_HAVE_F16 && GK_SIMD_F32_STEP > 1
    gk_f32x acc[GK_SIMD_ACC];
    for (int a = 0; a < GK_SIMD_ACC; ++a) {
        acc[a] = gk_f32x_zero();
    }

    for (; i + GK_SIMD_BLOCK <= n; i += GK_SIMD_BLOCK) {
        for (int a = 0; a < GK_SIMD_ACC; ++a) {
            const int64_t off = i + (int64_t) a * GK_SIMD_F32_STEP;
            acc[a] = gk_f32x_fma(acc[a], gk_f32x_load_f16(x + off), gk_f32x_load(y + off));
        }
    }
    for (; i + GK_SIMD_F32_STEP <= n; i += GK_SIMD_F32_STEP) {
        acc[0] = gk_f32x_fma(acc[0], gk_f32x_load_f16(x + i), gk_f32x_load(y + i));
    }

    for (int stride = GK_SIMD_ACC / 2; stride > 0; stride /= 2) {
        for (int a = 0; a < stride; ++a) {
            acc[a] = gk_f32x_add(acc[a], acc[a + stride]);
        }
    }
    sum = gk_f32x_reduce(acc[0]);
#endif

    for (; i < n; ++i) {
        sum += gk_fp16_to_fp32(x[i]) * y[i];
    }
    *s = sum;
}

static void vec_dot_bf16(int n, float * s, size_t bs,
                         const void * vx, size_t bx,
                         const void * vy, size_t by, int nrc) {
    GK_ASSERT(nrc == 1);
    GK_UNUSED(bs); GK_UNUSED(bx); GK_UNUSED(by);

    const gk_bf16_t * x = (const gk_bf16_t *) vx;
    const float     * y = (const float *) vy;

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double) gk_bf16_to_fp32(x[i]) * (double) y[i];
    }
    *s = (float) sum;
}

// Widens the weight row a block at a time and multiplies in float. The window
// is a fixed stack buffer so this stays reentrant and allocation-free; it is
// sized to hold a whole super-block of any format the codec defines.
#define GK_DOT_WINDOW 256

static void vec_dot_quant(enum gk_type type, int n, float * s,
                          const void * vx, const void * vy) {
    const struct gk_type_traits * tr = gk_get_type_traits(type);

    const int64_t blck = tr->blck_size;
    GK_ASSERT(n % blck == 0);
    GK_ASSERT(GK_DOT_WINDOW % blck == 0);

    const char  * x = (const char *) vx;
    const float * y = (const float *) vy;

    float buf[GK_DOT_WINDOW];

    // The inner product over a window runs vectorised in float; the window
    // partials are summed in double. One double add per 256 elements costs
    // nothing and keeps this path's accuracy, which is the reason the integer
    // fast paths have something worth being checked against.
    double sum = 0.0;
    for (int64_t i = 0; i < n; i += GK_DOT_WINDOW) {
        const int64_t k = GK_MIN((int64_t) GK_DOT_WINDOW, n - i);

        tr->to_float(x + (i / blck) * tr->type_size, buf, k);

        sum += (double) gk_dot_f32(k, buf, y + i);
    }
    *s = (float) sum;
}

#define GK_QUANT_DOT(name, type_id) \
    static void vec_dot_##name(int n, float * s, size_t bs, \
                               const void * vx, size_t bx, \
                               const void * vy, size_t by, int nrc) { \
        GK_ASSERT(nrc == 1); \
        GK_UNUSED(bs); GK_UNUSED(bx); GK_UNUSED(by); \
        vec_dot_quant((enum gk_type) (type_id), n, s, vx, vy); \
    }

GK_QUANT_DOT(q1_0,    GK_TYPE_Q1_0)
GK_QUANT_DOT(q2_0,    GK_TYPE_Q2_0)
GK_QUANT_DOT(q4_0,    GK_TYPE_Q4_0)
GK_QUANT_DOT(q4_1,    GK_TYPE_Q4_1)
GK_QUANT_DOT(q5_0,    GK_TYPE_Q5_0)
GK_QUANT_DOT(q5_1,    GK_TYPE_Q5_1)
GK_QUANT_DOT(q8_0,    GK_TYPE_Q8_0)
GK_QUANT_DOT(q2_k,    GK_TYPE_Q2_K)
GK_QUANT_DOT(q3_k,    GK_TYPE_Q3_K)
GK_QUANT_DOT(q4_k,    GK_TYPE_Q4_K)
GK_QUANT_DOT(q5_k,    GK_TYPE_Q5_K)
GK_QUANT_DOT(q6_k,    GK_TYPE_Q6_K)
GK_QUANT_DOT(iq2_xxs, GK_TYPE_IQ2_XXS)
GK_QUANT_DOT(iq2_xs,  GK_TYPE_IQ2_XS)
GK_QUANT_DOT(iq2_s,   GK_TYPE_IQ2_S)
GK_QUANT_DOT(iq3_xxs, GK_TYPE_IQ3_XXS)
GK_QUANT_DOT(iq3_s,   GK_TYPE_IQ3_S)
GK_QUANT_DOT(iq1_s,   GK_TYPE_IQ1_S)
GK_QUANT_DOT(iq1_m,   GK_TYPE_IQ1_M)
GK_QUANT_DOT(iq4_nl,  GK_TYPE_IQ4_NL)
GK_QUANT_DOT(iq4_xs,  GK_TYPE_IQ4_XS)
GK_QUANT_DOT(tq1_0,   GK_TYPE_TQ1_0)
GK_QUANT_DOT(tq2_0,   GK_TYPE_TQ2_0)
GK_QUANT_DOT(mxfp4,   GK_TYPE_MXFP4)
GK_QUANT_DOT(nvfp4,   GK_TYPE_NVFP4)

// --------------------------------------------------------------------------
// the table
//
// blck_size and type_size are taken from the codec at first use rather than
// repeated here, so there is exactly one place where a block's size is stated.
// --------------------------------------------------------------------------

#define GK_TRAITS_FLOAT(id, nm, ctype, sfx) \
    [id] = { .name = nm, .blck_size = 1, .type_size = sizeof(ctype), \
             .is_quantized = false, \
             .to_float = to_float_##sfx, .from_float = from_float_##sfx, \
             .vec_dot_type = GK_TYPE_F32, .vec_dot = vec_dot_##sfx, .nrows = 1 }

// Types that are storage only: readable as floats for casts and dumps, but
// never an operand of a matmul, so they carry no dot.
#define GK_TRAITS_STORAGE(id, nm, ctype, sfx) \
    [id] = { .name = nm, .blck_size = 1, .type_size = sizeof(ctype), \
             .is_quantized = false, \
             .to_float = to_float_##sfx, .from_float = from_float_##sfx, \
             .vec_dot_type = GK_TYPE_F32, .vec_dot = NULL, .nrows = 1 }

#define GK_TRAITS_QUANT(id, nm, sfx) \
    [id] = { .name = nm, .blck_size = 0, .type_size = 0, \
             .is_quantized = true, \
             .to_float = to_float_##sfx, .from_float = from_float_##sfx, \
             .vec_dot_type = GK_TYPE_F32, .vec_dot = vec_dot_##sfx, .nrows = 1 }

static struct gk_type_traits g_traits[GK_TYPE_COUNT] = {
    GK_TRAITS_FLOAT(GK_TYPE_F32,  "f32",  float,     f32),
    GK_TRAITS_FLOAT(GK_TYPE_F16,  "f16",  gk_fp16_t, f16),
    GK_TRAITS_FLOAT(GK_TYPE_BF16, "bf16", gk_bf16_t, bf16),
    GK_TRAITS_STORAGE(GK_TYPE_F64, "f64", double,  f64),

    GK_TRAITS_STORAGE(GK_TYPE_I8,  "i8",  int8_t,  i8),
    GK_TRAITS_STORAGE(GK_TYPE_I16, "i16", int16_t, i16),
    GK_TRAITS_STORAGE(GK_TYPE_I32, "i32", int32_t, i32),
    GK_TRAITS_STORAGE(GK_TYPE_I64, "i64", int64_t, i64),

    [GK_TYPE_Q1_0] = { .name = "q1_0", .is_quantized = true,
                    .to_float = to_float_q1_0, .from_float = from_float_q1_0,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_q1_0_q8_0, .nrows = 1 },
    [GK_TYPE_Q2_0] = { .name = "q2_0", .is_quantized = true,
                    .to_float = to_float_q2_0, .from_float = from_float_q2_0,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_q2_0_q8_0, .nrows = 1 },
    [GK_TYPE_Q4_0] = { .name = "q4_0", .is_quantized = true,
                    .to_float = to_float_q4_0, .from_float = from_float_q4_0,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_q4_0_q8_0, .nrows = 1 },
    [GK_TYPE_Q4_1] = { .name = "q4_1", .is_quantized = true,
                    .to_float = to_float_q4_1, .from_float = from_float_q4_1,
                    .vec_dot_type = GK_TYPE_Q8_1, .vec_dot = gk_vec_dot_q4_1_q8_1, .nrows = 1 },
    [GK_TYPE_Q5_0] = { .name = "q5_0", .is_quantized = true,
                    .to_float = to_float_q5_0, .from_float = from_float_q5_0,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_q5_0_q8_0, .nrows = 1 },
    [GK_TYPE_Q5_1] = { .name = "q5_1", .is_quantized = true,
                    .to_float = to_float_q5_1, .from_float = from_float_q5_1,
                    .vec_dot_type = GK_TYPE_Q8_1, .vec_dot = gk_vec_dot_q5_1_q8_1, .nrows = 1 },
    [GK_TYPE_Q8_0] = { .name = "q8_0", .is_quantized = true,
                    .to_float = to_float_q8_0, .from_float = from_float_q8_0,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_q8_0_q8_0, .nrows = 1 },

    [GK_TYPE_Q2_K] = { .name = "q2_K", .is_quantized = true,
                    .to_float = to_float_q2_k, .from_float = from_float_q2_k,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_q2_K_q8_K, .nrows = 1 },
    [GK_TYPE_Q3_K] = { .name = "q3_K", .is_quantized = true,
                    .to_float = to_float_q3_k, .from_float = from_float_q3_k,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_q3_K_q8_K, .nrows = 1 },
    [GK_TYPE_Q4_K] = { .name = "q4_K", .is_quantized = true,
                    .to_float = to_float_q4_k, .from_float = from_float_q4_k,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_q4_K_q8_K, .nrows = 1 },
    [GK_TYPE_Q5_K] = { .name = "q5_K", .is_quantized = true,
                    .to_float = to_float_q5_k, .from_float = from_float_q5_k,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_q5_K_q8_K, .nrows = 1 },
    [GK_TYPE_Q6_K] = { .name = "q6_K", .is_quantized = true,
                    .to_float = to_float_q6_k, .from_float = from_float_q6_k,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_q6_K_q8_K, .nrows = 1 },

    [GK_TYPE_IQ2_XXS] = { .name = "iq2_xxs", .is_quantized = true,
                    .to_float = to_float_iq2_xxs, .from_float = from_float_iq2_xxs,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq2_xxs_q8_K, .nrows = 1 },
    [GK_TYPE_IQ2_XS] = { .name = "iq2_xs", .is_quantized = true,
                    .to_float = to_float_iq2_xs, .from_float = from_float_iq2_xs,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq2_xs_q8_K, .nrows = 1 },
    [GK_TYPE_IQ2_S] = { .name = "iq2_s", .is_quantized = true,
                    .to_float = to_float_iq2_s, .from_float = from_float_iq2_s,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq2_s_q8_K, .nrows = 1 },
    [GK_TYPE_IQ3_XXS] = { .name = "iq3_xxs", .is_quantized = true,
                    .to_float = to_float_iq3_xxs, .from_float = from_float_iq3_xxs,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq3_xxs_q8_K, .nrows = 1 },
    [GK_TYPE_IQ3_S] = { .name = "iq3_s", .is_quantized = true,
                    .to_float = to_float_iq3_s, .from_float = from_float_iq3_s,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq3_s_q8_K, .nrows = 1 },
    [GK_TYPE_IQ1_S] = { .name = "iq1_s", .is_quantized = true,
                    .to_float = to_float_iq1_s, .from_float = from_float_iq1_s,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq1_s_q8_K, .nrows = 1 },
    [GK_TYPE_IQ1_M] = { .name = "iq1_m", .is_quantized = true,
                    .to_float = to_float_iq1_m, .from_float = from_float_iq1_m,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq1_m_q8_K, .nrows = 1 },
    [GK_TYPE_IQ4_NL] = { .name = "iq4_nl", .is_quantized = true,
                    .to_float = to_float_iq4_nl, .from_float = from_float_iq4_nl,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_iq4_nl_q8_0, .nrows = 1 },
    [GK_TYPE_IQ4_XS] = { .name = "iq4_xs", .is_quantized = true,
                    .to_float = to_float_iq4_xs, .from_float = from_float_iq4_xs,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_iq4_xs_q8_K, .nrows = 1 },

    [GK_TYPE_TQ1_0] = { .name = "tq1_0", .is_quantized = true,
                    .to_float = to_float_tq1_0, .from_float = from_float_tq1_0,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_tq1_0_q8_K, .nrows = 1 },
    [GK_TYPE_TQ2_0] = { .name = "tq2_0", .is_quantized = true,
                    .to_float = to_float_tq2_0, .from_float = from_float_tq2_0,
                    .vec_dot_type = GK_TYPE_Q8_K, .vec_dot = gk_vec_dot_tq2_0_q8_K, .nrows = 1 },

    [GK_TYPE_MXFP4] = { .name = "mxfp4", .is_quantized = true,
                    .to_float = to_float_mxfp4, .from_float = from_float_mxfp4,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_mxfp4_q8_0, .nrows = 1 },
    [GK_TYPE_NVFP4] = { .name = "nvfp4", .is_quantized = true,
                    .to_float = to_float_nvfp4, .from_float = from_float_nvfp4,
                    .vec_dot_type = GK_TYPE_Q8_0, .vec_dot = gk_vec_dot_nvfp4_q8_0, .nrows = 1 },

    // Q8_1 and Q8_K are activation-side intermediates for the integer dots.
    // They never appear in a file, so they have an encoder but no decoder: a
    // matmul converts activations into them and nothing ever reads one back.
    [GK_TYPE_Q8_1] = { .name = "q8_1", .is_quantized = true,
                       .from_float = gk_quantize_row_q8_1,
                       .vec_dot_type = GK_TYPE_F32, .nrows = 1 },
    [GK_TYPE_Q8_K] = { .name = "q8_K", .is_quantized = true,
                       .from_float = gk_quantize_row_q8_K,
                       .vec_dot_type = GK_TYPE_F32, .nrows = 1 },
};

// The codec owns the block geometry; copy it in once so the hot paths read a
// plain field instead of calling through.
static void gk_traits_init(void) {
    static bool done = false;
    if (done) {
        return;
    }

    for (int i = 0; i < GK_TYPE_COUNT; ++i) {
        if (g_traits[i].name == NULL) {
            continue;
        }
        const int64_t blck = qz_blck_size((qz_type) i);
        if (blck > 0) {
            g_traits[i].blck_size = blck;
            g_traits[i].type_size = qz_type_size((qz_type) i);
        }
    }

    done = true;
}

const struct gk_type_traits * gk_get_type_traits(enum gk_type type) {
    GK_ASSERT((int) type >= 0 && (int) type < GK_TYPE_COUNT);
    gk_traits_init();
    return &g_traits[type];
}

// --------------------------------------------------------------------------
// public type queries
// --------------------------------------------------------------------------

const char * gk_type_name(enum gk_type type) {
    const char * name = gk_get_type_traits(type)->name;
    return name ? name : "unknown";
}

int64_t gk_blck_size(enum gk_type type) {
    return gk_get_type_traits(type)->blck_size;
}

size_t gk_type_size(enum gk_type type) {
    return gk_get_type_traits(type)->type_size;
}

bool gk_is_quantized(enum gk_type type) {
    return gk_get_type_traits(type)->is_quantized;
}

size_t gk_row_size(enum gk_type type, int64_t ne) {
    const struct gk_type_traits * tr = gk_get_type_traits(type);
    GK_ASSERT(ne % tr->blck_size == 0);
    return tr->type_size * ne / tr->blck_size;
}
