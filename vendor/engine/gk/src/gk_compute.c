// The CPU compute pass: evaluates a finished graph, one node at a time.
//
// Every kernel here is scalar and correctness-first. That is deliberate, and
// it is the order the whole library is being built in: this pass is the
// definition of what each op means, and the SIMD and GPU paths that come later
// are checked against it. Nothing in this file should be clever.
//
// One shape runs through all of it. Kernels work a row at a time, and they
// reach a row through two helpers rather than by indexing raw memory:
//
//     gk_row_read   - hand me row (i1,i2,i3) of this tensor as f32
//     gk_row_write  - store this f32 row back into that tensor
//
// Those two absorb every combination of storage the rest of the code would
// otherwise have to care about: f32 passes through untouched, f16/bf16 convert,
// a quantized row is decoded through the shared codec, and a permuted or
// strided tensor is gathered element by element. The cost is a copy per row on
// the paths that could have been read in place; the benefit is that each op is
// written once and is correct for every type it can legally receive, which is
// what makes the differential harness able to sweep the whole matrix.
//
// Ops that only reinterpret shape - reshape, view, permute, transpose - do no
// work at all. Their results already alias the operand's memory, so the pass
// skips them.

#include "gk_impl.h"
#include "gk_simd.h"

#include <float.h>
#include <math.h>

// MSVC does not define the common mathematical constants from <math.h>
// unless _USE_MATH_DEFINES is set before including that header.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --------------------------------------------------------------------------
// row access
// --------------------------------------------------------------------------

// Byte address of the first element of row (i1,i2,i3).
static inline const char * gk_row_ptr(const struct gk_tensor * t,
                                      int64_t i1, int64_t i2, int64_t i3) {
    return (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}

static inline char * gk_row_ptr_mut(struct gk_tensor * t,
                                    int64_t i1, int64_t i2, int64_t i3) {
    return (char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}

// True when a row's elements sit one after another, so the whole row can be
// converted in a single call instead of gathered.
static inline bool gk_row_is_packed(const struct gk_tensor * t) {
    return t->nb[0] == gk_type_size(t->type);
}

// Reads row (i1,i2,i3) as f32 into `buf`, which must hold ne[0] floats.
// Returns a pointer to the floats - `buf` itself, or the tensor's own memory
// when it is already f32 and packed, in which case nothing is copied.
static const float * gk_row_read(const struct gk_tensor * t,
                                 int64_t i1, int64_t i2, int64_t i3, float * buf) {
    const char * p = gk_row_ptr(t, i1, i2, i3);

    if (t->type == GK_TYPE_F32 && gk_row_is_packed(t)) {
        return (const float *) p;
    }

    const struct gk_type_traits * tr = gk_get_type_traits(t->type);
    GK_ASSERT(tr->to_float != NULL);

    if (gk_row_is_packed(t)) {
        tr->to_float(p, buf, t->ne[0]);
        return buf;
    }

    // Strided: the row was produced by a permute or a view with a gap, so the
    // elements have to be picked up one at a time. A block type cannot be
    // addressed element-wise, so this path is float storage only - which is
    // the only thing that can be strided in practice, since a permute never
    // moves quantized weights.
    GK_ASSERT(!tr->is_quantized);

    for (int64_t i = 0; i < t->ne[0]; ++i) {
        tr->to_float(p + i * t->nb[0], buf + i, 1);
    }
    return buf;
}

// Stores `src` (ne[0] floats) into row (i1,i2,i3).
static void gk_row_write(struct gk_tensor * t,
                         int64_t i1, int64_t i2, int64_t i3, const float * src) {
    char * p = gk_row_ptr_mut(t, i1, i2, i3);

    if (t->type == GK_TYPE_F32 && gk_row_is_packed(t)) {
        if ((const float *) p != src) {
            memcpy(p, src, (size_t) t->ne[0] * sizeof(float));
        }
        return;
    }

    const struct gk_type_traits * tr = gk_get_type_traits(t->type);
    GK_ASSERT(tr->from_float != NULL);

    if (gk_row_is_packed(t)) {
        tr->from_float(src, p, t->ne[0]);
        return;
    }

    GK_ASSERT(!tr->is_quantized);

    for (int64_t i = 0; i < t->ne[0]; ++i) {
        tr->from_float(src + i, p + i * t->nb[0], 1);
    }
}

// --------------------------------------------------------------------------
// scratch
//
// Kernels need a few rows of float workspace. The pass owns one block of it
// and hands out fixed slices, sized from the widest row in the graph, so no
// kernel allocates.
// --------------------------------------------------------------------------

#define GK_SCRATCH_ROWS 5

struct gk_compute_state {
    float * scratch;  // nth * GK_SCRATCH_ROWS slices
    int64_t row_size; // floats per slice
    int     ith;
    int     nth;
};

// Each thread gets its own set of slices, so a kernel never has to think about
// whether its workspace is shared.
static float * gk_scratch(struct gk_compute_state * st, int slot) {
    GK_ASSERT(slot >= 0 && slot < GK_SCRATCH_ROWS);
    const int64_t idx = (int64_t) st->ith * GK_SCRATCH_ROWS + slot;
    return st->scratch + idx * st->row_size;
}

// A slice reinterpreted as bytes, for holding a row of quantized activations.
// Every quantized format encodes an element in under four bytes - q8_K, the
// widest, uses 292 per 256 - so a slice sized for f32 always has room.
static void * gk_scratch_bytes(struct gk_compute_state * st, int slot, size_t need) {
    GK_ASSERT(need <= (size_t) st->row_size * sizeof(float));
    return (void *) gk_scratch(st, slot);
}

// --------------------------------------------------------------------------
// work partitioning
//
// Almost every kernel here writes one destination row at a time and each row
// is independent, so rows are the unit of work. Splitting them contiguously
// rather than round-robin keeps each thread on its own stretch of memory.
// --------------------------------------------------------------------------

static void gk_rows_for_thread(const struct gk_tensor * t, int ith, int nth,
                               int64_t * ir0, int64_t * ir1) {
    const int64_t nr  = t->ne[1] * t->ne[2] * t->ne[3];
    const int64_t per = (nr + nth - 1) / nth;

    *ir0 = per * ith;
    *ir1 = GK_MIN(*ir0 + per, nr);

    if (*ir0 > nr) {
        *ir0 = nr;
    }
}

// --------------------------------------------------------------------------
// elementwise binary
//
// `src1` broadcasts onto `src0`: each of its dimensions divides the matching
// one, and an index wraps with a modulo. Dimension 0 wraps too, which is what
// makes a per-row bias or a per-channel scale work without materialising it.
// --------------------------------------------------------------------------

enum gk_binary_kind { GK_BIN_ADD, GK_BIN_SUB, GK_BIN_MUL, GK_BIN_DIV };

static void gk_compute_binary(struct gk_compute_state * st,
                              struct gk_tensor * dst, enum gk_binary_kind kind) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    GK_ASSERT(gk_can_repeat(src1, src0));
    GK_ASSERT(gk_are_same_shape(src0, dst));

    float * a   = gk_scratch(st, 0);
    float * b   = gk_scratch(st, 1);
    float * out = gk_scratch(st, 2);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);
                const float * pb = gk_row_read(src1,
                                               i1 % src1->ne[1],
                                               i2 % src1->ne[2],
                                               i3 % src1->ne[3], b);

                const int64_t n1 = src1->ne[0];
                const int64_t n  = dst->ne[0];

                // The common case is operands of equal width, where the index
                // wrap is a no-op and the whole row is one vector loop. A
                // narrower operand still has to wrap per element.
                if (n1 == n) {
                    switch (kind) {
                        case GK_BIN_ADD: gk_vec_add_f32(n, pa, pb, out); break;
                        case GK_BIN_SUB: gk_vec_sub_f32(n, pa, pb, out); break;
                        case GK_BIN_MUL: gk_vec_mul_f32(n, pa, pb, out); break;
                        case GK_BIN_DIV:
                            for (int64_t i = 0; i < n; ++i) out[i] = pa[i] / pb[i];
                            break;
                    }
                } else {
                    switch (kind) {
                        case GK_BIN_ADD:
                            for (int64_t i = 0; i < n; ++i) out[i] = pa[i] + pb[i % n1];
                            break;
                        case GK_BIN_SUB:
                            for (int64_t i = 0; i < n; ++i) out[i] = pa[i] - pb[i % n1];
                            break;
                        case GK_BIN_MUL:
                            for (int64_t i = 0; i < n; ++i) out[i] = pa[i] * pb[i % n1];
                            break;
                        case GK_BIN_DIV:
                            for (int64_t i = 0; i < n; ++i) out[i] = pa[i] / pb[i % n1];
                            break;
                    }
                }

                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

// --------------------------------------------------------------------------
// elementwise unary
// --------------------------------------------------------------------------

// The error function, needed by the exact GELU. Written out rather than taken
// from libm because erff is not available everywhere we build, and this is a
// reference path where a few extra operations do not matter.
//
// Abramowitz and Stegun 7.1.26: a degree-5 polynomial in 1/(1+px) with a
// Gaussian factor, good to about 1.5e-7 absolute, which is below f32 rounding
// for the range GELU cares about.
static float gk_act_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabsf(x);

    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    const float t = 1.0f / (1.0f + p * x);
    const float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * expf(-x * x);

    return sign * y;
}

// The tanh approximation of GELU. This is the form the published weights were
// trained against, so it is the default rather than the exact one.
static float gk_act_gelu(float x) {
    const float c = 0.797884560802865f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static float gk_act_gelu_erf(float x) {
    return 0.5f * x * (1.0f + gk_act_erf(x * 0.7071067811865475f)); // x/sqrt(2)
}

static float gk_act_gelu_quick(float x) {
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

static float gk_act_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float gk_apply_unary(enum gk_unary_op op, float x) {
    switch (op) {
        case GK_UNARY_OP_ABS:         return fabsf(x);
        case GK_UNARY_OP_SGN:         return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        case GK_UNARY_OP_NEG:         return -x;
        case GK_UNARY_OP_STEP:        return x > 0.0f ? 1.0f : 0.0f;
        case GK_UNARY_OP_TANH:        return tanhf(x);
        case GK_UNARY_OP_ELU:         return x > 0.0f ? x : expm1f(x);
        case GK_UNARY_OP_RELU:        return x > 0.0f ? x : 0.0f;
        case GK_UNARY_OP_SIGMOID:     return 1.0f / (1.0f + expf(-x));
        case GK_UNARY_OP_GELU:        return gk_act_gelu(x);
        case GK_UNARY_OP_GELU_QUICK:  return gk_act_gelu_quick(x);
        case GK_UNARY_OP_GELU_ERF:    return gk_act_gelu_erf(x);
        case GK_UNARY_OP_SILU:        return gk_act_silu(x);
        case GK_UNARY_OP_HARDSWISH:   return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
        case GK_UNARY_OP_HARDSIGMOID: return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
        case GK_UNARY_OP_EXP:         return expf(x);
        case GK_UNARY_OP_EXPM1:       return expm1f(x);
        case GK_UNARY_OP_SOFTPLUS:    return x > 20.0f ? x : logf(1.0f + expf(x));
        case GK_UNARY_OP_FLOOR:       return floorf(x);
        case GK_UNARY_OP_CEIL:        return ceilf(x);
        case GK_UNARY_OP_ROUND:       return rintf(x);
        case GK_UNARY_OP_TRUNC:       return truncf(x);
        default:
            GK_ABORT("unary op %s is not implemented", gk_unary_op_name(op));
    }
}

// xielu's constants arrive pre-constrained in the op params: [1] and [2] are
// already softplus'd at build time, so the kernel is two multiplies per side.
static float gk_apply_xielu(const struct gk_tensor * dst, float x) {
    const float alpha_n = gk_get_op_params_f32(dst, 1);
    const float alpha_p = gk_get_op_params_f32(dst, 2);
    const float beta    = gk_get_op_params_f32(dst, 3);
    const float eps     = gk_get_op_params_f32(dst, 4);

    if (x > 0.0f) {
        return alpha_p * x * x + beta * x;
    }
    const float mx = fminf(x, eps);
    return (expm1f(mx) - x) * alpha_n + beta * x;
}

static void gk_compute_unary(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const enum gk_unary_op op = gk_get_unary_op(dst);

    GK_ASSERT(gk_are_same_shape(src0, dst));

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);
                if (op == GK_UNARY_OP_XIELU) {
                    for (int64_t i = 0; i < dst->ne[0]; ++i) {
                        out[i] = gk_apply_xielu(dst, pa[i]);
                    }
                } else {
                    for (int64_t i = 0; i < dst->ne[0]; ++i) {
                        out[i] = gk_apply_unary(op, pa[i]);
                    }
                }
                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

static void gk_compute_leaky_relu(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const float slope = gk_get_op_params_f32(dst, 0);

    GK_ASSERT(gk_are_same_shape(src0, dst));

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const float * pa = gk_row_read(src0, i1, i2, i3, a);
        for (int64_t i = 0; i < dst->ne[0]; ++i) {
            out[i] = pa[i] > 0.0f ? pa[i] : pa[i] * slope;
        }
        gk_row_write(dst, i1, i2, i3, out);
    }
}

// sqr/sqrt/log/sin/cos carry their own op ids, so they dispatch separately
static void gk_compute_simple_unary(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(gk_are_same_shape(src0, dst));

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);
                for (int64_t i = 0; i < dst->ne[0]; ++i) {
                    switch (dst->op) {
                        case GK_OP_SQR:  out[i] = pa[i] * pa[i];  break;
                        case GK_OP_SQRT: out[i] = sqrtf(pa[i]);   break;
                        case GK_OP_LOG:  out[i] = logf(pa[i]);    break;
                        case GK_OP_SIN:  out[i] = sinf(pa[i]);    break;
                        case GK_OP_COS:  out[i] = cosf(pa[i]);    break;
                        default: GK_ABORT("not a simple unary: %s", gk_op_name(dst->op));
                    }
                }
                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

// --------------------------------------------------------------------------
// gated linear units
// --------------------------------------------------------------------------

static void gk_compute_glu(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    const enum gk_glu_op op = gk_get_glu_op(dst);

    const int64_t n = dst->ne[0];

    float * a   = gk_scratch(st, 0);
    float * b   = gk_scratch(st, 1);
    float * out = gk_scratch(st, 2);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    const bool swapped = gk_get_op_params_i32(dst, 1) != 0;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);

                // Splitting a single row: the activation is applied to the
                // FIRST half and multiplied by the second, unless `swapped`
                // exchanges the halves. Getting this round the wrong way
                // produces plausible-looking output that is silently wrong,
                // so it is worth stating explicitly:
                //
                //     out = act(row[0..n]) * row[n..2n]
                //
                // With a separate gate tensor, `src0` is the activated side
                // and `src1` is the multiplier.
                const float * act = swapped ? pa + n : pa;
                const float * mul = swapped ? pa     : pa + n;

                if (src1 != NULL) {
                    act = pa;
                    mul = gk_row_read(src1, i1, i2, i3, b);
                }

                if (op == GK_GLU_OP_SWIGLU_OAI) {
                    const float alpha = gk_get_op_params_f32(dst, 2);
                    const float limit = gk_get_op_params_f32(dst, 3);

                    for (int64_t i = 0; i < n; ++i) {
                        const float x = fminf(act[i], limit);
                        const float y = fminf(fmaxf(mul[i], -limit), limit);
                        const float g = x / (1.0f + expf(alpha * (-x)));
                        out[i] = g * (y + 1.0f);
                    }
                    gk_row_write(dst, i1, i2, i3, out);
                    continue;
                }

                for (int64_t i = 0; i < n; ++i) {
                    const float x = act[i];
                    const float m = mul[i];

                    switch (op) {
                        case GK_GLU_OP_REGLU:       out[i] = (x > 0.0f ? x : 0.0f) * m; break;
                        case GK_GLU_OP_GEGLU:       out[i] = gk_act_gelu(x) * m;        break;
                        case GK_GLU_OP_SWIGLU:      out[i] = gk_act_silu(x) * m;        break;
                        case GK_GLU_OP_GEGLU_ERF:   out[i] = gk_act_gelu_erf(x) * m;    break;
                        case GK_GLU_OP_GEGLU_QUICK: out[i] = gk_act_gelu_quick(x) * m;  break;
                        default:
                            GK_ABORT("glu op %s is not implemented", gk_glu_op_name(op));
                    }
                }

                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

// --------------------------------------------------------------------------
// scale, clamp
// --------------------------------------------------------------------------

static void gk_compute_scale(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const float s    = gk_get_op_params_f32(dst, 0);
    const float bias = gk_get_op_params_f32(dst, 1);

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);
                gk_scale_f32(dst->ne[0], pa, out, s, bias);
                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

static void gk_compute_clamp(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const float lo = gk_get_op_params_f32(dst, 0);
    const float hi = gk_get_op_params_f32(dst, 1);

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);
                for (int64_t i = 0; i < dst->ne[0]; ++i) {
                    out[i] = fminf(hi, fmaxf(lo, pa[i]));
                }
                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

// --------------------------------------------------------------------------
// normalisation
//
// All four normalise along dimension 0. They differ only in what statistic
// they divide by, so they share one loop.
//
// The sums accumulate in double. A row of several thousand activations summed
// in f32 loses enough to shift the variance visibly, and this is the reference
// - the fast paths can make their own accuracy trade, but they get checked
// against this.
// --------------------------------------------------------------------------

static void gk_compute_norm(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const float eps = gk_get_op_params_f32(dst, 0);
    const int64_t n = dst->ne[0];

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);

                if (dst->op == GK_OP_RMS_NORM) {
                    const float sum = gk_sumsq_f32(n, pa);
                    const float scale = 1.0f / sqrtf(sum / (float) n + eps);
                    gk_scale_f32(n, pa, out, scale, 0.0f);
                } else if (dst->op == GK_OP_NORM) {
                    const float mean = gk_sum_f32(n, pa) / (float) n;

                    // The variance is computed about the mean in one pass over
                    // the centred values rather than as E[x^2] - E[x]^2, which
                    // cancels catastrophically when the mean dominates.
                    for (int64_t i = 0; i < n; ++i) {
                        out[i] = pa[i] - mean;
                    }
                    const float var = gk_sumsq_f32(n, out);
                    const float scale = 1.0f / sqrtf(var / (float) n + eps);
                    gk_scale_f32(n, out, out, scale, 0.0f);
                } else { // GK_OP_L2_NORM
                    const float sum = gk_sumsq_f32(n, pa);
                    const float scale = 1.0f / sqrtf(fmaxf(sum, eps));
                    gk_scale_f32(n, pa, out, scale, 0.0f);
                }

                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

// Group norm splits the channels of each image into `n_groups` and normalises
// over each group's whole spatial extent, so its statistic spans dimensions
// 0, 1 and part of 2 - which is why it does not share the loop above.
static void gk_compute_group_norm(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const int   n_groups = gk_get_op_params_i32(dst, 0);
    const float eps      = gk_get_op_params_f32(dst, 1);

    GK_ASSERT(n_groups > 0);
    GK_ASSERT(src0->ne[2] % n_groups == 0);

    const int64_t per_group = src0->ne[2] / n_groups;
    const int64_t n         = dst->ne[0];

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    // The statistic spans a whole group, so a group is the unit of work here
    // rather than a row.
    const int64_t n_units = dst->ne[3] * n_groups;
    const int64_t per_u   = (n_units + st->nth - 1) / st->nth;
    const int64_t u0      = GK_MIN(per_u * st->ith, n_units);
    const int64_t u1      = GK_MIN(u0 + per_u, n_units);

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t i3 = u / n_groups;
        const int     g  = (int) (u % n_groups);
        {
            const int64_t c0 = (int64_t) g * per_group;
            const int64_t c1 = c0 + per_group;

            double sum = 0.0;
            int64_t count = 0;

            for (int64_t i2 = c0; i2 < c1; ++i2) {
                for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                    const float * pa = gk_row_read(src0, i1, i2, i3, a);
                    for (int64_t i = 0; i < n; ++i) {
                        sum += (double) pa[i];
                    }
                    count += n;
                }
            }

            const double mean = sum / (double) count;

            double var = 0.0;
            for (int64_t i2 = c0; i2 < c1; ++i2) {
                for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                    const float * pa = gk_row_read(src0, i1, i2, i3, a);
                    for (int64_t i = 0; i < n; ++i) {
                        const double d = (double) pa[i] - mean;
                        var += d * d;
                    }
                }
            }

            const float scale = 1.0f / sqrtf((float) (var / (double) count) + eps);

            for (int64_t i2 = c0; i2 < c1; ++i2) {
                for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                    const float * pa = gk_row_read(src0, i1, i2, i3, a);
                    for (int64_t i = 0; i < n; ++i) {
                        out[i] = (float) (((double) pa[i] - mean)) * scale;
                    }
                    gk_row_write(dst, i1, i2, i3, out);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// matrix multiply
//
//     dst[i0, i1, i2, i3] = dot( a_row(i0, i2/r2, i3/r3), b_row(i1, i2, i3) )
//
// `a` is the weight and is read along its fastest dimension, which is what
// lets a quantized weight be used without transposing it. The dot itself comes
// from the type's traits, so a format's fast path is picked up here without
// this function changing.
//
// The higher dimensions of `a` broadcast onto `b`'s: grouped-query attention
// has fewer key heads than query heads, and each key head serves a group.
//
// ### Why this is tiled
//
// The obvious loop - for each activation column, dot it against every weight
// row - reads the entire weight matrix once per column. A 4096x4096 q4_K
// weight is about 9 MB, far past any cache, so a 32-column multiply streams
// 9 MB from memory 32 times and runs at memory speed rather than at the
// kernels' speed.
//
// So the column loop is blocked: `GK_GEMM_NC` activation columns are converted
// up front, and each weight row, once loaded, is dotted against all of them
// before moving on. That divides the weight traffic by NC. The activation
// panel is small enough to sit in L1 next to the weight row - which is what
// bounds NC, not the arithmetic.
//
// ### Why the work is split over weight rows too
//
// Splitting the output by rows of `dst` - the unit every other kernel here
// uses - means splitting by *activation column*. When there is one column,
// which is every matmul in single-token decoding, that is one unit of work and
// every thread but the first idles. Measured, threads made that case slower,
// not faster, because the idle threads still pay the barrier.
//
// The unit of work here is therefore a tile of the output: a block of weight
// rows by a block of columns. Both axes divide, so both the decode case and
// the batched case spread across the pool.
//
// Tiles are ordered with the weight-row block varying fastest, so a thread's
// contiguous slice of the tile space usually stays within one column block and
// converts that block's activations once rather than per tile.
//
// None of this changes the arithmetic: every output element is still one
// `vec_dot` over the whole of k, so results stay bit-identical across thread
// counts and block sizes.
// --------------------------------------------------------------------------

// Activation columns per pass over the weight rows. Each loaded weight row is
// used this many times before it can be evicted.
//
// Measured on a 4096x4096, 32-column multiply (GFLOP/s):
//
//     NC     2       4       8
//     f32    15.23   23.48   25.04
//     q4_K   42.16   47.27   46.24
//
// 4 is the choice because 8 needs nine scratch slices instead of five - the
// workspace is sized by the widest row in the graph and multiplied by the
// thread count, so that is not free - and it buys f32 6% while costing the
// quantized formats 2%. Those are what actually gets run.
#define GK_GEMM_NC 4

// Weight rows per tile. This only sets how finely the work divides between
// threads; the arithmetic does not depend on it.
#define GK_GEMM_MR 64

// One scratch slice per panel column, plus one to stage a row being converted.
_Static_assert(GK_GEMM_NC + 1 <= GK_SCRATCH_ROWS, "gemm panel needs more scratch slices");

static void gk_compute_mul_mat(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    GK_ASSERT(src0->ne[0] == src1->ne[0]);
    GK_ASSERT(dst->ne[0] == src0->ne[1]);
    GK_ASSERT(dst->ne[1] == src1->ne[1]);
    GK_ASSERT(dst->type == GK_TYPE_F32);

    const struct gk_type_traits * tr = gk_get_type_traits(src0->type);
    GK_ASSERT(tr->vec_dot != NULL);

    // The dot takes its right-hand operand in `vec_dot_type`. A format on the
    // float path reports f32 and the activations are passed straight through;
    // one on an integer path names q8_0, q8_1 or q8_K, and the activation rows
    // are converted into that here.
    const enum gk_type vdt = tr->vec_dot_type;
    const struct gk_type_traits * vtr = gk_get_type_traits(vdt);

    if (vdt != GK_TYPE_F32) {
        GK_ASSERT(vtr->from_float != NULL);
        GK_ASSERT(src0->ne[0] % vtr->blck_size == 0);
    }

    // Weight rows are read through the raw pointer the dot expects, so they
    // have to be packed. A permuted weight would need a repack first, which is
    // what gk_cont is for.
    GK_ASSERT(gk_row_is_packed(src0));

    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];

    const int64_t k   = src0->ne[0];
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t n_rb = (ne0 + GK_GEMM_MR - 1) / GK_GEMM_MR;
    const int64_t n_cb = (ne1 + GK_GEMM_NC - 1) / GK_GEMM_NC;

    const int64_t n_tiles = n_rb * n_cb * dst->ne[2] * dst->ne[3];

    const int64_t per = (n_tiles + st->nth - 1) / st->nth;
    const int64_t t0  = GK_MIN(per * (int64_t) st->ith, n_tiles);
    const int64_t t1  = GK_MIN(t0 + per, n_tiles);

    // The converted activation panel, and which column block it holds. -1 is
    // not a reachable key, so the first tile always fills it.
    const void * panel[GK_GEMM_NC];
    int64_t panel_key = -1;
    int64_t panel_n   = 0;
    int64_t panel_j0  = 0;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t irb  = t % n_rb;              // weight-row block, fastest
        const int64_t rest = t / n_rb;              // identifies (jc, i2, i3)

        const int64_t jc = rest % n_cb;
        const int64_t i2 = (rest / n_cb) % dst->ne[2];
        const int64_t i3 = rest / (n_cb * dst->ne[2]);

        if (rest != panel_key) {
            panel_key = rest;
            panel_j0  = jc * GK_GEMM_NC;
            panel_n   = GK_MIN((int64_t) GK_GEMM_NC, ne1 - panel_j0);

            for (int64_t j = 0; j < panel_n; ++j) {
                if (vdt == GK_TYPE_F32) {
                    // gk_row_read returns the tensor's own memory when the row
                    // is already f32 and packed, so this usually copies nothing
                    // - but each column still needs its own slice for the case
                    // where it does copy.
                    panel[j] = gk_row_read(src1, panel_j0 + j, i2, i3,
                                           gk_scratch(st, (int) j));
                } else {
                    const float * pb = gk_row_read(src1, panel_j0 + j, i2, i3,
                                                   gk_scratch(st, GK_GEMM_NC));
                    void * q = gk_scratch_bytes(st, (int) j, gk_row_size(vdt, k));
                    vtr->from_float(pb, q, k);
                    panel[j] = q;
                }
            }
        }

        const int64_t i00 = irb * GK_GEMM_MR;
        const int64_t i01 = GK_MIN(i00 + GK_GEMM_MR, ne0);

        char * dp = (char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3];

        for (int64_t i0 = i00; i0 < i01; ++i0) {
            const char * pa = gk_row_ptr(src0, i0, i2 / r2, i3 / r3);

            // The reuse: `pa` is read from memory once and dotted panel_n
            // times, all of it from L1.
            for (int64_t j = 0; j < panel_n; ++j) {
                float s = 0.0f;
                tr->vec_dot((int) k, &s, 0, pa, 0, panel[j], 0, 1);

                *(float *) (dp + i0 * dst->nb[0] + (panel_j0 + j) * dst->nb[1]) = s;
            }
        }
    }
}

// Mixture of experts. `ids` holds, for each token, which experts to route it
// to; every (token, slot) pair picks one expert row-block out of `as`.
static void gk_compute_mul_mat_id(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * as  = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];
    const struct gk_tensor * ids = dst->src[2];

    GK_ASSERT(ids->type == GK_TYPE_I32);
    GK_ASSERT(dst->type == GK_TYPE_F32);

    const struct gk_type_traits * tr = gk_get_type_traits(as->type);
    GK_ASSERT(tr->vec_dot != NULL);
    GK_ASSERT(gk_row_is_packed(as));

    const enum gk_type vdt = tr->vec_dot_type;
    const struct gk_type_traits * vtr = gk_get_type_traits(vdt);

    const int64_t k       = as->ne[0];
    const int64_t n_out   = dst->ne[0];  // rows per expert
    const int64_t n_slots = dst->ne[1];  // experts chosen per token
    const int64_t n_tok   = dst->ne[2];

    float * b   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    const int64_t per_tok = (n_tok + st->nth - 1) / st->nth;
    const int64_t it0 = GK_MIN(per_tok * st->ith, n_tok);
    const int64_t it1 = GK_MIN(it0 + per_tok, n_tok);

    for (int64_t it = it0; it < it1; ++it) {
        // one activation row per token, shared across that token's slots
        const float * pb = gk_row_read(src1, 0, it, 0, b);

        const void * pdot = pb;
        if (vdt != GK_TYPE_F32) {
            void * q = gk_scratch_bytes(st, 2, gk_row_size(vdt, k));
            vtr->from_float(pb, q, k);
            pdot = q;
        }

        for (int64_t is = 0; is < n_slots; ++is) {
            const int32_t * id_row =
                (const int32_t *) gk_row_ptr(ids, it, 0, 0);
            const int32_t expert = id_row[is];

            GK_ASSERT(expert >= 0 && expert < as->ne[2]);

            for (int64_t i0 = 0; i0 < n_out; ++i0) {
                const char * pa = gk_row_ptr(as, i0, expert, 0);

                float s = 0.0f;
                tr->vec_dot((int) k, &s, 0, pa, 0, pdot, 0, 1);
                out[i0] = s;
            }

            gk_row_write(dst, is, it, 0, out);
        }
    }
}

// --------------------------------------------------------------------------
// copy, cont, dup
//
// One kernel: read every row of the source and write it to the matching
// position of the destination, converting type on the way. `cont` is the same
// operation with a contiguous destination, which is exactly what makes a
// permuted tensor usable again.
// --------------------------------------------------------------------------

static void gk_compute_copy(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(gk_nelements(src0) == gk_nelements(dst));

    float * a = gk_scratch(st, 0);

    // Same shape: rows line up, so copy row by row.
    if (gk_are_same_shape(src0, dst)) {
        int64_t ir0, ir1;
        gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t i1 = ir % dst->ne[1];
            const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
            const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

            const float * pa = gk_row_read(src0, i1, i2, i3, a);
            gk_row_write(dst, i1, i2, i3, pa);
        }
        return;
    }

    // Different shape: the copy is defined over the flat element order, so
    // walk the source in order and fill the destination in order. This is the
    // path a reshape-through-copy takes.
    //
    // The walk carries position from one row to the next, so it cannot be
    // split across threads without first computing where each thread would
    // start. It is a rare path - a reshape that cannot be expressed as a view
    // - so one thread does it and the rest wait at the next barrier.
    if (st->ith != 0) {
        return;
    }

    float * out = gk_scratch(st, 1);

    int64_t d1 = 0, d2 = 0, d3 = 0, di = 0;

    for (int64_t i3 = 0; i3 < src0->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);

                for (int64_t i = 0; i < src0->ne[0]; ++i) {
                    out[di++] = pa[i];

                    if (di == dst->ne[0]) {
                        gk_row_write(dst, d1, d2, d3, out);
                        di = 0;
                        if (++d1 == dst->ne[1]) {
                            d1 = 0;
                            if (++d2 == dst->ne[2]) {
                                d2 = 0;
                                ++d3;
                            }
                        }
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// gather and broadcast
// --------------------------------------------------------------------------

static void gk_compute_get_rows(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    GK_ASSERT(src1->type == GK_TYPE_I32);
    GK_ASSERT(dst->ne[0] == src0->ne[0]);

    float * a = gk_scratch(st, 0);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const int32_t * idx = (const int32_t *)
                    ((const char *) src1->data
                     + i1 * src1->nb[0] + i2 * src1->nb[1] + i3 * src1->nb[2]);

                const int64_t r = *idx;
                GK_ASSERT(r >= 0 && r < src0->ne[1]);

                const float * pa = gk_row_read(src0, r, i2, i3, a);
                gk_row_write(dst, i1, i2, i3, pa);
            }
        }
    }
}

static void gk_compute_repeat(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(gk_can_repeat(src0, dst));

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0,
                                               i1 % src0->ne[1],
                                               i2 % src0->ne[2],
                                               i3 % src0->ne[3], a);

                for (int64_t i = 0; i < dst->ne[0]; ++i) {
                    out[i] = pa[i % src0->ne[0]];
                }

                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

static void gk_compute_concat(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    const int dim = gk_get_op_params_i32(dst, 0);

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                // Which operand a position comes from is decided by whether
                // the index along `dim` is past the first operand's extent.
                int64_t j1 = i1, j2 = i2, j3 = i3;
                const struct gk_tensor * src = src0;

                bool from_second = false;
                switch (dim) {
                    case 1: from_second = i1 >= src0->ne[1]; if (from_second) j1 -= src0->ne[1]; break;
                    case 2: from_second = i2 >= src0->ne[2]; if (from_second) j2 -= src0->ne[2]; break;
                    case 3: from_second = i3 >= src0->ne[3]; if (from_second) j3 -= src0->ne[3]; break;
                    default: break; // dim 0 is handled inside the row
                }
                if (from_second) {
                    src = src1;
                }

                if (dim == 0) {
                    const float * pa = gk_row_read(src0, i1, i2, i3, a);
                    memcpy(out, pa, (size_t) src0->ne[0] * sizeof(float));

                    const float * pb = gk_row_read(src1, i1, i2, i3, a);
                    memcpy(out + src0->ne[0], pb, (size_t) src1->ne[0] * sizeof(float));

                    gk_row_write(dst, i1, i2, i3, out);
                } else {
                    const float * pa = gk_row_read(src, j1, j2, j3, a);
                    gk_row_write(dst, i1, i2, i3, pa);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// softmax and masking
//
// The maximum is subtracted before exponentiating so the largest term is
// exactly 1 and nothing overflows - a row of attention logits can otherwise
// reach values where expf saturates.
//
// `max_bias` turns on ALiBi: a per-head linear penalty on distance, folded
// into the mask rather than added as a separate op. The slopes are powers of a
// ratio chosen so the heads span the range geometrically.
// --------------------------------------------------------------------------

// The ALiBi slope for head h. The first 2^floor(log2(n_head)) heads step down
// geometrically; any heads past that power of two interleave a second, √-of-
// the-first sequence. Both branches matter: a model with a non-power-of-two
// head count was trained against exactly this assignment.
static float gk_alibi_slope(float max_bias, int64_t h, int64_t n_head_log2) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float m0 = powf(2.0f, -max_bias / (float) n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);

    return h < n_head_log2
        ? powf(m0, (float) (h + 1))
        : powf(m1, (float) (2 * (h - n_head_log2) + 1));
}

static void gk_compute_soft_max(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0  = dst->src[0];
    const struct gk_tensor * mask  = dst->src[1];
    const struct gk_tensor * sinks = dst->src[2];

    const float scale    = gk_get_op_params_f32(dst, 0);
    const float max_bias = gk_get_op_params_f32(dst, 1);

    const int64_t n = dst->ne[0];

    // ALiBi indexes heads by dimension 2, rounded up to a power of two.
    const int64_t n_head = src0->ne[2];
    int64_t n_head_log2 = 1;
    while (n_head_log2 * 2 <= n_head) {
        n_head_log2 *= 2;
    }

    float * a   = gk_scratch(st, 0);
    float * m   = gk_scratch(st, 1);
    float * out = gk_scratch(st, 2);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            const float slope = gk_alibi_slope(max_bias, i2, n_head_log2);
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);

                const float * pm = NULL;
                if (mask != NULL) {
                    // the mask broadcasts over heads and batches
                    pm = gk_row_read(mask, i1,
                                     i2 % mask->ne[2],
                                     i3 % mask->ne[3], m);
                }

                float max = -INFINITY;
                for (int64_t i = 0; i < n; ++i) {
                    out[i] = pa[i] * scale + (pm ? slope * pm[i] : 0.0f);
                    if (out[i] > max) {
                        max = out[i];
                    }
                }

                // A sink is one extra virtual logit per head, included in the
                // normalisation but absent from the output row.
                float sink = -INFINITY;
                if (sinks != NULL) {
                    sink = ((const float *) sinks->data)[i2];
                    max  = fmaxf(max, sink);
                }

                // A fully masked row is all -inf. Exponentiating that gives
                // 0/0; define it as a uniform zero row rather than NaN, which
                // is what a padded position needs.
                if (!isfinite(max)) {
                    for (int64_t i = 0; i < n; ++i) {
                        out[i] = 0.0f;
                    }
                    gk_row_write(dst, i1, i2, i3, out);
                    continue;
                }

                double sum = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    const float e = expf(out[i] - max);
                    out[i] = e;
                    sum += e;
                }

                if (sinks != NULL) {
                    sum += (double) expf(sink - max);
                }

                const float inv = (float) (1.0 / sum);
                for (int64_t i = 0; i < n; ++i) {
                    out[i] *= inv;
                }

                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

static void gk_compute_diag_mask(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const int n_past = gk_get_op_params_i32(dst, 0);
    const float fill = dst->op == GK_OP_DIAG_MASK_INF ? -INFINITY : 0.0f;

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);

                for (int64_t i = 0; i < dst->ne[0]; ++i) {
                    out[i] = i > n_past + i1 ? fill : pa[i];
                }

                gk_row_write(dst, i1, i2, i3, out);
            }
        }
    }
}

// --------------------------------------------------------------------------
// rotary position embedding
//
// Each pair of channels is rotated by an angle proportional to the position
// and to that pair's frequency. Two pairings are in use and they are not
// interchangeable - a model trained with one produces nonsense under the
// other:
//
//   normal   channel 2i is paired with 2i+1
//   neox     channel i is paired with i + n_dims/2
//
// Channels at or beyond `n_dims` are passed through untouched, which is how
// models that rotate only part of the head dimension work.
//
// YaRN, when `ext_factor` is non-zero, interpolates between the unscaled and
// the frequency-scaled angle, with the blend depending on how many full
// rotations that channel completes over the original context. Low frequencies
// get scaled (they encode absolute position, which has to stretch) and high
// ones do not (they encode local offsets, which should not).
// --------------------------------------------------------------------------

// Number of rotations a channel of the given frequency completes across the
// original context length - the quantity YaRN's blend is defined against.
float gk_rope_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return (float) n_dims * logf((float) n_ctx_orig / (n_rot * 2.0f * (float) M_PI)) /
           (2.0f * logf(base));
}

void gk_rope_corr_dims(int n_dims, int n_ctx_orig, float base,
                              float beta_fast, float beta_slow, float dims[2]) {
    dims[0] = floorf(gk_rope_corr_dim(n_dims, n_ctx_orig, beta_fast, base));
    dims[1] = ceilf (gk_rope_corr_dim(n_dims, n_ctx_orig, beta_slow, base));
    dims[0] = fmaxf(0.0f, dims[0]);
    dims[1] = fminf((float) (n_dims - 1), dims[1]);
}

// Ramps from 1 at the low-frequency end to 0 at the high-frequency end, over
// the band between the two correction dimensions.
static float gk_rope_ramp(float low, float high, int i) {
    const float y = ((float) (i / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static void gk_rope_angle(float theta_base, float freq_scale, const float corr_dims[2],
                          int i, float ext_factor, float attn_factor,
                          float * cos_out, float * sin_out) {
    float theta = theta_base * freq_scale;
    float mscale = attn_factor;

    if (ext_factor != 0.0f) {
        const float ramp = gk_rope_ramp(corr_dims[0], corr_dims[1], i) * ext_factor;

        // blend the scaled and unscaled angles across the band
        theta = theta * (1.0f - ramp) + theta_base * ramp;

        // YaRN's temperature correction: the extra attention entropy from
        // stretching the context is compensated by a small uniform gain
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    *cos_out = cosf(theta) * mscale;
    *sin_out = sinf(theta) * mscale;
}

// Fills `cache` with (cos, sin) per pair for one position. The single-axis
// form advances one angle geometrically; the multi-axis form runs four angles
// (time, height, width, extra) and picks per pair by which section it falls
// in - or, for the interleaved variant, by its index modulo three.
static void gk_rope_cache_single(float p, float freq_scale, const float * freq_factors,
                                 const float corr_dims[2], int64_t n, float ext_factor,
                                 float mscale, float theta_scale, float * cache) {
    float theta = p;
    for (int64_t i0 = 0; i0 < n; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        gk_rope_angle(theta / ff, freq_scale, corr_dims, (int) i0,
                      ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        theta *= theta_scale;
    }
}

static void gk_rope_cache_multi(float p_t, float p_h, float p_w, float p_e,
                                const int sections[4], bool is_imrope, bool indep_sects,
                                float freq_scale, const float * freq_factors,
                                const float corr_dims[2], int64_t n, float ext_factor,
                                float mscale, float theta_scale, float * cache) {
    float theta_t = p_t, theta_h = p_h, theta_w = p_w, theta_e = p_e;

    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const int sec_w = sections[1] + sections[0];
    const int sec_e = sections[2] + sec_w;
    GK_ASSERT(sect_dims <= n);

    for (int64_t i0 = 0; i0 < n; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;

        const int sector = (int) ((i0 / 2) % sect_dims);
        if (indep_sects) {
            // the vision rope restarts each axis's angle at its section
            if (sector == 0) {
                theta_t = p_t;
            } else if (sector == sections[0]) {
                theta_h = p_h;
            } else if (sector == sec_w) {
                theta_w = p_w;
            } else if (sector == sec_e) {
                theta_e = p_e;
            }
        }

        float theta = theta_t;
        if (is_imrope) {
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
        } else {
            if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            } else if (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        gk_rope_angle(theta / ff, freq_scale, corr_dims, (int) i0,
                      ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

static void gk_compute_rope(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * pos  = dst->src[1];
    const struct gk_tensor * ff   = dst->src[2];

    const int   n_dims     = gk_get_op_params_i32(dst, 1);
    const int   mode       = gk_get_op_params_i32(dst, 2);
    const int   n_ctx_orig = gk_get_op_params_i32(dst, 4);
    const float freq_base  = gk_get_op_params_f32(dst, 5);
    const float freq_scale = gk_get_op_params_f32(dst, 6);
    const float ext_factor = gk_get_op_params_f32(dst, 7);
    const float attn_factor= gk_get_op_params_f32(dst, 8);
    const float beta_fast  = gk_get_op_params_f32(dst, 9);
    const float beta_slow  = gk_get_op_params_f32(dst, 10);

    int sections[4];
    for (int i = 0; i < 4; ++i) {
        sections[i] = gk_get_op_params_i32(dst, 11 + i);
    }

    const bool is_imrope  = mode == GK_ROPE_TYPE_IMROPE;
    const bool mrope_used = (mode & GK_ROPE_TYPE_MROPE) != 0; // vision and imrope included
    const bool is_vision  = mode == GK_ROPE_TYPE_VISION;
    const bool neox       = (mode & GK_ROPE_TYPE_NEOX) != 0;

    GK_ASSERT(pos->type == GK_TYPE_I32);
    GK_ASSERT(n_dims <= src0->ne[0]);
    GK_ASSERT(n_dims % 2 == 0);

    if (mrope_used) {
        GK_ASSERT(pos->ne[0] == src0->ne[2] * 4);
        GK_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    } else {
        GK_ASSERT(pos->ne[0] == src0->ne[2]);
    }
    if (is_vision) {
        GK_ASSERT(n_dims == src0->ne[0] / 2);
    }

    const float theta_scale = powf(freq_base, -2.0f / (float) n_dims);

    float corr_dims[2];
    gk_rope_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const int32_t * positions = (const int32_t *) pos->data;
    const int64_t n_pos = src0->ne[2];

    const float * freq_factors = NULL;
    if (ff != NULL) {
        GK_ASSERT(ff->type == GK_TYPE_F32);
        freq_factors = (const float *) ff->data;
    }

    float * a     = gk_scratch(st, 0);
    float * out   = gk_scratch(st, 1);
    float * cache = gk_scratch(st, 2);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    int64_t cached_i2 = -1;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        // the angles depend only on the position (dim 2), so they are
        // computed once per token rather than once per head
        if (cached_i2 != i2) {
            if (!mrope_used) {
                gk_rope_cache_single((float) positions[i2], freq_scale, freq_factors,
                        corr_dims, src0->ne[0], ext_factor, attn_factor,
                        theta_scale, cache);
            } else {
                gk_rope_cache_multi((float) positions[i2],
                        (float) positions[i2 + n_pos],
                        (float) positions[i2 + n_pos * 2],
                        (float) positions[i2 + n_pos * 3],
                        sections, is_imrope, is_vision,
                        freq_scale, freq_factors, corr_dims, src0->ne[0],
                        ext_factor, attn_factor, theta_scale, cache);
            }
            cached_i2 = i2;
        }

        const float * pa = gk_row_read(src0, i1, i2, i3, a);

        // three pairings share the rotation: normal pairs (2i, 2i+1), neox
        // and the multi-axis ropes pair (i, i + n_dims/2), and the vision
        // rope pairs (i, i + n_dims) across the full width
        const int64_t n_rot    = is_vision ? dst->ne[0] : n_dims;
        const int64_t n_offset = is_vision ? n_dims
                               : (neox || mrope_used) ? n_dims / 2 : 1;
        const int     ic_scale = (neox || mrope_used) ? 2 : 1;

        for (int64_t i = n_dims; i < dst->ne[0] && !is_vision; ++i) {
            out[i] = pa[i]; // channels past n_dims pass through unrotated
        }

        for (int64_t i0 = 0; i0 < n_rot; i0 += 2) {
            const int64_t ic = i0 / ic_scale;

            const float cos_t = cache[i0 + 0];
            const float sin_t = cache[i0 + 1];

            const float x0 = pa[ic];
            const float x1 = pa[ic + n_offset];

            out[ic]            = x0 * cos_t - x1 * sin_t;
            out[ic + n_offset] = x0 * sin_t + x1 * cos_t;
        }

        gk_row_write(dst, i1, i2, i3, out);
    }
}

// --------------------------------------------------------------------------
// fused attention
//
// One pass of online softmax attention per query row: walk the keys once,
// maintain the running maximum M and the running normaliser S, and rescale
// the value accumulator whenever the maximum moves. The accumulator is f32
// regardless of the value type - this pass is the reference for accuracy, and
// a half-precision accumulator would put the error floor above what the
// differential tests can distinguish.
//
// K may be quantized: the query row is converted once to K's dot operand type
// and the same vec_dot the matmul uses runs against every key row.
// --------------------------------------------------------------------------

static void gk_compute_flash_attn_ext(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * q     = dst->src[0];
    const struct gk_tensor * k     = dst->src[1];
    const struct gk_tensor * v     = dst->src[2];
    const struct gk_tensor * mask  = dst->src[3];
    const struct gk_tensor * sinks = dst->src[4];

    const int64_t DK = k->ne[0];
    const int64_t DV = v->ne[0];

    GK_ASSERT(q->ne[0] == DK);
    GK_ASSERT(dst->ne[0] == DV);
    GK_ASSERT(dst->ne[2] == q->ne[1]);

    GK_ASSERT(q->nb[0] == gk_type_size(q->type));
    GK_ASSERT(k->nb[0] == gk_type_size(k->type));
    GK_ASSERT(v->nb[0] == gk_type_size(v->type));
    GK_ASSERT(q->type == GK_TYPE_F32);
    GK_ASSERT(dst->type == GK_TYPE_F32);

    float scale         = gk_get_op_params_f32(dst, 0);
    const float max_bias      = gk_get_op_params_f32(dst, 1);
    const float logit_softcap = gk_get_op_params_f32(dst, 2);

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // broadcast factors for grouped-query attention
    const int64_t rk2 = q->ne[2] / k->ne[2];
    const int64_t rk3 = q->ne[3] / k->ne[3];
    const int64_t rv2 = q->ne[2] / v->ne[2];
    const int64_t rv3 = q->ne[3] / v->ne[3];

    const int64_t n_head = q->ne[2];
    int64_t n_head_log2 = 1;
    while (n_head_log2 * 2 <= n_head) {
        n_head_log2 *= 2;
    }

    const struct gk_type_traits * ktr = gk_get_type_traits(k->type);
    GK_ASSERT(ktr->vec_dot != NULL);

    const enum gk_type vdt = ktr->vec_dot_type;
    const struct gk_type_traits * vdtr = gk_get_type_traits(vdt);
    const struct gk_type_traits * vtr  = gk_get_type_traits(v->type);

    GK_ASSERT(vdt == GK_TYPE_F32 || vdtr->from_float != NULL);
    GK_ASSERT(v->type == GK_TYPE_F32 || vtr->to_float != NULL);

    float * VKQ = gk_scratch(st, 0);                  // f32 value accumulator
    float * V32 = gk_scratch(st, 1);                  // staging for a converted V row
    void  * Qq  = gk_scratch_bytes(st, 2, gk_row_size(vdt, DK)); // converted query row

    // one unit of work per query row; each writes a disjoint slice of dst
    const int64_t nr  = q->ne[1] * q->ne[2] * q->ne[3];
    const int64_t per = (nr + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, nr);
    const int64_t r1  = GK_MIN(r0 + per, nr);

    for (int64_t ir = r0; ir < r1; ++ir) {
        const int64_t iq3 = ir / (q->ne[2] * q->ne[1]);
        const int64_t iq2 = (ir - iq3 * q->ne[2] * q->ne[1]) / q->ne[1];
        const int64_t iq1 = ir - iq3 * q->ne[2] * q->ne[1] - iq2 * q->ne[1];

        const float slope = gk_alibi_slope(max_bias, iq2, n_head_log2);

        float S = 0.0f;
        float M = -INFINITY;

        memset(VKQ, 0, (size_t) DV * sizeof(float));

        const gk_fp16_t * mp = NULL;
        if (mask != NULL) {
            mp = (const gk_fp16_t *) ((const char *) mask->data
                    + iq1 * mask->nb[1]
                    + (iq2 % mask->ne[2]) * mask->nb[2]
                    + (iq3 % mask->ne[3]) * mask->nb[3]);
        }

        const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
        const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

        const float * pq = (const float *) ((const char *) q->data
                + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        if (vdt == GK_TYPE_F32) {
            memcpy(Qq, pq, (size_t) DK * sizeof(float));
        } else {
            vdtr->from_float(pq, Qq, DK);
        }

        for (int64_t ic = 0; ic < k->ne[1]; ++ic) {
            const float mv = mp ? slope * gk_fp16_to_fp32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s = 0.0f;
            const char * k_row = (const char *) k->data
                + ic * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
            ktr->vec_dot((int) DK, &s, 0, k_row, 0, Qq, 0, 1);

            s *= scale;
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            s += mv;

            float vs = 1.0f; // weight of this value row
            if (s > M) {
                const float ms = expf(M - s); // rescale of everything so far
                M = s;
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ[d] *= ms;
                }
                S = S * ms + 1.0f;
            } else {
                vs = expf(s - M);
                S += vs;
            }

            const char * v_row = (const char *) v->data
                + ic * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];

            const float * pv;
            if (v->type == GK_TYPE_F32) {
                pv = (const float *) v_row;
            } else {
                vtr->to_float(v_row, V32, DV);
                pv = V32;
            }

            for (int64_t d = 0; d < DV; ++d) {
                VKQ[d] += pv[d] * vs;
            }
        }

        // the sink is one more virtual position with no value row
        if (sinks != NULL) {
            const float s = ((const float *) sinks->data)[iq2];
            if (s > M) {
                const float ms = expf(M - s);
                M = s;
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ[d] *= ms;
                }
                S = S * ms + 1.0f;
            } else {
                S += expf(s - M);
            }
        }

        const float inv = S == 0.0f ? 0.0f : 1.0f / S;
        for (int64_t d = 0; d < DV; ++d) {
            VKQ[d] *= inv;
        }

        // heads and batch swap on the way out: dst is [DV, n_head, n_batch, ne3]
        memcpy((char *) dst->data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3],
               VKQ, (size_t) DV * sizeof(float));
    }
}

// --------------------------------------------------------------------------
// reductions and ordering
// --------------------------------------------------------------------------

static void gk_compute_sum_rows(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const bool mean = dst->op == GK_OP_MEAN;

    float * a = gk_scratch(st, 0);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);

                double sum = 0.0;
                for (int64_t i = 0; i < src0->ne[0]; ++i) {
                    sum += (double) pa[i];
                }
                if (mean) {
                    sum /= (double) src0->ne[0];
                }

                const float v = (float) sum;
                gk_row_write(dst, i1, i2, i3, &v);
            }
        }
    }
}

static void gk_compute_sum(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    // One scalar out of every element: splitting this would need a reduction
    // tree across threads for a result that is one number. Not worth it.
    if (st->ith != 0) {
        return;
    }

    float * a = gk_scratch(st, 0);

    double sum = 0.0;
    for (int64_t i3 = 0; i3 < src0->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);
                for (int64_t i = 0; i < src0->ne[0]; ++i) {
                    sum += (double) pa[i];
                }
            }
        }
    }

    const float v = (float) sum;
    gk_row_write(dst, 0, 0, 0, &v);
}

static void gk_compute_argmax(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(dst->type == GK_TYPE_I32);

    float * a = gk_scratch(st, 0);

    const int64_t nr  = src0->ne[1];
    const int64_t per = (nr + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, nr);
    const int64_t r1  = GK_MIN(r0 + per, nr);

    for (int64_t i1 = r0; i1 < r1; ++i1) {
        const float * pa = gk_row_read(src0, i1, 0, 0, a);

        int32_t best = 0;
        float   bestv = pa[0];
        for (int64_t i = 1; i < src0->ne[0]; ++i) {
            if (pa[i] > bestv) {
                bestv = pa[i];
                best  = (int32_t) i;
            }
        }

        ((int32_t *) dst->data)[i1] = best;
    }
}

// Insertion sort: rows here are vocabulary-sized at worst and this runs once
// per sampling step, not per layer. A comparison sort with better asymptotics
// is easy to swap in behind the same result, which is required to be a stable
// ordering so that equal values keep their index order.
static void gk_compute_argsort(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(dst->type == GK_TYPE_I32);

    const enum gk_sort_order order =
        (enum gk_sort_order) gk_get_op_params_i32(dst, 0);

    const int64_t n = src0->ne[0];

    float * a = gk_scratch(st, 0);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);
        {
            {
                const float * pa = gk_row_read(src0, i1, i2, i3, a);

                int32_t * idx = (int32_t *) gk_row_ptr_mut(dst, i1, i2, i3);
                int32_t * tmp = (int32_t *) gk_scratch_bytes(st, 1, (size_t) n * sizeof(int32_t));

                for (int64_t i = 0; i < n; ++i) {
                    idx[i] = (int32_t) i;
                }

                // A bottom-up merge sort, ping-ponging between the output row
                // and one scratch slot.
                //
                // This was an insertion sort, which is fine for the router
                // rows this op sees most of the time - a hundred experts - and
                // quadratic everywhere else. A sampler argsorts the whole
                // vocabulary, a quarter of a million wide, and that is around
                // 10^10 comparisons: tens of seconds, per token, on one
                // thread, because a single row cannot be split across them.
                int32_t * from = idx;
                int32_t * to   = tmp;

                for (int64_t width = 1; width < n; width *= 2) {
                    for (int64_t lo = 0; lo < n; lo += 2 * width) {
                        const int64_t mid = GK_MIN(lo + width, n);
                        const int64_t hi  = GK_MIN(lo + 2 * width, n);

                        int64_t p = lo, q = mid, o = lo;

                        while (p < mid && q < hi) {
                            // Ties go to the lower index, which is what makes
                            // the order total: two equal values are ordered by
                            // where they came from, so the answer does not
                            // depend on how the sort happened to move them.
                            const float vp = pa[from[p]];
                            const float vq = pa[from[q]];

                            bool take_p;
                            if (vp != vq) {
                                take_p = order == GK_SORT_ORDER_ASC ? vp < vq : vp > vq;
                            } else {
                                take_p = from[p] < from[q];
                            }

                            to[o++] = take_p ? from[p++] : from[q++];
                        }
                        while (p < mid) { to[o++] = from[p++]; }
                        while (q < hi)  { to[o++] = from[q++]; }
                    }

                    int32_t * swap = from; from = to; to = swap;
                }

                if (from != idx) {
                    memcpy(idx, from, (size_t) n * sizeof(int32_t));
                }
            }
        }
    }
}

// The partial-selection top_k: indices of the k largest per row, in no
// promised order. The selection itself is a bounded insertion into the first
// k slots, so a vocabulary-sized row pays O(n*k) rather than a full sort.
static void gk_compute_top_k(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(dst->type == GK_TYPE_I32);
    GK_ASSERT(src0->type == GK_TYPE_F32);
    GK_ASSERT(gk_row_is_packed(src0));

    const int64_t n = src0->ne[0];
    const int64_t k = dst->ne[0];

    int32_t * idx = (int32_t *) gk_scratch_bytes(st, 0, (size_t) n * sizeof(int32_t));

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const float * pa = (const float *) gk_row_ptr(src0, i1, i2, i3);

        for (int64_t i = 0; i < n; ++i) {
            idx[i] = (int32_t) i;
        }

        // partial selection sort of the first k slots: descending by value,
        // index order breaking ties - the same order a stable full sort gives
        for (int64_t i = 0; i < k; ++i) {
            int64_t best = i;
            for (int64_t j = i + 1; j < n; ++j) {
                if (pa[idx[j]] > pa[idx[best]]) {
                    best = j;
                }
            }
            const int32_t t = idx[i];
            idx[i] = idx[best];
            idx[best] = t;
        }

        int32_t * out = (int32_t *) gk_row_ptr_mut(dst, i1, i2, i3);
        memcpy(out, idx, (size_t) k * sizeof(int32_t));

        // deliberately break the sorted order, so nothing learns to rely on
        // an ordering the op does not promise
        if (k > 1) {
            const int32_t t = out[0];
            out[0] = out[1];
            out[1] = t;
        }
    }
}

static void gk_compute_cumsum(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(gk_are_same_shape(src0, dst));

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const float * pa = gk_row_read(src0, i1, i2, i3, a);

        float run = 0.0f;
        for (int64_t i = 0; i < dst->ne[0]; ++i) {
            run += pa[i];
            out[i] = run;
        }
        gk_row_write(dst, i1, i2, i3, out);
    }
}

static void gk_compute_count_equal(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    GK_ASSERT(dst->type == GK_TYPE_I64);
    GK_ASSERT(src0->type == GK_TYPE_I32 && src1->type == GK_TYPE_I32);

    // one scalar out; not worth a cross-thread reduction
    if (st->ith != 0) {
        return;
    }

    int64_t count = 0;
    for (int64_t i3 = 0; i3 < src0->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                const int32_t * pa = (const int32_t *) gk_row_ptr(src0, i1, i2, i3);
                const int32_t * pb = (const int32_t *) gk_row_ptr(src1, i1, i2, i3);
                for (int64_t i = 0; i < src0->ne[0]; ++i) {
                    count += pa[i] == pb[i];
                }
            }
        }
    }

    *(int64_t *) dst->data = count;
}

// dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]]
static void gk_compute_add_id(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];
    const struct gk_tensor * ids  = dst->src[2];

    GK_ASSERT(ids->type == GK_TYPE_I32);

    float * a   = gk_scratch(st, 0);
    float * b   = gk_scratch(st, 1);
    float * out = gk_scratch(st, 2);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const int32_t row = *(const int32_t *)
            ((const char *) ids->data + i1 * ids->nb[0] + i2 * ids->nb[1]);
        GK_ASSERT(row >= 0 && row < src1->ne[1]);

        const float * pa = gk_row_read(src0, i1, i2, i3, a);
        const float * pb = gk_row_read(src1, row, 0, 0, b);

        for (int64_t i = 0; i < dst->ne[0]; ++i) {
            out[i] = pa[i] + pb[i];
        }
        gk_row_write(dst, i1, i2, i3, out);
    }
}

// acc and set: the result is a copy of src0 with the window described by the
// op params overwritten (set) or accumulated into (acc) from src1. The window
// walk carries running state, so one thread does it and the rest wait at the
// node barrier - these ops sit outside every inference hot path.
static void gk_compute_acc_set(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    if (st->ith != 0) {
        return;
    }

    const size_t nb1     = (size_t) gk_get_op_params_i32(dst, 0);
    const size_t nb2     = (size_t) gk_get_op_params_i32(dst, 1);
    const size_t nb3     = (size_t) gk_get_op_params_i32(dst, 2);
    const size_t offset  = (size_t) gk_get_op_params_i32(dst, 3);
    const bool   inplace = gk_get_op_params_i32(dst, 4) != 0;

    GK_ASSERT(dst->type == GK_TYPE_F32 && src0->type == GK_TYPE_F32 && src1->type == GK_TYPE_F32);

    if (!inplace) {
        // dst and src0 are both contiguous f32 by the builder's contract
        if (dst->data != src0->data) {
            memcpy(dst->data, src0->data, gk_nbytes(dst));
        }
    }

    const bool add = dst->op == GK_OP_ACC;

    for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src1->ne[1]; ++i1) {
                const float * pb = (const float *) gk_row_ptr(src1, i1, i2, i3);
                float * pd = (float *) ((char *) dst->data
                        + offset + i1 * nb1 + i2 * nb2 + i3 * nb3);

                for (int64_t i = 0; i < src1->ne[0]; ++i) {
                    pd[i] = add ? pd[i] + pb[i] : pb[i];
                }
            }
        }
    }
}

// Writes the rows of src[0] into the view target src[2] at the row indices
// src[1] names, converting into the destination's storage type - this is the
// op a quantized KV cache is filled through.
static void gk_compute_set_rows(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * b   = dst->src[0];
    const struct gk_tensor * c   = dst->src[1];

    GK_ASSERT(b->type == GK_TYPE_F32 || b->type == GK_TYPE_F16);
    GK_ASSERT(c->type == GK_TYPE_I64 || c->type == GK_TYPE_I32);

    const struct gk_type_traits * dtr = gk_get_type_traits(dst->type);
    GK_ASSERT(dtr->from_float != NULL || dst->type == GK_TYPE_F32);

    const int64_t nc = b->ne[0];
    const int64_t nr = b->ne[1];

    float * buf = gk_scratch(st, 0);

    const int64_t per = (nr + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, nr);
    const int64_t r1  = GK_MIN(r0 + per, nr);

    for (int64_t i3 = 0; i3 < b->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < b->ne[2]; ++i2) {
            for (int64_t i = r0; i < r1; ++i) {
                const char * pi = (const char *) c->data
                    + i * c->nb[0] + (i2 % c->ne[1]) * c->nb[1] + (i3 % c->ne[2]) * c->nb[2];

                const int64_t row = c->type == GK_TYPE_I64
                    ? *(const int64_t *) pi
                    : (int64_t) *(const int32_t *) pi;

                GK_ASSERT(row >= 0 && row < dst->ne[1]);

                const float * pb = gk_row_read(b, i, i2, i3, buf);
                char * pd = (char *) dst->data
                    + row * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];

                if (dst->type == GK_TYPE_F32) {
                    memcpy(pd, pb, (size_t) nc * sizeof(float));
                } else {
                    dtr->from_float(pb, pd, nc);
                }
            }
        }
    }
}

static void gk_compute_diag(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const float * pa = gk_row_read(src0, 0, i2, i3, a);

        for (int64_t i = 0; i < dst->ne[0]; ++i) {
            out[i] = i == i1 ? pa[i] : 0.0f;
        }
        gk_row_write(dst, i1, i2, i3, out);
    }
}

static void gk_compute_tri(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    const enum gk_tri_type type = (enum gk_tri_type) gk_get_op_params_i32(dst, 0);

    float * a   = gk_scratch(st, 0);
    float * out = gk_scratch(st, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const float * pa = gk_row_read(src0, i1, i2, i3, a);

        for (int64_t i = 0; i < dst->ne[0]; ++i) {
            bool keep;
            switch (type) {
                case GK_TRI_TYPE_LOWER:      keep = i <  i1; break;
                case GK_TRI_TYPE_LOWER_DIAG: keep = i <= i1; break;
                case GK_TRI_TYPE_UPPER:      keep = i >  i1; break;
                default:                     keep = i >= i1; break; // UPPER_DIAG
            }
            out[i] = keep ? pa[i] : 0.0f;
        }
        gk_row_write(dst, i1, i2, i3, out);
    }
}

static void gk_compute_fill(struct gk_compute_state * st, struct gk_tensor * dst) {
    const float c = gk_get_op_params_f32(dst, 0);

    float * out = gk_scratch(st, 0);
    for (int64_t i = 0; i < dst->ne[0]; ++i) {
        out[i] = c;
    }

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        gk_row_write(dst, i1, i2, i3, out);
    }
}

static void gk_compute_arange(struct gk_compute_state * st, struct gk_tensor * dst) {
    GK_ASSERT(dst->type == GK_TYPE_F32);

    const float start = gk_get_op_params_f32(dst, 0);
    const float step  = gk_get_op_params_f32(dst, 2);

    const int64_t n = gk_nelements(dst);
    float * out = (float *) dst->data;

    for (int64_t i = st->ith; i < n; i += st->nth) {
        out[i] = start + step * (float) i;
    }
}

static inline int64_t gk_wrap_index(int64_t i, int64_t n) {
    return ((i % n) + n) % n;
}

static void gk_compute_roll(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32);
    GK_ASSERT(gk_row_is_packed(src0) && gk_row_is_packed(dst));

    const int s0 = gk_get_op_params_i32(dst, 0);
    const int s1 = gk_get_op_params_i32(dst, 1);
    const int s2 = gk_get_op_params_i32(dst, 2);
    const int s3 = gk_get_op_params_i32(dst, 3);

    const int64_t n = dst->ne[0];

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const int64_t j1 = gk_wrap_index(i1 - s1, src0->ne[1]);
        const int64_t j2 = gk_wrap_index(i2 - s2, src0->ne[2]);
        const int64_t j3 = gk_wrap_index(i3 - s3, src0->ne[3]);

        const float * src_row = (const float *) gk_row_ptr(src0, j1, j2, j3);
        float * dst_row = (float *) gk_row_ptr_mut(dst, i1, i2, i3);

        // within the row the shift is a two-piece copy
        const int64_t s = gk_wrap_index(-s0, n);
        memcpy(dst_row, src_row + s, (size_t) (n - s) * sizeof(float));
        memcpy(dst_row + (n - s), src_row, (size_t) s * sizeof(float));
    }
}

static void gk_compute_timestep_embedding(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);

    const int dim        = gk_get_op_params_i32(dst, 0);
    const int max_period = gk_get_op_params_i32(dst, 1);

    const int half = dim / 2;

    float * out = gk_scratch(st, 0);

    // one destination row per timestep; rows are independent
    const int64_t nr  = dst->ne[1];
    const int64_t per = (nr + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, nr);
    const int64_t r1  = GK_MIN(r0 + per, nr);

    for (int64_t i = r0; i < r1; ++i) {
        const float t = ((const float *) src0->data)[i];

        for (int j = 0; j < half; ++j) {
            const float freq = expf(-logf((float) max_period) * (float) j / (float) half);
            const float arg  = t * freq;
            out[j]        = cosf(arg);
            out[j + half] = sinf(arg);
        }
        if (dim % 2 != 0) {
            out[2 * half] = 0.0f;
        }

        gk_row_write(dst, i, 0, 0, out);
    }
}

static void gk_compute_pad(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);
    GK_ASSERT(gk_is_contiguous(dst));

    const int lp0 = gk_get_op_params_i32(dst, 0);
    const int lp1 = gk_get_op_params_i32(dst, 2);
    const int rp1 = gk_get_op_params_i32(dst, 3);
    const int lp2 = gk_get_op_params_i32(dst, 4);
    const int rp2 = gk_get_op_params_i32(dst, 5);
    const int lp3 = gk_get_op_params_i32(dst, 6);
    const int rp3 = gk_get_op_params_i32(dst, 7);
    const bool circular = gk_get_op_params_i32(dst, 8) != 0;

    const int rp0 = gk_get_op_params_i32(dst, 1);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        float * out = (float *) gk_row_ptr_mut(dst, i1, i2, i3);

        if (circular) {
            const int64_t j1 = gk_wrap_index(i1 - lp1, src0->ne[1]);
            const int64_t j2 = gk_wrap_index(i2 - lp2, src0->ne[2]);
            const int64_t j3 = gk_wrap_index(i3 - lp3, src0->ne[3]);

            for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                const int64_t j0 = gk_wrap_index(i0 - lp0, src0->ne[0]);
                out[i0] = *(const float *) ((const char *) src0->data
                        + j0 * src0->nb[0] + j1 * src0->nb[1]
                        + j2 * src0->nb[2] + j3 * src0->nb[3]);
            }
            continue;
        }

        const bool inside_123 =
            i1 >= lp1 && i1 < dst->ne[1] - rp1 &&
            i2 >= lp2 && i2 < dst->ne[2] - rp2 &&
            i3 >= lp3 && i3 < dst->ne[3] - rp3;

        if (!inside_123) {
            memset(out, 0, (size_t) dst->ne[0] * sizeof(float));
            continue;
        }

        for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
            if (i0 >= lp0 && i0 < dst->ne[0] - rp0) {
                out[i0] = *(const float *) ((const char *) src0->data
                        + (i0 - lp0) * src0->nb[0] + (i1 - lp1) * src0->nb[1]
                        + (i2 - lp2) * src0->nb[2] + (i3 - lp3) * src0->nb[3]);
            } else {
                out[i0] = 0.0f;
            }
        }
    }
}

static void gk_compute_pad_reflect_1d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);

    const int p0 = gk_get_op_params_i32(dst, 0);
    const int p1 = gk_get_op_params_i32(dst, 1);

    const int64_t n = src0->ne[0];

    float * a = gk_scratch(st, 0);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const float * pa = gk_row_read(src0, i1, i2, i3, a);
        float * out = (float *) gk_row_ptr_mut(dst, i1, i2, i3);

        for (int64_t i = 0; i < n; ++i) {
            out[p0 + i] = pa[i];
        }
        for (int i = 1; i <= p0; ++i) {
            out[p0 - i] = pa[i];
        }
        for (int i = 1; i <= p1; ++i) {
            out[p0 + n - 1 + i] = pa[n - 1 - i];
        }
    }
}

// --------------------------------------------------------------------------
// resampling
//
// All four filters follow the same frame: for each output pixel, work out the
// source position(s) it draws from and blend. The bit-for-bit shapes of the
// index arithmetic - truncation for nearest, the half-pixel offset, clamping
// at the borders - are part of the op's meaning, not implementation detail:
// a model's projector was trained against exactly these.
// --------------------------------------------------------------------------

static void gk_compute_upscale(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);

    const int32_t mode_flags = gk_get_op_params_i32(dst, 0);
    const enum gk_scale_mode mode = (enum gk_scale_mode) (mode_flags & 0xFF);

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1];

    float sf0 = (float) dst->ne[0] / src0->ne[0];
    float sf1 = (float) dst->ne[1] / src0->ne[1];
    const float sf2 = (float) dst->ne[2] / src0->ne[2];
    const float sf3 = (float) dst->ne[3] / src0->ne[3];
    float pixel_offset = 0.5f;

    if (mode_flags & GK_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = dst->ne[0] > 1 && ne00 > 1 ? (float) (dst->ne[0] - 1) / (ne00 - 1) : sf0;
        sf1 = dst->ne[1] > 1 && ne01 > 1 ? (float) (dst->ne[1] - 1) / (ne01 - 1) : sf1;
    }

    #define GK_SRC_PX(x, y, c, b) (*(const float *) ((const char *) src0->data \
            + (x) * src0->nb[0] + (y) * src0->nb[1] + (c) * src0->nb[2] + (b) * src0->nb[3]))

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const int64_t i02 = (int64_t) (i2 / sf2);
        const int64_t i03 = (int64_t) (i3 / sf3);

        float * out = (float *) gk_row_ptr_mut(dst, i1, i2, i3);

        if (mode == GK_SCALE_MODE_NEAREST) {
            const int64_t i01 = (int64_t) (i1 / sf1);
            for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                const int64_t i00 = (int64_t) (i0 / sf0);
                out[i0] = GK_SRC_PX(i00, i01, i02, i03);
            }
        } else if (mode == GK_SCALE_MODE_BILINEAR && (mode_flags & GK_SCALE_FLAG_ANTIALIAS)) {
            // triangle filter widened to the downscale factor, the shape
            // F.interpolate(..., antialias=True) uses
            const float support1  = fmaxf(1.0f, 1.0f / sf1);
            const float invscale1 = 1.0f / support1;
            const float support0  = fmaxf(1.0f, 1.0f / sf0);
            const float invscale0 = 1.0f / support0;

            const float y = ((float) i1 + pixel_offset) / sf1;
            const int64_t y_min = GK_MAX((int64_t) (y - support1 + pixel_offset), 0);
            const int64_t y_max = GK_MIN((int64_t) (y + support1 + pixel_offset), ne01);

            for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                const float x = ((float) i0 + pixel_offset) / sf0;
                const int64_t x_min = GK_MAX((int64_t) (x - support0 + pixel_offset), 0);
                const int64_t x_max = GK_MIN((int64_t) (x + support0 + pixel_offset), ne00);

                float val = 0.0f;
                float total = 0.0f;

                for (int64_t sy = y_min; sy < y_max; ++sy) {
                    const float wy = fmaxf(1.0f - fabsf(((float) sy - y + pixel_offset) * invscale1), 0.0f);
                    for (int64_t sx = x_min; sx < x_max; ++sx) {
                        const float wx = fmaxf(1.0f - fabsf(((float) sx - x + pixel_offset) * invscale0), 0.0f);
                        const float w = wx * wy;
                        if (w <= 0.0f) {
                            continue;
                        }
                        val   += GK_SRC_PX(sx, sy, i02, i03) * w;
                        total += w;
                    }
                }
                out[i0] = total > 0.0f ? val / total : val;
            }
        } else if (mode == GK_SCALE_MODE_BILINEAR) {
            const float y = ((float) i1 + pixel_offset) / sf1 - pixel_offset;
            int64_t y0 = (int64_t) floorf(y);
            int64_t y1 = y0 + 1;
            float dy = y - (float) y0;

            y0 = GK_MAX((int64_t) 0, GK_MIN(y0, ne01 - 1));
            y1 = GK_MAX((int64_t) 0, GK_MIN(y1, ne01 - 1));
            dy = fmaxf(0.0f, fminf(dy, 1.0f));

            for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                const float x = ((float) i0 + pixel_offset) / sf0 - pixel_offset;
                int64_t x0 = (int64_t) floorf(x);
                int64_t x1 = x0 + 1;
                float dx = x - (float) x0;

                x0 = GK_MAX((int64_t) 0, GK_MIN(x0, ne00 - 1));
                x1 = GK_MAX((int64_t) 0, GK_MIN(x1, ne00 - 1));
                dx = fmaxf(0.0f, fminf(dx, 1.0f));

                const float p00 = GK_SRC_PX(x0, y0, i02, i03);
                const float p10 = GK_SRC_PX(x1, y0, i02, i03);
                const float p01 = GK_SRC_PX(x0, y1, i02, i03);
                const float p11 = GK_SRC_PX(x1, y1, i02, i03);

                out[i0] = p00 * (1 - dx) * (1 - dy) + p10 * dx * (1 - dy)
                        + p01 * (1 - dx) * dy       + p11 * dx * dy;
            }
        } else if (mode == GK_SCALE_MODE_BICUBIC) {
            // the alpha = -0.75 convolution kernel, evaluated as two nested
            // 1-D passes; border pixels clamp
            const float A = -0.75f;
            #define GK_CUBIC_W1(x) ((((A) + 2) * (x) - ((A) + 3)) * (x) * (x) + 1)
            #define GK_CUBIC_W2(x) (((((A) * (x)) - 5 * (A)) * (x) + 8 * (A)) * (x) - 4 * (A))
            #define GK_CUBIC(p0, p1, p2, p3, x) \
                ((p0) * GK_CUBIC_W2((x) + 1) + (p1) * GK_CUBIC_W1(x) + \
                 (p2) * GK_CUBIC_W1(1 - (x)) + (p3) * GK_CUBIC_W2(2 - (x)))

            const float y = ((float) i1 + pixel_offset) / sf1 - pixel_offset;
            const int64_t y0 = (int64_t) floorf(y);
            const float dy = y - (float) y0;

            for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                const float x = ((float) i0 + pixel_offset) / sf0 - pixel_offset;
                const int64_t x0 = (int64_t) floorf(x);
                const float dx = x - (float) x0;

                float rows[4];
                for (int dr = -1; dr <= 2; ++dr) {
                    const int64_t sy = GK_MAX((int64_t) 0, GK_MIN(y0 + dr, ne01 - 1));
                    float px[4];
                    for (int dc = -1; dc <= 2; ++dc) {
                        const int64_t sx = GK_MAX((int64_t) 0, GK_MIN(x0 + dc, ne00 - 1));
                        px[dc + 1] = GK_SRC_PX(sx, sy, i02, i03);
                    }
                    rows[dr + 1] = GK_CUBIC(px[0], px[1], px[2], px[3], dx);
                }
                out[i0] = GK_CUBIC(rows[0], rows[1], rows[2], rows[3], dy);
            }
            #undef GK_CUBIC
            #undef GK_CUBIC_W1
            #undef GK_CUBIC_W2
        } else {
            GK_ABORT("upscale mode %d is not implemented", (int) mode);
        }
    }

    #undef GK_SRC_PX
}

// --------------------------------------------------------------------------
// convolution
//
// The composite conv builders lower to im2col + matmul, so the kernels here
// are the unrolling itself plus the direct forms. One numeric convention runs
// through all of them: when the kernel tensor is f16, the input samples are
// rounded to f16 before the multiply - that is what the im2col route does by
// construction, and the direct kernels reproduce it so the two routes agree.
// --------------------------------------------------------------------------

static inline float gk_conv_sample(const struct gk_tensor * t,
                                   int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * p = (const char *) t->data
        + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    if (t->type == GK_TYPE_F16) {
        return gk_fp16_to_fp32(*(const gk_fp16_t *) p);
    }
    return *(const float *) p;
}

static void gk_compute_im2col(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0]; // kernel, shape only
    const struct gk_tensor * src1 = dst->src[1]; // image

    GK_ASSERT(src1->type == GK_TYPE_F32);
    GK_ASSERT(dst->type == GK_TYPE_F32 || dst->type == GK_TYPE_F16);
    GK_ASSERT(gk_is_contiguous(dst));

    const int s0 = gk_get_op_params_i32(dst, 0);
    const int s1 = gk_get_op_params_i32(dst, 1);
    const int p0 = gk_get_op_params_i32(dst, 2);
    const int p1 = gk_get_op_params_i32(dst, 3);
    const int d0 = gk_get_op_params_i32(dst, 4);
    const int d1 = gk_get_op_params_i32(dst, 5);
    const bool is_2D = gk_get_op_params_i32(dst, 6) != 0;

    const int64_t N  = is_2D ? src1->ne[3] : src1->ne[2];
    const int64_t IC = is_2D ? src1->ne[2] : src1->ne[1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW = src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW = src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW = dst->ne[1];

    const size_t ofs_n = is_2D ? src1->nb[3] : src1->nb[2];
    const size_t ofs_c = is_2D ? src1->nb[2] : src1->nb[1];
    const size_t ofs_h = is_2D ? src1->nb[1] : 0;

    // split by input channel: each (channel, position) cell is written by
    // exactly one thread, so the result does not depend on the count
    for (int64_t in = 0; in < N; ++in) {
        for (int64_t ioh = 0; ioh < OH; ++ioh) {
            for (int64_t iow = 0; iow < OW; ++iow) {
                char * cell = (char *) dst->data
                    + ((in * OH * OW + ioh * OW + iow) * (IC * KH * KW)) * gk_type_size(dst->type);

                for (int64_t iic = st->ith; iic < IC; iic += st->nth) {
                    const char * plane = (const char *) src1->data + in * ofs_n + iic * ofs_c;

                    for (int64_t ikh = 0; ikh < KH; ++ikh) {
                        for (int64_t ikw = 0; ikw < KW; ++ikw) {
                            const int64_t iiw = iow * s0 + ikw * d0 - p0;
                            const int64_t iih = ioh * s1 + ikh * d1 - p1;

                            float v = 0.0f;
                            if (iih >= 0 && iih < IH && iiw >= 0 && iiw < IW) {
                                v = *(const float *) (plane + iih * ofs_h + iiw * src1->nb[0]);
                            }

                            const int64_t at = iic * (KH * KW) + ikh * KW + ikw;
                            if (dst->type == GK_TYPE_F16) {
                                ((gk_fp16_t *) cell)[at] = gk_fp32_to_fp16(v);
                            } else {
                                ((float *) cell)[at] = v;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void gk_compute_im2col_3d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    GK_ASSERT(src1->type == GK_TYPE_F32);
    GK_ASSERT(dst->type == GK_TYPE_F32 || dst->type == GK_TYPE_F16);
    GK_ASSERT(gk_is_contiguous(dst));

    const int s0 = gk_get_op_params_i32(dst, 0);
    const int s1 = gk_get_op_params_i32(dst, 1);
    const int s2 = gk_get_op_params_i32(dst, 2);
    const int p0 = gk_get_op_params_i32(dst, 3);
    const int p1 = gk_get_op_params_i32(dst, 4);
    const int p2 = gk_get_op_params_i32(dst, 5);
    const int d0 = gk_get_op_params_i32(dst, 6);
    const int d1 = gk_get_op_params_i32(dst, 7);
    const int d2 = gk_get_op_params_i32(dst, 8);
    const int64_t IC = gk_get_op_params_i32(dst, 9);

    const int64_t N  = src1->ne[3] / IC;
    const int64_t ID = src1->ne[2], IH = src1->ne[1], IW = src1->ne[0];
    const int64_t KD = src0->ne[2], KH = src0->ne[1], KW = src0->ne[0];
    const int64_t OD = dst->ne[3] / N, OH = dst->ne[2], OW = dst->ne[1];

    const int64_t KD_KH_KW = KD * KH * KW;
    const int64_t KH_KW    = KH * KW;
    const int64_t cell_n   = IC * KD_KH_KW;

    for (int64_t in = 0; in < N; ++in) {
        for (int64_t iod = 0; iod < OD; ++iod) {
            for (int64_t ioh = 0; ioh < OH; ++ioh) {
                for (int64_t iow = 0; iow < OW; ++iow) {
                    char * cell = (char *) dst->data
                        + ((in * OD + iod) * OH * OW + ioh * OW + iow) * cell_n * gk_type_size(dst->type);

                    for (int64_t iic = st->ith; iic < IC; iic += st->nth) {
                        const char * vol = (const char *) src1->data + (in * IC + iic) * src1->nb[3];

                        for (int64_t ikd = 0; ikd < KD; ++ikd) {
                            for (int64_t ikh = 0; ikh < KH; ++ikh) {
                                for (int64_t ikw = 0; ikw < KW; ++ikw) {
                                    const int64_t iid = iod * s2 + ikd * d2 - p2;
                                    const int64_t iih = ioh * s1 + ikh * d1 - p1;
                                    const int64_t iiw = iow * s0 + ikw * d0 - p0;

                                    float v = 0.0f;
                                    if (iid >= 0 && iid < ID && iih >= 0 && iih < IH &&
                                        iiw >= 0 && iiw < IW) {
                                        v = *(const float *) (vol + iid * src1->nb[2]
                                                + iih * src1->nb[1] + iiw * src1->nb[0]);
                                    }

                                    const int64_t at = iic * KD_KH_KW + ikd * KH_KW + ikh * KW + ikw;
                                    if (dst->type == GK_TYPE_F16) {
                                        ((gk_fp16_t *) cell)[at] = gk_fp32_to_fp16(v);
                                    } else {
                                        ((float *) cell)[at] = v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// direct 2-D convolution: kernel [KW, KH, IC, OC], input [W, H, IC, N],
// result [OW, OH, OC, N]
static void gk_compute_conv_2d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * a = dst->src[0];
    const struct gk_tensor * b = dst->src[1];

    GK_ASSERT(b->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);
    GK_ASSERT(a->type == GK_TYPE_F32 || a->type == GK_TYPE_F16);

    const int s0 = gk_get_op_params_i32(dst, 0);
    const int s1 = gk_get_op_params_i32(dst, 1);
    const int p0 = gk_get_op_params_i32(dst, 2);
    const int p1 = gk_get_op_params_i32(dst, 3);
    const int d0 = gk_get_op_params_i32(dst, 4);
    const int d1 = gk_get_op_params_i32(dst, 5);

    const bool round_src = a->type == GK_TYPE_F16;

    const int64_t IC = b->ne[2], IW = b->ne[0], IH = b->ne[1];
    const int64_t KW = a->ne[0], KH = a->ne[1];

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t oy = ir % dst->ne[1];
        const int64_t oc = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t in = ir / (dst->ne[1] * dst->ne[2]);

        float * out = (float *) gk_row_ptr_mut(dst, oy, oc, in);

        for (int64_t ox = 0; ox < dst->ne[0]; ++ox) {
            float acc = 0.0f;

            for (int64_t ic = 0; ic < IC; ++ic) {
                for (int64_t ky = 0; ky < KH; ++ky) {
                    const int64_t sy = oy * s1 + ky * d1 - p1;
                    if (sy < 0 || sy >= IH) {
                        continue;
                    }
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        const int64_t sx = ox * s0 + kx * d0 - p0;
                        if (sx < 0 || sx >= IW) {
                            continue;
                        }

                        float sv = *(const float *) ((const char *) b->data
                                + sx * b->nb[0] + sy * b->nb[1] + ic * b->nb[2] + in * b->nb[3]);
                        if (round_src) {
                            sv = gk_fp16_to_fp32(gk_fp32_to_fp16(sv));
                        }

                        acc += gk_conv_sample(a, kx, ky, ic, oc) * sv;
                    }
                }
            }

            out[ox] = acc;
        }
    }
}

// direct depthwise 2-D convolution: kernel [KW, KH, 1, C]; handles both the
// WHCN layout and the channels-fastest (CWHN) one the builder detects
static void gk_compute_conv_2d_dw(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * a = dst->src[0];
    const struct gk_tensor * b = dst->src[1];

    GK_ASSERT(b->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);

    const int s0 = gk_get_op_params_i32(dst, 0);
    const int s1 = gk_get_op_params_i32(dst, 1);
    const int p0 = gk_get_op_params_i32(dst, 2);
    const int p1 = gk_get_op_params_i32(dst, 3);
    const int d0 = gk_get_op_params_i32(dst, 4);
    const int d1 = gk_get_op_params_i32(dst, 5);

    const int64_t C  = b->ne[2], N = b->ne[3];
    const int64_t IW = b->ne[0], IH = b->ne[1];
    const int64_t KW = a->ne[0], KH = a->ne[1];
    const int64_t OW = dst->ne[0], OH = dst->ne[1];

    // one (channel, image) plane per unit of work
    const int64_t n_units = C * N;
    const int64_t per = (n_units + st->nth - 1) / st->nth;
    const int64_t u0  = GK_MIN(per * st->ith, n_units);
    const int64_t u1  = GK_MIN(u0 + per, n_units);

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t ic = u % C;
        const int64_t in = u / C;

        for (int64_t oy = 0; oy < OH; ++oy) {
            for (int64_t ox = 0; ox < OW; ++ox) {
                float acc = 0.0f;

                for (int64_t ky = 0; ky < KH; ++ky) {
                    const int64_t sy = oy * s1 + ky * d1 - p1;
                    if (sy < 0 || sy >= IH) {
                        continue;
                    }
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        const int64_t sx = ox * s0 + kx * d0 - p0;
                        if (sx < 0 || sx >= IW) {
                            continue;
                        }

                        const float sv = *(const float *) ((const char *) b->data
                                + sx * b->nb[0] + sy * b->nb[1] + ic * b->nb[2] + in * b->nb[3]);

                        acc += gk_conv_sample(a, kx, ky, 0, ic) * sv;
                    }
                }

                *(float *) ((char *) dst->data + ox * dst->nb[0] + oy * dst->nb[1]
                        + ic * dst->nb[2] + in * dst->nb[3]) = acc;
            }
        }
    }
}

// direct 3-D convolution: kernel [KW, KH, KD, IC*OC] with IC fastest, input
// [W, H, D, IC*N] with IC fastest, result [OW, OH, OD, OC*N] with OC fastest
static void gk_compute_conv_3d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * a = dst->src[0];
    const struct gk_tensor * b = dst->src[1];

    GK_ASSERT(b->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);

    const int s0 = gk_get_op_params_i32(dst, 0);
    const int s1 = gk_get_op_params_i32(dst, 1);
    const int s2 = gk_get_op_params_i32(dst, 2);
    const int p0 = gk_get_op_params_i32(dst, 3);
    const int p1 = gk_get_op_params_i32(dst, 4);
    const int p2 = gk_get_op_params_i32(dst, 5);
    const int d0 = gk_get_op_params_i32(dst, 6);
    const int d1 = gk_get_op_params_i32(dst, 7);
    const int d2 = gk_get_op_params_i32(dst, 8);
    const int64_t C  = gk_get_op_params_i32(dst, 9);
    const int64_t OC = gk_get_op_params_i32(dst, 11);

    const bool round_src = a->type == GK_TYPE_F16;

    const int64_t IW = b->ne[0], IH = b->ne[1], ID = b->ne[2];
    const int64_t KW = a->ne[0], KH = a->ne[1], KD = a->ne[2];

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t oy   = ir % dst->ne[1];
        const int64_t oz   = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t ocn  = ir / (dst->ne[1] * dst->ne[2]);
        const int64_t ioc  = ocn % OC;
        const int64_t in   = ocn / OC;

        float * out = (float *) gk_row_ptr_mut(dst, oy, oz, ocn);

        for (int64_t ox = 0; ox < dst->ne[0]; ++ox) {
            float acc = 0.0f;

            for (int64_t ic = 0; ic < C; ++ic) {
                for (int64_t kz = 0; kz < KD; ++kz) {
                    const int64_t sz = oz * s2 + kz * d2 - p2;
                    if (sz < 0 || sz >= ID) {
                        continue;
                    }
                    for (int64_t ky = 0; ky < KH; ++ky) {
                        const int64_t sy = oy * s1 + ky * d1 - p1;
                        if (sy < 0 || sy >= IH) {
                            continue;
                        }
                        for (int64_t kx = 0; kx < KW; ++kx) {
                            const int64_t sx = ox * s0 + kx * d0 - p0;
                            if (sx < 0 || sx >= IW) {
                                continue;
                            }

                            float sv = *(const float *) ((const char *) b->data
                                    + sx * b->nb[0] + sy * b->nb[1] + sz * b->nb[2]
                                    + (in * C + ic) * b->nb[3]);
                            if (round_src) {
                                sv = gk_fp16_to_fp32(gk_fp32_to_fp16(sv));
                            }

                            acc += gk_conv_sample(a, kx, ky, kz, ioc * C + ic) * sv;
                        }
                    }
                }
            }

            out[ox] = acc;
        }
    }
}

// transposed 1-D convolution: kernel [K, Cout, Cin], input [L, Cin], result
// [(L-1)*s + K, Cout]. Each output row (a Cout) accumulates independently.
static void gk_compute_conv_transpose_1d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * a = dst->src[0];
    const struct gk_tensor * b = dst->src[1];

    GK_ASSERT(b->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);

    const int s0 = gk_get_op_params_i32(dst, 0);

    const bool round_src = a->type == GK_TYPE_F16;

    const int64_t K   = a->ne[0];
    const int64_t Cin = a->ne[2];
    const int64_t L   = b->ne[0];

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t oc = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];

        float * out = (float *) gk_row_ptr_mut(dst, oc, i2, 0);
        memset(out, 0, (size_t) dst->ne[0] * sizeof(float));

        for (int64_t il = 0; il < L; ++il) {
            for (int64_t ik = 0; ik < K; ++ik) {
                float v = 0.0f;
                for (int64_t ic = 0; ic < Cin; ++ic) {
                    float sv = *(const float *) ((const char *) b->data
                            + il * b->nb[0] + ic * b->nb[1] + i2 * b->nb[2]);
                    if (round_src) {
                        sv = gk_fp16_to_fp32(gk_fp32_to_fp16(sv));
                    }
                    v += gk_conv_sample(a, ik, oc, ic, 0) * sv;
                }
                out[il * s0 + ik] += v;
            }
        }
    }
}

// transposed 2-D convolution: kernel [KW, KH, Cout, Cin], input [W, H, Cin, N],
// result [OW, OH, Cout, N]. Each (Cout, image) plane is independent.
static void gk_compute_conv_transpose_2d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * a = dst->src[0];
    const struct gk_tensor * b = dst->src[1];

    GK_ASSERT(b->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);

    const int stride = gk_get_op_params_i32(dst, 0);

    const bool round_src = a->type == GK_TYPE_F16;

    const int64_t KW = a->ne[0], KH = a->ne[1];
    const int64_t Cin = a->ne[3];
    const int64_t W = b->ne[0], H = b->ne[1];

    const int64_t n_units = dst->ne[2] * dst->ne[3];
    const int64_t per = (n_units + st->nth - 1) / st->nth;
    const int64_t u0  = GK_MIN(per * st->ith, n_units);
    const int64_t u1  = GK_MIN(u0 + per, n_units);

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t oc = u % dst->ne[2];
        const int64_t in = u / dst->ne[2];

        char * plane = (char *) dst->data + oc * dst->nb[2] + in * dst->nb[3];
        for (int64_t oy = 0; oy < dst->ne[1]; ++oy) {
            memset(plane + oy * dst->nb[1], 0, (size_t) dst->ne[0] * sizeof(float));
        }

        for (int64_t sy = 0; sy < H; ++sy) {
            for (int64_t sx = 0; sx < W; ++sx) {
                for (int64_t ky = 0; ky < KH; ++ky) {
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        float v = 0.0f;
                        for (int64_t ic = 0; ic < Cin; ++ic) {
                            float sv = *(const float *) ((const char *) b->data
                                    + sx * b->nb[0] + sy * b->nb[1] + ic * b->nb[2] + in * b->nb[3]);
                            if (round_src) {
                                sv = gk_fp16_to_fp32(gk_fp32_to_fp16(sv));
                            }
                            v += gk_conv_sample(a, kx, ky, oc, ic) * sv;
                        }

                        *(float *) (plane + (sy * stride + ky) * dst->nb[1]
                                + (sx * stride + kx) * dst->nb[0]) += v;
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// pooling and windows
// --------------------------------------------------------------------------

static void gk_compute_pool_1d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 || src0->type == GK_TYPE_F16);
    GK_ASSERT(dst->type == GK_TYPE_F32);

    const enum gk_op_pool op = (enum gk_op_pool) gk_get_op_params_i32(dst, 0);
    const int k0 = gk_get_op_params_i32(dst, 1);
    const int s0 = gk_get_op_params_i32(dst, 2);
    const int p0 = gk_get_op_params_i32(dst, 3);

    const int64_t IW = src0->ne[0];

    float * a = gk_scratch(st, 0);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const float * pa = gk_row_read(src0, i1, i2, i3, a);
        float * out = (float *) gk_row_ptr_mut(dst, i1, i2, i3);

        for (int64_t ow = 0; ow < dst->ne[0]; ++ow) {
            float res = op == GK_OP_POOL_MAX ? -FLT_MAX : 0.0f;
            int count = 0;

            const int64_t base = ow * s0 - p0;
            for (int ki = 0; ki < k0; ++ki) {
                const int64_t j = base + ki;
                if (j < 0 || j >= IW) {
                    continue;
                }
                const float v = pa[j];
                if (op == GK_OP_POOL_MAX) {
                    res = fmaxf(res, v);
                } else {
                    res += v;
                }
                ++count;
            }

            if (op == GK_OP_POOL_AVG) {
                res = count > 0 ? res / (float) count : 0.0f;
            }
            out[ow] = res;
        }
    }
}

static void gk_compute_pool_2d(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 || src0->type == GK_TYPE_F16);
    GK_ASSERT(dst->type == GK_TYPE_F32);

    const enum gk_op_pool op = (enum gk_op_pool) gk_get_op_params_i32(dst, 0);
    const int k0 = gk_get_op_params_i32(dst, 1);
    const int k1 = gk_get_op_params_i32(dst, 2);
    const int s0 = gk_get_op_params_i32(dst, 3);
    const int s1 = gk_get_op_params_i32(dst, 4);
    const int p0 = gk_get_op_params_i32(dst, 5);
    const int p1 = gk_get_op_params_i32(dst, 6);

    const int64_t IW = src0->ne[0], IH = src0->ne[1];
    const float ka = (float) (k0 * k1);

    float * a = gk_scratch(st, 0);

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t oy = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        float * out = (float *) gk_row_ptr_mut(dst, oy, i2, i3);

        for (int64_t ox = 0; ox < dst->ne[0]; ++ox) {
            float res = op == GK_OP_POOL_MAX ? -FLT_MAX : 0.0f;

            const int64_t ix = ox * s0 - p0;
            const int64_t iy = oy * s1 - p1;

            for (int ky = 0; ky < k1; ++ky) {
                if (iy + ky < 0 || iy + ky >= IH) {
                    continue;
                }
                const float * row = gk_row_read(src0, iy + ky, i2, i3, a);
                for (int kx = 0; kx < k0; ++kx) {
                    const int64_t j = ix + kx;
                    if (j < 0 || j >= IW) {
                        continue;
                    }
                    if (op == GK_OP_POOL_MAX) {
                        res = fmaxf(res, row[j]);
                    } else {
                        res += row[j];
                    }
                }
            }

            // avg pool divides by the full kernel area, padding included -
            // that asymmetry with pool_1d is inherited behaviour the models
            // were trained against
            if (op == GK_OP_POOL_AVG) {
                res /= ka;
            }
            out[ox] = res;
        }
    }
}

static void gk_compute_win_part(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);
    GK_ASSERT(gk_is_contiguous(src0) && gk_is_contiguous(dst));

    const int npx = gk_get_op_params_i32(dst, 0);
    const int w   = gk_get_op_params_i32(dst, 2);

    const float * src = (const float *) src0->data;
    float * out = (float *) dst->data;

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = ir / (dst->ne[1] * dst->ne[2]);

        const int64_t px = i3 % npx;
        const int64_t py = i3 / npx;

        const int64_t i01 = px * w + i1;
        const int64_t i02 = py * w + i2;

        float * row = out + ((i3 * dst->ne[2] + i2) * dst->ne[1] + i1) * dst->ne[0];

        if (i01 >= src0->ne[1] || i02 >= src0->ne[2]) {
            memset(row, 0, (size_t) dst->ne[0] * sizeof(float));
        } else {
            memcpy(row, src + (i02 * src0->ne[1] + i01) * src0->ne[0],
                   (size_t) dst->ne[0] * sizeof(float));
        }
    }
}

static void gk_compute_win_unpart(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];

    GK_ASSERT(src0->type == GK_TYPE_F32 && dst->type == GK_TYPE_F32);
    GK_ASSERT(gk_is_contiguous(src0) && gk_is_contiguous(dst));

    const int w = gk_get_op_params_i32(dst, 0);

    const int64_t px  = (w - dst->ne[1] % w) % w;
    const int64_t npx = (px + dst->ne[1]) / w;

    const float * src = (const float *) src0->data;
    float * out = (float *) dst->data;

    int64_t ir0, ir1;
    gk_rows_for_thread(dst, st->ith, st->nth, &ir0, &ir1);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i1 = ir % dst->ne[1];
        const int64_t i2 = (ir / dst->ne[1]) % dst->ne[2];

        const int64_t ip1 = i1 / w;
        const int64_t ip2 = i2 / w;

        const int64_t j = ((ip2 * npx + ip1) * src0->ne[2] + (i2 % w)) * src0->ne[1] + (i1 % w);

        memcpy(out + (i2 * dst->ne[1] + i1) * dst->ne[0],
               src + j * src0->ne[0],
               (size_t) dst->ne[0] * sizeof(float));
    }
}

// --------------------------------------------------------------------------
// recurrent layers
//
// Every kernel in this section carries a per-head [S, S] state across the
// token loop, so tokens are inherently sequential; the parallelism is across
// heads and sequences, which own disjoint states and disjoint output slices.
// That split keeps the bit-identical-across-thread-counts property.
// --------------------------------------------------------------------------

static void gk_compute_ssm_conv(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * sx = dst->src[0];
    const struct gk_tensor * c  = dst->src[1];

    GK_ASSERT(sx->type == GK_TYPE_F32 && c->type == GK_TYPE_F32);

    const int64_t nc  = c->ne[0];   // d_conv
    const int64_t ncs = sx->ne[0];  // d_conv - 1 + n_t
    const int64_t nr  = sx->ne[1];  // d_inner
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    GK_ASSERT(sx->nb[0] == sizeof(float));
    GK_ASSERT(c->nb[0] == sizeof(float));
    GK_ASSERT(sx->nb[1] == (size_t) ncs * sizeof(float));

    const int64_t per = (nr + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, nr);
    const int64_t r1  = GK_MIN(r0 + per, nr);

    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t i2 = 0; i2 < n_t; ++i2) {
            // a sliding window over the padded input
            const float * s = (const float *) ((const char *) sx->data
                    + r0 * sx->nb[1] + i2 * sx->nb[0] + i3 * sx->nb[2]);
            const float * w = (const float *) ((const char *) c->data + r0 * c->nb[1]);
            float * x = (float *) ((char *) dst->data
                    + r0 * dst->nb[0] + i2 * dst->nb[1] + i3 * dst->nb[2]);

            for (int64_t i1 = 0; i1 < r1 - r0; ++i1) {
                float sum = 0.0f;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    sum += s[i0 + i1 * ncs] * w[i0 + i1 * nc];
                }
                x[i1] = sum;
            }
        }
    }
}

static void gk_compute_ssm_scan(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * s_in = dst->src[0]; // {d_state, dim, n_head, n_seqs+}
    const struct gk_tensor * x    = dst->src[1]; // {dim, n_head, n_tok, n_seqs}
    const struct gk_tensor * dt   = dst->src[2]; // {n_head, n_tok, n_seqs}
    const struct gk_tensor * A    = dst->src[3]; // {d_state, n_head} or {1, n_head}
    const struct gk_tensor * B    = dst->src[4]; // {d_state, n_group, n_tok, n_seqs}
    const struct gk_tensor * C    = dst->src[5]; // same shape as B
    const struct gk_tensor * ids  = dst->src[6]; // {n_seqs}

    const int64_t nc = s_in->ne[0]; // d_state
    const int64_t nr = s_in->ne[1]; // dim
    const int64_t nh = x->ne[1];    // n_head
    const int64_t ng = B->ne[1];
    const int64_t nt = x->ne[2];
    const int64_t ns = x->ne[3];

    const int64_t s_off = gk_nelements(x) * sizeof(float);

    GK_ASSERT(gk_nelements(x) + nc * nr * nh * ns == gk_nelements(dst));
    GK_ASSERT(nh % ng == 0);

    // heads split across threads; a head's state rows are its own
    const int64_t per = (nh + st->nth - 1) / st->nth;
    const int64_t h0  = GK_MIN(per * st->ith, nh);
    const int64_t h1  = GK_MIN(h0 + per, nh);

    const int32_t * seq_ids = (const int32_t *) ids->data;

    for (int64_t i3 = 0; i3 < ns; ++i3) {
        const float * s0 = (const float *) ((const char *) s_in->data
                + seq_ids[i3] * s_in->nb[3]);
        float * s = (float *) ((char *) dst->data + i3 * s_in->nb[3] + s_off);

        for (int64_t i2 = 0; i2 < nt; ++i2) {
            const float * xt = (const float *) ((const char *) x->data
                    + i2 * x->nb[2] + i3 * x->nb[3]);
            const float * dtt = (const float *) ((const char *) dt->data
                    + i2 * dt->nb[1] + i3 * dt->nb[2]);
            const float * Ad = (const float *) A->data;
            const float * Bt = (const float *) ((const char *) B->data
                    + i2 * B->nb[2] + i3 * B->nb[3]);
            const float * Ct = (const float *) ((const char *) C->data
                    + i2 * C->nb[2] + i3 * C->nb[3]);
            float * y = (float *) ((char *) dst->data
                    + i2 * nh * nr * sizeof(float) + i3 * nt * nh * nr * sizeof(float));

            for (int64_t h = h0; h < h1; ++h) {
                const float dts = dtt[h] > 20.0f ? dtt[h] : logf(1.0f + expf(dtt[h]));
                const int64_t g = h / (nh / ng);

                if (A->ne[0] == 1) {
                    // Mamba-2: one decay per head
                    const float dA = expf(dts * Ad[h]);

                    for (int64_t i1 = 0; i1 < nr; ++i1) {
                        const int64_t ii = i1 + h * nr;
                        const float x_dt = xt[ii] * dts;

                        float sum = 0.0f;
                        for (int64_t i0 = 0; i0 < nc; ++i0) {
                            const int64_t i  = i0 + ii * nc;
                            const int64_t ig = i0 + g * nc;
                            const float state = s0[i] * dA + Bt[ig] * x_dt;
                            sum += state * Ct[ig];
                            s[i] = state;
                        }
                        y[ii] = sum;
                    }
                } else {
                    // Mamba-1: a decay per state element
                    for (int64_t i1 = 0; i1 < nr; ++i1) {
                        const int64_t ii = i1 + h * nr;
                        const float x_dt = xt[ii] * dts;

                        float sum = 0.0f;
                        for (int64_t i0 = 0; i0 < nc; ++i0) {
                            const int64_t i  = i0 + ii * nc;
                            const int64_t ig = i0 + g * nc;
                            const float state = s0[i] * expf(dts * Ad[i0 + h * nc]) + Bt[ig] * x_dt;
                            sum += state * Ct[ig];
                            s[i] = state;
                        }
                        y[ii] = sum;
                    }
                }
            }

            // from the second token on, the state chains through the output
            s0 = s;
        }
    }
}

// The three RWKV-shaped kernels share their layout: inputs [S, H, n_tokens],
// state [S*S*H per sequence], output rows first and final states after them.
static void gk_compute_rwkv_wkv6(struct gk_compute_state * st, struct gk_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[5]->ne[1];
    const int64_t S = C / H;

    float * out   = (float *) dst->data;
    float * state = (float *) dst->data + C * T;

    const float * k  = (const float *) dst->src[0]->data;
    const float * v  = (const float *) dst->src[1]->data;
    const float * r  = (const float *) dst->src[2]->data;
    const float * tf = (const float *) dst->src[3]->data;
    const float * td = (const float *) dst->src[4]->data;

    GK_ASSERT(C % H == 0);

    const int64_t hpt = (H + st->nth - 1) / st->nth;
    const int64_t hs  = GK_MIN(hpt * st->ith, H);
    const int64_t he  = GK_MIN(hs + hpt, H);

    // the output accumulates over the row loop, so this thread's head slices
    // are zeroed first - each thread owns its own, so no barrier is needed
    for (int64_t t = 0; t < T; ++t) {
        memset(out + t * C + hs * S, 0, (size_t) (he - hs) * S * sizeof(float));
    }

    for (int64_t t = 0; t < T; ++t) {
        const int64_t t_off = t * C;
        const int64_t state_off = S * C * (t / (T / n_seqs));
        float * state_cur = state + state_off;
        const float * state_prev = t % (T / n_seqs)
            ? state_cur
            : (const float *) dst->src[5]->data + state_off;

        for (int64_t h = hs; h < he; ++h) {
            const int64_t h_off = h * S;
            const int64_t th_off = t_off + h_off;
            const int64_t h2d = h * S * S;

            for (int64_t i = 0; i < S; ++i) {
                const float kv = k[th_off + i];
                const float rv = r[th_off + i];
                const float tfv = tf[h_off + i];
                const float tdv = td[th_off + i];

                for (int64_t j = 0; j < S; ++j) {
                    const float kvv = v[th_off + j] * kv;
                    const float prev = state_prev[h2d + i * S + j];
                    const float tmp = kvv * tfv + prev;
                    out[th_off + j] += tmp * rv;
                    state_cur[h2d + i * S + j] = prev * tdv + kvv;
                }
            }
        }
    }
}

static void gk_compute_rwkv_wkv7(struct gk_compute_state * st, struct gk_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[6]->ne[1];
    const int64_t S = C / H;

    float * out   = (float *) dst->data;
    float * state = (float *) dst->data + C * T;

    const float * r = (const float *) dst->src[0]->data;
    const float * w = (const float *) dst->src[1]->data;
    const float * k = (const float *) dst->src[2]->data;
    const float * v = (const float *) dst->src[3]->data;
    const float * a = (const float *) dst->src[4]->data;
    const float * b = (const float *) dst->src[5]->data;

    GK_ASSERT(C % H == 0);

    const int64_t hpt = (H + st->nth - 1) / st->nth;
    const int64_t hs  = GK_MIN(hpt * st->ith, H);
    const int64_t he  = GK_MIN(hs + hpt, H);

    for (int64_t t = 0; t < T; ++t) {
        const int64_t t_off = t * C;
        const int64_t state_off = S * C * (t / (T / n_seqs));
        float * state_cur = state + state_off;
        const float * state_prev = t % (T / n_seqs)
            ? state_cur
            : (const float *) dst->src[6]->data + state_off;

        for (int64_t h = hs; h < he; ++h) {
            const int64_t th_off = t_off + h * S;
            const int64_t h2d = h * S * S;

            for (int64_t i = 0; i < S; ++i) {
                const float vv = v[th_off + i];

                // in-context learning rate: project the previous state row
                // onto a, then fold it back in through b
                float sa = 0.0f;
                for (int64_t j = 0; j < S; ++j) {
                    sa += a[th_off + j] * state_prev[h2d + i * S + j];
                }

                float res = 0.0f;
                for (int64_t j = 0; j < S; ++j) {
                    const float kvv = vv * k[th_off + j];
                    const float prev = state_prev[h2d + i * S + j];
                    const float cur = prev * w[th_off + j] + kvv + sa * b[th_off + j];
                    state_cur[h2d + i * S + j] = cur;
                    res += cur * r[th_off + j];
                }
                out[th_off + i] = res;
            }
        }
    }
}

static void gk_compute_gla(struct gk_compute_state * st, struct gk_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[4]->ne[1];
    const int64_t S = C / H;
    const float scale = gk_get_op_params_f32(dst, 0);

    float * out   = (float *) dst->data;
    float * state = (float *) dst->data + C * T;

    const float * k = (const float *) dst->src[0]->data;
    const float * v = (const float *) dst->src[1]->data;
    const float * q = (const float *) dst->src[2]->data;
    const float * g = (const float *) dst->src[3]->data;

    GK_ASSERT(C % H == 0);

    const int64_t hpt = (H + st->nth - 1) / st->nth;
    const int64_t hs  = GK_MIN(hpt * st->ith, H);
    const int64_t he  = GK_MIN(hs + hpt, H);

    for (int64_t t = 0; t < T; ++t) {
        memset(out + t * C + hs * S, 0, (size_t) (he - hs) * S * sizeof(float));
    }

    for (int64_t t = 0; t < T; ++t) {
        const int64_t t_off = t * C;
        const int64_t state_off = S * C * (t / (T / n_seqs));
        float * state_cur = state + state_off;
        const float * state_prev = t % (T / n_seqs)
            ? state_cur
            : (const float *) dst->src[4]->data + state_off;

        for (int64_t h = hs; h < he; ++h) {
            const int64_t th_off = t_off + h * S;
            const int64_t h2d = h * S * S;

            for (int64_t i = 0; i < S; ++i) {
                const float kv = k[th_off + i];
                const float qv = q[th_off + i] * scale;
                const float gv = g[th_off + i];

                for (int64_t j = 0; j < S; ++j) {
                    const float kvv = v[th_off + j] * kv;
                    const float prev = state_prev[h2d + i * S + j];
                    const float tmp = prev * gv + kvv;
                    out[th_off + j] += tmp * qv;
                    state_cur[h2d + i * S + j] = tmp;
                }
            }
        }
    }
}

static void gk_compute_gated_delta_net(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * q    = dst->src[0];
    const struct gk_tensor * k    = dst->src[1];
    const struct gk_tensor * v    = dst->src[2];
    const struct gk_tensor * g    = dst->src[3];
    const struct gk_tensor * beta = dst->src[4];
    const struct gk_tensor * s_in = dst->src[5];

    const int64_t S = v->ne[0];
    const int64_t H = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];

    const bool kda = g->ne[0] == S;

    const int64_t K = gk_get_op_params_i32(dst, 0);
    GK_ASSERT(K >= 1);

    const int64_t attn_elems = S * H * n_tokens * n_seqs;
    const int64_t snap_elems = S * S * H * n_seqs;

    float * attn_base  = (float *) dst->data;
    float * state_base = (float *) dst->data + attn_elems;

    float * delta = gk_scratch(st, 0); // S floats

    // q/k may carry fewer heads than v and broadcast over the ratio
    const int64_t rq3 = v->ne[3] / q->ne[3];
    const int64_t rk3 = v->ne[3] / k->ne[3];

    const float scale = 1.0f / sqrtf((float) S);

    const int64_t n_units = H * n_seqs;
    const int64_t per = (n_units + st->nth - 1) / st->nth;
    const int64_t u0  = GK_MIN(per * st->ith, n_units);
    const int64_t u1  = GK_MIN(u0 + per, n_units);

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t iv1 = u % H; // head
        const int64_t iv3 = u / H; // sequence

        const int64_t iq1 = iv1 % q->ne[1];
        const int64_t ik1 = iv1 % k->ne[1];
        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        // slot 0 of the output doubles as the working state; older snapshots
        // are copied out as the token loop passes them
        float * s_work = state_base + (iv3 * H + iv1) * S * S;

        const float * s0 = (const float *) s_in->data
            + iv3 * (s_in->nb[3] / sizeof(float)) + iv1 * S * S;
        memcpy(s_work, s0, (size_t) S * S * sizeof(float));

        float * attn = attn_base + (iv3 * n_tokens * H + iv1) * S;

        for (int64_t t = 0; t < n_tokens; ++t) {
            const float * q_d = (const float *) ((const char *) q->data
                    + iq3 * q->nb[3] + t * q->nb[2] + iq1 * q->nb[1]);
            const float * k_d = (const float *) ((const char *) k->data
                    + ik3 * k->nb[3] + t * k->nb[2] + ik1 * k->nb[1]);
            const float * v_d = (const float *) ((const char *) v->data
                    + iv3 * v->nb[3] + t * v->nb[2] + iv1 * v->nb[1]);

            const float beta_v = *(const float *) ((const char *) beta->data
                    + iv3 * beta->nb[3] + t * beta->nb[2] + iv1 * beta->nb[1]);
            const float * g_d = (const float *) ((const char *) g->data
                    + iv3 * g->nb[3] + t * g->nb[2] + iv1 * g->nb[1]);

            // decay the state: per channel for KDA, one scalar otherwise.
            // the state is stored transposed - row j holds column j of S.
            if (kda) {
                for (int64_t i = 0; i < S; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                for (int64_t j = 0; j < S; ++j) {
                    for (int64_t i = 0; i < S; ++i) {
                        s_work[j * S + i] *= delta[i];
                    }
                }
            } else {
                const float dg = expf(g_d[0]);
                for (int64_t i = 0; i < S * S; ++i) {
                    s_work[i] *= dg;
                }
            }

            // the delta rule: how far the state's prediction of v misses
            for (int64_t j = 0; j < S; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < S; ++i) {
                    sum += s_work[j * S + i] * k_d[i];
                }
                delta[j] = (v_d[j] - sum) * beta_v;
            }

            // rank-one update, then read the state out against q
            for (int64_t j = 0; j < S; ++j) {
                const float dj = delta[j];
                for (int64_t i = 0; i < S; ++i) {
                    s_work[j * S + i] += k_d[i] * dj;
                }
            }

            for (int64_t j = 0; j < S; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < S; ++i) {
                    sum += s_work[j * S + i] * q_d[i];
                }
                attn[j] = sum * scale;
            }

            attn += S * H;

            if (K > 1) {
                const int64_t slot = n_tokens - 1 - t;
                if (slot > 0 && slot < K) {
                    memcpy(state_base + slot * snap_elems + (iv3 * H + iv1) * S * S,
                           s_work, (size_t) S * S * sizeof(float));
                }
            }
        }
    }
}

static void gk_compute_solve_tri(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * a = dst->src[0]; // lower-triangular A
    const struct gk_tensor * b = dst->src[1]; // right-hand side

    const int64_t n = b->ne[1]; // A is n x n
    const int64_t kk = b->ne[0]; // RHS columns

    // one (batch, column) strand per unit: forward substitution is sequential
    // down the rows, but every column is independent
    const int64_t n_units = a->ne[2] * a->ne[3] * kk;
    const int64_t per = (n_units + st->nth - 1) / st->nth;
    const int64_t u0  = GK_MIN(per * st->ith, n_units);
    const int64_t u1  = GK_MIN(u0 + per, n_units);

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t i03 = u / (a->ne[2] * kk);
        const int64_t i02 = (u - i03 * a->ne[2] * kk) / kk;
        const int64_t i01 = u - i03 * a->ne[2] * kk - i02 * kk;

        const float * A = (const float *) ((const char *) a->data + i02 * a->nb[2] + i03 * a->nb[3]);
        const float * B = (const float *) ((const char *) b->data + i02 * b->nb[2] + i03 * b->nb[3]);
        float * X = (float *) ((char *) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        for (int64_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int64_t t = 0; t < i; ++t) {
                sum += A[i * n + t] * X[t * kk + i01];
            }
            X[i * kk + i01] = (B[i * kk + i01] - sum) / A[i * n + i];
        }
    }
}

static void gk_compute_lightning_indexer(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * q = dst->src[0];
    const struct gk_tensor * k = dst->src[1];
    const struct gk_tensor * w = dst->src[2];
    const struct gk_tensor * m = dst->src[3];

    GK_ASSERT(q->type == GK_TYPE_F32 && w->type == GK_TYPE_F32 && m->type == GK_TYPE_F16);

    const int64_t n_embd   = q->ne[0];
    const int64_t n_head   = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_stream = q->ne[3];
    const int64_t n_kv     = k->ne[2];

    const struct gk_type_traits * ktr = gk_get_type_traits(k->type);
    GK_ASSERT(k->type == GK_TYPE_F32 || ktr->to_float != NULL);

    float * kbuf = gk_scratch(st, 0);

    const int64_t per = (n_kv + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, n_kv);
    const int64_t r1  = GK_MIN(r0 + per, n_kv);

    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            const float * w_row = (const float *) ((const char *) w->data
                    + t * w->nb[1] + s * w->nb[3]);
            const gk_fp16_t * m_row = (const gk_fp16_t *) ((const char *) m->data
                    + t * m->nb[1] + (s % m->ne[3]) * m->nb[3]);
            float * out = (float *) ((char *) dst->data + t * dst->nb[1] + s * dst->nb[3]);

            for (int64_t ik = r0; ik < r1; ++ik) {
                const char * k_row = (const char *) k->data + ik * k->nb[2] + s * k->nb[3];

                const float * kf;
                if (k->type == GK_TYPE_F32) {
                    kf = (const float *) k_row;
                } else {
                    ktr->to_float(k_row, kbuf, n_embd);
                    kf = kbuf;
                }

                float score = 0.0f;
                for (int64_t h = 0; h < n_head; ++h) {
                    const float * q_row = (const float *) ((const char *) q->data
                            + h * q->nb[1] + t * q->nb[2] + s * q->nb[3]);
                    float qk = 0.0f;
                    for (int64_t i = 0; i < n_embd; ++i) {
                        qk += q_row[i] * kf[i];
                    }
                    // scores rectify before weighting; the weights arrive
                    // prescaled by the caller
                    score += fmaxf(qk, 0.0f) * w_row[h];
                }

                out[ik] = score + gk_fp16_to_fp32(m_row[ik]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// DeepSeek V4 hyper-connections
// --------------------------------------------------------------------------

#define GK_DSV4_HC 4

static void gk_dsv4_norm_cols(float * comb, float eps) {
    for (int i = 0; i < GK_DSV4_HC; ++i) {
        float sum = eps;
        for (int j = 0; j < GK_DSV4_HC; ++j) {
            sum += comb[i + GK_DSV4_HC * j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < GK_DSV4_HC; ++j) {
            comb[i + GK_DSV4_HC * j] *= inv;
        }
    }
}

static void gk_dsv4_norm_rows(float * comb, float eps) {
    for (int j = 0; j < GK_DSV4_HC; ++j) {
        float sum = eps;
        for (int i = 0; i < GK_DSV4_HC; ++i) {
            sum += comb[i + GK_DSV4_HC * j];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < GK_DSV4_HC; ++i) {
            comb[i + GK_DSV4_HC * j] *= inv;
        }
    }
}

static void gk_compute_dsv4_hc_comb(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * mixes = dst->src[0];
    const struct gk_tensor * scale = dst->src[1];
    const struct gk_tensor * base  = dst->src[2];

    enum { hc = GK_DSV4_HC, comb_off = 2 * hc };

    const int64_t n_tokens = mixes->ne[1];

    const float eps    = gk_get_op_params_f32(dst, 0);
    const int   n_iter = gk_get_op_params_i32(dst, 1);

    const float scale_comb = *(const float *) ((const char *) scale->data + 2 * scale->nb[0]);

    const int64_t per = (n_tokens + st->nth - 1) / st->nth;
    const int64_t t0  = GK_MIN(per * st->ith, n_tokens);
    const int64_t t1  = GK_MIN(t0 + per, n_tokens);

    for (int64_t it = t0; it < t1; ++it) {
        float comb[hc * hc];

        // per source stream: a softmax over destinations, plus the eps floor
        for (int j = 0; j < hc; ++j) {
            float max = -INFINITY;
            for (int i = 0; i < hc; ++i) {
                const int idx = i + hc * j;
                const float xv = *(const float *) ((const char *) mixes->data
                        + (comb_off + idx) * mixes->nb[0] + it * mixes->nb[1]);
                const float bv = *(const float *) ((const char *) base->data
                        + (comb_off + idx) * base->nb[0]);
                comb[idx] = xv * scale_comb + bv;
                max = fmaxf(max, comb[idx]);
            }

            float sum = 0.0f;
            for (int i = 0; i < hc; ++i) {
                const float e = expf(comb[i + hc * j] - max);
                comb[i + hc * j] = e;
                sum += e;
            }

            const float inv = 1.0f / sum;
            for (int i = 0; i < hc; ++i) {
                comb[i + hc * j] = comb[i + hc * j] * inv + eps;
            }
        }

        // Sinkhorn-style alternating normalisation to near-doubly-stochastic
        gk_dsv4_norm_cols(comb, eps);
        for (int i = 1; i < n_iter; ++i) {
            gk_dsv4_norm_rows(comb, eps);
            gk_dsv4_norm_cols(comb, eps);
        }

        for (int j = 0; j < hc; ++j) {
            for (int i = 0; i < hc; ++i) {
                *(float *) ((char *) dst->data
                        + i * dst->nb[0] + j * dst->nb[1] + it * dst->nb[2]) = comb[i + hc * j];
            }
        }
    }
}

static void gk_compute_dsv4_hc_pre(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * x = dst->src[0];
    const struct gk_tensor * w = dst->src[1];

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int64_t nr = n_embd * n_tokens;
    const int64_t per = (nr + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, nr);
    const int64_t r1  = GK_MIN(r0 + per, nr);

    for (int64_t ir = r0; ir < r1; ++ir) {
        const int64_t i0 = ir % n_embd;
        const int64_t it = ir / n_embd;

        float sum = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const float xv = *(const float *) ((const char *) x->data
                    + i0 * x->nb[0] + ih * x->nb[1] + it * x->nb[2]);
            const float wv = *(const float *) ((const char *) w->data
                    + ih * w->nb[0] + it * w->nb[1]);
            sum += xv * wv;
        }

        *(float *) ((char *) dst->data + i0 * dst->nb[0] + it * dst->nb[1]) = sum;
    }
}

static void gk_compute_dsv4_hc_post(struct gk_compute_state * st, struct gk_tensor * dst) {
    const struct gk_tensor * x    = dst->src[0];
    const struct gk_tensor * res  = dst->src[1];
    const struct gk_tensor * post = dst->src[2];
    const struct gk_tensor * comb = dst->src[3];

    const int64_t n_embd = x->ne[0];
    const int64_t hc     = res->ne[1];

    const int64_t nr = gk_nelements(dst);
    const int64_t per = (nr + st->nth - 1) / st->nth;
    const int64_t r0  = GK_MIN(per * st->ith, nr);
    const int64_t r1  = GK_MIN(r0 + per, nr);

    for (int64_t ir = r0; ir < r1; ++ir) {
        const int64_t i0 = ir % n_embd;
        const int64_t id = (ir / n_embd) % hc;
        const int64_t it = ir / (n_embd * hc);

        const float xv = *(const float *) ((const char *) x->data
                + i0 * x->nb[0] + it * x->nb[1]);
        const float pv = *(const float *) ((const char *) post->data
                + id * post->nb[0] + it * post->nb[1]);

        float sum = xv * pv;
        for (int64_t is = 0; is < hc; ++is) {
            const float rv = *(const float *) ((const char *) res->data
                    + i0 * res->nb[0] + is * res->nb[1] + it * res->nb[2]);
            const float cv = *(const float *) ((const char *) comb->data
                    + id * comb->nb[0] + is * comb->nb[1] + it * comb->nb[2]);
            sum += rv * cv;
        }

        *(float *) ((char *) dst->data
                + i0 * dst->nb[0] + id * dst->nb[1] + it * dst->nb[2]) = sum;
    }
}

// --------------------------------------------------------------------------
// custom ops - hand the node to the caller's function
// --------------------------------------------------------------------------

static void gk_compute_custom(struct gk_compute_state * st, struct gk_tensor * dst) {
    struct gk_custom_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    // the op may cap its own parallelism
    int nth = st->nth;
    if (p.n_tasks != GK_N_TASKS_MAX && p.n_tasks < nth) {
        nth = p.n_tasks;
    }
    if (st->ith >= nth) {
        return;
    }

    switch (dst->op) {
        case GK_OP_MAP_CUSTOM1:
            ((gk_custom1_op_t) p.fun)(dst, dst->src[0], st->ith, nth, p.userdata);
            break;
        case GK_OP_MAP_CUSTOM2:
            ((gk_custom2_op_t) p.fun)(dst, dst->src[0], dst->src[1], st->ith, nth, p.userdata);
            break;
        case GK_OP_MAP_CUSTOM3:
            ((gk_custom3_op_t) p.fun)(dst, dst->src[0], dst->src[1], dst->src[2],
                                      st->ith, nth, p.userdata);
            break;
        case GK_OP_CUSTOM:
            ((gk_custom_op_t) p.fun)(dst, st->ith, nth, p.userdata);
            break;
        default:
            GK_ABORT("not a custom op: %s", gk_op_name(dst->op));
    }
}

// --------------------------------------------------------------------------
// dispatch
// --------------------------------------------------------------------------

static void gk_compute_node(struct gk_compute_state * st, struct gk_tensor * node) {
    // A node with a zero extent has no elements to write. The per-op kernels
    // below assume at least one row and would divide by, or index past, a zero
    // dimension, so an empty result is a no-op here.
    if (gk_is_empty(node)) {
        return;
    }

    switch (node->op) {
        // Pure reinterpretations: the result already aliases the operand's
        // memory, so there is nothing to do.
        case GK_OP_NONE:
        case GK_OP_RESHAPE:
        case GK_OP_VIEW:
        case GK_OP_PERMUTE:
        case GK_OP_TRANSPOSE:
            break;

        case GK_OP_ADD: gk_compute_binary(st, node, GK_BIN_ADD); break;
        case GK_OP_SUB: gk_compute_binary(st, node, GK_BIN_SUB); break;
        case GK_OP_MUL: gk_compute_binary(st, node, GK_BIN_MUL); break;
        case GK_OP_DIV: gk_compute_binary(st, node, GK_BIN_DIV); break;

        case GK_OP_SQR:
        case GK_OP_SQRT:
        case GK_OP_LOG:
        case GK_OP_SIN:
        case GK_OP_COS:
            gk_compute_simple_unary(st, node);
            break;

        case GK_OP_UNARY: gk_compute_unary(st, node); break;
        case GK_OP_GLU:   gk_compute_glu  (st, node); break;

        case GK_OP_SCALE: gk_compute_scale(st, node); break;
        case GK_OP_CLAMP: gk_compute_clamp(st, node); break;

        case GK_OP_NORM:
        case GK_OP_RMS_NORM:
        case GK_OP_L2_NORM:
            gk_compute_norm(st, node);
            break;

        case GK_OP_GROUP_NORM: gk_compute_group_norm(st, node); break;

        case GK_OP_MUL_MAT:    gk_compute_mul_mat   (st, node); break;
        case GK_OP_MUL_MAT_ID: gk_compute_mul_mat_id(st, node); break;

        case GK_OP_DUP:
        case GK_OP_CPY:
        case GK_OP_CONT:
            gk_compute_copy(st, node);
            break;

        case GK_OP_GET_ROWS: gk_compute_get_rows(st, node); break;
        case GK_OP_REPEAT:   gk_compute_repeat  (st, node); break;
        case GK_OP_CONCAT:   gk_compute_concat  (st, node); break;

        case GK_OP_SOFT_MAX: gk_compute_soft_max(st, node); break;

        case GK_OP_DIAG_MASK_INF:
        case GK_OP_DIAG_MASK_ZERO:
            gk_compute_diag_mask(st, node);
            break;

        case GK_OP_ROPE:           gk_compute_rope          (st, node); break;
        case GK_OP_FLASH_ATTN_EXT: gk_compute_flash_attn_ext(st, node); break;

        case GK_OP_SUM:      gk_compute_sum     (st, node); break;
        case GK_OP_SUM_ROWS:
        case GK_OP_MEAN:
            gk_compute_sum_rows(st, node);
            break;

        case GK_OP_ARGMAX:  gk_compute_argmax (st, node); break;
        case GK_OP_ARGSORT: gk_compute_argsort(st, node); break;
        case GK_OP_TOP_K:   gk_compute_top_k  (st, node); break;

        case GK_OP_LEAKY_RELU:  gk_compute_leaky_relu (st, node); break;
        case GK_OP_CUMSUM:      gk_compute_cumsum     (st, node); break;
        case GK_OP_COUNT_EQUAL: gk_compute_count_equal(st, node); break;
        case GK_OP_ADD_ID:      gk_compute_add_id     (st, node); break;

        case GK_OP_ACC:
        case GK_OP_SET:
            gk_compute_acc_set(st, node);
            break;

        case GK_OP_SET_ROWS: gk_compute_set_rows(st, node); break;
        case GK_OP_DIAG:     gk_compute_diag    (st, node); break;
        case GK_OP_TRI:      gk_compute_tri     (st, node); break;
        case GK_OP_FILL:     gk_compute_fill    (st, node); break;
        case GK_OP_ARANGE:   gk_compute_arange  (st, node); break;
        case GK_OP_ROLL:     gk_compute_roll    (st, node); break;

        case GK_OP_TIMESTEP_EMBEDDING: gk_compute_timestep_embedding(st, node); break;

        case GK_OP_PAD:            gk_compute_pad           (st, node); break;
        case GK_OP_PAD_REFLECT_1D: gk_compute_pad_reflect_1d(st, node); break;
        case GK_OP_UPSCALE:        gk_compute_upscale       (st, node); break;

        case GK_OP_SSM_CONV:          gk_compute_ssm_conv         (st, node); break;
        case GK_OP_SSM_SCAN:          gk_compute_ssm_scan         (st, node); break;
        case GK_OP_RWKV_WKV6:         gk_compute_rwkv_wkv6        (st, node); break;
        case GK_OP_RWKV_WKV7:         gk_compute_rwkv_wkv7        (st, node); break;
        case GK_OP_GATED_LINEAR_ATTN: gk_compute_gla              (st, node); break;
        case GK_OP_GATED_DELTA_NET:   gk_compute_gated_delta_net  (st, node); break;
        case GK_OP_SOLVE_TRI:         gk_compute_solve_tri        (st, node); break;
        case GK_OP_LIGHTNING_INDEXER: gk_compute_lightning_indexer(st, node); break;
        case GK_OP_DSV4_HC_COMB:      gk_compute_dsv4_hc_comb     (st, node); break;
        case GK_OP_DSV4_HC_PRE:       gk_compute_dsv4_hc_pre      (st, node); break;
        case GK_OP_DSV4_HC_POST:      gk_compute_dsv4_hc_post     (st, node); break;

        case GK_OP_MAP_CUSTOM1:
        case GK_OP_MAP_CUSTOM2:
        case GK_OP_MAP_CUSTOM3:
        case GK_OP_CUSTOM:
            gk_compute_custom(st, node);
            break;

        case GK_OP_IM2COL:            gk_compute_im2col           (st, node); break;
        case GK_OP_IM2COL_3D:         gk_compute_im2col_3d        (st, node); break;
        case GK_OP_CONV_2D:           gk_compute_conv_2d          (st, node); break;
        case GK_OP_CONV_2D_DW:        gk_compute_conv_2d_dw       (st, node); break;
        case GK_OP_CONV_3D:           gk_compute_conv_3d          (st, node); break;
        case GK_OP_CONV_TRANSPOSE_1D: gk_compute_conv_transpose_1d(st, node); break;
        case GK_OP_CONV_TRANSPOSE_2D: gk_compute_conv_transpose_2d(st, node); break;
        case GK_OP_POOL_1D:           gk_compute_pool_1d          (st, node); break;
        case GK_OP_POOL_2D:           gk_compute_pool_2d          (st, node); break;
        case GK_OP_WIN_PART:          gk_compute_win_part         (st, node); break;
        case GK_OP_WIN_UNPART:        gk_compute_win_unpart       (st, node); break;

        default:
            GK_ABORT("op %s has no CPU kernel yet", gk_op_name(node->op));
    }
}

// Mirrors the dispatch above: which ops this pass can actually evaluate. Kept
// immediately next to the switch so the two do not drift, and used by the
// backend to answer `supports_op` - which is what lets a scheduler route an op
// this backend cannot do somewhere that can.
bool gk_cpu_has_kernel(enum gk_op op) {
    switch (op) {
        case GK_OP_NONE:
        case GK_OP_RESHAPE:
        case GK_OP_VIEW:
        case GK_OP_PERMUTE:
        case GK_OP_TRANSPOSE:
        case GK_OP_ADD:
        case GK_OP_SUB:
        case GK_OP_MUL:
        case GK_OP_DIV:
        case GK_OP_SQR:
        case GK_OP_SQRT:
        case GK_OP_LOG:
        case GK_OP_SIN:
        case GK_OP_COS:
        case GK_OP_UNARY:
        case GK_OP_GLU:
        case GK_OP_SCALE:
        case GK_OP_CLAMP:
        case GK_OP_NORM:
        case GK_OP_RMS_NORM:
        case GK_OP_L2_NORM:
        case GK_OP_GROUP_NORM:
        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
        case GK_OP_DUP:
        case GK_OP_CPY:
        case GK_OP_CONT:
        case GK_OP_GET_ROWS:
        case GK_OP_REPEAT:
        case GK_OP_CONCAT:
        case GK_OP_SOFT_MAX:
        case GK_OP_DIAG_MASK_INF:
        case GK_OP_DIAG_MASK_ZERO:
        case GK_OP_ROPE:
        case GK_OP_FLASH_ATTN_EXT:
        case GK_OP_SUM:
        case GK_OP_SUM_ROWS:
        case GK_OP_MEAN:
        case GK_OP_ARGMAX:
        case GK_OP_ARGSORT:
        case GK_OP_TOP_K:
        case GK_OP_LEAKY_RELU:
        case GK_OP_CUMSUM:
        case GK_OP_COUNT_EQUAL:
        case GK_OP_ADD_ID:
        case GK_OP_ACC:
        case GK_OP_SET:
        case GK_OP_SET_ROWS:
        case GK_OP_DIAG:
        case GK_OP_TRI:
        case GK_OP_FILL:
        case GK_OP_ARANGE:
        case GK_OP_ROLL:
        case GK_OP_TIMESTEP_EMBEDDING:
        case GK_OP_PAD:
        case GK_OP_PAD_REFLECT_1D:
        case GK_OP_UPSCALE:
        case GK_OP_IM2COL:
        case GK_OP_IM2COL_3D:
        case GK_OP_CONV_2D:
        case GK_OP_CONV_2D_DW:
        case GK_OP_CONV_3D:
        case GK_OP_CONV_TRANSPOSE_1D:
        case GK_OP_CONV_TRANSPOSE_2D:
        case GK_OP_POOL_1D:
        case GK_OP_POOL_2D:
        case GK_OP_WIN_PART:
        case GK_OP_WIN_UNPART:
        case GK_OP_SSM_CONV:
        case GK_OP_SSM_SCAN:
        case GK_OP_RWKV_WKV6:
        case GK_OP_RWKV_WKV7:
        case GK_OP_GATED_LINEAR_ATTN:
        case GK_OP_GATED_DELTA_NET:
        case GK_OP_SOLVE_TRI:
        case GK_OP_LIGHTNING_INDEXER:
        case GK_OP_DSV4_HC_COMB:
        case GK_OP_DSV4_HC_PRE:
        case GK_OP_DSV4_HC_POST:
        case GK_OP_MAP_CUSTOM1:
        case GK_OP_MAP_CUSTOM2:
        case GK_OP_MAP_CUSTOM3:
        case GK_OP_CUSTOM:
            return true;
        default:
            return false;
    }
}

// The widest row any kernel in this graph will touch. Kernels concatenate two
// rows in one buffer, so the concat case has to be counted at its full width.
static int64_t gk_graph_max_row(const struct gk_cgraph * graph) {
    int64_t max = 1;

    for (int i = 0; i < graph->n_nodes; ++i) {
        const struct gk_tensor * node = graph->nodes[i];

        if (node->ne[0] > max) {
            max = node->ne[0];
        }
        for (int s = 0; s < GK_MAX_SRC; ++s) {
            if (node->src[s] != NULL && node->src[s]->ne[0] > max) {
                max = node->src[s]->ne[0];
            }
        }
    }

    return max;
}

// One node's worth of work, handed to every thread in the pool.
struct gk_node_job {
    struct gk_tensor * node;
    float *            scratch;
    int64_t            row_size;
};

static void gk_node_worker(void * ctx, int ith, int nth) {
    struct gk_node_job * job = (struct gk_node_job *) ctx;

    struct gk_compute_state st = {
        .scratch  = job->scratch,
        .row_size = job->row_size,
        .ith      = ith,
        .nth      = nth,
    };

    gk_compute_node(&st, job->node);
}

enum gk_status gk_graph_compute_with_pool(struct gk_cgraph * graph, struct gk_pool * pool) {
    const int nth = pool != NULL ? gk_pool_n_threads(pool) : 1;
    const int64_t row = gk_graph_max_row(graph);

    // Every thread gets its own slices, so the block scales with the pool.
    const size_t n_floats = (size_t) row * GK_SCRATCH_ROWS * (size_t) nth;
    float * scratch = (float *) malloc(n_floats * sizeof(float));
    if (scratch == NULL) {
        return GK_STATUS_ALLOC_FAILED;
    }

    enum gk_status status = GK_STATUS_SUCCESS;

    for (int i = 0; i < graph->n_nodes; ++i) {
        struct gk_tensor * node = graph->nodes[i];

        // A node in the graph but not flagged for computing is a placeholder
        // branch from gk_build_forward_select. It holds its slot and no more.
        if (node->op != GK_OP_NONE && !(node->flags & GK_TENSOR_FLAG_COMPUTE)) {
            continue;
        }

        // An empty node is dropped by gk_compute_node anyway, but it has to be
        // dropped here too, ahead of the storage check: zero bytes of result is
        // exactly the case an allocator is entitled to answer with a null
        // pointer, so the check below would call a legal node unallocated. The
        // GPU backends skip empty nodes at the same point and the two have to
        // agree - a graph that runs on one and refuses on the other is the
        // hardest kind of split-placement bug to find.
        if (gk_is_empty(node)) {
            continue;
        }

        // A node whose result nothing wrote storage for cannot be evaluated.
        // This catches a graph handed over before its buffers were assigned,
        // which otherwise shows up as a null dereference deep in a kernel.
        if (node->data == NULL && node->op != GK_OP_NONE) {
            gk_logf("gk: node %d (%s, op %s) has no data assigned\n",
                    i, node->name, gk_op_name(node->op));
            status = GK_STATUS_NO_STORAGE;
            break;
        }

        struct gk_node_job job = {
            .node     = node,
            .scratch  = scratch,
            .row_size = row,
        };

        // One barrier per node. Nodes depend on the ones before them, so the
        // synchronisation is not optional; what it costs is a fixed overhead
        // per node, which is why very small graphs can be faster on one
        // thread than on many.
        if (pool != NULL) {
            gk_pool_run(pool, gk_node_worker, &job);
        } else {
            gk_node_worker(&job, 0, 1);
        }
    }

    free(scratch);
    return status;
}

enum gk_status gk_graph_compute(struct gk_cgraph * graph, int n_threads) {
    if (n_threads == 1) {
        return gk_graph_compute_with_pool(graph, NULL);
    }

    struct gk_pool * pool = gk_pool_create(n_threads);
    if (pool == NULL) {
        return GK_STATUS_ALLOC_FAILED;
    }

    const enum gk_status status = gk_graph_compute_with_pool(graph, pool);

    gk_pool_free(pool);
    return status;
}
