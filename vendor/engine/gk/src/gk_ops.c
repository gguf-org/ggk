// The op builders.
//
// Every function here does the same three things: check that the operands make
// sense, allocate a result tensor of the right shape and type, and record the
// op and its sources. None of them compute anything or touch tensor data -
// that happens later, when a backend walks the finished graph.
//
// Two conventions run through the file:
//
//   * an `_inplace` variant writes its result over its first operand's storage
//     by making the result a view of it. The graph still carries a node, so
//     the dependency is explicit and the scheduler can see it;
//
//   * ops that reshape or reinterpret return a view - the result shares the
//     operand's memory and only ne/nb differ. Those are free at run time and
//     the compute pass skips them entirely.

#include "gk_impl.h"

#include <math.h>

// Result of a view-producing op: shares storage with `a`, carries `a` as its
// source so the dependency survives into the graph.
static struct gk_tensor * gk_view_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, int n_dims, const int64_t * ne, size_t offset) {

    struct gk_tensor * result = gk_new_tensor_view(ctx, a->type, n_dims, ne, a, offset);
    gk_format_name(result, "%s (view)", a->name);

    // The offset is a size_t, so it needs two int32 slots on a 64-bit build -
    // one is not enough and writing it into one overruns the array.
    int32_t params[sizeof(size_t) / sizeof(int32_t)] = { 0 };
    memcpy(params, &offset, sizeof(offset));
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_VIEW;
    result->src[0] = a;

    return result;
}

// --------------------------------------------------------------------------
// elementwise binary
//
// The second operand may be smaller than the first, in which case it is
// broadcast: every dimension of `b` must divide the matching dimension of `a`.
// Broadcasting is resolved in the kernel by taking the source index modulo the
// operand's extent, so nothing is materialised here.
// --------------------------------------------------------------------------

static struct gk_tensor * gk_binary_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
        enum gk_op op, bool inplace) {

    GK_ASSERT(gk_can_repeat(b, a));

    struct gk_tensor * result = inplace
        ? gk_view_tensor(ctx, a)
        : gk_dup_tensor(ctx, a);

    result->op     = op;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_add(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_ADD, false);
}

struct gk_tensor * gk_add_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_ADD, true);
}

struct gk_tensor * gk_sub(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_SUB, false);
}

struct gk_tensor * gk_sub_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_SUB, true);
}

struct gk_tensor * gk_mul(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_MUL, false);
}

struct gk_tensor * gk_mul_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_MUL, true);
}

struct gk_tensor * gk_div(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_DIV, false);
}

struct gk_tensor * gk_div_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_binary_impl(ctx, a, b, GK_OP_DIV, true);
}

// --------------------------------------------------------------------------
// elementwise unary
// --------------------------------------------------------------------------

static struct gk_tensor * gk_unary_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, enum gk_unary_op op, bool inplace) {

    // An activation reads and writes the same positions, so a non-contiguous
    // operand would need the kernel to carry strides it otherwise ignores.
    GK_ASSERT(gk_is_contiguous_1(a));

    struct gk_tensor * result = inplace
        ? gk_view_tensor(ctx, a)
        : gk_dup_tensor(ctx, a);

    gk_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GK_OP_UNARY;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_unary(struct gk_ctx * ctx, struct gk_tensor * a, enum gk_unary_op op) {
    return gk_unary_impl(ctx, a, op, false);
}

struct gk_tensor * gk_unary_inplace(struct gk_ctx * ctx, struct gk_tensor * a, enum gk_unary_op op) {
    return gk_unary_impl(ctx, a, op, true);
}

#define GK_UNARY_BUILDER(name, op) \
    struct gk_tensor * gk_##name(struct gk_ctx * ctx, struct gk_tensor * a) { \
        return gk_unary_impl(ctx, a, op, false); \
    } \
    struct gk_tensor * gk_##name##_inplace(struct gk_ctx * ctx, struct gk_tensor * a) { \
        return gk_unary_impl(ctx, a, op, true); \
    }

GK_UNARY_BUILDER(abs,         GK_UNARY_OP_ABS)
GK_UNARY_BUILDER(sgn,         GK_UNARY_OP_SGN)
GK_UNARY_BUILDER(neg,         GK_UNARY_OP_NEG)
GK_UNARY_BUILDER(step,        GK_UNARY_OP_STEP)
GK_UNARY_BUILDER(tanh,        GK_UNARY_OP_TANH)
GK_UNARY_BUILDER(elu,         GK_UNARY_OP_ELU)
GK_UNARY_BUILDER(relu,        GK_UNARY_OP_RELU)
GK_UNARY_BUILDER(sigmoid,     GK_UNARY_OP_SIGMOID)
GK_UNARY_BUILDER(gelu,        GK_UNARY_OP_GELU)
GK_UNARY_BUILDER(gelu_quick,  GK_UNARY_OP_GELU_QUICK)
GK_UNARY_BUILDER(gelu_erf,    GK_UNARY_OP_GELU_ERF)
GK_UNARY_BUILDER(silu,        GK_UNARY_OP_SILU)
GK_UNARY_BUILDER(hardswish,   GK_UNARY_OP_HARDSWISH)
GK_UNARY_BUILDER(hardsigmoid, GK_UNARY_OP_HARDSIGMOID)
GK_UNARY_BUILDER(exp,         GK_UNARY_OP_EXP)
GK_UNARY_BUILDER(expm1,       GK_UNARY_OP_EXPM1)
GK_UNARY_BUILDER(softplus,    GK_UNARY_OP_SOFTPLUS)
GK_UNARY_BUILDER(floor,       GK_UNARY_OP_FLOOR)
GK_UNARY_BUILDER(ceil,        GK_UNARY_OP_CEIL)
GK_UNARY_BUILDER(round,       GK_UNARY_OP_ROUND)
GK_UNARY_BUILDER(trunc,       GK_UNARY_OP_TRUNC)

// Squares and roots are their own ops rather than unary variants because the
// backward pass and several fused paths want to recognise them by op id.
static struct gk_tensor * gk_simple_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, enum gk_op op, bool inplace) {
    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);
    result->op     = op;
    result->src[0] = a;
    return result;
}

#define GK_SIMPLE_BUILDER(name, op) \
    struct gk_tensor * gk_##name(struct gk_ctx * ctx, struct gk_tensor * a) { \
        return gk_simple_impl(ctx, a, op, false); \
    } \
    struct gk_tensor * gk_##name##_inplace(struct gk_ctx * ctx, struct gk_tensor * a) { \
        return gk_simple_impl(ctx, a, op, true); \
    }

GK_SIMPLE_BUILDER(sqr,  GK_OP_SQR)
GK_SIMPLE_BUILDER(sqrt, GK_OP_SQRT)
GK_SIMPLE_BUILDER(log,  GK_OP_LOG)
GK_SIMPLE_BUILDER(sin,  GK_OP_SIN)
GK_SIMPLE_BUILDER(cos,  GK_OP_COS)

struct gk_tensor * gk_leaky_relu(struct gk_ctx * ctx, struct gk_tensor * a,
                                 float negative_slope, bool inplace) {
    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);

    gk_set_op_params(result, &negative_slope, sizeof(negative_slope));

    result->op     = GK_OP_LEAKY_RELU;
    result->src[0] = a;

    return result;
}

// The four raw constants are folded here, at build time, into the two the
// kernel actually multiplies by - softplus is the constraining function the
// paper runs the alphas through, and there is no reason to re-run it per
// element.
struct gk_tensor * gk_xielu(struct gk_ctx * ctx, struct gk_tensor * a,
                            float alpha_n, float alpha_p, float beta, float eps) {
    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    // softplus overflows exp before it converges to the identity; past 20 the
    // difference from x is below f32 resolution anyway
    const float sp_n = alpha_n > 20.0f ? alpha_n : logf(1.0f + expf(alpha_n));
    const float sp_p = alpha_p > 20.0f ? alpha_p : logf(1.0f + expf(alpha_p));

    gk_set_op_params_i32(result, 0, (int32_t) GK_UNARY_OP_XIELU);
    gk_set_op_params_f32(result, 1, beta + sp_n);
    gk_set_op_params_f32(result, 2, sp_p);
    gk_set_op_params_f32(result, 3, beta);
    gk_set_op_params_f32(result, 4, eps);

    result->op     = GK_OP_UNARY;
    result->src[0] = a;

    return result;
}

// --------------------------------------------------------------------------
// gated linear units
//
// The gate either comes from splitting `a` down the middle, or from a separate
// tensor `b` of matching shape. Splitting is the common case in feed-forward
// blocks where one projection produces both halves.
// --------------------------------------------------------------------------

static struct gk_tensor * gk_glu_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
        enum gk_glu_op op, bool swapped) {

    GK_ASSERT(gk_is_contiguous_1(a));

    if (b != NULL) {
        GK_ASSERT(gk_is_contiguous_1(b));
        GK_ASSERT(gk_are_same_shape(a, b));
        GK_ASSERT(a->type == b->type);
    }

    // without a separate gate, `a` holds both halves along dimension 0
    const int64_t ne0 = b != NULL ? a->ne[0] : a->ne[0] / 2;
    GK_ASSERT(b != NULL || a->ne[0] % 2 == 0);

    const int64_t ne[GK_MAX_DIMS] = { ne0, a->ne[1], a->ne[2], a->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, a->type, GK_MAX_DIMS, ne);

    gk_set_op_params_i32(result, 0, (int32_t) op);
    gk_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GK_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_glu_split(
        struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b, enum gk_glu_op op) {
    return gk_glu_impl(ctx, a, b, op, false);
}

struct gk_tensor * gk_glu(struct gk_ctx * ctx, struct gk_tensor * a, enum gk_glu_op op, bool swapped) {
    return gk_glu_impl(ctx, a, NULL, op, swapped);
}

struct gk_tensor * gk_swiglu_oai(struct gk_ctx * ctx, struct gk_tensor * a,
                                 struct gk_tensor * b, float alpha, float limit) {
    struct gk_tensor * result = gk_glu_impl(ctx, a, b, GK_GLU_OP_SWIGLU_OAI, false);

    gk_set_op_params_f32(result, 2, alpha);
    gk_set_op_params_f32(result, 3, limit);

    return result;
}

// --------------------------------------------------------------------------
// scale and clamp
// --------------------------------------------------------------------------

static struct gk_tensor * gk_scale_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, float s, float bias, bool inplace) {

    GK_ASSERT(gk_is_padded_1(a));

    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);

    float params[2] = { s, bias };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_SCALE;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_scale(struct gk_ctx * ctx, struct gk_tensor * a, float s) {
    return gk_scale_impl(ctx, a, s, 0.0f, false);
}

struct gk_tensor * gk_scale_inplace(struct gk_ctx * ctx, struct gk_tensor * a, float s) {
    return gk_scale_impl(ctx, a, s, 0.0f, true);
}

struct gk_tensor * gk_scale_bias(struct gk_ctx * ctx, struct gk_tensor * a, float s, float b) {
    return gk_scale_impl(ctx, a, s, b, false);
}

struct gk_tensor * gk_clamp(struct gk_ctx * ctx, struct gk_tensor * a, float min, float max) {
    // clamp always writes through the operand's storage
    struct gk_tensor * result = gk_view_tensor(ctx, a);

    float params[2] = { min, max };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_CLAMP;
    result->src[0] = a;

    return result;
}

// --------------------------------------------------------------------------
// normalisation
// --------------------------------------------------------------------------

static struct gk_tensor * gk_norm_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, float eps, enum gk_op op, bool inplace) {

    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);

    gk_set_op_params(result, &eps, sizeof(eps));

    result->op     = op;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_norm(struct gk_ctx * ctx, struct gk_tensor * a, float eps) {
    return gk_norm_impl(ctx, a, eps, GK_OP_NORM, false);
}

struct gk_tensor * gk_norm_inplace(struct gk_ctx * ctx, struct gk_tensor * a, float eps) {
    return gk_norm_impl(ctx, a, eps, GK_OP_NORM, true);
}

struct gk_tensor * gk_rms_norm(struct gk_ctx * ctx, struct gk_tensor * a, float eps) {
    return gk_norm_impl(ctx, a, eps, GK_OP_RMS_NORM, false);
}

struct gk_tensor * gk_rms_norm_inplace(struct gk_ctx * ctx, struct gk_tensor * a, float eps) {
    return gk_norm_impl(ctx, a, eps, GK_OP_RMS_NORM, true);
}

struct gk_tensor * gk_l2_norm(struct gk_ctx * ctx, struct gk_tensor * a, float eps) {
    return gk_norm_impl(ctx, a, eps, GK_OP_L2_NORM, false);
}

struct gk_tensor * gk_group_norm(
        struct gk_ctx * ctx, struct gk_tensor * a, int n_groups, float eps) {

    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    gk_set_op_params_i32(result, 0, n_groups);
    gk_set_op_params_f32(result, 1, eps);

    result->op     = GK_OP_GROUP_NORM;
    result->src[0] = a;

    return result;
}

// --------------------------------------------------------------------------
// matrix multiply
//
// Shapes follow the GGUF convention, which is the transpose of the usual
// mathematical one: `a` is the weight, stored with the reduction dimension
// first, and the result is [a->ne[1], b->ne[1], ...]. Written out, the kernel
// computes for every pair of rows
//
//     dst[i1, i0] = dot(a_row[i0], b_row[i1])
//
// so both operands are read along their fastest dimension, which is what makes
// a quantized weight usable without transposing it first.
//
// The higher dimensions broadcast: `b` may carry more heads or batches than
// `a`, and each of `a`'s is reused across the matching group.
// --------------------------------------------------------------------------

static bool gk_can_mul_mat(const struct gk_tensor * a, const struct gk_tensor * b) {
    return a->ne[0] == b->ne[0]
        && b->ne[2] % a->ne[2] == 0
        && b->ne[3] % a->ne[3] == 0;
}

struct gk_tensor * gk_mul_mat(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    GK_ASSERT(gk_can_mul_mat(a, b));
    GK_ASSERT(!gk_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    result->op     = GK_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// Mixture-of-experts: `ids` selects, per token, which of `as`'s experts to
// apply. `as` carries the experts along dimension 2.
struct gk_tensor * gk_mul_mat_id(
        struct gk_ctx * ctx, struct gk_tensor * as, struct gk_tensor * b, struct gk_tensor * ids) {

    GK_ASSERT(!gk_is_transposed(as));
    GK_ASSERT(ids->type == GK_TYPE_I32);
    GK_ASSERT(as->ne[3] == 1);       // experts live in dim 2, so dim 3 is unused
    GK_ASSERT(b->ne[3] == 1);
    GK_ASSERT(as->ne[0] == b->ne[0]); // shared reduction dimension
    GK_ASSERT(ids->ne[1] == b->ne[2]);

    const int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    result->op     = GK_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    return result;
}

struct gk_tensor * gk_out_prod(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    GK_ASSERT(!gk_is_transposed(a));

    const int64_t ne[4] = { a->ne[0], b->ne[0], a->ne[2], b->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    result->op     = GK_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// --------------------------------------------------------------------------
// shape ops
//
// All of these return views. Reshape requires a contiguous operand because it
// reinterprets the element order; permute and transpose only reorder the
// ne/nb pairs and work on anything.
// --------------------------------------------------------------------------

struct gk_tensor * gk_reshape(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    GK_ASSERT(gk_is_contiguous(a));
    GK_ASSERT(gk_nelements(a) == gk_nelements(b));

    struct gk_tensor * result = gk_new_tensor_view(ctx, a->type, GK_MAX_DIMS, b->ne, a, 0);
    gk_format_name(result, "%s (reshaped)", a->name);

    result->op     = GK_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

static struct gk_tensor * gk_reshape_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, int n_dims, const int64_t * ne) {

    GK_ASSERT(gk_is_contiguous(a));

    int64_t n = 1;
    for (int i = 0; i < n_dims; ++i) {
        n *= ne[i];
    }
    GK_ASSERT(gk_nelements(a) == n);

    struct gk_tensor * result = gk_new_tensor_view(ctx, a->type, n_dims, ne, a, 0);
    gk_format_name(result, "%s (reshaped)", a->name);

    result->op     = GK_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_reshape_1d(struct gk_ctx * ctx, struct gk_tensor * a, int64_t ne0) {
    return gk_reshape_impl(ctx, a, 1, &ne0);
}

struct gk_tensor * gk_reshape_2d(struct gk_ctx * ctx, struct gk_tensor * a,
                                 int64_t ne0, int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    return gk_reshape_impl(ctx, a, 2, ne);
}

struct gk_tensor * gk_reshape_3d(struct gk_ctx * ctx, struct gk_tensor * a,
                                 int64_t ne0, int64_t ne1, int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    return gk_reshape_impl(ctx, a, 3, ne);
}

struct gk_tensor * gk_reshape_4d(struct gk_ctx * ctx, struct gk_tensor * a,
                                 int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return gk_reshape_impl(ctx, a, 4, ne);
}

struct gk_tensor * gk_view_1d(struct gk_ctx * ctx, struct gk_tensor * a,
                              int64_t ne0, size_t offset) {
    return gk_view_impl(ctx, a, 1, &ne0, offset);
}

struct gk_tensor * gk_view_2d(struct gk_ctx * ctx, struct gk_tensor * a,
                              int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
    const int64_t ne[2] = { ne0, ne1 };
    struct gk_tensor * result = gk_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1] * ne1;
    result->nb[3] = result->nb[2];

    return result;
}

struct gk_tensor * gk_view_3d(struct gk_ctx * ctx, struct gk_tensor * a,
                              int64_t ne0, int64_t ne1, int64_t ne2,
                              size_t nb1, size_t nb2, size_t offset) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    struct gk_tensor * result = gk_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2] * ne2;

    return result;
}

struct gk_tensor * gk_view_4d(struct gk_ctx * ctx, struct gk_tensor * a,
                              int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                              size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    struct gk_tensor * result = gk_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    return result;
}

// Reorders dimensions by rewriting ne/nb. `axisN` says where dimension N of
// the operand ends up in the result.
struct gk_tensor * gk_permute(struct gk_ctx * ctx, struct gk_tensor * a,
                              int axis0, int axis1, int axis2, int axis3) {
    GK_ASSERT(axis0 >= 0 && axis0 < GK_MAX_DIMS);
    GK_ASSERT(axis1 >= 0 && axis1 < GK_MAX_DIMS);
    GK_ASSERT(axis2 >= 0 && axis2 < GK_MAX_DIMS);
    GK_ASSERT(axis3 >= 0 && axis3 < GK_MAX_DIMS);

    // a permutation, not just any mapping
    GK_ASSERT(axis0 != axis1 && axis0 != axis2 && axis0 != axis3);
    GK_ASSERT(axis1 != axis2 && axis1 != axis3);
    GK_ASSERT(axis2 != axis3);

    struct gk_tensor * result = gk_view_tensor(ctx, a);
    gk_format_name(result, "%s (permuted)", a->name);

    int64_t ne[GK_MAX_DIMS];
    size_t  nb[GK_MAX_DIMS];

    ne[axis0] = a->ne[0]; ne[axis1] = a->ne[1]; ne[axis2] = a->ne[2]; ne[axis3] = a->ne[3];
    nb[axis0] = a->nb[0]; nb[axis1] = a->nb[1]; nb[axis2] = a->nb[2]; nb[axis3] = a->nb[3];

    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        result->ne[i] = ne[i];
        result->nb[i] = nb[i];
    }

    result->op     = GK_OP_PERMUTE;
    result->src[0] = a;

    gk_set_op_params_i32(result, 0, axis0);
    gk_set_op_params_i32(result, 1, axis1);
    gk_set_op_params_i32(result, 2, axis2);
    gk_set_op_params_i32(result, 3, axis3);

    return result;
}

struct gk_tensor * gk_transpose(struct gk_ctx * ctx, struct gk_tensor * a) {
    struct gk_tensor * result = gk_view_tensor(ctx, a);
    gk_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];
    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GK_OP_TRANSPOSE;
    result->src[0] = a;

    return result;
}

// Materialises a contiguous copy. This is the op that pays for a permute: up
// to here nothing has moved, and `cont` is where it does.
static struct gk_tensor * gk_cont_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, int n_dims, const int64_t * ne) {

    struct gk_tensor * result = gk_new_tensor(ctx, a->type, n_dims, ne);
    gk_format_name(result, "%s (cont)", a->name);

    result->op     = GK_OP_CONT;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_cont(struct gk_ctx * ctx, struct gk_tensor * a) {
    return gk_cont_impl(ctx, a, GK_MAX_DIMS, a->ne);
}

struct gk_tensor * gk_cont_2d(struct gk_ctx * ctx, struct gk_tensor * a,
                              int64_t ne0, int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    GK_ASSERT(gk_nelements(a) == ne0 * ne1);
    return gk_cont_impl(ctx, a, 2, ne);
}

struct gk_tensor * gk_cont_3d(struct gk_ctx * ctx, struct gk_tensor * a,
                              int64_t ne0, int64_t ne1, int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    GK_ASSERT(gk_nelements(a) == ne0 * ne1 * ne2);
    return gk_cont_impl(ctx, a, 3, ne);
}

struct gk_tensor * gk_cont_4d(struct gk_ctx * ctx, struct gk_tensor * a,
                              int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    GK_ASSERT(gk_nelements(a) == ne0 * ne1 * ne2 * ne3);
    return gk_cont_impl(ctx, a, 4, ne);
}

// Copies `a` into `b`'s storage and type, converting if they differ. Returns a
// view of `b`, so the result is whatever `b` addresses.
struct gk_tensor * gk_cpy(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    GK_ASSERT(gk_nelements(a) == gk_nelements(b));

    struct gk_tensor * result = gk_view_tensor(ctx, b);
    if (b->name[0] != '\0') {
        gk_format_name(result, "%s (copy of %s)", b->name, a->name);
    } else {
        gk_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GK_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_cast(struct gk_ctx * ctx, struct gk_tensor * a, enum gk_type type) {
    struct gk_tensor * result = gk_new_tensor(ctx, type, GK_MAX_DIMS, a->ne);
    gk_format_name(result, "%s (cast)", a->name);

    result->op     = GK_OP_CPY;
    result->src[0] = a;
    result->src[1] = result;

    return result;
}

struct gk_tensor * gk_dup(struct gk_ctx * ctx, struct gk_tensor * a) {
    struct gk_tensor * result = gk_dup_tensor(ctx, a);
    result->op     = GK_OP_DUP;
    result->src[0] = a;
    return result;
}

// --------------------------------------------------------------------------
// gather and broadcast
// --------------------------------------------------------------------------

// Row lookup - the embedding op. `b` holds row indices; the result collects
// those rows of `a`. A quantized `a` is dequantized on the way out, because an
// embedding row feeds arithmetic immediately.
struct gk_tensor * gk_get_rows(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    GK_ASSERT(a->ne[2] == b->ne[1]);
    GK_ASSERT(b->ne[3] == 1);
    GK_ASSERT(b->type == GK_TYPE_I32);

    const enum gk_type type =
        (a->type == GK_TYPE_I32) ? GK_TYPE_I32 : GK_TYPE_F32;

    const int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[1], b->ne[2] };
    struct gk_tensor * result = gk_new_tensor(ctx, type, 4, ne);

    result->op     = GK_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_repeat(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    GK_ASSERT(gk_can_repeat(a, b));

    struct gk_tensor * result = gk_new_tensor(ctx, a->type, GK_MAX_DIMS, b->ne);

    result->op     = GK_OP_REPEAT;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_repeat_4d(struct gk_ctx * ctx, struct gk_tensor * a,
                                int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const bool can_repeat = gk_is_empty(a) ||
        (ne0 % a->ne[0] == 0 && ne1 % a->ne[1] == 0 &&
         ne2 % a->ne[2] == 0 && ne3 % a->ne[3] == 0);
    GK_ASSERT(can_repeat);

    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    struct gk_tensor * result = gk_new_tensor(ctx, a->type, 4, ne);

    result->op     = GK_OP_REPEAT;
    result->src[0] = a;

    return result;
}

// Joins two tensors along `dim`; every other dimension must match.
struct gk_tensor * gk_concat(struct gk_ctx * ctx, struct gk_tensor * a,
                             struct gk_tensor * b, int dim) {
    GK_ASSERT(dim >= 0 && dim < GK_MAX_DIMS);

    int64_t ne[GK_MAX_DIMS];
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        if (i == dim) {
            ne[i] = a->ne[i] + b->ne[i];
        } else {
            GK_ASSERT(a->ne[i] == b->ne[i]);
            ne[i] = a->ne[i];
        }
    }

    struct gk_tensor * result = gk_new_tensor(ctx, a->type, GK_MAX_DIMS, ne);

    gk_set_op_params_i32(result, 0, dim);

    result->op     = GK_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// --------------------------------------------------------------------------
// attention pieces
// --------------------------------------------------------------------------

// Softmax over rows, with the two things attention always wants folded in: a
// scale applied before the exponential, and an additive mask. `mask` may be
// f16 to halve the bandwidth of a large attention bias.
struct gk_tensor * gk_soft_max_ext(
        struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * mask,
        float scale, float max_bias) {

    GK_ASSERT(gk_is_contiguous(a));

    if (mask != NULL) {
        GK_ASSERT(mask->type == GK_TYPE_F16 || mask->type == GK_TYPE_F32);
        GK_ASSERT(gk_is_contiguous(mask));
        GK_ASSERT(mask->ne[0] == a->ne[0]);
        GK_ASSERT(mask->ne[1] >= a->ne[1]);
    }

    if (max_bias > 0.0f) {
        GK_ASSERT(mask != NULL);
    }

    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    float params[2] = { scale, max_bias };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    return result;
}

struct gk_tensor * gk_soft_max(struct gk_ctx * ctx, struct gk_tensor * a) {
    return gk_soft_max_ext(ctx, a, NULL, 1.0f, 0.0f);
}

void gk_soft_max_add_sinks(struct gk_tensor * a, struct gk_tensor * sinks) {
    if (sinks == NULL) {
        a->src[2] = NULL;
        return;
    }

    GK_ASSERT(a->op == GK_OP_SOFT_MAX);
    GK_ASSERT(a->src[2] == NULL);
    GK_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GK_ASSERT(sinks->type == GK_TYPE_F32);

    a->src[2] = sinks;
}

// --------------------------------------------------------------------------
// fused attention
// --------------------------------------------------------------------------

struct gk_tensor * gk_flash_attn_ext(struct gk_ctx * ctx,
                                     struct gk_tensor * q, struct gk_tensor * k,
                                     struct gk_tensor * v, struct gk_tensor * mask,
                                     float scale, float max_bias, float logit_softcap) {
    GK_ASSERT(gk_can_mul_mat(k, q));
    GK_ASSERT(q->ne[3] == k->ne[3]);
    GK_ASSERT(q->ne[3] == v->ne[3]);
    GK_ASSERT(k->ne[1] == v->ne[1]); // one V row per K row

    if (mask != NULL) {
        GK_ASSERT(mask->type == GK_TYPE_F16);
        GK_ASSERT(gk_is_contiguous(mask));
        GK_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GK_ASSERT(q->ne[3] % mask->ne[3] == 0);
        GK_ASSERT(mask->ne[1] >= q->ne[1]);
    }

    if (max_bias > 0.0f) {
        GK_ASSERT(mask != NULL);
    }

    // the head and batch dimensions come out swapped: [DV, n_head, n_batch, ne3]
    const int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    float params[3] = { scale, max_bias, logit_softcap };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    return result;
}

void gk_flash_attn_ext_set_prec(struct gk_tensor * a, enum gk_prec prec) {
    GK_ASSERT(a->op == GK_OP_FLASH_ATTN_EXT);
    gk_set_op_params_i32(a, 3, (int32_t) prec); // slots 0-2 hold the float params
}

enum gk_prec gk_flash_attn_ext_get_prec(const struct gk_tensor * a) {
    GK_ASSERT(a->op == GK_OP_FLASH_ATTN_EXT);
    return (enum gk_prec) gk_get_op_params_i32(a, 3);
}

void gk_flash_attn_ext_add_sinks(struct gk_tensor * a, struct gk_tensor * sinks) {
    if (sinks == NULL) {
        a->src[4] = NULL;
        return;
    }

    GK_ASSERT(a->op == GK_OP_FLASH_ATTN_EXT);
    GK_ASSERT(a->src[4] == NULL);
    GK_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GK_ASSERT(sinks->type == GK_TYPE_F32);

    a->src[4] = sinks;
}

static struct gk_tensor * gk_diag_mask_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, int n_past, enum gk_op op, bool inplace) {

    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);

    gk_set_op_params_i32(result, 0, n_past);

    result->op     = op;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_diag_mask_inf(struct gk_ctx * ctx, struct gk_tensor * a, int n_past) {
    return gk_diag_mask_impl(ctx, a, n_past, GK_OP_DIAG_MASK_INF, false);
}

struct gk_tensor * gk_diag_mask_inf_inplace(struct gk_ctx * ctx, struct gk_tensor * a, int n_past) {
    return gk_diag_mask_impl(ctx, a, n_past, GK_OP_DIAG_MASK_INF, true);
}

struct gk_tensor * gk_diag_mask_zero(struct gk_ctx * ctx, struct gk_tensor * a, int n_past) {
    return gk_diag_mask_impl(ctx, a, n_past, GK_OP_DIAG_MASK_ZERO, false);
}

// --------------------------------------------------------------------------
// rotary position embedding
//
// Rotates pairs of channels by an angle that depends on the position and on
// the channel's frequency. The parameter block is wide because every variant
// in circulation - NeoX pair ordering, multi-axis vision ropes, YaRN scaling,
// long-context extension factors - is the same rotation with different
// bookkeeping, and they are best kept in one op.
// --------------------------------------------------------------------------

static struct gk_tensor * gk_rope_impl(
        struct gk_ctx * ctx,
        struct gk_tensor * a, struct gk_tensor * b, struct gk_tensor * c,
        int n_dims, const int * sections, int mode, int n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {

    GK_ASSERT(b->type == GK_TYPE_I32);

    // a multi-axis rope reads four position ids per token
    if (mode & GK_ROPE_TYPE_MROPE) {
        GK_ASSERT(a->ne[2] * 4 == b->ne[0]);
    } else {
        GK_ASSERT(a->ne[2] == b->ne[0]);
    }

    if (c != NULL) {
        GK_ASSERT(c->type == GK_TYPE_F32);
        GK_ASSERT(c->ne[0] >= n_dims / 2);
    }

    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    int32_t params[15] = { 0 };
    params[1] = n_dims;
    params[2] = mode;
    params[4] = n_ctx_orig;
    memcpy(params +  5, &freq_base,   sizeof(float));
    memcpy(params +  6, &freq_scale,  sizeof(float));
    memcpy(params +  7, &ext_factor,  sizeof(float));
    memcpy(params +  8, &attn_factor, sizeof(float));
    memcpy(params +  9, &beta_fast,   sizeof(float));
    memcpy(params + 10, &beta_slow,   sizeof(float));
    if ((mode & GK_ROPE_TYPE_MROPE) && sections != NULL) {
        memcpy(params + 11, sections, sizeof(int32_t) * GK_MROPE_SECTIONS);
    }
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct gk_tensor * gk_rope_ext(
        struct gk_ctx * ctx,
        struct gk_tensor * a, struct gk_tensor * b, struct gk_tensor * c,
        int n_dims, int mode, int n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    return gk_rope_impl(ctx, a, b, c, n_dims, NULL, mode, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

struct gk_tensor * gk_rope_multi(
        struct gk_ctx * ctx,
        struct gk_tensor * a, struct gk_tensor * b, struct gk_tensor * c,
        int n_dims, int sections[GK_MROPE_SECTIONS], int mode, int n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    return gk_rope_impl(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

struct gk_tensor * gk_rope(struct gk_ctx * ctx, struct gk_tensor * a,
                           struct gk_tensor * b, int n_dims, int mode) {
    return gk_rope_ext(ctx, a, b, NULL, n_dims, mode, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

// --------------------------------------------------------------------------
// reductions and ordering
// --------------------------------------------------------------------------

struct gk_tensor * gk_sum(struct gk_ctx * ctx, struct gk_tensor * a) {
    struct gk_tensor * result = gk_new_tensor_1d(ctx, a->type, 1);
    result->op     = GK_OP_SUM;
    result->src[0] = a;
    return result;
}

struct gk_tensor * gk_sum_rows(struct gk_ctx * ctx, struct gk_tensor * a) {
    const int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, a->type, 4, ne);
    result->op     = GK_OP_SUM_ROWS;
    result->src[0] = a;
    return result;
}

struct gk_tensor * gk_mean(struct gk_ctx * ctx, struct gk_tensor * a) {
    const int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);
    result->op     = GK_OP_MEAN;
    result->src[0] = a;
    return result;
}

struct gk_tensor * gk_argmax(struct gk_ctx * ctx, struct gk_tensor * a) {
    GK_ASSERT(gk_is_matrix(a));
    GK_ASSERT(a->ne[0] <= INT32_MAX);

    struct gk_tensor * result = gk_new_tensor_1d(ctx, GK_TYPE_I32, a->ne[1]);
    result->op     = GK_OP_ARGMAX;
    result->src[0] = a;
    return result;
}

struct gk_tensor * gk_argsort(struct gk_ctx * ctx, struct gk_tensor * a, enum gk_sort_order order) {
    GK_ASSERT(a->ne[0] <= INT32_MAX);

    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_I32, GK_MAX_DIMS, a->ne);

    gk_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GK_OP_ARGSORT;
    result->src[0] = a;

    return result;
}

// A real kernel: a partial selection, so a vocabulary-sized row does not pay
// for a full sort. The indices come back in no promised order.
struct gk_tensor * gk_top_k(struct gk_ctx * ctx, struct gk_tensor * a, int k) {
    GK_ASSERT(a->ne[0] >= k);

    struct gk_tensor * result = gk_new_tensor_4d(ctx, GK_TYPE_I32,
                                                 k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GK_OP_TOP_K;
    result->src[0] = a;

    return result;
}

// The deterministic form: a full descending sort, viewed down to k.
struct gk_tensor * gk_argsort_top_k(struct gk_ctx * ctx, struct gk_tensor * a, int k) {
    GK_ASSERT(a->ne[0] >= k);

    struct gk_tensor * result = gk_argsort(ctx, a, GK_SORT_ORDER_DESC);
    return gk_view_4d(ctx, result, k, result->ne[1], result->ne[2], result->ne[3],
                      result->nb[1], result->nb[2], result->nb[3], 0);
}

struct gk_tensor * gk_cumsum(struct gk_ctx * ctx, struct gk_tensor * a) {
    GK_ASSERT(a->type == GK_TYPE_F32);

    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    result->op     = GK_OP_CUMSUM;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_count_equal(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    GK_ASSERT(gk_are_same_shape(a, b));

    struct gk_tensor * result = gk_new_tensor_1d(ctx, GK_TYPE_I64, 1);

    result->op     = GK_OP_COUNT_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// --------------------------------------------------------------------------
// data placement
// --------------------------------------------------------------------------

struct gk_tensor * gk_add_id(struct gk_ctx * ctx, struct gk_tensor * a,
                             struct gk_tensor * b, struct gk_tensor * ids) {
    GK_ASSERT(a->ne[0] == b->ne[0]);
    GK_ASSERT(a->ne[1] == ids->ne[0]);
    GK_ASSERT(a->ne[2] == ids->ne[1]);
    GK_ASSERT(ids->type == GK_TYPE_I32);

    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    result->op     = GK_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    return result;
}

// acc and set share a parameter block: the window's strides and offset, plus
// whether the result may write over `a` or has to copy it first.
static struct gk_tensor * gk_acc_set_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
        size_t nb1, size_t nb2, size_t nb3, size_t offset,
        enum gk_op op, bool inplace) {

    GK_ASSERT(gk_nelements(b) <= gk_nelements(a));
    GK_ASSERT(gk_is_contiguous(a));
    GK_ASSERT(a->type == GK_TYPE_F32);
    GK_ASSERT(b->type == GK_TYPE_F32);

    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);

    int32_t params[] = { (int32_t) nb1, (int32_t) nb2, (int32_t) nb3,
                         (int32_t) offset, inplace ? 1 : 0 };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = op;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_acc(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                          size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return gk_acc_set_impl(ctx, a, b, nb1, nb2, nb3, offset, GK_OP_ACC, false);
}

struct gk_tensor * gk_acc_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                  size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return gk_acc_set_impl(ctx, a, b, nb1, nb2, nb3, offset, GK_OP_ACC, true);
}

struct gk_tensor * gk_set(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                          size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return gk_acc_set_impl(ctx, a, b, nb1, nb2, nb3, offset, GK_OP_SET, false);
}

struct gk_tensor * gk_set_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                  size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    return gk_acc_set_impl(ctx, a, b, nb1, nb2, nb3, offset, GK_OP_SET, true);
}

struct gk_tensor * gk_set_1d(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                             size_t offset) {
    return gk_acc_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, GK_OP_SET, false);
}

struct gk_tensor * gk_set_1d_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                     size_t offset) {
    return gk_acc_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, GK_OP_SET, true);
}

struct gk_tensor * gk_set_rows(struct gk_ctx * ctx, struct gk_tensor * a,
                               struct gk_tensor * b, struct gk_tensor * c) {
    GK_ASSERT(a->ne[0] == b->ne[0]);
    GK_ASSERT(a->ne[2] == b->ne[2]);
    GK_ASSERT(a->ne[3] == b->ne[3]);
    GK_ASSERT(b->ne[1] == c->ne[0]);
    GK_ASSERT(b->ne[2] % c->ne[1] == 0);
    GK_ASSERT(b->ne[3] % c->ne[2] == 0);
    GK_ASSERT(c->ne[3] == 1);
    GK_ASSERT(b->type == GK_TYPE_F32 || b->type == GK_TYPE_F16);
    GK_ASSERT(c->type == GK_TYPE_I64 || c->type == GK_TYPE_I32);

    struct gk_tensor * result = gk_view_tensor(ctx, a);

    result->op     = GK_OP_SET_ROWS;
    // The destination rides in src[2], the payload in src[0]. Backwards, but
    // it is the order the rest of the engine was built against and the graph
    // machinery does not care which slot the view parent sits in.
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a;

    return result;
}

struct gk_tensor * gk_diag(struct gk_ctx * ctx, struct gk_tensor * a) {
    GK_ASSERT(a->ne[1] == 1);

    const int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, a->type, 4, ne);

    result->op     = GK_OP_DIAG;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_tri(struct gk_ctx * ctx, struct gk_tensor * a, enum gk_tri_type type) {
    GK_ASSERT(a->type == GK_TYPE_F32);
    GK_ASSERT(gk_is_contiguous(a));
    GK_ASSERT(a->ne[0] == a->ne[1]);

    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    gk_set_op_params_i32(result, 0, (int32_t) type);

    result->op     = GK_OP_TRI;
    result->src[0] = a;

    return result;
}

static struct gk_tensor * gk_fill_impl(struct gk_ctx * ctx, struct gk_tensor * a,
                                       float c, bool inplace) {
    GK_ASSERT(a->type == GK_TYPE_F32 || a->type == GK_TYPE_F16);
    GK_ASSERT(gk_is_contiguous(a));

    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);

    gk_set_op_params_f32(result, 0, c);

    result->op     = GK_OP_FILL;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_fill(struct gk_ctx * ctx, struct gk_tensor * a, float c) {
    return gk_fill_impl(ctx, a, c, false);
}

struct gk_tensor * gk_fill_inplace(struct gk_ctx * ctx, struct gk_tensor * a, float c) {
    return gk_fill_impl(ctx, a, c, true);
}

struct gk_tensor * gk_arange(struct gk_ctx * ctx, float start, float stop, float step) {
    GK_ASSERT(stop > start);

    const int64_t steps = (int64_t) ceilf((stop - start) / step);

    struct gk_tensor * result = gk_new_tensor_1d(ctx, GK_TYPE_F32, steps);

    gk_set_op_params_f32(result, 0, start);
    gk_set_op_params_f32(result, 1, stop);
    gk_set_op_params_f32(result, 2, step);

    result->op = GK_OP_ARANGE;

    return result;
}

struct gk_tensor * gk_roll(struct gk_ctx * ctx, struct gk_tensor * a,
                           int shift0, int shift1, int shift2, int shift3) {
    GK_ASSERT(a->nb[0] == gk_type_size(a->type));
    GK_ASSERT(abs(shift0) < a->ne[0]);
    GK_ASSERT(abs(shift1) < a->ne[1]);
    GK_ASSERT(abs(shift2) < a->ne[2]);
    GK_ASSERT(abs(shift3) < a->ne[3]);

    struct gk_tensor * result = gk_dup_tensor(ctx, a);

    gk_set_op_params_i32(result, 0, shift0);
    gk_set_op_params_i32(result, 1, shift1);
    gk_set_op_params_i32(result, 2, shift2);
    gk_set_op_params_i32(result, 3, shift3);

    result->op     = GK_OP_ROLL;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_timestep_embedding(struct gk_ctx * ctx, struct gk_tensor * timesteps,
                                         int dim, int max_period) {
    struct gk_tensor * result = gk_new_tensor_2d(ctx, GK_TYPE_F32, dim, timesteps->ne[0]);

    gk_set_op_params_i32(result, 0, dim);
    gk_set_op_params_i32(result, 1, max_period);

    result->op     = GK_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    return result;
}

// --------------------------------------------------------------------------
// padding and resampling
// --------------------------------------------------------------------------

static struct gk_tensor * gk_pad_impl(
        struct gk_ctx * ctx, struct gk_tensor * a,
        int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3,
        bool circular) {

    struct gk_tensor * result = gk_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    gk_set_op_params_i32(result, 0, lp0);
    gk_set_op_params_i32(result, 1, rp0);
    gk_set_op_params_i32(result, 2, lp1);
    gk_set_op_params_i32(result, 3, rp1);
    gk_set_op_params_i32(result, 4, lp2);
    gk_set_op_params_i32(result, 5, rp2);
    gk_set_op_params_i32(result, 6, lp3);
    gk_set_op_params_i32(result, 7, rp3);
    gk_set_op_params_i32(result, 8, circular ? 1 : 0);

    result->op     = GK_OP_PAD;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_pad(struct gk_ctx * ctx, struct gk_tensor * a,
                          int p0, int p1, int p2, int p3) {
    return gk_pad_impl(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3, false);
}

struct gk_tensor * gk_pad_ext(struct gk_ctx * ctx, struct gk_tensor * a,
                              int lp0, int rp0, int lp1, int rp1,
                              int lp2, int rp2, int lp3, int rp3) {
    return gk_pad_impl(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, false);
}

struct gk_tensor * gk_pad_circular(struct gk_ctx * ctx, struct gk_tensor * a,
                                   int p0, int p1, int p2, int p3) {
    return gk_pad_impl(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3, true);
}

struct gk_tensor * gk_pad_ext_circular(struct gk_ctx * ctx, struct gk_tensor * a,
                                       int lp0, int rp0, int lp1, int rp1,
                                       int lp2, int rp2, int lp3, int rp3) {
    return gk_pad_impl(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, true);
}

struct gk_tensor * gk_pad_reflect_1d(struct gk_ctx * ctx, struct gk_tensor * a, int p0, int p1) {
    GK_ASSERT(p0 >= 0 && p1 >= 0);
    GK_ASSERT(p0 < a->ne[0]);
    GK_ASSERT(p1 < a->ne[0]);
    GK_ASSERT(a->type == GK_TYPE_F32);

    struct gk_tensor * result = gk_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1, a->ne[1], a->ne[2], a->ne[3]);

    gk_set_op_params_i32(result, 0, p0);
    gk_set_op_params_i32(result, 1, p1);

    result->op     = GK_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    return result;
}

static struct gk_tensor * gk_interpolate_impl(
        struct gk_ctx * ctx, struct gk_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode) {

    GK_ASSERT((mode & 0xFF) < GK_SCALE_MODE_COUNT);
    // antialias is only defined for the bilinear filter
    GK_ASSERT(!(mode & GK_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GK_SCALE_MODE_BILINEAR);
    GK_ASSERT(a->type == GK_TYPE_F32);

    struct gk_tensor * result = gk_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    gk_set_op_params_i32(result, 0, (int32_t) mode);

    result->op     = GK_OP_UPSCALE;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_upscale(struct gk_ctx * ctx, struct gk_tensor * a,
                              int scale_factor, enum gk_scale_mode mode) {
    GK_ASSERT(scale_factor > 1);
    return gk_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor,
                               a->ne[2], a->ne[3], mode);
}

struct gk_tensor * gk_interpolate(struct gk_ctx * ctx, struct gk_tensor * a,
                                  int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                  uint32_t mode) {
    return gk_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// --------------------------------------------------------------------------
// convolution
// --------------------------------------------------------------------------

static int64_t gk_conv_out_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

static int64_t gk_conv_transpose_out_size(int64_t ins, int64_t ks, int s, int p) {
    return (ins - 1) * s - 2 * p + ks;
}

static int64_t gk_pool_out_size(int64_t ins, int ks, int s, float p) {
    return (int64_t) ((ins + 2 * p - ks) / s) + 1;
}

struct gk_tensor * gk_im2col(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                             int s0, int s1, int p0, int p1, int d0, int d1,
                             bool is_2D, enum gk_type dst_type) {
    if (is_2D) {
        GK_ASSERT(a->ne[2] == b->ne[2]);
    } else {
        GK_ASSERT(b->ne[1] == a->ne[1]);
        GK_ASSERT(b->ne[3] == 1);
    }

    const int64_t OH = is_2D ? gk_conv_out_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    const int64_t OW =         gk_conv_out_size(b->ne[0], a->ne[0], s0, p0, d0);

    GK_ASSERT((!is_2D || OH > 0) && OW > 0);

    const int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ? b->ne[3] : 1,
    };

    struct gk_tensor * result = gk_new_tensor(ctx, dst_type, 4, ne);

    int32_t params[] = { s0, s1, p0, p1, d0, d1, is_2D ? 1 : 0 };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_im2col_3d(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                int64_t IC, int s0, int s1, int s2, int p0, int p1, int p2,
                                int d0, int d1, int d2, enum gk_type dst_type) {
    const int64_t N  = b->ne[3] / IC;
    const int64_t OD = gk_conv_out_size(b->ne[2], a->ne[2], s2, p2, d2);
    const int64_t OH = gk_conv_out_size(b->ne[1], a->ne[1], s1, p1, d1);
    const int64_t OW = gk_conv_out_size(b->ne[0], a->ne[0], s0, p0, d0);

    GK_ASSERT(OD > 0 && OH > 0 && OW > 0);

    const int64_t ne[4] = { a->ne[0] * a->ne[1] * a->ne[2] * IC, OW, OH, OD * N };

    struct gk_tensor * result = gk_new_tensor(ctx, dst_type, 4, ne);

    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t) IC };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// The composites. Each one is im2col followed by a matmul, with the reshapes
// that line the patch rows up against the flattened kernel. The intermediate
// is f16 unless the kernel is bf16 - halving that buffer is nearly free
// because the matmul dots in f16 anyway.
struct gk_tensor * gk_conv_1d(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                              int s0, int p0, int d0) {
    struct gk_tensor * im2col = gk_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false,
            a->type == GK_TYPE_BF16 ? GK_TYPE_F32 : GK_TYPE_F16);        // [N, OL, IC*K]

    struct gk_tensor * result =
        gk_mul_mat(ctx,
                gk_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
                gk_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2]));

    return gk_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]); // [N, OC, OL]
}

struct gk_tensor * gk_conv_1d_ph(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                 int s, int d) {
    return gk_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

struct gk_tensor * gk_conv_1d_dw(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                 int s0, int p0, int d0) {
    struct gk_tensor * new_b = gk_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    struct gk_tensor * im2col = gk_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false,
            a->type == GK_TYPE_BF16 ? GK_TYPE_F32 : GK_TYPE_F16);

    struct gk_tensor * result = gk_mul_mat(ctx, im2col, a);

    return gk_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);
}

struct gk_tensor * gk_conv_1d_dw_ph(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                    int s0, int d0) {
    return gk_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

struct gk_tensor * gk_conv_2d(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                              int s0, int s1, int p0, int p1, int d0, int d1) {
    struct gk_tensor * im2col = gk_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true,
            a->type == GK_TYPE_BF16 ? GK_TYPE_F32 : GK_TYPE_F16); // [N, OH, OW, IC*KH*KW]

    struct gk_tensor * result =
        gk_mul_mat(ctx,
                gk_reshape_2d(ctx, im2col, im2col->ne[0],
                              im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                gk_reshape_2d(ctx, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]));

    result = gk_reshape_4d(ctx, result,
                           im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    return gk_cont(ctx, gk_permute(ctx, result, 0, 1, 3, 2));                      // [N, OC, OH, OW]
}

struct gk_tensor * gk_conv_2d_sk_p0(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

struct gk_tensor * gk_conv_2d_s1_ph(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b) {
    return gk_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

struct gk_tensor * gk_conv_2d_dw(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                 int s0, int s1, int p0, int p1, int d0, int d1) {
    struct gk_tensor * new_a = gk_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    struct gk_tensor * im2col = gk_im2col(ctx, new_a,
            gk_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
            s0, s1, p0, p1, d0, d1, true,
            a->type == GK_TYPE_BF16 ? GK_TYPE_F32 : GK_TYPE_F16); // [N*IC, OH, OW, KH*KW]
    struct gk_tensor * new_b = gk_reshape_4d(ctx, im2col,
            im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]);

    new_a = gk_reshape_4d(ctx, new_a,
            new_a->ne[0] * new_a->ne[1], new_a->ne[2], new_a->ne[3], 1);

    struct gk_tensor * result = gk_mul_mat(ctx, new_a, new_b);
    return gk_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]);
}

struct gk_tensor * gk_conv_2d_direct(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                     int s0, int s1, int p0, int p1, int d0, int d1) {
    GK_ASSERT(a->ne[2] == b->ne[2]);

    const int64_t ne[4] = {
        gk_conv_out_size(b->ne[0], a->ne[0], s0, p0, d0),
        gk_conv_out_size(b->ne[1], a->ne[1], s1, p1, d1),
        a->ne[3],
        b->ne[3],
    };

    struct gk_tensor * result = gk_new_tensor(ctx, b->type, 4, ne);

    int32_t params[] = { s0, s1, p0, p1, d0, d1 };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// True for a [W, H, C, N] map stored channels-fastest: CWHN order in memory,
// permuted to look WHCN. The depthwise kernel handles that layout natively.
static bool gk_is_contiguous_channels(const struct gk_tensor * t) {
    return t->nb[0] > t->nb[2] && t->nb[1] > t->nb[0] && t->nb[2] == gk_type_size(t->type);
}

struct gk_tensor * gk_conv_2d_dw_direct(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                        int s0, int s1, int p0, int p1, int d0, int d1) {
    GK_ASSERT(a->ne[2] == 1);
    GK_ASSERT(a->ne[3] == b->ne[2]);

    const int64_t ne[4] = {
        gk_conv_out_size(b->ne[0], a->ne[0], s0, p0, d0),
        gk_conv_out_size(b->ne[1], a->ne[1], s1, p1, d1),
        b->ne[2],
        b->ne[3],
    };

    struct gk_tensor * result = gk_new_tensor(ctx, b->type, 4, ne);

    if (gk_is_contiguous_channels(b)) {
        // the result keeps the input's CWHN order
        const size_t ts = gk_type_size(result->type);
        result->nb[0] = result->ne[2] * ts;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = ts;
    }

    int32_t params[] = { s0, s1, p0, p1, d0, d1 };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_conv_3d(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                              int64_t IC, int s0, int s1, int s2,
                              int p0, int p1, int p2, int d0, int d1, int d2) {
    struct gk_tensor * im2col = gk_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2,
            a->type == GK_TYPE_BF16 ? GK_TYPE_F32 : GK_TYPE_F16); // [N*OD, OH, OW, IC*KD*KH*KW]

    const int64_t OC = a->ne[3] / IC;
    const int64_t N  = b->ne[3] / IC;

    struct gk_tensor * result =
        gk_mul_mat(ctx,
                gk_reshape_2d(ctx, im2col, im2col->ne[0],
                              im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                gk_reshape_2d(ctx, a, a->ne[0] * a->ne[1] * a->ne[2] * IC, OC));

    const int64_t OD = im2col->ne[3] / N;
    result = gk_reshape_4d(ctx, result, im2col->ne[1] * im2col->ne[2], OD, N, OC);
    result = gk_cont(ctx, gk_permute(ctx, result, 0, 1, 3, 2));
    return gk_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N);
}

struct gk_tensor * gk_conv_3d_direct(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                     int s0, int s1, int s2, int p0, int p1, int p2,
                                     int d0, int d1, int d2,
                                     int n_channels, int n_batch, int n_channels_out) {
    GK_ASSERT(a->ne[3] == (int64_t) n_channels * n_channels_out);
    GK_ASSERT(b->ne[3] == (int64_t) n_channels * n_batch);

    const int64_t ne[4] = {
        gk_conv_out_size(b->ne[0], a->ne[0], s0, p0, d0),
        gk_conv_out_size(b->ne[1], a->ne[1], s1, p1, d1),
        gk_conv_out_size(b->ne[2], a->ne[2], s2, p2, d2),
        (int64_t) n_channels_out * n_batch,
    };

    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2,
                         n_channels, n_batch, n_channels_out };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_CONV_3D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_conv_transpose_1d(struct gk_ctx * ctx, struct gk_tensor * a,
                                        struct gk_tensor * b, int s0, int p0, int d0) {
    GK_ASSERT(b->ne[1] == a->ne[2]); // shared input-channel count

    // the output length is computed for p 0, d 1 - the only configuration the
    // kernel below implements, and the only one the engine builds
    const int64_t ne[4] = {
        (b->ne[0] - 1) * s0 + a->ne[0],
        a->ne[1], b->ne[2], 1,
    };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_conv_transpose_2d_p0(struct gk_ctx * ctx, struct gk_tensor * a,
                                           struct gk_tensor * b, int stride) {
    GK_ASSERT(a->ne[3] == b->ne[2]);

    const int64_t ne[4] = {
        gk_conv_transpose_out_size(b->ne[0], a->ne[0], stride, 0),
        gk_conv_transpose_out_size(b->ne[1], a->ne[1], stride, 0),
        a->ne[2], b->ne[3],
    };

    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    gk_set_op_params_i32(result, 0, stride);

    result->op     = GK_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_pool_1d(struct gk_ctx * ctx, struct gk_tensor * a,
                              enum gk_op_pool op, int k0, int s0, int p0) {
    const int64_t ne[4] = {
        gk_pool_out_size(a->ne[0], k0, s0, (float) p0),
        a->ne[1], a->ne[2], a->ne[3],
    };
    GK_ASSERT(ne[0] > 0);

    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    int32_t params[] = { (int32_t) op, k0, s0, p0 };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_POOL_1D;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_pool_2d(struct gk_ctx * ctx, struct gk_tensor * a,
                              enum gk_op_pool op, int k0, int k1, int s0, int s1,
                              float p0, float p1) {
    const int64_t ne[4] = {
        gk_pool_out_size(a->ne[0], k0, s0, p0),
        gk_pool_out_size(a->ne[1], k1, s1, p1),
        a->ne[2], a->ne[3],
    };
    GK_ASSERT(ne[0] > 0 && ne[1] > 0);

    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    int32_t params[] = { (int32_t) op, k0, k1, s0, s1, (int32_t) p0, (int32_t) p1 };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_POOL_2D;
    result->src[0] = a;

    return result;
}

struct gk_tensor * gk_win_part(struct gk_ctx * ctx, struct gk_tensor * a, int w) {
    GK_ASSERT(a->ne[3] == 1);
    GK_ASSERT(a->type == GK_TYPE_F32);

    const int px = (int) ((w - a->ne[1] % w) % w);
    const int py = (int) ((w - a->ne[2] % w) % w);

    const int npx = (int) ((px + a->ne[1]) / w);
    const int npy = (int) ((py + a->ne[2]) / w);

    const int64_t ne[4] = { a->ne[0], w, w, (int64_t) npx * npy };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_WIN_PART;
    result->src[0] = a;

    return result;
}

// --------------------------------------------------------------------------
// recurrent layers
// --------------------------------------------------------------------------

struct gk_tensor * gk_ssm_conv(struct gk_ctx * ctx, struct gk_tensor * sx, struct gk_tensor * c) {
    GK_ASSERT(sx->ne[3] == 1);
    GK_ASSERT(gk_is_matrix(c));

    const int64_t d_conv  = c->ne[0];
    const int64_t d_inner = c->ne[1];
    const int64_t n_t     = sx->ne[0] - d_conv + 1;
    const int64_t n_s     = sx->ne[2];

    GK_ASSERT(sx->ne[1] == d_inner);
    GK_ASSERT(n_t >= 0);

    struct gk_tensor * result = gk_new_tensor_3d(ctx, GK_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GK_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    return result;
}

struct gk_tensor * gk_ssm_scan(struct gk_ctx * ctx, struct gk_tensor * s, struct gk_tensor * x,
                               struct gk_tensor * dt, struct gk_tensor * A,
                               struct gk_tensor * B, struct gk_tensor * C,
                               struct gk_tensor * ids) {
    GK_ASSERT(gk_is_contiguous(s));
    GK_ASSERT(gk_is_contiguous(dt));
    GK_ASSERT(gk_is_contiguous(A));
    GK_ASSERT(x->nb[0] == gk_type_size(x->type));
    GK_ASSERT(B->nb[0] == gk_type_size(B->type));
    GK_ASSERT(C->nb[0] == gk_type_size(C->type));
    GK_ASSERT(gk_are_same_shape(B, C));
    GK_ASSERT(ids->type == GK_TYPE_I32);

    const int64_t d_state  = s->ne[0];
    const int64_t head_dim = x->ne[0];
    const int64_t n_head   = x->ne[1];
    const int64_t n_seqs   = x->ne[3];

    GK_ASSERT(dt->ne[0] == n_head);
    GK_ASSERT(s->ne[1] == head_dim);
    GK_ASSERT(s->ne[2] == n_head);
    GK_ASSERT(B->ne[0] == d_state);
    GK_ASSERT(ids->ne[0] == n_seqs);
    GK_ASSERT(A->ne[1] == n_head);
    GK_ASSERT(A->ne[0] == 1 || A->ne[0] == d_state);

    // the per-token outputs, then the final states, in one flat result
    struct gk_tensor * result = gk_new_tensor_1d(ctx, GK_TYPE_F32,
            gk_nelements(x) + s->ne[0] * s->ne[1] * s->ne[2] * ids->ne[0]);

    result->op     = GK_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    return result;
}

// The three RWKV-shaped builders share the output convention: [S*H,
// n_tokens + S*n_seqs] with the token rows first and the states after.
struct gk_tensor * gk_rwkv_wkv6(struct gk_ctx * ctx, struct gk_tensor * k, struct gk_tensor * v,
                                struct gk_tensor * r, struct gk_tensor * tf,
                                struct gk_tensor * td, struct gk_tensor * state) {
    GK_ASSERT(gk_is_contiguous(k) && gk_is_contiguous(v) && gk_is_contiguous(r));
    GK_ASSERT(gk_is_contiguous(tf) && gk_is_contiguous(td) && gk_is_contiguous(state));

    const int64_t S = k->ne[0];
    const int64_t n_tokens = k->ne[2];
    const int64_t n_seqs = state->ne[1];

    const int64_t ne[4] = { S * k->ne[1], n_tokens + S * n_seqs, 1, 1 };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    result->op     = GK_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    return result;
}

struct gk_tensor * gk_rwkv_wkv7(struct gk_ctx * ctx, struct gk_tensor * r, struct gk_tensor * w,
                                struct gk_tensor * k, struct gk_tensor * v,
                                struct gk_tensor * a, struct gk_tensor * b,
                                struct gk_tensor * state) {
    GK_ASSERT(gk_is_contiguous(r) && gk_is_contiguous(w) && gk_is_contiguous(k));
    GK_ASSERT(gk_is_contiguous(v) && gk_is_contiguous(a) && gk_is_contiguous(b));
    GK_ASSERT(gk_is_contiguous(state));

    const int64_t S = k->ne[0];
    const int64_t n_tokens = k->ne[2];
    const int64_t n_seqs = state->ne[1];

    const int64_t ne[4] = { S * k->ne[1], n_tokens + S * n_seqs, 1, 1 };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    result->op     = GK_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    return result;
}

struct gk_tensor * gk_gated_linear_attn(struct gk_ctx * ctx, struct gk_tensor * k,
                                        struct gk_tensor * v, struct gk_tensor * q,
                                        struct gk_tensor * g, struct gk_tensor * state,
                                        float scale) {
    GK_ASSERT(gk_is_contiguous(k) && gk_is_contiguous(v) && gk_is_contiguous(q));
    GK_ASSERT(gk_is_contiguous(g) && gk_is_contiguous(state));

    const int64_t S = k->ne[0];
    const int64_t n_tokens = k->ne[2];
    const int64_t n_seqs = state->ne[1];

    const int64_t ne[4] = { S * k->ne[1], n_tokens + S * n_seqs, 1, 1 };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    gk_set_op_params_f32(result, 0, scale);

    result->op     = GK_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    return result;
}

struct gk_tensor * gk_gated_delta_net(struct gk_ctx * ctx, struct gk_tensor * q,
                                      struct gk_tensor * k, struct gk_tensor * v,
                                      struct gk_tensor * g, struct gk_tensor * beta,
                                      struct gk_tensor * state, int64_t K) {
    GK_ASSERT(gk_is_contiguous(g) && gk_is_contiguous(beta) && gk_is_contiguous(state));
    GK_ASSERT(q->type == GK_TYPE_F32 && k->type == GK_TYPE_F32 && v->type == GK_TYPE_F32);
    GK_ASSERT(g->type == GK_TYPE_F32 && beta->type == GK_TYPE_F32 && state->type == GK_TYPE_F32);

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];

    GK_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GK_ASSERT(beta->ne[0] == 1);
    GK_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v);
    GK_ASSERT(state->ne[2] == H && state->ne[3] == n_seqs);
    GK_ASSERT(K >= 1);

    const int64_t ne[4] = { S_v * H, n_tokens * n_seqs + K * S_v * n_seqs, 1, 1 };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    gk_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GK_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    return result;
}

struct gk_tensor * gk_solve_tri(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                bool left, bool lower, bool uni) {
    GK_ASSERT(a->type == GK_TYPE_F32 && b->type == GK_TYPE_F32);
    GK_ASSERT(a->ne[0] == a->ne[1]);
    GK_ASSERT(a->ne[1] == b->ne[1]);
    GK_ASSERT(a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3]);
    GK_ASSERT(gk_is_contiguous(a) && gk_is_contiguous(b));

    // the one variant anything builds; the flags stay in the signature so a
    // second variant is an implementation, not an interface change
    GK_ASSERT(lower && left && !uni);

    struct gk_tensor * result = gk_new_tensor_4d(ctx, GK_TYPE_F32,
            b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GK_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct gk_tensor * gk_lightning_indexer(struct gk_ctx * ctx, struct gk_tensor * q,
                                        struct gk_tensor * k, struct gk_tensor * weights,
                                        struct gk_tensor * mask) {
    GK_ASSERT(q->type == GK_TYPE_F32);
    GK_ASSERT(weights->type == GK_TYPE_F32);
    GK_ASSERT(mask->type == GK_TYPE_F16);
    GK_ASSERT(q->ne[0] == k->ne[0]);
    GK_ASSERT(mask->ne[0] == k->ne[2]);
    GK_ASSERT(q->ne[1] == weights->ne[0]);
    GK_ASSERT(k->ne[1] == 1);
    GK_ASSERT(mask->ne[1] == q->ne[2]);
    GK_ASSERT(q->ne[2] == weights->ne[1]);
    GK_ASSERT(q->ne[3] == k->ne[3]);

    const int64_t ne[4] = { k->ne[2], q->ne[2], 1, q->ne[3] };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 4, ne);

    result->op     = GK_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    return result;
}

struct gk_tensor * gk_dsv4_hc_comb(struct gk_ctx * ctx, struct gk_tensor * mixes,
                                   struct gk_tensor * scale, struct gk_tensor * base,
                                   float eps, int32_t n_iter) {
    GK_ASSERT(mixes->type == GK_TYPE_F32);
    GK_ASSERT(scale->type == GK_TYPE_F32);
    GK_ASSERT(base->type == GK_TYPE_F32);
    GK_ASSERT(n_iter > 0);

    // (2 + hc) * hc mixing lanes; the kernel is written for hc 4
    GK_ASSERT(mixes->ne[0] == 24);
    GK_ASSERT(scale->ne[0] >= 3);
    GK_ASSERT(base->ne[0] == mixes->ne[0]);

    struct gk_tensor * result = gk_new_tensor_3d(ctx, GK_TYPE_F32, 4, 4, mixes->ne[1]);

    gk_set_op_params_f32(result, 0, eps);
    gk_set_op_params_i32(result, 1, n_iter);

    result->op     = GK_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    return result;
}

struct gk_tensor * gk_dsv4_hc_pre(struct gk_ctx * ctx, struct gk_tensor * x,
                                  struct gk_tensor * weights) {
    GK_ASSERT(x->type == GK_TYPE_F32 && weights->type == GK_TYPE_F32);
    GK_ASSERT(weights->ne[0] == x->ne[1]);
    GK_ASSERT(weights->ne[1] == x->ne[2]);

    struct gk_tensor * result = gk_new_tensor_2d(ctx, GK_TYPE_F32, x->ne[0], x->ne[2]);

    result->op     = GK_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    return result;
}

struct gk_tensor * gk_dsv4_hc_post(struct gk_ctx * ctx, struct gk_tensor * x,
                                   struct gk_tensor * residual, struct gk_tensor * post,
                                   struct gk_tensor * comb) {
    GK_ASSERT(x->type == GK_TYPE_F32 && residual->type == GK_TYPE_F32);
    GK_ASSERT(post->type == GK_TYPE_F32 && comb->type == GK_TYPE_F32);

    const int64_t hc = residual->ne[1];
    GK_ASSERT(residual->ne[0] == x->ne[0]);
    GK_ASSERT(residual->ne[2] == x->ne[1]);
    GK_ASSERT(post->ne[0] == hc && post->ne[1] == x->ne[1]);
    GK_ASSERT(comb->ne[0] == hc && comb->ne[1] == hc && comb->ne[2] == x->ne[1]);

    struct gk_tensor * result = gk_new_tensor_3d(ctx, GK_TYPE_F32, x->ne[0], hc, x->ne[1]);

    result->op     = GK_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    return result;
}

// --------------------------------------------------------------------------
// custom ops
//
// The function pointer, task count and user pointer travel in the op params.
// n_tasks caps how many threads call the function; GK_N_TASKS_MAX means all
// of them.
// --------------------------------------------------------------------------

static struct gk_tensor * gk_map_custom_impl(
        struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b, struct gk_tensor * c,
        void * fun, int n_tasks, void * userdata, enum gk_op op, bool inplace) {

    GK_ASSERT(n_tasks == GK_N_TASKS_MAX || n_tasks > 0);

    struct gk_tensor * result = inplace ? gk_view_tensor(ctx, a) : gk_dup_tensor(ctx, a);

    struct gk_custom_params params = { fun, n_tasks, userdata };
    gk_set_op_params(result, &params, sizeof(params));

    result->op     = op;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct gk_tensor * gk_map_custom1(struct gk_ctx * ctx, struct gk_tensor * a,
                                  gk_custom1_op_t fun, int n_tasks, void * userdata) {
    return gk_map_custom_impl(ctx, a, NULL, NULL, (void *) fun, n_tasks, userdata,
                              GK_OP_MAP_CUSTOM1, false);
}

struct gk_tensor * gk_map_custom1_inplace(struct gk_ctx * ctx, struct gk_tensor * a,
                                  gk_custom1_op_t fun, int n_tasks, void * userdata) {
    return gk_map_custom_impl(ctx, a, NULL, NULL, (void *) fun, n_tasks, userdata,
                              GK_OP_MAP_CUSTOM1, true);
}

struct gk_tensor * gk_map_custom2(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                  gk_custom2_op_t fun, int n_tasks, void * userdata) {
    return gk_map_custom_impl(ctx, a, b, NULL, (void *) fun, n_tasks, userdata,
                              GK_OP_MAP_CUSTOM2, false);
}

struct gk_tensor * gk_map_custom2_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                  gk_custom2_op_t fun, int n_tasks, void * userdata) {
    return gk_map_custom_impl(ctx, a, b, NULL, (void *) fun, n_tasks, userdata,
                              GK_OP_MAP_CUSTOM2, true);
}

struct gk_tensor * gk_map_custom3(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                  struct gk_tensor * c,
                                  gk_custom3_op_t fun, int n_tasks, void * userdata) {
    return gk_map_custom_impl(ctx, a, b, c, (void *) fun, n_tasks, userdata,
                              GK_OP_MAP_CUSTOM3, false);
}

struct gk_tensor * gk_map_custom3_inplace(struct gk_ctx * ctx, struct gk_tensor * a, struct gk_tensor * b,
                                  struct gk_tensor * c,
                                  gk_custom3_op_t fun, int n_tasks, void * userdata) {
    return gk_map_custom_impl(ctx, a, b, c, (void *) fun, n_tasks, userdata,
                              GK_OP_MAP_CUSTOM3, true);
}

struct gk_tensor * gk_custom_4d(struct gk_ctx * ctx, enum gk_type type,
                                int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                struct gk_tensor ** args, int n_args,
                                gk_custom_op_t fun, int n_tasks, void * userdata) {
    GK_ASSERT(n_args < GK_MAX_SRC);

    struct gk_tensor * result = gk_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    struct gk_custom_params params = { (void *) fun, n_tasks, userdata };
    gk_set_op_params(result, &params, sizeof(params));

    result->op = GK_OP_CUSTOM;
    for (int i = 0; i < n_args; ++i) {
        result->src[i] = args[i];
    }

    return result;
}

struct gk_tensor * gk_custom_inplace(struct gk_ctx * ctx, struct gk_tensor * a,
                                     struct gk_tensor ** args, int n_args,
                                     gk_custom_op_t fun, int n_tasks, void * userdata) {
    GK_ASSERT(n_args < GK_MAX_SRC - 1);

    struct gk_tensor * result = gk_view_tensor(ctx, a);

    struct gk_custom_params params = { (void *) fun, n_tasks, userdata };
    gk_set_op_params(result, &params, sizeof(params));

    result->op     = GK_OP_CUSTOM;
    result->src[0] = a;
    for (int i = 0; i < n_args; ++i) {
        result->src[i + 1] = args[i];
    }

    return result;
}

struct gk_tensor * gk_win_unpart(struct gk_ctx * ctx, struct gk_tensor * a, int w0, int h0, int w) {
    GK_ASSERT(a->type == GK_TYPE_F32);

    const int64_t ne[4] = { a->ne[0], w0, h0, 1 };
    struct gk_tensor * result = gk_new_tensor(ctx, GK_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    gk_set_op_params(result, params, sizeof(params));

    result->op     = GK_OP_WIN_UNPART;
    result->src[0] = a;

    return result;
}
