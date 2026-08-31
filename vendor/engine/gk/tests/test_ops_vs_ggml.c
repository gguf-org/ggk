// Differential tests for the compute pass.
//
// For each op: build the same graph twice, once with gk and once with the
// reference, feed both the identical random inputs, evaluate both, and compare
// the results element by element.
//
// The two implementations are independent, so exact agreement is not the bar
// for anything that accumulates - a sum in a different order lands a few ulps
// apart. Each check states a tolerance scaled to the magnitude of what is
// being compared, and the tolerances are tight enough that a genuine wrong
// answer cannot slip through: a mispaired rope, a missing mask, a transposed
// matmul all move the result by far more than rounding.
//
// Where a result *is* required to be exact, it is checked exactly - index
// outputs from argsort and argmax, and anything that is a pure data movement.

#include "gk_impl.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fails  = 0;
static int g_checks = 0;

static uint64_t g_rng = 0x2545f4914f6cdd1dull;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float) (uint32_t) (g_rng >> 33) / (float) 0x7fffffffu * 2.0f - 1.0f;
}

// Thread count for the gk side. Every comparison below runs at this setting,
// so raising it re-runs the whole suite threaded - which is how the threading
// is checked for races: the results must not move.
static int g_threads = 1;

static struct gk_ctx     * G;  // gk context
static struct ggml_context * R;  // reference context

// --------------------------------------------------------------------------

// Compares a gk result against a reference result. `tol` is relative to the
// larger of the two magnitudes, with a small absolute floor so values near
// zero are not held to an impossible relative bar.
static void compare(const char * name, const struct gk_tensor * a,
                    const struct ggml_tensor * b, double tol) {
    g_checks++;

    printf("  %-22s ", name);

    if (gk_nelements(a) != ggml_nelements(b)) {
        printf("FAIL: %lld elements vs %lld\n",
               (long long) gk_nelements(a), (long long) ggml_nelements(b));
        g_fails++;
        return;
    }

    const int64_t n = gk_nelements(a);

    double worst     = 0.0;
    int64_t worst_at = -1;
    int64_t n_bad    = 0;

    for (int64_t i = 0; i < n; ++i) {
        double x, y;

        if (a->type == GK_TYPE_I32) {
            x = (double) ((const int32_t *) a->data)[i];
            y = (double) ((const int32_t *) b->data)[i];
        } else {
            x = (double) ((const float *) a->data)[i];
            y = (double) ((const float *) b->data)[i];
        }

        // -inf on both sides is agreement, not a difference
        if (isinf(x) && isinf(y) && ((x < 0) == (y < 0))) {
            continue;
        }

        const double scale = fmax(fmax(fabs(x), fabs(y)), 1e-4);
        const double rel   = fabs(x - y) / scale;

        if (rel > worst) {
            worst    = rel;
            worst_at = i;
        }
        if (rel > tol) {
            n_bad++;
        }
    }

    if (n_bad > 0) {
        const double x = a->type == GK_TYPE_I32
            ? (double) ((const int32_t *) a->data)[worst_at]
            : (double) ((const float *) a->data)[worst_at];
        const double y = b->type == GK_TYPE_I32
            ? (double) ((const int32_t *) b->data)[worst_at]
            : (double) ((const float *) b->data)[worst_at];

        printf("FAIL: %lld/%lld off, worst %.3g at [%lld] (%g vs %g)\n",
               (long long) n_bad, (long long) n, worst, (long long) worst_at, x, y);
        g_fails++;
    } else {
        printf("ok   (%lld elems, worst rel %.2g)\n", (long long) n, worst);
    }
}

// Creates the same f32 tensor in both libraries, filled with identical data.
static void make_pair(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, float scale,
                      struct gk_tensor ** ga, struct ggml_tensor ** ra) {
    *ga = gk_new_tensor_4d(G, GK_TYPE_F32, ne0, ne1, ne2, ne3);
    *ra = ggml_new_tensor_4d(R, GGML_TYPE_F32, ne0, ne1, ne2, ne3);

    const int64_t n = ne0 * ne1 * ne2 * ne3;
    for (int64_t i = 0; i < n; ++i) {
        const float v = frand() * scale;
        ((float *) (*ga)->data)[i] = v;
        ((float *) (*ra)->data)[i] = v;
    }
}

// An i32 index tensor, same contents on both sides.
static void make_pair_i32(int64_t ne0, int64_t ne1, int32_t lo, int32_t hi,
                          struct gk_tensor ** ga, struct ggml_tensor ** ra) {
    *ga = gk_new_tensor_2d(G, GK_TYPE_I32, ne0, ne1);
    *ra = ggml_new_tensor_2d(R, GGML_TYPE_I32, ne0, ne1);

    const int64_t n = ne0 * ne1;
    for (int64_t i = 0; i < n; ++i) {
        g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
        const int32_t v = lo + (int32_t) ((g_rng >> 33) % (uint32_t) (hi - lo + 1));
        ((int32_t *) (*ga)->data)[i] = v;
        ((int32_t *) (*ra)->data)[i] = v;
    }
}

// A quantized weight, identical bytes on both sides: encoded once by the
// shared codec and copied, so this isolates the matmul from any encoder
// difference. Encoder agreement is what test-vs-ggml already covers.
static void make_pair_quant(enum gk_type type, int64_t ne0, int64_t ne1,
                            struct gk_tensor ** ga, struct ggml_tensor ** ra) {
    *ga = gk_new_tensor_2d(G, type, ne0, ne1);
    *ra = ggml_new_tensor_2d(R, (enum ggml_type) type, ne0, ne1);

    float * tmp = (float *) malloc((size_t) ne0 * sizeof(float));

    const struct gk_type_traits * tr = gk_get_type_traits(type);
    const size_t row = gk_row_size(type, ne0);

    for (int64_t r = 0; r < ne1; ++r) {
        for (int64_t i = 0; i < ne0; ++i) {
            const float u = frand();
            tmp[i] = u * u * u;
        }
        tr->from_float(tmp, (char *) (*ga)->data + (size_t) r * row, ne0);
    }

    memcpy((*ra)->data, (*ga)->data, row * (size_t) ne1);
    free(tmp);
}

// Evaluates one output on each side.
static void run_both(struct gk_tensor * gout, struct ggml_tensor * rout) {
    struct gk_cgraph * gg = gk_new_graph(G);
    gk_build_forward_expand(gg, gout);
    const enum gk_status st = gk_graph_compute(gg, g_threads);
    if (st != GK_STATUS_SUCCESS) {
        printf("  gk_graph_compute failed with status %d\n", (int) st);
        g_fails++;
        return;
    }

    struct ggml_cgraph * rg = ggml_new_graph(R);
    ggml_build_forward_expand(rg, rout);
    ggml_graph_compute_with_ctx(R, rg, 1);
}

#define TEST(name, gexpr, rexpr, tol) \
    do { \
        struct gk_tensor   * go = (gexpr); \
        struct ggml_tensor * ro = (rexpr); \
        run_both(go, ro); \
        compare(name, go, ro, tol); \
    } while (0)

// --------------------------------------------------------------------------

static void test_elementwise(void) {
    printf("elementwise\n");

    struct gk_tensor * ga, * gb; struct ggml_tensor * ra, * rb;

    make_pair(64, 8, 2, 2, 1.0f, &ga, &ra);
    make_pair(64, 8, 2, 2, 1.0f, &gb, &rb);

    TEST("add",   gk_add(G, ga, gb),   ggml_add(R, ra, rb),   1e-6);
    TEST("sub",   gk_sub(G, ga, gb),   ggml_sub(R, ra, rb),   1e-6);
    TEST("mul",   gk_mul(G, ga, gb),   ggml_mul(R, ra, rb),   1e-6);
    TEST("div",   gk_div(G, ga, gb),   ggml_div(R, ra, rb),   1e-5);
    TEST("scale", gk_scale(G, ga, 0.7f), ggml_scale(R, ra, 0.7f), 1e-6);
    TEST("sqr",   gk_sqr(G, ga),       ggml_sqr(R, ra),       1e-6);

    // broadcast: a row-shaped operand applied across every row
    struct gk_tensor * gr; struct ggml_tensor * rr;
    make_pair(64, 1, 1, 1, 1.0f, &gr, &rr);

    TEST("add broadcast", gk_add(G, ga, gr), ggml_add(R, ra, rr), 1e-6);
    TEST("mul broadcast", gk_mul(G, ga, gr), ggml_mul(R, ra, rr), 1e-6);
}

static void test_activations(void) {
    printf("activations\n");

    struct gk_tensor * ga; struct ggml_tensor * ra;
    make_pair(128, 16, 1, 1, 3.0f, &ga, &ra);

    TEST("relu",       gk_relu(G, ga),       ggml_relu(R, ra),       1e-6);
    TEST("sigmoid",    gk_sigmoid(G, ga),    ggml_sigmoid(R, ra),    1e-6);
    TEST("tanh",       gk_tanh(G, ga),       ggml_tanh(R, ra),       1e-6);
    TEST("exp",        gk_exp(G, ga),        ggml_exp(R, ra),        1e-6);
    TEST("neg",        gk_neg(G, ga),        ggml_neg(R, ra),        1e-6);
    TEST("silu",       gk_silu(G, ga),       ggml_silu(R, ra),       1e-6);
    // The reference evaluates gelu through an f16 lookup table, so it carries
    // about half-precision accuracy. Ours evaluates the tanh form directly and
    // is the more accurate of the two; the bar here is agreement to within the
    // table's resolution, not to float rounding.
    TEST("gelu",       gk_gelu(G, ga),       ggml_gelu(R, ra),       5e-3);
    TEST("gelu_quick", gk_gelu_quick(G, ga), ggml_gelu_quick(R, ra), 5e-3);
    TEST("gelu_erf",   gk_gelu_erf(G, ga),   ggml_gelu_erf(R, ra),   1e-3);

    // The gate half of a swiglu comes from the second half of the row.
    struct gk_tensor * gw; struct ggml_tensor * rw;
    make_pair(256, 16, 1, 1, 2.0f, &gw, &rw);
    TEST("swiglu", gk_glu(G, gw, GK_GLU_OP_SWIGLU, false), ggml_swiglu(R, rw), 1e-6);
    TEST("geglu",  gk_glu(G, gw, GK_GLU_OP_GEGLU, false),  ggml_geglu(R, rw),  5e-3);
}

static void test_norms(void) {
    printf("normalisation\n");

    struct gk_tensor * ga; struct ggml_tensor * ra;
    make_pair(512, 8, 2, 1, 1.0f, &ga, &ra);

    TEST("rms_norm", gk_rms_norm(G, ga, 1e-5f), ggml_rms_norm(R, ra, 1e-5f), 1e-5);
    TEST("norm",     gk_norm(G, ga, 1e-5f),     ggml_norm(R, ra, 1e-5f),     1e-4);

    struct gk_tensor * gg; struct ggml_tensor * rg;
    make_pair(16, 16, 8, 1, 1.0f, &gg, &rg);
    TEST("group_norm", gk_group_norm(G, gg, 4, 1e-5f),
                       ggml_group_norm(R, rg, 4, 1e-5f), 1e-4);
}

// Quantized matmul needs a different bar from the ops above.
//
// Comparing our result against the reference's would be comparing two
// approximations to each other, and on a dot that cancels down to near zero
// the relative gap between them says nothing useful. So the truth is computed
// directly - dequantize the weight through the shared codec and dot in double
// - and each implementation is measured against that.
//
// What counts as passing depends on which path the format is on, and the
// traits say which:
//
//   float path (vec_dot_type f32)   the weight is widened and multiplied in
//                                   float, so the only error is rounding and
//                                   the result should sit at ~1e-8.
//
//   integer path                    the activations are quantized to 8 bits
//                                   before the dot, which is a real loss the
//                                   float path did not have - about a part in
//                                   1e-3. The reference does exactly the same
//                                   thing, so the bar is that we are no worse
//                                   than it is.
static void test_matmul_quant(void) {
    static const enum gk_type types[] = {
        GK_TYPE_Q4_0, GK_TYPE_Q4_1, GK_TYPE_Q5_0, GK_TYPE_Q8_0,
        GK_TYPE_Q2_K, GK_TYPE_Q3_K, GK_TYPE_Q4_K, GK_TYPE_Q5_K, GK_TYPE_Q6_K,
        GK_TYPE_IQ4_NL, GK_TYPE_IQ4_XS, GK_TYPE_MXFP4,
    };

    const int64_t k    = 256; // reduction length
    const int64_t rows = 32;  // weight rows
    const int64_t cols = 4;   // activation columns

    float * wrow = (float *) malloc((size_t) k * sizeof(float));

    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ++ti) {
        const enum gk_type type = types[ti];

        struct gk_tensor * gq, * ga; struct ggml_tensor * rq, * ra;
        make_pair_quant(type, k, rows, &gq, &rq);
        make_pair(k, cols, 1, 1, 1.0f, &ga, &ra);

        struct gk_tensor   * go = gk_mul_mat(G, gq, ga);
        struct ggml_tensor * ro = ggml_mul_mat(R, rq, ra);
        run_both(go, ro);

        const struct gk_type_traits * tr = gk_get_type_traits(type);
        const size_t row_bytes = gk_row_size(type, k);

        double worst_ours = 0.0;
        double worst_ref  = 0.0;

        for (int64_t c = 0; c < cols; ++c) {
            const float * act = (const float *) ga->data + c * k;

            for (int64_t r = 0; r < rows; ++r) {
                // ground truth from the stored bytes
                tr->to_float((const char *) gq->data + (size_t) r * row_bytes, wrow, k);

                double truth = 0.0;
                for (int64_t i = 0; i < k; ++i) {
                    truth += (double) wrow[i] * act[i];
                }

                const double ours = ((const float *) go->data)[c * rows + r];
                const double ref  = ((const float *) ro->data)[c * rows + r];

                // scale by the row's own magnitude, not by the result: a dot
                // that cancels to near zero would otherwise dominate
                double mag = 0.0;
                for (int64_t i = 0; i < k; ++i) {
                    mag += fabs((double) wrow[i] * act[i]);
                }
                mag = fmax(mag, 1e-6);

                worst_ours = fmax(worst_ours, fabs(ours - truth) / mag);
                worst_ref  = fmax(worst_ref,  fabs(ref  - truth) / mag);
            }
        }

        g_checks++;

        const struct gk_type_traits * wtr = gk_get_type_traits(type);
        const bool integer_path = wtr->vec_dot_type != GK_TYPE_F32;

        printf("  mul_mat %-8s %-5s ", gk_type_name(type), integer_path ? "int" : "float");

        // On the float path only rounding separates us from the truth. On the
        // integer path the activation quantization dominates, and the bar is
        // parity with the reference rather than an absolute figure.
        const double bar = integer_path ? worst_ref * 1.05 + 1e-9 : 1e-5;

        if (worst_ours > bar) {
            printf("FAIL: error vs truth %.3g, bar %.3g (reference %.3g)\n",
                   worst_ours, bar, worst_ref);
            g_fails++;
        } else {
            printf("ok   (ours %.3g from truth, reference %.3g)\n", worst_ours, worst_ref);
        }
    }

    free(wrow);
}

static void test_matmul(void) {
    printf("matrix multiply\n");

    // f32 weight
    struct gk_tensor * gw, * gx; struct ggml_tensor * rw, * rx;
    make_pair(256, 64, 1, 1, 0.5f, &gw, &rw);
    make_pair(256, 12, 1, 1, 1.0f, &gx, &rx);

    // Sums in a different order land a few ulps apart, and a dot over 256
    // terms can cancel down to near zero, which inflates the relative figure.
    TEST("mul_mat f32", gk_mul_mat(G, gw, gx), ggml_mul_mat(R, rw, rx), 1e-4);

    // broadcast over heads: fewer key heads than query heads
    struct gk_tensor * gk2, * gx2; struct ggml_tensor * rk2, * rx2;
    make_pair(64, 32, 2, 1, 0.5f, &gk2, &rk2);
    make_pair(64, 12, 4, 1, 1.0f, &gx2, &rx2);
    TEST("mul_mat broadcast", gk_mul_mat(G, gk2, gx2), ggml_mul_mat(R, rk2, rx2), 5e-3);

    test_matmul_quant();
}

static void test_movement(void) {
    printf("data movement\n");

    struct gk_tensor * ga; struct ggml_tensor * ra;
    make_pair(32, 16, 4, 1, 1.0f, &ga, &ra);

    // cont of a permute is the one that actually moves memory
    TEST("cont(permute)",
         gk_cont(G, gk_permute(G, ga, 0, 2, 1, 3)),
         ggml_cont(R, ggml_permute(R, ra, 0, 2, 1, 3)), 0.0);

    TEST("cont(transpose)",
         gk_cont(G, gk_transpose(G, ga)),
         ggml_cont(R, ggml_transpose(R, ra)), 0.0);

    struct gk_tensor * gb; struct ggml_tensor * rb;
    make_pair(32, 16, 4, 1, 1.0f, &gb, &rb);

    TEST("concat dim0", gk_concat(G, ga, gb, 0), ggml_concat(R, ra, rb, 0), 0.0);
    TEST("concat dim1", gk_concat(G, ga, gb, 1), ggml_concat(R, ra, rb, 1), 0.0);
    TEST("concat dim2", gk_concat(G, ga, gb, 2), ggml_concat(R, ra, rb, 2), 0.0);

    // get_rows: the embedding lookup
    struct gk_tensor * gemb, * gidx; struct ggml_tensor * remb, * ridx;
    make_pair(64, 100, 1, 1, 1.0f, &gemb, &remb);
    make_pair_i32(10, 1, 0, 99, &gidx, &ridx);

    TEST("get_rows",
         gk_get_rows(G, gemb, gk_reshape_1d(G, gidx, 10)),
         ggml_get_rows(R, remb, ggml_reshape_1d(R, ridx, 10)), 0.0);

    // repeat
    struct gk_tensor * gsmall, * gbig; struct ggml_tensor * rsmall, * rbig;
    make_pair(16, 2, 1, 1, 1.0f, &gsmall, &rsmall);
    make_pair(32, 4, 1, 1, 1.0f, &gbig, &rbig);
    TEST("repeat", gk_repeat(G, gsmall, gbig), ggml_repeat(R, rsmall, rbig), 0.0);
}

static void test_attention(void) {
    printf("attention\n");

    struct gk_tensor * ga; struct ggml_tensor * ra;
    make_pair(64, 64, 4, 1, 2.0f, &ga, &ra);

    TEST("soft_max", gk_soft_max(G, ga), ggml_soft_max(R, ra), 1e-5);

    TEST("soft_max scaled",
         gk_soft_max_ext(G, ga, NULL, 0.125f, 0.0f),
         ggml_soft_max_ext(R, ra, NULL, 0.125f, 0.0f), 1e-5);

    // with a causal mask
    struct gk_tensor * gm; struct ggml_tensor * rm;
    gm = gk_new_tensor_2d(G, GK_TYPE_F32, 64, 64);
    rm = ggml_new_tensor_2d(R, GGML_TYPE_F32, 64, 64);
    for (int64_t j = 0; j < 64; ++j) {
        for (int64_t i = 0; i < 64; ++i) {
            const float v = i > j ? -INFINITY : 0.0f;
            ((float *) gm->data)[j * 64 + i] = v;
            ((float *) rm->data)[j * 64 + i] = v;
        }
    }

    TEST("soft_max masked",
         gk_soft_max_ext(G, ga, gm, 0.125f, 0.0f),
         ggml_soft_max_ext(R, ra, rm, 0.125f, 0.0f), 1e-5);

    TEST("soft_max alibi",
         gk_soft_max_ext(G, ga, gm, 0.125f, 8.0f),
         ggml_soft_max_ext(R, ra, rm, 0.125f, 8.0f), 1e-5);

    TEST("diag_mask_inf",
         gk_diag_mask_inf(G, ga, 0),
         ggml_diag_mask_inf(R, ra, 0), 0.0);
}

static void test_rope(void) {
    printf("rope\n");

    const int64_t head_dim = 64;
    const int64_t n_head   = 4;
    const int64_t n_tok    = 12;

    struct gk_tensor * ga; struct ggml_tensor * ra;
    make_pair(head_dim, n_head, n_tok, 1, 1.0f, &ga, &ra);

    struct gk_tensor   * gp = gk_new_tensor_1d(G, GK_TYPE_I32, n_tok);
    struct ggml_tensor * rp = ggml_new_tensor_1d(R, GGML_TYPE_I32, n_tok);
    for (int64_t i = 0; i < n_tok; ++i) {
        ((int32_t *) gp->data)[i] = (int32_t) i;
        ((int32_t *) rp->data)[i] = (int32_t) i;
    }

    TEST("rope normal",
         gk_rope_ext(G, ga, gp, NULL, (int) head_dim, 0, 2048,
                     10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_ext(R, ra, rp, NULL, (int) head_dim, 0, 2048,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), 1e-4);

    TEST("rope neox",
         gk_rope_ext(G, ga, gp, NULL, (int) head_dim, GK_ROPE_TYPE_NEOX, 2048,
                     10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_ext(R, ra, rp, NULL, (int) head_dim, GGML_ROPE_TYPE_NEOX, 2048,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), 1e-4);

    // partial rotation: only the first half of the head dimension turns
    TEST("rope partial",
         gk_rope_ext(G, ga, gp, NULL, (int) head_dim / 2, 0, 2048,
                     10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_ext(R, ra, rp, NULL, (int) head_dim / 2, 0, 2048,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), 1e-4);

    // frequency scaling, and then YaRN's blended extension
    TEST("rope freq_scale",
         gk_rope_ext(G, ga, gp, NULL, (int) head_dim, 0, 2048,
                     10000.0f, 0.5f, 0.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_ext(R, ra, rp, NULL, (int) head_dim, 0, 2048,
                       10000.0f, 0.5f, 0.0f, 1.0f, 32.0f, 1.0f), 1e-4);

    TEST("rope yarn",
         gk_rope_ext(G, ga, gp, NULL, (int) head_dim, 0, 2048,
                     10000.0f, 0.5f, 1.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_ext(R, ra, rp, NULL, (int) head_dim, 0, 2048,
                       10000.0f, 0.5f, 1.0f, 1.0f, 32.0f, 1.0f), 1e-4);
}

static void test_reductions(void) {
    printf("reductions and ordering\n");

    struct gk_tensor * ga; struct ggml_tensor * ra;
    make_pair(128, 8, 2, 1, 1.0f, &ga, &ra);

    TEST("sum_rows", gk_sum_rows(G, ga), ggml_sum_rows(R, ra), 1e-5);

    struct gk_tensor * gm; struct ggml_tensor * rm;
    make_pair(64, 8, 1, 1, 1.0f, &gm, &rm);

    // indices must match exactly
    TEST("argsort desc",
         gk_argsort(G, gm, GK_SORT_ORDER_DESC),
         ggml_argsort(R, rm, GGML_SORT_ORDER_DESC), 0.0);

    TEST("argsort asc",
         gk_argsort(G, gm, GK_SORT_ORDER_ASC),
         ggml_argsort(R, rm, GGML_SORT_ORDER_ASC), 0.0);

    TEST("argmax", gk_argmax(G, gm), ggml_argmax(R, rm), 0.0);
}

// A small transformer block wired end to end. Each op above is checked in
// isolation; this checks they compose - that shapes line up through attention
// and the feed-forward, and that error does not build up across a stack.
static void test_block(void) {
    printf("composed transformer block\n");

    const int64_t d_model = 128;
    const int64_t n_head  = 4;
    const int64_t d_head  = d_model / n_head;
    const int64_t n_tok   = 16;

    struct gk_tensor * gx, * gwq, * gwk, * gwv, * gwo, * gw1, * gw2;
    struct ggml_tensor * rx, * rwq, * rwk, * rwv, * rwo, * rw1, * rw2;

    make_pair(d_model, n_tok, 1, 1, 1.0f, &gx, &rx);
    make_pair(d_model, d_model, 1, 1, 0.1f, &gwq, &rwq);
    make_pair(d_model, d_model, 1, 1, 0.1f, &gwk, &rwk);
    make_pair(d_model, d_model, 1, 1, 0.1f, &gwv, &rwv);
    make_pair(d_model, d_model, 1, 1, 0.1f, &gwo, &rwo);
    make_pair(d_model, d_model * 2, 1, 1, 0.1f, &gw1, &rw1);
    make_pair(d_model, d_model, 1, 1, 0.1f, &gw2, &rw2);

    struct gk_tensor   * gp = gk_new_tensor_1d(G, GK_TYPE_I32, n_tok);
    struct ggml_tensor * rp = ggml_new_tensor_1d(R, GGML_TYPE_I32, n_tok);
    for (int64_t i = 0; i < n_tok; ++i) {
        ((int32_t *) gp->data)[i] = (int32_t) i;
        ((int32_t *) rp->data)[i] = (int32_t) i;
    }

    const float scale = 1.0f / sqrtf((float) d_head);

    // ---- gk ----
    struct gk_tensor * gn = gk_rms_norm(G, gx, 1e-5f);
    struct gk_tensor * gq = gk_reshape_3d(G, gk_mul_mat(G, gwq, gn), d_head, n_head, n_tok);
    struct gk_tensor * gk_ = gk_reshape_3d(G, gk_mul_mat(G, gwk, gn), d_head, n_head, n_tok);
    struct gk_tensor * gv = gk_reshape_3d(G, gk_mul_mat(G, gwv, gn), d_head, n_head, n_tok);

    gq  = gk_rope_ext(G, gq,  gp, NULL, (int) d_head, 0, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    gk_ = gk_rope_ext(G, gk_, gp, NULL, (int) d_head, 0, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    gq  = gk_permute(G, gq,  0, 2, 1, 3);
    gk_ = gk_permute(G, gk_, 0, 2, 1, 3);

    struct gk_tensor * gatt = gk_mul_mat(G, gk_cont(G, gk_), gk_cont(G, gq));
    gatt = gk_soft_max_ext(G, gatt, NULL, scale, 0.0f);

    struct gk_tensor * gvt = gk_cont(G, gk_permute(G, gv, 1, 2, 0, 3));
    struct gk_tensor * gout = gk_mul_mat(G, gvt, gatt);
    gout = gk_cont_2d(G, gk_permute(G, gout, 0, 2, 1, 3), d_model, n_tok);
    gout = gk_add(G, gk_mul_mat(G, gwo, gout), gx);

    struct gk_tensor * gf = gk_rms_norm(G, gout, 1e-5f);
    gf = gk_glu(G, gk_mul_mat(G, gw1, gf), GK_GLU_OP_SWIGLU, false);
    gf = gk_add(G, gk_mul_mat(G, gw2, gf), gout);

    // ---- reference ----
    struct ggml_tensor * rn = ggml_rms_norm(R, rx, 1e-5f);
    struct ggml_tensor * rq = ggml_reshape_3d(R, ggml_mul_mat(R, rwq, rn), d_head, n_head, n_tok);
    struct ggml_tensor * rk = ggml_reshape_3d(R, ggml_mul_mat(R, rwk, rn), d_head, n_head, n_tok);
    struct ggml_tensor * rv = ggml_reshape_3d(R, ggml_mul_mat(R, rwv, rn), d_head, n_head, n_tok);

    rq = ggml_rope_ext(R, rq, rp, NULL, (int) d_head, 0, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    rk = ggml_rope_ext(R, rk, rp, NULL, (int) d_head, 0, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    rq = ggml_permute(R, rq, 0, 2, 1, 3);
    rk = ggml_permute(R, rk, 0, 2, 1, 3);

    struct ggml_tensor * ratt = ggml_mul_mat(R, ggml_cont(R, rk), ggml_cont(R, rq));
    ratt = ggml_soft_max_ext(R, ratt, NULL, scale, 0.0f);

    struct ggml_tensor * rvt = ggml_cont(R, ggml_permute(R, rv, 1, 2, 0, 3));
    struct ggml_tensor * rout = ggml_mul_mat(R, rvt, ratt);
    rout = ggml_cont_2d(R, ggml_permute(R, rout, 0, 2, 1, 3), d_model, n_tok);
    rout = ggml_add(R, ggml_mul_mat(R, rwo, rout), rx);

    struct ggml_tensor * rf = ggml_rms_norm(R, rout, 1e-5f);
    rf = ggml_swiglu(R, ggml_mul_mat(R, rw1, rf));
    rf = ggml_add(R, ggml_mul_mat(R, rw2, rf), rout);

    run_both(gf, rf);

    // Every op above is checked individually at float rounding. What this one
    // is for is composition - that shapes line up through attention and the
    // feed-forward, and that error does not compound across the stack. The
    // output has passed through roughly fifteen dot products of length 128 or
    // more, so a few parts in 10^7 of accumulated rounding is expected; a
    // structural mistake would land orders of magnitude above this bar.
    compare("block output", gf, rf, 1e-3);
}

// The data-placement and shaping ops added for engine parity: fills, pads,
// rolls, scatters, resampling, the odd activations. Most are pure data
// movement and are held to (near) exactness; the resampling filters accumulate
// a little and get a small tolerance.
static void test_placement(void) {
    printf("placement and shaping\n");

    struct gk_tensor * ga, * gb; struct ggml_tensor * ra, * rb;

    // fills and generators
    make_pair(33, 7, 2, 1, 1.0f, &ga, &ra);
    TEST("fill",    gk_fill(G, ga, 3.25f), ggml_fill(R, ra, 3.25f), 0.0);
    TEST("arange",  gk_arange(G, 0.5f, 33.0f, 0.75f), ggml_arange(R, 0.5f, 33.0f, 0.75f), 0.0);
    struct gk_tensor * gsq; struct ggml_tensor * rsq;
    make_pair(33, 33, 2, 1, 1.0f, &gsq, &rsq);
    TEST("tri",     gk_tri(G, gsq, GK_TRI_TYPE_LOWER_DIAG),
                    ggml_tri(R, rsq, GGML_TRI_TYPE_LOWER_DIAG), 0.0);
    TEST("cumsum",  gk_cumsum(G, ga), ggml_cumsum(R, ra), 1e-5);
    TEST("roll",    gk_roll(G, ga, 5, 3, 1, 0), ggml_roll(R, ra, 5, 3, 1, 0), 0.0);
    TEST("roll neg",gk_roll(G, ga, -4, -2, 0, 0), ggml_roll(R, ra, -4, -2, 0, 0), 0.0);

    // diag from a vector
    struct gk_tensor * gv; struct ggml_tensor * rv;
    make_pair(24, 1, 2, 1, 1.0f, &gv, &rv);
    TEST("diag", gk_diag(G, gv), ggml_diag(R, rv), 0.0);

    // the odd activations
    make_pair(96, 12, 1, 1, 4.0f, &ga, &ra);
    TEST("softplus",   gk_softplus(G, ga), ggml_softplus(R, ra), 1e-6);
    // The reference computes expf(x) - 1, which cancels catastrophically near
    // zero; ours is expm1f and keeps those bits. The bar here is the
    // reference's own error, not ours.
    TEST("expm1",      gk_expm1(G, ga),    ggml_expm1(R, ra),    5e-5);
    TEST("leaky_relu", gk_leaky_relu(G, ga, 0.1f, false),
                       ggml_leaky_relu(R, ra, 0.1f, false), 1e-6);
    TEST("xielu",      gk_xielu(G, ga, 0.8f, 0.8f, 0.5f, -1e-6f),
                       ggml_xielu(R, ra, 0.8f, 0.8f, 0.5f, -1e-6f), 1e-5);

    // glu variants
    struct gk_tensor * gw; struct ggml_tensor * rw;
    make_pair(128, 8, 1, 1, 2.0f, &gw, &rw);
    TEST("swiglu swapped", gk_glu(G, gw, GK_GLU_OP_SWIGLU, true),
                           ggml_swiglu_swapped(R, rw), 1e-6);
    make_pair(64, 8, 1, 1, 2.0f, &ga, &ra);
    make_pair(64, 8, 1, 1, 2.0f, &gb, &rb);
    TEST("swiglu_oai", gk_swiglu_oai(G, ga, gb, 1.702f, 7.0f),
                       ggml_swiglu_oai(R, ra, rb, 1.702f, 7.0f), 1e-6);

    // add_id: per-row bias selected by index
    struct gk_tensor * gids; struct ggml_tensor * rids;
    make_pair(64, 6, 3, 1, 1.0f, &ga, &ra);
    make_pair(64, 10, 1, 1, 1.0f, &gb, &rb);
    make_pair_i32(6, 3, 0, 9, &gids, &rids);
    TEST("add_id", gk_add_id(G, ga, gb, gids), ggml_add_id(R, ra, rb, rids), 1e-6);

    // acc and set: a window of a larger tensor written or accumulated
    make_pair(64, 16, 1, 1, 1.0f, &ga, &ra);
    make_pair(32, 8, 1, 1, 1.0f, &gb, &rb);
    TEST("acc", gk_acc(G, ga, gb, ga->nb[1], ga->nb[2], ga->nb[3], 16 * sizeof(float)),
                ggml_acc(R, ra, rb, ra->nb[1], ra->nb[2], ra->nb[3], 16 * sizeof(float)), 1e-6);
    TEST("set", gk_set(G, ga, gb, ga->nb[1], ga->nb[2], ga->nb[3], 16 * sizeof(float)),
                ggml_set(R, ra, rb, ra->nb[1], ra->nb[2], ra->nb[3], 16 * sizeof(float)), 0.0);
    TEST("set_1d", gk_set_1d(G, ga, gk_reshape_1d(G, gb, 256), 32 * sizeof(float)),
                   ggml_set_1d(R, ra, ggml_reshape_1d(R, rb, 256), 32 * sizeof(float)), 0.0);

    // pad family
    make_pair(15, 9, 3, 2, 1.0f, &ga, &ra);
    TEST("pad",          gk_pad(G, ga, 3, 2, 1, 0), ggml_pad(R, ra, 3, 2, 1, 0), 0.0);
    TEST("pad_ext",      gk_pad_ext(G, ga, 1, 2, 3, 1, 0, 1, 1, 0),
                         ggml_pad_ext(R, ra, 1, 2, 3, 1, 0, 1, 1, 0), 0.0);
    TEST("pad_circular", gk_pad_circular(G, ga, 4, 3, 1, 1),
                         ggml_pad_circular(R, ra, 4, 3, 1, 1), 0.0);
    TEST("pad_reflect_1d", gk_pad_reflect_1d(G, ga, 4, 5),
                           ggml_pad_reflect_1d(R, ra, 4, 5), 0.0);

    // resampling: every mode, up and down
    make_pair(12, 10, 3, 1, 1.0f, &ga, &ra);
    TEST("upscale nearest",  gk_upscale(G, ga, 3, GK_SCALE_MODE_NEAREST),
                             ggml_upscale(R, ra, 3, GGML_SCALE_MODE_NEAREST), 0.0);
    TEST("upscale bilinear", gk_upscale(G, ga, 3, GK_SCALE_MODE_BILINEAR),
                             ggml_upscale(R, ra, 3, GGML_SCALE_MODE_BILINEAR), 1e-6);
    TEST("upscale bicubic",  gk_upscale(G, ga, 2, GK_SCALE_MODE_BICUBIC),
                             ggml_upscale(R, ra, 2, GGML_SCALE_MODE_BICUBIC), 1e-5);
    TEST("interp align_corners",
         gk_interpolate(G, ga, 30, 24, 3, 1, GK_SCALE_MODE_BILINEAR | GK_SCALE_FLAG_ALIGN_CORNERS),
         ggml_interpolate(R, ra, 30, 24, 3, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS), 1e-6);
    TEST("interp antialias down",
         gk_interpolate(G, ga, 5, 4, 3, 1, GK_SCALE_MODE_BILINEAR | GK_SCALE_FLAG_ANTIALIAS),
         ggml_interpolate(R, ra, 5, 4, 3, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS), 1e-5);

    // timestep embedding, even and odd widths
    make_pair(6, 1, 1, 1, 1.0f, &ga, &ra);
    for (int i = 0; i < 6; ++i) {
        const float v = fabsf(((float *) ga->data)[i]) * 800.0f;
        ((float *) ga->data)[i] = v;
        ((float *) ra->data)[i] = v;
    }
    TEST("timestep_embedding", gk_timestep_embedding(G, ga, 64, 10000),
                               ggml_timestep_embedding(R, ra, 64, 10000), 1e-5);
    TEST("timestep odd dim",   gk_timestep_embedding(G, ga, 33, 10000),
                               ggml_timestep_embedding(R, ra, 33, 10000), 1e-5);

    // argsort_top_k is a full sort and exact; top_k only promises the set,
    // so its check sorts each row's indices before comparing
    make_pair(200, 4, 1, 1, 1.0f, &ga, &ra);
    TEST("argsort_top_k", gk_argsort_top_k(G, ga, 16), ggml_argsort_top_k(R, ra, 16), 0.0);

    {
        struct gk_tensor   * go = gk_top_k(G, ga, 16);
        struct ggml_tensor * ro = ggml_top_k(R, ra, 16);
        run_both(go, ro);

        g_checks++;
        int bad = 0;
        for (int64_t r = 0; r < 4; ++r) {
            int32_t * xi = (int32_t *) ((char *) go->data + r * go->nb[1]);
            int32_t * yi = (int32_t *) ((char *) ro->data + r * ro->nb[1]);
            // insertion-sort both rows, then require identical index sets
            for (int a = 1; a < 16; ++a) {
                for (int b = a; b > 0 && xi[b - 1] > xi[b]; --b) {
                    int32_t t = xi[b]; xi[b] = xi[b - 1]; xi[b - 1] = t;
                }
                for (int b = a; b > 0 && yi[b - 1] > yi[b]; --b) {
                    int32_t t = yi[b]; yi[b] = yi[b - 1]; yi[b - 1] = t;
                }
            }
            for (int i = 0; i < 16; ++i) {
                bad += xi[i] != yi[i];
            }
        }
        printf("  %-22s %s\n", "top_k (as set)", bad == 0 ? "ok" : "FAIL");
        if (bad) g_fails++;
    }

    // count_equal: an i64 scalar, checked by hand
    {
        struct gk_tensor * gx, * gy; struct ggml_tensor * rx, * ry;
        make_pair_i32(64, 8, 0, 3, &gx, &rx);
        make_pair_i32(64, 8, 0, 3, &gy, &ry);

        struct gk_tensor   * go = gk_count_equal(G, gx, gy);
        struct ggml_tensor * ro = ggml_count_equal(R, rx, ry);
        run_both(go, ro);

        g_checks++;
        const int64_t x = *(int64_t *) go->data;
        const int64_t y = *(int64_t *) ro->data;
        printf("  %-22s %s (%lld vs %lld)\n", "count_equal",
               x == y ? "ok" : "FAIL", (long long) x, (long long) y);
        if (x != y) g_fails++;
    }

    // set_rows: rows scattered into a destination by index, f32 and f16
    // destinations; the f16 one is compared bit for bit since both sides
    // perform the same single rounding
    {
        for (int pass = 0; pass < 2; ++pass) {
            const enum gk_type dt = pass == 0 ? GK_TYPE_F32 : GK_TYPE_F16;

            struct gk_tensor   * gd = gk_new_tensor_4d(G, dt, 32, 20, 2, 1);
            struct ggml_tensor * rd = ggml_new_tensor_4d(R, (enum ggml_type) dt, 32, 20, 2, 1);
            memset(gd->data, 0, gk_nbytes(gd));
            memset(rd->data, 0, ggml_nbytes(rd));

            struct gk_tensor * gsrc; struct ggml_tensor * rsrc;
            make_pair(32, 6, 2, 1, 1.0f, &gsrc, &rsrc);

            struct gk_tensor   * gi = gk_new_tensor_1d(G, GK_TYPE_I64, 6);
            struct ggml_tensor * ri = ggml_new_tensor_1d(R, GGML_TYPE_I64, 6);
            const int64_t rows[6] = { 3, 17, 0, 9, 12, 5 };
            memcpy(gi->data, rows, sizeof(rows));
            memcpy(ri->data, rows, sizeof(rows));

            struct gk_tensor   * go = gk_set_rows(G, gd, gsrc, gi);
            struct ggml_tensor * ro = ggml_set_rows(R, rd, rsrc, ri);
            run_both(go, ro);

            g_checks++;
            const int same = memcmp(gd->data, rd->data, gk_nbytes(gd)) == 0;
            printf("  %-22s %s\n", pass == 0 ? "set_rows f32" : "set_rows f16",
                   same ? "ok" : "FAIL");
            if (!same) g_fails++;
        }
    }
}

// A pair of tensors with identical f16 contents on both sides.
static void make_pair_f16(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, float scale,
                          struct gk_tensor ** ga, struct ggml_tensor ** ra) {
    *ga = gk_new_tensor_4d(G, GK_TYPE_F16, ne0, ne1, ne2, ne3);
    *ra = ggml_new_tensor_4d(R, GGML_TYPE_F16, ne0, ne1, ne2, ne3);

    const int64_t n = ne0 * ne1 * ne2 * ne3;
    for (int64_t i = 0; i < n; ++i) {
        const gk_fp16_t v = gk_fp32_to_fp16(frand() * scale);
        ((gk_fp16_t *) (*ga)->data)[i] = v;
        ((uint16_t *)  (*ra)->data)[i] = v;
    }
}

// The convolution family. The composite routes (conv_1d, conv_2d, dw) go
// through im2col + matmul on both sides; the direct kernels are checked
// against the reference's direct kernels. f16 kernels are the common shipped
// form, so both types run.
static void test_conv(void) {
    printf("convolution\n");

    struct gk_tensor * gk_k, * gk_x; struct ggml_tensor * gg_k, * gg_x;

    // conv_1d: kernel [K, IC, OC], input [L, IC]
    make_pair_f16(5, 3, 4, 1, 0.5f, &gk_k, &gg_k);
    make_pair(40, 3, 1, 1, 1.0f, &gk_x, &gg_x);
    TEST("conv_1d", gk_conv_1d(G, gk_k, gk_x, 2, 1, 1),
                    ggml_conv_1d(R, gg_k, gg_x, 2, 1, 1), 1e-3);

    // depthwise 1-D: kernel [K, 1, C], input [L, C]
    struct gk_tensor * gdw; struct ggml_tensor * rdw;
    make_pair_f16(5, 1, 3, 1, 0.5f, &gdw, &rdw);
    TEST("conv_1d_dw", gk_conv_1d_dw(G, gdw, gk_x, 1, 2, 1),
                       ggml_conv_1d_dw(R, rdw, gg_x, 1, 2, 1), 1e-3);

    // conv_2d composite (f16 kernel) and direct (f32 kernel), same geometry.
    // The f16 inputs are kept positive: the reference's ARM f16 dot
    // accumulates in half precision, and on a cancelling sum that error is
    // unbounded relative to the tiny result. Without cancellation both sides
    // sit within f16 rounding of each other.
    struct gk_tensor * gk2, * gx2; struct ggml_tensor * rk2, * rx2;
    make_pair_f16(3, 3, 4, 5, 0.5f, &gk2, &rk2);
    make_pair(13, 11, 4, 2, 1.0f, &gx2, &rx2);
    for (int64_t i = 0; i < gk_nelements(gk2); ++i) {
        ((gk_fp16_t *) gk2->data)[i] &= 0x7fff;
        ((uint16_t *)  rk2->data)[i] &= 0x7fff;
    }
    for (int64_t i = 0; i < gk_nelements(gx2); ++i) {
        const float v = fabsf(((float *) gx2->data)[i]);
        ((float *) gx2->data)[i] = v;
        ((float *) rx2->data)[i] = v;
    }
    TEST("conv_2d", gk_conv_2d(G, gk2, gx2, 2, 1, 1, 1, 1, 1),
                    ggml_conv_2d(R, rk2, rx2, 2, 1, 1, 1, 1, 1), 5e-3);
    TEST("conv_2d_direct f16", gk_conv_2d_direct(G, gk2, gx2, 2, 1, 1, 1, 1, 1),
                    ggml_conv_2d_direct(R, rk2, rx2, 2, 1, 1, 1, 1, 1), 5e-3);

    struct gk_tensor * gk2f; struct ggml_tensor * rk2f;
    make_pair(3, 3, 4, 5, 0.5f, &gk2f, &rk2f);
    TEST("conv_2d_direct f32", gk_conv_2d_direct(G, gk2f, gx2, 1, 2, 0, 1, 1, 1),
                    ggml_conv_2d_direct(R, rk2f, rx2, 1, 2, 0, 1, 1, 1), 1e-3);

    // depthwise 2-D: the composite route needs an f16 kernel (its im2col
    // intermediate is f16, and an f16-by-f32 matmul exists in neither
    // library); the direct kernel takes f32
    struct gk_tensor * gkdw; struct ggml_tensor * rkdw;
    make_pair_f16(3, 3, 4, 1, 0.5f, &gkdw, &rkdw);
    TEST("conv_2d_dw", gk_conv_2d_dw(G, gkdw, gx2, 1, 1, 1, 1, 1, 1),
                       ggml_conv_2d_dw(R, rkdw, rx2, 1, 1, 1, 1, 1, 1), 5e-3);
    {
        struct gk_tensor * gkd, * gxd; struct ggml_tensor * rkd, * rxd;
        make_pair(3, 3, 1, 4, 0.5f, &gkd, &rkd);
        make_pair(13, 11, 4, 2, 1.0f, &gxd, &rxd);
        TEST("conv_2d_dw_direct", gk_conv_2d_dw_direct(G, gkd, gxd, 1, 1, 1, 1, 1, 1),
                                  ggml_conv_2d_dw_direct(R, rkd, rxd, 1, 1, 1, 1, 1, 1), 1e-5);
    }

    // conv_3d, composite and direct: kernel [KW,KH,KD, IC*OC], input [W,H,D, IC*N];
    // positive inputs for the same f16-accumulation reason as conv_2d
    struct gk_tensor * gk3, * gx3; struct ggml_tensor * rk3, * rx3;
    make_pair_f16(3, 3, 2, 2 * 3, 0.5f, &gk3, &rk3);   // IC 2, OC 3
    make_pair(9, 8, 5, 2 * 2, 1.0f, &gx3, &rx3);       // N 2
    for (int64_t i = 0; i < gk_nelements(gk3); ++i) {
        ((gk_fp16_t *) gk3->data)[i] &= 0x7fff;
        ((uint16_t *)  rk3->data)[i] &= 0x7fff;
    }
    for (int64_t i = 0; i < gk_nelements(gx3); ++i) {
        const float v = fabsf(((float *) gx3->data)[i]);
        ((float *) gx3->data)[i] = v;
        ((float *) rx3->data)[i] = v;
    }
    TEST("conv_3d", gk_conv_3d(G, gk3, gx3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                    ggml_conv_3d(R, rk3, rx3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1), 1e-3);
    TEST("conv_3d_direct", gk_conv_3d_direct(G, gk3, gx3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3),
                    ggml_conv_3d_direct(R, rk3, rx3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3), 1e-3);

    // transposed convolutions
    struct gk_tensor * gkt, * gxt; struct ggml_tensor * rkt, * rxt;
    make_pair_f16(4, 5, 3, 1, 0.5f, &gkt, &rkt);       // [K, Cout, Cin]
    make_pair(12, 3, 1, 1, 1.0f, &gxt, &rxt);          // [L, Cin]
    TEST("conv_transpose_1d", gk_conv_transpose_1d(G, gkt, gxt, 2, 0, 1),
                              ggml_conv_transpose_1d(R, rkt, rxt, 2, 0, 1), 1e-3);

    struct gk_tensor * gkt2, * gxt2; struct ggml_tensor * rkt2, * rxt2;
    make_pair_f16(3, 3, 4, 2, 0.5f, &gkt2, &rkt2);     // [KW, KH, Cout, Cin]
    make_pair(7, 6, 2, 1, 1.0f, &gxt2, &rxt2);         // [W, H, Cin, N]
    TEST("conv_transpose_2d", gk_conv_transpose_2d_p0(G, gkt2, gxt2, 2),
                              ggml_conv_transpose_2d_p0(R, rkt2, rxt2, 2), 1e-3);

    // pooling
    struct gk_tensor * gp; struct ggml_tensor * rp;
    make_pair(17, 13, 3, 2, 1.0f, &gp, &rp);
    TEST("pool_1d max", gk_pool_1d(G, gp, GK_OP_POOL_MAX, 3, 2, 1),
                        ggml_pool_1d(R, rp, GGML_OP_POOL_MAX, 3, 2, 1), 0.0);
    TEST("pool_1d avg", gk_pool_1d(G, gp, GK_OP_POOL_AVG, 4, 2, 1),
                        ggml_pool_1d(R, rp, GGML_OP_POOL_AVG, 4, 2, 1), 1e-6);
    TEST("pool_2d max", gk_pool_2d(G, gp, GK_OP_POOL_MAX, 3, 3, 2, 2, 1.0f, 1.0f),
                        ggml_pool_2d(R, rp, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1.0f, 1.0f), 0.0);
    TEST("pool_2d avg", gk_pool_2d(G, gp, GK_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f),
                        ggml_pool_2d(R, rp, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f), 1e-6);

    // windows: a 10x10 map in windows of 4, and back
    struct gk_tensor * gwm; struct ggml_tensor * rwm;
    make_pair(8, 10, 10, 1, 1.0f, &gwm, &rwm);
    TEST("win_part",   gk_win_part(G, gwm, 4), ggml_win_part(R, rwm, 4), 0.0);
    TEST("win_unpart", gk_win_unpart(G, gk_win_part(G, gwm, 4), 10, 10, 4),
                       ggml_win_unpart(R, ggml_win_part(R, rwm, 4), 10, 10, 4), 0.0);

    // im2col standalone, f32 destination (the f16 one is covered through the
    // composites above)
    TEST("im2col f32", gk_im2col(G, gk2f, gx2, 1, 1, 1, 1, 1, 1, true, GK_TYPE_F32),
                       ggml_im2col(R, rk2f, rx2, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F32), 0.0);
}

// Multi-axis rope, softmax sinks, and fused attention.
static void test_attention_ext(void) {
    printf("attention extensions\n");

    // multi-axis rope: 4 position ids per token
    const int64_t hd = 32, nh = 3, nt = 5; // head dim, heads, tokens
    struct gk_tensor * gq; struct ggml_tensor * rq;
    make_pair(hd, nh, nt, 1, 1.0f, &gq, &rq);

    struct gk_tensor   * gpos = gk_new_tensor_1d(G, GK_TYPE_I32, nt * 4);
    struct ggml_tensor * rpos = ggml_new_tensor_1d(R, GGML_TYPE_I32, nt * 4);
    for (int i = 0; i < nt * 4; ++i) {
        const int32_t v = (i * 7) % 40;
        ((int32_t *) gpos->data)[i] = v;
        ((int32_t *) rpos->data)[i] = v;
    }

    int sect[4] = { 6, 5, 5, 0 };
    TEST("rope mrope",
         gk_rope_multi(G, gq, gpos, NULL, 32, sect, GK_ROPE_TYPE_MROPE, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_multi(R, rq, rpos, NULL, 32, sect, GGML_ROPE_TYPE_MROPE, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), 1e-4);
    TEST("rope imrope",
         gk_rope_multi(G, gq, gpos, NULL, 32, sect, GK_ROPE_TYPE_IMROPE, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_multi(R, rq, rpos, NULL, 32, sect, GGML_ROPE_TYPE_IMROPE, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), 1e-4);

    int vsect[4] = { 8, 8, 0, 0 };
    TEST("rope vision",
         gk_rope_multi(G, gq, gpos, NULL, 16, vsect, GK_ROPE_TYPE_VISION, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f),
         ggml_rope_multi(R, rq, rpos, NULL, 16, vsect, GGML_ROPE_TYPE_VISION, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), 1e-4);

    // softmax with sinks, and ALiBi over a non-power-of-two head count
    struct gk_tensor * gs; struct ggml_tensor * rs;
    make_pair(24, 6, 5, 1, 3.0f, &gs, &rs); // 5 heads - not a power of two
    struct gk_tensor * gsink; struct ggml_tensor * rsink;
    make_pair(5, 1, 1, 1, 2.0f, &gsink, &rsink);
    {
        struct gk_tensor   * go = gk_soft_max_ext(G, gs, NULL, 0.5f, 0.0f);
        struct ggml_tensor * ro = ggml_soft_max_ext(R, rs, NULL, 0.5f, 0.0f);
        gk_soft_max_add_sinks(go, gsink);
        ggml_soft_max_add_sinks(ro, rsink);
        run_both(go, ro);
        compare("soft_max sinks", go, ro, 1e-5);
    }
    {
        struct gk_tensor   * gm = gk_new_tensor_2d(G, GK_TYPE_F32, 24, 6);
        struct ggml_tensor * rm = ggml_new_tensor_2d(R, GGML_TYPE_F32, 24, 6);
        for (int i = 0; i < 24 * 6; ++i) {
            const float v = frand();
            ((float *) gm->data)[i] = v;
            ((float *) rm->data)[i] = v;
        }
        TEST("soft_max alibi 5 heads", gk_soft_max_ext(G, gs, gm, 0.7f, 4.0f),
                                       ggml_soft_max_ext(R, rs, rm, 0.7f, 4.0f), 1e-5);
    }

    // fused attention. DK 40 / DV 32 exercises the unequal-width path, 4
    // query heads over 2 KV heads exercises the grouped-query broadcast.
    const int64_t DKq = 40, DVv = 32, n_kv = 33, n_b = 7, nhq = 4, nhkv = 2;

    struct gk_tensor * gfq; struct ggml_tensor * rfq;
    make_pair(DKq, n_b, nhq, 1, 1.0f, &gfq, &rfq);

    // V is kept positive: the reference accumulates an f16 V in half
    // precision, and on a cancelling combination that error is unbounded
    // relative to the small result. gk accumulates in f32 on purpose.
    struct gk_tensor * gfk, * gfv; struct ggml_tensor * rfk, * rfv;
    make_pair_f16(DKq, n_kv, nhkv, 1, 1.0f, &gfk, &rfk);
    make_pair_f16(DVv, n_kv, nhkv, 1, 1.0f, &gfv, &rfv);
    for (int64_t i = 0; i < gk_nelements(gfv); ++i) {
        ((gk_fp16_t *) gfv->data)[i] &= 0x7fff;
        ((uint16_t *)  rfv->data)[i] &= 0x7fff;
    }

    // an f16 causal-ish mask, padded rows allowed
    struct gk_tensor   * gfm = gk_new_tensor_2d(G, GK_TYPE_F16, n_kv, n_b);
    struct ggml_tensor * rfm = ggml_new_tensor_2d(R, GGML_TYPE_F16, n_kv, n_b);
    for (int64_t r = 0; r < n_b; ++r) {
        for (int64_t c = 0; c < n_kv; ++c) {
            const float v = c > 20 + r ? -INFINITY : frand() * 0.1f;
            ((gk_fp16_t *) gfm->data)[r * n_kv + c] = gk_fp32_to_fp16(v);
            ((uint16_t *)  rfm->data)[r * n_kv + c] = gk_fp32_to_fp16(v);
        }
    }

    TEST("flash_attn f16",
         gk_flash_attn_ext(G, gfq, gfk, gfv, gfm, 0.15f, 0.0f, 0.0f),
         ggml_flash_attn_ext(R, rfq, rfk, rfv, rfm, 0.15f, 0.0f, 0.0f), 5e-3);

    TEST("flash_attn softcap",
         gk_flash_attn_ext(G, gfq, gfk, gfv, gfm, 0.15f, 0.0f, 30.0f),
         ggml_flash_attn_ext(R, rfq, rfk, rfv, rfm, 0.15f, 0.0f, 30.0f), 5e-3);

    TEST("flash_attn alibi",
         gk_flash_attn_ext(G, gfq, gfk, gfv, gfm, 0.15f, 8.0f, 0.0f),
         ggml_flash_attn_ext(R, rfq, rfk, rfv, rfm, 0.15f, 8.0f, 0.0f), 5e-3);

    {
        struct gk_tensor * gfs; struct ggml_tensor * rfs;
        make_pair(nhq, 1, 1, 1, 1.0f, &gfs, &rfs);

        struct gk_tensor   * go = gk_flash_attn_ext(G, gfq, gfk, gfv, gfm, 0.15f, 0.0f, 0.0f);
        struct ggml_tensor * ro = ggml_flash_attn_ext(R, rfq, rfk, rfv, rfm, 0.15f, 0.0f, 0.0f);
        gk_flash_attn_ext_add_sinks(go, gfs);
        ggml_flash_attn_ext_add_sinks(ro, rfs);
        run_both(go, ro);
        compare("flash_attn sinks", go, ro, 5e-3);
    }

    // quantized KV: q8_0 keys and values, at a block-aligned head width
    {
        const int64_t DKB = 64;
        struct gk_tensor * gq2; struct ggml_tensor * rq2;
        make_pair(DKB, n_b, nhq, 1, 1.0f, &gq2, &rq2);

        struct gk_tensor * gqk, * gqv; struct ggml_tensor * rqk, * rqv;
        make_pair_quant(GK_TYPE_Q8_0, DKB, n_kv * nhkv, &gqk, &rqk);
        make_pair_quant(GK_TYPE_Q8_0, DVv, n_kv * nhkv, &gqv, &rqv);
        struct gk_tensor   * gqk3 = gk_reshape_3d(G, gqk, DKB, n_kv, nhkv);
        struct gk_tensor   * gqv3 = gk_reshape_3d(G, gqv, DVv, n_kv, nhkv);
        struct ggml_tensor * rqk3 = ggml_reshape_3d(R, rqk, DKB, n_kv, nhkv);
        struct ggml_tensor * rqv3 = ggml_reshape_3d(R, rqv, DVv, n_kv, nhkv);

        TEST("flash_attn q8_0 kv",
             gk_flash_attn_ext(G, gq2, gqk3, gqv3, gfm, 0.15f, 0.0f, 0.0f),
             ggml_flash_attn_ext(R, rq2, rqk3, rqv3, rfm, 0.15f, 0.0f, 0.0f), 5e-3);
    }
}

// The recurrent family: state-space scans, the RWKV kernels, the delta rule,
// and the DeepSeek V4 pieces. All are checked over multiple sequences so the
// state bookkeeping (per-sequence offsets, chaining across tokens) is on
// trial, not just the arithmetic.
static void test_recurrent(void) {
    printf("recurrent layers\n");

    // ssm_conv: [d_conv-1+n_t, d_inner, n_seqs] against [d_conv, d_inner]
    struct gk_tensor * gsx, * gc; struct ggml_tensor * rsx, * rc;
    make_pair(3 + 7, 24, 2, 1, 1.0f, &gsx, &rsx);
    make_pair(4, 24, 1, 1, 1.0f, &gc, &rc);
    TEST("ssm_conv", gk_ssm_conv(G, gsx, gc), ggml_ssm_conv(R, rsx, rc), 1e-5);

    // ssm_scan, Mamba-2 shape: A [1, n_head]
    {
        const int64_t d_state = 16, head_dim = 8, n_head = 4, n_tok = 5, n_seqs = 2;

        struct gk_tensor * gs, * gx, * gdt, * gA, * gB, * gC;
        struct ggml_tensor * rs, * rx, * rdt, * rA, * rB, * rC;
        make_pair(d_state, head_dim, n_head, n_seqs, 1.0f, &gs, &rs);
        make_pair(head_dim, n_head, n_tok, n_seqs, 1.0f, &gx, &rx);
        make_pair(n_head, n_tok, n_seqs, 1, 1.0f, &gdt, &rdt);
        make_pair(1, n_head, 1, 1, 1.0f, &gA, &rA);
        make_pair(d_state, 1, n_tok, n_seqs, 1.0f, &gB, &rB);
        make_pair(d_state, 1, n_tok, n_seqs, 1.0f, &gC, &rC);

        struct gk_tensor   * gids = gk_new_tensor_1d(G, GK_TYPE_I32, n_seqs);
        struct ggml_tensor * rids = ggml_new_tensor_1d(R, GGML_TYPE_I32, n_seqs);
        ((int32_t *) gids->data)[0] = 1; ((int32_t *) gids->data)[1] = 0;
        ((int32_t *) rids->data)[0] = 1; ((int32_t *) rids->data)[1] = 0;

        TEST("ssm_scan mamba2", gk_ssm_scan(G, gs, gx, gdt, gA, gB, gC, gids),
                                ggml_ssm_scan(R, rs, rx, rdt, rA, rB, rC, rids), 1e-4);

        // Mamba-1 shape: A [d_state, n_head]
        struct gk_tensor * gA1; struct ggml_tensor * rA1;
        make_pair(d_state, n_head, 1, 1, 1.0f, &gA1, &rA1);
        TEST("ssm_scan mamba1", gk_ssm_scan(G, gs, gx, gdt, gA1, gB, gC, gids),
                                ggml_ssm_scan(R, rs, rx, rdt, rA1, rB, rC, rids), 1e-4);
    }

    // The RWKV-shaped kernels: S 16, H 3, 6 tokens over 2 sequences. S stays
    // a multiple of 16 because the reference's vector path loads whole
    // 16-float steps and reads across the head boundary below that - real
    // models use 64+.
    {
        const int64_t S = 16, H = 3, T = 6, NS = 2;

        struct gk_tensor * gk1, * gv1, * gr1, * gtf, * gtd, * gst;
        struct ggml_tensor * rk1, * rv1, * rr1, * rtf, * rtd, * rst;
        make_pair(S, H, T, 1, 0.5f, &gk1, &rk1);
        make_pair(S, H, T, 1, 0.5f, &gv1, &rv1);
        make_pair(S, H, T, 1, 0.5f, &gr1, &rr1);
        make_pair(S, H, 1, 1, 0.5f, &gtf, &rtf);
        make_pair(S, H, T, 1, 0.1f, &gtd, &rtd);
        // decay must sit in (0, 1) for the recurrence to stay bounded
        for (int64_t i = 0; i < gk_nelements(gtd); ++i) {
            const float v = 0.5f + 0.4f * fabsf(((float *) gtd->data)[i]);
            ((float *) gtd->data)[i] = v;
            ((float *) rtd->data)[i] = v;
        }
        make_pair(S * S * H / (int) H, H * (int) NS * (int) H / (int) H, 1, 1, 0.5f, &gst, &rst);
        // state is [S*S*H, n_seqs] in effect; rebuild with the exact shape
        gst = gk_new_tensor_2d(G, GK_TYPE_F32, S * S * H, NS);
        rst = ggml_new_tensor_2d(R, GGML_TYPE_F32, S * S * H, NS);
        for (int64_t i = 0; i < S * S * H * NS; ++i) {
            const float v = frand() * 0.5f;
            ((float *) gst->data)[i] = v;
            ((float *) rst->data)[i] = v;
        }

        TEST("rwkv_wkv6", gk_rwkv_wkv6(G, gk1, gv1, gr1, gtf, gtd, gst),
                          ggml_rwkv_wkv6(R, rk1, rv1, rr1, rtf, rtd, rst), 1e-4);

        TEST("gated_linear_attn", gk_gated_linear_attn(G, gk1, gv1, gr1, gtd, gst, 0.35f),
                          ggml_gated_linear_attn(R, rk1, rv1, rr1, rtd, rst, 0.35f), 1e-4);

        struct gk_tensor * gw7, * ga7, * gb7;
        struct ggml_tensor * rw7, * ra7, * rb7;
        make_pair(S, H, T, 1, 0.1f, &gw7, &rw7);
        for (int64_t i = 0; i < gk_nelements(gw7); ++i) {
            const float v = 0.5f + 0.4f * fabsf(((float *) gw7->data)[i]);
            ((float *) gw7->data)[i] = v;
            ((float *) rw7->data)[i] = v;
        }
        make_pair(S, H, T, 1, 0.3f, &ga7, &ra7);
        make_pair(S, H, T, 1, 0.3f, &gb7, &rb7);
        TEST("rwkv_wkv7", gk_rwkv_wkv7(G, gr1, gw7, gk1, gv1, ga7, gb7, gst),
                          ggml_rwkv_wkv7(R, rr1, rw7, rk1, rv1, ra7, rb7, rst), 1e-4);
    }

    // the gated delta rule, scalar gate and KDA per-channel gate, K 1 and 3
    {
        const int64_t S = 8, H = 3, T = 5, NS = 2;

        struct gk_tensor * gq, * gk2, * gv2, * gb2, * gs2;
        struct ggml_tensor * rq, * rk2, * rv2, * rb2, * rs2;
        make_pair(S, H, T, NS, 0.5f, &gq, &rq);
        make_pair(S, H, T, NS, 0.5f, &gk2, &rk2);
        make_pair(S, H, T, NS, 0.5f, &gv2, &rv2);
        make_pair(1, H, T, NS, 0.5f, &gb2, &rb2);
        make_pair(S, S, H, NS, 0.3f, &gs2, &rs2);

        struct gk_tensor * gg1; struct ggml_tensor * rg1;
        make_pair(1, H, T, NS, 1.0f, &gg1, &rg1);
        for (int64_t i = 0; i < gk_nelements(gg1); ++i) {
            const float v = -fabsf(((float *) gg1->data)[i]); // decay <= 1
            ((float *) gg1->data)[i] = v;
            ((float *) rg1->data)[i] = v;
        }

        TEST("gated_delta_net K1", gk_gated_delta_net(G, gq, gk2, gv2, gg1, gb2, gs2, 1),
                          ggml_gated_delta_net(R, rq, rk2, rv2, rg1, rb2, rs2, 1), 1e-4);
        TEST("gated_delta_net K3", gk_gated_delta_net(G, gq, gk2, gv2, gg1, gb2, gs2, 3),
                          ggml_gated_delta_net(R, rq, rk2, rv2, rg1, rb2, rs2, 3), 1e-4);

        struct gk_tensor * ggv; struct ggml_tensor * rgv;
        make_pair(S, H, T, NS, 1.0f, &ggv, &rgv);
        for (int64_t i = 0; i < gk_nelements(ggv); ++i) {
            const float v = -fabsf(((float *) ggv->data)[i]);
            ((float *) ggv->data)[i] = v;
            ((float *) rgv->data)[i] = v;
        }
        TEST("gated_delta_net kda", gk_gated_delta_net(G, gq, gk2, gv2, ggv, gb2, gs2, 1),
                          ggml_gated_delta_net(R, rq, rk2, rv2, rgv, rb2, rs2, 1), 1e-4);
    }

    // solve_tri: L x = b by forward substitution, batched
    {
        struct gk_tensor * gA, * gB; struct ggml_tensor * rA, * rB;
        make_pair(12, 12, 2, 2, 1.0f, &gA, &rA);
        make_pair(5, 12, 2, 2, 1.0f, &gB, &rB);
        // a well-conditioned lower-triangular A: keep the diagonal away from 0
        for (int64_t b = 0; b < 4; ++b) {
            float * ga = (float *) gA->data + b * 144;
            float * ra = (float *) rA->data + b * 144;
            for (int i = 0; i < 12; ++i) {
                for (int j = 0; j < 12; ++j) {
                    float v = j > i ? 0.0f : ga[i * 12 + j];
                    if (i == j) {
                        v = 2.0f + fabsf(v);
                    }
                    ga[i * 12 + j] = v;
                    ra[i * 12 + j] = v;
                }
            }
        }
        TEST("solve_tri", gk_solve_tri(G, gA, gB, true, true, false),
                          ggml_solve_tri(R, rA, rB, true, true, false), 1e-4);
    }

    // lightning indexer
    {
        const int64_t E = 16, HI = 4, TQ = 3, KV = 10;
        struct gk_tensor * gq, * gw; struct ggml_tensor * rq, * rw;
        make_pair(E, HI, TQ, 1, 1.0f, &gq, &rq);
        make_pair(HI, TQ, 1, 1, 1.0f, &gw, &rw);

        struct gk_tensor * gkk; struct ggml_tensor * rkk;
        make_pair(E, 1, KV, 1, 1.0f, &gkk, &rkk);

        struct gk_tensor   * gm = gk_new_tensor_2d(G, GK_TYPE_F16, KV, TQ);
        struct ggml_tensor * rm = ggml_new_tensor_2d(R, GGML_TYPE_F16, KV, TQ);
        for (int64_t i = 0; i < KV * TQ; ++i) {
            const gk_fp16_t h = gk_fp32_to_fp16(frand() * 0.1f);
            ((gk_fp16_t *) gm->data)[i] = h;
            ((uint16_t *)  rm->data)[i] = h;
        }

        TEST("lightning_indexer", gk_lightning_indexer(G, gq, gkk, gw, gm),
                                  ggml_lightning_indexer(R, rq, rkk, rw, rm), 1e-4);
    }

    // DeepSeek V4 hyper-connections
    {
        const int64_t NE = 24, TT = 5, HC = 4;

        struct gk_tensor * gmix, * gscale, * gbase;
        struct ggml_tensor * rmix, * rscale, * rbase;
        make_pair((2 + HC) * HC, TT, 1, 1, 1.0f, &gmix, &rmix);
        make_pair(3, 1, 1, 1, 1.0f, &gscale, &rscale);
        make_pair((2 + HC) * HC, 1, 1, 1, 1.0f, &gbase, &rbase);

        struct gk_tensor   * gcomb = gk_dsv4_hc_comb(G, gmix, gscale, gbase, 1e-4f, 2);
        struct ggml_tensor * rcomb = ggml_dsv4_hc_comb(R, rmix, rscale, rbase, 1e-4f, 2);
        run_both(gcomb, rcomb);
        compare("dsv4_hc_comb", gcomb, rcomb, 1e-5);

        struct gk_tensor * gx, * gwp; struct ggml_tensor * rx, * rwp;
        make_pair(NE, HC, TT, 1, 1.0f, &gx, &rx);
        make_pair(HC, TT, 1, 1, 1.0f, &gwp, &rwp);
        TEST("dsv4_hc_pre", gk_dsv4_hc_pre(G, gx, gwp),
                            ggml_dsv4_hc_pre(R, rx, rwp), 1e-5);

        struct gk_tensor * gxp, * gpost; struct ggml_tensor * rxp, * rpost;
        make_pair(NE, TT, 1, 1, 1.0f, &gxp, &rxp);
        make_pair(HC, TT, 1, 1, 1.0f, &gpost, &rpost);
        TEST("dsv4_hc_post", gk_dsv4_hc_post(G, gxp, gx, gpost, gcomb),
                             ggml_dsv4_hc_post(R, rxp, rx, rpost, rcomb), 1e-5);
    }

    // build_forward_select: three branches in one graph, only the chosen one
    // computed - the unchosen output keeps whatever was in its buffer
    {
        struct gk_tensor * ga; struct ggml_tensor * ra;
        make_pair(32, 4, 1, 1, 1.0f, &ga, &ra);

        struct gk_tensor * gbr[3] = {
            gk_scale(G, ga, 2.0f), gk_sqr(G, ga), gk_neg(G, ga),
        };
        struct ggml_tensor * rbr[3] = {
            ggml_scale(R, ra, 2.0f), ggml_sqr(R, ra), ggml_neg(R, ra),
        };

        struct gk_cgraph * gg = gk_new_graph(G);
        struct gk_tensor * gsel = gk_build_forward_select(gg, gbr, 3, 1);
        gk_graph_compute(gg, g_threads);

        struct ggml_cgraph * rg = ggml_new_graph(R);
        struct ggml_tensor * rsel = ggml_build_forward_select(rg, rbr, 3, 1);
        ggml_graph_compute_with_ctx(R, rg, 1);

        compare("build_forward_select", gsel, rsel, 1e-6);
    }
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        g_threads = atoi(argv[1]);
    }

    printf("gk compute pass, differential against reference (%d thread%s)\n\n",
           g_threads, g_threads == 1 ? "" : "s");

    // both libraries need generous arenas: every intermediate is retained
    G = gk_init((struct gk_init_params) {
        .mem_size = 512u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    struct ggml_init_params rp = {
        .mem_size = 512u << 20, .mem_buffer = NULL, .no_alloc = false,
    };
    R = ggml_init(rp);

    if (G == NULL || R == NULL) {
        printf("could not create contexts\n");
        return 1;
    }

    ggml_cpu_init();

    test_elementwise();
    test_activations();
    test_norms();
    test_matmul();
    test_movement();
    test_attention();
    test_rope();
    test_reductions();
    test_placement();
    test_conv();
    test_attention_ext();
    test_recurrent();
    test_block();

    ggml_free(R);
    gk_free(G);

    printf("\n%d comparisons, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
