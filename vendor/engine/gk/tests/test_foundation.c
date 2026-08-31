// Invariants the rest of gk is allowed to assume.
//
// These are the properties every kernel written later will lean on without
// re-checking: that a row of a quantized type is a whole number of blocks,
// that default strides really are contiguous, that a view of a view collapses,
// and that a graph comes out of the builder in dependency order. They are
// cheap and they fail loudly, which is what makes the op work that follows
// safe to do quickly.

#include "gk_impl.h"

#include "qz_codebook.h"
#include "qz_fp.h"
#include "qz_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fails = 0;
static int g_checks = 0;

#define CHECK(cond) \
    do { \
        g_checks++; \
        if (!(cond)) { \
            g_fails++; \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

#define CHECK_MSG(cond, ...) \
    do { \
        g_checks++; \
        if (!(cond)) { \
            g_fails++; \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__); \
            printf("\n"); \
        } \
    } while (0)

// --------------------------------------------------------------------------

static void test_type_geometry(void) {
    printf("type geometry\n");

    for (int i = 0; i < GK_TYPE_COUNT; ++i) {
        const enum gk_type t = (enum gk_type) i;
        const struct gk_type_traits * tr = gk_get_type_traits(t);

        if (tr->name == NULL) {
            continue; // an id the format defines but this library does not carry
        }

        const int64_t blck = gk_blck_size(t);
        const size_t  size = gk_type_size(t);

        CHECK_MSG(blck > 0, "%s: block size %lld", gk_type_name(t), (long long) blck);
        CHECK_MSG(size > 0, "%s: type size %zu",   gk_type_name(t), size);

        // A block must not encode to more bytes than it holds elements at f32,
        // or it is not a compression of anything.
        if (tr->is_quantized) {
            CHECK_MSG(size < (size_t) blck * sizeof(float),
                      "%s: %zu bytes for %lld elements is not a saving",
                      gk_type_name(t), size, (long long) blck);
        } else {
            CHECK_MSG(blck == 1, "%s: unquantized types encode one element per block",
                      gk_type_name(t));
        }

        // row_size has to agree with the block geometry for every legal length
        for (int64_t n = blck; n <= blck * 8; n += blck) {
            CHECK_MSG(gk_row_size(t, n) == size * (size_t) (n / blck),
                      "%s: row_size(%lld) disagrees with block geometry",
                      gk_type_name(t), (long long) n);
        }
    }
}

static void test_fp_roundtrip(void) {
    printf("narrow float round trips\n");

    // Values that exercise the interesting parts of the encodings: zero, the
    // subnormal range, the rounding boundary, and the overflow edge.
    static const float xs[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f,
        6.103515625e-05f,   // smallest normal half
        5.960464477539063e-08f, // smallest subnormal half
        3.14159265f, 1e-8f, 1e8f,
    };

    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); ++i) {
        const float x = xs[i];

        const float h = gk_fp16_to_fp32(gk_fp32_to_fp16(x));
        const float b = gk_bf16_to_fp32(gk_fp32_to_bf16(x));

        // f16 carries 11 significand bits, bf16 carries 8; anything inside
        // their range should come back within one ulp of that.
        if (x != 0.0f && x >= -65504.0f && x <= 65504.0f && x > 1e-7f) {
            CHECK_MSG(fabsf(h - x) <= fabsf(x) * 1e-3f,
                      "f16 round trip of %g gave %g", (double) x, (double) h);
        }
        if (x != 0.0f) {
            CHECK_MSG(fabsf(b - x) <= fabsf(x) * 1e-2f,
                      "bf16 round trip of %g gave %g", (double) x, (double) b);
        }
    }

    // bf16 is defined as the top half of the f32 bit pattern, so a value whose
    // low half is already zero must survive exactly.
    union { uint32_t u; float f; } v;
    v.u = 0x3f800000u; // 1.0
    CHECK(gk_bf16_to_fp32(gk_fp32_to_bf16(v.f)) == v.f);
}

static void test_context_and_strides(void) {
    printf("context, tensors and strides\n");

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false,
    });
    CHECK(ctx != NULL);

    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 32, 8);
    CHECK(a != NULL);
    CHECK(a->ne[0] == 32 && a->ne[1] == 8 && a->ne[2] == 1 && a->ne[3] == 1);
    CHECK(gk_nelements(a) == 32 * 8);
    CHECK(gk_nrows(a) == 8);
    CHECK(gk_n_dims(a) == 2);
    CHECK(gk_is_matrix(a));
    CHECK(!gk_is_vector(a));

    // default strides describe a contiguous tensor
    CHECK(a->nb[0] == sizeof(float));
    CHECK(a->nb[1] == 32 * sizeof(float));
    CHECK(a->nb[2] == 32 * 8 * sizeof(float));
    CHECK(gk_is_contiguous(a));
    CHECK(!gk_is_permuted(a));
    CHECK(!gk_is_transposed(a));
    CHECK(gk_nbytes(a) == 32 * 8 * sizeof(float));

    // data was actually carved and is aligned
    CHECK(a->data != NULL);
    CHECK(((uintptr_t) a->data % GK_MEM_ALIGN) == 0);

    // a quantized tensor's first stride is a whole block
    struct gk_tensor * q = gk_new_tensor_2d(ctx, GK_TYPE_Q4_K, 256, 4);
    CHECK(q != NULL);
    CHECK(q->nb[0] == gk_type_size(GK_TYPE_Q4_K));
    CHECK(q->nb[1] == gk_row_size(GK_TYPE_Q4_K, 256));
    CHECK(gk_is_contiguous(q));
    CHECK(gk_nbytes(q) == gk_row_size(GK_TYPE_Q4_K, 256) * 4);

    // views share storage and collapse rather than chaining
    struct gk_tensor * v1 = gk_view_tensor(ctx, a);
    CHECK(v1->view_src == a);
    CHECK(v1->data == a->data);
    CHECK(gk_are_same_shape(v1, a));
    CHECK(gk_are_same_stride(v1, a));

    struct gk_tensor * v2 = gk_view_tensor(ctx, v1);
    CHECK_MSG(v2->view_src == a, "a view of a view must resolve to the owner");

    // no_alloc leaves data unset but still produces correct shapes
    gk_set_no_alloc(ctx, true);
    struct gk_tensor * n = gk_new_tensor_1d(ctx, GK_TYPE_F32, 64);
    CHECK(n != NULL);
    CHECK(n->data == NULL);
    CHECK(gk_nbytes(n) == 64 * sizeof(float));
    gk_set_no_alloc(ctx, false);

    // the arena reports growth and never rewinds
    const size_t used = gk_used_mem(ctx);
    CHECK(used > 0);
    (void) gk_new_tensor_1d(ctx, GK_TYPE_F32, 16);
    CHECK(gk_used_mem(ctx) > used);

    // broadcast compatibility
    struct gk_tensor * small = gk_new_tensor_2d(ctx, GK_TYPE_F32, 32, 1);
    CHECK(gk_can_repeat(small, a));
    CHECK(!gk_can_repeat(a, small));

    gk_free(ctx);
}

static void test_out_of_space(void) {
    printf("arena exhaustion\n");

    // Deliberately tiny: the allocator must return NULL rather than run past
    // the end of the buffer.
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1024, .mem_buffer = NULL, .no_alloc = false,
    });
    CHECK(ctx != NULL);

    struct gk_tensor * big = gk_new_tensor_1d(ctx, GK_TYPE_F32, 1024 * 1024);
    CHECK_MSG(big == NULL, "an oversized request must fail, not overrun");

    gk_free(ctx);
}

// Graphs really do carry tensors with a zero extent - a vision encoder run on
// a batch that turned out to hold no image, a sequence that contributes no
// tokens to a ubatch. The ops that consume them have to become no-ops instead
// of failing, which means the broadcast test has to answer before it divides
// by a zero extent. Getting the two directions backwards aborts a multimodal
// server on its first request, so both are pinned here.
static void test_empty_tensors(void) {
    printf("zero-extent tensors\n");

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 4 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true,
    });
    CHECK(ctx != NULL);

    struct gk_tensor * empty  = gk_new_tensor_2d(ctx, GK_TYPE_F32, 576, 0);
    struct gk_tensor * empty2 = gk_new_tensor_2d(ctx, GK_TYPE_F32, 576, 0);
    struct gk_tensor * weight = gk_new_tensor_2d(ctx, GK_TYPE_F32, 576, 1);
    struct gk_tensor * rows   = gk_new_tensor_2d(ctx, GK_TYPE_F32, 576, 4);

    CHECK(gk_is_empty(empty));
    CHECK(!gk_is_empty(weight));
    CHECK(gk_nelements(empty) == 0);
    CHECK(gk_nbytes(empty) == 0);

    // a normal operand broadcasts onto an empty result: 0 % n == 0
    CHECK_MSG(gk_can_repeat(weight, empty),
              "a weight must broadcast onto an empty result");
    // empty onto empty is the identity case
    CHECK_MSG(gk_can_repeat(empty, empty2), "empty repeats onto empty");
    // an empty source cannot fill a non-empty result - there is nothing to read
    CHECK_MSG(!gk_can_repeat(empty, rows),
              "an empty source must not fill a non-empty result");
    CHECK_MSG(!gk_can_repeat(empty, weight),
              "an empty source must not fill a non-empty result");

    // the ops built on top of it must therefore accept the empty operand
    struct gk_tensor * mul = gk_mul(ctx, empty, weight);
    CHECK(mul != NULL && gk_is_empty(mul));
    struct gk_tensor * add = gk_add(ctx, empty, weight);
    CHECK(add != NULL && gk_is_empty(add));

    gk_free(ctx);
}

// Builds a small diamond so the walk has a shared subexpression to dedup:
//
//     x -> b -> d
//       -> c -> d
//
// d must come last, b and c must both appear exactly once, and x must be a
// leaf rather than a node.
static void test_graph_order(void) {
    printf("graph construction and ordering\n");

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true,
    });
    CHECK(ctx != NULL);

    struct gk_tensor * x = gk_new_tensor_1d(ctx, GK_TYPE_F32, 16);
    gk_set_name(x, "x");

    // The op builders are not written yet, so wire the nodes by hand - the
    // walk only reads op/src, which is exactly what a builder would set.
    struct gk_tensor * b = gk_new_tensor_1d(ctx, GK_TYPE_F32, 16);
    b->op = GK_OP_SQR; b->src[0] = x; gk_set_name(b, "b");

    struct gk_tensor * c = gk_new_tensor_1d(ctx, GK_TYPE_F32, 16);
    c->op = GK_OP_SQRT; c->src[0] = x; gk_set_name(c, "c");

    struct gk_tensor * d = gk_new_tensor_1d(ctx, GK_TYPE_F32, 16);
    d->op = GK_OP_ADD; d->src[0] = b; d->src[1] = c; gk_set_name(d, "d");

    struct gk_cgraph * g = gk_new_graph(ctx);
    CHECK(g != NULL);

    gk_build_forward_expand(g, d);

    CHECK_MSG(gk_graph_n_nodes(g) == 3, "expected 3 nodes, got %d", gk_graph_n_nodes(g));
    CHECK_MSG(g->n_leafs == 1, "expected 1 leaf, got %d", g->n_leafs);
    CHECK(g->leafs[0] == x);

    // the output is last, and every source precedes its consumer
    CHECK(gk_graph_node(g, -1) == d);

    int pos_b = -1, pos_c = -1, pos_d = -1;
    for (int i = 0; i < gk_graph_n_nodes(g); ++i) {
        struct gk_tensor * t = gk_graph_node(g, i);
        if (t == b) pos_b = i;
        if (t == c) pos_c = i;
        if (t == d) pos_d = i;
    }
    CHECK(pos_b >= 0 && pos_c >= 0 && pos_d >= 0);
    CHECK_MSG(pos_b < pos_d && pos_c < pos_d, "sources must precede their consumer");

    // expanding the same output again must not duplicate anything
    gk_build_forward_expand(g, d);
    CHECK_MSG(gk_graph_n_nodes(g) == 3, "re-expanding duplicated nodes");

    CHECK(gk_graph_get_tensor(g, "d") == d);
    CHECK(gk_graph_get_tensor(g, "nope") == NULL);
    CHECK(gk_graph_contains(g, b));

    gk_graph_clear(g);
    CHECK(gk_graph_n_nodes(g) == 0);

    gk_free(ctx);
}

// A long chain is what would blow a recursive walk's stack; 100k links is well
// past any real model and costs nothing to check.
static void test_graph_deep(void) {
    printf("deep graph walk\n");

    const int depth = 100000;

    // The graph's own arrays scale with the node count too, so they have to be
    // in the budget - the tensors alone are not the whole cost.
    const size_t mem_size =
        (size_t) (depth + 16) * (GK_TENSOR_SIZE + GK_OBJECT_SIZE) +
        gk_graph_overhead_custom((size_t) depth + 16, false) +
        (1u << 20);

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = mem_size, .mem_buffer = NULL, .no_alloc = true,
    });
    CHECK(ctx != NULL);
    if (ctx == NULL) {
        return;
    }

    struct gk_tensor * t = gk_new_tensor_1d(ctx, GK_TYPE_F32, 4);
    CHECK(t != NULL);

    for (int i = 0; i < depth; ++i) {
        struct gk_tensor * n = gk_new_tensor_1d(ctx, GK_TYPE_F32, 4);
        if (n == NULL) {
            CHECK_MSG(false, "context exhausted at depth %d", i);
            gk_free(ctx);
            return;
        }
        n->op = GK_OP_SQR;
        n->src[0] = t;
        t = n;
    }

    struct gk_cgraph * g = gk_new_graph_custom(ctx, (size_t) depth + 16, false);
    CHECK(g != NULL);
    if (g == NULL) {
        gk_free(ctx);
        return;
    }

    gk_build_forward_expand(g, t);
    CHECK_MSG(gk_graph_n_nodes(g) == depth, "expected %d nodes, got %d",
              depth, gk_graph_n_nodes(g));
    CHECK(gk_graph_node(g, -1) == t);

    gk_free(ctx);
}

static void test_op_names(void) {
    printf("op names\n");

    for (int i = 0; i < GK_OP_COUNT; ++i) {
        const char * name = gk_op_name((enum gk_op) i);
        CHECK_MSG(strcmp(name, "unknown") != 0, "op %d has no name", i);
    }
    for (int i = 0; i < GK_UNARY_OP_COUNT; ++i) {
        CHECK_MSG(strcmp(gk_unary_op_name((enum gk_unary_op) i), "unknown") != 0,
                  "unary op %d has no name", i);
    }
    for (int i = 0; i < GK_GLU_OP_COUNT; ++i) {
        CHECK_MSG(strcmp(gk_glu_op_name((enum gk_glu_op) i), "unknown") != 0,
                  "glu op %d has no name", i);
    }
}

// --------------------------------------------------------------------------
// Every dot product, against the codec's own decoder
//
// The differential harness checks dots against the reference library, which is
// the strongest evidence available - but only for formats the reference has.
// nvfp4, q1_0 and q2_0 are not in it, so an integer dot for them would
// otherwise ship with no verification at all.
//
// This closes that gap without needing an oracle. Truth is the weight decoded
// through the shared codec and dotted in double; the dot under test is the
// engine's own. The codec decoder is the authority on what the bytes mean and
// is independent of the dot, so a mis-ordered unpack or a wrong offset shows up
// immediately.
//
// The bar depends on which path the format took, and the test derives that from
// the traits rather than being told: a format whose vec_dot_type is not f32
// quantizes the activations to 8 bits, which is a real loss the float path does
// not have.
// --------------------------------------------------------------------------

static uint64_t g_dot_rng = 0x243f6a8885a308d3ull;

static float dot_rand(void) {
    g_dot_rng = g_dot_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float) (uint32_t) (g_dot_rng >> 33) / (float) 0x7fffffffu * 2.0f - 1.0f;
}

static void test_vec_dot_vs_codec(void) {
    printf("every dot product agrees with the codec's decoder\n");

    const int64_t k = 1024;   // a whole number of blocks for every format

    float * w   = (float *) malloc((size_t) k * sizeof(float));
    float * a   = (float *) malloc((size_t) k * sizeof(float));
    float * dec = (float *) malloc((size_t) k * sizeof(float));

    // uniform importances: the formats that demand an imatrix only need one to
    // exist, and a flat one keeps the encoding reproducible
    float * imatrix = (float *) malloc((size_t) k * sizeof(float));
    for (int64_t i = 0; i < k; ++i) {
        imatrix[i] = 1.0f;
    }

    for (int t = 0; t < GK_TYPE_COUNT; ++t) {
        const struct gk_type_traits * tr = gk_get_type_traits((enum gk_type) t);

        if (tr == NULL || !tr->is_quantized || tr->vec_dot == NULL ||
            tr->from_float == NULL || tr->to_float == NULL) {
            continue;
        }
        if (k % tr->blck_size != 0) {
            continue;
        }

        g_dot_rng = 0x243f6a8885a308d3ull;

        // The weight magnitude is varied per sixteen elements - the finest
        // group any format here scales independently. With uniform random data
        // neighbouring groups get near-identical scales, so mixing two of them
        // up barely moves the result: a deliberately swapped pair of iq2_xs
        // half-group scales scored 0.0021 against a 0.002 bar, which is luck
        // rather than a test. Giving adjacent groups genuinely different
        // magnitudes makes that class of bug loud.
        static const float group_gain[4] = { 1.0f, 0.3f, 0.75f, 0.2f };

        for (int64_t i = 0; i < k; ++i) {
            w[i] = dot_rand() * group_gain[(i / 16) % 4];
            a[i] = dot_rand();
        }

        // Encoded through the codec directly rather than through the traits'
        // from_float, because iq2_xxs, iq2_xs and iq1_s refuse to encode
        // without an importance matrix and from_float has nowhere to pass one.
        // Without this they silently produced an unencoded buffer, and the
        // comparison below passed while proving nothing.
        const size_t wbytes = gk_row_size((enum gk_type) t, k);
        void * qw = calloc(1, wbytes);

        qz_quantize_init((qz_type) t);
        const size_t wrote = qz_quantize_chunk((qz_type) t, w, qw, 0, 1, k, imatrix);

        CHECK_MSG(wrote == wbytes, "%s: encoder wrote %zu bytes, expected %zu",
                  tr->name, wrote, wbytes);

        tr->to_float(qw, dec, k);

        // truth from the decoded weight, so the format's own quantization error
        // is not counted - only the dot is on trial
        double truth = 0.0, mag = 0.0;
        for (int64_t i = 0; i < k; ++i) {
            truth += (double) dec[i] * (double) a[i];
            mag   += fabs((double) dec[i] * (double) a[i]);
        }

        // An all-zero weight row would make the comparison below pass no matter
        // what the dot did. That is exactly what happened for the formats that
        // need an importance matrix, so it is now a failure rather than a
        // silently green line.
        CHECK_MSG(mag > 0.0, "%s: the encoded weight row decodes to all zeros", tr->name);

        const enum gk_type vdt = tr->vec_dot_type;
        const struct gk_type_traits * vtr = gk_get_type_traits(vdt);

        const void * pdot = a;
        void * qa = NULL;
        if (vdt != GK_TYPE_F32) {
            qa = malloc(gk_row_size(vdt, k));
            vtr->from_float(a, qa, k);
            pdot = qa;
        }

        float got = 0.0f;
        tr->vec_dot((int) k, &got, 0, qw, 0, pdot, 0, 1);

        // normalised by the sum of magnitudes, so a near-zero dot from
        // cancellation does not turn a tiny absolute error into a huge relative one
        const double err = mag > 0.0 ? fabs((double) got - truth) / mag : 0.0;
        // 8-bit activations give a per-element relative error near 1/254; over
        // k elements that averages down to about 1e-4, and every correct format
        // below measures between 1e-5 and 2.6e-4. The bar sits just above the
        // worst of those rather than at a comfortable round number, because a
        // loose bar is not a test - at 1e-2 a deliberately swapped pair of
        // nvfp4 sub-group scales scored 5.6e-3 and passed.
        const double bar = vdt == GK_TYPE_F32 ? 1e-6 : 1e-3;

        CHECK_MSG(err < bar, "%s: dot is %.9g, codec says %.9g (err %.3g, bar %.3g)",
                  tr->name, (double) got, truth, err, bar);

        printf("  %-8s %-5s err %8.2e\n", tr->name,
               vdt == GK_TYPE_F32 ? "float" : "int", err);

        free(qw);
        free(qa);
    }

    free(w);
    free(a);
    free(dec);
    free(imatrix);
}

// nvfp4 carries a UE4M3 scale per 16 weights, which its dot product would
// otherwise decode arithmetically in the inner loop - measured as most of that
// kernel. The codec therefore ships the decode tabulated, next to its other
// value tables, and the kernel reads that. A table is only as good as its
// agreement with the formula it came from, so this sweeps every entry.
// q3_K unpacks sixteen 6-bit scales out of twelve bytes, and its vector
// kernel does that arithmetically in a register rather than calling the scalar
// version - the stack buffer the scalar one fills is precisely what made that
// kernel slow. Two implementations of one bit layout is a thing to be nervous
// about, so they are swept against each other. Twelve bytes is 96 bits, too
// wide to enumerate, so this is a large random sweep plus the boundary
// patterns (all-zero, all-ones) that a random sweep is least likely to hit.
static void test_q3_k_scale_unpack(void) {
    printf("q3_K scale unpack, vector against scalar\n");

    uint32_t rs = 12345u;
    int mismatches = 0;

    for (int t = 0; t < 20000; ++t) {
        uint8_t src[12];
        for (int i = 0; i < 12; ++i) {
            if (t == 0) {
                src[i] = 0x00;
            } else if (t == 1) {
                src[i] = 0xff;
            } else {
                rs = rs * 1664525u + 1013904223u;
                src[i] = (uint8_t) (rs >> 24);
            }
        }

        int8_t want[16], got[16];
        gk_q3_k_unpack_scalar(src, want);
        gk_q3_k_unpack_vector(src, got);

        if (memcmp(want, got, sizeof(want)) != 0) {
            if (mismatches < 3) {
                printf("  case %d:\n    scalar", t);
                for (int i = 0; i < 16; ++i) { printf(" %4d", want[i]); }
                printf("\n    vector");
                for (int i = 0; i < 16; ++i) { printf(" %4d", got[i]); }
                printf("\n");
            }
            mismatches++;
        }
    }

    CHECK_MSG(mismatches == 0, "%d of 20000 q3_K scale unpacks disagree", mismatches);
}

static void test_ue4m3_vs_codec(void) {
    printf("UE4M3 table against the codec's converter\n");

    int mismatches = 0;

    for (int i = 0; i < 256; ++i) {
        const float want = qz_ue4m3_to_fp32((uint8_t) i);
        const float got  = qz_ue4m3_values[i];

        // bit equality, not a tolerance: both are decodes of the same 8-bit
        // code, and there is nothing here to round
        if (memcmp(&got, &want, sizeof(float)) != 0) {
            if (mismatches < 5) {
                printf("  code 0x%02x: converter %a, table %a\n",
                       i, (double) want, (double) got);
            }
            mismatches++;
        }
    }

    CHECK_MSG(mismatches == 0, "%d of 256 UE4M3 table entries disagree", mismatches);
}

int main(void) {
    printf("gk foundation tests\n\n");

    test_type_geometry();
    test_fp_roundtrip();
    test_context_and_strides();
    test_out_of_space();
    test_empty_tensors();
    test_graph_order();
    test_graph_deep();
    test_op_names();
    test_vec_dot_vs_codec();
    test_ue4m3_vs_codec();
    test_q3_k_scale_unpack();

    printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
