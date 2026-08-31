// Threading tests. Self-contained - no reference library needed.
//
// The property being checked is stronger than "the answers are close": every
// output element is computed by exactly one thread, and the arithmetic that
// produces it does not depend on how many threads exist. So the results must be
// *bit identical* at every thread count.
//
// Most kernels get there by splitting destination rows. Matmul does not - it
// tiles the output on two axes - so it gets its own sweep further down, checked
// against a naive matmul rather than only against other thread counts.
//
// That matters beyond tidiness. A kernel that accumulated across the split, or
// that let two threads touch the same row, would usually still produce
// plausible numbers - close enough to pass a tolerance check, and different
// run to run. Requiring bit equality turns that class of bug into a hard
// failure, and makes inference reproducible regardless of how many cores the
// machine happens to have.

#include "gk_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fails  = 0;
static int g_checks = 0;

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

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float) (uint32_t) (g_rng >> 33) / (float) 0x7fffffffu * 2.0f - 1.0f;
}

// Builds a graph exercising every kernel that partitions differently: the
// row-split majority, the group-split norm, the token-split expert matmul, the
// single-threaded reduction, and a rope whose position comes from the row
// index. If any of those mis-derives its indices from the thread's slice, the
// output moves.
static struct gk_tensor * build_graph(struct gk_ctx * ctx, uint64_t seed) {
    g_rng = seed;

    const int64_t d = 256;
    const int64_t n = 48;

    struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, d, n);
    for (int64_t i = 0; i < d * n; ++i) {
        ((float *) x->data)[i] = frand();
    }

    struct gk_tensor * w = gk_new_tensor_2d(ctx, GK_TYPE_F32, d, d);
    for (int64_t i = 0; i < d * d; ++i) {
        ((float *) w->data)[i] = frand() * 0.1f;
    }

    struct gk_tensor * pos = gk_new_tensor_1d(ctx, GK_TYPE_I32, n);
    for (int64_t i = 0; i < n; ++i) {
        ((int32_t *) pos->data)[i] = (int32_t) i;
    }

    struct gk_tensor * t = gk_rms_norm(ctx, x, 1e-5f);
    t = gk_mul_mat(ctx, w, t);
    t = gk_silu(ctx, t);
    t = gk_add(ctx, t, x);

    // rope wants (head_dim, n_head, n_tok)
    struct gk_tensor * r = gk_reshape_3d(ctx, t, 64, 4, n);
    r = gk_rope_ext(ctx, r, pos, NULL, 64, 0, 2048,
                    10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    t = gk_reshape_2d(ctx, gk_cont(ctx, r), d, n);

    // A softmax with ALiBi, whose slope is derived from the head index. The
    // slope multiplies the mask, so ALiBi without one is meaningless and the
    // builder rejects it.
    struct gk_tensor * mask = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 4);
    for (int64_t j = 0; j < 4; ++j) {
        for (int64_t i = 0; i < 64; ++i) {
            ((float *) mask->data)[j * 64 + i] = i > j * 16 ? -INFINITY : 0.0f;
        }
    }

    struct gk_tensor * s = gk_reshape_3d(ctx, t, 64, 4, n);
    s = gk_soft_max_ext(ctx, s, mask, 0.125f, 8.0f);
    t = gk_reshape_2d(ctx, gk_cont(ctx, s), d, n);

    // group norm splits by group, not by row
    struct gk_tensor * g = gk_reshape_3d(ctx, t, 16, 16, n);
    g = gk_group_norm(ctx, g, 4, 1e-5f);
    t = gk_reshape_2d(ctx, gk_cont(ctx, g), d, n);

    t = gk_glu(ctx, t, GK_GLU_OP_SWIGLU, false);   // halves the row
    t = gk_sum_rows(ctx, t);                 // one value per row

    return t;
}

// Runs the graph at `nth` threads and returns a copy of the output bytes.
static float * run_at(int nth, int64_t * n_out) {
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    if (ctx == NULL) {
        return NULL;
    }

    struct gk_tensor * out = build_graph(ctx, 0x1234abcdull);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    const enum gk_status st = gk_graph_compute(graph, nth);
    if (st != GK_STATUS_SUCCESS) {
        printf("  compute failed at %d threads, status %d\n", nth, (int) st);
        gk_free(ctx);
        return NULL;
    }

    *n_out = gk_nelements(out);
    float * copy = (float *) malloc((size_t) *n_out * sizeof(float));
    memcpy(copy, out->data, (size_t) *n_out * sizeof(float));

    gk_free(ctx);
    return copy;
}

static void test_determinism(void) {
    printf("thread count does not change the result\n");

    int64_t n1 = 0;
    float * base = run_at(1, &n1);
    CHECK_MSG(base != NULL, "single-threaded run failed");
    if (base == NULL) {
        return;
    }

    // A count above the row count is deliberate: the last threads get an empty
    // slice, and an off-by-one in the partition shows up here rather than in
    // production on a machine with more cores than the developer's.
    static const int counts[] = { 2, 3, 4, 5, 7, 8, 16, 64 };

    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
        const int nth = counts[i];

        int64_t n = 0;
        float * got = run_at(nth, &n);

        if (got == NULL) {
            CHECK_MSG(false, "run at %d threads failed", nth);
            continue;
        }

        if (n != n1) {
            CHECK_MSG(false, "%d threads produced %lld values, not %lld",
                      nth, (long long) n, (long long) n1);
            free(got);
            continue;
        }

        int64_t diff = 0;
        int64_t first = -1;
        for (int64_t j = 0; j < n; ++j) {
            if (memcmp(&base[j], &got[j], sizeof(float)) != 0) {
                if (first < 0) {
                    first = j;
                }
                diff++;
            }
        }

        CHECK_MSG(diff == 0,
                  "%d threads: %lld/%lld values differ, first at [%lld] (%.9g vs %.9g)",
                  nth, (long long) diff, (long long) n, (long long) first,
                  first >= 0 ? (double) base[first] : 0.0,
                  first >= 0 ? (double) got[first] : 0.0);

        free(got);
    }

    free(base);
}

// Matmul gets its own sweep because it is the one kernel that does not split
// by destination row. It tiles the output on both axes - a block of weight rows
// by a block of activation columns - so its index arithmetic is the only place
// here where a thread's slice has to be decomposed rather than just offset.
//
// The shapes are chosen to break that decomposition if it is wrong:
//
//   - one column, which is every matmul in single-token decoding. Under the
//     old row split this was a single unit of work, so every thread but one
//     idled and the case could not go wrong; now it is the case that divides.
//   - extents that are not multiples of either block size, so the tail tiles
//     are partial in both directions.
//   - a quantized weight, which sends the activation panel through a
//     conversion the float path skips.
//   - a third dimension with broadcast, since the tile index has to unpack to
//     (row block, column block, i2, i3) and a wrong divisor still produces
//     plausible numbers.
struct matmul_shape {
    const char * name;
    enum gk_type wtype;
    int64_t k, rows, cols, n2, r2;
};

// A deliberately naive matmul: one column at a time, one weight row at a time,
// no tiling and no threads.
//
// This exists because comparing thread counts against each other is *not*
// enough. An indexing bug in the tiled kernel is usually deterministic - it
// gets the same wrong answer at one thread as at sixteen - so a
// consistency-only check passes it. Two deliberately introduced index bugs did
// pass, which is why this is here.
//
// It calls the same `vec_dot` and the same activation conversion as the kernel,
// so the arithmetic is identical and the comparison can be exact. The only
// thing that differs is the index arithmetic, which is exactly what is on trial.
static float * reference_matmul(const struct gk_tensor * w,
                                const struct gk_tensor * x,
                                const struct gk_tensor * out) {
    const struct gk_type_traits * tr  = gk_get_type_traits(w->type);
    const enum gk_type            vdt = tr->vec_dot_type;
    const struct gk_type_traits * vtr = gk_get_type_traits(vdt);

    const int64_t k  = w->ne[0];
    const int64_t r2 = x->ne[2] / w->ne[2];
    const int64_t r3 = x->ne[3] / w->ne[3];

    float * ref  = (float *) malloc((size_t) gk_nelements(out) * sizeof(float));
    void  * qbuf = malloc(gk_row_size(vdt, k) + 64);

    for (int64_t i3 = 0; i3 < out->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < out->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < out->ne[1]; ++i1) {
                const float * pb = (const float *) ((const char *) x->data
                    + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]);

                const void * pdot = pb;
                if (vdt != GK_TYPE_F32) {
                    vtr->from_float(pb, qbuf, k);
                    pdot = qbuf;
                }

                for (int64_t i0 = 0; i0 < out->ne[0]; ++i0) {
                    const char * pa = (const char *) w->data
                        + i0 * w->nb[1] + (i2 / r2) * w->nb[2] + (i3 / r3) * w->nb[3];

                    float v = 0.0f;
                    tr->vec_dot((int) k, &v, 0, pa, 0, pdot, 0, 1);

                    ref[((i3 * out->ne[2] + i2) * out->ne[1] + i1) * out->ne[0] + i0] = v;
                }
            }
        }
    }

    free(qbuf);
    return ref;
}

// Runs one shape at `nth` threads. When `ref_out` is non-NULL it also returns
// the naive result for the same inputs.
static float * run_matmul_at(const struct matmul_shape * s, int nth,
                             int64_t * n_out, float ** ref_out) {
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    if (ctx == NULL) {
        return NULL;
    }

    g_rng = 0xfeedfaceull;

    struct gk_tensor * w = gk_new_tensor_3d(ctx, s->wtype, s->k, s->rows, s->n2);
    struct gk_tensor * x = gk_new_tensor_3d(ctx, GK_TYPE_F32, s->k, s->cols,
                                            s->n2 * s->r2);

    // build the weight through the codec so a quantized type is filled legally
    float * tmp = (float *) malloc((size_t) s->k * sizeof(float));
    const struct gk_type_traits * tr = gk_get_type_traits(s->wtype);
    const size_t rb = gk_row_size(s->wtype, s->k);

    for (int64_t r = 0; r < s->rows * s->n2; ++r) {
        for (int64_t i = 0; i < s->k; ++i) {
            tmp[i] = frand() * 0.1f;
        }
        tr->from_float(tmp, (char *) w->data + (size_t) r * rb, s->k);
    }
    free(tmp);

    for (int64_t i = 0; i < gk_nelements(x); ++i) {
        ((float *) x->data)[i] = frand();
    }

    struct gk_tensor * out = gk_mul_mat(ctx, w, x);

    // before computing, so a kernel writing out of its tile is visible as a
    // value that was never overwritten
    memset(out->data, 0x7f, (size_t) gk_nelements(out) * sizeof(float));

    if (ref_out != NULL) {
        *ref_out = reference_matmul(w, x, out);
    }

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    if (gk_graph_compute(graph, nth) != GK_STATUS_SUCCESS) {
        gk_free(ctx);
        return NULL;
    }

    *n_out = gk_nelements(out);
    float * copy = (float *) malloc((size_t) *n_out * sizeof(float));
    memcpy(copy, out->data, (size_t) *n_out * sizeof(float));

    gk_free(ctx);
    return copy;
}

static void test_matmul_tiling(void) {
    printf("matmul tiling matches a naive matmul exactly, at every thread count\n");

    static const struct matmul_shape shapes[] = {
        { "1 column, f32",        GK_TYPE_F32,  256,  512,   1, 1, 1 },
        { "1 column, q4_K",       GK_TYPE_Q4_K, 256,  512,   1, 1, 1 },
        { "1 column, q6_K",       GK_TYPE_Q6_K, 256,  512,   1, 1, 1 },
        { "partial tiles, f32",   GK_TYPE_F32,  256,   97,  13, 1, 1 },
        { "partial tiles, q4_K",  GK_TYPE_Q4_K, 256,   97,  13, 1, 1 },
        { "fewer rows than tile", GK_TYPE_F32,  256,    3,   2, 1, 1 },
        { "3d, no broadcast",     GK_TYPE_F32,  256,   65,   5, 3, 1 },
        { "3d, broadcast x2",     GK_TYPE_Q4_K, 256,   65,   5, 2, 2 },
        { "wide batch",           GK_TYPE_Q4_K, 256,  129,  37, 1, 1 },
        // These have a single column block and several slabs, so the column
        // index does not change when the slab does. That is the only way to
        // catch an activation panel cached on the column block alone - with two
        // or more column blocks the two indices happen to move together and a
        // stale panel is never observed.
        { "3d, 1 column, f32",    GK_TYPE_F32,  256,   70,   1, 3, 1 },
        { "3d, 1 column, q4_K",   GK_TYPE_Q4_K, 256,   70,   1, 3, 1 },
        { "3d, 2 cols, bcast",    GK_TYPE_Q4_K, 256,   70,   2, 2, 2 },
    };

    static const int counts[] = { 2, 3, 5, 8, 16, 64 };

    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        int64_t n1 = 0;
        float * ref  = NULL;
        float * base = run_matmul_at(&shapes[s], 1, &n1, &ref);

        if (base == NULL || ref == NULL) {
            CHECK_MSG(false, "%s: single-threaded run failed", shapes[s].name);
            free(base);
            free(ref);
            continue;
        }

        int bad = 0;

        // Every thread count is compared against the naive result, not against
        // each other, so a bug that is wrong the same way everywhere still fails.
        for (size_t i = 0; i <= sizeof(counts) / sizeof(counts[0]); ++i) {
            const int nth = i == 0 ? 1 : counts[i - 1];

            int64_t n = 0;
            float * got = i == 0 ? base : run_matmul_at(&shapes[s], nth, &n, NULL);

            if (got == NULL || (i > 0 && n != n1)) {
                CHECK_MSG(false, "%s: run at %d threads failed", shapes[s].name, nth);
                if (i > 0) {
                    free(got);
                }
                bad = 1;
                continue;
            }

            int64_t diff = 0, first = -1;
            for (int64_t j = 0; j < n1; ++j) {
                if (memcmp(&ref[j], &got[j], sizeof(float)) != 0) {
                    if (first < 0) {
                        first = j;
                    }
                    diff++;
                }
            }

            CHECK_MSG(diff == 0,
                      "%s at %d threads: %lld/%lld differ from the naive result, "
                      "first at [%lld] (%.9g vs %.9g)",
                      shapes[s].name, nth, (long long) diff, (long long) n1,
                      (long long) first,
                      first >= 0 ? (double) ref[first] : 0.0,
                      first >= 0 ? (double) got[first] : 0.0);
            if (diff != 0) {
                bad = 1;
            }
            if (i > 0) {
                free(got);
            }
        }

        // Matching would be vacuous if both sides were all zero.
        int64_t nonzero = 0;
        for (int64_t j = 0; j < n1; ++j) {
            if (ref[j] != 0.0f) {
                nonzero++;
            }
        }
        CHECK_MSG(nonzero > 0, "%s: every output is zero", shapes[s].name);

        printf("  %-22s %s\n", shapes[s].name, bad ? "MISMATCH" : "ok");
        free(base);
        free(ref);
    }
}

// The pool is created and destroyed around every gk_graph_compute call in the
// convenience form, so churning it is the normal case, not an edge one.
static void test_pool_lifecycle(void) {
    printf("pool creation and teardown\n");

    for (int i = 0; i < 50; ++i) {
        struct gk_pool * pool = gk_pool_create(4);
        CHECK_MSG(pool != NULL, "pool creation failed on iteration %d", i);
        if (pool == NULL) {
            return;
        }
        CHECK_MSG(gk_pool_n_threads(pool) == 4, "pool reported the wrong size");
        gk_pool_free(pool);
    }

    // a pool that is created and freed without ever running anything
    struct gk_pool * idle = gk_pool_create(8);
    CHECK_MSG(idle != NULL, "idle pool creation failed");
    gk_pool_free(idle);

    // asking for zero means one per core
    struct gk_pool * autop = gk_pool_create(0);
    CHECK_MSG(autop != NULL, "auto-sized pool creation failed");
    if (autop != NULL) {
        CHECK_MSG(gk_pool_n_threads(autop) >= 1, "auto-sized pool has no threads");
        gk_pool_free(autop);
    }
}

// Counts how many times each index ran, and that every index appears exactly
// once per run - the contract gk_pool_run offers its callers.
struct count_ctx {
    int counts[64];
    int nth_seen;
};

static void counting_fn(void * ctx, int ith, int nth) {
    struct count_ctx * c = (struct count_ctx *) ctx;
    if (ith < 64) {
        c->counts[ith]++;
    }
    c->nth_seen = nth;
}

static void test_pool_dispatch(void) {
    printf("every thread index runs exactly once per job\n");

    struct gk_pool * pool = gk_pool_create(8);
    CHECK_MSG(pool != NULL, "pool creation failed");
    if (pool == NULL) {
        return;
    }

    struct count_ctx c;
    memset(&c, 0, sizeof(c));

    const int jobs = 100;
    for (int i = 0; i < jobs; ++i) {
        gk_pool_run(pool, counting_fn, &c);
    }

    CHECK_MSG(c.nth_seen == 8, "workers saw nth = %d, expected 8", c.nth_seen);

    for (int i = 0; i < 8; ++i) {
        CHECK_MSG(c.counts[i] == jobs,
                  "index %d ran %d times, expected %d", i, c.counts[i], jobs);
    }
    for (int i = 8; i < 64; ++i) {
        CHECK_MSG(c.counts[i] == 0, "index %d ran but should not exist", i);
    }

    gk_pool_free(pool);
}

int main(void) {
    printf("gk threading tests\n\n");

    test_pool_lifecycle();
    test_pool_dispatch();
    test_determinism();
    test_matmul_tiling();

    printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
