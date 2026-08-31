// A benchmark for the kernels that dominate inference.
//
// The point is to make the SIMD work checkable rather than assumed. Build the
// library twice - once with the host's vector instructions and once without -
// and run this against both; the difference is the actual gain, on the actual
// kernels, rather than a claim about what the compiler ought to be doing.
//
//   cmake -B build-simd   -DGK_NATIVE=ON
//   cmake -B build-scalar -DGK_NATIVE=OFF
//
// Every case reports the arithmetic rate, so the numbers are comparable across
// shapes as well as across builds.

#include "gk_impl.h"

#include "qz_quant.h"
#include "gk_simd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

static double now_sec(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double) counter.QuadPart / (double) frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
#endif
}

static uint64_t g_rng = 0x243f6a8885a308d3ull;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float) (uint32_t) (g_rng >> 33) / (float) 0x7fffffffu * 2.0f - 1.0f;
}

// Runs `fn` enough times to spend at least `min_sec`, and returns the fastest
// per-iteration time. The minimum rather than the mean: the thing being
// measured is how fast the kernel can go, and the slow samples are the
// machine's noise, not the kernel's.
static double time_it(void (*fn)(void *), void * ctx, double min_sec) {
    // one untimed pass so caches and any first-use initialisation are warm
    fn(ctx);

    double best = 1e30;
    double spent = 0.0;
    int iters = 0;

    while (spent < min_sec || iters < 3) {
        const double t0 = now_sec();
        fn(ctx);
        const double dt = now_sec() - t0;

        if (dt < best) {
            best = dt;
        }
        spent += dt;
        iters++;
    }

    return best;
}

// --------------------------------------------------------------------------

struct matmul_ctx {
    struct gk_cgraph * graph;
    int                n_threads;
};

static void run_graph(void * p) {
    struct matmul_ctx * c = (struct matmul_ctx *) p;
    gk_graph_compute(c->graph, c->n_threads);
}

// A single matmul of a `rows x k` weight against `cols` activation columns.
// Reported as GFLOP/s counting the two operations of each multiply-add.
static void bench_matmul(enum gk_type type, int64_t k, int64_t rows, int64_t cols,
                         int n_threads) {
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1024u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    if (ctx == NULL) {
        printf("  (out of memory)\n");
        return;
    }

    struct gk_tensor * w = gk_new_tensor_2d(ctx, type, k, rows);
    struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, k, cols);

    float * tmp = (float *) malloc((size_t) k * sizeof(float));
    const size_t row_bytes = gk_row_size(type, k);

    // Encoded through the codec rather than the traits' from_float, and with a
    // uniform importance matrix: iq2_xxs, iq2_xs and iq1_s refuse to encode
    // without one, and from_float has nowhere to pass it. Without this the
    // weight buffer stays whatever the arena held and the benchmark times a dot
    // over meaningless bytes.
    float * imatrix = (float *) malloc((size_t) k * sizeof(float));
    for (int64_t i = 0; i < k; ++i) {
        imatrix[i] = 1.0f;
    }

    qz_quantize_init((qz_type) type);

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < k; ++i) {
            tmp[i] = frand() * 0.1f;
        }
        if (qz_quantize_chunk((qz_type) type, tmp,
                              (char *) w->data + (size_t) r * row_bytes,
                              0, 1, k, imatrix) != row_bytes) {
            printf("  %-8s (encoder refused this shape)\n", gk_type_name(type));
            free(tmp);
            free(imatrix);
            gk_free(ctx);
            return;
        }
    }
    for (int64_t i = 0; i < k * cols; ++i) {
        ((float *) x->data)[i] = frand();
    }
    free(tmp);
    free(imatrix);

    struct gk_tensor * out = gk_mul_mat(ctx, w, x);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    struct matmul_ctx c = { .graph = graph, .n_threads = n_threads };
    const double t = time_it(run_graph, &c, 0.30);

    const double flops = 2.0 * (double) k * (double) rows * (double) cols;

    printf("  %-8s k=%-5lld rows=%-5lld cols=%-3lld  %8.3f ms  %7.2f GFLOP/s\n",
           gk_type_name(type), (long long) k, (long long) rows, (long long) cols,
           t * 1e3, flops / t / 1e9);

    gk_free(ctx);
}

// --------------------------------------------------------------------------

struct elem_ctx {
    struct gk_cgraph * graph;
    int64_t            bytes;
};

// Elementwise work is limited by memory, not arithmetic, so it is reported as
// bandwidth. Reaching a useful fraction of the machine's memory bandwidth is
// the ceiling here; more vector width past that buys nothing.
static void bench_elementwise(const char * label, int64_t n, int op) {
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 512u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    struct gk_tensor * a = gk_new_tensor_1d(ctx, GK_TYPE_F32, n);
    struct gk_tensor * b = gk_new_tensor_1d(ctx, GK_TYPE_F32, n);

    for (int64_t i = 0; i < n; ++i) {
        ((float *) a->data)[i] = frand();
        ((float *) b->data)[i] = frand() + 2.0f;
    }

    struct gk_tensor * out = NULL;
    int64_t streams = 3; // two read, one written

    switch (op) {
        case 0: out = gk_add(ctx, a, b); break;
        case 1: out = gk_mul(ctx, a, b); break;
        case 2: out = gk_scale(ctx, a, 0.5f);      streams = 2; break;
        case 3: out = gk_rms_norm(ctx, a, 1e-5f);  streams = 2; break;
        case 4: out = gk_silu(ctx, a);             streams = 2; break;
        default: break;
    }

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    struct matmul_ctx c = { .graph = graph, .n_threads = 1 };
    const double t = time_it(run_graph, &c, 0.30);

    const double bytes = (double) n * (double) streams * sizeof(float);

    printf("  %-10s n=%-9lld  %8.3f ms  %7.2f GB/s\n",
           label, (long long) n, t * 1e3, bytes / t / 1e9);

    gk_free(ctx);
}

// --------------------------------------------------------------------------

// A transformer block, which is what the arithmetic above actually adds up to.
static void bench_block(enum gk_type wtype, int n_threads) {
    const int64_t d_model = 1024;
    const int64_t n_head  = 16;
    const int64_t d_head  = d_model / n_head;
    const int64_t n_tok   = 64;

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1024u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    if (ctx == NULL) {
        printf("  (out of memory)\n");
        return;
    }

    float * tmp = (float *) malloc((size_t) d_model * 4 * sizeof(float));
    const struct gk_type_traits * tr = gk_get_type_traits(wtype);

    struct gk_tensor * (*mk_w)(struct gk_ctx *, enum gk_type, int64_t, int64_t) =
        gk_new_tensor_2d;

    #define MKW(name, k, rows) \
        struct gk_tensor * name = mk_w(ctx, wtype, (k), (rows)); \
        { \
            const size_t rb = gk_row_size(wtype, (k)); \
            for (int64_t r = 0; r < (rows); ++r) { \
                for (int64_t i = 0; i < (k); ++i) tmp[i] = frand() * 0.1f; \
                tr->from_float(tmp, (char *) name->data + (size_t) r * rb, (k)); \
            } \
        }

    MKW(wq, d_model, d_model)
    MKW(wk, d_model, d_model)
    MKW(wv, d_model, d_model)
    MKW(wo, d_model, d_model)
    MKW(w1, d_model, d_model * 2)
    MKW(w2, d_model, d_model)
    #undef MKW

    free(tmp);

    struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, d_model, n_tok);
    for (int64_t i = 0; i < d_model * n_tok; ++i) {
        ((float *) x->data)[i] = frand();
    }

    struct gk_tensor * pos = gk_new_tensor_1d(ctx, GK_TYPE_I32, n_tok);
    for (int64_t i = 0; i < n_tok; ++i) {
        ((int32_t *) pos->data)[i] = (int32_t) i;
    }

    const float scale = 1.0f / sqrtf((float) d_head);

    struct gk_tensor * nrm = gk_rms_norm(ctx, x, 1e-5f);
    struct gk_tensor * q = gk_reshape_3d(ctx, gk_mul_mat(ctx, wq, nrm), d_head, n_head, n_tok);
    struct gk_tensor * k = gk_reshape_3d(ctx, gk_mul_mat(ctx, wk, nrm), d_head, n_head, n_tok);
    struct gk_tensor * v = gk_reshape_3d(ctx, gk_mul_mat(ctx, wv, nrm), d_head, n_head, n_tok);

    q = gk_rope_ext(ctx, q, pos, NULL, (int) d_head, 0, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = gk_rope_ext(ctx, k, pos, NULL, (int) d_head, 0, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    q = gk_cont(ctx, gk_permute(ctx, q, 0, 2, 1, 3));
    k = gk_cont(ctx, gk_permute(ctx, k, 0, 2, 1, 3));

    struct gk_tensor * att = gk_soft_max_ext(ctx, gk_mul_mat(ctx, k, q), NULL, scale, 0.0f);
    struct gk_tensor * vt  = gk_cont(ctx, gk_permute(ctx, v, 1, 2, 0, 3));
    struct gk_tensor * o   = gk_mul_mat(ctx, vt, att);

    o = gk_cont_2d(ctx, gk_permute(ctx, o, 0, 2, 1, 3), d_model, n_tok);
    o = gk_add(ctx, gk_mul_mat(ctx, wo, o), x);

    struct gk_tensor * f = gk_rms_norm(ctx, o, 1e-5f);
    f = gk_glu(ctx, gk_mul_mat(ctx, w1, f), GK_GLU_OP_SWIGLU, false);
    f = gk_add(ctx, gk_mul_mat(ctx, w2, f), o);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, f);

    struct matmul_ctx c = { .graph = graph, .n_threads = n_threads };
    const double t = time_it(run_graph, &c, 0.50);

    // the six projections dominate; attention adds two more matmuls per head
    const double proj_flops = 2.0 * (double) d_model * (double) d_model * (double) n_tok * 7.0;

    printf("  %-8s d=%lld tok=%lld threads=%-3d  %8.3f ms  %7.2f GFLOP/s  %6.1f tok/s\n",
           gk_type_name(wtype), (long long) d_model, (long long) n_tok, n_threads,
           t * 1e3, proj_flops / t / 1e9, (double) n_tok / t);

    gk_free(ctx);
}

int main(int argc, char ** argv) {
    int max_threads = 0;
    if (argc > 1) {
        max_threads = atoi(argv[1]);
    }
    if (max_threads <= 0) {
        max_threads = 8;
    }

    printf("gk benchmark\n");
    printf("  vector unit: %s, %d floats wide, %d accumulators\n",
           GK_SIMD_NAME, GK_SIMD_F32_STEP, GK_SIMD_ACC);
    printf("  f16 vector convert: %s\n\n", GK_SIMD_HAVE_F16 ? "yes" : "no");

    printf("matrix multiply, single thread\n");
    bench_matmul(GK_TYPE_F32,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_F16,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q8_0, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q4_0, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q4_1, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q5_0, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q5_1, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q2_K, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q3_K, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q4_K, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q5_K, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q6_K, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ4_NL, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ4_XS, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ2_XXS,4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ2_XS, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ2_S,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ3_XXS,4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ1_S,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ1_M,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_IQ3_S,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_MXFP4,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_NVFP4,  4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_TQ2_0,  4096, 4096, 1, 1);
    printf("\n");

    printf("matrix multiply, threaded (1 column - the decode case)\n");
    bench_matmul(GK_TYPE_Q4_K, 4096, 4096, 1, 1);
    bench_matmul(GK_TYPE_Q4_K, 4096, 4096, 1, 2);
    bench_matmul(GK_TYPE_Q4_K, 4096, 4096, 1, 4);
    bench_matmul(GK_TYPE_Q4_K, 4096, 4096, 1, 8);
    printf("\n");

    printf("matrix multiply, batched (32 columns)\n");
    bench_matmul(GK_TYPE_F32,  4096, 4096, 32, 1);
    bench_matmul(GK_TYPE_Q4_K, 4096, 4096, 32, 1);
    printf("\n");

    printf("elementwise and norms, single thread\n");
    bench_elementwise("add",      1 << 22, 0);
    bench_elementwise("mul",      1 << 22, 1);
    bench_elementwise("scale",    1 << 22, 2);
    bench_elementwise("rms_norm", 1 << 22, 3);
    bench_elementwise("silu",     1 << 22, 4);
    printf("\n");

    printf("transformer block\n");
    for (int t = 1; t <= max_threads; t *= 2) {
        bench_block(GK_TYPE_Q4_K, t);
    }
    printf("\n");

    return 0;
}
