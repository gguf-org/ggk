// Backend and graph-allocator tests. Self-contained.
//
// The allocator is the part worth testing hard. Its whole purpose is to hand
// the same memory to different tensors at different times, so a mistake there
// does not crash - it silently returns the wrong numbers, because a tensor
// reads bytes that something else has since overwritten.
//
// The check that catches that is comparing a graph run through the allocator
// against the same graph with every tensor separately allocated. If a lifetime
// is computed too short, the two disagree.

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

static uint64_t g_rng;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float) (uint32_t) (g_rng >> 33) / (float) 0x7fffffffu * 2.0f - 1.0f;
}

// --------------------------------------------------------------------------

static void test_buffer_type(void) {
    printf("CPU buffer type\n");

    gk_backend_buffer_type_t buft = gk_backend_cpu_buffer_type();

    CHECK_MSG(strcmp(gk_backend_buft_name(buft), "CPU") == 0, "wrong buffer type name");
    CHECK_MSG(gk_backend_buft_is_host(buft), "CPU memory should report as host memory");
    CHECK_MSG(gk_backend_buft_get_alignment(buft) >= GK_MEM_ALIGN, "alignment too weak");

    gk_backend_buffer_t buf = gk_backend_buft_alloc_buffer(buft, 4096);
    CHECK_MSG(buf != NULL, "buffer allocation failed");
    if (buf == NULL) {
        return;
    }

    CHECK_MSG(gk_backend_buffer_get_size(buf) == 4096, "buffer reports the wrong size");

    void * base = gk_backend_buffer_get_base(buf);
    CHECK_MSG(base != NULL, "buffer has no base pointer");
    CHECK_MSG(((uintptr_t) base % GK_MEM_ALIGN) == 0, "buffer base is not aligned");

    gk_backend_buffer_clear(buf, 0);
    CHECK_MSG(((const unsigned char *) base)[0] == 0, "clear did not write");
    CHECK_MSG(((const unsigned char *) base)[4095] == 0, "clear did not reach the end");

    gk_backend_buffer_free(buf);

    // a zero-sized buffer is legitimate and must not be a special case for
    // every caller
    gk_backend_buffer_t empty = gk_backend_buft_alloc_buffer(buft, 0);
    CHECK_MSG(empty != NULL, "zero-sized buffer should still produce an object");
    if (empty != NULL) {
        CHECK_MSG(gk_backend_buffer_get_size(empty) == 0, "zero buffer has nonzero size");
        gk_backend_buffer_free(empty);
    }
}

// Borrowed memory must survive the buffer being freed - this is the path a
// memory-mapped model file takes, and freeing it would unmap the model.
static void test_buffer_from_ptr(void) {
    printf("buffer over borrowed memory\n");

    float * mine = (float *) malloc(256 * sizeof(float));
    for (int i = 0; i < 256; ++i) {
        mine[i] = (float) i;
    }

    gk_backend_buffer_t buf = gk_backend_cpu_buffer_from_ptr(mine, 256 * sizeof(float));
    CHECK_MSG(buf != NULL, "wrapping a pointer failed");
    if (buf == NULL) {
        free(mine);
        return;
    }

    CHECK_MSG(gk_backend_buffer_get_base(buf) == mine, "wrapped buffer moved the base");

    gk_backend_buffer_free(buf);

    CHECK_MSG(mine[0] == 0.0f && mine[255] == 255.0f,
              "freeing a wrapping buffer must not touch the borrowed memory");
    free(mine);
}

static void test_tensor_transfer(void) {
    printf("tensor set and get\n");

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    struct gk_tensor * t = gk_new_tensor_1d(ctx, GK_TYPE_F32, 64);

    float in[64], out[64];
    for (int i = 0; i < 64; ++i) {
        in[i] = (float) i * 1.5f;
    }

    gk_backend_tensor_set(t, in, 0, sizeof(in));
    gk_backend_tensor_get(t, out, 0, sizeof(out));

    int bad = 0;
    for (int i = 0; i < 64; ++i) {
        if (in[i] != out[i]) {
            bad++;
        }
    }
    CHECK_MSG(bad == 0, "%d values did not survive a set/get round trip", bad);

    // a partial write must leave the rest alone
    float half[8];
    for (int i = 0; i < 8; ++i) {
        half[i] = -1.0f;
    }
    gk_backend_tensor_set(t, half, 4 * sizeof(float), sizeof(half));
    gk_backend_tensor_get(t, out, 0, sizeof(out));

    CHECK_MSG(out[3] == in[3], "a partial write ran backwards past its offset");
    CHECK_MSG(out[4] == -1.0f && out[11] == -1.0f, "a partial write did not land");
    CHECK_MSG(out[12] == in[12], "a partial write ran past its length");

    gk_free(ctx);
}

// --------------------------------------------------------------------------

// A graph with a long chain and a residual that skips over it. The residual is
// the interesting part: `x` has to stay alive across everything between its
// definition and the add, so an allocator that released it early would hand
// its bytes to an intermediate and corrupt the result.
static struct gk_tensor * build_graph(struct gk_ctx * ctx, int n_layers) {
    g_rng = 0xfeedfacecafebeefull;

    const int64_t d = 128;
    const int64_t n = 16;

    struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, d, n);
    struct gk_tensor * w = gk_new_tensor_2d(ctx, GK_TYPE_F32, d, d);

    for (int64_t i = 0; i < d * n; ++i) {
        ((float *) x->data)[i] = frand();
    }
    for (int64_t i = 0; i < d * d; ++i) {
        ((float *) w->data)[i] = frand() * 0.1f;
    }

    struct gk_tensor * t = x;
    for (int l = 0; l < n_layers; ++l) {
        struct gk_tensor * h = gk_rms_norm(ctx, t, 1e-5f);
        h = gk_mul_mat(ctx, w, h);
        h = gk_silu(ctx, h);
        t = gk_add(ctx, h, t); // residual: `t` must outlive the whole branch
    }

    gk_set_output(t);
    return t;
}

static void test_allocator(void) {
    printf("graph allocator\n");

    const int n_layers = 12;

    // ---- reference: every tensor gets its own storage ----------------------
    struct gk_ctx * plain = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    struct gk_tensor * out_plain = build_graph(plain, n_layers);

    struct gk_cgraph * gp = gk_new_graph(plain);
    gk_build_forward_expand(gp, out_plain);
    CHECK_MSG(gk_graph_compute(gp, 1) == GK_STATUS_SUCCESS, "reference run failed");

    const int64_t n_out = gk_nelements(out_plain);
    float * expected = (float *) malloc((size_t) n_out * sizeof(float));
    memcpy(expected, out_plain->data, (size_t) n_out * sizeof(float));

    const size_t plain_bytes = gk_used_mem(plain);

    // ---- the allocator: shapes only, storage assigned from one buffer -------
    // no_alloc leaves every node's data null, which is what the allocator
    // exists to fill in.
    struct gk_ctx * lean = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });

    // The inputs still need real storage, so they are built in their own
    // allocating context and referenced from the lean graph.
    struct gk_ctx * inputs = gk_init((struct gk_init_params) {
        .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    g_rng = 0xfeedfacecafebeefull;

    const int64_t d = 128, n = 16;
    struct gk_tensor * x = gk_new_tensor_2d(inputs, GK_TYPE_F32, d, n);
    struct gk_tensor * w = gk_new_tensor_2d(inputs, GK_TYPE_F32, d, d);
    for (int64_t i = 0; i < d * n; ++i) {
        ((float *) x->data)[i] = frand();
    }
    for (int64_t i = 0; i < d * d; ++i) {
        ((float *) w->data)[i] = frand() * 0.1f;
    }

    struct gk_tensor * t = x;
    for (int l = 0; l < n_layers; ++l) {
        struct gk_tensor * h = gk_rms_norm(lean, t, 1e-5f);
        h = gk_mul_mat(lean, w, h);
        h = gk_silu(lean, h);
        t = gk_add(lean, h, t);
    }
    gk_set_output(t);

    struct gk_cgraph * gl = gk_new_graph(lean);
    gk_build_forward_expand(gl, t);

    struct gk_gallocr * galloc = gk_gallocr_new(gk_backend_cpu_buffer_type());
    CHECK_MSG(galloc != NULL, "allocator creation failed");
    if (galloc == NULL) {
        return;
    }

    CHECK_MSG(gk_gallocr_reserve(galloc, gl), "reserve failed");
    CHECK_MSG(gk_gallocr_alloc_graph(galloc, gl), "alloc_graph failed");

    const size_t pooled_bytes = gk_gallocr_get_buffer_size(galloc);

    CHECK_MSG(gk_graph_compute(gl, 1) == GK_STATUS_SUCCESS, "allocated run failed");

    // ---- the results must be identical --------------------------------------
    int64_t bad = 0;
    for (int64_t i = 0; i < n_out; ++i) {
        if (memcmp(&expected[i], &((const float *) t->data)[i], sizeof(float)) != 0) {
            bad++;
        }
    }
    CHECK_MSG(bad == 0,
              "%lld/%lld outputs differ from the separately-allocated run - a "
              "tensor's storage was reused while it was still live",
              (long long) bad, (long long) n_out);

    // ---- and the point of it all: it should use far less memory -------------
    printf("    %d layers: separate %.2f MB, pooled %.2f MB (%.1fx less)\n",
           n_layers,
           (double) plain_bytes / (1024.0 * 1024.0),
           (double) pooled_bytes / (1024.0 * 1024.0),
           (double) plain_bytes / (double) (pooled_bytes ? pooled_bytes : 1));

    CHECK_MSG(pooled_bytes < plain_bytes,
              "pooled allocation (%zu) should be smaller than separate (%zu)",
              pooled_bytes, plain_bytes);

    // The live set does not grow with depth, so neither should the buffer.
    // Allow a couple of layers' worth of slack for the residual and the ends.
    const size_t per_layer = (size_t) d * n * sizeof(float);
    CHECK_MSG(pooled_bytes < per_layer * 8,
              "buffer (%zu) scales with depth; expected roughly a constant live set "
              "of a few times %zu", pooled_bytes, per_layer);

    gk_gallocr_free(galloc);
    gk_free(lean);
    gk_free(inputs);
    gk_free(plain);
    free(expected);
}

// Re-allocating the same graph must be idempotent: an inference loop calls
// alloc_graph once per token and would drift or grow if it were not.
static void test_allocator_reuse(void) {
    printf("allocator reuse across runs\n");

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_ctx * inputs = gk_init((struct gk_init_params) {
        .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    g_rng = 0x5555aaaaull;

    struct gk_tensor * x = gk_new_tensor_2d(inputs, GK_TYPE_F32, 64, 8);
    for (int i = 0; i < 64 * 8; ++i) {
        ((float *) x->data)[i] = frand();
    }

    struct gk_tensor * t = gk_rms_norm(ctx, x, 1e-5f);
    t = gk_silu(ctx, t);
    gk_set_output(t);

    struct gk_cgraph * g = gk_new_graph(ctx);
    gk_build_forward_expand(g, t);

    struct gk_gallocr * galloc = gk_gallocr_new(gk_backend_cpu_buffer_type());
    CHECK_MSG(gk_gallocr_reserve(galloc, g), "reserve failed");

    const size_t first_size = gk_gallocr_get_buffer_size(galloc);
    float first[64 * 8];

    for (int run = 0; run < 20; ++run) {
        CHECK_MSG(gk_gallocr_alloc_graph(galloc, g), "alloc_graph failed on run %d", run);
        CHECK_MSG(gk_graph_compute(g, 1) == GK_STATUS_SUCCESS, "run %d failed", run);

        if (run == 0) {
            memcpy(first, t->data, sizeof(first));
        } else {
            CHECK_MSG(memcmp(first, t->data, sizeof(first)) == 0,
                      "run %d produced different output from run 0", run);
        }
    }

    CHECK_MSG(gk_gallocr_get_buffer_size(galloc) == first_size,
              "the buffer grew across identical runs (%zu -> %zu)",
              first_size, gk_gallocr_get_buffer_size(galloc));

    gk_gallocr_free(galloc);
    gk_free(ctx);
    gk_free(inputs);
}

static void test_backend(void) {
    printf("CPU backend\n");

    gk_backend_t backend = gk_backend_cpu_init(4);
    CHECK_MSG(backend != NULL, "backend creation failed");
    if (backend == NULL) {
        return;
    }

    CHECK_MSG(strcmp(gk_backend_name(backend), "CPU") == 0, "wrong backend name");
    CHECK_MSG(gk_backend_cpu_n_threads(backend) == 4, "backend has the wrong thread count");
    CHECK_MSG(gk_backend_get_default_buffer_type(backend) == gk_backend_cpu_buffer_type(),
              "backend does not prefer CPU memory");

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    g_rng = 0x1010101ull;

    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, 8);
    for (int i = 0; i < 128 * 8; ++i) {
        ((float *) a->data)[i] = frand();
    }

    // supports_op is asked about the node, so build real ones
    struct gk_tensor * ok = gk_rms_norm(ctx, a, 1e-5f);
    CHECK_MSG(gk_backend_supports_op(backend, ok), "backend should support rms_norm");

    // the training steps are the ops this library deliberately does not have
    struct gk_tensor * unimpl = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, 8);
    unimpl->op = GK_OP_OPT_STEP_ADAMW;
    CHECK_MSG(!gk_backend_supports_op(backend, unimpl),
              "backend should not claim ops it has no kernel for");

    struct gk_cgraph * g = gk_new_graph(ctx);
    gk_build_forward_expand(g, ok);
    CHECK_MSG(gk_backend_graph_compute(backend, g) == GK_STATUS_SUCCESS,
              "backend graph compute failed");

    // the pool is owned by the backend, so this exercises reuse across graphs
    for (int i = 0; i < 10; ++i) {
        CHECK_MSG(gk_backend_graph_compute(backend, g) == GK_STATUS_SUCCESS,
                  "repeat compute %d failed", i);
    }

    gk_free(ctx);
    gk_backend_free(backend);
}

int main(void) {
    printf("gk backend and allocator tests\n\n");

    test_buffer_type();
    test_buffer_from_ptr();
    test_tensor_transfer();
    test_backend();
    test_allocator();
    test_allocator_reuse();

    printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
