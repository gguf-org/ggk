// Two or more real GPUs, and a graph that has to cross between them.
//
// test-sched already covers the scheduler's reasoning with backend doubles,
// but both of those doubles are backed by host memory: the staging path runs,
// the bus never does. test-cuda covers the device path, but only ever on one
// device. What neither reaches is the pair - a weight in device 0's memory and
// the next layer's weight in device 1's, which is what a diffusion model split
// across two cards actually looks like.
//
// That gap is where the failure this test was written for lived. A build whose
// device code was compiled for one card's architecture and not the other's
// works perfectly until a second device is given a node, and then every launch
// on it fails. Discovery now probes for that and refuses to register a device
// it has no code for, so reaching this test at all means every device here can
// run something; what remains to be checked is that the answer is right.
//
// With one GPU this reports what it found and passes. It is not a failure to
// own one graphics card.

#include "gk_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// The scheduler's own ceiling (gk_sched.c); not exported, and no machine this
// runs on is near it.
#define MAX_DEVICES 16

static int g_fails  = 0;
static int g_checks = 0;

#define CHECK_MSG(cond, ...)                        \
    do {                                            \
        g_checks++;                                 \
        if (!(cond)) {                              \
            g_fails++;                              \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                    \
            printf("\n");                           \
        }                                           \
    } while (0)

static uint64_t g_rng;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float) (uint32_t) (g_rng >> 33) / (float) 0x7fffffffu * 2.0f - 1.0f;
}

// --------------------------------------------------------------------------
// the graph
//
// A stack of layers, each one a norm, a matmul against that layer's weight, an
// activation and a scale. The matmul is what anchors a layer to a device: its
// weight lives in one device's memory and the scheduler has to follow it
// there, carrying the running activation across the bus each time the layer's
// device changes.
// --------------------------------------------------------------------------

#define N_LAYERS 6

static const int64_t g_dim = 128;
static const int64_t g_tok = 12;

static struct gk_tensor * build_layers(struct gk_ctx * ctx, struct gk_tensor * x,
                                       struct gk_tensor ** weights) {
    struct gk_tensor * t = x;
    for (int l = 0; l < N_LAYERS; ++l) {
        t = gk_rms_norm(ctx, t, 1e-5f);
        t = gk_mul_mat(ctx, weights[l], t);
        t = gk_silu(ctx, t);
        t = gk_scale(ctx, t, 0.75f);
    }
    gk_set_output(t);
    return t;
}

static void fill_reference(float * x, float * w) {
    g_rng = 0x9e3779b97f4a7c15ull;
    for (int64_t i = 0; i < g_dim * g_tok; ++i) {
        x[i] = frand();
    }
    for (int l = 0; l < N_LAYERS; ++l) {
        for (int64_t i = 0; i < g_dim * g_dim; ++i) {
            w[l * g_dim * g_dim + i] = frand() * 0.1f;
        }
    }
}

// The same computation on the CPU, in host memory, as the answer to compare
// against. Deliberately built from the same builder: a reference that is a
// second transcription of the maths would drift from it.
static int run_on_cpu(const float * x_src, const float * w_src, float * out) {
    gk_backend_t cpu = gk_backend_cpu_init(0);
    if (cpu == NULL) {
        return 1;
    }

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, g_dim, g_tok);
    memcpy(x->data, x_src, (size_t) g_dim * g_tok * sizeof(float));

    struct gk_tensor * weights[N_LAYERS];
    for (int l = 0; l < N_LAYERS; ++l) {
        weights[l] = gk_new_tensor_2d(ctx, GK_TYPE_F32, g_dim, g_dim);
        memcpy(weights[l]->data, w_src + (size_t) l * g_dim * g_dim,
               (size_t) g_dim * g_dim * sizeof(float));
    }

    struct gk_tensor * y = build_layers(ctx, x, weights);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, y);

    const enum gk_status status = gk_backend_graph_compute(cpu, graph);
    if (status == GK_STATUS_SUCCESS) {
        memcpy(out, y->data, (size_t) g_dim * g_tok * sizeof(float));
    }

    gk_free(ctx);
    gk_backend_free(cpu);
    return status == GK_STATUS_SUCCESS ? 0 : 1;
}

// --------------------------------------------------------------------------
// weights, spread across the devices
// --------------------------------------------------------------------------

struct device_weights {
    gk_backend_buffer_t buffer;
};

// Puts one layer's weight in one device's memory, the way a model loader
// would: a buffer from that device's buffer type, the tensor pointed into it,
// and the bytes written through the backend rather than by the host.
static bool place_weight(gk_backend_buffer_type_t buft, struct gk_tensor * w,
                         const float * src, struct device_weights * owner) {
    const size_t align = gk_backend_buft_get_alignment(buft);
    const size_t need  = gk_pad_size(gk_nbytes(w), align);

    gk_backend_buffer_t buf = gk_backend_buft_alloc_buffer(buft, need);
    if (buf == NULL) {
        return false;
    }

    w->data = (char *) gk_backend_buffer_get_base(buf);
    gk_backend_buffer_init_tensor(buf, w);

    // Device memory: the host may not write through the pointer.
    gk_backend_tensor_set(w, src, 0, gk_nbytes(w));

    owner->buffer = buf;
    return true;
}

// --------------------------------------------------------------------------
// the run
// --------------------------------------------------------------------------

static int run_across(gk_backend_t * backends, gk_device_t * devices, int n_dev,
                      const float * x_src, const float * w_src, const float * expected) {
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    // The weights and the input outlive the graph's arena reuse, so they get
    // their own context - the scheduler is free to reuse graph memory, and a
    // weight it decided was dead would come back as somebody else's bytes.
    struct gk_ctx * params = gk_init((struct gk_init_params) {
        .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = true,
    });

    struct device_weights owned[N_LAYERS + 1];
    struct gk_tensor *    weights[N_LAYERS];
    struct gk_tensor *    x     = NULL;
    struct gk_tensor *    y     = NULL;
    struct gk_cgraph *    graph = NULL;
    struct gk_sched *     sched = NULL;
    enum gk_status        status = GK_STATUS_SUCCESS;
    bool                  placed = true;
    int                   failed = 0;

    memset(owned, 0, sizeof(owned));
    memset(weights, 0, sizeof(weights));

    // The input starts on device 0; every layer after the first that lands
    // elsewhere is a crossing.
    x = gk_new_tensor_2d(params, GK_TYPE_F32, g_dim, g_tok);
    if (!place_weight(gk_backend_get_default_buffer_type(backends[0]), x, x_src, &owned[N_LAYERS])) {
        CHECK_MSG(false, "could not allocate the input on %s", gk_backend_name(backends[0]));
        placed = false;
    }

    for (int l = 0; placed && l < N_LAYERS; ++l) {
        const int dev = l % n_dev;
        weights[l] = gk_new_tensor_2d(params, GK_TYPE_F32, g_dim, g_dim);
        gk_format_name(weights[l], "w%d@%s", l, gk_device_name(devices[dev]));
        if (!place_weight(gk_backend_get_default_buffer_type(backends[dev]), weights[l],
                          w_src + (size_t) l * g_dim * g_dim, &owned[l])) {
            CHECK_MSG(false, "could not allocate weight %d on %s", l, gk_device_name(devices[dev]));
            placed = false;
        }
    }

    if (placed) {
        y     = build_layers(ctx, x, weights);
        graph = gk_new_graph(ctx);
        gk_build_forward_expand(graph, y);

        sched = gk_sched_new(backends, n_dev);
        if (sched == NULL) {
            CHECK_MSG(false, "could not create a scheduler over %d devices", n_dev);
            placed = false;
        }
    }

    if (!placed) {
        failed = 1;
    } else {
        status = gk_sched_graph_compute(sched, graph);
        CHECK_MSG(status == GK_STATUS_SUCCESS,
                  "compute across %d devices failed with status %d", n_dev, (int) status);
    }

    if (placed && status == GK_STATUS_SUCCESS) {
        const int splits = gk_sched_n_splits(sched);
        printf("    %d layers over %d devices produced %d splits\n", N_LAYERS, n_dev, splits);

        // With the weights alternating, a run that did not cross would mean
        // the scheduler ignored where they live - the answer might still come
        // out right while the second device sat idle, which is the failure
        // this whole file exists to notice.
        if (n_dev > 1) {
            CHECK_MSG(splits >= n_dev,
                      "expected the graph to cross between devices, got %d splits", splits);
        }

        float * got = (float *) malloc((size_t) g_dim * g_tok * sizeof(float));
        gk_backend_tensor_get(y, got, 0, (size_t) g_dim * g_tok * sizeof(float));

        // Layer on layer of matmul and silu, on hardware whose reductions do
        // not have to agree with the CPU's SIMD order, so this is a tolerance
        // rather than an equality - the same one test-cuda uses for its
        // composite graphs.
        double max_abs = 0.0;
        int    bad     = 0;
        for (int64_t i = 0; i < g_dim * g_tok; ++i) {
            const double diff = fabs((double) got[i] - (double) expected[i]);
            if (diff > max_abs) {
                max_abs = diff;
            }
            if (diff > 4e-3) {
                bad++;
            }
        }
        printf("    max abs error against the CPU: %.6g\n", max_abs);
        CHECK_MSG(bad == 0,
                  "%d of %lld values disagree with the CPU by more than 4e-3 (worst %.6g)",
                  bad, (long long) (g_dim * g_tok), max_abs);

        free(got);
    }

    if (sched != NULL) {
        gk_sched_free(sched);
    }

    for (int i = 0; i < N_LAYERS + 1; ++i) {
        if (owned[i].buffer != NULL) {
            gk_backend_buffer_free(owned[i].buffer);
        }
    }
    gk_free(ctx);
    gk_free(params);
    return failed;
}

// A tensor copied straight from one device's memory to another's, which is the
// transfer every crossing above is built out of. Checked on its own because a
// peer copy that silently moves nothing produces a wrong answer far away from
// here, and because the fallback when the pair cannot see each other - staging
// through the host - is a different path that has to give the same bytes.
static void test_device_to_device_copy(gk_backend_t * backends, gk_device_t * devices, int n_dev) {
    if (n_dev < 2) {
        return;
    }

    printf("  device-to-device tensor copy\n");

    const int64_t n = 4096;

    float * src = (float *) malloc((size_t) n * sizeof(float));
    float * got = (float *) malloc((size_t) n * sizeof(float));
    g_rng = 0x5deece66dull;
    for (int64_t i = 0; i < n; ++i) {
        src[i] = frand();
    }

    for (int a = 0; a < n_dev; ++a) {
        for (int b = 0; b < n_dev; ++b) {
            if (a == b) {
                continue;
            }

            struct gk_ctx * ctx = gk_init((struct gk_init_params) {
                .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
            });

            struct gk_tensor * ta = gk_new_tensor_1d(ctx, GK_TYPE_F32, n);
            struct gk_tensor * tb = gk_new_tensor_1d(ctx, GK_TYPE_F32, n);

            struct device_weights owner_a = { NULL };
            struct device_weights owner_b = { NULL };

            const bool ok_a = place_weight(gk_backend_get_default_buffer_type(backends[a]),
                                           ta, src, &owner_a);

            // The destination is allocated the same way but left as whatever
            // the device had; the copy is what has to put the values there.
            gk_backend_buffer_type_t buft_b = gk_backend_get_default_buffer_type(backends[b]);
            gk_backend_buffer_t buf_b =
                gk_backend_buft_alloc_buffer(buft_b, gk_pad_size(gk_nbytes(tb),
                                                                 gk_backend_buft_get_alignment(buft_b)));
            if (ok_a && buf_b != NULL) {
                tb->data = (char *) gk_backend_buffer_get_base(buf_b);
                gk_backend_buffer_init_tensor(buf_b, tb);
                gk_backend_buffer_clear(buf_b, 0);
                owner_b.buffer = buf_b;

                gk_backend_tensor_copy(ta, tb);
                gk_backend_synchronize(backends[b]);

                memset(got, 0, (size_t) n * sizeof(float));
                gk_backend_tensor_get(tb, got, 0, (size_t) n * sizeof(float));

                int bad = 0;
                for (int64_t i = 0; i < n; ++i) {
                    if (got[i] != src[i]) {
                        bad++;
                    }
                }
                CHECK_MSG(bad == 0, "%s -> %s copied %d of %lld values wrongly",
                          gk_device_name(devices[a]), gk_device_name(devices[b]),
                          bad, (long long) n);
            } else {
                CHECK_MSG(false, "could not allocate for the %s -> %s copy",
                          gk_device_name(devices[a]), gk_device_name(devices[b]));
            }

            if (owner_a.buffer != NULL) { gk_backend_buffer_free(owner_a.buffer); }
            if (owner_b.buffer != NULL) { gk_backend_buffer_free(owner_b.buffer); }
            gk_free(ctx);
        }
    }

    free(got);
    free(src);
}

// --------------------------------------------------------------------------
// the arrangement the diffusion engine actually produces
//
// The layered graph above crosses between devices, but it is a chain: every
// value is consumed by the node after it, and it is made of matmuls. A
// diffusion graph is neither. It carries views, permutes and conts through
// attention; it has skip connections that hand a value from an early block to
// one near the end, so a tensor produced on one device is read many splits
// later on another; and the engine does not place nodes by where their weights
// are - it pins whole runs of them with gk_sched_set_tensor_backend, from a
// partition it computed beforehand.
//
// Each of those is a different demand on the staging path, and none of them is
// exercised by a chain of matmuls.
// --------------------------------------------------------------------------

#define MAX_BUILDER_INPUTS 16

typedef struct gk_tensor * (*multi_builder)(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in);

static struct gk_tensor * build_attention(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t D = 128, T = 37, HD = 32, NH = 4;

    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, T);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[2] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[4] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[5] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D * 2);
    in[6] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D * 2, D);
    *n_in = 7;

    struct gk_tensor * x = in[0];
    for (int layer = 0; layer < 3; ++layer) {
        struct gk_tensor * h = gk_norm(ctx, x, 1e-5f);

        struct gk_tensor * q = gk_mul_mat(ctx, in[1], h);
        struct gk_tensor * k = gk_mul_mat(ctx, in[2], h);
        struct gk_tensor * v = gk_mul_mat(ctx, in[3], h);

        q = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, q, HD, NH, T), 0, 2, 1, 3));
        k = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, k, HD, NH, T), 0, 2, 1, 3));
        v = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, v, HD, NH, T), 0, 2, 1, 3));

        struct gk_tensor * att = gk_mul_mat(ctx, k, q);
        att = gk_soft_max_ext(ctx, att, NULL, 1.0f / sqrtf((float) HD), 0.0f);

        struct gk_tensor * o = gk_mul_mat(ctx, gk_cont(ctx, gk_transpose(ctx, v)), att);
        o = gk_cont(ctx, gk_permute(ctx, o, 0, 2, 1, 3));
        o = gk_reshape_2d(ctx, o, D, T);
        o = gk_mul_mat(ctx, in[4], o);

        x = gk_add(ctx, x, o);

        h = gk_norm(ctx, x, 1e-5f);
        h = gk_mul_mat(ctx, in[5], h);
        h = gk_gelu(ctx, h);
        h = gk_mul_mat(ctx, in[6], h);
        x = gk_add(ctx, x, h);
    }
    gk_set_output(x);
    return x;
}

// A U-Net's defining shape: what the downward path produces is kept, and the
// upward path concatenates it back in. The first skip is made in the first
// block and read in the last, so it spans the whole graph - and therefore
// every device boundary in it.
static struct gk_tensor * build_unet_skips(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t W = 16, H = 16, C = 32;
    const int n_levels = 3;

    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, W, H, C, 1);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, C, C);       // down
    in[2] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, C * 2, C);   // up, after concat
    in[3] = gk_new_tensor_1d(ctx, GK_TYPE_F32, C);
    *n_in = 4;

    struct gk_tensor * skips[8];
    struct gk_tensor * x = in[0];

    for (int level = 0; level < n_levels; ++level) {
        struct gk_tensor * h = gk_group_norm(ctx, x, 8, 1e-6f);
        h = gk_silu(ctx, h);
        h = gk_conv_2d(ctx, in[1], h, 1, 1, 1, 1, 1, 1);
        h = gk_add(ctx, h, gk_reshape_4d(ctx, in[3], 1, 1, C, 1));
        x = gk_add(ctx, x, h);
        skips[level] = x;
    }

    for (int level = n_levels - 1; level >= 0; --level) {
        // Reaches back across every split made since the skip was produced.
        struct gk_tensor * h = gk_concat(ctx, x, skips[level], 2);
        h = gk_conv_2d(ctx, in[2], h, 1, 1, 1, 1, 1, 1);
        h = gk_group_norm(ctx, h, 8, 1e-6f);
        h = gk_silu(ctx, h);
        x = gk_add(ctx, x, h);
    }

    gk_set_output(x);
    return x;
}

// In-place ops, which is what makes this graph different from the others here.
//
// An in-place op's result is a view of its first operand: it has no storage of
// its own and writes over what it was given. Nothing about that is visible in
// the pinning below, which places nodes in blocks the way an engine places
// transformer blocks - so a run of pins will happily put one of these on a
// device other than the one holding the tensor it overwrites. Staging cannot
// rescue it either: staging copies a value into memory the reader can address,
// and the whole point of an in-place op is that later nodes read the original
// back, so a write to a copy is a write that goes nowhere.
// The distance is the point. An in-place op sitting directly on top of the
// value it overwrites is placed next to it by any reasonable pinning, and
// nothing goes wrong. The one that goes wrong is far from what it writes: the
// value is produced at the start of the graph, the write happens near the end,
// and by then the pins have moved on to another device.
static struct gk_tensor * build_inplace_chain(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t D = 128, T = 24;

    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, T);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[2] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, T);
    *n_in = 3;

    struct gk_tensor * base = gk_mul_mat(ctx, in[1], in[0]);

    struct gk_tensor * t = base;
    for (int layer = 0; layer < 10; ++layer) {
        t = gk_mul_mat(ctx, in[1], t);
        t = gk_silu(ctx, t);
    }

    // Writes over `base`, which was produced at the top of the graph and by now
    // is a long way behind the pins. `t` has already been computed from it, so
    // the overwrite is well defined; what is not well defined is which device's
    // memory it lands in.
    struct gk_tensor * written = gk_add_inplace(ctx, base, in[2]);
    written = gk_scale_inplace(ctx, written, 0.5f);

    struct gk_tensor * out = gk_add(ctx, t, written);
    gk_set_output(out);
    return out;
}

static bool is_view_op(enum gk_op op) {
    return op == GK_OP_NONE || op == GK_OP_VIEW || op == GK_OP_RESHAPE ||
           op == GK_OP_PERMUTE || op == GK_OP_TRANSPOSE;
}

// Pins runs of nodes to devices in turn, the way the engine pins the segments
// of its layer-split partition. Views are left unpinned, because a view has no
// storage of its own and pinning one only confuses where its parent lives -
// the same rule the engine follows.
//
// Every fifth node is left unpinned as well, and that is the point rather than
// an economy. A graph whose every node is pinned is placed by the pins alone,
// and placement is then trivially the same however many times it is derived.
// The engine's graphs are not like that: whole classes of node fall outside its
// partition and are placed by the scheduler's own rules, which read where
// tensors already live - so those are the nodes whose placement can change
// after an allocation has given them somewhere to live.
static int pin_in_blocks(struct gk_sched * sched, gk_backend_t * backends, int n_dev,
                         struct gk_cgraph * graph) {
    const int n_nodes = gk_graph_n_nodes(graph);
    const int per_dev = (n_nodes + n_dev - 1) / n_dev;

    int pinned = 0;
    for (int i = 0; i < n_nodes; ++i) {
        struct gk_tensor * node = gk_graph_node(graph, i);
        if (is_view_op(node->op) || i % 5 == 0) {
            continue;
        }
        const int dev = (i / per_dev) % n_dev;
        if (gk_backend_supports_op(backends[dev], node)) {
            gk_sched_set_tensor_backend(sched, node, backends[dev]);
            pinned++;
        }
    }
    return pinned;
}

// `prealloc` reproduces how the diffusion engine drives the scheduler: it
// allocates the graph, writes its inputs into the storage that allocation just
// handed out, and only then computes. The scheduler has to treat the second
// call as already-allocated work; re-deriving the layout underneath a caller
// who has written into it is what corrupts a run.
static int run_pinned_composite(gk_backend_t * backends, gk_device_t * devices, int n_dev,
                                const char * name, multi_builder build, double tol,
                                bool prealloc) {
    GK_UNUSED(devices);

    // The reference: the same graph, on the CPU, in host memory.
    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * cpu_in[MAX_BUILDER_INPUTS];
    int n_in = 0;
    struct gk_tensor * cpu_out = build(cpu_ctx, cpu_in, &n_in);

    g_rng = 0x2545f4914f6cdd1dull;
    for (int i = 0; i < n_in; ++i) {
        const int64_t n = gk_nelements(cpu_in[i]);
        if (cpu_in[i]->type == GK_TYPE_F32) {
            for (int64_t j = 0; j < n; ++j) {
                ((float *) cpu_in[i]->data)[j] = frand() * 0.5f;
            }
        } else {
            // f16 weights: written through the type's own converter rather
            // than by hand, so the bytes are exactly what a loader would have
            // produced.
            float * tmp = (float *) malloc((size_t) n * sizeof(float));
            for (int64_t j = 0; j < n; ++j) {
                tmp[j] = frand() * 0.5f;
            }
            gk_get_type_traits(cpu_in[i]->type)->from_float(tmp, cpu_in[i]->data, n);
            free(tmp);
        }
    }

    gk_backend_t cpu = gk_backend_cpu_init(0);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cpu_out);
    const enum gk_status cpu_status = gk_backend_graph_compute(cpu, cpu_graph);

    const int64_t n_out = gk_nelements(cpu_out);
    float * expected = (float *) malloc((size_t) n_out * sizeof(float));
    if (cpu_status == GK_STATUS_SUCCESS) {
        memcpy(expected, cpu_out->data, (size_t) n_out * sizeof(float));
    }
    CHECK_MSG(cpu_status == GK_STATUS_SUCCESS, "%s: the CPU reference failed", name);

    // The run under test: same graph, inputs in device 0's memory, nodes
    // pinned across the devices in blocks.
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_ctx * params = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });

    struct gk_tensor * gpu_in[MAX_BUILDER_INPUTS];
    int n_gpu_in = 0;
    struct gk_tensor * gpu_out = build(ctx, gpu_in, &n_gpu_in);

    // The builder made the inputs in the graph context; they need storage of
    // their own, so they are rebuilt in the params context and swapped in.
    struct device_weights owned[MAX_BUILDER_INPUTS];
    memset(owned, 0, sizeof(owned));

    gk_backend_buffer_type_t buft = gk_backend_get_default_buffer_type(backends[0]);
    bool placed = true;
    for (int i = 0; i < n_gpu_in && placed; ++i) {
        const size_t align = gk_backend_buft_get_alignment(buft);
        gk_backend_buffer_t buf =
            gk_backend_buft_alloc_buffer(buft, gk_pad_size(gk_nbytes(gpu_in[i]), align));
        if (buf == NULL) {
            CHECK_MSG(false, "%s: could not allocate input %d", name, i);
            placed = false;
            break;
        }
        gpu_in[i]->data = (char *) gk_backend_buffer_get_base(buf);
        gk_backend_buffer_init_tensor(buf, gpu_in[i]);
        owned[i].buffer = buf;
    }

    if (placed) {
        struct gk_cgraph * graph = gk_new_graph_custom(ctx, 8192, false);
        gk_build_forward_expand(graph, gpu_out);

        // The devices, and the CPU last - the arrangement the diffusion engine
        // builds, where the CPU is the fallback for anything no device will
        // take. It matters here beyond that: the placement pass treats the last
        // backend as the home for unallocated graph inputs, and it gives the
        // smoothing pass somewhere to move work to, so the graph is placed by
        // rules rather than by pins alone.
        gk_backend_t sched_backends[MAX_DEVICES + 1];
        int n_sched = 0;
        for (int i = 0; i < n_dev; ++i) {
            sched_backends[n_sched++] = backends[i];
        }
        sched_backends[n_sched++] = cpu;

        struct gk_sched * sched = gk_sched_new(sched_backends, n_sched);
        if (sched == NULL) {
            CHECK_MSG(false, "%s: could not create a scheduler", name);
        } else {
            const int pinned = pin_in_blocks(sched, backends, n_dev, graph);

            int splits_after_alloc = -1;
            if (prealloc) {
                if (!gk_sched_alloc_graph(sched, graph)) {
                    CHECK_MSG(false, "%s: alloc_graph failed", name);
                }
                splits_after_alloc = gk_sched_n_splits(sched);
            }

            // Written after allocation, as the engine writes its inputs.
            for (int i = 0; i < n_gpu_in; ++i) {
                gk_backend_tensor_set(gpu_in[i], cpu_in[i]->data, 0, gk_nbytes(gpu_in[i]));
            }

            const enum gk_status status = gk_sched_graph_compute(sched, graph);

            // The invariant behind the whole arrangement: computing an
            // allocated graph must run the placement that allocation was built
            // from. A different split count means it was placed again, against
            // storage that already existed - and the numbers below can still
            // come out close enough to look healthy when it does, which is why
            // this is checked rather than inferred from the output.
            if (prealloc) {
                CHECK_MSG(gk_sched_n_splits(sched) == splits_after_alloc,
                          "%s: placement changed between alloc and compute (%d splits -> %d)",
                          name, splits_after_alloc, gk_sched_n_splits(sched));
            }
            CHECK_MSG(status == GK_STATUS_SUCCESS,
                      "%s: compute failed with status %d", name, (int) status);

            if (status == GK_STATUS_SUCCESS) {
                float * got = (float *) malloc((size_t) n_out * sizeof(float));
                gk_backend_tensor_get(gpu_out, got, 0, (size_t) n_out * sizeof(float));

                // Relative, because these graphs do not agree on scale: the
                // attention block's outputs are around one and the in-place
                // chain's are in the thousands, having gone through eleven
                // matmuls. The two devices reduce in different orders, so what
                // is being allowed for is proportional drift - an absolute
                // bound would either fail the large graph or excuse the small
                // one.
                double max_rel = 0.0;
                int    bad     = 0;
                for (int64_t i = 0; i < n_out; ++i) {
                    const double diff  = fabs((double) got[i] - (double) expected[i]);
                    const double scale = 1.0 + fabs((double) expected[i]);
                    const double rel   = diff / scale;
                    if (rel > max_rel) {
                        max_rel = rel;
                    }
                    if (rel > tol) {
                        bad++;
                    }
                }
                printf("    %-14s %-14s %d nodes, %d pinned, %d splits, max rel error %.6g\n",
                       name, prealloc ? "(pre-alloc)" : "(compute only)",
                       gk_graph_n_nodes(graph), pinned,
                       gk_sched_n_splits(sched), max_rel);
                CHECK_MSG(bad == 0,
                          "%s: %d of %lld outputs differ from the CPU by more than %g relative (worst %.6g)",
                          name, bad, (long long) n_out, tol, max_rel);
                free(got);
            }
            gk_sched_free(sched);
        }
    }

    for (int i = 0; i < MAX_BUILDER_INPUTS; ++i) {
        if (owned[i].buffer != NULL) {
            gk_backend_buffer_free(owned[i].buffer);
        }
    }
    free(expected);
    gk_free(ctx);
    gk_free(params);
    gk_free(cpu_ctx);
    gk_backend_free(cpu);
    return 0;
}

// The same scheduler, twice, with a larger graph the second time.
//
// This is the shape of a diffusion sampler rather than an edge case: the first
// step runs one batch, and every step after it runs two, because classifier-
// free guidance evaluates the conditioned and unconditioned prompts together.
// The buffers were sized for the first graph, and the second does not fit them.
//
// Growing is the easy half. The half that goes wrong quietly is what happens if
// the pass that hands out addresses does not know how big the buffer is: it
// keeps counting past the end, reports success, and the first kernel to write
// there lands outside the allocation. On a device that is either a fault or -
// if the driver happens to have something mapped after it - a wrong answer.
static void test_graph_that_grows(gk_backend_t * backends, gk_device_t * devices, int n_dev) {
    GK_UNUSED(devices);

    printf("  the same scheduler with a larger graph the second time\n");

    gk_backend_t cpu = gk_backend_cpu_init(0);
    struct gk_sched * sched = gk_sched_new(backends, n_dev);
    if (sched == NULL) {
        CHECK_MSG(false, "could not create a scheduler");
        gk_backend_free(cpu);
        return;
    }

    // Second run is eight times the first, which no amount of reuse inside the
    // first graph's buffers can absorb.
    const int64_t batches[2] = { 4, 32 };

    for (int run = 0; run < 2; ++run) {
        const int64_t dim = 256;
        const int64_t tok = batches[run];

        struct gk_ctx * ctx = gk_init((struct gk_init_params) {
            .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
        });

        struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, dim, tok);
        struct gk_tensor * w = gk_new_tensor_2d(ctx, GK_TYPE_F32, dim, dim);

        gk_backend_buffer_type_t buft = gk_backend_get_default_buffer_type(backends[0]);
        const size_t align = gk_backend_buft_get_alignment(buft);

        gk_backend_buffer_t xbuf = gk_backend_buft_alloc_buffer(buft, gk_pad_size(gk_nbytes(x), align));
        gk_backend_buffer_t wbuf = gk_backend_buft_alloc_buffer(buft, gk_pad_size(gk_nbytes(w), align));
        if (xbuf == NULL || wbuf == NULL) {
            CHECK_MSG(false, "run %d: could not allocate inputs", run);
            gk_backend_buffer_free(xbuf);
            gk_backend_buffer_free(wbuf);
            gk_free(ctx);
            continue;
        }

        x->data = (char *) gk_backend_buffer_get_base(xbuf);
        gk_backend_buffer_init_tensor(xbuf, x);
        w->data = (char *) gk_backend_buffer_get_base(wbuf);
        gk_backend_buffer_init_tensor(wbuf, w);

        float * xs = (float *) malloc((size_t) dim * tok * sizeof(float));
        float * ws = (float *) malloc((size_t) dim * dim * sizeof(float));
        g_rng = 0x1234567800000000ull + (uint64_t) run;
        for (int64_t i = 0; i < dim * tok; ++i) { xs[i] = frand(); }
        for (int64_t i = 0; i < dim * dim; ++i) { ws[i] = frand() * 0.05f; }
        gk_backend_tensor_set(w, ws, 0, gk_nbytes(w));

        struct gk_tensor * t = x;
        for (int layer = 0; layer < 8; ++layer) {
            t = gk_rms_norm(ctx, t, 1e-5f);
            t = gk_mul_mat(ctx, w, t);
            t = gk_silu(ctx, t);
        }
        gk_set_output(t);

        struct gk_cgraph * graph = gk_new_graph(ctx);
        gk_build_forward_expand(graph, t);

        gk_sched_reset(sched);
        pin_in_blocks(sched, backends, n_dev, graph);

        if (!gk_sched_alloc_graph(sched, graph)) {
            CHECK_MSG(false, "run %d: alloc_graph failed", run);
        } else {
            gk_backend_tensor_set(x, xs, 0, gk_nbytes(x));

            const enum gk_status status = gk_sched_graph_compute(sched, graph);
            CHECK_MSG(status == GK_STATUS_SUCCESS,
                      "run %d (batch %lld): compute failed with status %d",
                      run, (long long) tok, (int) status);

            if (status == GK_STATUS_SUCCESS) {
                // Every output finite is the check that matters: writing past a
                // buffer shows up here as whatever the neighbouring allocation
                // held, and that is reliably not a number this graph produces.
                float * got = (float *) malloc((size_t) dim * tok * sizeof(float));
                gk_backend_tensor_get(t, got, 0, (size_t) dim * tok * sizeof(float));

                int bad = 0;
                for (int64_t i = 0; i < dim * tok; ++i) {
                    if (!(got[i] > -1e6f && got[i] < 1e6f)) {
                        bad++;
                    }
                }
                CHECK_MSG(bad == 0, "run %d (batch %lld): %d outputs are not finite",
                          run, (long long) tok, bad);
                free(got);
            }
        }

        free(ws);
        free(xs);
        gk_backend_buffer_free(xbuf);
        gk_backend_buffer_free(wbuf);
        gk_free(ctx);
    }

    gk_sched_free(sched);
    gk_backend_free(cpu);
}

int main(void) {
    gk_backend_t backends[MAX_DEVICES];
    gk_device_t  devices[MAX_DEVICES];
    int          n_dev = 0;

    const int n_registered = gk_device_count();
    for (int i = 0; i < n_registered && n_dev < MAX_DEVICES; ++i) {
        gk_device_t dev = gk_device_get(i);
        const enum gk_device_type type = gk_device_type_of(dev);
        if (type != GK_DEVICE_TYPE_GPU && type != GK_DEVICE_TYPE_IGPU) {
            continue;
        }

        gk_backend_t backend = gk_device_init_backend(dev);
        if (backend == NULL) {
            printf("  could not initialize a backend for %s; skipping it\n", gk_device_name(dev));
            continue;
        }

        size_t free_mem = 0, total_mem = 0;
        gk_device_memory(dev, &free_mem, &total_mem);
        printf("  %s: %s (%zu MiB free of %zu MiB)\n",
               gk_device_name(dev), gk_device_description(dev),
               free_mem >> 20, total_mem >> 20);

        devices[n_dev]  = dev;
        backends[n_dev] = backend;
        n_dev++;
    }

    if (n_dev == 0) {
        fprintf(stderr, "CUDA-family backend was built but no usable GPU was discovered\n");
        return 1;
    }

    float * x_src    = (float *) malloc((size_t) g_dim * g_tok * sizeof(float));
    float * w_src    = (float *) malloc((size_t) N_LAYERS * g_dim * g_dim * sizeof(float));
    float * expected = (float *) malloc((size_t) g_dim * g_tok * sizeof(float));
    fill_reference(x_src, w_src);

    if (run_on_cpu(x_src, w_src, expected) != 0) {
        fprintf(stderr, "the CPU reference run failed\n");
        return 1;
    }

    printf("layered graph with its weights spread across the devices\n");
    run_across(backends, devices, n_dev, x_src, w_src, expected);

    test_device_to_device_copy(backends, devices, n_dev);
    test_graph_that_grows(backends, devices, n_dev);

    printf("composite graphs, pinned across the devices in blocks\n");
    run_pinned_composite(backends, devices, n_dev, "attention",  build_attention,  4e-3, false);
    run_pinned_composite(backends, devices, n_dev, "unet skips", build_unet_skips, 4e-2, false);
    run_pinned_composite(backends, devices, n_dev, "attention",  build_attention,  4e-3, true);
    run_pinned_composite(backends, devices, n_dev, "unet skips", build_unet_skips, 4e-2, true);
    run_pinned_composite(backends, devices, n_dev, "in-place",   build_inplace_chain, 4e-3, true);

    if (n_dev == 1) {
        printf("  only one GPU on this machine - the crossing itself was not exercised\n");
    }

    for (int i = 0; i < n_dev; ++i) {
        gk_backend_free(backends[i]);
    }
    free(expected);
    free(w_src);
    free(x_src);

    printf("%d/%d checks failed\n", g_fails, g_checks);
    return g_fails == 0 ? 0 : 1;
}
