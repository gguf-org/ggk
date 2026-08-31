// Scheduler tests.
//
// The scheduler's job only becomes visible with more than one backend, and a
// machine running this suite usually has one device. So the second backend
// here is a test double, in two flavours: one that shares the CPU's memory and
// refuses a nominated op, and one that also has a buffer type of its own and
// calls it device memory.
//
// Between them they drive everything that would otherwise be written blind and
// stay unexercised until someone ran the suite on a machine with a GPU - the
// assignment rules, the smoothing pass, where the splits land, which values
// have to be staged across a memory boundary, and whether the pieces still
// compose into the right answer.
//
// What remains untested here is a transfer that really crosses a bus: both
// doubles are backed by host memory, because the kernels that run on them are
// the CPU's. Everything above that transfer is the same code either way.

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
// a backend that refuses one op
// --------------------------------------------------------------------------

struct picky_ctx {
    gk_backend_t inner;
    enum gk_op   refuse;
};

static const char * picky_name(gk_backend_t b) {
    GK_UNUSED(b);
    return "picky";
}

static void picky_free(gk_backend_t b) {
    struct picky_ctx * c = (struct picky_ctx *) b->context;
    gk_backend_free(c->inner);
    free(c);
    free(b);
}

static gk_backend_buffer_type_t picky_buft(gk_backend_t b) {
    struct picky_ctx * c = (struct picky_ctx *) b->context;
    return gk_backend_get_default_buffer_type(c->inner);
}

static enum gk_status picky_compute(gk_backend_t b, struct gk_cgraph * g) {
    struct picky_ctx * c = (struct picky_ctx *) b->context;
    return gk_backend_graph_compute(c->inner, g);
}

static bool picky_supports(gk_backend_t b, const struct gk_tensor * op) {
    struct picky_ctx * c = (struct picky_ctx *) b->context;
    if (op->op == c->refuse) {
        return false;
    }
    return gk_backend_supports_op(c->inner, op);
}

static gk_backend_t picky_init(enum gk_op refuse) {
    struct picky_ctx * c = (struct picky_ctx *) malloc(sizeof(struct picky_ctx));
    c->inner  = gk_backend_cpu_init(1);
    c->refuse = refuse;

    // Zeroed, not just filled in: the interface has optional entries beyond
    // the five below and gk reads them, so a backend that leaves them as
    // whatever malloc returned is a crash waiting for the first caller that
    // asks something the backend did not think about.
    gk_backend_t b = (gk_backend_t) calloc(1, sizeof(struct gk_backend));

    b->iface.get_name                = picky_name;
    b->iface.free                    = picky_free;
    b->iface.get_default_buffer_type = picky_buft;
    b->iface.graph_compute           = picky_compute;
    b->iface.supports_op             = picky_supports;
    b->context                       = c;
    b->device                        = NULL; // a test double, not a device

    return b;
}

// --------------------------------------------------------------------------
// a backend whose memory is somewhere else
//
// The double above shares the CPU's buffer type, so a graph split across it
// needs no data movement. This one has its own, and reports it as device
// memory: the scheduler must therefore stage every value that crosses a
// boundary into the consuming backend's memory before it can be read.
//
// The storage is still host memory - the kernels have to run on it, and the
// pool that runs them is the CPU's - but nothing in gk is told that. What is
// exercised is the whole staging path: which values need copying, where the
// copies are allocated, and whether the pieces still compose into the same
// computation. That is the part that would otherwise stay unexercised on a
// machine with no GPU, which is every machine this suite runs on today.
// --------------------------------------------------------------------------

static const char * remote_buft_name(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return "remote";
}

static void remote_buffer_free(gk_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * remote_buffer_get_base(gk_backend_buffer_t buffer) {
    return buffer->context;
}

static void remote_buffer_set(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                              const void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void remote_buffer_get(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                              void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void remote_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const struct gk_backend_buffer_i g_remote_buffer_iface = {
    /* .free_buffer   = */ remote_buffer_free,
    /* .get_base      = */ remote_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .set_tensor    = */ remote_buffer_set,
    /* .get_tensor    = */ remote_buffer_get,
    /* .clear         = */ remote_buffer_clear,
    /* .memset_tensor = */ NULL,
    /* .cpy_tensor    = */ NULL,
};

static gk_backend_buffer_t remote_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    void * base = malloc(size);
    if (base == NULL) {
        return NULL;
    }
    return gk_backend_buffer_init(buft, &g_remote_buffer_iface, base, size);
}

static size_t remote_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return 64;
}

static struct gk_backend_buffer_type g_remote_buft = {
    .iface = {
        .get_name       = remote_buft_name,
        .alloc_buffer   = remote_buft_alloc,
        .get_alignment  = remote_buft_alignment,
        .get_alloc_size = NULL,
        .is_host        = NULL, // the point of the double: not host memory
    },
    .context = NULL,
    .device  = NULL,
};

struct remote_ctx {
    gk_backend_t inner;
};

static const char * remote_name(gk_backend_t b) {
    GK_UNUSED(b);
    return "remote";
}

static void remote_free(gk_backend_t b) {
    struct remote_ctx * c = (struct remote_ctx *) b->context;
    gk_backend_free(c->inner);
    free(c);
    free(b);
}

static gk_backend_buffer_type_t remote_buft(gk_backend_t b) {
    GK_UNUSED(b);
    return &g_remote_buft;
}

static enum gk_status remote_compute(gk_backend_t b, struct gk_cgraph * g) {
    struct remote_ctx * c = (struct remote_ctx *) b->context;
    return gk_backend_graph_compute(c->inner, g);
}

static bool remote_supports_op(gk_backend_t b, const struct gk_tensor * op) {
    struct remote_ctx * c = (struct remote_ctx *) b->context;
    // Refuses the same op the picky double does, so the graph is forced to
    // cross the memory boundary once per layer.
    if (op->op == GK_OP_UNARY) {
        return false;
    }
    return gk_backend_supports_op(c->inner, op);
}

static bool remote_supports_buft(gk_backend_t b, gk_backend_buffer_type_t buft) {
    GK_UNUSED(b);
    return buft == &g_remote_buft;
}

static gk_backend_t remote_init(void) {
    struct remote_ctx * c = (struct remote_ctx *) malloc(sizeof(struct remote_ctx));
    c->inner = gk_backend_cpu_init(1);

    gk_backend_t b = (gk_backend_t) calloc(1, sizeof(struct gk_backend));

    b->iface.get_name                = remote_name;
    b->iface.free                    = remote_free;
    b->iface.get_default_buffer_type = remote_buft;
    b->iface.graph_compute           = remote_compute;
    b->iface.supports_op             = remote_supports_op;
    b->iface.supports_buft           = remote_supports_buft;
    b->context                       = c;

    return b;
}

// --------------------------------------------------------------------------

// Alternates ops so that refusing one of them forces repeated boundaries
// rather than a single clean cut.
static struct gk_tensor * build_graph(struct gk_ctx * ctx, struct gk_ctx * inputs,
                                      int n_layers, struct gk_tensor ** out_x) {
    g_rng = 0xc0ffee1234ull;

    const int64_t d = 64;
    const int64_t n = 8;

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
        t = gk_rms_norm(ctx, t, 1e-5f);
        t = gk_mul_mat(ctx, w, t);
        t = gk_silu(ctx, t);   // the op the picky backend will refuse
        t = gk_scale(ctx, t, 0.5f);
    }

    gk_set_output(t);

    if (out_x != NULL) {
        *out_x = x;
    }
    return t;
}

static float * run(gk_backend_t * backends, int n_backends, int n_layers,
                   int * n_splits_out, int64_t * n_out) {
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_ctx * inputs = gk_init((struct gk_init_params) {
        .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    struct gk_tensor * out = build_graph(ctx, inputs, n_layers, NULL);

    struct gk_cgraph * g = gk_new_graph(ctx);
    gk_build_forward_expand(g, out);

    struct gk_sched * sched = gk_sched_new(backends, n_backends);
    if (sched == NULL) {
        gk_free(ctx);
        gk_free(inputs);
        return NULL;
    }

    if (!gk_sched_reserve(sched, g)) {
        printf("  reserve failed\n");
        gk_sched_free(sched);
        gk_free(ctx);
        gk_free(inputs);
        return NULL;
    }

    const enum gk_status st = gk_sched_graph_compute(sched, g);
    if (st != GK_STATUS_SUCCESS) {
        printf("  compute failed with status %d\n", (int) st);
        gk_sched_free(sched);
        gk_free(ctx);
        gk_free(inputs);
        return NULL;
    }

    if (n_splits_out != NULL) {
        *n_splits_out = gk_sched_n_splits(sched);
    }

    *n_out = gk_nelements(out);
    float * copy = (float *) malloc((size_t) *n_out * sizeof(float));
    memcpy(copy, out->data, (size_t) *n_out * sizeof(float));

    gk_sched_free(sched);
    gk_free(ctx);
    gk_free(inputs);

    return copy;
}

static void test_single_backend(void) {
    printf("one backend: a single split, and the right answer\n");

    gk_backend_t cpu = gk_backend_cpu_init(2);
    gk_backend_t list[1] = { cpu };

    int splits = -1;
    int64_t n = 0;
    float * got = run(list, 1, 4, &splits, &n);

    CHECK_MSG(got != NULL, "scheduled run failed");
    CHECK_MSG(splits == 1, "one backend should give one split, got %d", splits);

    // compare against running the same graph without a scheduler
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * out = build_graph(ctx, ctx, 4, NULL);
    struct gk_cgraph * g = gk_new_graph(ctx);
    gk_build_forward_expand(g, out);
    CHECK_MSG(gk_graph_compute(g, 1) == GK_STATUS_SUCCESS, "plain run failed");

    if (got != NULL) {
        int64_t bad = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (memcmp(&got[i], &((const float *) out->data)[i], sizeof(float)) != 0) {
                bad++;
            }
        }
        CHECK_MSG(bad == 0, "%lld/%lld values differ from the unscheduled run",
                  (long long) bad, (long long) n);
        free(got);
    }

    gk_free(ctx);
    gk_backend_free(cpu);
}

static void test_two_backends(void) {
    printf("two backends: work lands where it is supported\n");

    const int n_layers = 4;

    // reference: one plain CPU backend
    gk_backend_t cpu = gk_backend_cpu_init(1);
    gk_backend_t one[1] = { cpu };

    int64_t n_ref = 0;
    float * ref = run(one, 1, n_layers, NULL, &n_ref);
    CHECK_MSG(ref != NULL, "reference run failed");

    // Backend 0 refuses silu, so every silu has to land on backend 1 and the
    // graph cuts around each one.
    gk_backend_t picky = picky_init(GK_OP_UNARY);
    gk_backend_t other = gk_backend_cpu_init(1);
    gk_backend_t two[2] = { picky, other };

    int splits = -1;
    int64_t n = 0;
    float * got = run(two, 2, n_layers, &splits, &n);

    CHECK_MSG(got != NULL, "two-backend run failed");
    CHECK_MSG(splits > 1, "expected the graph to be cut, got %d split(s)", splits);

    printf("    %d layers produced %d splits across 2 backends\n", n_layers, splits);

    if (got != NULL && ref != NULL) {
        CHECK_MSG(n == n_ref, "output sizes differ (%lld vs %lld)",
                  (long long) n, (long long) n_ref);

        int64_t bad = 0;
        for (int64_t i = 0; i < n && i < n_ref; ++i) {
            if (memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
                bad++;
            }
        }
        CHECK_MSG(bad == 0,
                  "%lld/%lld values differ once the graph is split - the pieces "
                  "did not compose back into the same computation",
                  (long long) bad, (long long) n);
    }

    free(got);
    free(ref);

    gk_backend_free(picky);
    gk_backend_free(other);
    gk_backend_free(cpu);
}

// The assignment rule that matters most once devices are real: a node whose
// operand already lives in a particular backend's memory should run there.
// Weights are the case - they are the largest thing in the graph and moving
// them is the one cost worth reorganising everything else to avoid.
//
// Here the weight is placed in an explicit backend buffer rather than a
// context, which is what a loaded model looks like. Every matmul reading it
// must be assigned to that backend even though the surrounding ops drifted
// elsewhere, and the graph should therefore cut back and forth rather than
// settling on one side.
static void test_weights_pin_nodes(void) {
    printf("a weight in a backend's memory pulls its matmul there\n");

    const int n_layers = 4;

    gk_backend_t picky = picky_init(GK_OP_UNARY);
    gk_backend_t other = gk_backend_cpu_init(1);
    gk_backend_t two[2] = { picky, other };

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_ctx * inputs = gk_init((struct gk_init_params) {
        .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = true,
    });

    const int64_t d = 64, n = 8;

    struct gk_tensor * x = gk_new_tensor_2d(inputs, GK_TYPE_F32, d, n);
    struct gk_tensor * w = gk_new_tensor_2d(inputs, GK_TYPE_F32, d, d);

    // Give the inputs real storage in a backend buffer, the way a model loader
    // would, so they carry a buffer and the scheduler can see where they live.
    gk_backend_buffer_type_t buft = gk_backend_get_default_buffer_type(picky);
    const size_t need = gk_nbytes(x) + gk_nbytes(w) + 2 * gk_backend_buft_get_alignment(buft);

    gk_backend_buffer_t buf = gk_backend_buft_alloc_buffer(buft, need);
    CHECK_MSG(buf != NULL, "weight buffer allocation failed");
    if (buf == NULL) {
        return;
    }

    char * base = (char *) gk_backend_buffer_get_base(buf);
    x->data = base;
    gk_backend_buffer_init_tensor(buf, x);
    w->data = base + gk_pad_size(gk_nbytes(x), gk_backend_buft_get_alignment(buft));
    gk_backend_buffer_init_tensor(buf, w);

    CHECK_MSG(x->buffer == buf && w->buffer == buf, "tensors did not record their buffer");

    g_rng = 0xabcdef01ull;
    for (int64_t i = 0; i < d * n; ++i) {
        ((float *) x->data)[i] = frand();
    }
    for (int64_t i = 0; i < d * d; ++i) {
        ((float *) w->data)[i] = frand() * 0.1f;
    }

    struct gk_tensor * t = x;
    for (int l = 0; l < n_layers; ++l) {
        t = gk_rms_norm(ctx, t, 1e-5f);
        t = gk_mul_mat(ctx, w, t);
        t = gk_silu(ctx, t);
        t = gk_scale(ctx, t, 0.5f);
    }
    gk_set_output(t);

    struct gk_cgraph * g = gk_new_graph(ctx);
    gk_build_forward_expand(g, t);

    struct gk_sched * sched = gk_sched_new(two, 2);
    CHECK_MSG(gk_sched_reserve(sched, g), "reserve failed");
    CHECK_MSG(gk_sched_graph_compute(sched, g) == GK_STATUS_SUCCESS, "compute failed");

    const int splits = gk_sched_n_splits(sched);
    printf("    %d layers with a pinned weight produced %d splits\n", n_layers, splits);

    // Each layer's silu is forced off backend 0 and each matmul is pulled back
    // to it, so the graph must alternate rather than settle.
    CHECK_MSG(splits >= n_layers,
              "expected at least one cut per layer (%d), got %d - the pinned "
              "weight did not pull its matmul back", n_layers, splits);

    gk_sched_free(sched);
    gk_backend_buffer_free(buf);
    gk_free(ctx);
    gk_free(inputs);
    gk_backend_free(picky);
    gk_backend_free(other);
}

// The case the whole staging path exists for: two backends whose memories are
// not the same one. Every value crossing a boundary has to be copied into the
// consuming backend's memory first, and the answer has to come out unchanged.
static void test_separate_memories(void) {
    printf("two memories: values are staged across the boundary\n");

    const int n_layers = 4;

    gk_backend_t cpu = gk_backend_cpu_init(1);
    gk_backend_t one[1] = { cpu };

    int64_t n_ref = 0;
    float * ref = run(one, 1, n_layers, NULL, &n_ref);
    CHECK_MSG(ref != NULL, "reference run failed");

    // Backend 0 has its own memory and refuses silu; backend 1 is the host.
    gk_backend_t remote = remote_init();
    gk_backend_t host   = gk_backend_cpu_init(1);
    gk_backend_t two[2] = { remote, host };

    int splits = -1;
    int64_t n = 0;
    float * got = run(two, 2, n_layers, &splits, &n);

    CHECK_MSG(got != NULL, "split-memory run failed");
    CHECK_MSG(splits > 1, "expected the graph to be cut, got %d split(s)", splits);

    printf("    %d layers produced %d splits across two memories\n", n_layers, splits);

    if (got != NULL && ref != NULL) {
        int64_t bad = 0;
        for (int64_t i = 0; i < n && i < n_ref; ++i) {
            if (memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
                bad++;
            }
        }
        CHECK_MSG(bad == 0,
                  "%lld/%lld values differ once the graph is split across two "
                  "memories - a staged input did not arrive",
                  (long long) bad, (long long) n);
    }

    free(got);
    free(ref);

    gk_backend_free(remote);
    gk_backend_free(host);
    gk_backend_free(cpu);
}

// Input tensors are filled directly by the engine between graph allocation
// and execution. They must therefore remain in host-addressable memory even
// when their first consumer runs on a device. The scheduler stages the value
// into device memory for that consumer; allocating the input itself on the
// device makes the caller write through a device pointer. This is the exact
// shape used by llama's KV-cache index inputs.
static void test_inputs_stay_on_host(void) {
    printf("graph inputs stay host-addressable before device execution\n");

    gk_backend_t remote = remote_init();
    gk_backend_t host   = gk_backend_cpu_init(1);
    gk_backend_t two[2] = { remote, host };

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
    });

    struct gk_tensor * input = gk_new_tensor_1d(ctx, GK_TYPE_F32, 64);
    gk_set_input(input);
    struct gk_tensor * out = gk_scale(ctx, input, 2.0f);
    gk_set_output(out);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    struct gk_sched * sched = gk_sched_new(two, 2);
    CHECK_MSG(sched != NULL, "scheduler creation failed");
    CHECK_MSG(sched != NULL && gk_sched_alloc_graph(sched, graph),
              "input graph allocation failed");

    if (input->buffer != NULL) {
        CHECK_MSG(gk_backend_buffer_is_host(input->buffer),
                  "input followed its device consumer into non-host memory");
        CHECK_MSG(gk_sched_get_tensor_backend(sched, input) == host,
                  "scheduler reports the device as the input's backend");
    } else {
        CHECK_MSG(false, "input was not allocated");
    }

    gk_sched_free(sched);
    gk_free(ctx);
    gk_backend_free(remote);
    gk_backend_free(host);
}

// A backend that supports nothing at all must be reported, not worked around.
static void test_unsupported_op(void) {
    printf("an op nothing supports is refused, not guessed at\n");

    gk_backend_t picky = picky_init(GK_OP_RMS_NORM);
    gk_backend_t list[1] = { picky };

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = false,
    });

    g_rng = 1;
    struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 4);
    for (int i = 0; i < 64 * 4; ++i) {
        ((float *) x->data)[i] = frand();
    }

    struct gk_tensor * t = gk_rms_norm(ctx, x, 1e-5f);
    struct gk_cgraph * g = gk_new_graph(ctx);
    gk_build_forward_expand(g, t);

    struct gk_sched * sched = gk_sched_new(list, 1);
    CHECK_MSG(sched != NULL, "scheduler creation failed");

    CHECK_MSG(!gk_sched_reserve(sched, g),
              "reserve should fail when no backend supports an op in the graph");

    gk_sched_free(sched);
    gk_free(ctx);
    gk_backend_free(picky);
}

int main(void) {
    printf("gk scheduler tests\n\n");

    test_single_backend();
    test_two_backends();
    test_separate_memories();
    test_inputs_stay_on_host();
    test_weights_pin_nodes();
    test_unsupported_op();

    printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
