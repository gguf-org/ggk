// The CUDA/HIP backend: devices, memory, and running a graph on them.
//
// The kernels are next door; this file is the part that makes them reachable
// through gk's backend interface - what a buffer of device memory is, how a
// tensor's bytes get in and out of one, and what happens when the scheduler
// hands over a range of nodes.
//
// Three things here are worth stating outright because they are the whole
// difference between this and the CPU backend:
//
//   * Device memory is not host memory. A tensor in it has a `data` pointer
//     that must never be dereferenced on the host, which is why gk's
//     tensor_set/tensor_get exist and why the engine goes through them.
//
//   * Work is queued, not done. Every kernel below is launched on a stream and
//     returns immediately; nothing may read a result before `synchronize`.
//     The scheduler knows this and synchronizes before it copies across a
//     split boundary.
//
//   * One stream per backend, and a backend belongs to one device. Two
//     backends on the same device share its memory and can copy between
//     themselves without going through the host; two on different devices can
//     too, when the driver says the pair can see each other.

#include "gk_cuda_ops.cuh"

#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

// Device memory is handed out in multiples of this. Coalesced loads want their
// rows aligned and the driver's own allocations are far coarser than this
// anyway, so it costs nothing to promise.
#define GK_CUDA_ALIGN 128

// --------------------------------------------------------------------------
// per-device state
//
// Built once, at discovery, and shared by every backend and buffer on that
// device. The buffer types in particular have to be stable addresses: the
// scheduler compares buffer types by pointer to decide what lives where.
// --------------------------------------------------------------------------

struct gk_cuda_device_ctx {
    int  index;
    char name[32];
    char description[256];

    size_t total_memory;
    bool   integrated;
    int    n_sm;         // multiprocessors; a launcher sizing a grid wants it
    int    cc;           // compute capability; the tensor-core path needs 8.0
    int    smem_max;     // shared memory a block may opt in to, past the 48 KB default

    // What gk_device_features hands back. Built once at registration and kept
    // here rather than formatted on demand, because the caller is promised
    // pointers that outlive the call.
    struct gk_feature features[6];
    char              cc_str[16];
    char              n_sm_str[16];
    char              smem_str[16];

    struct gk_backend_buffer_type buft;
    struct gk_backend_buffer_type host_buft;
    struct gk_device              device;
};

static struct gk_cuda_device_ctx g_cuda_devices[GK_CUDA_MAX_DEVICES];
static int  g_cuda_n_devices;
static bool g_cuda_discovered;

// --------------------------------------------------------------------------
// device buffers
// --------------------------------------------------------------------------

struct gk_cuda_buffer_ctx {
    int    device;
    void * base;
    bool   owned; // false for a buffer wrapping memory somebody else allocated
};

static void gk_cuda_buffer_free(gk_backend_buffer_t buffer) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;
    if (ctx == NULL) {
        return;
    }
    if (ctx->owned) {
        GK_CUDA_CHECK(gkSetDevice(ctx->device));
        GK_CUDA_CHECK(gkFree(ctx->base));
    }
    free(ctx);
}

static void * gk_cuda_buffer_get_base(gk_backend_buffer_t buffer) {
    return ((struct gk_cuda_buffer_ctx *) buffer->context)->base;
}

static void gk_cuda_buffer_set_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                      const void * data, size_t offset, size_t size) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    // Synchronous by contract: the caller may free `data` the moment this
    // returns, so the copy cannot be left in flight.
    //
    // The wait is on the per-thread stream rather than the legacy default one,
    // and that is the whole point. A plain gkMemcpy is ordered against every
    // other stream on the device, so a model loader filling tensors from
    // several threads has them queue behind one another instead of overlapping.
    // Same contract, same cost for a single thread, several times the
    // throughput for a loader that uses more than one.
    GK_CUDA_CHECK(gkMemcpyAsync((char *) tensor->data + offset, data, size,
                                gkMemcpyHostToDevice, gkStreamPerThread));
    GK_CUDA_CHECK(gkStreamSynchronize(gkStreamPerThread));
}

static void gk_cuda_buffer_get_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                      void * data, size_t offset, size_t size) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    GK_CUDA_CHECK(gkMemcpyAsync(data, (const char *) tensor->data + offset, size,
                                gkMemcpyDeviceToHost, gkStreamPerThread));
    GK_CUDA_CHECK(gkStreamSynchronize(gkStreamPerThread));
}

static void gk_cuda_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    GK_CUDA_CHECK(gkMemset(ctx->base, value, buffer->size));
}

static void gk_cuda_buffer_memset_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                         uint8_t value, size_t offset, size_t size) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    GK_CUDA_CHECK(gkMemset((char *) tensor->data + offset, value, size));
}

// The device-to-device path. Same device: a plain device copy. Different
// devices: a peer copy, if the driver has told us the pair can see each other.
// Anything else returns false and the caller stages it through the host.
static bool gk_cuda_buffer_cpy_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * src,
                                      struct gk_tensor * dst) {
    struct gk_cuda_buffer_ctx * dctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    if (src->buffer == NULL) {
        return false;
    }

    const gk_backend_buffer_type_t src_buft = gk_backend_buffer_get_type(src->buffer);

    // Pinned host memory is readable by the device directly, so a copy out of
    // one is a host-to-device copy rather than a staged round trip.
    if (gk_backend_buft_is_host(src_buft)) {
        GK_CUDA_CHECK(gkSetDevice(dctx->device));
        GK_CUDA_CHECK(gkMemcpy(dst->data, src->data, gk_nbytes(src), gkMemcpyHostToDevice));
        return true;
    }

    // Is the source one of ours?
    int src_device = -1;
    for (int i = 0; i < g_cuda_n_devices; ++i) {
        if (&g_cuda_devices[i].buft == src_buft) {
            src_device = g_cuda_devices[i].index;
            break;
        }
    }
    if (src_device < 0) {
        return false;
    }

    if (src_device == dctx->device) {
        GK_CUDA_CHECK(gkSetDevice(dctx->device));
        GK_CUDA_CHECK(gkMemcpy(dst->data, src->data, gk_nbytes(src), gkMemcpyDeviceToDevice));
        return true;
    }

    int can_peer = 0;
    GK_CUDA_CHECK(gkDeviceCanAccessPeer(&can_peer, dctx->device, src_device));
    if (!can_peer) {
        return false;
    }

    GK_CUDA_CHECK(gkSetDevice(dctx->device));
    GK_CUDA_CHECK(gkMemcpyPeerAsync(dst->data, dctx->device, src->data, src_device,
                                    gk_nbytes(src), 0));
    GK_CUDA_CHECK(gkStreamSynchronize(0));
    return true;
}

static const struct gk_backend_buffer_i g_cuda_buffer_iface = {
    /* .free_buffer   = */ gk_cuda_buffer_free,
    /* .get_base      = */ gk_cuda_buffer_get_base,
    /* .init_tensor   = */ NULL, // nothing per tensor; the pointer is enough
    /* .set_tensor    = */ gk_cuda_buffer_set_tensor,
    /* .get_tensor    = */ gk_cuda_buffer_get_tensor,
    /* .clear         = */ gk_cuda_buffer_clear,
    /* .memset_tensor = */ gk_cuda_buffer_memset_tensor,
    /* .cpy_tensor    = */ gk_cuda_buffer_cpy_tensor,
};

// --------------------------------------------------------------------------
// the device buffer type
// --------------------------------------------------------------------------

static const char * gk_cuda_buft_name(gk_backend_buffer_type_t buft) {
    return ((struct gk_cuda_device_ctx *) buft->context)->name;
}

static gk_backend_buffer_t gk_cuda_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    struct gk_cuda_device_ctx * dev = (struct gk_cuda_device_ctx *) buft->context;

    struct gk_cuda_buffer_ctx * ctx =
        (struct gk_cuda_buffer_ctx *) malloc(sizeof(struct gk_cuda_buffer_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->device = dev->index;
    ctx->owned  = true;

    GK_CUDA_CHECK(gkSetDevice(dev->index));

    const gkError_t err = gkMalloc(&ctx->base, size);
    if (err != gkSuccess) {
        gk_logf("gk %s: failed to allocate %zu bytes on %s: %s\n",
                GK_CUDA_BACKEND_NAME, size, dev->name, gkGetErrorString(err));
        free(ctx);
        return NULL;
    }

    gk_backend_buffer_t buffer = gk_backend_buffer_init(buft, &g_cuda_buffer_iface, ctx, size);
    if (buffer == NULL) {
        GK_CUDA_CHECK(gkFree(ctx->base));
        free(ctx);
        return NULL;
    }

    return buffer;
}

static size_t gk_cuda_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_CUDA_ALIGN;
}

static const struct gk_backend_buffer_type_i g_cuda_buft_iface = {
    /* .get_name       = */ gk_cuda_buft_name,
    /* .alloc_buffer   = */ gk_cuda_buft_alloc,
    /* .get_alignment  = */ gk_cuda_buft_alignment,
    /* .get_alloc_size = */ NULL, // a tensor costs its own bytes here
    /* .is_host        = */ NULL, // device memory: the default false is right
};

// --------------------------------------------------------------------------
// pinned host memory
//
// Allocated by the device, addressable by the host, and copied from without
// the driver having to stage it - which is what makes it worth a separate
// buffer type rather than plain malloc. The engine puts the KV cache overflow
// and the input tensors here when it wants both sides to reach them.
// --------------------------------------------------------------------------

struct gk_cuda_host_buffer_ctx {
    void * base;
};

static void gk_cuda_host_buffer_free(gk_backend_buffer_t buffer) {
    struct gk_cuda_host_buffer_ctx * ctx = (struct gk_cuda_host_buffer_ctx *) buffer->context;
    if (ctx == NULL) {
        return;
    }
    GK_CUDA_CHECK(gkFreeHost(ctx->base));
    free(ctx);
}

static void * gk_cuda_host_buffer_get_base(gk_backend_buffer_t buffer) {
    return ((struct gk_cuda_host_buffer_ctx *) buffer->context)->base;
}

static void gk_cuda_host_buffer_set(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                    const void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void gk_cuda_host_buffer_get(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                    void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void gk_cuda_host_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    struct gk_cuda_host_buffer_ctx * ctx = (struct gk_cuda_host_buffer_ctx *) buffer->context;
    memset(ctx->base, value, buffer->size);
}

static const struct gk_backend_buffer_i g_cuda_host_buffer_iface = {
    /* .free_buffer   = */ gk_cuda_host_buffer_free,
    /* .get_base      = */ gk_cuda_host_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .set_tensor    = */ gk_cuda_host_buffer_set,
    /* .get_tensor    = */ gk_cuda_host_buffer_get,
    /* .clear         = */ gk_cuda_host_buffer_clear,
    /* .memset_tensor = */ NULL,
    /* .cpy_tensor    = */ NULL,
};

static const char * gk_cuda_host_buft_name(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_CUDA_BACKEND_NAME "_Host";
}

static gk_backend_buffer_t gk_cuda_host_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    struct gk_cuda_device_ctx * dev = (struct gk_cuda_device_ctx *) buft->context;

    struct gk_cuda_host_buffer_ctx * ctx =
        (struct gk_cuda_host_buffer_ctx *) malloc(sizeof(struct gk_cuda_host_buffer_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    GK_CUDA_CHECK(gkSetDevice(dev->index));

    const gkError_t err = gkHostAlloc(&ctx->base, size);
    if (err != gkSuccess) {
        // Pinned memory is a finite resource and running out of it is not
        // fatal - ordinary host memory works, just slower - so this is a
        // warning and a NULL rather than an abort.
        gk_logf("gk %s: could not pin %zu bytes of host memory: %s\n",
                GK_CUDA_BACKEND_NAME, size, gkGetErrorString(err));
        free(ctx);
        return NULL;
    }

    return gk_backend_buffer_init(buft, &g_cuda_host_buffer_iface, ctx, size);
}

static size_t gk_cuda_host_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_MEM_ALIGN;
}

static bool gk_cuda_host_buft_is_host(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return true;
}

static const struct gk_backend_buffer_type_i g_cuda_host_buft_iface = {
    /* .get_name       = */ gk_cuda_host_buft_name,
    /* .alloc_buffer   = */ gk_cuda_host_buft_alloc,
    /* .get_alignment  = */ gk_cuda_host_buft_alignment,
    /* .get_alloc_size = */ NULL,
    /* .is_host        = */ gk_cuda_host_buft_is_host,
};

// --------------------------------------------------------------------------
// the backend
// --------------------------------------------------------------------------

struct gk_cu_graph_cache;

struct gk_cuda_backend_ctx {
    struct gk_cuda_device_ctx * dev;
    gkStream_t                  stream;
    struct gk_cuda_scratch      scratch;
    struct gk_cu_graph_cache *  graphs; // captured-graph cache; NULL until first used
};

// --------------------------------------------------------------------------
// captured graphs
//
// A decode step is a few hundred tiny kernels, and on Windows every launch is
// a WDDM submission; the launches cost more than the math. But the server's
// decode graph is byte-identical from one token to the next - the KV cache is
// written through set_rows with the row indices as a graph *input*, and the
// attention shapes are padded to a 256-token window - so the same launches,
// with the same pointers and the same geometry, repeat for hundreds of calls.
// That is the one situation CUDA graphs are for: record the launches once,
// then replay the whole step as a single submission.
//
// The contract is exact repetition, checked, not assumed. Every node's data
// pointer, geometry, op params and operand pointers are recorded; a graph
// replays only when all of them match what was captured. The first sighting
// of a topology just records it; the second executes through stream capture
// and keeps the instantiated graph; from the third on, one launch. Anything
// that breaks the match - a prompt of a different length, the 256-token
// window rolling over, the scratch buffer moving - falls back to plain
// launches and re-records, which is never wrong, only slower.
//
// GK_CUDA_GRAPHS=0 turns it off. GK_CUDA_GRAPH_LOG=1 says what it is doing
// and, more usefully, *why* a graph did not replay - the first node that
// failed the comparison and which of its facts moved.
// --------------------------------------------------------------------------

#define GK_CU_GRAPH_CACHE     8   // distinct topologies kept per backend
#define GK_CU_GRAPH_MIN_NODES 8   // below this, capture overhead beats the win

// How many operands get their full geometry recorded. Operand *pointers* are
// recorded for all of them; shapes matter for the first few (flash attention
// reads the KV length from src1/src2's extents while its own shape holds
// still, so a dst-only record would replay a stale attention span).
#define GK_CU_GRAPH_SRC_SHAPES 4

struct gk_cu_node_props {
    void *  data;
    int32_t op;
    int32_t type;
    int32_t flags;
    int32_t n_src;
    int64_t ne[GK_MAX_DIMS];
    size_t  nb[GK_MAX_DIMS];
    int32_t op_params[GK_MAX_OP_PARAMS / sizeof(int32_t)];
    void *  src_data[GK_MAX_SRC];
    int32_t src_type[GK_MAX_SRC];
    int64_t src_ne[GK_CU_GRAPH_SRC_SHAPES][GK_MAX_DIMS];
    size_t  src_nb[GK_CU_GRAPH_SRC_SHAPES][GK_MAX_DIMS];
};

static void gk_cu_node_props_fill(struct gk_cu_node_props * p, const struct gk_tensor * t) {
    // memset first so the padding and the unused slots compare equal.
    memset(p, 0, sizeof(*p));

    p->data  = t->data;
    p->op    = (int32_t) t->op;
    p->type  = (int32_t) t->type;
    p->flags = t->flags;

    memcpy(p->ne, t->ne, sizeof(p->ne));
    memcpy(p->nb, t->nb, sizeof(p->nb));
    memcpy(p->op_params, t->op_params, sizeof(p->op_params));

    for (int s = 0; s < GK_MAX_SRC; ++s) {
        const struct gk_tensor * src = t->src[s];
        if (src == NULL) {
            continue;
        }
        p->n_src       = s + 1;
        p->src_data[s] = src->data;
        p->src_type[s] = (int32_t) src->type;
        if (s < GK_CU_GRAPH_SRC_SHAPES) {
            memcpy(p->src_ne[s], src->ne, sizeof(p->src_ne[s]));
            memcpy(p->src_nb[s], src->nb, sizeof(p->src_nb[s]));
        }
    }
}

struct gk_cu_graph_entry {
    std::vector<gk_cu_node_props> props;
    gkGraphExec_t exec   = NULL;
    uint64_t      gen    = 0;     // scratch generation the capture baked in
    uint64_t      tick   = 0;     // last use, for eviction
    bool          broken = false; // capture failed once; do not retry this topology
    bool          used   = false;

    // GK_CUDA_GRAPH_PROF only: event-record nodes captured *into* the graph
    // every bucket_size launches, so a replay times itself. The events are
    // re-recorded by every replay; the harvest on the next call reads the
    // previous replay's timestamps. A few dozen events per graph is cheap
    // enough to leave running; a pair per node (GK_LAUNCH_PROFILE=2) is not.
    std::vector<gkEvent_t>   pev;
    std::vector<double>      pms;    // accumulated ms per bucket
    std::vector<std::string> plabel; // first node of each bucket, for the dump
    int64_t                  pn = 0; // replays accumulated

    void prof_reset() {
        for (size_t i = 0; i < pev.size(); ++i) {
            if (pev[i] != NULL) {
                GK_CUDA_CHECK(gkEventDestroy(pev[i]));
            }
        }
        pev.clear();
        pms.clear();
        plabel.clear();
        pn = 0;
    }
};

struct gk_cu_graph_cache {
    gk_cu_graph_entry entries[GK_CU_GRAPH_CACHE];
    std::vector<gk_cu_node_props> staging; // this call's props, reused across calls
    uint64_t tick = 0;

    // Replay timing, GK_CUDA_GRAPH_LOG=2 only: one event pair around the
    // graph launch, harvested on a later call once the stream has moved past
    // it. Two events per replayed *graph* is measurement; two per node, which
    // is what GK_LAUNCH_PROFILE=2 does, distorts what it measures.
    gkEvent_t tev0    = NULL;
    gkEvent_t tev1    = NULL;
    bool      armed   = false;
    double    dev_ms  = 0.0;
    int64_t   dev_n   = 0;
};

// What the lookup decided for this call.
enum gk_cu_graph_action {
    GK_CU_GRAPH_PLAIN,   // no match (or graphs off): launch normally
    GK_CU_GRAPH_CAPTURE, // seen once before, identical: capture while launching
    GK_CU_GRAPH_REPLAY,  // captured and still valid: launch the graph
};

static int64_t g_graph_replays  = 0;
static int64_t g_graph_captures = 0;
static int64_t g_graph_breaks   = 0;

static void gk_cu_graph_log_dump(void) {
    gk_logf("gk cuda graphs: %lld replays, %lld captures, %lld broken\n",
            (long long) g_graph_replays, (long long) g_graph_captures,
            (long long) g_graph_breaks);
}

static int gk_cu_graph_log_level(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GK_CUDA_GRAPH_LOG");
        on = e == NULL || e[0] == '\0' || e[0] == '0' ? 0 : (e[0] == '2' ? 2 : 1);
        if (on) {
            atexit(gk_cu_graph_log_dump);
        }
    }
    return on;
}

static bool gk_cu_graph_log_on(void) {
    return gk_cu_graph_log_level() != 0;
}

static bool gk_cu_graphs_on(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GK_CUDA_GRAPHS");
        bool v = !(e != NULL && e[0] == '0'); // default on

        // The mat-mul diagnostics synchronize inside an op, which invalidates
        // every capture; a run that asks for them has chosen measurement.
        const char * s1 = getenv("GK_MM_SPLIT");
        const char * s2 = getenv("GK_MM_FP4_STATS");
        if ((s1 != NULL && s1[0] != '\0' && s1[0] != '0') ||
            (s2 != NULL && s2[0] != '\0' && s2[0] != '0')) {
            v = false;
        }
        on = v ? 1 : 0;
    }
    return on != 0;
}

// Why a candidate entry failed to match, for the log: the first differing
// node and which of its recorded facts moved.
static void gk_cu_graph_log_mismatch(const struct gk_cu_graph_entry * e,
                                     struct gk_cgraph * graph,
                                     const std::vector<gk_cu_node_props> & now) {
    static int logged = 0;
    if (logged >= (gk_cu_graph_log_level() >= 2 ? 4096 : 16)) {
        return; // enough to see the pattern; a rolling mismatch would flood
    }

    for (size_t i = 0; i < now.size(); ++i) {
        const struct gk_cu_node_props * a = &e->props[i];
        const struct gk_cu_node_props * b = &now[i];
        if (memcmp(a, b, sizeof(*a)) == 0) {
            continue;
        }

        const char * what =
            a->op != b->op || a->type != b->type          ? "op/type"    :
            a->data != b->data                            ? "data ptr"   :
            memcmp(a->ne, b->ne, sizeof(a->ne)) != 0      ? "shape"      :
            memcmp(a->nb, b->nb, sizeof(a->nb)) != 0      ? "strides"    :
            memcmp(a->op_params, b->op_params,
                   sizeof(a->op_params)) != 0             ? "op params"  :
            memcmp(a->src_data, b->src_data,
                   sizeof(a->src_data)) != 0              ? "src ptr"    :
            memcmp(a->src_ne, b->src_ne,
                   sizeof(a->src_ne)) != 0                ? "src shape"  : "src strides";

        gk_logf("gk cuda graphs: no replay, node %zu (%s, op %s) changed: %s (%p -> %p)\n",
                i, gk_graph_node(graph, (int) i)->name,
                gk_op_name((enum gk_op) b->op), what, a->data, b->data);
        logged++;
        return;
    }
}

// Records or matches this call's graph. On a full match the entry comes back
// with the action that fits its state; on a miss the topology is recorded
// into the least-recently-used slot and the call runs plain.
static enum gk_cu_graph_action gk_cu_graph_lookup(struct gk_cuda_backend_ctx * ctx,
                                                  struct gk_cgraph * graph, int n,
                                                  struct gk_cu_graph_entry ** out) {
    *out = NULL;

    if (ctx->graphs == NULL) {
        ctx->graphs = new gk_cu_graph_cache();
    }
    struct gk_cu_graph_cache * gc = ctx->graphs;
    gc->tick++;

    // The health of the whole scheme in one recurring line: if replays are
    // not the overwhelming share, the cache is thrashing and the log above
    // says which node keeps moving.
    if (gk_cu_graph_log_on() && gc->tick % 1024 == 0) {
        gk_logf("gk cuda graphs: %s: %llu lookups, %lld replays, %lld captures, %lld broken\n",
                ctx->dev->name, (unsigned long long) gc->tick,
                (long long) g_graph_replays, (long long) g_graph_captures,
                (long long) g_graph_breaks);
    }

    gc->staging.resize((size_t) n);
    for (int i = 0; i < n; ++i) {
        gk_cu_node_props_fill(&gc->staging[(size_t) i], gk_graph_node(graph, i));
    }

    for (int k = 0; k < GK_CU_GRAPH_CACHE; ++k) {
        struct gk_cu_graph_entry * e = &gc->entries[k];
        if (!e->used || e->props.size() != (size_t) n) {
            continue;
        }
        if (memcmp(e->props.data(), gc->staging.data(),
                   (size_t) n * sizeof(gk_cu_node_props)) != 0) {
            continue; // a stale sibling; only a full miss below is worth a log line
        }

        e->tick = gc->tick;
        *out    = e;

        if (e->broken) {
            return GK_CU_GRAPH_PLAIN;
        }
        if (e->exec != NULL && e->gen == ctx->scratch.gen) {
            return GK_CU_GRAPH_REPLAY;
        }
        if (e->exec != NULL) {
            // The scratch moved since the capture, so the recorded launches
            // point into freed memory. The grow already drained the stream;
            // the sync here is against a launch of this very graph still in
            // flight, and costs nothing when there is none.
            GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
            GK_CUDA_CHECK(gkGraphExecDestroy(e->exec));
            e->exec = NULL;
        }
        return GK_CU_GRAPH_CAPTURE;
    }

    // Not seen before: remember it in the stalest slot. The log names the
    // nearest miss - the first same-length entry's first differing node - so
    // a graph that never replays says why.
    if (gk_cu_graph_log_on()) {
        for (int k = 0; k < GK_CU_GRAPH_CACHE; ++k) {
            struct gk_cu_graph_entry * e = &gc->entries[k];
            if (e->used && e->props.size() == (size_t) n) {
                gk_cu_graph_log_mismatch(e, graph, gc->staging);
                break;
            }
        }
    }

    struct gk_cu_graph_entry * victim = &gc->entries[0];
    for (int k = 1; k < GK_CU_GRAPH_CACHE; ++k) {
        struct gk_cu_graph_entry * e = &gc->entries[k];
        if (!e->used) {
            victim = e;
            break;
        }
        if (e->tick < victim->tick) {
            victim = e;
        }
    }
    if (victim->exec != NULL) {
        GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
        GK_CUDA_CHECK(gkGraphExecDestroy(victim->exec));
        victim->exec = NULL;
    }
    victim->prof_reset();
    victim->props  = gc->staging;
    victim->gen    = 0;
    victim->tick   = gc->tick;
    victim->broken = false;
    victim->used   = true;

    return GK_CU_GRAPH_PLAIN;
}

static void gk_cu_graph_cache_free(struct gk_cuda_backend_ctx * ctx) {
    if (ctx->graphs == NULL) {
        return;
    }
    // The caller has already drained the stream.
    for (int k = 0; k < GK_CU_GRAPH_CACHE; ++k) {
        if (ctx->graphs->entries[k].exec != NULL) {
            GK_CUDA_CHECK(gkGraphExecDestroy(ctx->graphs->entries[k].exec));
        }
        ctx->graphs->entries[k].prof_reset();
    }
    if (ctx->graphs->tev0 != NULL) { GK_CUDA_CHECK(gkEventDestroy(ctx->graphs->tev0)); }
    if (ctx->graphs->tev1 != NULL) { GK_CUDA_CHECK(gkEventDestroy(ctx->graphs->tev1)); }
    delete ctx->graphs;
    ctx->graphs = NULL;
}

// One node: launch, and name a geometry rejection as it happens. The error
// check is a host-side flag read, not a synchronization.
static inline enum gk_status gk_cuda_launch_one(struct gk_cuda_backend_ctx * ctx,
                                                struct gk_tensor * node) {
    if (!gk_cuda_compute_op(ctx->stream, &ctx->scratch, node)) {
        gk_logf("gk %s: no kernel for op %s (node %s)\n",
                GK_CUDA_BACKEND_NAME, gk_op_name(node->op), node->name);
        return GK_STATUS_NO_STORAGE;
    }

    const gkError_t err = gkGetLastError();
    if (err != gkSuccess) {
        gk_logf("gk %s: %s (node %s, op %s, ne = [%lld %lld %lld %lld])\n",
                GK_CUDA_BACKEND_NAME, gkGetErrorString(err),
                node->name, gk_op_name(node->op),
                (long long) node->ne[0], (long long) node->ne[1],
                (long long) node->ne[2], (long long) node->ne[3]);
        return GK_STATUS_NO_STORAGE;
    }
    return GK_STATUS_SUCCESS;
}

// --------------------------------------------------------------------------
// fusion plan
//
// On this driver a kernel costs ~5-8 us of fixed replay latency whatever it
// computes, so at decode shapes a transformer's (rms_norm, mul-by-weight)
// pair - four to six of them per layer - is two overheads for one kernel's
// worth of math. The plan marks each such pair; the launch loops below run
// the fused kernel at the head and skip the tail.
//
// The conditions are checked here, once per launching pass, so the launcher
// itself has nothing to decide: the mul must directly follow its norm,
// consume it as one operand, be the norm's *only* consumer in this graph,
// match its shape, and the norm must not be something the outside reads.
// GK_CUDA_FUSE=0 turns the whole thing off.
// --------------------------------------------------------------------------

#define GK_CU_FUSE_NONE  0
#define GK_CU_FUSE_HEAD  1 // fused (rms_norm, mul): covers this node and the next
#define GK_CU_FUSE_SKIP  2 // covered by an earlier node's fused launch
#define GK_CU_FUSE_HEAD3 3 // fused (add, rms_norm, mul): covers three nodes

// The tail fusions: the head is the *last* node of its chain and the parts
// sit anywhere behind it, marked SKIP where they stand. This is the shape a
// DiT block takes - its norms are separated from their muls by the nodes
// that build the weight, its gates and rope are spelled out in elementwise
// steps - so the adjacent pairs above never fire there. `aux` carries the
// part indices the head's launcher needs.
#define GK_CU_FUSE_RMSMUL_T    4 // at the mul: (rms_norm ... mul)
#define GK_CU_FUSE_ADDRMSMUL_T 5 // at the mul: (add, rms_norm ... mul)
#define GK_CU_FUSE_MADD        6 // at the add: c + y*g, g one row
#define GK_CU_FUSE_MADDS       7 // at the second add: c + y*g + t
#define GK_CU_FUSE_UNARYMUL    8 // at the mul: unary(x) * y
#define GK_CU_FUSE_ROPE2       9 // at the add: x1*f1 + x2*f2, repeats elided
#define GK_CU_FUSE_MMACT      10 // at the unary: applied in the matmul epilogue

static bool gk_cu_fuse_on(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GK_CUDA_FUSE");
        on = !(e != NULL && e[0] == '0');
    }
    return on != 0;
}

// Contiguous f32 with a vectorizable row: the only layout the fused kernels
// speak. Anything else is launched unfused rather than fused slowly - the
// first cut fused everything through the generic accessor and lost ~5% to
// the very pairs it replaced.
static bool gk_cu_fuse_flat(const struct gk_tensor * t) {
    return t != NULL && (int) t->type == GK_TYPE_F32 && gk_is_contiguous(t) &&
           t->ne[0] % 4 == 0;
}

// The norm weight: one contiguous f32 row matching the normed extent.
static bool gk_cu_fuse_weight(const struct gk_tensor * w, int64_t ne0) {
    return gk_cu_fuse_flat(w) && w->ne[0] == ne0 &&
           w->ne[1] == 1 && w->ne[2] == 1 && w->ne[3] == 1;
}

// Whether the tail fusions run; `GK_CUDA_FUSE_TAIL=0` takes only them out
// of a bisect while the adjacent pairs stay.
static bool gk_cu_fuse_tail_on(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GK_CUDA_FUSE_TAIL");
        on = !(e != NULL && e[0] == '0');
    }
    return on != 0;
}

// GK_CUDA_FUSE_MASK: bisect aid - a bitmask of patterns allowed to fire,
// bit k = pattern k (e.g. 0x200 = rope2 only, 0x5ff = all but rope2).
// Unset allows everything.
static bool gk_cu_fuse_pat_on(int pat) {
    static long mask = -1;
    if (mask < 0) {
        const char * e = getenv("GK_CUDA_FUSE_MASK");
        mask = e != NULL ? strtol(e, NULL, 0) : 0x7ff;
    }
    return (mask & (1l << pat)) != 0;
}

// Whether this node's launch writes its output at all: the view family does
// not, and treating a view as a writer would veto every window it sits in.
static bool gk_cu_fuse_writes(const struct gk_tensor * t) {
    switch (t->op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return false;
        default:
            return true;
    }
}

static bool gk_cu_fuse_overlap(const struct gk_tensor * a, const struct gk_tensor * b) {
    const uint8_t * al = (const uint8_t *) a->data;
    const uint8_t * ah = al + gk_nbytes(a);
    const uint8_t * bl = (const uint8_t *) b->data;
    const uint8_t * bh = bl + gk_nbytes(b);
    return al < bh && bl < ah;
}

static bool gk_cu_fuse_same_ne(const struct gk_tensor * a, const struct gk_tensor * b) {
    return a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] &&
           a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

// A fused head reads its parts' sources at its own position while writing
// its output. The allocator considers a part's source dead at the part, so
// the output may have been given (part of) that very storage: unfused that
// was fine - the part's read ran before the head's write - fused it is a
// cross-thread race inside one kernel, and the corruption varies with warp
// timing. Writing straight onto a read is safe only when every thread
// writes exactly the element it read: same address, same extents, same
// type. A broadcast row never satisfies that against a full-size output,
// and any partial overlap is a race.
static bool gk_cu_fuse_out_ok(const struct gk_tensor * out,
                              std::initializer_list<const struct gk_tensor *> reads) {
    for (const struct gk_tensor * r : reads) {
        if (r == NULL || !gk_cu_fuse_overlap(out, r)) {
            continue;
        }
        if (r->data == out->data && r->type == out->type &&
            gk_cu_fuse_same_ne(r, out)) {
            continue; // exact elementwise aliasing: in-thread read-before-write
        }
        return false;
    }
    return true;
}

// The strict flavor, for heads whose read pattern is not elementwise (a
// matmul's tiles, rope's pair layout): any overlap at all is a race.
static bool gk_cu_fuse_out_disjoint(const struct gk_tensor * out,
                                    std::initializer_list<const struct gk_tensor *> reads) {
    for (const struct gk_tensor * r : reads) {
        if (r != NULL && gk_cu_fuse_overlap(out, r)) {
            return false;
        }
    }
    return true;
}

static void gk_cu_fuse_plan(struct gk_cuda_scratch * scratch,
                            struct gk_cgraph * graph, int n, std::vector<uint8_t> & tag,
                            std::vector<int32_t> & aux) {
    tag.assign((size_t) n, GK_CU_FUSE_NONE);
    aux.assign((size_t) n * 2, -1);

    if (!gk_cu_fuse_on()) {
        return;
    }

    std::unordered_map<const struct gk_tensor *, int> uses;
    uses.reserve((size_t) n * 2);
    for (int i = 0; i < n; ++i) {
        const struct gk_tensor * node = gk_graph_node(graph, i);
        for (int s = 0; s < GK_MAX_SRC && node->src[s] != NULL; ++s) {
            uses[node->src[s]]++;
        }
    }

    for (int i = 0; i + 1 < n; ++i) {
        struct gk_tensor * a = gk_graph_node(graph, i);
        struct gk_tensor * b = gk_graph_node(graph, i + 1);

        // The residual chain: add, then the norm that reads the sum, then
        // the norm's weight. The add's output is still written - it is the
        // residual stream and has other readers - so only the *norm's*
        // consumers constrain anything.
        if (i + 2 < n && a->op == GK_OP_ADD && b->op == GK_OP_RMS_NORM &&
            b->src[0] == a) {
            struct gk_tensor * c = gk_graph_node(graph, i + 2);

            const bool same_abc =
                b->ne[0] == a->ne[0] && b->ne[1] == a->ne[1] &&
                b->ne[2] == a->ne[2] && b->ne[3] == a->ne[3] &&
                c->ne[0] == a->ne[0] && c->ne[1] == a->ne[1] &&
                c->ne[2] == a->ne[2] && c->ne[3] == a->ne[3];
            const bool add_elementwise = // no broadcast: a residual never has one
                a->src[0] != NULL && a->src[1] != NULL &&
                a->src[0]->ne[0] == a->ne[0] && a->src[0]->ne[1] == a->ne[1] &&
                a->src[0]->ne[2] == a->ne[2] && a->src[0]->ne[3] == a->ne[3] &&
                a->src[1]->ne[0] == a->ne[0] && a->src[1]->ne[1] == a->ne[1] &&
                a->src[1]->ne[2] == a->ne[2] && a->src[1]->ne[3] == a->ne[3];

            const struct gk_tensor * w3 = c->src[0] == b ? c->src[1] : c->src[0];
            if (c->op == GK_OP_MUL && (c->src[0] == b || c->src[1] == b) &&
                uses[b] == 1 &&
                (b->flags & (GK_TENSOR_FLAG_OUTPUT | GK_TENSOR_FLAG_INPUT)) == 0 &&
                same_abc && add_elementwise &&
                gk_cu_fuse_flat(a) && gk_cu_fuse_flat(c) &&
                gk_cu_fuse_flat(a->src[0]) && gk_cu_fuse_flat(a->src[1]) &&
                gk_cu_fuse_weight(w3, c->ne[0]) &&
                (int) b->type == GK_TYPE_F32 && gk_cu_fuse_pat_on(GK_CU_FUSE_HEAD3) &&
                gk_cu_fuse_out_ok(c, { a->src[0], a->src[1], w3, a }) &&
                gk_cu_fuse_out_ok(a, { a->src[0], a->src[1], w3 })) {
                tag[(size_t) i]     = GK_CU_FUSE_HEAD3;
                tag[(size_t) i + 1] = GK_CU_FUSE_SKIP;
                tag[(size_t) i + 2] = GK_CU_FUSE_SKIP;
                i += 2;
                continue;
            }
        }

        if (a->op != GK_OP_RMS_NORM || b->op != GK_OP_MUL) {
            continue;
        }
        if (b->src[0] != a && b->src[1] != a) {
            continue;
        }
        if (uses[a] != 1 || (a->flags & (GK_TENSOR_FLAG_OUTPUT | GK_TENSOR_FLAG_INPUT)) != 0) {
            continue;
        }
        // The fused kernel writes b with a's row statistics, so the two must
        // agree on every extent; a mul that broadcasts the *norm* is not
        // this pattern.
        if (b->ne[0] != a->ne[0] || b->ne[1] != a->ne[1] ||
            b->ne[2] != a->ne[2] || b->ne[3] != a->ne[3]) {
            continue;
        }
        if ((int) a->type != GK_TYPE_F32 ||
            !gk_cu_fuse_flat(b) || !gk_cu_fuse_flat(a->src[0]) ||
            !gk_cu_fuse_weight(b->src[0] == a ? b->src[1] : b->src[0], b->ne[0])) {
            continue;
        }
        if (!gk_cu_fuse_pat_on(GK_CU_FUSE_HEAD) ||
            !gk_cu_fuse_out_ok(b, { a->src[0],
                                    b->src[0] == a ? b->src[1] : b->src[0] })) {
            continue;
        }

        tag[(size_t) i]     = GK_CU_FUSE_HEAD;
        tag[(size_t) i + 1] = GK_CU_FUSE_SKIP;
        ++i; // the pair is settled; the tail cannot head another pair
    }

    if (!gk_cu_fuse_tail_on()) {
        return;
    }

    // ----- the tail fusions -----

    std::unordered_map<const struct gk_tensor *, int> pos;
    pos.reserve((size_t) n * 2);
    for (int i = 0; i < n; ++i) {
        pos[gk_graph_node(graph, i)] = i;
    }

    // The fusion plan runs after allocation, and that is what makes a
    // non-adjacent chain checkable at all: a part's input dies - to the
    // allocator - at the part, so a node between the part and the head may
    // have been given that very storage for its own output. The check is by
    // actual pointer ranges: no writer in the window may overlap a tensor
    // the fused launch still reads (its own producer excepted - that write
    // is the dependency), and no node in the window may read an output the
    // fused launch defers to the head.
    auto window_ok = [&](int from, int to,
                         std::initializer_list<const struct gk_tensor *> reads,
                         std::initializer_list<const struct gk_tensor *> late,
                         std::initializer_list<int> parts) -> bool {
        if (to - from > 24) {
            return false;
        }
        for (int j = from + 1; j < to; ++j) {
            bool is_part = false;
            for (int p : parts) {
                if (p == j) { is_part = true; break; }
            }
            if (is_part) {
                continue;
            }

            const struct gk_tensor * nj = gk_graph_node(graph, j);
            if (gk_cu_fuse_writes(nj)) {
                for (const struct gk_tensor * r : reads) {
                    // a read that is the writer itself, or a view into it,
                    // is the dependency this window exists to preserve -
                    // rope's frequency views live inside the cont that
                    // produces them
                    if (r == NULL || nj == r ||
                        (r->view_src != NULL && r->view_src == nj)) {
                        continue;
                    }
                    if (gk_cu_fuse_overlap(nj, r)) {
                        return false;
                    }
                }
            }
            for (int s = 0; s < GK_MAX_SRC && nj->src[s] != NULL; ++s) {
                for (const struct gk_tensor * l : late) {
                    if (nj->src[s] == l) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    auto clean = [&](const struct gk_tensor * t, int * idx) -> bool {
        auto it = pos.find(t);
        if (it == pos.end() || tag[(size_t) it->second] != GK_CU_FUSE_NONE) {
            return false;
        }
        *idx = it->second;
        return (t->flags & (GK_TENSOR_FLAG_OUTPUT | GK_TENSOR_FLAG_INPUT)) == 0;
    };

    for (int i = 0; i < n; ++i) {
        if (tag[(size_t) i] != GK_CU_FUSE_NONE) {
            continue;
        }
        struct gk_tensor * h = gk_graph_node(graph, i);

        if (h->op == GK_OP_MUL && h->src[0] != NULL && h->src[1] != NULL) {
            // (rms_norm [after an add] ... mul-by-weight), the parts behind:
            // a DiT builds the weight *between* the norm and the mul.
            struct gk_tensor * nm =
                h->src[0]->op == GK_OP_RMS_NORM ? h->src[0] :
                h->src[1]->op == GK_OP_RMS_NORM ? h->src[1] : NULL;

            int ni = -1;
            if (nm != NULL && clean(nm, &ni) && uses[nm] == 1 &&
                (int) nm->type == GK_TYPE_F32 && gk_cu_fuse_same_ne(h, nm) &&
                gk_cu_fuse_flat(h) && gk_cu_fuse_flat(nm->src[0]) &&
                gk_cu_fuse_weight(h->src[0] == nm ? h->src[1] : h->src[0], h->ne[0])) {
                const struct gk_tensor * w = h->src[0] == nm ? h->src[1] : h->src[0];

                // the residual add feeding the norm comes along when it is
                // untouched in between - its output is still written
                struct gk_tensor * ad = nm->src[0];
                int ai = -1;
                bool done = false;
                if (ad->op == GK_OP_ADD && ad->src[0] != NULL && ad->src[1] != NULL) {
                    auto ait = pos.find(ad);
                    if (ait != pos.end() && tag[(size_t) ait->second] == GK_CU_FUSE_NONE &&
                        (ad->flags & GK_TENSOR_FLAG_INPUT) == 0) {
                        ai = ait->second;
                        const bool ew =
                            gk_cu_fuse_same_ne(ad, ad->src[0]) &&
                            gk_cu_fuse_same_ne(ad, ad->src[1]) &&
                            gk_cu_fuse_same_ne(ad, h);
                        if (ew && gk_cu_fuse_flat(ad) &&
                            gk_cu_fuse_pat_on(GK_CU_FUSE_ADDRMSMUL_T) &&
                            gk_cu_fuse_flat(ad->src[0]) && gk_cu_fuse_flat(ad->src[1]) &&
                            gk_cu_fuse_out_ok(h, { ad->src[0], ad->src[1], w, ad }) &&
                            gk_cu_fuse_out_ok(ad, { ad->src[0], ad->src[1], w }) &&
                            window_ok(ai, i, { ad->src[0], ad->src[1], w },
                                      { ad, nm }, { ni })) {
                            tag[(size_t) ai] = GK_CU_FUSE_SKIP;
                            tag[(size_t) ni] = GK_CU_FUSE_SKIP;
                            tag[(size_t) i]  = GK_CU_FUSE_ADDRMSMUL_T;
                            aux[(size_t) i * 2]     = ni;
                            aux[(size_t) i * 2 + 1] = ai;
                            done = true;
                        }
                    }
                }
                if (!done && gk_cu_fuse_pat_on(GK_CU_FUSE_RMSMUL_T) &&
                    gk_cu_fuse_out_ok(h, { nm->src[0], w }) &&
                    window_ok(ni, i, { nm->src[0], w }, { nm }, {})) {
                    tag[(size_t) ni] = GK_CU_FUSE_SKIP;
                    tag[(size_t) i]  = GK_CU_FUSE_RMSMUL_T;
                    aux[(size_t) i * 2] = ni;
                }
                continue;
            }

            // unary(x) * y - the attention gate, and the MLP's geglu when
            // the allocator left the gate projection readable
            struct gk_tensor * u =
                h->src[0]->op == GK_OP_UNARY ? h->src[0] :
                h->src[1]->op == GK_OP_UNARY ? h->src[1] : NULL;

            int ui = -1;
            if (u != NULL && clean(u, &ui) && uses[u] == 1 &&
                (int) u->type == GK_TYPE_F32 && gk_cu_fuse_same_ne(h, u) &&
                gk_cu_fuse_same_ne(h, h->src[0] == u ? h->src[1] : h->src[0]) &&
                gk_cu_fuse_flat(h) && gk_cu_fuse_flat(u->src[0]) &&
                gk_cu_fuse_flat(h->src[0] == u ? h->src[1] : h->src[0])) {
                const struct gk_tensor * y = h->src[0] == u ? h->src[1] : h->src[0];
                if (gk_cu_fuse_pat_on(GK_CU_FUSE_UNARYMUL) &&
                    gk_cu_fuse_out_ok(h, { u->src[0], y }) &&
                    window_ok(ui, i, { u->src[0], y }, { u }, {})) {
                    tag[(size_t) ui] = GK_CU_FUSE_SKIP;
                    tag[(size_t) i]  = GK_CU_FUSE_UNARYMUL;
                    aux[(size_t) i * 2] = ui;
                }
                continue;
            }
        }

        // unary directly on a gate projection: applied in the matmul's own
        // epilogue, so the 68 MB activation pass never runs. This is the
        // geglu pair the unary*mul fusion cannot reach - the allocator
        // recycles the gate projection under it - approached from the other
        // side: nothing here outlives the matmul launch.
        if (h->op == GK_OP_UNARY && h->src[0] != NULL &&
            h->src[0]->op == GK_OP_MUL_MAT) {
            struct gk_tensor * m = h->src[0];

            int mi3 = -1;
            if (clean(m, &mi3) && uses[m] == 1 &&
                gk_cu_fuse_same_ne(h, m) &&
                (int) h->type == GK_TYPE_F32 && gk_is_contiguous(h) &&
                (int) m->type == GK_TYPE_F32 && gk_is_contiguous(m) &&
                gk_cuda_mm_act_fusable(scratch, m) &&
                gk_cu_fuse_pat_on(GK_CU_FUSE_MMACT) &&
                gk_cu_fuse_out_disjoint(h, { m->src[0], m->src[1] }) &&
                window_ok(mi3, i, { m->src[1] }, { m }, {})) {
                tag[(size_t) mi3] = GK_CU_FUSE_SKIP;
                tag[(size_t) i]   = GK_CU_FUSE_MMACT;
                aux[(size_t) i * 2] = mi3;
            }
            continue;
        }

        if (h->op == GK_OP_ADD && h->src[0] != NULL && h->src[1] != NULL) {
            // The hand-built rope pair: add(mul(repeat(x1), f1),
            // mul(repeat(x2), f2)), where each x is [1,K,T,H], each f
            // [2,K,T,1], the output [2,K,T,H]. The repeats materialize x at
            // full width twice; collapsed, x is read once.
            struct gk_tensor * m1 = h->src[0];
            struct gk_tensor * m2 = h->src[1];
            if (m1 != m2 && m1->op == GK_OP_MUL && m2->op == GK_OP_MUL &&
                h->ne[0] == 2 && (int) h->type == GK_TYPE_F32 && gk_is_contiguous(h) &&
                ((uintptr_t) h->data & 7u) == 0) {
                int mi[2] = { -1, -1 };
                int ri[2] = { -1, -1 };
                bool ok = true;
                for (int k = 0; k < 2 && ok; ++k) {
                    struct gk_tensor * m = k == 0 ? m1 : m2;
                    ok = clean(m, &mi[k]) && uses[m] == 1 && gk_cu_fuse_same_ne(m, h);
                    if (!ok) { break; }

                    struct gk_tensor * r =
                        m->src[0] != NULL && m->src[0]->op == GK_OP_REPEAT ? m->src[0] :
                        m->src[1] != NULL && m->src[1]->op == GK_OP_REPEAT ? m->src[1] : NULL;
                    const struct gk_tensor * f = r == m->src[0] ? m->src[1] : m->src[0];

                    ok = r != NULL && f != NULL && clean(r, &ri[k]) && uses[r] == 1 &&
                         gk_cu_fuse_same_ne(r, h) &&
                         (int) r->src[0]->type == GK_TYPE_F32 &&
                         gk_is_contiguous(r->src[0]) &&
                         r->src[0]->ne[0] == 1 && r->src[0]->ne[1] == h->ne[1] &&
                         r->src[0]->ne[2] == h->ne[2] && r->src[0]->ne[3] == h->ne[3] &&
                         (int) f->type == GK_TYPE_F32 && gk_is_contiguous(f) &&
                         f->ne[0] == 2 && f->ne[1] == h->ne[1] &&
                         f->ne[2] == h->ne[2] && f->ne[3] == 1 &&
                         ((uintptr_t) f->data & 7u) == 0;
                }
                if (ok) {
                    const struct gk_tensor * f1 = m1->src[0]->op == GK_OP_REPEAT
                                                ? m1->src[1] : m1->src[0];
                    const struct gk_tensor * f2 = m2->src[0]->op == GK_OP_REPEAT
                                                ? m2->src[1] : m2->src[0];
                    const struct gk_tensor * r1 = m1->src[0]->op == GK_OP_REPEAT
                                                ? m1->src[0] : m1->src[1];
                    const struct gk_tensor * r2 = m2->src[0]->op == GK_OP_REPEAT
                                                ? m2->src[0] : m2->src[1];
                    const int from = ri[0] < ri[1] ? ri[0] : ri[1];
                    if (gk_cu_fuse_pat_on(GK_CU_FUSE_ROPE2) &&
                        gk_cu_fuse_out_disjoint(h, { r1->src[0], r2->src[0], f1, f2 }) &&
                        window_ok(from, i, { r1->src[0], r2->src[0], f1, f2 }, {},
                                  { ri[0], ri[1], mi[0], mi[1] })) {
                        tag[(size_t) ri[0]] = GK_CU_FUSE_SKIP;
                        tag[(size_t) ri[1]] = GK_CU_FUSE_SKIP;
                        tag[(size_t) mi[0]] = GK_CU_FUSE_SKIP;
                        tag[(size_t) mi[1]] = GK_CU_FUSE_SKIP;
                        tag[(size_t) i]     = GK_CU_FUSE_ROPE2;
                        continue;
                    }
                }
            }

            // c + y*g (+ t): the gate and modulate cluster, g and t one
            // broadcast row each
            for (int side = 0; side < 2; ++side) {
                struct gk_tensor * m = h->src[side];
                struct gk_tensor * c = h->src[side ^ 1];
                if (m->op != GK_OP_MUL || m->src[0] == NULL || m->src[1] == NULL) {
                    continue;
                }

                int mi2 = -1;
                if (!(clean(m, &mi2) && uses[m] == 1 && gk_cu_fuse_same_ne(m, h))) {
                    continue;
                }

                const bool g0 = gk_cu_fuse_weight(m->src[0], h->ne[0]) &&
                                gk_cu_fuse_same_ne(m->src[1], h);
                const bool g1 = gk_cu_fuse_weight(m->src[1], h->ne[0]) &&
                                gk_cu_fuse_same_ne(m->src[0], h);
                if (g0 == g1) {   // none, or ambiguous
                    continue;
                }
                const struct gk_tensor * g = g0 ? m->src[0] : m->src[1];
                const struct gk_tensor * y = g0 ? m->src[1] : m->src[0];

                if (!(gk_cu_fuse_flat(h) && gk_cu_fuse_flat(y) &&
                      gk_cu_fuse_flat(c) && gk_cu_fuse_same_ne(c, h))) {
                    continue;
                }

                // a trailing broadcast shift extends the pair to
                // x*(1+s)+t in one pass - the modulate spelling
                if (uses[h] == 1 &&
                    (h->flags & (GK_TENSOR_FLAG_OUTPUT | GK_TENSOR_FLAG_INPUT)) == 0) {
                    for (int j = i + 1; j < n && j <= i + 8; ++j) {
                        struct gk_tensor * h2 = gk_graph_node(graph, j);
                        if (h2->src[0] != h && h2->src[1] != h) {
                            continue;
                        }
                        if (h2->op != GK_OP_ADD ||
                            tag[(size_t) j] != GK_CU_FUSE_NONE ||
                            !gk_cu_fuse_same_ne(h2, h) || !gk_cu_fuse_flat(h2)) {
                            break;
                        }
                        const struct gk_tensor * t = h2->src[0] == h ? h2->src[1]
                                                                     : h2->src[0];
                        if (!gk_cu_fuse_weight(t, h->ne[0])) {
                            break;
                        }
                        if (gk_cu_fuse_pat_on(GK_CU_FUSE_MADDS) &&
                            gk_cu_fuse_out_ok(h2, { c, y, g, t }) &&
                            window_ok(mi2, j, { c, y, g, t }, { h }, { i })) {
                            tag[(size_t) mi2] = GK_CU_FUSE_SKIP;
                            tag[(size_t) i]   = GK_CU_FUSE_SKIP;
                            tag[(size_t) j]   = GK_CU_FUSE_MADDS;
                            aux[(size_t) j * 2]     = mi2;
                            aux[(size_t) j * 2 + 1] = i;
                        }
                        break;
                    }
                    if (tag[(size_t) i] != GK_CU_FUSE_NONE) {
                        break;
                    }
                }

                if (gk_cu_fuse_pat_on(GK_CU_FUSE_MADD) &&
                    gk_cu_fuse_out_ok(h, { c, y, g }) &&
                    window_ok(mi2, i, { c, y, g }, {}, {})) {
                    tag[(size_t) mi2] = GK_CU_FUSE_SKIP;
                    tag[(size_t) i]   = GK_CU_FUSE_MADD;
                    aux[(size_t) i * 2] = mi2;
                }
                break;
            }
        }
    }

    // GK_CUDA_FUSE_DUMP=1: what the plan decided, once per distinct node
    // count - the fast way to see a pattern quietly not firing.
    static int dump = -1;
    if (dump < 0) {
        const char * e = getenv("GK_CUDA_FUSE_DUMP");
        dump = e != NULL && e[0] == '1' ? 1 : 0;
    }
    if (dump == 1) {
        static std::unordered_map<int, bool> seen;
        if (!seen[n]) {
            seen[n] = true;
            int cnt[11] = { 0 };
            for (int i = 0; i < n; ++i) {
                cnt[tag[(size_t) i]]++;
            }
            gk_logf("gk cuda fuse plan (%d nodes): head %d head3 %d rmsmul_t %d "
                    "addrmsmul_t %d madd %d madds %d unarymul %d rope2 %d mmact %d skip %d\n",
                    n, cnt[GK_CU_FUSE_HEAD], cnt[GK_CU_FUSE_HEAD3],
                    cnt[GK_CU_FUSE_RMSMUL_T], cnt[GK_CU_FUSE_ADDRMSMUL_T],
                    cnt[GK_CU_FUSE_MADD], cnt[GK_CU_FUSE_MADDS],
                    cnt[GK_CU_FUSE_UNARYMUL], cnt[GK_CU_FUSE_ROPE2],
                    cnt[GK_CU_FUSE_MMACT], cnt[GK_CU_FUSE_SKIP]);
        }
    }
}

// The shared "launch node i under the plan" step: 0, 1 or 2 following nodes
// are folded into this launch, or the node is a tail head whose parts were
// skipped where they stood, or the node itself is such a skipped part.
// Returns the number of nodes consumed, or 0 on failure with the status in
// *st.
static inline int gk_cu_launch_planned(struct gk_cuda_backend_ctx * ctx,
                                       struct gk_cgraph * graph, int i,
                                       uint8_t t, const std::vector<int32_t> & aux,
                                       enum gk_status * st) {
    *st = GK_STATUS_SUCCESS;

    if (t == GK_CU_FUSE_SKIP) {
        // a part of a tail chain, launched at its head
        return 1;
    }

    if (t != GK_CU_FUSE_NONE) {
        struct gk_tensor * h  = gk_graph_node(graph, i);
        const int32_t     a0 = aux[(size_t) i * 2];
        const int32_t     a1 = aux[(size_t) i * 2 + 1];

        switch (t) {
            case GK_CU_FUSE_HEAD:
                gk_cuda_fused_rms_mul(ctx->stream, h, gk_graph_node(graph, i + 1));
                break;
            case GK_CU_FUSE_HEAD3:
                gk_cuda_fused_add_rms_mul(ctx->stream, h,
                                          gk_graph_node(graph, i + 1),
                                          gk_graph_node(graph, i + 2));
                break;
            case GK_CU_FUSE_RMSMUL_T:
                gk_cuda_fused_rms_mul_x(ctx->stream, gk_graph_node(graph, a0), h);
                break;
            case GK_CU_FUSE_ADDRMSMUL_T:
                gk_cuda_fused_add_rms_mul_x(ctx->stream, gk_graph_node(graph, a1),
                                            gk_graph_node(graph, a0), h);
                break;
            case GK_CU_FUSE_MADD:
                gk_cuda_fused_madd(ctx->stream, gk_graph_node(graph, a0), h, NULL);
                break;
            case GK_CU_FUSE_MADDS:
                gk_cuda_fused_madd(ctx->stream, gk_graph_node(graph, a0),
                                   gk_graph_node(graph, a1), h);
                break;
            case GK_CU_FUSE_UNARYMUL:
                gk_cuda_fused_unary_mul(ctx->stream, gk_graph_node(graph, a0), h);
                break;
            case GK_CU_FUSE_ROPE2:
                gk_cuda_fused_rope_pair(ctx->stream, h->src[0], h->src[1], h);
                break;
            case GK_CU_FUSE_MMACT:
                gk_cuda_mul_mat_act(ctx->stream, &ctx->scratch,
                                    gk_graph_node(graph, a0), h);
                break;
            default:
                break;
        }

        const gkError_t err = gkGetLastError();
        if (err != gkSuccess) {
            gk_logf("gk %s: %s (fused chain at node %d)\n",
                    GK_CUDA_BACKEND_NAME, gkGetErrorString(err), i);
            *st = GK_STATUS_NO_STORAGE;
            return 0;
        }
        return t == GK_CU_FUSE_HEAD ? 2 : t == GK_CU_FUSE_HEAD3 ? 3 : 1;
    }

    *st = gk_cuda_launch_one(ctx, gk_graph_node(graph, i));
    return *st == GK_STATUS_SUCCESS ? 1 : 0;
}

// GK_CUDA_GRAPH_DUMP=N: print every graph of at least N nodes, once per
// distinct node count, as one line per node - op, name, extents, sources.
// This is how a fusion candidate is found: the op profile says which ops
// cost, this says which ops are *adjacent* and who else reads them.
static void gk_cu_graph_dump(struct gk_cgraph * graph, int n) {
    static long min_n = -2;
    if (min_n == -2) {
        const char * e = getenv("GK_CUDA_GRAPH_DUMP");
        min_n = e != NULL && e[0] != '\0' ? strtol(e, NULL, 10) : -1;
    }
    if (min_n < 0 || n < min_n) {
        return;
    }

    static std::unordered_map<int, bool> seen;
    if (seen[n]) {
        return;
    }
    seen[n] = true;

    gk_logf("gk cuda graph dump: %d nodes\n", n);
    for (int i = 0; i < n; ++i) {
        const struct gk_tensor * t = gk_graph_node(graph, i);
        char line[512];
        int  off = snprintf(line, sizeof(line),
                            "  %4d %-12s %-6s [%lld %lld %lld %lld] %s <-",
                            i, gk_op_name(t->op), gk_type_name(t->type),
                            (long long) t->ne[0], (long long) t->ne[1],
                            (long long) t->ne[2], (long long) t->ne[3], t->name);
        for (int s = 0; s < GK_MAX_SRC && t->src[s] != NULL && off < (int) sizeof(line) - 2; ++s) {
            off += snprintf(line + off, sizeof(line) - (size_t) off, " %s[%lld,%lld,%lld,%lld]",
                            t->src[s]->name,
                            (long long) t->src[s]->ne[0], (long long) t->src[s]->ne[1],
                            (long long) t->src[s]->ne[2], (long long) t->src[s]->ne[3]);
        }
        gk_logf("%s\n", line);
    }
}

// The bare launch loop: every node in order, fused pairs as one. This is
// what runs when nothing is being measured - directly, under stream capture,
// and as the fallback when a capture goes wrong.
static enum gk_status gk_cuda_launch_nodes(struct gk_cuda_backend_ctx * ctx,
                                           struct gk_cgraph * graph, int n) {
    std::vector<uint8_t> tag;
    std::vector<int32_t> aux;
    gk_cu_fuse_plan(&ctx->scratch, graph, n, tag, aux);
    gk_cu_graph_dump(graph, n);

    int i = 0;
    while (i < n) {
        enum gk_status st;
        const int took = gk_cu_launch_planned(ctx, graph, i, tag[(size_t) i], aux, &st);
        if (took == 0) {
            return st;
        }
        i += took;
    }
    return GK_STATUS_SUCCESS;
}

// GK_CUDA_GRAPH_PROF=N: bucket size for the replay-time profile, 0 = off.
static int gk_cu_graph_prof_bucket(void) {
    static int b = -1;
    if (b < 0) {
        const char * e = getenv("GK_CUDA_GRAPH_PROF");
        b = 0;
        if (e != NULL && e[0] != '\0') {
            const long v = strtol(e, NULL, 10);
            b = v > 0 ? (int) v : (v != 0 ? 32 : 0);
        }
    }
    return b;
}

// The same loop with an event dropped every `bucket` launches, used only
// under capture: the events become nodes of the graph and every replay
// re-times itself. The harvest lives in the replay path.
// A plain event record captured into a graph becomes an *internal* node:
// executing it signals the event but leaves no timestamp, and ElapsedTime on
// it answers cudaErrorInvalidValue. The `external` flavor is the one whose
// replays behave like real records, timestamps included.
static inline gkError_t gk_cu_event_record_ext(gkEvent_t ev, gkStream_t stream) {
#if defined(GK_USE_HIP)
    return hipEventRecord(ev, stream); // no external flavor; prof stays CUDA-only
#else
    return cudaEventRecordWithFlags(ev, stream, cudaEventRecordExternal);
#endif
}

static enum gk_status gk_cuda_launch_nodes_bucketed(struct gk_cuda_backend_ctx * ctx,
                                                    struct gk_cgraph * graph, int n,
                                                    struct gk_cu_graph_entry * ge, int bucket) {
    std::vector<uint8_t> tag;
    std::vector<int32_t> aux;
    gk_cu_fuse_plan(&ctx->scratch, graph, n, tag, aux);

    GK_CUDA_CHECK(gk_cu_event_record_ext(ge->pev[0], ctx->stream));
    int i = 0;
    while (i < n) {
        enum gk_status st;
        const int took = gk_cu_launch_planned(ctx, graph, i, tag[(size_t) i], aux, &st);
        if (took == 0) {
            return st;
        }
        // Record every boundary the fused step crossed, so the event count
        // and indexing stay exactly the unfused loop's.
        for (int j = i; j < i + took; ++j) {
            if ((j + 1) % bucket == 0 || j + 1 == n) {
                GK_CUDA_CHECK(gk_cu_event_record_ext(ge->pev[(size_t) (j / bucket) + 1], ctx->stream));
            }
        }
        i += took;
    }
    return GK_STATUS_SUCCESS;
}

static void gk_cu_graph_prof_dump(struct gk_cuda_backend_ctx * ctx,
                                  struct gk_cu_graph_entry * ge) {
    const size_t nb = ge->pms.size();

    double total = 0.0;
    for (size_t k = 0; k < nb; ++k) {
        total += ge->pms[k];
    }

    std::vector<size_t> order(nb);
    for (size_t k = 0; k < nb; ++k) {
        order[k] = k;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b2) {
        return ge->pms[a] > ge->pms[b2];
    });

    gk_logf("gk cuda graph prof: %s: %.2f ms/graph over %lld replays, hottest buckets:\n",
            ctx->dev->name, total / (double) ge->pn, (long long) ge->pn);
    for (size_t r = 0; r < nb && r < 20; ++r) {
        const size_t k = order[r];
        gk_logf("  %8.1f us  %4.1f%%  %s\n",
                1000.0 * ge->pms[k] / (double) ge->pn,
                total > 0.0 ? 100.0 * ge->pms[k] / total : 0.0,
                ge->plabel[k].c_str());
    }
}

// The graph-aware compute path: replay when the recorded launches still hold,
// capture when a topology proves it repeats, plain launches otherwise.
static enum gk_status gk_cuda_compute_graphed(struct gk_cuda_backend_ctx * ctx,
                                              struct gk_cgraph * graph, int n) {
    struct gk_cu_graph_entry * ge = NULL;
    enum gk_cu_graph_action action = GK_CU_GRAPH_PLAIN;

    if (gk_cu_graphs_on() && n >= GK_CU_GRAPH_MIN_NODES) {
        action = gk_cu_graph_lookup(ctx, graph, n, &ge);
    }

    if (action == GK_CU_GRAPH_REPLAY) {
        struct gk_cu_graph_cache * gc = ctx->graphs;

        // Harvest the previous replay's bucket events before this launch
        // re-records them. The stream has been synchronized since (the
        // engine reads the logits between tokens), so a completed last event
        // means every timestamp is in place.
        if (gk_cu_graph_prof_bucket() > 0 && gk_cu_graph_log_on()) {
            static int64_t seen = 0;
            if (++seen % 100 == 0 && !ge->pev.empty()) {
                float ms = -1.0f;
                const gkError_t eerr = gkEventElapsedTime(&ms, ge->pev[0], ge->pev.back());
                gk_logf("gk cuda graph prof: guard: pev=%zu qback=%d elapsed=%d %.3f ms pn=%lld\n",
                        ge->pev.size(), (int) gkEventQuery(ge->pev.back()),
                        (int) eerr, ms, (long long) ge->pn);
            }
        }
        // A graph-recorded event does not answer cudaEventQuery the way a
        // stream-recorded one does; whether its timestamps are usable is
        // asked of ElapsedTime directly. The engine synchronizes between
        // tokens, so by the next call the previous replay has fully retired.
        if (!ge->pev.empty()) {
            float span = 0.0f;
            if (gkEventElapsedTime(&span, ge->pev[0], ge->pev.back()) == gkSuccess) {
            for (size_t k = 0; k + 1 < ge->pev.size(); ++k) {
                float ms = 0.0f;
                if (gkEventElapsedTime(&ms, ge->pev[k], ge->pev[k + 1]) == gkSuccess) {
                    ge->pms[k] += ms;
                }
            }
            ge->pn++;
            // Replays spread across several sibling entries (the compute
            // buffer rotates through a handful of base addresses), so any
            // one entry accumulates slowly; dump early rather than never.
            if (ge->pn % 40 == 0) {
                gk_cu_graph_prof_dump(ctx, ge);
            }
            }
        }

        const bool timing = gk_cu_graph_log_level() >= 2;
        if (timing) {
            if (gc->armed && gkEventQuery(gc->tev1) == gkSuccess) {
                float ms = 0.0f;
                if (gkEventElapsedTime(&ms, gc->tev0, gc->tev1) == gkSuccess) {
                    gc->dev_ms += ms;
                    gc->dev_n  += 1;
                    if (gc->dev_n % 200 == 0) {
                        gk_logf("gk cuda graphs: %s: device %.2f ms/graph over %lld replays\n",
                                ctx->dev->name, gc->dev_ms / (double) gc->dev_n,
                                (long long) gc->dev_n);
                    }
                }
                gc->armed = false;
            }
            if (!gc->armed) {
                if (gc->tev0 == NULL && gkEventCreate(&gc->tev0) != gkSuccess) { gc->tev0 = NULL; }
                if (gc->tev1 == NULL && gkEventCreate(&gc->tev1) != gkSuccess) { gc->tev1 = NULL; }
                if (gc->tev0 != NULL && gc->tev1 != NULL) {
                    GK_CUDA_CHECK(gkEventRecord(gc->tev0, ctx->stream));
                }
            }
        }
        if (gkGraphLaunch(ge->exec, ctx->stream) == gkSuccess) {
            if (timing && !gc->armed && gc->tev0 != NULL && gc->tev1 != NULL) {
                GK_CUDA_CHECK(gkEventRecord(gc->tev1, ctx->stream));
                gc->armed = true;
            }
            g_graph_replays++;
            return GK_STATUS_SUCCESS;
        }
        // A replay refusing to launch is not a reason to fail the step;
        // plain launches always work.
        (void) gkGetLastError();
        GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
        GK_CUDA_CHECK(gkGraphExecDestroy(ge->exec));
        ge->exec   = NULL;
        ge->broken = true;
        g_graph_breaks++;
        action = GK_CU_GRAPH_PLAIN;
    }

    bool capturing = false;
    int  pbucket   = 0;
    if (action == GK_CU_GRAPH_CAPTURE) {
        // Whatever events a previous capture recorded belong to a graph that
        // is about to be replaced; a stale set would harvest frozen
        // timestamps forever.
        ge->prof_reset();

        // The profile's events have to exist before capture starts; their
        // records become nodes of the graph being recorded.
        pbucket = gk_cu_graph_prof_bucket();
        if (pbucket > 0) {
            const size_t nb = (size_t) ((n + pbucket - 1) / pbucket) + 1;
            ge->pev.assign(nb, NULL);
            for (size_t k = 0; k < nb; ++k) {
                if (gkEventCreate(&ge->pev[k]) != gkSuccess) {
                    ge->pev[k] = NULL;
                    ge->prof_reset();
                    pbucket = 0;
                    break;
                }
            }
            if (pbucket > 0) {
                ge->pms.assign(nb - 1, 0.0);
                ge->plabel.resize(nb - 1);
                for (size_t k = 0; k + 1 < nb; ++k) {
                    const struct gk_tensor * first = gk_graph_node(graph, (int) k * pbucket);
                    char buf[160];
                    snprintf(buf, sizeof(buf), "nodes %5d.. %s (%s)",
                             (int) k * pbucket, first->name, gk_op_name(first->op));
                    ge->plabel[k] = buf;
                }
            }
        }

        capturing = gkStreamBeginCapture(ctx->stream, gkStreamCaptureModeRelaxed) == gkSuccess;
        if (!capturing) {
            (void) gkGetLastError();
            ge->broken = true;
            g_graph_breaks++;
        }
    }

    enum gk_status status = capturing && pbucket > 0
        ? gk_cuda_launch_nodes_bucketed(ctx, graph, n, ge, pbucket)
        : gk_cuda_launch_nodes(ctx, graph, n);

    if (capturing) {
        gkGraph_t g = NULL;
        const gkError_t cerr = gkStreamEndCapture(ctx->stream, &g);

        if (status != GK_STATUS_SUCCESS) {
            // The loop failed mid-capture; nothing was executed and nothing
            // will be. Fail the graph exactly as the plain path would.
            if (g != NULL) {
                GK_CUDA_CHECK(gkGraphDestroy(g));
            }
            ge->broken = true;
            g_graph_breaks++;
            return status;
        }

        gkGraphExec_t exec = NULL;
        if (cerr == gkSuccess && g != NULL &&
            gkGraphInstantiate(&exec, g) == gkSuccess && exec != NULL) {
            ge->exec = exec;
            ge->gen  = ctx->scratch.gen;
            g_graph_captures++;
            if (gk_cu_graph_log_on()) {
                // How many *CUDA* nodes the capture actually produced - the
                // kernels (views launch nothing, mat-vecs launch two) plus
                // any profiling event nodes. Per-kernel replay cost is the
                // graph's device time over this, not over the gk node count.
                size_t n_cuda = 0;
#if defined(GK_USE_HIP)
                (void) hipGraphGetNodes(g, NULL, &n_cuda);
#else
                (void) cudaGraphGetNodes(g, NULL, &n_cuda);
#endif
                gk_logf("gk cuda graphs: captured %d gk nodes -> %zu cuda nodes on %s\n",
                        n, n_cuda, ctx->dev->name);

                // The composition, because at ~8 us of fixed cost per kernel
                // on this driver the op mix IS the time: what to fuse or
                // batch is read straight off this table.
                int op_count[GK_OP_COUNT] = { 0 };
                for (int i = 0; i < n; ++i) {
                    op_count[(int) gk_graph_node(graph, i)->op]++;
                }
                for (int o = 0; o < GK_OP_COUNT; ++o) {
                    if (op_count[o] > 0) {
                        gk_logf("    %4d x %s\n", op_count[o], gk_op_name((enum gk_op) o));
                    }
                }
            }
        } else {
            // An op synchronized or allocated mid-capture. Remember that this
            // topology cannot be captured rather than paying the failed
            // attempt again every round.
            (void) gkGetLastError();
            ge->broken = true;
            g_graph_breaks++;
            if (gk_cu_graph_log_on()) {
                gk_logf("gk cuda graphs: capture failed (%s), staying on plain launches\n",
                        cerr != gkSuccess ? gkGetErrorString(cerr) : "instantiate");
            }
        }
        if (g != NULL) {
            GK_CUDA_CHECK(gkGraphDestroy(g));
        }

        // Captured launches never executed: the work of this call still has
        // to happen. Replay the fresh graph, or launch plain if it broke.
        if (ge->exec != NULL && gkGraphLaunch(ge->exec, ctx->stream) == gkSuccess) {
            g_graph_replays++;
            return GK_STATUS_SUCCESS;
        }
        return gk_cuda_launch_nodes(ctx, graph, n);
    }

    return status;
}

static const char * gk_cuda_backend_name(gk_backend_t backend) {
    return ((struct gk_cuda_backend_ctx *) backend->context)->dev->name;
}

static void gk_cuda_backend_free(gk_backend_t backend) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;
    if (ctx != NULL) {
        GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
        GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
        // After the wait, so nothing is still reading it.
        gk_cu_graph_cache_free(ctx);
        if (ctx->scratch.ptr != NULL) {
            GK_CUDA_CHECK(gkFree(ctx->scratch.ptr));
        }
        GK_CUDA_CHECK(gkStreamDestroy(ctx->stream));
        free(ctx);
    }
    free(backend);
}

static gk_backend_buffer_type_t gk_cuda_backend_buft(gk_backend_t backend) {
    return &((struct gk_cuda_backend_ctx *) backend->context)->dev->buft;
}

// --------------------------------------------------------------------------
// per-op profile
//
// What the scheduler report is for placement, this is for time. GK_SCHED_REPORT
// answers "which backend ran this op"; this answers "where did the graph's
// milliseconds go", which is the only question worth asking once every op is
// already on the device and the whole thing is still slower than it should be.
//
// Set GK_OP_PROFILE in the environment. Every node is timed by synchronizing
// the stream around its launch - that serializes what would otherwise overlap,
// so the total reads high, but a graph this deep is nearly serial anyway and
// the shares are what the number is for. Rows are keyed by op and by the shape
// that op saw, because "mul_mat is expensive" is not actionable and "mul_mat,
// f16 weights, 2880x4096" is.
// --------------------------------------------------------------------------

#define GK_CU_PROF_MAX 4096

struct gk_cu_prof_row {
    char   key[96];
    double ms;
    int64_t calls;
    double flops;   // multiply-accumulates, for the shapes where it means something
};

static struct gk_cu_prof_row g_prof[GK_CU_PROF_MAX];
static int                   g_prof_n       = 0;
static int                   g_prof_enabled = -1;
static double                g_prof_total   = 0.0;

static int gk_cu_prof_cmp(const void * a, const void * b) {
    const double x = ((const struct gk_cu_prof_row *) a)->ms;
    const double y = ((const struct gk_cu_prof_row *) b)->ms;
    return x < y ? 1 : (x > y ? -1 : 0);
}

struct gk_cu_scratch_stats g_gk_scratch_stats;

static void gk_cu_prof_dump(void) {
    if (g_prof_n == 0) {
        return;
    }

    extern double g_gk_mm_quant_ms;
    extern double g_gk_mm_tile_ms;
    gk_logf("\ngk cuda nvfp4 mma: %.1f ms quantizing activations, %.1f ms in the tile\n",
            g_gk_mm_quant_ms, g_gk_mm_tile_ms);

    {
        // GK_MM_FP4_STATS, if it was on. Zero counters mean it was not.
        double sq_err = 0.0, sq_ref = 0.0;
        unsigned long long zero = 0, groups = 0;

        gk_cuda_fp4_stats(&sq_err, &sq_ref, &zero, &groups);

        if (groups != 0) {
            gk_logf("\ngk cuda nvfp4 activations: rms error %.4f%% of signal, "
                    "%llu of %llu groups of 16 zeroed by scale underflow (%.3f%%)\n",
                    100.0 * (sq_ref > 0.0 ? sqrt(sq_err / sq_ref) : 0.0),
                    (unsigned long long) zero, (unsigned long long) groups,
                    100.0 * (double) zero / (double) groups);
        }
    }

    gk_logf("\ngk cuda scratch: %lld calls, %lld grows (%lld failed), %.1f ms in the grow path\n",
            (long long) g_gk_scratch_stats.calls, (long long) g_gk_scratch_stats.grows,
            (long long) g_gk_scratch_stats.fails, g_gk_scratch_stats.grow_ms);

    qsort(g_prof, (size_t) g_prof_n, sizeof(g_prof[0]), gk_cu_prof_cmp);

    gk_logf("\ngk cuda profile: %.1f ms over %d distinct shapes\n", g_prof_total, g_prof_n);
    gk_logf("  %-62s %10s %7s %8s %10s\n", "op / shape", "ms", "%", "calls", "GFLOP/s");

    for (int i = 0; i < g_prof_n; ++i) {
        char rate[16] = "-";
        if (g_prof[i].flops > 0.0 && g_prof[i].ms > 0.0) {
            snprintf(rate, sizeof(rate), "%.0f", g_prof[i].flops * 2.0 / (g_prof[i].ms * 1e6));
        }
        gk_logf("  %-62s %10.2f %6.1f%% %8lld %10s\n",
                g_prof[i].key, g_prof[i].ms,
                100.0 * g_prof[i].ms / g_prof_total,
                (long long) g_prof[i].calls, rate);
    }
}

static void gk_cu_prof_add(const char * key, double ms, double flops) {
    for (int i = 0; i < g_prof_n; ++i) {
        if (strcmp(g_prof[i].key, key) == 0) {
            g_prof[i].ms    += ms;
            g_prof[i].flops += flops;
            g_prof[i].calls++;
            g_prof_total    += ms;
            return;
        }
    }

    if (g_prof_n >= GK_CU_PROF_MAX) {
        return;
    }

    struct gk_cu_prof_row * row = &g_prof[g_prof_n++];
    snprintf(row->key, sizeof(row->key), "%s", key);
    row->ms    = ms;
    row->flops = flops;
    row->calls = 1;
    g_prof_total += ms;
}

// The name a row gets, and the work it did. Only the ops whose cost depends on
// more than the output size say anything beyond their shape.
static void gk_cu_prof_key(const struct gk_tensor * node, char * out, size_t out_size,
                           double * flops) {
    *flops = 0.0;

    switch ((int) node->op) {
        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID: {
            const struct gk_tensor * a = node->src[0];
            const struct gk_tensor * b = node->src[1];
            const int64_t k = a->ne[0];
            const int64_t m = node->ne[0];
            const int64_t n = node->ne[1] * node->ne[2] * node->ne[3];
            snprintf(out, out_size, "%s %-6s %lldx%lldx%lld [%s]",
                     gk_op_name(node->op), gk_type_name(a->type),
                     (long long) m, (long long) n, (long long) k,
                     node->op == GK_OP_MUL_MAT ? gk_cuda_mm_last_path() : "-");
            *flops = (double) m * (double) n * (double) k;
            GK_UNUSED(b);
            break;
        }
        case GK_OP_FLASH_ATTN_EXT: {
            const struct gk_tensor * q = node->src[0];
            const struct gk_tensor * k = node->src[1];
            snprintf(out, out_size, "flash_attn d=%lld q=%lld kv=%lld h=%lld",
                     (long long) q->ne[0], (long long) q->ne[1],
                     (long long) k->ne[1], (long long) q->ne[2]);
            *flops = 2.0 * (double) q->ne[0] * (double) q->ne[1] *
                     (double) k->ne[1] * (double) q->ne[2];
            break;
        }
        case GK_OP_IM2COL: {
            snprintf(out, out_size, "im2col %lldx%lldx%lld",
                     (long long) node->ne[0], (long long) node->ne[1],
                     (long long) node->ne[2]);
            break;
        }
        default:
            snprintf(out, out_size, "%s %lldx%lldx%lldx%lld",
                     gk_op_name(node->op),
                     (long long) node->ne[0], (long long) node->ne[1],
                     (long long) node->ne[2], (long long) node->ne[3]);
            break;
    }
}

// How many graphs to run before dumping, or 0 for "only at exit". A server does
// not exit on demand - there is no way to send it a signal from another OS, and
// a forced kill runs no atexit handler - so a long-running process could be
// profiled and then never print. `GK_OP_PROFILE=N` for N > 1 dumps every N
// graphs instead, which also makes the numbers a rate rather than a total.
static int  g_prof_every = 0;
static long g_prof_graphs = 0;

static bool gk_cu_prof_on(void) {
    if (g_prof_enabled < 0) {
        const char * e = getenv("GK_OP_PROFILE");
        g_prof_enabled = e != NULL && e[0] != '0';
        if (g_prof_enabled) {
            const int n = atoi(e);
            g_prof_every = n > 1 ? n : 0;
            atexit(gk_cu_prof_dump);
        }
    }
    return g_prof_enabled != 0;
}

// --------------------------------------------------------------------------
// launch profile
//
// GK_OP_PROFILE answers "which op costs the milliseconds". This answers a
// different question that shape-level profile cannot: is the GPU actually
// busy for those milliseconds, or is it idle waiting for this thread to hand
// it the next kernel?
//
// A graph this deep is thousands of launches, and a launch costs host time
// whether or not the device has anything left to do. Timing the loop without
// synchronizing gives the host cost - every launch here is asynchronous, so
// the loop returns as soon as the last one is *queued*. Synchronizing after
// it gives what the device still had left. If the queueing time is the larger
// of the two, the kernels are irrelevant: the device finished each one before
// the next arrived, and the fix is fewer launches, not faster ones.
//
// Set GK_LAUNCH_PROFILE=1.
// --------------------------------------------------------------------------

static double            g_launch_host_ms = 0.0;   // time spent queueing
static double            g_launch_wait_ms = 0.0;   // device time left after queueing
static long long         g_launch_nodes   = 0;
static long long         g_launch_graphs  = 0;
static int               g_launch_enabled = -1;

// GK_LAUNCH_PROFILE=2 additionally brackets every launch in a pair of events.
// Host timing cannot separate "the device is idle waiting for this thread"
// from "this thread is blocked because the device is behind" - both show up as
// time inside the loop. Events are recorded in the stream, so the gap between
// one kernel's end event and the next kernel's start event is device idle time
// and nothing else. Summing the brackets gives busy; the span from the first
// start to the last end gives elapsed; the difference is what a captured graph
// could give back.
static double            g_launch_busy_ms = 0.0;
static double            g_launch_span_ms = 0.0;
static bool              g_launch_events  = false;

static void gk_cu_launch_prof_dump(void) {
    if (g_launch_graphs == 0) {
        return;
    }

    const double total = g_launch_host_ms + g_launch_wait_ms;

    gk_logf("\ngk cuda launch profile: %lld graphs, %lld nodes (%.0f nodes/graph)\n",
            (long long) g_launch_graphs, (long long) g_launch_nodes,
            (double) g_launch_nodes / (double) g_launch_graphs);
    gk_logf("  host, queueing launches   %9.1f ms  %5.1f%%   (%.2f us/node)\n",
            g_launch_host_ms, total > 0.0 ? 100.0 * g_launch_host_ms / total : 0.0,
            g_launch_nodes ? 1000.0 * g_launch_host_ms / (double) g_launch_nodes : 0.0);
    gk_logf("  device, after queueing    %9.1f ms  %5.1f%%\n",
            g_launch_wait_ms, total > 0.0 ? 100.0 * g_launch_wait_ms / total : 0.0);

    if (g_launch_span_ms > 0.0) {
        gk_logf("  device busy (events)      %9.1f ms  %5.1f%% of the %.1f ms the stream spanned\n",
                g_launch_busy_ms, 100.0 * g_launch_busy_ms / g_launch_span_ms, g_launch_span_ms);
        gk_logf("  device idle between ops   %9.1f ms  %5.1f%%   <- what a captured graph can recover\n",
                g_launch_span_ms - g_launch_busy_ms,
                100.0 * (g_launch_span_ms - g_launch_busy_ms) / g_launch_span_ms);
    } else {
        gk_logf("  (set GK_LAUNCH_PROFILE=2 to bracket every launch in events and separate\n"
                "   device-idle from host-blocked - host time alone cannot tell them apart.)\n");
    }
}

static bool gk_cu_launch_prof_on(void) {
    if (g_launch_enabled < 0) {
        const char * e = getenv("GK_LAUNCH_PROFILE");
        g_launch_enabled = e != NULL && e[0] != '0';
        g_launch_events  = e != NULL && e[0] == '2';
        if (g_launch_enabled) {
            atexit(gk_cu_launch_prof_dump);
        }
    }
    return g_launch_enabled != 0;
}

// One pair per node, reused across graphs and grown on demand. Creating them
// per graph would time the event allocator as much as the kernels.
static std::vector<gkEvent_t> g_launch_ev;

static void gk_cu_launch_ev_reserve(int n) {
    const size_t want = (size_t) n * 2;
    while (g_launch_ev.size() < want) {
        gkEvent_t e = NULL;
        if (gkEventCreate(&e) != gkSuccess) {
            g_launch_events = false;   // out of events: fall back to host timing
            return;
        }
        g_launch_ev.push_back(e);
    }
}

// GK_NODE_HASH=1: a checksum of every node's output, in graph order, with the
// per-node synchronization to make it meaningful. Two runs that should agree -
// one with the allocator reusing memory and one without, or one on each
// backend - diff down to the first node where they stop agreeing, which is
// where the bug is rather than where the symptom is.
static bool gk_cu_node_hash_on(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GK_NODE_HASH");
        on = e != NULL && e[0] != '0';
    }
    return on != 0;
}

static void gk_cu_node_hash(gkStream_t stream, int index, const struct gk_tensor * node) {
    if (node->data == NULL || !gk_is_contiguous(node)) {
        gk_logf("nhash %4d %-14s %-22s -\n", index, gk_op_name(node->op), node->name);
        return;
    }

    const size_t bytes = gk_nbytes(node);
    if (bytes == 0 || bytes > (32u << 20)) {
        gk_logf("nhash %4d %-14s %-22s -\n", index, gk_op_name(node->op), node->name);
        return;
    }

    GK_CUDA_CHECK(gkStreamSynchronize(stream));

    unsigned char * host = (unsigned char *) malloc(bytes);
    if (host == NULL) {
        return;
    }
    GK_CUDA_CHECK(gkMemcpy(host, node->data, bytes, gkMemcpyDeviceToHost));

    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        h = (h ^ host[i]) * 1099511628211ull;
    }
    free(host);

    gk_logf("nhash %4d %-14s %-22s %016llx ne=[%lld %lld %lld %lld] dst=%p s0=%p(%s) s1=%p(%s) %s\n",
            index, gk_op_name(node->op), node->name, h,
            (long long) node->ne[0], (long long) node->ne[1],
            (long long) node->ne[2], (long long) node->ne[3],
            node->data,
            node->src[0] ? node->src[0]->data : NULL,
            node->src[0] ? gk_type_name(node->src[0]->type) : "-",
            node->src[1] ? node->src[1]->data : NULL,
            node->src[1] ? gk_type_name(node->src[1]->type) : "-",
            node->op == GK_OP_MUL_MAT ? gk_cuda_mm_last_path() : "");
}

static enum gk_status gk_cuda_backend_compute(gk_backend_t backend, struct gk_cgraph * graph) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));

    // The per-node check below reads the thread's error flag, and that flag
    // carries whatever the last unchecked runtime call left in it - from
    // anywhere, including code that has nothing to do with this graph. Clear it
    // once here so that what the loop reports is this graph's, not a stale
    // error attributed to node 0 because node 0 happened to look next.
    (void) gkGetLastError();

    const int  n    = gk_graph_n_nodes(graph);
    const bool prof = gk_cu_prof_on();

    // A new execution: whatever activation the scratch held belongs to the
    // previous graph's numbers.
    ctx->scratch.pass++;

    // Deliberately not `prof`: the per-node profile synchronizes around every
    // launch, which is exactly the overlap this measurement is about. The two
    // are mutually exclusive and the launch one wins.
    const bool lprof = !prof && gk_cu_launch_prof_on();

    // The unmeasured run - which is every production run - goes through the
    // graph cache. The profiled paths below cannot: both synchronize inside
    // the loop, which is meaningless under capture and worse under replay,
    // where there are no per-node launches to measure at all.
    const bool nhash = gk_cu_node_hash_on();

    if (!prof && !lprof && !nhash) {
        return gk_cuda_compute_graphed(ctx, graph, n);
    }

    std::chrono::steady_clock::time_point lt0;
    const bool lev = lprof && g_launch_events;
    if (lprof) {
        if (lev) {
            gk_cu_launch_ev_reserve(n);
        }
        lt0 = std::chrono::steady_clock::now();
    }

    for (int i = 0; i < n; ++i) {
        struct gk_tensor * node = gk_graph_node(graph, i);

        if (lev && g_launch_events) {
            GK_CUDA_CHECK(gkEventRecord(g_launch_ev[(size_t) i * 2], ctx->stream));
        }

        std::chrono::steady_clock::time_point t0;
        if (prof) {
            GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
            t0 = std::chrono::steady_clock::now();
        }

        if (!gk_cuda_compute_op(ctx->stream, &ctx->scratch, node)) {
            gk_logf("gk %s: no kernel for op %s (node %s)\n",
                    GK_CUDA_BACKEND_NAME, gk_op_name(node->op), node->name);
            return GK_STATUS_NO_STORAGE;
        }

        // A launch is rejected synchronously when its geometry is wrong, so the
        // check belongs next to the launch that caused it: checking once at the
        // end of the graph would name whichever node happened to be queued last
        // and leave the real one unnamed. It is a host-side flag read, not a
        // synchronization - the queue keeps running behind it.
        const gkError_t err = gkGetLastError();
        if (err != gkSuccess) {
            gk_logf("gk %s: %s (node %s, op %s, ne = [%lld %lld %lld %lld])\n",
                    GK_CUDA_BACKEND_NAME, gkGetErrorString(err),
                    node->name, gk_op_name(node->op),
                    (long long) node->ne[0], (long long) node->ne[1],
                    (long long) node->ne[2], (long long) node->ne[3]);
            return GK_STATUS_NO_STORAGE;
        }

        if (nhash) {
            gk_cu_node_hash(ctx->stream, i, node);
        }

        if (lev && g_launch_events) {
            GK_CUDA_CHECK(gkEventRecord(g_launch_ev[(size_t) i * 2 + 1], ctx->stream));
        }

        if (prof) {
            GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
            const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            char   key[96];
            double flops = 0.0;
            gk_cu_prof_key(node, key, sizeof(key), &flops);
            gk_cu_prof_add(key,
                           std::chrono::duration<double, std::milli>(t1 - t0).count(),
                           flops);
        }
    }

    if (prof && g_prof_every > 0 && ++g_prof_graphs % g_prof_every == 0) {
        gk_cu_prof_dump();
    }

    if (lprof) {
        const std::chrono::steady_clock::time_point lt1 = std::chrono::steady_clock::now();
        GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
        const std::chrono::steady_clock::time_point lt2 = std::chrono::steady_clock::now();

        g_launch_host_ms += std::chrono::duration<double, std::milli>(lt1 - lt0).count();
        g_launch_wait_ms += std::chrono::duration<double, std::milli>(lt2 - lt1).count();
        g_launch_nodes   += n;
        g_launch_graphs  += 1;

        // A server dies by taskkill, which skips atexit; a periodic dump is
        // the only way the numbers ever reach the log there.
        if (g_launch_graphs % 200 == 0) {
            gk_cu_launch_prof_dump();
        }

        if (lev && g_launch_events && n > 0) {
            // The stream is already drained, so every event has a timestamp.
            // Sum the brackets for busy time and take first-to-last for the
            // span; their difference is device idle.
            double busy = 0.0;
            for (int i = 0; i < n; ++i) {
                float ms = 0.0f;
                if (gkEventElapsedTime(&ms, g_launch_ev[(size_t) i * 2],
                                       g_launch_ev[(size_t) i * 2 + 1]) == gkSuccess) {
                    busy += ms;
                }
            }

            float span = 0.0f;
            if (gkEventElapsedTime(&span, g_launch_ev[0],
                                   g_launch_ev[(size_t) (n - 1) * 2 + 1]) == gkSuccess) {
                g_launch_span_ms += span;
                g_launch_busy_ms += busy;
            }
        }
    }

    return GK_STATUS_SUCCESS;
}

static bool gk_cuda_backend_supports_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);
    return gk_cuda_supports_op(op);
}

static bool gk_cuda_backend_supports_buft(gk_backend_t backend, gk_backend_buffer_type_t buft) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    // Its own memory only. Pinned host memory is addressable from the device,
    // but "addressable" is not "readable at speed": a kernel that walks an
    // operand more than once - a mat-vec re-reading the activation column per
    // output row - multiplies every in-place read by the row count, over the
    // bus. Declining it here makes the scheduler stage such a tensor into
    // device memory once per graph instead; the pinning still pays for that
    // copy's speed, which is what it is for.
    return buft == &ctx->dev->buft;
}

// Whether it is worth pulling an op here that would otherwise run elsewhere.
// The trade is one weight's worth of transfer against the op's work, so the
// answer is yes only for the ops whose work grows with the batch: a matmul
// over many tokens pays the copy back, an elementwise add never does.
//
// The scheduler only asks about a node whose weight is in host memory, which
// is why there is no flash-attention case here. Attention has no weight; its
// operands are the KV cache, and a cache is not a thing to move.
static bool gk_cuda_backend_offload_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);

    const int64_t min_batch = 32;

    switch ((int) op->op) {
        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            return op->src[1]->ne[1] >= min_batch;
        default:
            return false;
    }
}

static void gk_cuda_backend_synchronize(gk_backend_t backend) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
    GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
}

static void gk_cuda_backend_set_async(gk_backend_t backend, struct gk_tensor * tensor,
                                      const void * data, size_t offset, size_t size) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
    GK_CUDA_CHECK(gkMemcpyAsync((char *) tensor->data + offset, data, size,
                                gkMemcpyHostToDevice, ctx->stream));
}

static void gk_cuda_backend_get_async(gk_backend_t backend, const struct gk_tensor * tensor,
                                      void * data, size_t offset, size_t size) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
    GK_CUDA_CHECK(gkMemcpyAsync(data, (const char *) tensor->data + offset, size,
                                gkMemcpyDeviceToHost, ctx->stream));
}

static const struct gk_backend_i g_cuda_backend_iface = {
    /* .get_name                = */ gk_cuda_backend_name,
    /* .free                    = */ gk_cuda_backend_free,
    /* .get_default_buffer_type = */ gk_cuda_backend_buft,
    /* .graph_compute           = */ gk_cuda_backend_compute,
    /* .supports_op             = */ gk_cuda_backend_supports_op,
    /* .supports_buft           = */ gk_cuda_backend_supports_buft,
    /* .offload_op              = */ gk_cuda_backend_offload_op,
    /* .synchronize             = */ gk_cuda_backend_synchronize,
    /* .set_tensor_async        = */ gk_cuda_backend_set_async,
    /* .get_tensor_async        = */ gk_cuda_backend_get_async,
};

// --------------------------------------------------------------------------
// the device vtable
// --------------------------------------------------------------------------

static const char * gk_cuda_device_name(gk_device_t dev) {
    return ((struct gk_cuda_device_ctx *) dev->context)->name;
}

static const char * gk_cuda_device_description(gk_device_t dev) {
    return ((struct gk_cuda_device_ctx *) dev->context)->description;
}

static void gk_cuda_device_memory(gk_device_t dev, size_t * free_out, size_t * total_out) {
    struct gk_cuda_device_ctx * ctx = (struct gk_cuda_device_ctx *) dev->context;

    size_t free_mem = 0, total_mem = ctx->total_memory;

    GK_CUDA_CHECK(gkSetDevice(ctx->index));
    GK_CUDA_CHECK(gkMemGetInfo(&free_mem, &total_mem));

    if (free_out  != NULL) { *free_out  = free_mem; }
    if (total_out != NULL) { *total_out = total_mem; }
}

static enum gk_device_type gk_cuda_device_type(gk_device_t dev) {
    struct gk_cuda_device_ctx * ctx = (struct gk_cuda_device_ctx *) dev->context;
    // An integrated part shares the host's memory, and the scheduler treats
    // that differently: there is no bus to avoid crossing.
    return ctx->integrated ? GK_DEVICE_TYPE_IGPU : GK_DEVICE_TYPE_GPU;
}

static gk_backend_t gk_cuda_device_init_backend(gk_device_t dev) {
    struct gk_cuda_device_ctx * d = (struct gk_cuda_device_ctx *) dev->context;

    struct gk_cuda_backend_ctx * ctx =
        (struct gk_cuda_backend_ctx *) malloc(sizeof(struct gk_cuda_backend_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->dev          = d;
    ctx->graphs       = NULL;
    ctx->scratch.ptr  = NULL;
    ctx->scratch.size = 0;
    ctx->scratch.gen  = 0;
    ctx->scratch.aq_src    = NULL;
    ctx->scratch.aq_tensor = NULL;
    ctx->scratch.aq_blk  = 0;
    ctx->scratch.aq_grp  = 0;
    ctx->scratch.aq_pass = 0;
    ctx->scratch.pass    = 0;
    ctx->scratch.n_sm = d->n_sm;
    ctx->scratch.cc   = d->cc;
    ctx->scratch.smem_max = d->smem_max;

    GK_CUDA_CHECK(gkSetDevice(d->index));

    const gkError_t err = gkStreamCreate(&ctx->stream);
    if (err != gkSuccess) {
        gk_logf("gk %s: could not create a stream on %s: %s\n",
                GK_CUDA_BACKEND_NAME, d->name, gkGetErrorString(err));
        free(ctx);
        return NULL;
    }

    gk_backend_t backend = (gk_backend_t) malloc(sizeof(struct gk_backend));
    if (backend == NULL) {
        GK_CUDA_CHECK(gkStreamDestroy(ctx->stream));
        free(ctx);
        return NULL;
    }

    backend->iface   = g_cuda_backend_iface;
    backend->context = ctx;
    backend->device  = &d->device;

    return backend;
}

static gk_backend_buffer_type_t gk_cuda_device_buft(gk_device_t dev) {
    return &((struct gk_cuda_device_ctx *) dev->context)->buft;
}

static gk_backend_buffer_type_t gk_cuda_device_host_buft(gk_device_t dev) {
    return &((struct gk_cuda_device_ctx *) dev->context)->host_buft;
}

static bool gk_cuda_device_supports_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_cuda_supports_op(op);
}

static bool gk_cuda_device_supports_buft(gk_device_t dev, gk_backend_buffer_type_t buft) {
    // Own device memory only - see gk_cuda_backend_supports_buft for why
    // pinned host memory is deliberately not claimed here.
    struct gk_cuda_device_ctx * ctx = (struct gk_cuda_device_ctx *) dev->context;
    return buft == &ctx->buft;
}

static bool gk_cuda_device_offload_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_cuda_backend_offload_op(NULL, op);
}

// What this binary was compiled for, filled in by the build.
#ifndef GK_CUDA_ARCH_LIST
#define GK_CUDA_ARCH_LIST "unknown"
#endif

static const struct gk_feature * gk_cuda_device_features(gk_device_t dev) {
    return ((struct gk_cuda_device_ctx *) dev->context)->features;
}

// Filled at registration, once the properties are known. "built for" is the
// build's list rather than this device's own capability on purpose: a device
// running on a nearer arch than it was compiled for is the usual reason a GPU
// is slower than it should be, and the two numbers side by side say so.
static void gk_cuda_device_fill_features(struct gk_cuda_device_ctx * d) {
    snprintf(d->cc_str,   sizeof(d->cc_str),   "%d.%d", d->cc / 10, d->cc % 10);
    snprintf(d->n_sm_str, sizeof(d->n_sm_str), "%d",    d->n_sm);
    snprintf(d->smem_str, sizeof(d->smem_str), "%d KiB", d->smem_max / 1024);

    int n = 0;
    #define GK_CUDA_FEATURE(nm, val)                                             \
        do {                                                                     \
            GK_ASSERT(n < (int) (sizeof(d->features) / sizeof(d->features[0]))); \
            d->features[n].name  = (nm);                                         \
            d->features[n].value = (val);                                        \
            n++;                                                                 \
        } while (0)

    GK_CUDA_FEATURE("compute capability", d->cc_str);
    GK_CUDA_FEATURE("SMs",                d->n_sm_str);
    GK_CUDA_FEATURE("shared memory",      d->smem_str);
    GK_CUDA_FEATURE("built for",          GK_CUDA_ARCH_LIST);
    if (d->integrated) {
        GK_CUDA_FEATURE("integrated", "1");
    }
    GK_CUDA_FEATURE(NULL, NULL);

    #undef GK_CUDA_FEATURE
}

static const struct gk_device_i g_cuda_device_iface = {
    /* .get_name             = */ gk_cuda_device_name,
    /* .get_description      = */ gk_cuda_device_description,
    /* .get_memory           = */ gk_cuda_device_memory,
    /* .get_type             = */ gk_cuda_device_type,
    /* .init_backend         = */ gk_cuda_device_init_backend,
    /* .buffer_type          = */ gk_cuda_device_buft,
    /* .host_buffer_type     = */ gk_cuda_device_host_buft,
    /* .buffer_from_host_ptr = */ NULL, // mapping arbitrary host pages is not
                                        // something this backend does; the
                                        // loader copies weights in instead
    /* .supports_op          = */ gk_cuda_device_supports_op,
    /* .supports_buft        = */ gk_cuda_device_supports_buft,
    /* .offload_op           = */ gk_cuda_device_offload_op,
    /* .get_features         = */ gk_cuda_device_features,
};

// --------------------------------------------------------------------------
// discovery
// --------------------------------------------------------------------------

// Launched at discovery to find out whether the binary actually contains code
// this device can run. It does nothing; the launch itself is the question.
static __global__ void gk_cuda_probe_kernel(void) {}

// Whether a device can run any of the code in this binary.
//
// A build carries SASS for the architectures it was told about and PTX for at
// most one of them. Put a card in the machine that is newer than both and every
// launch on it fails with "no kernel image is available for execution on the
// device" - the first launch and every one after, because nothing about it is
// transient. It costs one empty launch to find that out here instead, before
// the device has been registered and before the scheduler has put a graph on
// it, and the difference in what the user sees is the whole point: a named
// device and a rebuild flag at startup, rather than a failed generation with a
// node number in it.
//
// The launch is on the default stream and is checked synchronously, because a
// rejected launch reports through gkGetLastError immediately.
static bool gk_cuda_device_has_kernel_image(int index) {
    if (gkSetDevice(index) != gkSuccess) {
        return false;
    }

    // Whatever came before is not this launch's fault; the flag is per-thread
    // and sticky until read.
    (void) gkGetLastError();

    gk_cuda_probe_kernel<<<1, 1>>>();

    const gkError_t err = gkGetLastError();
    if (err == gkSuccess) {
        return true;
    }

    if (err != gkErrorNoKernelImage) {
        // Something else is wrong with the device - out of memory at context
        // creation, a driver that has fallen over. Not this function's
        // question, and not a reason to claim the binary is at fault, so let
        // the device register and fail with its own error where it happens.
        return true;
    }

    return false;
}

// Whether this binary carries SASS a device of capability `cc` can run.
//
// A cubin runs on any part of the same major version with a minor at least as
// high as its own: sm_80 code runs on an sm_89 card, sm_89 code does not run on
// sm_86 and nothing crosses a major boundary. Anything else the device has to
// JIT from PTX.
static bool gk_cuda_arch_list_has_sass_for(int cc) {
    const char * p = GK_CUDA_ARCH_LIST;

    while (*p) {
        if (*p < '0' || *p > '9') {
            p++;
            continue;
        }

        int arch = 0;
        while (*p >= '0' && *p <= '9') {
            arch = arch * 10 + (*p - '0');
            p++;
        }

        if (arch / 10 == cc / 10 && arch % 10 <= cc % 10) {
            return true;
        }
    }

    return false;
}

extern "C" void gk_cuda_register_devices(void) {
    if (g_cuda_discovered) {
        return;
    }
    g_cuda_discovered = true;

    int count = 0;
    const gkError_t err = gkGetDeviceCount(&count);
    if (err != gkSuccess) {
        // No driver, no devices, or a driver too old for this build: all of
        // them mean "this machine has none", which is not an error - the CPU
        // backend is always there.
        return;
    }

    if (count > GK_CUDA_MAX_DEVICES) {
        count = GK_CUDA_MAX_DEVICES;
    }

    for (int i = 0; i < count; ++i) {
        gkDeviceProp_t prop;
        if (gkGetDeviceProperties(&prop, i) != gkSuccess) {
            continue;
        }

        if (!gk_cuda_device_has_kernel_image(i)) {
            // Registering it would mean the scheduler eventually places work
            // on it, and every one of those launches fails. Leaving it out
            // costs the device and keeps the run: the remaining GPUs, or the
            // CPU, take the graph instead.
            gk_logf("gk %s: skipping device %d (%s, compute capability %d.%d) - this build has "
                    "no code for it (built for: %s). Rebuild with "
                    "-DGK_CUDA_ARCHITECTURES=%d to use it.\n",
                    GK_CUDA_BACKEND_NAME, i, prop.name, prop.major, prop.minor,
                    GK_CUDA_ARCH_LIST, prop.major * 10 + prop.minor);
            continue;
        }

        if (!gk_cuda_arch_list_has_sass_for(prop.major * 10 + prop.minor)) {
            // The probe launch succeeded, so the driver can JIT this build's
            // PTX for the device - but it has to compile every kernel the run
            // touches before running it, and for a graph the size of a
            // diffusion model that is minutes of apparently frozen process
            // with no output between "loading tensors completed" and the first
            // step. It is cached afterwards (%APPDATA%\NVIDIA\ComputeCache,
            // ~/.nv/ComputeCache), so the run after this one looks fine and
            // nothing about it looks like a build problem. Say so once, here,
            // where the cost is about to be paid.
            gk_logf("gk %s: device %d (%s, compute capability %d.%d) has no native code in this "
                    "build (built for: %s); it will run on JIT-compiled PTX. The first launch "
                    "on it can take minutes before anything appears to happen - the driver "
                    "caches the result, so later runs start normally. Rebuild with "
                    "-DGK_CUDA_ARCHITECTURES=%d to avoid it.\n",
                    GK_CUDA_BACKEND_NAME, i, prop.name, prop.major, prop.minor,
                    GK_CUDA_ARCH_LIST, prop.major * 10 + prop.minor);
        }

        struct gk_cuda_device_ctx * d = &g_cuda_devices[g_cuda_n_devices];
        memset(d, 0, sizeof(*d));

        d->index        = i;
        d->total_memory = prop.totalGlobalMem;
        d->integrated   = prop.integrated != 0;
        d->n_sm         = prop.multiProcessorCount;
        d->cc           = prop.major * 10 + prop.minor;
        // Every part gives a block 48 KB without asking; Ampere and later will
        // give most of the multiprocessor's store to one block on request.
        d->smem_max     = (int) prop.sharedMemPerBlockOptin;
        if (d->smem_max < (int) prop.sharedMemPerBlock) {
            d->smem_max = (int) prop.sharedMemPerBlock;
        }

        snprintf(d->name, sizeof(d->name), "%s%d", GK_CUDA_BACKEND_NAME, i);
        snprintf(d->description, sizeof(d->description), "%s", prop.name);

        gk_cuda_device_fill_features(d);

        d->buft.iface   = g_cuda_buft_iface;
        d->buft.context = d;
        d->buft.device  = &d->device;

        d->host_buft.iface   = g_cuda_host_buft_iface;
        d->host_buft.context = d;
        d->host_buft.device  = &d->device;

        d->device.iface   = g_cuda_device_iface;
        d->device.backend = GK_CUDA_BACKEND_NAME;
        d->device.index   = i;
        d->device.context = d;
        snprintf(d->device.name, sizeof(d->device.name), "%s", d->name);

        // Peer access, where the pair supports it, is what lets the scheduler
        // move a tensor from one device to another without a host bounce. It
        // is enabled once here rather than per copy, because enabling it is
        // not free and the set of devices does not change.
        //
        // Paired against the devices already registered rather than every
        // index below this one: a device that failed the kernel-image probe is
        // not going to be given work, so there is no copy to make faster.
        for (int k = 0; k < g_cuda_n_devices; ++k) {
            const int peer = g_cuda_devices[k].index;
            int can = 0;
            if (gkDeviceCanAccessPeer(&can, i, peer) == gkSuccess && can) {
                GK_CUDA_CHECK(gkSetDevice(i));
                gkDeviceEnablePeerAccess(peer, 0);
                GK_CUDA_CHECK(gkSetDevice(peer));
                gkDeviceEnablePeerAccess(i, 0);
            }
        }

        g_cuda_n_devices++;

        gk_device_register(&d->device);
    }

    // Peer access that was already enabled comes back as an error rather than
    // as success, and an unread error is not discarded - it is handed to
    // whoever calls gkGetLastError next. That is the graph loop, which would
    // blame it on its first node. Read it here, where it means nothing.
    (void) gkGetLastError();
}

// --------------------------------------------------------------------------
// the direct entry point
// --------------------------------------------------------------------------

extern "C" gk_backend_t gk_backend_cuda_init(int device) {
    gk_cuda_register_devices();

    // `device` is a CUDA device index - the number in "CUDA1" and the one
    // CUDA_VISIBLE_DEVICES speaks - not a position in the registered array.
    // The two stop agreeing the moment a device is skipped for having no
    // kernel image, and the caller has no way of knowing that happened.
    for (int i = 0; i < g_cuda_n_devices; ++i) {
        if (g_cuda_devices[i].index == device) {
            return gk_cuda_device_init_backend(&g_cuda_devices[i].device);
        }
    }

    return NULL;
}
