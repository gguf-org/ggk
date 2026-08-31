// The backend interface, and the CPU backend that implements it.
//
// Three things, layered:
//
//   buffer type   a kind of memory - "CPU host memory", "device 0 memory".
//                 Knows how to allocate a buffer, what alignment it needs, and
//                 how much space a given tensor costs in it.
//   buffer        one allocation of that kind. Tensors are pointed into it.
//   backend       something that can evaluate a graph, plus the buffer type it
//                 prefers for its own tensors.
//
// Splitting buffer types from backends looks like ceremony with only a CPU
// backend in the tree, and it is the thing that stops being ceremony the
// moment there are two devices: a tensor has to record which memory it lives
// in independently of which device computes on it, because those are not
// always the same. Pinned host memory is allocated by a GPU backend but read
// directly by the CPU; a weight shared between devices lives in one buffer and
// is read by several. `tensor->buffer` answers "where is this", the backend
// answers "who runs this", and keeping them separate is what lets the
// scheduler place a graph across devices later without changing any kernel.
//
// Everything is a small vtable rather than a compile-time switch, so a GPU
// backend is a new file rather than an edit to this one.

#include "gk_impl.h"

#include <stdlib.h>

// --------------------------------------------------------------------------
// buffer type
// --------------------------------------------------------------------------

const char * gk_backend_buft_name(gk_backend_buffer_type_t buft) {
    return buft->iface.get_name(buft);
}

gk_backend_buffer_t gk_backend_buft_alloc_buffer(gk_backend_buffer_type_t buft, size_t size) {
    // A zero-sized buffer is legitimate - a graph whose tensors are all views
    // needs no storage of its own - and every caller would otherwise have to
    // special-case it.
    if (size == 0) {
        return gk_backend_buffer_init(buft, NULL, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t gk_backend_buft_get_alignment(gk_backend_buffer_type_t buft) {
    return buft->iface.get_alignment(buft);
}

size_t gk_backend_buft_get_alloc_size(gk_backend_buffer_type_t buft,
                                      const struct gk_tensor * tensor) {
    if (buft->iface.get_alloc_size != NULL) {
        return buft->iface.get_alloc_size(buft, tensor);
    }
    return gk_nbytes(tensor);
}

bool gk_backend_buft_is_host(gk_backend_buffer_type_t buft) {
    if (buft->iface.is_host != NULL) {
        return buft->iface.is_host(buft);
    }
    return false;
}

// --------------------------------------------------------------------------
// buffer
// --------------------------------------------------------------------------

gk_backend_buffer_t gk_backend_buffer_init(gk_backend_buffer_type_t buft,
                                           const struct gk_backend_buffer_i * iface,
                                           void * context, size_t size) {
    gk_backend_buffer_t buffer =
        (gk_backend_buffer_t) malloc(sizeof(struct gk_backend_buffer));
    if (buffer == NULL) {
        return NULL;
    }

    buffer->buft    = buft;
    buffer->context = context;
    buffer->size    = size;

    if (iface != NULL) {
        buffer->iface = *iface;
    } else {
        memset(&buffer->iface, 0, sizeof(buffer->iface));
    }

    return buffer;
}

void gk_backend_buffer_free(gk_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }
    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    free(buffer);
}

void * gk_backend_buffer_get_base(gk_backend_buffer_t buffer) {
    if (buffer->size == 0) {
        return NULL;
    }
    return buffer->iface.get_base(buffer);
}

size_t gk_backend_buffer_get_size(gk_backend_buffer_t buffer) {
    return buffer->size;
}

gk_backend_buffer_type_t gk_backend_buffer_get_type(gk_backend_buffer_t buffer) {
    return buffer->buft;
}

size_t gk_backend_buffer_get_alignment(gk_backend_buffer_t buffer) {
    return gk_backend_buft_get_alignment(buffer->buft);
}

bool gk_backend_buffer_is_host(gk_backend_buffer_t buffer) {
    return gk_backend_buft_is_host(buffer->buft);
}

// Records which buffer a tensor lives in and gives the backend a chance to set
// up anything it needs per tensor. Views inherit their parent's buffer.
void gk_backend_buffer_init_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor) {
    tensor->buffer = buffer;

    if (buffer->iface.init_tensor != NULL) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}

void gk_backend_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    if (buffer->size == 0) {
        return;
    }
    if (buffer->iface.clear != NULL) {
        buffer->iface.clear(buffer, value);
    }
}

gk_device_t gk_backend_buft_get_device(gk_backend_buffer_type_t buft) {
    return buft->device;
}

// --------------------------------------------------------------------------
// tensor transfer
//
// These are the only sanctioned way to move data in and out of a tensor whose
// storage might not be host memory. Writing through `tensor->data` happens to
// work for the CPU backend and will not for any other, so the tests and the
// engine go through here.
// --------------------------------------------------------------------------

void gk_backend_tensor_set(struct gk_tensor * tensor, const void * data,
                           size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    if (tensor->data == NULL) { gk_logf("gk: tensor_set on unallocated tensor %s\n", tensor->name); }
    GK_ASSERT(tensor->data != NULL);
    GK_ASSERT(offset + size <= gk_nbytes(tensor));

    gk_backend_buffer_t buf = tensor->buffer;

    if (buf != NULL && buf->iface.set_tensor != NULL) {
        buf->iface.set_tensor(buf, tensor, data, offset, size);
        return;
    }

    // No buffer attached means a context-allocated tensor, which is host
    // memory by construction.
    memcpy((char *) tensor->data + offset, data, size);
}

void gk_backend_tensor_get(const struct gk_tensor * tensor, void * data,
                           size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    GK_ASSERT(tensor->data != NULL);
    GK_ASSERT(offset + size <= gk_nbytes(tensor));

    gk_backend_buffer_t buf = tensor->buffer;

    if (buf != NULL && buf->iface.get_tensor != NULL) {
        buf->iface.get_tensor(buf, tensor, data, offset, size);
        return;
    }

    memcpy(data, (const char *) tensor->data + offset, size);
}

void gk_backend_tensor_memset(struct gk_tensor * tensor, uint8_t value,
                              size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    GK_ASSERT(tensor->data != NULL);
    GK_ASSERT(offset + size <= gk_nbytes(tensor));

    gk_backend_buffer_t buf = tensor->buffer;

    if (buf != NULL && buf->iface.memset_tensor != NULL) {
        buf->iface.memset_tensor(buf, tensor, value, offset, size);
        return;
    }

    // A device buffer with no memset of its own still has set_tensor, so the
    // pattern goes out through that rather than through a pointer we may not
    // be allowed to dereference.
    if (buf != NULL && !gk_backend_buffer_is_host(buf) && buf->iface.set_tensor != NULL) {
        enum { CHUNK = 4096 };
        uint8_t pattern[CHUNK];
        memset(pattern, value, sizeof(pattern));

        for (size_t done = 0; done < size; ) {
            const size_t n = GK_MIN((size_t) CHUNK, size - done);
            buf->iface.set_tensor(buf, tensor, pattern, offset + done, n);
            done += n;
        }
        return;
    }

    memset((char *) tensor->data + offset, value, size);
}

// Three cases, cheapest first: the same memory needs nothing, two buffers a
// device can reach let that device do the copy, and anything else goes out to
// the host and back in. The last one is always available, which is what makes
// a pair of unrelated devices work at all.
void gk_backend_tensor_copy(const struct gk_tensor * src, struct gk_tensor * dst) {
    GK_ASSERT(gk_are_same_shape(src, dst));
    GK_ASSERT(src->type == dst->type);

    if (src == dst || src->data == dst->data) {
        return;
    }

    if (dst->buffer != NULL && dst->buffer->iface.cpy_tensor != NULL) {
        if (dst->buffer->iface.cpy_tensor(dst->buffer, src, dst)) {
            return;
        }
    }

    const size_t nbytes = gk_nbytes(src);

    // Either side being host memory means one of the two transfers is a plain
    // read or write and no staging buffer is needed.
    if (src->buffer == NULL || gk_backend_buffer_is_host(src->buffer)) {
        gk_backend_tensor_set(dst, (const char *) src->data, 0, nbytes);
        return;
    }
    if (dst->buffer == NULL || gk_backend_buffer_is_host(dst->buffer)) {
        gk_backend_tensor_get(src, (char *) dst->data, 0, nbytes);
        return;
    }

    void * staging = malloc(nbytes);
    if (staging == NULL) {
        gk_logf("gk: no memory to stage a %zu byte tensor copy\n", nbytes);
        return;
    }

    gk_backend_tensor_get(src, staging, 0, nbytes);
    gk_backend_tensor_set(dst, staging, 0, nbytes);

    free(staging);
}

// --------------------------------------------------------------------------
// backend
// --------------------------------------------------------------------------

const char * gk_backend_name(gk_backend_t backend) {
    return backend->iface.get_name(backend);
}

void gk_backend_free(gk_backend_t backend) {
    if (backend == NULL) {
        return;
    }
    backend->iface.free(backend);
}

gk_backend_buffer_type_t gk_backend_get_default_buffer_type(gk_backend_t backend) {
    return backend->iface.get_default_buffer_type(backend);
}

enum gk_status gk_backend_graph_compute(gk_backend_t backend, struct gk_cgraph * graph) {
    const enum gk_status st = backend->iface.graph_compute(backend, graph);
    if (st != GK_STATUS_SUCCESS) {
        return st;
    }

    // A device backend has only queued the work at this point. Waiting here is
    // what makes the results readable, and it is the contract every caller
    // above already assumes: they compute a graph and then read its output
    // without a synchronize of their own. On the CPU this is a no-op.
    gk_backend_synchronize(backend);
    return st;
}

enum gk_status gk_backend_graph_compute_async(gk_backend_t backend, struct gk_cgraph * graph) {
    return backend->iface.graph_compute(backend, graph);
}

bool gk_backend_supports_op(gk_backend_t backend, const struct gk_tensor * op) {
    return backend->iface.supports_op(backend, op);
}

bool gk_backend_supports_buft(gk_backend_t backend, gk_backend_buffer_type_t buft) {
    if (backend->iface.supports_buft != NULL) {
        return backend->iface.supports_buft(backend, buft);
    }
    // The default is the host's answer: a backend that never said otherwise
    // computes out of host memory and can only read host memory.
    return gk_backend_buft_is_host(buft);
}

bool gk_backend_offload_op(gk_backend_t backend, const struct gk_tensor * op) {
    if (backend->iface.offload_op != NULL) {
        return backend->iface.offload_op(backend, op);
    }
    return false;
}

void gk_backend_synchronize(gk_backend_t backend) {
    if (backend->iface.synchronize != NULL) {
        backend->iface.synchronize(backend);
    }
}

void gk_backend_tensor_set_async(gk_backend_t backend, struct gk_tensor * tensor,
                                 const void * data, size_t offset, size_t size) {
    if (backend->iface.set_tensor_async != NULL) {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
        return;
    }
    gk_backend_tensor_set(tensor, data, offset, size);
}

void gk_backend_tensor_get_async(gk_backend_t backend, const struct gk_tensor * tensor,
                                 void * data, size_t offset, size_t size) {
    if (backend->iface.get_tensor_async != NULL) {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
        return;
    }
    gk_backend_tensor_get(tensor, data, offset, size);
}

gk_device_t gk_backend_get_device(gk_backend_t backend) {
    return backend->device;
}

// ==========================================================================
// the CPU backend
// ==========================================================================

struct gk_cpu_buffer_ctx {
    void * base;   // what to hand out
    void * owned;  // what to free, or NULL when the memory came from outside
};

static void gk_cpu_buffer_free(gk_backend_buffer_t buffer) {
    struct gk_cpu_buffer_ctx * ctx = (struct gk_cpu_buffer_ctx *) buffer->context;
    if (ctx == NULL) {
        return;
    }
    free(ctx->owned);
    free(ctx);
}

static void * gk_cpu_buffer_get_base(gk_backend_buffer_t buffer) {
    struct gk_cpu_buffer_ctx * ctx = (struct gk_cpu_buffer_ctx *) buffer->context;
    return ctx->base;
}

static void gk_cpu_buffer_set_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                     const void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void gk_cpu_buffer_get_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                     void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void gk_cpu_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    struct gk_cpu_buffer_ctx * ctx = (struct gk_cpu_buffer_ctx *) buffer->context;
    memset(ctx->base, value, buffer->size);
}

static void gk_cpu_buffer_memset_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                        uint8_t value, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

// Host memory can be read from wherever the source is, so this accepts any
// source whose own buffer can hand its bytes over.
static bool gk_cpu_buffer_cpy_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * src,
                                     struct gk_tensor * dst) {
    GK_UNUSED(buffer);

    if (src->buffer != NULL && !gk_backend_buffer_is_host(src->buffer)) {
        return false; // let the caller stage it; only the owner knows how
    }

    memcpy(dst->data, src->data, gk_nbytes(src));
    return true;
}

static const struct gk_backend_buffer_i g_cpu_buffer_iface = {
    /* .free_buffer   = */ gk_cpu_buffer_free,
    /* .get_base      = */ gk_cpu_buffer_get_base,
    /* .init_tensor   = */ NULL, // host memory needs no per-tensor setup
    /* .set_tensor    = */ gk_cpu_buffer_set_tensor,
    /* .get_tensor    = */ gk_cpu_buffer_get_tensor,
    /* .clear         = */ gk_cpu_buffer_clear,
    /* .memset_tensor = */ gk_cpu_buffer_memset_tensor,
    /* .cpy_tensor    = */ gk_cpu_buffer_cpy_tensor,
};

// --------------------------------------------------------------------------

static const char * gk_cpu_buft_name(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return "CPU";
}

// Rounds a pointer up to the alignment the kernels want. Allocating the extra
// by hand rather than with aligned_alloc keeps this portable to MSVC, which
// has no aligned_alloc, without a second code path.
static gk_backend_buffer_t gk_cpu_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    struct gk_cpu_buffer_ctx * ctx =
        (struct gk_cpu_buffer_ctx *) malloc(sizeof(struct gk_cpu_buffer_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->owned = malloc(size + GK_MEM_ALIGN);
    if (ctx->owned == NULL) {
        free(ctx);
        gk_logf("gk: failed to allocate %zu bytes of CPU buffer\n", size);
        return NULL;
    }

    const uintptr_t raw = (uintptr_t) ctx->owned;
    ctx->base = (void *) ((raw + GK_MEM_ALIGN - 1) & ~(uintptr_t) (GK_MEM_ALIGN - 1));

    gk_backend_buffer_t buffer =
        gk_backend_buffer_init(buft, &g_cpu_buffer_iface, ctx, size);
    if (buffer == NULL) {
        free(ctx->owned);
        free(ctx);
        return NULL;
    }

    return buffer;
}

static size_t gk_cpu_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_MEM_ALIGN;
}

static bool gk_cpu_buft_is_host(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return true;
}

// The device pointer is filled in when the registry builds the CPU device;
// until then it is NULL, which every caller already reads as "the host".
static struct gk_backend_buffer_type g_cpu_buft = {
    // Keep this nested initializer literal.  MSVC rejects copying the
    // separate static vtable object here as a non-constant initializer.
    {
        /* .get_name       = */ gk_cpu_buft_name,
        /* .alloc_buffer   = */ gk_cpu_buft_alloc,
        /* .get_alignment  = */ gk_cpu_buft_alignment,
        /* .get_alloc_size = */ NULL, // host storage costs exactly gk_nbytes
        /* .is_host        = */ gk_cpu_buft_is_host,
    },
    /* .context = */ NULL,
    /* .device  = */ NULL,
};

gk_backend_buffer_type_t gk_backend_cpu_buffer_type(void) {
    return &g_cpu_buft;
}

// Wraps memory the caller already owns - a memory-mapped model file, most
// usefully, which is how weights get read without being copied.
gk_backend_buffer_t gk_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    struct gk_cpu_buffer_ctx * ctx =
        (struct gk_cpu_buffer_ctx *) malloc(sizeof(struct gk_cpu_buffer_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->base  = ptr;
    ctx->owned = NULL; // borrowed; freeing the buffer must not free this

    return gk_backend_buffer_init(&g_cpu_buft, &g_cpu_buffer_iface, ctx, size);
}

// --------------------------------------------------------------------------

struct gk_cpu_backend_ctx {
    struct gk_pool * pool;
    int              n_threads;
};

static const char * gk_cpu_backend_name(gk_backend_t backend) {
    GK_UNUSED(backend);
    return "CPU";
}

static void gk_cpu_backend_free(gk_backend_t backend) {
    struct gk_cpu_backend_ctx * ctx = (struct gk_cpu_backend_ctx *) backend->context;
    if (ctx != NULL) {
        gk_pool_free(ctx->pool);
        free(ctx);
    }
    free(backend);
}

static gk_backend_buffer_type_t gk_cpu_backend_buft(gk_backend_t backend) {
    GK_UNUSED(backend);
    return &g_cpu_buft;
}

// The pool lives on the backend, so a run of graphs shares one set of threads
// instead of creating and tearing them down per graph.
static enum gk_status gk_cpu_backend_compute(gk_backend_t backend, struct gk_cgraph * graph) {
    struct gk_cpu_backend_ctx * ctx = (struct gk_cpu_backend_ctx *) backend->context;
    return gk_graph_compute_with_pool(graph, ctx->pool);
}

static bool gk_cpu_backend_supports_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);
    return gk_cpu_has_kernel(op->op);
}

// The CPU can compute out of any memory the host can address, which includes
// a device's pinned host buffers as well as its own.
static bool gk_cpu_backend_supports_buft(gk_backend_t backend, gk_backend_buffer_type_t buft) {
    GK_UNUSED(backend);
    return gk_backend_buft_is_host(buft);
}

static const struct gk_backend_i g_cpu_backend_iface = {
    /* .get_name                = */ gk_cpu_backend_name,
    /* .free                    = */ gk_cpu_backend_free,
    /* .get_default_buffer_type = */ gk_cpu_backend_buft,
    /* .graph_compute           = */ gk_cpu_backend_compute,
    /* .supports_op             = */ gk_cpu_backend_supports_op,
    /* .supports_buft           = */ gk_cpu_backend_supports_buft,
    /* .offload_op              = */ NULL, // there is nowhere further to offload to
    /* .synchronize             = */ NULL, // the compute pass returns finished
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
};

gk_backend_t gk_backend_cpu_init(int n_threads) {
    struct gk_cpu_backend_ctx * ctx =
        (struct gk_cpu_backend_ctx *) malloc(sizeof(struct gk_cpu_backend_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->pool = gk_pool_create(n_threads);
    if (ctx->pool == NULL) {
        free(ctx);
        return NULL;
    }
    ctx->n_threads = gk_pool_n_threads(ctx->pool);

    gk_backend_t backend = (gk_backend_t) malloc(sizeof(struct gk_backend));
    if (backend == NULL) {
        gk_pool_free(ctx->pool);
        free(ctx);
        return NULL;
    }

    backend->iface   = g_cpu_backend_iface;
    backend->context = ctx;
    // Asking the registry rather than reading the buffer type's field: the
    // field is filled in by registration, and this may be the call that
    // happens first. Discovery creates no backends, so there is no loop.
    backend->device  = gk_device_by_type(GK_DEVICE_TYPE_CPU);

    return backend;
}

int gk_backend_cpu_n_threads(gk_backend_t backend) {
    struct gk_cpu_backend_ctx * ctx = (struct gk_cpu_backend_ctx *) backend->context;
    return ctx->n_threads;
}

// Swaps the pool for one of a different width. The old threads are joined
// first, so no graph can still be running on them.
void gk_backend_cpu_set_n_threads(gk_backend_t backend, int n_threads) {
    struct gk_cpu_backend_ctx * ctx = (struct gk_cpu_backend_ctx *) backend->context;

    struct gk_pool * pool = gk_pool_create(n_threads);
    if (pool == NULL) {
        return; // keep the pool we have rather than none
    }

    gk_pool_free(ctx->pool);
    ctx->pool      = pool;
    ctx->n_threads = gk_pool_n_threads(pool);
}
