// The Metal backend: devices, memory, and running a graph on an Apple GPU.
//
// Two things make this different from the CUDA backend next door, and both
// come from the same fact - on every Mac gk runs on, the GPU and the CPU share
// physical memory.
//
//   * A buffer is allocated with shared storage, so its contents have a host
//     address. `tensor->data` is therefore a real pointer and getting a tensor
//     out is a memcpy rather than a transfer. The buffer type still reports
//     itself as device memory: the scheduler uses that answer to decide where
//     a node runs, and "the GPU's memory" is the right answer for placement
//     even when the pages happen to be reachable from both sides.
//
//   * Kernels are bound buffers, not pointers, so a tensor's address has to be
//     resolved back to the buffer that contains it. Every live buffer is
//     registered with its device for exactly that lookup.
//
// The kernel source is embedded and compiled once per backend, at creation.
// Compiling at load time rather than shipping a .metallib is what keeps the
// library from depending on a file it cannot guarantee is installed.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gk_impl.h"

#include <math.h>

#include "gk_metal_shaders.h" // generated: gk_metal_shader_source

#include <stdlib.h>
#include <string.h>

#define GK_METAL_MAX_DEVICES 8
#define GK_METAL_MAX_BUFFERS 256

// Metal's own minimum for a bound buffer offset on discrete parts, and a
// harmless over-alignment on the unified ones.
#define GK_METAL_ALIGN 256

// Threads per group for the row-wise kernels. 256 is a whole number of SIMD
// groups on every Apple GPU, which the two-stage reductions assume.
#define GK_METAL_ROW_TG 256

// --------------------------------------------------------------------------
// the kernel parameter block, mirrored from gk_metal.metal
// --------------------------------------------------------------------------

struct gk_mtl_tview {
    int64_t ne[4];
    int64_t nb[4];
    int32_t type;
    int32_t pad;
};

struct gk_mtl_params {
    struct gk_mtl_tview src0;
    struct gk_mtl_tview src1;
    struct gk_mtl_tview src2;
    struct gk_mtl_tview dst;

    float   f[8];
    int32_t i[16];

    int64_t n;
    int32_t flags;
    int32_t pad2;
};

// Which pipelines exist. The order is the order they are looked up in, and the
// names are the kernel names in the shader source.
enum gk_metal_kernel {
    GK_METAL_K_BINARY = 0,
    GK_METAL_K_UNARY,
    GK_METAL_K_SIMPLE,
    GK_METAL_K_AFFINE,
    GK_METAL_K_GLU,
    GK_METAL_K_COPY,
    GK_METAL_K_GET_ROWS,
    GK_METAL_K_SET_ROWS,
    GK_METAL_K_REPEAT,
    GK_METAL_K_CONCAT,
    GK_METAL_K_ADD_ID,
    GK_METAL_K_DIAG_MASK,
    GK_METAL_K_PAD,
    GK_METAL_K_IM2COL,
    GK_METAL_K_NORM,
    GK_METAL_K_GROUP_NORM,
    GK_METAL_K_SOFT_MAX,
    GK_METAL_K_SUM_ROWS,
    GK_METAL_K_ROPE,
    GK_METAL_K_ROPE_PASSTHROUGH,
    GK_METAL_K_MUL_MAT,
    GK_METAL_K_MUL_MAT_TILED,
    GK_METAL_K_MUL_MAT_ID,
    GK_METAL_K_COUNT,
};

static const char * g_metal_kernel_names[GK_METAL_K_COUNT] = {
    "gk_mtl_binary",
    "gk_mtl_unary_op",
    "gk_mtl_simple",
    "gk_mtl_affine",
    "gk_mtl_glu",
    "gk_mtl_copy",
    "gk_mtl_get_rows",
    "gk_mtl_set_rows",
    "gk_mtl_repeat",
    "gk_mtl_concat",
    "gk_mtl_add_id",
    "gk_mtl_diag_mask",
    "gk_mtl_pad",
    "gk_mtl_im2col",
    "gk_mtl_norm",
    "gk_mtl_group_norm",
    "gk_mtl_soft_max",
    "gk_mtl_sum_rows",
    "gk_mtl_rope",
    "gk_mtl_rope_passthrough",
    "gk_mtl_mul_mat",
    "gk_mtl_mul_mat_tiled",
    "gk_mtl_mul_mat_id",
};

// --------------------------------------------------------------------------
// per-device state
// --------------------------------------------------------------------------

struct gk_metal_buffer_ctx;

struct gk_metal_device_ctx {
    int index;
    char name[32];
    char description[256];

    id<MTLDevice> mtl;

    size_t total_memory;

    // Every live buffer on this device, so a tensor's address can be resolved
    // back to the buffer holding it. Nothing else needs the list, and it is
    // short - a model is a handful of large buffers, not many small ones.
    struct gk_metal_buffer_ctx * buffers[GK_METAL_MAX_BUFFERS];
    int                          n_buffers;

    struct gk_backend_buffer_type buft;
    struct gk_device              device;
};

static struct gk_metal_device_ctx g_metal_devices[GK_METAL_MAX_DEVICES];
static int  g_metal_n_devices;
static bool g_metal_discovered;

struct gk_metal_buffer_ctx {
    struct gk_metal_device_ctx * dev;
    id<MTLBuffer>                mtl;
    void *                       base;
    size_t                       size;
};

// --------------------------------------------------------------------------
// buffers
// --------------------------------------------------------------------------

static void gk_metal_device_add_buffer(struct gk_metal_device_ctx * dev,
                                       struct gk_metal_buffer_ctx * buf) {
    if (dev->n_buffers == GK_METAL_MAX_BUFFERS) {
        gk_logf("gk Metal: more than %d live buffers on %s; tensor lookup will fail\n",
                GK_METAL_MAX_BUFFERS, dev->name);
        return;
    }
    dev->buffers[dev->n_buffers++] = buf;
}

static void gk_metal_device_drop_buffer(struct gk_metal_device_ctx * dev,
                                        struct gk_metal_buffer_ctx * buf) {
    for (int i = 0; i < dev->n_buffers; ++i) {
        if (dev->buffers[i] == buf) {
            dev->buffers[i] = dev->buffers[--dev->n_buffers];
            return;
        }
    }
}

// The buffer a host address falls inside, and how far into it. This is the
// lookup the whole binding scheme rests on: Metal binds buffers, gk hands out
// pointers, and this is where the two meet.
static id<MTLBuffer> gk_metal_resolve(struct gk_metal_device_ctx * dev,
                                      const void * addr, size_t * offset) {
    for (int i = 0; i < dev->n_buffers; ++i) {
        struct gk_metal_buffer_ctx * b = dev->buffers[i];
        const char * base = (const char *) b->base;

        if ((const char *) addr >= base && (const char *) addr < base + b->size) {
            *offset = (size_t) ((const char *) addr - base);
            return b->mtl;
        }
    }

    *offset = 0;
    return nil;
}

static void gk_metal_buffer_free(gk_backend_buffer_t buffer) {
    struct gk_metal_buffer_ctx * ctx = (struct gk_metal_buffer_ctx *) buffer->context;
    if (ctx == NULL) {
        return;
    }

    gk_metal_device_drop_buffer(ctx->dev, ctx);
    ctx->mtl = nil; // ARC releases the buffer with the last reference
    free(ctx);
}

static void * gk_metal_buffer_get_base(gk_backend_buffer_t buffer) {
    return ((struct gk_metal_buffer_ctx *) buffer->context)->base;
}

static void gk_metal_buffer_set_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                       const void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    // Shared storage: the GPU sees these bytes without a transfer.
    memcpy((char *) tensor->data + offset, data, size);
}

static void gk_metal_buffer_get_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                       void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void gk_metal_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    struct gk_metal_buffer_ctx * ctx = (struct gk_metal_buffer_ctx *) buffer->context;
    memset(ctx->base, value, ctx->size);
}

static void gk_metal_buffer_memset_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                          uint8_t value, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static const struct gk_backend_buffer_i g_metal_buffer_iface = {
    /* .free_buffer   = */ gk_metal_buffer_free,
    /* .get_base      = */ gk_metal_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .set_tensor    = */ gk_metal_buffer_set_tensor,
    /* .get_tensor    = */ gk_metal_buffer_get_tensor,
    /* .clear         = */ gk_metal_buffer_clear,
    /* .memset_tensor = */ gk_metal_buffer_memset_tensor,
    /* .cpy_tensor    = */ NULL, // every path here is a host-addressable copy
};

static const char * gk_metal_buft_name(gk_backend_buffer_type_t buft) {
    return ((struct gk_metal_device_ctx *) buft->context)->name;
}

static gk_backend_buffer_t gk_metal_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    struct gk_metal_device_ctx * dev = (struct gk_metal_device_ctx *) buft->context;

    struct gk_metal_buffer_ctx * ctx =
        (struct gk_metal_buffer_ctx *) malloc(sizeof(struct gk_metal_buffer_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->dev  = dev;
    ctx->size = size;
    ctx->mtl  = [dev->mtl newBufferWithLength:size options:MTLResourceStorageModeShared];

    if (ctx->mtl == nil) {
        gk_logf("gk Metal: failed to allocate %zu bytes on %s\n", size, dev->name);
        free(ctx);
        return NULL;
    }

    ctx->base = [ctx->mtl contents];

    gk_backend_buffer_t buffer = gk_backend_buffer_init(buft, &g_metal_buffer_iface, ctx, size);
    if (buffer == NULL) {
        ctx->mtl = nil;
        free(ctx);
        return NULL;
    }

    gk_metal_device_add_buffer(dev, ctx);

    return buffer;
}

static size_t gk_metal_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_METAL_ALIGN;
}

static const struct gk_backend_buffer_type_i g_metal_buft_iface = {
    /* .get_name       = */ gk_metal_buft_name,
    /* .alloc_buffer   = */ gk_metal_buft_alloc,
    /* .get_alignment  = */ gk_metal_buft_alignment,
    /* .get_alloc_size = */ NULL,
    /* .is_host        = */ NULL, // reported as device memory, deliberately:
                                  // see the note at the top of this file
};

// --------------------------------------------------------------------------
// the backend
// --------------------------------------------------------------------------

struct gk_metal_backend_ctx {
    struct gk_metal_device_ctx * dev;

    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> pipelines[GK_METAL_K_COUNT];

    // The command buffer the current graph is being encoded into, kept so
    // `synchronize` has something to wait on.
    id<MTLCommandBuffer> pending;
};

static const char * gk_metal_backend_name(gk_backend_t backend) {
    return ((struct gk_metal_backend_ctx *) backend->context)->dev->name;
}

static void gk_metal_backend_free(gk_backend_t backend) {
    struct gk_metal_backend_ctx * ctx = (struct gk_metal_backend_ctx *) backend->context;
    if (ctx != NULL) {
        if (ctx->pending != nil) {
            [ctx->pending waitUntilCompleted];
        }
        for (int i = 0; i < GK_METAL_K_COUNT; ++i) {
            ctx->pipelines[i] = nil;
        }
        ctx->library = nil;
        ctx->queue   = nil;
        ctx->pending = nil;
        free(ctx);
    }
    free(backend);
}

static gk_backend_buffer_type_t gk_metal_backend_buft(gk_backend_t backend) {
    return &((struct gk_metal_backend_ctx *) backend->context)->dev->buft;
}

// --------------------------------------------------------------------------
// building the parameter block
// --------------------------------------------------------------------------

static struct gk_mtl_tview gk_metal_view(const struct gk_tensor * t) {
    struct gk_mtl_tview v;
    memset(&v, 0, sizeof(v));

    if (t == NULL) {
        return v;
    }

    for (int i = 0; i < 4; ++i) {
        v.ne[i] = t->ne[i];
        v.nb[i] = (int64_t) t->nb[i];
    }
    v.type = (int32_t) t->type;

    return v;
}

static struct gk_mtl_params gk_metal_params(const struct gk_tensor * node) {
    struct gk_mtl_params p;
    memset(&p, 0, sizeof(p));

    p.src0 = gk_metal_view(node->src[0]);
    p.src1 = gk_metal_view(node->src[1]);
    p.src2 = gk_metal_view(node->src[2]);
    p.dst  = gk_metal_view(node);

    p.n = node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3];

    return p;
}

// Binds a tensor's storage at `index`, or nothing if the tensor is absent.
static void gk_metal_bind(id<MTLComputeCommandEncoder> enc,
                          struct gk_metal_device_ctx * dev,
                          const struct gk_tensor * t, int index) {
    if (t == NULL || t->data == NULL) {
        return;
    }

    size_t offset = 0;
    id<MTLBuffer> buf = gk_metal_resolve(dev, t->data, &offset);

    if (buf == nil) {
        gk_logf("gk Metal: %s is not in any buffer on %s\n", t->name, dev->name);
        return;
    }

    [enc setBuffer:buf offset:offset atIndex:index];
}

static void gk_metal_dispatch_flat(id<MTLComputeCommandEncoder> enc,
                                   id<MTLComputePipelineState> pipe, int64_t n) {
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256) {
        tg = 256;
    }
    if ((NSUInteger) n < tg) {
        tg = (NSUInteger) (n > 0 ? n : 1);
    }

    [enc dispatchThreads:MTLSizeMake((NSUInteger) n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

static void gk_metal_dispatch_rows(id<MTLComputeCommandEncoder> enc, int64_t n_rows) {
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) n_rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(GK_METAL_ROW_TG, 1, 1)];
}

// --------------------------------------------------------------------------
// op dispatch
// --------------------------------------------------------------------------

// The float types the generic load/store paths handle.
static bool gk_metal_is_float(enum gk_type t) {
    return t == GK_TYPE_F32 || t == GK_TYPE_F16 || t == GK_TYPE_BF16;
}

// The quantized formats gk_metal.metal decodes. The lattice families are
// absent on purpose - they need their codebooks resident - so a weight in one
// of them makes its matmul unsupported and it runs on the CPU.
static bool gk_metal_type_supported(enum gk_type t) {
    switch (t) {
        case GK_TYPE_F32: case GK_TYPE_F16: case GK_TYPE_BF16:
        case GK_TYPE_Q4_0: case GK_TYPE_Q4_1: case GK_TYPE_Q5_0: case GK_TYPE_Q5_1:
        case GK_TYPE_Q8_0: case GK_TYPE_Q2_K: case GK_TYPE_Q3_K: case GK_TYPE_Q4_K:
        case GK_TYPE_Q5_K: case GK_TYPE_Q6_K: case GK_TYPE_IQ4_NL: case GK_TYPE_IQ4_XS:
        case GK_TYPE_MXFP4: case GK_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

static bool gk_metal_readable(const struct gk_tensor * t) {
    if (t == NULL) {
        return true;
    }
    if (gk_metal_is_float(t->type) || t->type == GK_TYPE_I32 || t->type == GK_TYPE_I64) {
        return true;
    }
    return gk_metal_type_supported(t->type) && t->nb[0] == gk_type_size(t->type);
}

static bool gk_metal_supports_op(const struct gk_tensor * op) {
    switch (op->op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return true;

        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            return op->type == GK_TYPE_F32 &&
                   gk_metal_type_supported(op->src[0]->type) && gk_metal_readable(op->src[0]) &&
                   gk_metal_is_float(op->src[1]->type);

        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV:
        case GK_OP_ADD_ID:
        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS:
        case GK_OP_UNARY: case GK_OP_GLU: case GK_OP_LEAKY_RELU:
        case GK_OP_SCALE: case GK_OP_CLAMP: case GK_OP_FILL:
        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM: case GK_OP_GROUP_NORM:
        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
        case GK_OP_GET_ROWS: case GK_OP_REPEAT: case GK_OP_CONCAT:
        case GK_OP_SOFT_MAX: case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
        case GK_OP_ROPE: case GK_OP_SUM_ROWS: case GK_OP_MEAN: case GK_OP_PAD:
            break;

        case GK_OP_SET_ROWS:
            return gk_metal_is_float(op->type) && gk_metal_is_float(op->src[0]->type);

        case GK_OP_IM2COL:
            // mirrors the CPU pass's asserts: f32 image in, f32 or f16 out
            return (op->type == GK_TYPE_F32 || op->type == GK_TYPE_F16) &&
                   op->src[1]->type == GK_TYPE_F32;

        default:
            return false;
    }

    if (!gk_metal_is_float(op->type)) {
        return false;
    }

    for (int i = 0; i < GK_MAX_SRC; ++i) {
        if (!gk_metal_readable(op->src[i])) {
            return false;
        }
    }

    return true;
}

// Encodes one node. Returns false for an op with no kernel, which the
// scheduler's supports_op should already have kept away from here.
static bool gk_metal_encode(struct gk_metal_backend_ctx * ctx,
                            id<MTLComputeCommandEncoder> enc,
                            struct gk_tensor * node) {
    struct gk_metal_device_ctx * dev = ctx->dev;

    struct gk_tensor * src0 = node->src[0];
    struct gk_tensor * src1 = node->src[1];
    struct gk_tensor * src2 = node->src[2];

    struct gk_mtl_params p = gk_metal_params(node);

    enum gk_metal_kernel which;
    bool row_wise = false;
    int64_t n_rows = node->ne[1] * node->ne[2] * node->ne[3];

    switch (node->op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return true;

        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV:
            which = GK_METAL_K_BINARY;
            p.i[0] = node->op == GK_OP_ADD ? 0 : node->op == GK_OP_SUB ? 1 :
                     node->op == GK_OP_MUL ? 2 : 3;
            break;

        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS:
            which = GK_METAL_K_SIMPLE;
            p.i[0] = node->op == GK_OP_SQR ? 0 : node->op == GK_OP_SQRT ? 1 :
                     node->op == GK_OP_LOG ? 2 : node->op == GK_OP_SIN ? 3 : 4;
            break;

        case GK_OP_UNARY:
            which = GK_METAL_K_UNARY;
            p.i[0] = (int) gk_get_unary_op(node);
            for (int i = 0; i < 4; ++i) {
                p.f[i] = gk_get_op_params_f32(node, i + 1);
            }
            break;

        case GK_OP_SCALE:
            which  = GK_METAL_K_AFFINE;
            p.i[0] = 0;
            p.f[0] = gk_get_op_params_f32(node, 0);
            p.f[1] = gk_get_op_params_f32(node, 1);
            break;

        case GK_OP_CLAMP:
            which  = GK_METAL_K_AFFINE;
            p.i[0] = 1;
            p.f[0] = gk_get_op_params_f32(node, 0);
            p.f[1] = gk_get_op_params_f32(node, 1);
            break;

        case GK_OP_FILL:
            which  = GK_METAL_K_AFFINE;
            p.i[0] = 2;
            p.f[0] = gk_get_op_params_f32(node, 0);
            break;

        case GK_OP_LEAKY_RELU:
            which  = GK_METAL_K_AFFINE;
            p.i[0] = 3;
            p.f[0] = gk_get_op_params_f32(node, 0);
            break;

        case GK_OP_GLU:
            which  = GK_METAL_K_GLU;
            p.i[0] = (int) gk_get_glu_op(node);
            p.i[1] = gk_get_op_params_i32(node, 1);
            p.f[0] = gk_get_op_params_f32(node, 2);
            p.f[1] = gk_get_op_params_f32(node, 3);
            p.flags = src1 != NULL ? 1 : 0;
            break;

        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
            which   = GK_METAL_K_COPY;
            p.flags = gk_are_same_shape(src0, node) ? 1 : 0;
            break;

        case GK_OP_GET_ROWS:
            which = GK_METAL_K_GET_ROWS;
            break;

        case GK_OP_SET_ROWS:
            which   = GK_METAL_K_SET_ROWS;
            p.flags = src1->type == GK_TYPE_I64 ? 1 : 0;
            p.n     = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
            break;

        case GK_OP_REPEAT:
            which = GK_METAL_K_REPEAT;
            break;

        case GK_OP_CONCAT:
            which  = GK_METAL_K_CONCAT;
            p.i[0] = gk_get_op_params_i32(node, 0);
            break;

        case GK_OP_ADD_ID:
            which = GK_METAL_K_ADD_ID;
            break;

        case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
            which  = GK_METAL_K_DIAG_MASK;
            p.i[0] = gk_get_op_params_i32(node, 0);
            p.f[0] = node->op == GK_OP_DIAG_MASK_INF ? -INFINITY : 0.0f;
            break;

        case GK_OP_PAD:
            which = GK_METAL_K_PAD;
            for (int i = 0; i < 8; ++i) {
                p.i[i] = gk_get_op_params_i32(node, i);
            }
            p.flags = gk_get_op_params_i32(node, 8) != 0 ? 1 : 0;
            break;

        case GK_OP_IM2COL:
            which = GK_METAL_K_IM2COL;
            for (int i = 0; i < 7; ++i) { // s0, s1, p0, p1, d0, d1, is_2D
                p.i[i] = gk_get_op_params_i32(node, i);
            }
            break;

        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM:
            which    = GK_METAL_K_NORM;
            row_wise = true;
            p.i[0]   = node->op == GK_OP_RMS_NORM ? 0 : node->op == GK_OP_NORM ? 1 : 2;
            p.f[0]   = gk_get_op_params_f32(node, 0);
            break;

        case GK_OP_GROUP_NORM:
            which    = GK_METAL_K_GROUP_NORM;
            row_wise = true;
            p.i[0]   = gk_get_op_params_i32(node, 0);
            p.f[0]   = gk_get_op_params_f32(node, 1);
            n_rows   = node->ne[3] * p.i[0];
            break;

        case GK_OP_SOFT_MAX: {
            which    = GK_METAL_K_SOFT_MAX;
            row_wise = true;
            p.f[0]   = gk_get_op_params_f32(node, 0);
            p.f[1]   = gk_get_op_params_f32(node, 1);

            int64_t n_head_log2 = 1;
            while (n_head_log2 * 2 <= src0->ne[2]) {
                n_head_log2 *= 2;
            }
            p.i[0]  = (int) n_head_log2;
            p.flags = (src1 != NULL ? 1 : 0) | (src2 != NULL ? 2 : 0);
            break;
        }

        case GK_OP_SUM_ROWS: case GK_OP_MEAN:
            which    = GK_METAL_K_SUM_ROWS;
            row_wise = true;
            p.i[0]   = node->op == GK_OP_MEAN ? 1 : 0;
            break;

        case GK_OP_ROPE: {
            which = GK_METAL_K_ROPE;

            const int n_dims = gk_get_op_params_i32(node, 1);
            const int mode   = gk_get_op_params_i32(node, 2);

            p.i[0] = n_dims;
            p.i[1] = mode;
            for (int i = 0; i < 4; ++i) {
                p.i[2 + i] = gk_get_op_params_i32(node, 11 + i);
            }

            const float freq_base = gk_get_op_params_f32(node, 5);

            p.f[0] = gk_get_op_params_f32(node, 6); // freq_scale
            p.f[1] = gk_get_op_params_f32(node, 7); // ext_factor
            p.f[2] = gk_get_op_params_f32(node, 8); // attn_factor
            p.f[3] = powf(freq_base, -2.0f / (float) n_dims);

            float corr[2];
            gk_rope_corr_dims(n_dims, gk_get_op_params_i32(node, 4), freq_base,
                              gk_get_op_params_f32(node, 9), gk_get_op_params_f32(node, 10),
                              corr);
            p.f[4] = corr[0];
            p.f[5] = corr[1];

            p.flags = src2 != NULL ? 1 : 0;

            const bool vision = mode == GK_ROPE_TYPE_VISION;
            const int64_t n_rot = vision ? node->ne[0] : n_dims;
            p.n = n_rows * (n_rot / 2);
            break;
        }

        case GK_OP_MUL_MAT:
            which    = GK_METAL_K_MUL_MAT;
            row_wise = true; // dispatched as threadgroups, geometry below
            break;

        case GK_OP_MUL_MAT_ID:
            which    = GK_METAL_K_MUL_MAT_ID;
            row_wise = true;
            break;

        default:
            return false;
    }

    id<MTLComputePipelineState> pipe = ctx->pipelines[which];
    [enc setComputePipelineState:pipe];

    gk_metal_bind(enc, dev, src0, 0);
    gk_metal_bind(enc, dev, src1, 1);
    gk_metal_bind(enc, dev, node, 2);

    // buffer 4 is whatever third operand the op has: the expert ids, the
    // attention sinks, the rope frequency factors
    gk_metal_bind(enc, dev, src2, 4);

    if (node->op == GK_OP_MUL_MAT) {
        // Wide destinations take the tiled kernel; the row geometry below is
        // only worth it for matvec-shaped outputs, where a 32-wide tile would
        // sit mostly empty.
        if (node->ne[1] >= 16) {
            id<MTLComputePipelineState> tiled = ctx->pipelines[GK_METAL_K_MUL_MAT_TILED];
            [enc setComputePipelineState:tiled];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setThreadgroupMemoryLength:2 * 32 * 16 * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) ((node->ne[0] + 31) / 32),
                                                  (NSUInteger) ((node->ne[1] + 31) / 32),
                                                  (NSUInteger) (node->ne[2] * node->ne[3]))
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return true;
        }

        p.i[0] = node->ne[1] < 4 ? 1 : 4;

        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setThreadgroupMemoryLength:GK_METAL_ROW_TG / 32 * sizeof(float) atIndex:0];

        const int64_t n_col_groups = (node->ne[1] + p.i[0] - 1) / p.i[0];

        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) node->ne[0],
                                              (NSUInteger) n_col_groups,
                                              (NSUInteger) (node->ne[2] * node->ne[3]))
            threadsPerThreadgroup:MTLSizeMake(GK_METAL_ROW_TG, 1, 1)];
        return true;
    }

    if (node->op == GK_OP_MUL_MAT_ID) {
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setThreadgroupMemoryLength:GK_METAL_ROW_TG / 32 * sizeof(float) atIndex:0];

        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) node->ne[0],
                                              (NSUInteger) node->ne[1],
                                              (NSUInteger) node->ne[2])
            threadsPerThreadgroup:MTLSizeMake(GK_METAL_ROW_TG, 1, 1)];
        return true;
    }

    [enc setBytes:&p length:sizeof(p) atIndex:3];

    if (row_wise) {
        [enc setThreadgroupMemoryLength:GK_METAL_ROW_TG / 32 * sizeof(float) atIndex:0];
        gk_metal_dispatch_rows(enc, n_rows);
    } else {
        gk_metal_dispatch_flat(enc, pipe, p.n);
    }

    // Rope leaves the channels past n_dims untouched, so they are copied over
    // in a second pass rather than branched on in the rotation kernel.
    if (node->op == GK_OP_ROPE) {
        const int n_dims = gk_get_op_params_i32(node, 1);
        const int mode   = gk_get_op_params_i32(node, 2);

        if (mode != GK_ROPE_TYPE_VISION && node->ne[0] > n_dims) {
            struct gk_mtl_params q = p;
            q.i[0] = n_dims;
            q.n    = n_rows * (node->ne[0] - n_dims);

            [enc setComputePipelineState:ctx->pipelines[GK_METAL_K_ROPE_PASSTHROUGH]];
            gk_metal_bind(enc, dev, src0, 0);
            gk_metal_bind(enc, dev, node, 2);
            [enc setBytes:&q length:sizeof(q) atIndex:3];
            gk_metal_dispatch_flat(enc, ctx->pipelines[GK_METAL_K_ROPE_PASSTHROUGH], q.n);
        }
    }

    return true;
}

static enum gk_status gk_metal_backend_compute(gk_backend_t backend, struct gk_cgraph * graph) {
    struct gk_metal_backend_ctx * ctx = (struct gk_metal_backend_ctx *) backend->context;

    @autoreleasepool {
        // The previous graph has to have finished before this one overwrites
        // its intermediates - the scheduler reuses the same memory.
        if (ctx->pending != nil) {
            [ctx->pending waitUntilCompleted];
            ctx->pending = nil;
        }

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        const int n = gk_graph_n_nodes(graph);

        for (int i = 0; i < n; ++i) {
            struct gk_tensor * node = gk_graph_node(graph, i);

            if (!gk_metal_encode(ctx, enc, node)) {
                [enc endEncoding];
                gk_logf("gk Metal: no kernel for op %s (node %s)\n",
                        gk_op_name(node->op), node->name);
                return GK_STATUS_NO_STORAGE;
            }
        }

        [enc endEncoding];
        [cmd commit];

        ctx->pending = cmd;
    }

    return GK_STATUS_SUCCESS;
}

static bool gk_metal_backend_supports_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);
    return gk_metal_supports_op(op);
}

static bool gk_metal_backend_supports_buft(gk_backend_t backend, gk_backend_buffer_type_t buft) {
    struct gk_metal_backend_ctx * ctx = (struct gk_metal_backend_ctx *) backend->context;
    // Its own memory only. Host memory allocated elsewhere is addressable but
    // not in an MTLBuffer, and a kernel can only be handed the latter.
    return buft == &ctx->dev->buft;
}

static bool gk_metal_backend_offload_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);

    // Copies here are memcpys within shared memory rather than transfers over
    // a bus, so the bar for moving work in is lower than on a discrete part -
    // but not zero: a copy is still a pass over the data.
    const int64_t min_batch = 8;

    switch (op->op) {
        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            return op->src[1]->ne[1] >= min_batch;
        default:
            return false;
    }
}

static void gk_metal_backend_synchronize(gk_backend_t backend) {
    struct gk_metal_backend_ctx * ctx = (struct gk_metal_backend_ctx *) backend->context;

    if (ctx->pending != nil) {
        [ctx->pending waitUntilCompleted];
        ctx->pending = nil;
    }
}

static const struct gk_backend_i g_metal_backend_iface = {
    /* .get_name                = */ gk_metal_backend_name,
    /* .free                    = */ gk_metal_backend_free,
    /* .get_default_buffer_type = */ gk_metal_backend_buft,
    /* .graph_compute           = */ gk_metal_backend_compute,
    /* .supports_op             = */ gk_metal_backend_supports_op,
    /* .supports_buft           = */ gk_metal_backend_supports_buft,
    /* .offload_op              = */ gk_metal_backend_offload_op,
    /* .synchronize             = */ gk_metal_backend_synchronize,
    /* .set_tensor_async        = */ NULL, // shared memory: a write is a write
    /* .get_tensor_async        = */ NULL,
};

// --------------------------------------------------------------------------
// the device vtable
// --------------------------------------------------------------------------

static const char * gk_metal_device_name(gk_device_t dev) {
    return ((struct gk_metal_device_ctx *) dev->context)->name;
}

static const char * gk_metal_device_description(gk_device_t dev) {
    return ((struct gk_metal_device_ctx *) dev->context)->description;
}

static void gk_metal_device_memory(gk_device_t dev, size_t * free_out, size_t * total_out) {
    struct gk_metal_device_ctx * ctx = (struct gk_metal_device_ctx *) dev->context;

    const size_t total = ctx->total_memory;
    const size_t used  = (size_t) [ctx->mtl currentAllocatedSize];

    if (total_out != NULL) { *total_out = total; }
    if (free_out  != NULL) { *free_out  = total > used ? total - used : 0; }
}

static enum gk_device_type gk_metal_device_type(gk_device_t dev) {
    // Every Metal device gk runs on shares memory with the host, which is what
    // IGPU means to the scheduler.
    GK_UNUSED(dev);
    return GK_DEVICE_TYPE_IGPU;
}

static gk_backend_t gk_metal_device_init_backend(gk_device_t dev) {
    struct gk_metal_device_ctx * d = (struct gk_metal_device_ctx *) dev->context;

    struct gk_metal_backend_ctx * ctx =
        (struct gk_metal_backend_ctx *) calloc(1, sizeof(struct gk_metal_backend_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->dev   = d;
    ctx->queue = [d->mtl newCommandQueue];

    if (ctx->queue == nil) {
        gk_logf("gk Metal: could not create a command queue on %s\n", d->name);
        free(ctx);
        return NULL;
    }

    @autoreleasepool {
        NSError * error = nil;
        NSString * source = [NSString stringWithUTF8String:gk_metal_shader_source];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.fastMathEnabled = NO; // the CPU pass is the reference; matching
                                      // it matters more than the last few
                                      // percent of throughput

        ctx->library = [d->mtl newLibraryWithSource:source options:options error:&error];

        if (ctx->library == nil) {
            gk_logf("gk Metal: shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            ctx->queue = nil;
            free(ctx);
            return NULL;
        }

        for (int i = 0; i < GK_METAL_K_COUNT; ++i) {
            NSString * name = [NSString stringWithUTF8String:g_metal_kernel_names[i]];
            id<MTLFunction> fn = [ctx->library newFunctionWithName:name];

            if (fn == nil) {
                gk_logf("gk Metal: kernel %s is missing from the library\n",
                        g_metal_kernel_names[i]);
                ctx->library = nil;
                ctx->queue   = nil;
                free(ctx);
                return NULL;
            }

            ctx->pipelines[i] = [d->mtl newComputePipelineStateWithFunction:fn error:&error];

            if (ctx->pipelines[i] == nil) {
                gk_logf("gk Metal: pipeline for %s failed: %s\n", g_metal_kernel_names[i],
                        [[error localizedDescription] UTF8String]);
                ctx->library = nil;
                ctx->queue   = nil;
                free(ctx);
                return NULL;
            }
        }
    }

    gk_backend_t backend = (gk_backend_t) malloc(sizeof(struct gk_backend));
    if (backend == NULL) {
        free(ctx);
        return NULL;
    }

    backend->iface   = g_metal_backend_iface;
    backend->context = ctx;
    backend->device  = &d->device;

    return backend;
}

static gk_backend_buffer_type_t gk_metal_device_buft(gk_device_t dev) {
    return &((struct gk_metal_device_ctx *) dev->context)->buft;
}

static bool gk_metal_device_supports_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_metal_supports_op(op);
}

static bool gk_metal_device_supports_buft(gk_device_t dev, gk_backend_buffer_type_t buft) {
    return buft == &((struct gk_metal_device_ctx *) dev->context)->buft;
}

static bool gk_metal_device_offload_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_metal_backend_offload_op(NULL, op);
}

static const struct gk_device_i g_metal_device_iface = {
    /* .get_name             = */ gk_metal_device_name,
    /* .get_description      = */ gk_metal_device_description,
    /* .get_memory           = */ gk_metal_device_memory,
    /* .get_type             = */ gk_metal_device_type,
    /* .init_backend         = */ gk_metal_device_init_backend,
    /* .buffer_type          = */ gk_metal_device_buft,
    /* .host_buffer_type     = */ NULL, // shared memory already is host memory
    /* .buffer_from_host_ptr = */ NULL, // an MTLBuffer cannot be conjured
                                        // around an arbitrary page
    /* .supports_op          = */ gk_metal_device_supports_op,
    /* .supports_buft        = */ gk_metal_device_supports_buft,
    /* .offload_op           = */ gk_metal_device_offload_op,
    /* .get_features         = */ NULL, // one library, compiled at load; there
                                        // is no build variant to report
};

// --------------------------------------------------------------------------
// discovery
// --------------------------------------------------------------------------

void gk_metal_register_devices(void) {
    if (g_metal_discovered) {
        return;
    }
    g_metal_discovered = true;

    @autoreleasepool {
        id<MTLDevice> mtl = MTLCreateSystemDefaultDevice();

        if (mtl == nil) {
            return; // no Metal device: not an error, the CPU is always there
        }

        struct gk_metal_device_ctx * d = &g_metal_devices[g_metal_n_devices];
        memset(d, 0, sizeof(*d));

        d->index = 0;
        d->mtl   = mtl;

        // The working-set limit is what the driver will actually let a process
        // hold, which is the number a placement decision should be made
        // against - not the machine's total RAM.
        d->total_memory = (size_t) [mtl recommendedMaxWorkingSetSize];

        snprintf(d->name, sizeof(d->name), "Metal%d", d->index);
        snprintf(d->description, sizeof(d->description), "%s", [[mtl name] UTF8String]);

        d->buft.iface   = g_metal_buft_iface;
        d->buft.context = d;
        d->buft.device  = &d->device;

        d->device.iface   = g_metal_device_iface;
        d->device.backend = "Metal";
        d->device.index   = d->index;
        d->device.context = d;
        snprintf(d->device.name, sizeof(d->device.name), "%s", d->name);

        g_metal_n_devices++;

        gk_device_register(&d->device);
    }
}

gk_backend_t gk_backend_metal_init(void) {
    gk_metal_register_devices();

    if (g_metal_n_devices == 0) {
        return NULL;
    }

    return gk_metal_device_init_backend(&g_metal_devices[0].device);
}
