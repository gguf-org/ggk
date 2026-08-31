#pragma once

// Internals shared across the gk sources. Nothing here is installed.

#include "gk.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// diagnostics
// --------------------------------------------------------------------------

GK_API void gk_logf(const char * fmt, ...);

// Marked noreturn so a switch whose default aborts does not also need a
// dead return statement to satisfy the compiler.
#if defined(__GNUC__) || defined(__clang__)
#  define GK_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#  define GK_NORETURN __declspec(noreturn)
#else
#  define GK_NORETURN
#endif

GK_API GK_NORETURN void gk_abort(const char * file, int line, const char * fmt, ...);

#define GK_ABORT(...) gk_abort(__FILE__, __LINE__, __VA_ARGS__)

#define GK_ASSERT(x) \
    do { \
        if (!(x)) { \
            GK_ABORT("assertion failed: %s", #x); \
        } \
    } while (0)

// Shape assertions read better as their own macro - they fire often while a
// new op is being written and the message is worth the few lines.
#define GK_ASSERT_SHAPE(t, e0, e1, e2, e3) \
    do { \
        GK_ASSERT((t)->ne[0] == (e0)); \
        GK_ASSERT((t)->ne[1] == (e1)); \
        GK_ASSERT((t)->ne[2] == (e2)); \
        GK_ASSERT((t)->ne[3] == (e3)); \
    } while (0)

#define GK_UNUSED(x) (void) (x)

#define GK_MIN(a, b) ((a) < (b) ? (a) : (b))
#define GK_MAX(a, b) ((a) > (b) ? (a) : (b))

// Unpacks a tensor's extents and strides into locals named ne00.. / nb00..
// The digit pair is (source index, dimension), which is the naming every
// kernel below uses.
#define GK_TENSOR_LOCALS_1(type, prefix, t, suffix) \
    const type prefix##0##suffix = (t) ? (t)->suffix[0] : 0; GK_UNUSED(prefix##0##suffix);
#define GK_TENSOR_LOCALS_2(type, prefix, t, suffix) \
    GK_TENSOR_LOCALS_1(type, prefix, t, suffix) \
    const type prefix##1##suffix = (t) ? (t)->suffix[1] : 0; GK_UNUSED(prefix##1##suffix);
#define GK_TENSOR_LOCALS_3(type, prefix, t, suffix) \
    GK_TENSOR_LOCALS_2(type, prefix, t, suffix) \
    const type prefix##2##suffix = (t) ? (t)->suffix[2] : 0; GK_UNUSED(prefix##2##suffix);
#define GK_TENSOR_LOCALS(type, prefix, t, suffix) \
    GK_TENSOR_LOCALS_3(type, prefix, t, suffix) \
    const type prefix##3##suffix = (t) ? (t)->suffix[3] : 0; GK_UNUSED(prefix##3##suffix);

#define GK_TENSOR_UNARY_OP_LOCALS \
    GK_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GK_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GK_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GK_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GK_TENSOR_BINARY_OP_LOCALS \
    GK_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GK_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GK_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GK_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GK_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GK_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

// --------------------------------------------------------------------------
// type traits
//
// The first four fields mirror what the quantizer publishes for the same type
// - they are the format's own properties and are read straight from it, so the
// two libraries can never disagree about a block layout. The rest is what an
// engine needs on top of a codec:
//
//   to_float / from_float   row-at-a-time conversion, used by casts, by CPU
//                           fallbacks and by anything that has to see f32
//   vec_dot_type            the type the right-hand side of a matmul must be
//                           converted to before vec_dot will accept it. A
//                           quantized weight is never dequantized for a
//                           matmul; instead the activations are quantized to
//                           this type and the dot runs in integer arithmetic
//   vec_dot                 dot product of one row of the weight type with one
//                           row of vec_dot_type
//   nrows                   how many destination rows one vec_dot call fills;
//                           >1 for layouts that interleave rows
// --------------------------------------------------------------------------

typedef void (*gk_to_float_t)  (const void * src, float * dst, int64_t k);
typedef void (*gk_from_float_t)(const float * src, void * dst, int64_t k);
typedef void (*gk_vec_dot_t)   (int n, float * s, size_t bs,
                                const void * x, size_t bx,
                                const void * y, size_t by, int nrc);

struct gk_type_traits {
    const char *    name;
    int64_t         blck_size;
    size_t          type_size;
    bool            is_quantized;
    gk_to_float_t   to_float;
    gk_from_float_t from_float;
    enum gk_type    vec_dot_type;
    gk_vec_dot_t    vec_dot;
    int64_t         nrows;
};

GK_API const struct gk_type_traits * gk_get_type_traits(enum gk_type type);

// How a custom op travels in a tensor's op params: the function pointer, the
// thread cap, and the caller's context pointer.
struct gk_custom_params {
    void * fun;
    int    n_tasks;
    void * userdata;
};

// --------------------------------------------------------------------------
// narrow float conversions
//
// These forward to the quantizer's scalar conversions so there is one
// definition of f16/bf16 rounding in the tree. The row helpers are where SIMD
// gets attached later; the scalar path stays as the reference.
// --------------------------------------------------------------------------

typedef uint16_t gk_fp16_t;

typedef struct { uint16_t bits; } gk_bf16_t;

GK_API float     gk_fp16_to_fp32(gk_fp16_t x);
GK_API gk_fp16_t gk_fp32_to_fp16(float x);
GK_API float     gk_bf16_to_fp32(gk_bf16_t x);
GK_API gk_bf16_t gk_fp32_to_bf16(float x);

GK_API void gk_fp16_to_fp32_row(const gk_fp16_t * x, float * y, int64_t n);
GK_API void gk_fp32_to_fp16_row(const float * x, gk_fp16_t * y, int64_t n);
GK_API void gk_bf16_to_fp32_row(const gk_bf16_t * x, float * y, int64_t n);
GK_API void gk_fp32_to_bf16_row(const float * x, gk_bf16_t * y, int64_t n);

// --------------------------------------------------------------------------
// context internals
// --------------------------------------------------------------------------

struct gk_ctx {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct gk_object * objects_begin;
    struct gk_object * objects_end;
};

// Every allocation in the arena is preceded by one of these, so the arena can
// be walked in order without a separate index.
enum gk_object_type {
    GK_OBJECT_TYPE_TENSOR,
    GK_OBJECT_TYPE_GRAPH,
    GK_OBJECT_TYPE_WORK_BUFFER,
};

struct gk_object {
    size_t offs;
    size_t size;

    struct gk_object * next;

    enum gk_object_type type;

    char padding[4];
};

#define GK_OBJECT_SIZE sizeof(struct gk_object)

// Rounds `n` up to the arena's alignment.
static inline size_t gk_pad_size(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

// The graph's node list plus the visited-set it needs while being built.
struct gk_hash_set {
    size_t      size;
    uint32_t *  used; // one bit per slot
    struct gk_tensor ** keys;
};

struct gk_cgraph {
    int size;      // capacity
    int n_nodes;   // tensors to evaluate, in order
    int n_leafs;   // tensors with no sources

    struct gk_tensor ** nodes;
    struct gk_tensor ** grads;
    struct gk_tensor ** grad_accs;
    struct gk_tensor ** leafs;

    struct gk_hash_set visited_hash_set;
};

GK_API struct gk_object * gk_new_object(struct gk_ctx * ctx, enum gk_object_type type, size_t size);

// arena cost of one tensor: its header plus the struct
GK_API size_t gk_tensor_overhead(void);

// a graph's node capacity (as opposed to gk_graph_n_nodes, its count)
GK_API int gk_graph_size(struct gk_cgraph * g);

// the YaRN correction band, exported for the compatibility layer
GK_API void gk_rope_corr_dims(int n_dims, int n_ctx_orig, float base,
                              float beta_fast, float beta_slow, float dims[2]);

GK_API bool gk_graph_contains(const struct gk_cgraph * g, const struct gk_tensor * t);

// --------------------------------------------------------------------------
// thread pool
//
// Fork-join: `gk_pool_run` hands the same function to every thread and returns
// once all of them have finished it. Each thread takes its slice of the work
// from its index, so partitioning is the caller's decision, not the pool's.
// The calling thread participates as index 0 rather than idling.
// --------------------------------------------------------------------------

struct gk_pool;

typedef void (*gk_pool_fn)(void * ctx, int ith, int nth);

GK_API struct gk_pool * gk_pool_create(int n_threads); // <= 0 means one per core
GK_API void             gk_pool_free(struct gk_pool * pool);
GK_API int              gk_pool_n_threads(const struct gk_pool * pool);
GK_API void             gk_pool_run(struct gk_pool * pool, gk_pool_fn fn, void * ctx);

// --------------------------------------------------------------------------
// backend vtables
//
// Internal because only backend implementations fill these in. Callers use the
// gk_backend_* functions in gk.h, which dispatch through here.
// --------------------------------------------------------------------------

struct gk_backend_buffer_type_i {
    const char *        (*get_name)      (gk_backend_buffer_type_t buft);
    gk_backend_buffer_t (*alloc_buffer)  (gk_backend_buffer_type_t buft, size_t size);
    size_t              (*get_alignment) (gk_backend_buffer_type_t buft);
    // optional: how much space this tensor really costs here, when it is not
    // simply gk_nbytes - a device that pads rows would override it
    size_t              (*get_alloc_size)(gk_backend_buffer_type_t buft,
                                          const struct gk_tensor * tensor);
    bool                (*is_host)       (gk_backend_buffer_type_t buft);
};

struct gk_backend_buffer_type {
    struct gk_backend_buffer_type_i iface;
    void *      context;
    gk_device_t device; // which device's memory this is; NULL means the host's
};

struct gk_backend_buffer_i {
    void   (*free_buffer)  (gk_backend_buffer_t buffer);
    void * (*get_base)     (gk_backend_buffer_t buffer);
    void   (*init_tensor)  (gk_backend_buffer_t buffer, struct gk_tensor * tensor);
    void   (*set_tensor)   (gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                            const void * data, size_t offset, size_t size);
    void   (*get_tensor)   (gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                            void * data, size_t offset, size_t size);
    void   (*clear)        (gk_backend_buffer_t buffer, uint8_t value);
    // optional: fill a range without staging it through host memory
    void   (*memset_tensor)(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                            uint8_t value, size_t offset, size_t size);
    // optional: copy from another tensor when this buffer can reach it - the
    // device-to-device path. Returns false to say "not mine", and the caller
    // falls back to a host bounce.
    bool   (*cpy_tensor)   (gk_backend_buffer_t buffer, const struct gk_tensor * src,
                            struct gk_tensor * dst);
};

// The caller's tag on a buffer. gk assigns it no meaning of its own and only
// ever asks one question of it - whether a buffer holds weights, which the
// scheduler needs before it will consider moving an op away from them. The
// value is the one the ggml compatibility layer writes, which is the only
// thing that writes it.
#define GK_BUFFER_USAGE_WEIGHTS 1

struct gk_backend_buffer {
    struct gk_backend_buffer_i iface;
    gk_backend_buffer_type_t   buft;
    void *                     context;
    size_t                     size;
    int                        usage; // caller-defined tag; see GK_BUFFER_USAGE_WEIGHTS
};

struct gk_backend_i {
    const char *             (*get_name)               (gk_backend_t backend);
    void                     (*free)                   (gk_backend_t backend);
    gk_backend_buffer_type_t (*get_default_buffer_type)(gk_backend_t backend);
    enum gk_status           (*graph_compute)          (gk_backend_t backend,
                                                        struct gk_cgraph * graph);
    bool                     (*supports_op)            (gk_backend_t backend,
                                                        const struct gk_tensor * op);

    // Everything below is optional. A backend that leaves an entry NULL gets
    // the host-memory answer, which is the right one for the CPU and the
    // reason these are not pure virtual: adding a device concern to the
    // interface must not force an edit to the backend that has none.
    bool                     (*supports_buft)          (gk_backend_t backend,
                                                        gk_backend_buffer_type_t buft);
    bool                     (*offload_op)             (gk_backend_t backend,
                                                        const struct gk_tensor * op);
    void                     (*synchronize)            (gk_backend_t backend);
    void                     (*set_tensor_async)       (gk_backend_t backend,
                                                        struct gk_tensor * tensor,
                                                        const void * data, size_t offset, size_t size);
    void                     (*get_tensor_async)       (gk_backend_t backend,
                                                        const struct gk_tensor * tensor,
                                                        void * data, size_t offset, size_t size);
};

struct gk_backend {
    struct gk_backend_i iface;
    void *      context;
    gk_device_t device; // NULL for a backend that was created without one
};

// --------------------------------------------------------------------------
// devices
//
// A device is a vtable plus an index into whatever the owning backend calls
// its hardware. Each compiled-in device backend implements this once and
// registers its devices at startup; gk_device.c does nothing but keep the
// list and dispatch.
// --------------------------------------------------------------------------

struct gk_device_i {
    const char *             (*get_name)       (gk_device_t dev);
    const char *             (*get_description)(gk_device_t dev);
    void                     (*get_memory)     (gk_device_t dev, size_t * free, size_t * total);
    enum gk_device_type      (*get_type)       (gk_device_t dev);
    gk_backend_t             (*init_backend)   (gk_device_t dev);
    gk_backend_buffer_type_t (*buffer_type)    (gk_device_t dev);
    // optional: host memory this device can read without staging
    gk_backend_buffer_type_t (*host_buffer_type)(gk_device_t dev);
    // optional: wrap caller-owned host memory
    gk_backend_buffer_t      (*buffer_from_host_ptr)(gk_device_t dev, void * ptr, size_t size);
    bool                     (*supports_op)    (gk_device_t dev, const struct gk_tensor * op);
    bool                     (*supports_buft)  (gk_device_t dev, gk_backend_buffer_type_t buft);
    bool                     (*offload_op)     (gk_device_t dev, const struct gk_tensor * op);
    // optional: what this build compiled for this device, for the banner
    const struct gk_feature * (*get_features)  (gk_device_t dev);
};

struct gk_device {
    struct gk_device_i iface;
    const char *       backend;  // "CPU", "CUDA", "ROCm", "Metal", "Vulkan"
    int                index;    // which device of that backend
    void *             context;  // the backend's own per-device state
    char               name[32]; // "CUDA0" and friends, filled in at registration
};

// Called by a device backend from its discovery hook below. The registry takes
// the pointer, not a copy, so the struct must outlive the process' use of it -
// in practice they are all static or allocated once at discovery.
GK_API void gk_device_register(gk_device_t dev);

// Each compiled-in device backend defines one of these. They are called once,
// in this order, the first time anything asks for the device list.
#if defined(GK_USE_CUDA)
void gk_cuda_register_devices(void);
#endif
#if defined(GK_USE_METAL)
void gk_metal_register_devices(void);
#endif
#if defined(GK_USE_VULKAN)
void gk_vulkan_register_devices(void);
#endif

// Used by a backend implementation to construct its buffer object.
GK_API gk_backend_buffer_t gk_backend_buffer_init(
        gk_backend_buffer_type_t buft, const struct gk_backend_buffer_i * iface,
        void * context, size_t size);

// True when the CPU pass has a kernel for this op. Kept next to the dispatch
// switch so the two cannot drift apart.
GK_API bool gk_cpu_has_kernel(enum gk_op op);

// --------------------------------------------------------------------------
// integer dot paths (gk_vecdot.c)
//
// Activation encoders, and one dot per (weight format, activation format)
// pair. A weight format opts into these by naming them in its traits entry;
// anything that has not opted in keeps the float path in gk_traits.c.
// --------------------------------------------------------------------------

GK_API void gk_quantize_row_q8_0(const float * src, void * dst, int64_t k);
GK_API void gk_quantize_row_q8_1(const float * src, void * dst, int64_t k);
GK_API void gk_quantize_row_q8_K(const float * src, void * dst, int64_t k);

#define GK_DECL_DOT(name) \
    GK_API void name(int n, float * s, size_t bs, const void * vx, size_t bx, \
                     const void * vy, size_t by, int nrc)

GK_DECL_DOT(gk_vec_dot_q8_0_q8_0);
GK_DECL_DOT(gk_vec_dot_q4_0_q8_0);
GK_DECL_DOT(gk_vec_dot_q5_0_q8_0);
GK_DECL_DOT(gk_vec_dot_q4_1_q8_1);
GK_DECL_DOT(gk_vec_dot_q5_1_q8_1);
GK_DECL_DOT(gk_vec_dot_iq2_xxs_q8_K);
GK_DECL_DOT(gk_vec_dot_iq2_xs_q8_K);
GK_DECL_DOT(gk_vec_dot_iq2_s_q8_K);
GK_DECL_DOT(gk_vec_dot_iq3_xxs_q8_K);
GK_DECL_DOT(gk_vec_dot_iq3_s_q8_K);
GK_DECL_DOT(gk_vec_dot_iq1_s_q8_K);
GK_DECL_DOT(gk_vec_dot_iq1_m_q8_K);
GK_DECL_DOT(gk_vec_dot_q1_0_q8_0);
GK_DECL_DOT(gk_vec_dot_q2_0_q8_0);
GK_DECL_DOT(gk_vec_dot_iq4_nl_q8_0);
GK_DECL_DOT(gk_vec_dot_iq4_xs_q8_K);
GK_DECL_DOT(gk_vec_dot_mxfp4_q8_0);
GK_DECL_DOT(gk_vec_dot_nvfp4_q8_0);

// q3_K's sixteen 6-bit scales, unpacked both ways. The vector kernel does not
// call the scalar unpack - it cannot afford to touch memory - so this is the
// one place in gk with two implementations of the same bit layout, and
// test-foundation holds them against each other. `_vector` is the scalar one
// in a build without the vector path, where the check is trivially true.
GK_API void gk_q3_k_unpack_scalar(const uint8_t * src, int8_t * out);
GK_API void gk_q3_k_unpack_vector(const uint8_t * src, int8_t * out);
GK_DECL_DOT(gk_vec_dot_tq1_0_q8_K);
GK_DECL_DOT(gk_vec_dot_tq2_0_q8_K);
GK_DECL_DOT(gk_vec_dot_q2_K_q8_K);
GK_DECL_DOT(gk_vec_dot_q3_K_q8_K);
GK_DECL_DOT(gk_vec_dot_q4_K_q8_K);
GK_DECL_DOT(gk_vec_dot_q5_K_q8_K);
GK_DECL_DOT(gk_vec_dot_q6_K_q8_K);

#undef GK_DECL_DOT

#ifdef __cplusplus
}
#endif
