#pragma once

// gk - the gguf compute kernels.
//
// This is the tensor layer the server, the diffuser and the multimodal
// projectors run on. It is written to the same rules as ../quantizer: the
// formats it reads are fixed by what is already on disk in published model
// files, and everything above those bytes is our own.
//
// The object model is deliberately small. There are three things:
//
//   * a tensor - a typed, strided view of memory, up to four dimensions;
//   * a context - a bump arena that owns the tensor structs (never the data);
//   * a graph - a flat, topologically ordered list of tensors to evaluate.
//
// Building a graph never computes anything and never touches tensor data. The
// builders below only allocate a struct, record an op and its sources, and
// work out the result shape. Memory for the results is assigned later, in one
// pass over the finished graph, and the graph is then handed to a backend to
// run. Keeping those three phases apart is what lets the same graph be
// re-planned for a different device, or re-run with new inputs, without being
// rebuilt.
//
// Shapes follow the GGUF convention: ne[0] is the fastest-moving dimension and
// a "row" is ne[0] elements long. Strides are in bytes and are carried
// explicitly per dimension, so a permute or a transpose is only a rewrite of
// ne/nb - no data moves until something asks for a contiguous copy.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GK_API
#  if defined(GK_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#      ifdef GK_BUILD
#        define GK_API __declspec(dllexport)
#      else
#        define GK_API __declspec(dllimport)
#      endif
#    else
#      define GK_API __attribute__((visibility("default")))
#    endif
#  else
#    define GK_API
#  endif
#endif

#define GK_MAX_DIMS       4
#define GK_MAX_SRC       10
#define GK_MAX_OP_PARAMS 64

// Sizes a field inside struct gk_tensor, so every object file in a build has
// to agree on it. The engine raises it to 128 for the diffusion models' long
// tensor names.
#ifndef GK_MAX_NAME
#define GK_MAX_NAME      64
#endif

// tensor data is aligned to this; wide enough for any SIMD load we issue
#define GK_MEM_ALIGN 16

// --------------------------------------------------------------------------
// types
//
// The numeric formats a tensor can hold. The quantized entries are the GGUF
// block formats - their layouts live in ../quantizer/src/kernels/qz_format.h,
// which is the single definition of those bytes for both the quantizer and
// this library. The enum values are the ones written into GGUF files, so they
// are fixed by the format and must not be renumbered.
// --------------------------------------------------------------------------

enum gk_type {
    GK_TYPE_F32     = 0,
    GK_TYPE_F16     = 1,
    GK_TYPE_Q4_0    = 2,
    GK_TYPE_Q4_1    = 3,
    // 4 and 5 were Q4_2 and Q4_3; both were withdrawn before GGUF v2 and no
    // published file uses them
    GK_TYPE_Q5_0    = 6,
    GK_TYPE_Q5_1    = 7,
    GK_TYPE_Q8_0    = 8,
    GK_TYPE_Q8_1    = 9,
    GK_TYPE_Q2_K    = 10,
    GK_TYPE_Q3_K    = 11,
    GK_TYPE_Q4_K    = 12,
    GK_TYPE_Q5_K    = 13,
    GK_TYPE_Q6_K    = 14,
    GK_TYPE_Q8_K    = 15,
    GK_TYPE_IQ2_XXS = 16,
    GK_TYPE_IQ2_XS  = 17,
    GK_TYPE_IQ3_XXS = 18,
    GK_TYPE_IQ1_S   = 19,
    GK_TYPE_IQ4_NL  = 20,
    GK_TYPE_IQ3_S   = 21,
    GK_TYPE_IQ2_S   = 22,
    GK_TYPE_IQ4_XS  = 23,
    GK_TYPE_I8      = 24,
    GK_TYPE_I16     = 25,
    GK_TYPE_I32     = 26,
    GK_TYPE_I64     = 27,
    GK_TYPE_F64     = 28,
    GK_TYPE_IQ1_M   = 29,
    GK_TYPE_BF16    = 30,
    // 31..33 were the short-lived Q4_0_4_4/4_8/8_8 repack types; they are now
    // produced at load time from Q4_0 instead of stored
    GK_TYPE_TQ1_0   = 34,
    GK_TYPE_TQ2_0   = 35,
    // 36 and 37 were IQ4_NL_4_4 and IQ4_NL_4_8, withdrawn like the Q4_0_* set
    GK_TYPE_MXFP4   = 39,
    GK_TYPE_NVFP4   = 40,
    GK_TYPE_Q1_0    = 41,
    GK_TYPE_Q2_0    = 42,
    GK_TYPE_COUNT   = 43,
};

// --------------------------------------------------------------------------
// operations
//
// One entry per distinct compute kernel. Ops that only rewrite shape or
// strides (view, reshape, permute, transpose) still get an entry, because the
// graph has to carry the dependency even though evaluating them is a no-op.
// --------------------------------------------------------------------------

// The order here mirrors the engine's historical ggml_op enum exactly, entry
// for entry, because the compatibility layer shares tensor structs with code
// compiled against that enum - the integer stored in tensor->op has to mean
// the same thing on both sides. Entries this library does not implement (the
// *_BACK gradients, the training steps) still hold their slot for that reason;
// the dispatch aborts on them rather than the enum omitting them.
enum gk_op {
    GK_OP_NONE = 0,

    GK_OP_DUP,
    GK_OP_ADD,
    GK_OP_ADD_ID,
    GK_OP_ADD1,
    GK_OP_ACC,
    GK_OP_SUB,
    GK_OP_MUL,
    GK_OP_DIV,
    GK_OP_SQR,
    GK_OP_SQRT,
    GK_OP_LOG,
    GK_OP_SIN,
    GK_OP_COS,
    GK_OP_SUM,
    GK_OP_SUM_ROWS,
    GK_OP_CUMSUM,
    GK_OP_MEAN,
    GK_OP_ARGMAX,
    GK_OP_COUNT_EQUAL,
    GK_OP_REPEAT,
    GK_OP_REPEAT_BACK,
    GK_OP_CONCAT,
    GK_OP_SILU_BACK,
    GK_OP_NORM,       // layer norm
    GK_OP_RMS_NORM,
    GK_OP_RMS_NORM_BACK,
    GK_OP_GROUP_NORM,
    GK_OP_L2_NORM,

    GK_OP_MUL_MAT,
    GK_OP_MUL_MAT_ID, // per-token expert selection, for mixture-of-experts
    GK_OP_OUT_PROD,

    GK_OP_SCALE,
    GK_OP_SET,
    GK_OP_CPY,
    GK_OP_CONT,
    GK_OP_RESHAPE,
    GK_OP_VIEW,
    GK_OP_PERMUTE,
    GK_OP_TRANSPOSE,
    GK_OP_GET_ROWS,
    GK_OP_GET_ROWS_BACK,
    GK_OP_SET_ROWS,
    GK_OP_DIAG,
    GK_OP_DIAG_MASK_INF,
    GK_OP_DIAG_MASK_ZERO,
    GK_OP_SOFT_MAX,
    GK_OP_SOFT_MAX_BACK,
    GK_OP_ROPE,
    GK_OP_ROPE_BACK,
    GK_OP_CLAMP,
    GK_OP_CONV_TRANSPOSE_1D,
    GK_OP_IM2COL,
    GK_OP_IM2COL_BACK,
    GK_OP_IM2COL_3D,
    GK_OP_COL2IM_1D,
    GK_OP_CONV_2D,
    GK_OP_CONV_3D,
    GK_OP_CONV_2D_DW,
    GK_OP_CONV_TRANSPOSE_2D,
    GK_OP_POOL_1D,
    GK_OP_POOL_2D,
    GK_OP_POOL_2D_BACK,
    GK_OP_UPSCALE,
    GK_OP_PAD,
    GK_OP_PAD_REFLECT_1D,
    GK_OP_ROLL,
    GK_OP_ARANGE,
    GK_OP_TIMESTEP_EMBEDDING,
    GK_OP_ARGSORT,
    GK_OP_TOP_K,
    GK_OP_LEAKY_RELU,
    GK_OP_TRI,
    GK_OP_FILL,

    GK_OP_FLASH_ATTN_EXT,
    GK_OP_FLASH_ATTN_BACK,
    GK_OP_SSM_CONV,   // state-space models (Mamba and friends)
    GK_OP_SSM_SCAN,
    GK_OP_WIN_PART,
    GK_OP_WIN_UNPART,
    GK_OP_GET_REL_POS,
    GK_OP_ADD_REL_POS,
    GK_OP_RWKV_WKV6,
    GK_OP_GATED_LINEAR_ATTN,
    GK_OP_RWKV_WKV7,
    GK_OP_SOLVE_TRI,
    GK_OP_GATED_DELTA_NET,
    GK_OP_LIGHTNING_INDEXER,
    GK_OP_DSV4_HC_COMB,
    GK_OP_DSV4_HC_PRE,
    GK_OP_DSV4_HC_POST,

    GK_OP_UNARY,

    GK_OP_MAP_CUSTOM1,
    GK_OP_MAP_CUSTOM2,
    GK_OP_MAP_CUSTOM3,

    GK_OP_CUSTOM,

    GK_OP_CROSS_ENTROPY_LOSS,
    GK_OP_CROSS_ENTROPY_LOSS_BACK,
    GK_OP_OPT_STEP_ADAMW,
    GK_OP_OPT_STEP_SGD,

    GK_OP_GLU,

    GK_OP_COUNT,
};

// Elementwise activations. These share GK_OP_UNARY and are selected by the
// first op param, so adding one does not widen the op dispatch table.
// Same rule as gk_op: this order is the compatibility contract.
enum gk_unary_op {
    GK_UNARY_OP_ABS = 0,
    GK_UNARY_OP_SGN,
    GK_UNARY_OP_NEG,
    GK_UNARY_OP_STEP,
    GK_UNARY_OP_TANH,
    GK_UNARY_OP_ELU,
    GK_UNARY_OP_RELU,
    GK_UNARY_OP_SIGMOID,
    GK_UNARY_OP_GELU,
    GK_UNARY_OP_GELU_QUICK,
    GK_UNARY_OP_SILU,
    GK_UNARY_OP_HARDSWISH,
    GK_UNARY_OP_HARDSIGMOID,
    GK_UNARY_OP_EXP,
    GK_UNARY_OP_EXPM1,
    GK_UNARY_OP_SOFTPLUS,
    GK_UNARY_OP_GELU_ERF,
    GK_UNARY_OP_XIELU,
    GK_UNARY_OP_FLOOR,
    GK_UNARY_OP_CEIL,
    GK_UNARY_OP_ROUND,
    GK_UNARY_OP_TRUNC,
    GK_UNARY_OP_COUNT,
};

// Gated linear units: the input splits in two halves, one is passed through an
// activation and multiplied into the other.
enum gk_glu_op {
    GK_GLU_OP_REGLU = 0,
    GK_GLU_OP_GEGLU,
    GK_GLU_OP_SWIGLU,
    GK_GLU_OP_SWIGLU_OAI,
    GK_GLU_OP_GEGLU_ERF,
    GK_GLU_OP_GEGLU_QUICK,
    GK_GLU_OP_COUNT,
};

enum gk_status {
    GK_STATUS_SUCCESS = 0,
    GK_STATUS_ALLOC_FAILED,
    GK_STATUS_NO_STORAGE, // a node was reached before its data was assigned
};

// Rotary variants. The pairing differs between them and a model trained with
// one produces nonsense under the other, so the value is part of the weights'
// contract, not a tuning knob.
#define GK_ROPE_TYPE_NORMAL 0
#define GK_ROPE_TYPE_NEOX   2
#define GK_ROPE_TYPE_MROPE  8
#define GK_ROPE_TYPE_VISION 24
#define GK_ROPE_TYPE_IMROPE 40 // interleaved M-RoPE

// how many sections a multi-axis rope carries
#define GK_MROPE_SECTIONS 4

enum gk_sort_order {
    GK_SORT_ORDER_ASC = 0,
    GK_SORT_ORDER_DESC,
};

// Triangular masks, for gk_tri and gk_solve_tri.
enum gk_tri_type {
    GK_TRI_TYPE_UPPER_DIAG = 0,
    GK_TRI_TYPE_UPPER      = 1,
    GK_TRI_TYPE_LOWER_DIAG = 2,
    GK_TRI_TYPE_LOWER      = 3,
};

enum gk_op_pool {
    GK_OP_POOL_MAX = 0,
    GK_OP_POOL_AVG,
    GK_OP_POOL_COUNT,
};

// Interpolation modes for gk_upscale / gk_interpolate. The flag bits start at
// bit 8 so a mode and its flags pack into one word.
enum gk_scale_mode {
    GK_SCALE_MODE_NEAREST  = 0,
    GK_SCALE_MODE_BILINEAR = 1,
    GK_SCALE_MODE_BICUBIC  = 2,
    GK_SCALE_MODE_COUNT,
};

enum gk_scale_flag {
    GK_SCALE_FLAG_ALIGN_CORNERS = (1 << 8),
    GK_SCALE_FLAG_ANTIALIAS     = (1 << 9),
};

// Requested arithmetic width for ops that offer a choice (matmul, attention).
enum gk_prec {
    GK_PREC_DEFAULT = 0,
    GK_PREC_F32     = 10,
};

// Structural hints a graph builder can attach to an op.
enum gk_op_hint {
    GK_HINT_NONE             = 0,
    GK_HINT_SRC0_IS_HADAMARD = 1,
};

enum gk_tensor_flag {
    GK_TENSOR_FLAG_INPUT   = 1,  // written from outside before the graph runs
    GK_TENSOR_FLAG_OUTPUT  = 2,  // read from outside after the graph runs
    GK_TENSOR_FLAG_PARAM   = 4,  // trainable
    GK_TENSOR_FLAG_LOSS    = 8,  // contributes to a training loss
    GK_TENSOR_FLAG_COMPUTE = 16, // must be evaluated even if nothing reads it
};

// --------------------------------------------------------------------------
// tensor
//
// `ne` is the extent per dimension and `nb` the stride in bytes per dimension.
// Unused trailing dimensions are 1, so rank is implicit and every tensor can
// be treated as 4-D. For a contiguous tensor:
//
//     nb[0] = type_size
//     nb[1] = nb[0] * (ne[0] / block_size)
//     nb[i] = nb[i-1] * ne[i-1]
//
// but nothing requires that - a view or a permute leaves nb describing the
// parent's layout, which is why every kernel reads strides rather than
// assuming them. For quantized types nb[0] is the size of a whole block and
// ne[0] must be a multiple of the block size, so a row is always a whole
// number of blocks.
//
// `data` is null while a graph is being built and is filled in by the
// allocator. `view_src` points at the tensor that actually owns the memory,
// with `view_offs` the byte offset into it; a tensor that owns its own memory
// has view_src null.
// --------------------------------------------------------------------------

struct gk_backend_buffer;

struct gk_tensor {
    enum gk_type type;

    struct gk_backend_buffer * buffer;

    int64_t ne[GK_MAX_DIMS];
    size_t  nb[GK_MAX_DIMS];

    enum gk_op op;

    // op parameters, stored as int32 so they stay aligned whatever the op
    // reads them back as
    int32_t op_params[GK_MAX_OP_PARAMS / sizeof(int32_t)];

    int32_t flags;

    struct gk_tensor * src[GK_MAX_SRC];

    struct gk_tensor * view_src;
    size_t             view_offs;

    void * data;

    char name[GK_MAX_NAME];

    void * extra; // backend scratch, owned by whichever backend set it

    char padding[8];
};

#define GK_TENSOR_SIZE sizeof(struct gk_tensor)

// --------------------------------------------------------------------------
// context
//
// A bump arena. Tensor structs are carved out of it in order and the whole
// thing is released at once; there is no per-tensor free, because graph
// building is a phase that ends. `no_alloc` builds shapes only, which is how
// the size of a graph is measured before its data buffer is committed.
// --------------------------------------------------------------------------

struct gk_ctx;

struct gk_init_params {
    size_t mem_size;
    void * mem_buffer; // null: the context allocates and owns mem_size bytes
    bool   no_alloc;   // do not reserve space for tensor data
};

GK_API struct gk_ctx * gk_init(struct gk_init_params params);
GK_API void            gk_free(struct gk_ctx * ctx);

GK_API size_t gk_used_mem(const struct gk_ctx * ctx);
GK_API void   gk_set_no_alloc(struct gk_ctx * ctx, bool no_alloc);
GK_API void * gk_mem_buffer(const struct gk_ctx * ctx);
GK_API size_t gk_mem_size  (const struct gk_ctx * ctx);
GK_API bool   gk_get_no_alloc(const struct gk_ctx * ctx);

// Walk the tensors an arena holds, in creation order; how a model loader
// enumerates what it built without keeping its own list.
GK_API struct gk_tensor * gk_get_first_tensor(const struct gk_ctx * ctx);
GK_API struct gk_tensor * gk_get_next_tensor (const struct gk_ctx * ctx, struct gk_tensor * t);
GK_API struct gk_tensor * gk_get_tensor      (struct gk_ctx * ctx, const char * name);

// --------------------------------------------------------------------------
// type queries
// --------------------------------------------------------------------------

GK_API const char * gk_type_name(enum gk_type type);
GK_API int64_t      gk_blck_size(enum gk_type type);  // elements per block
GK_API size_t       gk_type_size(enum gk_type type);  // bytes per block
GK_API bool         gk_is_quantized(enum gk_type type);
GK_API size_t       gk_row_size(enum gk_type type, int64_t ne);

GK_API const char * gk_op_name(enum gk_op op);
GK_API const char * gk_op_symbol(enum gk_op op);
GK_API const char * gk_unary_op_name(enum gk_unary_op op);
GK_API const char * gk_glu_op_name(enum gk_glu_op op);

// --------------------------------------------------------------------------
// tensor queries
// --------------------------------------------------------------------------

GK_API int64_t gk_nelements(const struct gk_tensor * t);
GK_API size_t  gk_nbytes   (const struct gk_tensor * t);
GK_API int64_t gk_nrows    (const struct gk_tensor * t);
GK_API int     gk_n_dims   (const struct gk_tensor * t); // trailing 1s dropped
GK_API size_t  gk_element_size(const struct gk_tensor * t);

GK_API bool gk_is_contiguous  (const struct gk_tensor * t);
// rows are packed but may be padded apart; the layout a scale can write over
GK_API bool gk_is_padded_1    (const struct gk_tensor * t);
GK_API bool gk_is_contiguous_1(const struct gk_tensor * t); // from dim 1 up
GK_API bool gk_is_contiguous_2(const struct gk_tensor * t); // from dim 2 up
GK_API bool gk_is_transposed  (const struct gk_tensor * t);
GK_API bool gk_is_permuted    (const struct gk_tensor * t);
GK_API bool gk_is_empty       (const struct gk_tensor * t);
GK_API bool gk_is_scalar      (const struct gk_tensor * t);
GK_API bool gk_is_vector      (const struct gk_tensor * t);
GK_API bool gk_is_matrix      (const struct gk_tensor * t);

GK_API bool gk_are_same_shape (const struct gk_tensor * a, const struct gk_tensor * b);
GK_API bool gk_are_same_stride(const struct gk_tensor * a, const struct gk_tensor * b);
// b's shape is a whole-number tiling of a's, so a can be broadcast onto b
GK_API bool gk_can_repeat(const struct gk_tensor * a, const struct gk_tensor * b);

GK_API const char *        gk_get_name(const struct gk_tensor * t);
GK_API struct gk_tensor *  gk_set_name(struct gk_tensor * t, const char * name);
GK_API struct gk_tensor *  gk_format_name(struct gk_tensor * t, const char * fmt, ...);

GK_API void gk_set_input (struct gk_tensor * t);
GK_API void gk_set_output(struct gk_tensor * t);

GK_API void    gk_set_op_params  (struct gk_tensor * t, const void * params, size_t size);
GK_API int32_t gk_get_op_params_i32(const struct gk_tensor * t, int i);
GK_API float   gk_get_op_params_f32(const struct gk_tensor * t, int i);
GK_API void    gk_set_op_params_i32(struct gk_tensor * t, int i, int32_t value);
GK_API void    gk_set_op_params_f32(struct gk_tensor * t, int i, float value);

GK_API enum gk_unary_op gk_get_unary_op(const struct gk_tensor * t);
GK_API enum gk_glu_op   gk_get_glu_op  (const struct gk_tensor * t);

// --------------------------------------------------------------------------
// tensor creation
// --------------------------------------------------------------------------

GK_API struct gk_tensor * gk_new_tensor(
        struct gk_ctx * ctx, enum gk_type type, int n_dims, const int64_t * ne);

GK_API struct gk_tensor * gk_new_tensor_1d(struct gk_ctx * ctx, enum gk_type type, int64_t ne0);
GK_API struct gk_tensor * gk_new_tensor_2d(struct gk_ctx * ctx, enum gk_type type, int64_t ne0, int64_t ne1);
GK_API struct gk_tensor * gk_new_tensor_3d(struct gk_ctx * ctx, enum gk_type type, int64_t ne0, int64_t ne1, int64_t ne2);
GK_API struct gk_tensor * gk_new_tensor_4d(struct gk_ctx * ctx, enum gk_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// A tensor whose storage is `src`'s, starting `offset` bytes in. Strides come
// out contiguous for the given shape; callers that need a strided window
// overwrite nb afterwards.
GK_API struct gk_tensor * gk_new_tensor_view(
        struct gk_ctx * ctx, enum gk_type type, int n_dims, const int64_t * ne,
        struct gk_tensor * src, size_t offset);

// same shape and type, fresh storage
GK_API struct gk_tensor * gk_dup_tensor (struct gk_ctx * ctx, const struct gk_tensor * src);
// same shape and type, no storage - a handle to be pointed at memory later
GK_API struct gk_tensor * gk_view_tensor(struct gk_ctx * ctx, struct gk_tensor * src);

GK_API struct gk_tensor * gk_new_i32(struct gk_ctx * ctx, int32_t value);
GK_API struct gk_tensor * gk_new_f32(struct gk_ctx * ctx, float   value);

// --------------------------------------------------------------------------
// graph
//
// Nodes are stored flat and in evaluation order. `gk_build_forward_expand`
// walks a tensor's sources depth-first and appends anything not already
// present, so calling it on the final tensor of a network produces a correct
// topological order in one pass. Calling it again on a second output extends
// the same graph.
// --------------------------------------------------------------------------

struct gk_cgraph;

GK_API size_t             gk_graph_overhead(void);
GK_API size_t             gk_graph_overhead_custom(size_t size, bool grads);
GK_API struct gk_cgraph * gk_new_graph(struct gk_ctx * ctx);
GK_API struct gk_cgraph * gk_new_graph_custom(struct gk_ctx * ctx, size_t size, bool grads);

GK_API void gk_build_forward_expand(struct gk_cgraph * graph, struct gk_tensor * tensor);

// Builds several alternative branches into one graph but flags only branch
// `idx` for evaluation. The others hold their places, so a graph's topology
// stays fixed while the computed path varies with the input - which is what
// lets a scheduler reuse its plan across calls.
GK_API struct gk_tensor * gk_build_forward_select(struct gk_cgraph * graph,
                                                  struct gk_tensor ** tensors,
                                                  int n_tensors, int idx);

GK_API int                gk_graph_n_nodes(const struct gk_cgraph * g);
GK_API struct gk_tensor * gk_graph_node   (struct gk_cgraph * g, int i); // i < 0 counts from the end
GK_API struct gk_tensor **gk_graph_nodes  (struct gk_cgraph * g);
GK_API void               gk_graph_add_node(struct gk_cgraph * g, struct gk_tensor * t);

GK_API int                gk_graph_n_leafs (const struct gk_cgraph * g);
GK_API struct gk_tensor * gk_graph_leaf    (struct gk_cgraph * g, int i);
GK_API void               gk_graph_add_leaf(struct gk_cgraph * g, struct gk_tensor * t);

GK_API void               gk_graph_clear(struct gk_cgraph * g);
GK_API void               gk_graph_cpy  (struct gk_cgraph * src, struct gk_cgraph * dst);
GK_API struct gk_cgraph   gk_graph_view (struct gk_cgraph * g, int start, int end);
GK_API struct gk_tensor * gk_graph_get_tensor(const struct gk_cgraph * g, const char * name);

// --------------------------------------------------------------------------
// op builders
//
// None of these compute anything. Each allocates a result tensor of the right
// shape and records the op and its sources; evaluation happens later, when a
// backend walks the graph.
//
// An `_inplace` variant writes over its first operand's storage. The node is
// still added to the graph, so the dependency stays visible to the scheduler.
// --------------------------------------------------------------------------

#define GK_BINARY(name) \
    GK_API struct gk_tensor * gk_##name(struct gk_ctx *, struct gk_tensor *, struct gk_tensor *); \
    GK_API struct gk_tensor * gk_##name##_inplace(struct gk_ctx *, struct gk_tensor *, struct gk_tensor *)

GK_BINARY(add);
GK_BINARY(sub);
GK_BINARY(mul);
GK_BINARY(div);

#undef GK_BINARY

#define GK_UNARY(name) \
    GK_API struct gk_tensor * gk_##name(struct gk_ctx *, struct gk_tensor *); \
    GK_API struct gk_tensor * gk_##name##_inplace(struct gk_ctx *, struct gk_tensor *)

GK_UNARY(abs);
GK_UNARY(sgn);
GK_UNARY(neg);
GK_UNARY(step);
GK_UNARY(tanh);
GK_UNARY(elu);
GK_UNARY(relu);
GK_UNARY(sigmoid);
GK_UNARY(gelu);
GK_UNARY(gelu_quick);
GK_UNARY(gelu_erf);
GK_UNARY(silu);
GK_UNARY(hardswish);
GK_UNARY(hardsigmoid);
GK_UNARY(exp);
GK_UNARY(expm1);
GK_UNARY(softplus);
GK_UNARY(floor);
GK_UNARY(ceil);
GK_UNARY(round);
GK_UNARY(trunc);

GK_UNARY(sqr);
GK_UNARY(sqrt);
GK_UNARY(log);
GK_UNARY(sin);
GK_UNARY(cos);

#undef GK_UNARY

GK_API struct gk_tensor * gk_unary(struct gk_ctx *, struct gk_tensor *, enum gk_unary_op);
GK_API struct gk_tensor * gk_unary_inplace(struct gk_ctx *, struct gk_tensor *, enum gk_unary_op);

// leaky_relu carries its slope, xielu its four constants; neither fits the
// plain unary builder's shape.
GK_API struct gk_tensor * gk_leaky_relu(struct gk_ctx *, struct gk_tensor *,
                                        float negative_slope, bool inplace);
GK_API struct gk_tensor * gk_xielu(struct gk_ctx *, struct gk_tensor *,
                                   float alpha_n, float alpha_p, float beta, float eps);

// `swapped` picks which half of a split row is the gate. With a separate gate
// tensor (gk_glu_split) there is nothing to swap.
GK_API struct gk_tensor * gk_glu(struct gk_ctx *, struct gk_tensor *, enum gk_glu_op, bool swapped);
GK_API struct gk_tensor * gk_glu_split(struct gk_ctx *, struct gk_tensor * a,
                                       struct gk_tensor * gate, enum gk_glu_op);
// the gpt-oss variant: clamped operands, a sigmoid with a temperature, +1 on the gate
GK_API struct gk_tensor * gk_swiglu_oai(struct gk_ctx *, struct gk_tensor * a,
                                        struct gk_tensor * gate, float alpha, float limit);

GK_API struct gk_tensor * gk_scale        (struct gk_ctx *, struct gk_tensor *, float s);
GK_API struct gk_tensor * gk_scale_inplace(struct gk_ctx *, struct gk_tensor *, float s);
GK_API struct gk_tensor * gk_scale_bias   (struct gk_ctx *, struct gk_tensor *, float s, float b);
GK_API struct gk_tensor * gk_clamp        (struct gk_ctx *, struct gk_tensor *, float min, float max);

GK_API struct gk_tensor * gk_norm            (struct gk_ctx *, struct gk_tensor *, float eps);
GK_API struct gk_tensor * gk_norm_inplace    (struct gk_ctx *, struct gk_tensor *, float eps);
GK_API struct gk_tensor * gk_rms_norm        (struct gk_ctx *, struct gk_tensor *, float eps);
GK_API struct gk_tensor * gk_rms_norm_inplace(struct gk_ctx *, struct gk_tensor *, float eps);
GK_API struct gk_tensor * gk_l2_norm         (struct gk_ctx *, struct gk_tensor *, float eps);
GK_API struct gk_tensor * gk_group_norm      (struct gk_ctx *, struct gk_tensor *, int n_groups, float eps);

GK_API struct gk_tensor * gk_mul_mat   (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b);
GK_API struct gk_tensor * gk_mul_mat_id(struct gk_ctx *, struct gk_tensor * as,
                                        struct gk_tensor * b, struct gk_tensor * ids);
GK_API struct gk_tensor * gk_out_prod   (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b);

GK_API struct gk_tensor * gk_reshape   (struct gk_ctx *, struct gk_tensor *, struct gk_tensor *);
GK_API struct gk_tensor * gk_reshape_1d(struct gk_ctx *, struct gk_tensor *, int64_t);
GK_API struct gk_tensor * gk_reshape_2d(struct gk_ctx *, struct gk_tensor *, int64_t, int64_t);
GK_API struct gk_tensor * gk_reshape_3d(struct gk_ctx *, struct gk_tensor *, int64_t, int64_t, int64_t);
GK_API struct gk_tensor * gk_reshape_4d(struct gk_ctx *, struct gk_tensor *, int64_t, int64_t, int64_t, int64_t);

GK_API struct gk_tensor * gk_view_1d(struct gk_ctx *, struct gk_tensor *,
                                     int64_t ne0, size_t offset);
GK_API struct gk_tensor * gk_view_2d(struct gk_ctx *, struct gk_tensor *,
                                     int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
GK_API struct gk_tensor * gk_view_3d(struct gk_ctx *, struct gk_tensor *,
                                     int64_t ne0, int64_t ne1, int64_t ne2,
                                     size_t nb1, size_t nb2, size_t offset);
GK_API struct gk_tensor * gk_view_4d(struct gk_ctx *, struct gk_tensor *,
                                     int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                     size_t nb1, size_t nb2, size_t nb3, size_t offset);

GK_API struct gk_tensor * gk_permute  (struct gk_ctx *, struct gk_tensor *, int, int, int, int);
GK_API struct gk_tensor * gk_transpose(struct gk_ctx *, struct gk_tensor *);

GK_API struct gk_tensor * gk_cont   (struct gk_ctx *, struct gk_tensor *);
GK_API struct gk_tensor * gk_cont_2d(struct gk_ctx *, struct gk_tensor *, int64_t, int64_t);
GK_API struct gk_tensor * gk_cont_3d(struct gk_ctx *, struct gk_tensor *, int64_t, int64_t, int64_t);
GK_API struct gk_tensor * gk_cont_4d(struct gk_ctx *, struct gk_tensor *, int64_t, int64_t, int64_t, int64_t);

GK_API struct gk_tensor * gk_cpy (struct gk_ctx *, struct gk_tensor *, struct gk_tensor *);
GK_API struct gk_tensor * gk_cast(struct gk_ctx *, struct gk_tensor *, enum gk_type);
GK_API struct gk_tensor * gk_dup (struct gk_ctx *, struct gk_tensor *);

GK_API struct gk_tensor * gk_get_rows (struct gk_ctx *, struct gk_tensor *, struct gk_tensor *);
GK_API struct gk_tensor * gk_repeat   (struct gk_ctx *, struct gk_tensor *, struct gk_tensor *);
GK_API struct gk_tensor * gk_repeat_4d(struct gk_ctx *, struct gk_tensor *,
                                       int64_t, int64_t, int64_t, int64_t);
GK_API struct gk_tensor * gk_concat   (struct gk_ctx *, struct gk_tensor *, struct gk_tensor *, int dim);

GK_API struct gk_tensor * gk_soft_max    (struct gk_ctx *, struct gk_tensor *);
GK_API struct gk_tensor * gk_soft_max_ext(struct gk_ctx *, struct gk_tensor * a,
                                          struct gk_tensor * mask, float scale, float max_bias);

// Attaches per-head sink logits to an existing soft_max node: each head gets
// one virtual position that absorbs probability mass without appearing in the
// output row.
GK_API void gk_soft_max_add_sinks(struct gk_tensor * a, struct gk_tensor * sinks);

GK_API struct gk_tensor * gk_diag_mask_inf        (struct gk_ctx *, struct gk_tensor *, int n_past);
GK_API struct gk_tensor * gk_diag_mask_inf_inplace(struct gk_ctx *, struct gk_tensor *, int n_past);
GK_API struct gk_tensor * gk_diag_mask_zero       (struct gk_ctx *, struct gk_tensor *, int n_past);

GK_API struct gk_tensor * gk_rope    (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * pos,
                                      int n_dims, int mode);
GK_API struct gk_tensor * gk_rope_ext(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * pos,
                                      struct gk_tensor * freq_factors,
                                      int n_dims, int mode, int n_ctx_orig,
                                      float freq_base, float freq_scale, float ext_factor,
                                      float attn_factor, float beta_fast, float beta_slow);

// The multi-axis rope for vision models: `sections` says how many cos/sin
// pairs each position axis (time, height, width, extra) owns, and `pos`
// carries four position ids per token. Mode is MROPE, IMROPE or VISION.
GK_API struct gk_tensor * gk_rope_multi(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * pos,
                                      struct gk_tensor * freq_factors,
                                      int n_dims, int sections[GK_MROPE_SECTIONS], int mode,
                                      int n_ctx_orig,
                                      float freq_base, float freq_scale, float ext_factor,
                                      float attn_factor, float beta_fast, float beta_slow);

// --------------------------------------------------------------------------
// fused attention
//
//   q [DK, n_batch, n_head,    ne3]
//   k [DK, n_kv,    n_head_kv, ne3]
//   v [DV, n_kv,    n_head_kv, ne3]   (not transposed)
//   mask f16 [n_kv, >=n_batch, ...]   (optional; broadcasts over heads)
//   result [DV, n_head, n_batch, ne3] (permuted)
//
// K and V may be quantized; the query row is converted to K's dot operand
// type once and reused across the whole sequence.
// --------------------------------------------------------------------------

GK_API struct gk_tensor * gk_flash_attn_ext(struct gk_ctx *,
                                            struct gk_tensor * q, struct gk_tensor * k,
                                            struct gk_tensor * v, struct gk_tensor * mask,
                                            float scale, float max_bias, float logit_softcap);

GK_API void         gk_flash_attn_ext_set_prec (struct gk_tensor * a, enum gk_prec prec);
GK_API enum gk_prec gk_flash_attn_ext_get_prec (const struct gk_tensor * a);
GK_API void         gk_flash_attn_ext_add_sinks(struct gk_tensor * a, struct gk_tensor * sinks);

// --------------------------------------------------------------------------
// recurrent layers
//
// These all share a shape convention: heads of size S over a token sequence,
// a carried [S, S] state per head per sequence, and an output that packs the
// per-token result rows first and the final states after them. The caller
// slices the two apart with views.
// --------------------------------------------------------------------------

// the short causal conv a state-space block starts with:
// sx [d_conv - 1 + n_t, d_inner, n_seqs], c [d_conv, d_inner]
GK_API struct gk_tensor * gk_ssm_conv(struct gk_ctx *, struct gk_tensor * sx, struct gk_tensor * c);

// the Mamba selective scan; A [1, n_head] (Mamba-2) or [d_state, n_head] (Mamba-1)
GK_API struct gk_tensor * gk_ssm_scan(struct gk_ctx *, struct gk_tensor * s, struct gk_tensor * x,
                                      struct gk_tensor * dt, struct gk_tensor * A,
                                      struct gk_tensor * B, struct gk_tensor * C,
                                      struct gk_tensor * ids);

GK_API struct gk_tensor * gk_rwkv_wkv6(struct gk_ctx *, struct gk_tensor * k, struct gk_tensor * v,
                                       struct gk_tensor * r, struct gk_tensor * tf,
                                       struct gk_tensor * td, struct gk_tensor * state);

GK_API struct gk_tensor * gk_rwkv_wkv7(struct gk_ctx *, struct gk_tensor * r, struct gk_tensor * w,
                                       struct gk_tensor * k, struct gk_tensor * v,
                                       struct gk_tensor * a, struct gk_tensor * b,
                                       struct gk_tensor * state);

GK_API struct gk_tensor * gk_gated_linear_attn(struct gk_ctx *, struct gk_tensor * k,
                                       struct gk_tensor * v, struct gk_tensor * q,
                                       struct gk_tensor * g, struct gk_tensor * state,
                                       float scale);

// the gated delta rule (Qwen3-Next and KDA): g is [1,...] for a scalar gate
// or [S_v,...] for a per-channel one. The output carries K state snapshots,
// most recent first.
GK_API struct gk_tensor * gk_gated_delta_net(struct gk_ctx *, struct gk_tensor * q,
                                       struct gk_tensor * k, struct gk_tensor * v,
                                       struct gk_tensor * g, struct gk_tensor * beta,
                                       struct gk_tensor * state, int64_t K);

// forward substitution: solves A x = b for lower-triangular A
GK_API struct gk_tensor * gk_solve_tri(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                       bool left, bool lower, bool uni);

// the DeepSeek sparse-attention indexer score pass
GK_API struct gk_tensor * gk_lightning_indexer(struct gk_ctx *, struct gk_tensor * q,
                                       struct gk_tensor * k, struct gk_tensor * weights,
                                       struct gk_tensor * mask);

// DeepSeek V4 hyper-connections: the combined-softmax mixing weights, the
// stream-collapse before a block, and the stream-redistribute after it
GK_API struct gk_tensor * gk_dsv4_hc_comb(struct gk_ctx *, struct gk_tensor * mixes,
                                       struct gk_tensor * scale, struct gk_tensor * base,
                                       float eps, int32_t n_iter);
GK_API struct gk_tensor * gk_dsv4_hc_pre (struct gk_ctx *, struct gk_tensor * x,
                                       struct gk_tensor * weights);
GK_API struct gk_tensor * gk_dsv4_hc_post(struct gk_ctx *, struct gk_tensor * x,
                                       struct gk_tensor * residual, struct gk_tensor * post,
                                       struct gk_tensor * comb);

// --------------------------------------------------------------------------
// custom ops - a caller-supplied kernel run as a graph node
// --------------------------------------------------------------------------

typedef void (*gk_custom1_op_t)(struct gk_tensor * dst, const struct gk_tensor * a,
                                int ith, int nth, void * userdata);
typedef void (*gk_custom2_op_t)(struct gk_tensor * dst, const struct gk_tensor * a,
                                const struct gk_tensor * b, int ith, int nth, void * userdata);
typedef void (*gk_custom3_op_t)(struct gk_tensor * dst, const struct gk_tensor * a,
                                const struct gk_tensor * b, const struct gk_tensor * c,
                                int ith, int nth, void * userdata);
typedef void (*gk_custom_op_t) (struct gk_tensor * dst, int ith, int nth, void * userdata);

#define GK_N_TASKS_MAX (-1)

GK_API struct gk_tensor * gk_map_custom1(struct gk_ctx *, struct gk_tensor * a,
                                         gk_custom1_op_t fun, int n_tasks, void * userdata);
GK_API struct gk_tensor * gk_map_custom1_inplace(struct gk_ctx *, struct gk_tensor * a,
                                         gk_custom1_op_t fun, int n_tasks, void * userdata);
GK_API struct gk_tensor * gk_map_custom2(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         gk_custom2_op_t fun, int n_tasks, void * userdata);
GK_API struct gk_tensor * gk_map_custom2_inplace(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         gk_custom2_op_t fun, int n_tasks, void * userdata);
GK_API struct gk_tensor * gk_map_custom3(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         struct gk_tensor * c,
                                         gk_custom3_op_t fun, int n_tasks, void * userdata);
GK_API struct gk_tensor * gk_map_custom3_inplace(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         struct gk_tensor * c,
                                         gk_custom3_op_t fun, int n_tasks, void * userdata);

GK_API struct gk_tensor * gk_custom_4d(struct gk_ctx *, enum gk_type type,
                                       int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                       struct gk_tensor ** args, int n_args,
                                       gk_custom_op_t fun, int n_tasks, void * userdata);
GK_API struct gk_tensor * gk_custom_inplace(struct gk_ctx *, struct gk_tensor * a,
                                       struct gk_tensor ** args, int n_args,
                                       gk_custom_op_t fun, int n_tasks, void * userdata);

GK_API struct gk_tensor * gk_sum     (struct gk_ctx *, struct gk_tensor *);
GK_API struct gk_tensor * gk_sum_rows(struct gk_ctx *, struct gk_tensor *);
GK_API struct gk_tensor * gk_cumsum  (struct gk_ctx *, struct gk_tensor *);
GK_API struct gk_tensor * gk_mean    (struct gk_ctx *, struct gk_tensor *);
GK_API struct gk_tensor * gk_argmax  (struct gk_ctx *, struct gk_tensor *);
GK_API struct gk_tensor * gk_argsort (struct gk_ctx *, struct gk_tensor *, enum gk_sort_order);

// Two ways to the same set: top_k is a real kernel (a partial selection, its
// indices in no promised order), argsort_top_k is a full sort viewed down to k
// (descending, deterministic order).
GK_API struct gk_tensor * gk_top_k        (struct gk_ctx *, struct gk_tensor *, int k);
GK_API struct gk_tensor * gk_argsort_top_k(struct gk_ctx *, struct gk_tensor *, int k);

// count of positions where a and b hold the same value, as a single i64
GK_API struct gk_tensor * gk_count_equal(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b);

// --------------------------------------------------------------------------
// data placement
// --------------------------------------------------------------------------

// dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]] - a per-row bias picked
// by an index, the shape mixture-of-experts biases arrive in.
GK_API struct gk_tensor * gk_add_id(struct gk_ctx *, struct gk_tensor * a,
                                    struct gk_tensor * b, struct gk_tensor * ids);

// copy of a with the window described by (nb1, nb2, nb3, offset) += b
GK_API struct gk_tensor * gk_acc        (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         size_t nb1, size_t nb2, size_t nb3, size_t offset);
GK_API struct gk_tensor * gk_acc_inplace(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         size_t nb1, size_t nb2, size_t nb3, size_t offset);

// copy of a with the window described by (nb1, nb2, nb3, offset) = b
GK_API struct gk_tensor * gk_set        (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         size_t nb1, size_t nb2, size_t nb3, size_t offset);
GK_API struct gk_tensor * gk_set_inplace(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         size_t nb1, size_t nb2, size_t nb3, size_t offset);
GK_API struct gk_tensor * gk_set_1d     (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         size_t offset);
GK_API struct gk_tensor * gk_set_1d_inplace(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                         size_t offset);

// Writes the rows of b into a at the row indices c names (i32 or i64), and is
// how a KV cache is updated in place. The destination keeps its own type: a
// quantized cache quantizes the incoming row here. Returns a view of a.
GK_API struct gk_tensor * gk_set_rows(struct gk_ctx *, struct gk_tensor * a,
                                      struct gk_tensor * b, struct gk_tensor * c);

// a vector placed on the diagonal of a square matrix of zeroes
GK_API struct gk_tensor * gk_diag(struct gk_ctx *, struct gk_tensor *);

// zero everything outside the named triangle
GK_API struct gk_tensor * gk_tri(struct gk_ctx *, struct gk_tensor *, enum gk_tri_type type);

GK_API struct gk_tensor * gk_fill        (struct gk_ctx *, struct gk_tensor *, float c);
GK_API struct gk_tensor * gk_fill_inplace(struct gk_ctx *, struct gk_tensor *, float c);

// [start, stop) in steps of `step`, f32, no source tensor
GK_API struct gk_tensor * gk_arange(struct gk_ctx *, float start, float stop, float step);

// rotate each dimension by a per-dimension shift, wrapping around
GK_API struct gk_tensor * gk_roll(struct gk_ctx *, struct gk_tensor *,
                                  int shift0, int shift1, int shift2, int shift3);

// the sinusoidal timestep embedding a diffusion model conditions on
GK_API struct gk_tensor * gk_timestep_embedding(struct gk_ctx *, struct gk_tensor * timesteps,
                                                int dim, int max_period);

// --------------------------------------------------------------------------
// padding and resampling
// --------------------------------------------------------------------------

GK_API struct gk_tensor * gk_pad    (struct gk_ctx *, struct gk_tensor *,
                                     int p0, int p1, int p2, int p3);
GK_API struct gk_tensor * gk_pad_ext(struct gk_ctx *, struct gk_tensor *,
                                     int lp0, int rp0, int lp1, int rp1,
                                     int lp2, int rp2, int lp3, int rp3);
GK_API struct gk_tensor * gk_pad_circular    (struct gk_ctx *, struct gk_tensor *,
                                     int p0, int p1, int p2, int p3);
GK_API struct gk_tensor * gk_pad_ext_circular(struct gk_ctx *, struct gk_tensor *,
                                     int lp0, int rp0, int lp1, int rp1,
                                     int lp2, int rp2, int lp3, int rp3);
GK_API struct gk_tensor * gk_pad_reflect_1d(struct gk_ctx *, struct gk_tensor *, int p0, int p1);

// nearest / bilinear / bicubic resampling of the first two dimensions;
// `mode` is a gk_scale_mode, optionally OR'd with gk_scale_flag bits
GK_API struct gk_tensor * gk_upscale    (struct gk_ctx *, struct gk_tensor *,
                                         int scale_factor, enum gk_scale_mode mode);
GK_API struct gk_tensor * gk_interpolate(struct gk_ctx *, struct gk_tensor *,
                                         int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                         uint32_t mode);

// --------------------------------------------------------------------------
// convolution
//
// Two routes to the same arithmetic. The im2col route unrolls input patches
// into rows and hands the reduction to matmul - the composite builders
// (gk_conv_1d, gk_conv_2d, ...) produce that little subgraph. The _direct ops
// are single kernels that walk the patches in place.
// --------------------------------------------------------------------------

// patches of b unrolled against kernel a: [N, IC, IH, IW] -> [N, OH, OW, IC*KH*KW]
GK_API struct gk_tensor * gk_im2col(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                    int s0, int s1, int p0, int p1, int d0, int d1,
                                    bool is_2D, enum gk_type dst_type);
GK_API struct gk_tensor * gk_im2col_3d(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                    int64_t IC, int s0, int s1, int s2, int p0, int p1, int p2,
                                    int d0, int d1, int d2, enum gk_type dst_type);

GK_API struct gk_tensor * gk_conv_1d      (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                           int s0, int p0, int d0);
GK_API struct gk_tensor * gk_conv_1d_ph   (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                           int s, int d);
GK_API struct gk_tensor * gk_conv_1d_dw   (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                           int s0, int p0, int d0);
GK_API struct gk_tensor * gk_conv_1d_dw_ph(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                           int s0, int d0);

GK_API struct gk_tensor * gk_conv_2d        (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                             int s0, int s1, int p0, int p1, int d0, int d1);
GK_API struct gk_tensor * gk_conv_2d_sk_p0  (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b);
GK_API struct gk_tensor * gk_conv_2d_s1_ph  (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b);
GK_API struct gk_tensor * gk_conv_2d_dw     (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                             int s0, int s1, int p0, int p1, int d0, int d1);
GK_API struct gk_tensor * gk_conv_2d_direct (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                             int s0, int s1, int p0, int p1, int d0, int d1);
GK_API struct gk_tensor * gk_conv_2d_dw_direct(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                             int s0, int s1, int p0, int p1, int d0, int d1);

GK_API struct gk_tensor * gk_conv_3d       (struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                            int64_t IC, int s0, int s1, int s2,
                                            int p0, int p1, int p2, int d0, int d1, int d2);
GK_API struct gk_tensor * gk_conv_3d_direct(struct gk_ctx *, struct gk_tensor * a, struct gk_tensor * b,
                                            int s0, int s1, int s2, int p0, int p1, int p2,
                                            int d0, int d1, int d2,
                                            int n_channels, int n_batch, int n_channels_out);

GK_API struct gk_tensor * gk_conv_transpose_1d(struct gk_ctx *, struct gk_tensor * a,
                                               struct gk_tensor * b, int s0, int p0, int d0);
GK_API struct gk_tensor * gk_conv_transpose_2d_p0(struct gk_ctx *, struct gk_tensor * a,
                                               struct gk_tensor * b, int stride);

GK_API struct gk_tensor * gk_pool_1d(struct gk_ctx *, struct gk_tensor *,
                                     enum gk_op_pool op, int k0, int s0, int p0);
GK_API struct gk_tensor * gk_pool_2d(struct gk_ctx *, struct gk_tensor *,
                                     enum gk_op_pool op, int k0, int k1, int s0, int s1,
                                     float p0, float p1);

// non-overlapping windows of a [C, W, H] map, zero-padded to a multiple of w,
// and the inverse
GK_API struct gk_tensor * gk_win_part  (struct gk_ctx *, struct gk_tensor *, int w);
GK_API struct gk_tensor * gk_win_unpart(struct gk_ctx *, struct gk_tensor *, int w0, int h0, int w);

// --------------------------------------------------------------------------
// evaluation
//
// Walks the graph in order and evaluates every node on the CPU. Each node's
// `data` must already point at storage - graph building leaves it null, so
// something has to assign buffers first.
// --------------------------------------------------------------------------

// `n_threads` <= 0 means one per core. A pool is created and torn down around
// the call; use gk_graph_compute_with_pool to keep one across many graphs,
// which is what an inference loop should do.
GK_API enum gk_status gk_graph_compute(struct gk_cgraph * graph, int n_threads);

struct gk_pool;
GK_API enum gk_status gk_graph_compute_with_pool(struct gk_cgraph * graph,
                                                 struct gk_pool * pool);

// --------------------------------------------------------------------------
// backends
//
// Three layers, and the split between the first two is the one that matters:
//
//   buffer type  a kind of memory - host, or a particular device's.
//   buffer       one allocation of that kind; tensors point into it.
//   backend      something that evaluates a graph.
//
// A tensor records which buffer it lives in, separately from which backend
// computes on it, because those are not always the same thing - pinned host
// memory is allocated by a device but read by the CPU, and a shared weight is
// read by several devices. Keeping the two apart is what lets a graph be
// placed across devices without any kernel knowing about it.
// --------------------------------------------------------------------------

typedef struct gk_backend_buffer_type * gk_backend_buffer_type_t;
typedef struct gk_backend_buffer      * gk_backend_buffer_t;
typedef struct gk_backend             * gk_backend_t;

GK_API const char *        gk_backend_buft_name          (gk_backend_buffer_type_t buft);
GK_API gk_backend_buffer_t gk_backend_buft_alloc_buffer  (gk_backend_buffer_type_t buft, size_t size);
GK_API size_t              gk_backend_buft_get_alignment (gk_backend_buffer_type_t buft);
GK_API size_t              gk_backend_buft_get_alloc_size(gk_backend_buffer_type_t buft,
                                                          const struct gk_tensor * tensor);
GK_API bool                gk_backend_buft_is_host       (gk_backend_buffer_type_t buft);

GK_API void                     gk_backend_buffer_free         (gk_backend_buffer_t buffer);
GK_API void *                   gk_backend_buffer_get_base     (gk_backend_buffer_t buffer);
GK_API size_t                   gk_backend_buffer_get_size     (gk_backend_buffer_t buffer);
GK_API gk_backend_buffer_type_t gk_backend_buffer_get_type     (gk_backend_buffer_t buffer);
GK_API size_t                   gk_backend_buffer_get_alignment(gk_backend_buffer_t buffer);
GK_API bool                     gk_backend_buffer_is_host      (gk_backend_buffer_t buffer);
GK_API void                     gk_backend_buffer_clear        (gk_backend_buffer_t buffer, uint8_t value);
GK_API void                     gk_backend_buffer_init_tensor  (gk_backend_buffer_t buffer,
                                                                struct gk_tensor * tensor);

// The only sanctioned way to move data in and out of a tensor. Writing through
// `tensor->data` happens to work for the CPU backend and will not for others.
GK_API void gk_backend_tensor_set(struct gk_tensor * tensor, const void * data,
                                  size_t offset, size_t size);
GK_API void gk_backend_tensor_get(const struct gk_tensor * tensor, void * data,
                                  size_t offset, size_t size);

// Fills a region of a tensor with one byte value, without staging it through
// host memory on a device that can do it itself.
GK_API void gk_backend_tensor_memset(struct gk_tensor * tensor, uint8_t value,
                                     size_t offset, size_t size);

// Copies one tensor into another of the same shape and type, wherever the two
// live. Same memory: nothing happens. Two buffers on one device: the device
// does it. Anything else: through a host bounce.
GK_API void gk_backend_tensor_copy(const struct gk_tensor * src, struct gk_tensor * dst);

GK_API const char *              gk_backend_name                   (gk_backend_t backend);
GK_API void                      gk_backend_free                   (gk_backend_t backend);
GK_API gk_backend_buffer_type_t  gk_backend_get_default_buffer_type(gk_backend_t backend);
// Evaluates the graph and waits for it. When this returns, every result the
// graph produced is readable - by the host, by another backend, by anything.
// A device backend queues its work on a stream that nothing else is ordered
// against, so without the wait a caller reading a result would race the
// kernels still writing it.
GK_API enum gk_status            gk_backend_graph_compute          (gk_backend_t backend,
                                                                    struct gk_cgraph * graph);

// The same, without the wait, for a caller that will synchronize itself. Only
// worth reaching for when several graphs are queued back to back; anything
// that reads a result in between wants the form above.
GK_API enum gk_status            gk_backend_graph_compute_async    (gk_backend_t backend,
                                                                    struct gk_cgraph * graph);
GK_API bool                      gk_backend_supports_op            (gk_backend_t backend,
                                                                    const struct gk_tensor * op);

// Whether this backend can read tensors that live in that memory directly.
// A device backend says yes to its own memory and to any host memory it can
// map; the CPU says yes only to host memory.
GK_API bool                      gk_backend_supports_buft          (gk_backend_t backend,
                                                                    gk_backend_buffer_type_t buft);

// Whether it is worth moving an op here even though its operands live
// elsewhere - true for the large matmuls of a prompt pass, where the copy is
// paid back many times over, and false for everything cheap.
GK_API bool                      gk_backend_offload_op             (gk_backend_t backend,
                                                                    const struct gk_tensor * op);

// Waits for everything this backend has been asked to do. Device backends
// queue work and return immediately; nothing may read their results before
// this returns.
GK_API void                      gk_backend_synchronize            (gk_backend_t backend);

// Transfers that may be queued behind the backend's own work rather than
// completing before they return. A caller that then reads host memory has to
// synchronize first; one that only feeds the same backend does not.
GK_API void gk_backend_tensor_set_async(gk_backend_t backend, struct gk_tensor * tensor,
                                        const void * data, size_t offset, size_t size);
GK_API void gk_backend_tensor_get_async(gk_backend_t backend, const struct gk_tensor * tensor,
                                        void * data, size_t offset, size_t size);

// The CPU backend. It owns a thread pool for its lifetime, so a run of graphs
// shares one set of threads.
GK_API gk_backend_t             gk_backend_cpu_init(int n_threads); // <= 0: one per core
GK_API int                      gk_backend_cpu_n_threads(gk_backend_t backend);
GK_API void                     gk_backend_cpu_set_n_threads(gk_backend_t backend, int n_threads);
GK_API gk_backend_buffer_type_t gk_backend_cpu_buffer_type(void);

// Wraps memory the caller owns - a memory-mapped model file, typically, so
// weights are read without being copied. Freeing the buffer does not free it.
GK_API gk_backend_buffer_t gk_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

// --------------------------------------------------------------------------
// devices
//
// A device is a thing a backend can be created on: the host, or one GPU. The
// list is fixed at the first call - each compiled-in backend is asked what it
// can see, once - because a caller that enumerates devices is about to place a
// model across them and must not have the answer change underneath it.
//
// The split between this and the backend interface above is the same one the
// buffer/backend split makes: a device is a piece of hardware and its
// properties, a backend is one stream of work submitted to it. Two backends on
// one device share its memory and can hand tensors to each other without a
// copy, which is exactly what the scheduler needs to know and what a bare
// backend handle cannot tell it.
// --------------------------------------------------------------------------

typedef struct gk_device * gk_device_t;

enum gk_device_type {
    GK_DEVICE_TYPE_CPU  = 0,
    GK_DEVICE_TYPE_GPU  = 1, // discrete: its own memory, across a bus
    GK_DEVICE_TYPE_IGPU = 2, // integrated: the host's memory, no copy needed
};

GK_API int         gk_device_count(void);
GK_API gk_device_t gk_device_get    (int index);
GK_API gk_device_t gk_device_by_name(const char * name);
GK_API gk_device_t gk_device_by_type(enum gk_device_type type); // the first of that type

GK_API const char *        gk_device_name       (gk_device_t dev); // "CUDA0", "Vulkan1", "CPU"
GK_API const char *        gk_device_description(gk_device_t dev); // what the driver calls it
GK_API const char *        gk_device_backend    (gk_device_t dev); // "CUDA", "ROCm", "Metal", ...
GK_API enum gk_device_type gk_device_type_of    (gk_device_t dev);
GK_API void                gk_device_memory     (gk_device_t dev, size_t * free, size_t * total);

// What this build chose for this device, rather than what the caller asked
// for: which vector instruction set the CPU kernels were compiled against,
// which compute capabilities a CUDA build carries code for. The values are
// strings because the set differs per backend and the consumer is a human
// reading a startup banner - a caller that needs a number should ask the
// specific question instead of parsing these.
//
// The list is terminated by an entry whose name is NULL, and the function
// never returns NULL: a device with nothing to report returns an empty list.
// Both the array and the strings in it are static and outlive any caller.
struct gk_feature {
    const char * name;
    const char * value;
};

GK_API const struct gk_feature * gk_device_features(gk_device_t dev);

GK_API gk_backend_t             gk_device_init_backend    (gk_device_t dev);
GK_API gk_backend_buffer_type_t gk_device_buffer_type     (gk_device_t dev);

// Memory the host can address that the device can read without a staging copy
// - pinned memory on a discrete GPU, plain host memory on an integrated one.
// NULL when the device has no such thing.
GK_API gk_backend_buffer_type_t gk_device_host_buffer_type(gk_device_t dev);

// Wraps host memory the caller owns, if the device can read it in place. NULL
// when it cannot, which tells a model loader to copy the weights instead of
// mapping them.
GK_API gk_backend_buffer_t gk_device_buffer_from_host_ptr(gk_device_t dev, void * ptr, size_t size);

GK_API bool gk_device_supports_op  (gk_device_t dev, const struct gk_tensor * op);
GK_API bool gk_device_supports_buft(gk_device_t dev, gk_backend_buffer_type_t buft);
GK_API bool gk_device_offload_op   (gk_device_t dev, const struct gk_tensor * op);

GK_API gk_device_t gk_backend_get_device    (gk_backend_t backend);
GK_API gk_device_t gk_backend_buft_get_device(gk_backend_buffer_type_t buft);

// --------------------------------------------------------------------------
// device backends
//
// Each is present only if it was compiled in; the ones that were not return
// NULL and report no devices, so a caller never has to be built differently
// from the library. Prefer the device list above to naming one of these
// directly - it is the same objects, discovered rather than assumed.
// --------------------------------------------------------------------------

GK_API gk_backend_t gk_backend_cuda_init  (int device); // CUDA, or ROCm in a HIP build
GK_API gk_backend_t gk_backend_metal_init (void);
GK_API gk_backend_t gk_backend_vulkan_init(int device);

// --------------------------------------------------------------------------
// graph allocator
//
// Assigns storage to a graph's nodes out of one buffer, reusing memory as
// soon as a tensor's last consumer has run. Without the reuse, a graph would
// need room for every intermediate at once; with it, a transformer needs only
// enough for the few that are live at any point.
//
// Usage is two phases: `reserve` measures a representative graph and sizes the
// buffer, then `alloc_graph` assigns pointers for each run. Re-reserving only
// happens when a graph needs more than the buffer holds.
// --------------------------------------------------------------------------

struct gk_gallocr;

GK_API struct gk_gallocr * gk_gallocr_new (gk_backend_buffer_type_t buft);
GK_API void                gk_gallocr_free(struct gk_gallocr * galloc);

GK_API bool   gk_gallocr_reserve        (struct gk_gallocr * galloc, struct gk_cgraph * graph);
GK_API bool   gk_gallocr_alloc_graph    (struct gk_gallocr * galloc, struct gk_cgraph * graph);
GK_API size_t gk_gallocr_get_buffer_size(struct gk_gallocr * galloc);

// The same allocator over several memories at once. Each node and leaf names
// which of the buffer types it belongs in, and the allocator keeps one buffer
// and one free list per type. This is what a graph split across devices needs:
// a node computed on the GPU has to be written into GPU memory, and the node
// after it that fell back to the CPU has to be written into host memory, in
// the same graph and in the same pass.
//
// The ids arrays are indexed by node/leaf position in the graph and hold an
// index into `bufts`. Passing NULL for either puts everything in buffer 0,
// which is exactly the single-memory case above.
GK_API struct gk_gallocr * gk_gallocr_new_n(gk_backend_buffer_type_t * bufts, int n_bufs);

GK_API bool gk_gallocr_reserve_n    (struct gk_gallocr * galloc, struct gk_cgraph * graph,
                                     const int * node_buffer_ids, const int * leaf_buffer_ids);
GK_API bool gk_gallocr_alloc_graph_n(struct gk_gallocr * galloc, struct gk_cgraph * graph,
                                     const int * node_buffer_ids, const int * leaf_buffer_ids);

GK_API size_t gk_gallocr_get_buffer_size_n(struct gk_gallocr * galloc, int buffer_id);

// --------------------------------------------------------------------------
// scheduler
//
// Places a graph across several backends and runs it: assigns each node a
// backend, cuts the node list where the backend changes, and moves tensors
// across the boundaries.
//
// With a single backend this degenerates to one split and does nothing the
// graph allocator does not already do. It earns its place once a second device
// exists - which is also the point at which its assignment quality starts to
// matter, since every extra split is a synchronisation point.
// --------------------------------------------------------------------------

struct gk_sched;

// Backends are given in priority order: an op that several of them support
// goes to the earliest one, so a list of (GPU, CPU) offloads what it can and
// falls back for the rest. `bufts`, when given, overrides the memory each
// backend's tensors are allocated in - a device backend that should compute
// out of pinned host memory is the case that needs it. `op_offload` allows an
// op whose weight the caller left in host memory to be pulled onto a device
// that would run it faster, at the cost of copying that weight there; an op
// with no weight is never moved, because there is nothing bounded to carry.
GK_API struct gk_sched * gk_sched_new(gk_backend_t * backends, int n_backends);
GK_API struct gk_sched * gk_sched_new_ext(gk_backend_t * backends,
                                          gk_backend_buffer_type_t * bufts,
                                          int n_backends, bool op_offload);
GK_API void              gk_sched_free(struct gk_sched * sched);

// A graph is placed across the backends once, and both entry points below do
// it: gk_sched_alloc_graph places and allocates, and gk_sched_graph_compute
// places and allocates only if that has not already happened for this graph.
//
// This matters to a caller that writes graph inputs, because the only place to
// write them is the storage allocation hands out - so the order is allocate,
// write, compute, and the compute must not move the work away from what was
// written. Placing twice would: the pass reads where tensors already live, and
// after an allocation that answer is different for every node in the graph.
GK_API bool           gk_sched_reserve      (struct gk_sched * sched, struct gk_cgraph * graph);
GK_API bool           gk_sched_alloc_graph  (struct gk_sched * sched, struct gk_cgraph * graph);
GK_API enum gk_status gk_sched_graph_compute(struct gk_sched * sched, struct gk_cgraph * graph);
GK_API void           gk_sched_synchronize  (struct gk_sched * sched);

// Drops the current placement. Anything the last graph's tensors pointed at is
// dangling afterwards, which is the price of reusing one buffer set across
// graphs; callers build a new graph rather than keeping the old one. This is
// also what ends the arrangement above: a new graph needs a reset before it,
// or it inherits the last one's storage.
GK_API void gk_sched_reset(struct gk_sched * sched);

GK_API int    gk_sched_n_splits       (const struct gk_sched * sched);
GK_API int    gk_sched_n_backends     (const struct gk_sched * sched);
GK_API size_t gk_sched_get_buffer_size(struct gk_sched * sched, int backend_index);

// Forces a node onto one backend, overriding the placement pass. Rarely right
// - the pass has more information than a caller usually does - but the engine
// uses it to keep specific tensors off a device.
GK_API void         gk_sched_set_tensor_backend(struct gk_sched * sched, struct gk_tensor * node,
                                                gk_backend_t backend);
GK_API gk_backend_t gk_sched_get_tensor_backend(struct gk_sched * sched, struct gk_tensor * node);

// Called with each node twice: `ask` to find out whether the caller wants to
// see it, then again once it has been computed. Returning false the second
// time cancels the rest of the graph. Installing one forces node-at-a-time
// evaluation, so nothing in a hot path should have it set.
typedef bool (*gk_sched_eval_callback)(struct gk_tensor * t, bool ask, void * user_data);

GK_API void gk_sched_set_eval_callback(struct gk_sched * sched,
                                       gk_sched_eval_callback callback, void * user_data);

#ifdef __cplusplus
}
#endif
