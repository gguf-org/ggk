#pragma once

// What every kernel in this directory shares: how a tensor is described to the
// device, how an element is read out of one, and the two reductions.
//
// The description is a flat struct rather than the gk_tensor the host holds,
// because a kernel needs only the numbers - a pointer, four extents, four byte
// strides and a type - and copying them into the launch is cheaper than
// chasing host pointers that mean nothing on the device.
//
// Strides are carried and honoured everywhere. It would be simpler to demand
// contiguous operands and make the caller materialise anything else, and it
// would also mean a `gk_cont` before half the ops in a transformer graph: a
// permuted view is the normal shape of attention's operands, not an edge case.

#include "gk_cuda_dequant.cuh"

#include <stdio.h>

// The most CUDA devices gk will track. Here rather than beside the device
// table it sizes, because the matmul dispatcher caches a per-device answer of
// its own and the two have to agree on the bound.
#define GK_CUDA_MAX_DEVICES 16

#include <chrono>

// --------------------------------------------------------------------------
// error handling
//
// A failed launch or a failed allocation is not recoverable in any useful
// sense here - the graph is half-run and the device state is unknown - so it
// is reported loudly and the caller is told the graph failed.
// --------------------------------------------------------------------------

#define GK_CUDA_CHECK(expr)                                                     \
    do {                                                                        \
        const gkError_t err_ = (expr);                                          \
        if (err_ != gkSuccess) {                                                \
            fprintf(stderr, "gk %s: %s failed at %s:%d: %s\n",                  \
                    GK_CUDA_BACKEND_NAME, #expr, __FILE__, __LINE__,            \
                    gkGetErrorString(err_));                                    \
        }                                                                       \
    } while (0)

// --------------------------------------------------------------------------
// tensor views
// --------------------------------------------------------------------------

struct gk_tview {
    const char * data;
    int64_t      ne[4];
    int64_t      nb[4];
    int          type;
};

struct gk_tview_mut {
    char *  data;
    int64_t ne[4];
    int64_t nb[4];
    int     type;
};

// Byte address of row (i1,i2,i3).
static __device__ __forceinline__ const char * gk_cu_row(const gk_tview & t,
                                                         int64_t i1, int64_t i2, int64_t i3) {
    return t.data + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
}

static __device__ __forceinline__ char * gk_cu_row(const gk_tview_mut & t,
                                                   int64_t i1, int64_t i2, int64_t i3) {
    return t.data + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
}

// One element. Float types are read through their stride, which is what makes
// permuted operands work; a quantized row is packed by definition, so its
// elements are found by block arithmetic instead.
static __device__ __forceinline__ float gk_cu_get(const gk_tview & t,
                                                  int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * row = gk_cu_row(t, i1, i2, i3);

    switch (t.type) {
        case GKT_F32:  return *(const float *)    (row + i0 * t.nb[0]);
        case GKT_F16:  return __half2float(*(const __half *) (row + i0 * t.nb[0]));
        case GKT_BF16: return gk_cu_bf2f(*(const uint16_t *) (row + i0 * t.nb[0]));
        case GKT_I32:  return (float) *(const int32_t *) (row + i0 * t.nb[0]);
        case GKT_I64:  return (float) *(const int64_t *) (row + i0 * t.nb[0]);
        default:       return gk_cu_row_elem(row, t.type, i0);
    }
}

// The same read with the type known at compile time. `t.type` is ignored and
// TYPE is trusted, so a caller has to have dispatched on the runtime type to
// reach the matching instantiation - which is what GK_CU_MM_DISPATCH does.
template <int TYPE>
static __device__ __forceinline__ float gk_cu_get_t(const gk_tview & t,
                                                    int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * row = gk_cu_row(t, i1, i2, i3);

    // The float types keep their stride, which is what makes a permuted
    // operand work; a quantized row is packed by definition and is indexed by
    // block arithmetic instead. Same rule as the runtime version.
    if (TYPE == GKT_F32)  { return *(const float *)    (row + i0 * t.nb[0]); }
    if (TYPE == GKT_F16)  { return __half2float(*(const __half *) (row + i0 * t.nb[0])); }
    if (TYPE == GKT_BF16) { return gk_cu_bf2f(*(const uint16_t *) (row + i0 * t.nb[0])); }
    if (TYPE == GKT_I32)  { return (float) *(const int32_t *) (row + i0 * t.nb[0]); }
    if (TYPE == GKT_I64)  { return (float) *(const int64_t *) (row + i0 * t.nb[0]); }

    return gk_cu_row_elem_t<TYPE>(row, i0);
}

// Reading back what a kernel has just written. Only the float types can be a
// destination, so this needs none of the quantized paths above.
static __device__ __forceinline__ float gk_cu_get_mut(const gk_tview_mut & t,
                                                      int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * p = gk_cu_row(t, i1, i2, i3) + i0 * t.nb[0];

    switch (t.type) {
        case GKT_F16:  return __half2float(*(const __half *) p);
        case GKT_BF16: return gk_cu_bf2f(*(const uint16_t *) p);
        default:       return *(const float *) p;
    }
}

static __device__ __forceinline__ void gk_cu_set(const gk_tview_mut & t,
                                                 int64_t i0, int64_t i1, int64_t i2, int64_t i3,
                                                 float v) {
    char * p = gk_cu_row(t, i1, i2, i3) + i0 * t.nb[0];

    switch (t.type) {
        case GKT_F32:  *(float *)  p = v; break;
        case GKT_F16:  *(__half *) p = __float2half(v); break;
        case GKT_BF16: {
            // round-to-nearest-even on the way down, the same rule the codec
            // uses, so a value written here and read on the host agree
            uint32_t u;
            memcpy(&u, &v, sizeof(u));
            const uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
            *(uint16_t *) p = (uint16_t) (rounded >> 16);
            break;
        }
        case GKT_I32:  *(int32_t *) p = (int32_t) v; break;
        default: break; // filtered by supports_op
    }
}

// The (i1,i2,i3) of a flat row index, the decomposition every kernel that
// works a row at a time starts with.
static __device__ __forceinline__ void gk_cu_unrow(int64_t ir, const int64_t ne[4],
                                                   int64_t * i1, int64_t * i2, int64_t * i3) {
    *i1 = ir % ne[1];
    *i2 = (ir / ne[1]) % ne[2];
    *i3 = ir / (ne[1] * ne[2]);
}

// --------------------------------------------------------------------------
// reductions
//
// Warp-wide first, then across warps through shared memory. Blocks are always
// a whole number of warps, so the second stage never has a partial warp.
// --------------------------------------------------------------------------

static __device__ __forceinline__ float gk_cu_warp_sum(float x) {
#pragma unroll
    for (int offset = GK_WARP_SIZE / 2; offset > 0; offset >>= 1) {
#if defined(GK_USE_HIP)
        x += __shfl_xor(x, offset, GK_WARP_SIZE);
#else
        x += __shfl_xor_sync(0xffffffff, x, offset, GK_WARP_SIZE);
#endif
    }
    return x;
}

static __device__ __forceinline__ float gk_cu_warp_max(float x) {
#pragma unroll
    for (int offset = GK_WARP_SIZE / 2; offset > 0; offset >>= 1) {
#if defined(GK_USE_HIP)
        x = fmaxf(x, __shfl_xor(x, offset, GK_WARP_SIZE));
#else
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, GK_WARP_SIZE));
#endif
    }
    return x;
}

// Block-wide sum. `scratch` must hold one float per warp in the block.
static __device__ __forceinline__ float gk_cu_block_sum(float x, float * scratch) {
    const int lane = threadIdx.x % GK_WARP_SIZE;
    const int warp = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = (blockDim.x + GK_WARP_SIZE - 1) / GK_WARP_SIZE;

    x = gk_cu_warp_sum(x);

    // An xor reduction leaves every lane holding the warp's answer, so a
    // single-warp block is already done.
    if (n_warps == 1) {
        return x;
    }

    if (lane == 0) {
        scratch[warp] = x;
    }
    __syncthreads();

    float total = 0.0f;
    for (int i = 0; i < n_warps; ++i) {
        total += scratch[i];
    }
    return total;
}

static __device__ __forceinline__ float gk_cu_block_max(float x, float * scratch) {
    const int lane = threadIdx.x % GK_WARP_SIZE;
    const int warp = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = (blockDim.x + GK_WARP_SIZE - 1) / GK_WARP_SIZE;

    x = gk_cu_warp_max(x);

    if (n_warps == 1) {
        return x;
    }

    if (lane == 0) {
        scratch[warp] = x;
    }
    __syncthreads();

    float best = scratch[0];
    for (int i = 1; i < n_warps; ++i) {
        best = fmaxf(best, scratch[i]);
    }
    return best;
}

// The ALiBi slope for head h: the first power-of-two heads step down
// geometrically and any beyond interleave a second sequence. Both branches
// matter - a model with a non-power-of-two head count was trained against
// exactly this assignment.
static __device__ __forceinline__ float gk_cu_alibi_slope(float max_bias, int64_t h,
                                                          int64_t n_head_log2) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float m0 = powf(2.0f, -max_bias / (float) n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);

    return h < n_head_log2
        ? powf(m0, (float) (h + 1))
        : powf(m1, (float) (2 * (h - n_head_log2) + 1));
}

// --------------------------------------------------------------------------
// division by a divisor the launch already knows
//
// Index arithmetic is where a kernel that moves one element per thread spends
// itself. Decomposing a flat index into the dimensions it stands for is a
// division per dimension, and a 64-bit integer division has no hardware behind
// it - it is tens of instructions each. A kernel doing five of them per output
// element is doing several hundred instructions of arithmetic to move four
// bytes, and its cost stops having anything to do with the memory it touches.
//
// Every divisor in that arithmetic is a tensor extent: fixed for the whole
// launch and known on the host. That is exactly the case a multiply-and-shift
// replaces. The magic number below is Hacker's Delight's unsigned form, in the
// variant that folds the correction into two shifts so nothing overflows 32
// bits and no wider intermediate is needed.
// --------------------------------------------------------------------------

// The largest dividend the form below is exact for. Callers divide element
// indices, so they check their extents against this and take the slower path
// rather than a wrong answer if a tensor is somehow past it.
#define GK_CU_FASTDIV_MAX 0x7fffffffu

struct gk_cu_fastdiv {
    uint32_t m;   // multiplier; zero when the divisor is one
    uint32_t l;   // shift
    uint32_t d;   // the divisor itself, for recovering the remainder
};

// Requires 1 <= d <= GK_CU_FASTDIV_MAX. A divisor past that would need a shift
// of 32 and an intermediate of 2^64, which is the check the callers make.
static __host__ __forceinline__ struct gk_cu_fastdiv gk_cu_fastdiv_make(uint32_t d) {
    struct gk_cu_fastdiv f;

    f.d = d;

    if (d <= 1) {
        f.m = 0;
        f.l = 0;
        return f;
    }

    uint32_t l = 0;
    while (((uint32_t) 1 << l) < d) {
        ++l;
    }

    f.m = (uint32_t) ((((uint64_t) 1 << (32 + l)) / d) - ((uint64_t) 1 << 32) + 1);
    f.l = l;

    return f;
}

static __device__ __forceinline__ uint32_t gk_cu_fastdiv_q(const struct gk_cu_fastdiv & f,
                                                           uint32_t n) {
    // A divisor of one is the common shape of a degenerate dimension, and the
    // multiply-shift form has no magic number that reproduces it. The compare
    // is uniform across the warp and costs nothing next to what it stands in
    // for.
    if (f.m == 0) {
        return n;
    }

    const uint32_t t = __umulhi(f.m, n);

    return (((n - t) >> 1) + t) >> (f.l - 1);
}

// Quotient and remainder together, which is what decomposing an index wants.
static __device__ __forceinline__ uint32_t gk_cu_fastdiv_qr(const struct gk_cu_fastdiv & f,
                                                            uint32_t n, uint32_t * rem) {
    const uint32_t q = gk_cu_fastdiv_q(f, n);
    *rem = n - q * f.d;
    return q;
}

// --------------------------------------------------------------------------
// scratch
//
// Device memory a kernel needs while it runs and not afterwards. Only the
// split attention path wants any so far: it hands each block a slice of the
// cache and has to put the partial results somewhere until the combine pass
// merges them.
//
// One buffer per backend, grown when a launch needs more than it holds and
// never shrunk. Allocating per launch is what this exists to avoid - a device
// allocation synchronizes, which would cost more than the kernel being served
// and would serialize the queue the rest of the backend works hard to keep
// full. Growth is rare in practice: the shapes a model runs settle after the
// first few nodes and the buffer stops moving.
// --------------------------------------------------------------------------

// Per-backend launch state: the scratch buffer, and what the launchers need
// to know about the device to size a grid. It is per-backend rather than
// global because a machine can have two unlike devices, and a grid sized for
// one of them on the other is a heuristic quietly applied to the wrong card.
struct gk_cuda_scratch {
    void * ptr;
    size_t size;
    int    n_sm;   // multiprocessors on this device
    int    cc;     // compute capability, major*10 + minor
    int    smem_max; // dynamic shared memory a block may opt in to, bytes

    // Bumped every time `ptr` moves. A captured graph bakes the scratch
    // pointer into its kernel launches, so a replay after a grow would read
    // memory that has been freed; the graph cache compares this against the
    // value it saw at capture and re-captures instead.
    uint64_t gen;

    // What the scratch currently holds, when that is a quantized activation.
    // The q/k/v projections - and gate/up - dot different weights against the
    // *same* activation, so the second and third quantize of it are the first
    // one's answer recomputed. A match on every field skips the pass; any
    // other scratch user clears the claim on entry (see gk_cu_scratch_get)
    // and the invariant holds by construction. `aq_pass` ties the claim to one
    // graph execution, because the same address holds new numbers next token.
    //
    // The claim names the *tensor* as well as its address, and it has to: the
    // graph allocator hands a dead tensor's storage to a later one, so within
    // a single execution one address is many tensors. Two of them the same
    // size - an attention output and the normalized activation that fed the
    // projection before it, both [n_embd, n_token] - matched on address,
    // block count and pass alike, and the second matmul then dotted its
    // weights against the first tensor's numbers. Nothing about that is
    // visible in the kernel: the answer is wrong, deterministic, and moves
    // when anything upstream changes the allocation.
    const void * aq_src;     // src1->data
    const void * aq_tensor;  // src1 itself, which the allocator does not recycle
    int64_t      aq_blk;
    int64_t      aq_grp;
    int64_t      aq_planes;  // whether the transposed scalar planes rode along
    uint64_t     aq_pass;
    uint64_t     pass;   // bumped by the backend at every graph_compute
};

// Returns a buffer of at least `bytes`, or NULL if the device has no room.
// The stream is needed because a grow frees the old buffer, and work already
// queued may still be reading it.
// What the grow path has cost so far. The comment above claims growth is rare;
// these are here so that claim is checked rather than assumed, because the two
// ways it can be wrong - a size that oscillates, or an allocation that fails and
// leaves the buffer empty for the next caller to allocate again - both turn a
// once-per-model cost into a per-node one, and neither is visible in a profile
// that attributes the time to whichever kernel happened to ask.
struct gk_cu_scratch_stats {
    int64_t calls;
    int64_t grows;
    int64_t fails;
    double  grow_ms;
};

extern struct gk_cu_scratch_stats g_gk_scratch_stats;

static __host__ __forceinline__ void * gk_cu_scratch_get(struct gk_cuda_scratch * s,
                                                         size_t bytes, gkStream_t stream) {
    g_gk_scratch_stats.calls++;

    // Whoever asked may write the buffer, so any quantized activation in it
    // is spoken for. The one caller that wants to keep it reads the aq fields
    // *before* calling and re-asserts them after.
    s->aq_src    = NULL;
    s->aq_tensor = NULL;

    if (s->ptr != NULL && s->size >= bytes) {
        return s->ptr;
    }

    g_gk_scratch_stats.grows++;
    s->gen++;
    const std::chrono::steady_clock::time_point t_grow0 = std::chrono::steady_clock::now();

    if (s->ptr != NULL) {
        // The queued work that was using the old buffer has to finish before
        // the memory goes back. This is the one synchronize in the path, and
        // it happens only when a shape grows past every shape before it.
        gkStreamSynchronize(stream);
        gkFree(s->ptr);
        s->ptr  = NULL;
        s->size = 0;
    }

    void * p = NULL;
    if (gkMalloc(&p, bytes) != gkSuccess) {
        g_gk_scratch_stats.fails++;
        g_gk_scratch_stats.grow_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_grow0).count();
        return NULL;
    }

    s->ptr  = p;
    s->size = bytes;

    g_gk_scratch_stats.grow_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_grow0).count();
    return p;
}

// --------------------------------------------------------------------------
// launch geometry
// --------------------------------------------------------------------------

#define GK_CUDA_BLOCK 256

// Grids are capped per dimension; a graph with a very large flat row count
// would exceed the limit, so kernels that use a flat index take a stride loop
// and this only sizes the first pass over it.
static __host__ __forceinline__ int gk_cu_blocks(int64_t n, int block) {
    const int64_t b = (n + block - 1) / block;
    return (int) (b > 65535 ? 65535 : (b < 1 ? 1 : b));
}
