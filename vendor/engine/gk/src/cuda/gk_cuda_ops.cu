// The CUDA/HIP kernels for everything except the matmuls.
//
// These are written to the same rule as the CPU pass they mirror: the CPU
// kernel is the definition of what an op means and this is a second
// implementation of that definition, so where there is a choice, the choice
// that reproduces the CPU's arithmetic wins over the one that runs faster.
// The accumulations that matter - the normalisation statistics, the softmax
// normaliser - are done the same way round and in the same precision, because
// a graph that is scheduled half here and half on the host must not produce
// visibly different numbers depending on where a layer happened to land.
//
// Two shapes cover almost everything:
//
//   elementwise   one thread per destination element, index decomposed from a
//                 flat id. Strides are honoured, so permuted operands work.
//   row-wise      one block per destination row, for the ops with a reduction
//                 along dimension 0 - the norms, the softmaxes, the sums.
//
// The ops that do not fit either - rope's per-position angles, attention's
// online softmax - say so in their own comments.

#include "gk_cuda_ops.cuh"

#include <float.h>
#include <stdlib.h>
#include <string.h>

// The op, unary and glu enums are used directly: this file includes gk.h
// through gk_impl.h, and an enumerator is a compile-time constant that device
// code can switch on as happily as host code can. Only the *type* codes are
// mirrored (in gk_cuda_dequant.cuh, which stands alone), and the assertions at
// the bottom of this file hold that mirror against the original.

// --------------------------------------------------------------------------
// elementwise
// --------------------------------------------------------------------------

// The flat destination index, decomposed. Every elementwise kernel below
// starts here, so the decomposition is written once.
struct gk_cu_idx {
    int64_t i0, i1, i2, i3;
};

static __device__ __forceinline__ gk_cu_idx gk_cu_decompose(int64_t k, const int64_t ne[4]) {
    gk_cu_idx x;
    x.i0 = k % ne[0];
    x.i1 = (k / ne[0]) % ne[1];
    x.i2 = (k / (ne[0] * ne[1])) % ne[2];
    x.i3 = k / (ne[0] * ne[1] * ne[2]);
    return x;
}

// RWKV-6 gives every slot of a head its own thread, so its block has to be at
// least as wide as the head, rounded up to a whole warp.

static __host__ __forceinline__ int gk_cu_round_warp(int64_t n) {
    return (int) ((n + GK_WARP_SIZE - 1) / GK_WARP_SIZE) * GK_WARP_SIZE;
}

// The other two recurrences give a whole warp to each row and stride their
// rows over the warps, so any warp-shaped block is correct and this is only a
// tuning choice.
#define GK_CU_RECURRENT_BLOCK 256

#define GK_CU_FLAT_LOOP(n) \
    for (int64_t k = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; \
         k < (n); k += (int64_t) gridDim.x * blockDim.x)

// add / sub / mul / div. src1 broadcasts onto src0: every dimension, dimension
// zero included, wraps with a modulo, which is what makes a per-row bias or a
// per-channel scale work without materialising it.
static __global__ void gk_cu_k_binary(gk_tview a, gk_tview b, gk_tview_mut d,
                                      int kind, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const float va = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        const float vb = gk_cu_get(b, x.i0 % b.ne[0], x.i1 % b.ne[1],
                                      x.i2 % b.ne[2], x.i3 % b.ne[3]);

        float r;
        switch (kind) {
            case 0:  r = va + vb; break;
            case 1:  r = va - vb; break;
            case 2:  r = va * vb; break;
            default: r = va / vb; break;
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, r);
    }
}

static __global__ void gk_cu_k_unary(gk_tview a, gk_tview_mut d, int op,
                                     float p1, float p2, float p3, float p4, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, gk_cu_unary(op, v, p1, p2, p3, p4));
    }
}

// sqr / sqrt / log / sin / cos, which carry their own op ids rather than
// travelling as unary variants
static __global__ void gk_cu_k_simple(gk_tview a, gk_tview_mut d, int which, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);

        float r;
        switch (which) {
            case 0:  r = v * v;     break;
            case 1:  r = sqrtf(v);  break;
            case 2:  r = logf(v);   break;
            case 3:  r = sinf(v);   break;
            default: r = cosf(v);   break;
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, r);
    }
}

static __global__ void gk_cu_k_scale(gk_tview a, gk_tview_mut d, float s, float bias, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, gk_cu_get(a, x.i0, x.i1, x.i2, x.i3) * s + bias);
    }
}

static __global__ void gk_cu_k_clamp(gk_tview a, gk_tview_mut d, float lo, float hi, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, fminf(hi, fmaxf(lo, v)));
    }
}

static __global__ void gk_cu_k_fill(gk_tview_mut d, float c, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, c);
    }
}

// --------------------------------------------------------------------------
// row-mapped elementwise
//
// The kernels above decompose a flat index into (i0,i1,i2,i3) with four 64-bit
// divisions, and a binary op pays four more taking the broadcast modulo of the
// second operand. A 64-bit division is software on this hardware - some tens
// of instructions - so an add of a million f32 elements spends an order of
// magnitude more instructions decomposing indices than it does adding, and
// measures at a flat ~13 G elements/s whatever the tensor's size. That flat
// rate is the tell: a bandwidth-bound kernel gets faster per element when its
// operands fit in L2, and this one does not.
//
// Mapping the grid to the shape instead - one block row per (i1,i2,i3), the
// block's threads walking i0 - costs one division per *block* and none per
// element, and lets a contiguous row be read four floats at a time.
//
// It applies when every operand is f32 with a unit innermost stride, which is
// every residual add, bias add, per-channel scale and activation in a
// diffusion UNet. Everything else - a permuted innermost axis, f16, a
// broadcast that is neither the full extent nor one - still goes through the
// flat kernels above, which stay correct for any stride, type and shape.
// --------------------------------------------------------------------------

// The launch, decided on the host. `ok` false means this shape has to use the
// flat kernel; nothing below is then valid.
struct gk_cu_rows_plan {
    bool    ok;
    bool    vec;        // four floats per thread per step
    bool    bcast0;     // src1 has one column, broadcast along i0
    dim3    grid;
    int     block;
    int64_t as[4], bs[4], ds[4];   // strides in elements, outer three used
    struct gk_cu_fastdiv ne2;      // splits blockIdx.z into (i2, i3)
};

// gridDim.y and gridDim.z are 16-bit on every device gk runs on; a tensor
// taller than this in either keeps the flat kernel.
#define GK_CU_ROWS_MAX_DIM 65535

// Enough blocks along i0 to cover the row without an unbounded grid; the
// kernels stride whatever is left.
#define GK_CU_ROWS_MAX_X 256

// The shortest row - in threads, so in float4s where the vector kernel is
// taken - worth giving a block of its own. Measured, against the fully general
// kernel, on strided rows the flat kernels may not take (bench-cuda,
// "rope-shaped elementwise", GPU ms):
//
//     row      threads   row-mapped   general
//      2         2          4.704       0.590
//     16         4          0.571       0.397
//     32         8          0.350       0.392
//     64        16          0.348       0.387
//
// Eight is where it turns over, and below it the row-mapped kernel does not
// merely lose - at a two-element row it is 8x slower, because the grid is then
// three million blocks retiring two elements each. `GK_EW_ROWS_MIN` overrides
// this: 1 forces the row-mapped kernel on every shape it can express, a large
// value forces none, which is how the table above was taken.
#define GK_CU_ROWS_MIN 8

static __host__ int64_t gk_cu_rows_min(void) {
    static const int64_t v = []() -> int64_t {
        const char * e = getenv("GK_EW_ROWS_MIN");
        const long   n = e != NULL ? strtol(e, NULL, 10) : 0;
        return n > 0 ? (int64_t) n : (int64_t) GK_CU_ROWS_MIN;
    }();
    return v;
}

static __host__ bool gk_cu_rows_strides(const struct gk_tensor * t, int64_t * s) {
    if (t->nb[0] != sizeof(float)) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (t->nb[i] % sizeof(float) != 0) {
            return false;
        }
        s[i] = (int64_t) (t->nb[i] / sizeof(float));
    }
    return true;
}

// `b` may be NULL for the one-operand kernels.
static __host__ struct gk_cu_rows_plan gk_cu_rows_plan_make(const struct gk_tensor * d,
                                                            const struct gk_tensor * a,
                                                            const struct gk_tensor * b) {
    struct gk_cu_rows_plan p;
    memset(&p, 0, sizeof(p));

    if (d->type != GKT_F32 || a->type != GKT_F32 || (b != NULL && b->type != GKT_F32)) {
        return p;
    }
    if (!gk_are_same_shape(a, d)) {
        return p;
    }
    if (d->ne[1] > GK_CU_ROWS_MAX_DIM || d->ne[2] * d->ne[3] > GK_CU_ROWS_MAX_DIM) {
        return p;
    }
    if (d->ne[2] > GK_CU_FASTDIV_MAX) {
        return p;
    }
    if (!gk_cu_rows_strides(d, p.ds) || !gk_cu_rows_strides(a, p.as)) {
        return p;
    }

    if (b != NULL) {
        // The flat kernel takes `i % b->ne[i]` in every dimension. Only the two
        // divisors that mean something - the full extent, or one - are
        // reproduced here as a stride and a zero; anything else falls back
        // rather than be approximated.
        for (int i = 0; i < 4; ++i) {
            if (b->ne[i] != d->ne[i] && b->ne[i] != 1) {
                return p;
            }
        }
        if (!gk_cu_rows_strides(b, p.bs)) {
            return p;
        }
        for (int i = 1; i < 4; ++i) {
            if (b->ne[i] == 1) {
                p.bs[i] = 0;
            }
        }
        p.bcast0 = b->ne[0] == 1 && d->ne[0] != 1;
    }

    // Four at a time needs the row length and every row start to be a multiple
    // of four floats, and the tensors themselves to be sixteen-byte aligned.
    p.vec = d->ne[0] % 4 == 0;
    for (int i = 1; i < 4 && p.vec; ++i) {
        p.vec = p.ds[i] % 4 == 0 && p.as[i] % 4 == 0 && (b == NULL || p.bs[i] % 4 == 0);
    }
    if (p.vec) {
        p.vec = ((uintptr_t) d->data % 16 == 0) && ((uintptr_t) a->data % 16 == 0) &&
                (b == NULL || (uintptr_t) b->data % 16 == 0);
    }
    if (p.vec && b != NULL && !p.bcast0) {
        p.vec = p.bs[0] == 1;
    }

    const int64_t per = p.vec ? d->ne[0] / 4 : d->ne[0];

    // A block per row is only a good trade while a row can keep a block busy.
    // Below a warp it is a bad one and it gets worse the shorter the row is:
    // rope's interleaved layout is {2, d_head/2, L, n_head}, where one row is
    // two elements, a 256-thread block retires two of them, and the grid is
    // three million blocks - 20x slower than the flat kernel's index
    // arithmetic on the same tensor. Hand those back; the caller's fallback
    // decomposes an index per element and still wins by an order of magnitude.
    if (per < gk_cu_rows_min()) {
        p.ok = false;
        return p;
    }

    // What is left is rounded to whole warps rather than taken as a fixed 256,
    // so a row of 64 fills its block instead of idling three quarters of it.
    int blk = (int) ((per + 31) / 32) * 32;
    if (blk > GK_CUDA_BLOCK) {
        blk = GK_CUDA_BLOCK;
    }

    int64_t gx = (per + blk - 1) / blk;
    if (gx > GK_CU_ROWS_MAX_X) {
        gx = GK_CU_ROWS_MAX_X;
    }

    p.grid  = dim3((unsigned) gx, (unsigned) d->ne[1], (unsigned) (d->ne[2] * d->ne[3]));
    p.block = blk;
    p.ne2   = gk_cu_fastdiv_make((uint32_t) d->ne[2]);
    p.ok    = true;

    return p;
}

// GK_EW_DUMP: the operand geometry behind each distinct elementwise shape, and
// which of the three kernels took it, once per (op, shape). This is the
// GK_MM_DUMP of the elementwise ops, and it exists for the same reason: an op
// with a shape and a rate does not say whether the fast path took it, and the
// three paths here differ by an order of magnitude on the same tensor. A rate
// that is 20x off the one the same op reaches at another shape is always this
// question.
static __host__ bool gk_cu_ew_dump_on(void) {
    static const bool on = getenv("GK_EW_DUMP") != NULL && getenv("GK_EW_DUMP")[0] != '0';
    return on;
}

static __host__ void gk_cu_ew_dump(const char * what, const char * path,
                                   const struct gk_tensor * d,
                                   const struct gk_tensor * a,
                                   const struct gk_tensor * b) {
    if (!gk_cu_ew_dump_on()) {
        return;
    }

    static char seen[128][96];
    static int  n_seen = 0;

    char key[96];
    snprintf(key, sizeof(key), "%s %lldx%lldx%lldx%lld", what,
             (long long) d->ne[0], (long long) d->ne[1],
             (long long) d->ne[2], (long long) d->ne[3]);

    for (int i = 0; i < n_seen; ++i) {
        if (strcmp(seen[i], key) == 0) {
            return;
        }
    }
    if (n_seen >= 128) {
        return;
    }
    snprintf(seen[n_seen++], sizeof(seen[0]), "%s", key);

    gk_logf("ew %-28s [%s]\n", key, path);

    const struct gk_tensor * ts[3] = { d, a, b };
    const char * nm[3] = { "dst ", "src0", "src1" };
    for (int t = 0; t < 3; ++t) {
        if (ts[t] == NULL) {
            continue;
        }
        gk_logf("   %s %-6s ne=[%lld %lld %lld %lld] nb=[%zu %zu %zu %zu] cont=%d\n",
                nm[t], gk_type_name(ts[t]->type),
                (long long) ts[t]->ne[0], (long long) ts[t]->ne[1],
                (long long) ts[t]->ne[2], (long long) ts[t]->ne[3],
                (size_t) ts[t]->nb[0], (size_t) ts[t]->nb[1],
                (size_t) ts[t]->nb[2], (size_t) ts[t]->nb[3],
                (int) gk_is_contiguous(ts[t]));
    }
}

static __device__ __forceinline__ float gk_cu_binary_apply(int kind, float va, float vb) {
    switch (kind) {
        case 0:  return va + vb;
        case 1:  return va - vb;
        case 2:  return va * vb;
        default: return va / vb;
    }
}

// The flat form of the same idea, for the case where every operand is
// contiguous. Then the destination's flat index *is* the address, so there is
// no decomposition to do at all - and unlike the row-mapped kernels, a block
// is fully busy however short a row is. A residual add in a UNet is
// {W, H, C, 1} with W=64: row-mapped, that is 64 lanes of a 256-thread block
// working and the other 192 idle.
//
// The second operand's broadcast is the only thing that needs an index, and it
// takes one 32-bit fast division rather than four 64-bit ones - and only in
// the shapes that broadcast at all.
enum gk_cu_bcast {
    GK_CU_BCAST_NONE = 0,   // src1 has the destination's shape
    GK_CU_BCAST_WRAP = 1,   // src1 is a leading block of it, repeated: {ne0, 1, 1, 1},
                            // {ne0, ne1, 1, 1}, {ne0, ne1, ne2, 1}
    GK_CU_BCAST_CH   = 2,   // src1 is one column:  {1, 1, ne2, ne3}
    GK_CU_BCAST_ONE  = 3,   // src1 is one element
};

// Whether a source broadcasts onto `d` in one of the four patterns above, or
// -1 for anything else. Contiguity is the caller's to check.
//
// WRAP covers every operand whose extents agree with the destination's over a
// leading run of dimensions and are one above it: both tensors are contiguous,
// so src1's flat index is then the destination's modulo src1's element count,
// whatever that run's length. Only {ne0,1,1,1} used to be admitted, which left
// a rope table of {2, d_head/2, L, 1} multiplying a {2, d_head/2, L, n_head}
// activation - contiguous on both sides and broadcast over nothing but the
// outermost axis - falling through to the row-mapped kernel, where a row is
// two elements long and 254 of a block's 256 lanes do nothing.
static __host__ int gk_cu_bcast_kind(const struct gk_tensor * d, const struct gk_tensor * b) {
    if (b->ne[0] == d->ne[0] && b->ne[1] == d->ne[1] &&
        b->ne[2] == d->ne[2] && b->ne[3] == d->ne[3]) {
        return GK_CU_BCAST_NONE;
    }
    if (b->ne[0] == 1 && b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1) {
        return GK_CU_BCAST_ONE;
    }
    if (b->ne[0] == 1 && b->ne[1] == 1 && b->ne[2] == d->ne[2] && b->ne[3] == d->ne[3]) {
        return GK_CU_BCAST_CH;
    }

    // A leading run that matches, then ones all the way out.
    int p = 0;
    while (p < 4 && b->ne[p] == d->ne[p]) {
        ++p;
    }
    for (int i = p; i < 4; ++i) {
        if (b->ne[i] != 1) {
            return -1;
        }
    }
    if (p > 0) {
        return GK_CU_BCAST_WRAP;
    }

    return -1;
}

static __host__ bool gk_cu_flat_ok(const struct gk_tensor * t) {
    return t->type == GKT_F32 && gk_is_contiguous(t);
}

// Sixteen-byte reads need the base aligned and the run a multiple of four.
static __host__ bool gk_cu_flat_vec(const struct gk_tensor * t, int64_t n) {
    return n % 4 == 0 && (uintptr_t) t->data % 16 == 0;
}

#define GK_CU_FLAT_BLOCKS(n) ((int) (((n) + GK_CUDA_BLOCK - 1) / GK_CUDA_BLOCK) > 65535 \
                              ? 65535 : (int) (((n) + GK_CUDA_BLOCK - 1) / GK_CUDA_BLOCK))

// `wrap` divides the flat index by src1's element count (in float4s in the
// vector kernel) and `plane` by ne0*ne1; only the one the broadcast needs is
// ever used.
template <int KIND, int BC>
static __global__ void gk_cu_k_binary_flat(const float * __restrict__ a,
                                           const float * __restrict__ b,
                                           float * __restrict__ d, int64_t n,
                                           struct gk_cu_fastdiv wrap,
                                           struct gk_cu_fastdiv plane) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x * blockDim.x) {
        float vb;
        if      (BC == GK_CU_BCAST_NONE) { vb = b[i]; }
        else if (BC == GK_CU_BCAST_ONE)  { vb = b[0]; }
        else if (BC == GK_CU_BCAST_WRAP) { uint32_t r; gk_cu_fastdiv_qr(wrap, (uint32_t) i, &r); vb = b[r]; }
        else                             { vb = b[gk_cu_fastdiv_q(plane, (uint32_t) i)]; }

        d[i] = gk_cu_binary_apply(KIND, a[i], vb);
    }
}

template <int KIND, int BC>
static __global__ void gk_cu_k_binary_flat4(const float4 * __restrict__ a,
                                            const float * __restrict__ b,
                                            float4 * __restrict__ d, int64_t n4,
                                            struct gk_cu_fastdiv wrap,
                                            struct gk_cu_fastdiv plane) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n4;
         i += (int64_t) gridDim.x * blockDim.x) {
        const float4 va = a[i];

        float4 vb;
        if (BC == GK_CU_BCAST_NONE) {
            vb = ((const float4 *) b)[i];
        } else if (BC == GK_CU_BCAST_WRAP) {
            uint32_t r;
            gk_cu_fastdiv_qr(wrap, (uint32_t) i, &r);
            vb = ((const float4 *) b)[r];
        } else {
            const float s = BC == GK_CU_BCAST_ONE ? b[0]
                                                  : b[gk_cu_fastdiv_q(plane, (uint32_t) i)];
            vb = make_float4(s, s, s, s);
        }

        d[i] = make_float4(gk_cu_binary_apply(KIND, va.x, vb.x),
                           gk_cu_binary_apply(KIND, va.y, vb.y),
                           gk_cu_binary_apply(KIND, va.z, vb.z),
                           gk_cu_binary_apply(KIND, va.w, vb.w));
    }
}

static __global__ void gk_cu_k_unary_flat(const float * __restrict__ a, float * __restrict__ d,
                                          int64_t n, int op,
                                          float p1, float p2, float p3, float p4) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x * blockDim.x) {
        d[i] = gk_cu_unary(op, a[i], p1, p2, p3, p4);
    }
}

static __global__ void gk_cu_k_unary_flat4(const float4 * __restrict__ a, float4 * __restrict__ d,
                                           int64_t n4, int op,
                                           float p1, float p2, float p3, float p4) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n4;
         i += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = a[i];
        d[i] = make_float4(gk_cu_unary(op, v.x, p1, p2, p3, p4),
                           gk_cu_unary(op, v.y, p1, p2, p3, p4),
                           gk_cu_unary(op, v.z, p1, p2, p3, p4),
                           gk_cu_unary(op, v.w, p1, p2, p3, p4));
    }
}

template <int KIND, bool BCAST0>
static __global__ void gk_cu_k_binary_rows(const float * __restrict__ a,
                                           const float * __restrict__ b,
                                           float * __restrict__ d,
                                           int64_t ne0,
                                           int64_t as1, int64_t as2, int64_t as3,
                                           int64_t bs0, int64_t bs1, int64_t bs2, int64_t bs3,
                                           int64_t ds1, int64_t ds2, int64_t ds3,
                                           struct gk_cu_fastdiv ne2) {
    uint32_t i2;
    const uint32_t i3 = gk_cu_fastdiv_qr(ne2, blockIdx.z, &i2);
    const uint32_t i1 = blockIdx.y;

    const float * ar = a + i1 * as1 + i2 * as2 + i3 * as3;
    const float * br = b + i1 * bs1 + i2 * bs2 + i3 * bs3;
    float       * dr = d + i1 * ds1 + i2 * ds2 + i3 * ds3;

    const float bv = BCAST0 ? br[0] : 0.0f;

    for (int64_t i0 = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i0 < ne0;
         i0 += (int64_t) gridDim.x * blockDim.x) {
        dr[i0] = gk_cu_binary_apply(KIND, ar[i0], BCAST0 ? bv : br[i0 * bs0]);
    }
}

template <int KIND, bool BCAST0>
static __global__ void gk_cu_k_binary_rows4(const float * __restrict__ a,
                                            const float * __restrict__ b,
                                            float * __restrict__ d,
                                            int64_t ne0_4,
                                            int64_t as1, int64_t as2, int64_t as3,
                                            int64_t bs1, int64_t bs2, int64_t bs3,
                                            int64_t ds1, int64_t ds2, int64_t ds3,
                                            struct gk_cu_fastdiv ne2) {
    uint32_t i2;
    const uint32_t i3 = gk_cu_fastdiv_qr(ne2, blockIdx.z, &i2);
    const uint32_t i1 = blockIdx.y;

    const float4 * ar = (const float4 *) (a + i1 * as1 + i2 * as2 + i3 * as3);
    const float4 * br = (const float4 *) (b + i1 * bs1 + i2 * bs2 + i3 * bs3);
    float4       * dr = (float4 *)       (d + i1 * ds1 + i2 * ds2 + i3 * ds3);

    const float bv = BCAST0 ? *(const float *) br : 0.0f;

    for (int64_t i0 = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i0 < ne0_4;
         i0 += (int64_t) gridDim.x * blockDim.x) {
        const float4 va = ar[i0];
        const float4 vb = BCAST0 ? make_float4(bv, bv, bv, bv) : br[i0];

        dr[i0] = make_float4(gk_cu_binary_apply(KIND, va.x, vb.x),
                             gk_cu_binary_apply(KIND, va.y, vb.y),
                             gk_cu_binary_apply(KIND, va.z, vb.z),
                             gk_cu_binary_apply(KIND, va.w, vb.w));
    }
}

// unary, and the five that carry their own op id. Both take the op as a
// runtime argument: the branch is uniform across the launch, and the cost this
// path exists to remove is the index arithmetic, not the switch.
static __global__ void gk_cu_k_unary_rows(const float * __restrict__ a,
                                          float * __restrict__ d,
                                          int64_t ne0,
                                          int64_t as1, int64_t as2, int64_t as3,
                                          int64_t ds1, int64_t ds2, int64_t ds3,
                                          int op, float p1, float p2, float p3, float p4,
                                          struct gk_cu_fastdiv ne2) {
    uint32_t i2;
    const uint32_t i3 = gk_cu_fastdiv_qr(ne2, blockIdx.z, &i2);
    const uint32_t i1 = blockIdx.y;

    const float * ar = a + i1 * as1 + i2 * as2 + i3 * as3;
    float       * dr = d + i1 * ds1 + i2 * ds2 + i3 * ds3;

    for (int64_t i0 = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i0 < ne0;
         i0 += (int64_t) gridDim.x * blockDim.x) {
        dr[i0] = gk_cu_unary(op, ar[i0], p1, p2, p3, p4);
    }
}

static __global__ void gk_cu_k_unary_rows4(const float * __restrict__ a,
                                           float * __restrict__ d,
                                           int64_t ne0_4,
                                           int64_t as1, int64_t as2, int64_t as3,
                                           int64_t ds1, int64_t ds2, int64_t ds3,
                                           int op, float p1, float p2, float p3, float p4,
                                           struct gk_cu_fastdiv ne2) {
    uint32_t i2;
    const uint32_t i3 = gk_cu_fastdiv_qr(ne2, blockIdx.z, &i2);
    const uint32_t i1 = blockIdx.y;

    const float4 * ar = (const float4 *) (a + i1 * as1 + i2 * as2 + i3 * as3);
    float4       * dr = (float4 *)       (d + i1 * ds1 + i2 * ds2 + i3 * ds3);

    for (int64_t i0 = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i0 < ne0_4;
         i0 += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = ar[i0];
        dr[i0] = make_float4(gk_cu_unary(op, v.x, p1, p2, p3, p4),
                             gk_cu_unary(op, v.y, p1, p2, p3, p4),
                             gk_cu_unary(op, v.z, p1, p2, p3, p4),
                             gk_cu_unary(op, v.w, p1, p2, p3, p4));
    }
}

static __global__ void gk_cu_k_leaky_relu(gk_tview a, gk_tview_mut d, float slope, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, v > 0.0f ? v : v * slope);
    }
}

// Gated linear units. With one operand the row splits in half and `swapped`
// says which half is the activated side; with two, src0 is activated and src1
// multiplies. Getting the halves the wrong way round produces plausible and
// wrong output, so the split is spelled out rather than inferred.
static __global__ void gk_cu_k_glu(gk_tview a, gk_tview b, bool has_b, gk_tview_mut d,
                                   int op, bool swapped, float alpha, float limit, int64_t n) {
    const int64_t half = d.ne[0];

    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        float act, mul;
        if (has_b) {
            act = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
            mul = gk_cu_get(b, x.i0, x.i1, x.i2, x.i3);
        } else {
            const int64_t ia = swapped ? x.i0 + half : x.i0;
            const int64_t im = swapped ? x.i0        : x.i0 + half;
            act = gk_cu_get(a, ia, x.i1, x.i2, x.i3);
            mul = gk_cu_get(a, im, x.i1, x.i2, x.i3);
        }

        float r;
        switch (op) {
            case GK_GLU_OP_REGLU:       r = (act > 0.0f ? act : 0.0f) * mul; break;
            case GK_GLU_OP_GEGLU:       r = gk_cu_gelu(act) * mul;           break;
            case GK_GLU_OP_SWIGLU:      r = gk_cu_silu(act) * mul;           break;
            case GK_GLU_OP_GEGLU_ERF:   r = gk_cu_gelu_erf(act) * mul;       break;
            case GK_GLU_OP_GEGLU_QUICK: r = gk_cu_gelu_quick(act) * mul;     break;
            default: {
                // the gpt-oss variant: both operands clamped, a sigmoid with a
                // temperature, and +1 on the gate
                const float xv = fminf(act, limit);
                const float yv = fminf(fmaxf(mul, -limit), limit);
                const float g  = xv / (1.0f + expf(alpha * (-xv)));
                r = g * (yv + 1.0f);
                break;
            }
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, r);
    }
}

// --------------------------------------------------------------------------
// normalisation
//
// One block per row, two passes over it: the statistic, then the scaling.
// The sums are f32 here where the CPU uses f64 for NORM's mean - a block
// reduction over a few thousand elements has a shallow enough tree that the
// difference stays well inside the tolerance the differential tests use, and
// f64 arithmetic on a consumer device costs an order of magnitude.
// --------------------------------------------------------------------------

#define GK_CU_NORM_BLOCK 256

static __global__ void gk_cu_k_norm(gk_tview a, gk_tview_mut d, int kind, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    const int64_t ir = blockIdx.x;
    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    const int64_t n = d.ne[0];

    if (kind == 1) { // NORM: mean first, then the variance about it
        float sum = 0.0f;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            sum += gk_cu_get(a, i, i1, i2, i3);
        }
        const float mean = gk_cu_block_sum(sum, scratch) / (float) n;
        __syncthreads();

        float var = 0.0f;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float c = gk_cu_get(a, i, i1, i2, i3) - mean;
            var += c * c;
        }
        const float scale = rsqrtf(gk_cu_block_sum(var, scratch) / (float) n + eps);

        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            gk_cu_set(d, i, i1, i2, i3, (gk_cu_get(a, i, i1, i2, i3) - mean) * scale);
        }
        return;
    }

    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = gk_cu_get(a, i, i1, i2, i3);
        sumsq += v * v;
    }
    const float total = gk_cu_block_sum(sumsq, scratch);

    // kind 0 is RMS_NORM, kind 2 is L2_NORM: the same sum, different divisor
    const float scale = kind == 0
        ? rsqrtf(total / (float) n + eps)
        : rsqrtf(fmaxf(total, eps));

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        gk_cu_set(d, i, i1, i2, i3, gk_cu_get(a, i, i1, i2, i3) * scale);
    }
}

// rms_norm and the weight multiply that always follows it, in one launch.
// A transformer runs this pair four to six times per layer, and at decode
// batch sizes both kernels are launch latency, not work: the pair costs two
// fixed overheads for arithmetic that fits comfortably in one. The math is
// the sequential pair's exactly, save one rounding: (x*scale)*w becomes
// x*scale*w in a single expression.
static __global__ void gk_cu_k_rms_norm_mul_f(const float * __restrict__ a,
                                              const float * __restrict__ w,
                                              float * __restrict__ d,
                                              int64_t n4, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    a += blockIdx.x * n4 * 4;
    d += blockIdx.x * n4 * 4;

    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = ((const float4 *) a)[i];
        sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float total = gk_cu_block_sum(sumsq, scratch);
    const float scale = rsqrtf(total / (float) (n4 * 4) + eps);

    for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v  = ((const float4 *) a)[i];
        const float4 wv = ((const float4 *) w)[i];
        ((float4 *) d)[i] = make_float4(v.x * scale * wv.x, v.y * scale * wv.y,
                                        v.z * scale * wv.z, v.w * scale * wv.w);
    }
}

// The residual step and the norm that reads it, in one launch. Unlike the
// pair below, the add's output cannot be elided - it *is* the residual
// stream, read again at the end of the block - so the kernel writes both
// destinations: the sum, and the normalized-and-weighted sum.
//
// Everything here is flat float4 over contiguous f32 - the fusion plan only
// approves that layout - because the first cut of these kernels went through
// the generic per-element accessor and *lost* to the pairs they replaced:
// six type switches and four-axis index math per element cost more than the
// launch they saved. The weight is one row, shared by every block.
//
// No __restrict__ on the sum: the allocator is free to place it over one of
// the addends, and the second pass reads the sum back rather than
// recomputing it for exactly that reason. Each thread rereads only elements
// it wrote itself, so the passes need no barrier between them.
static __global__ void gk_cu_k_add_rms_norm_mul_f(const float * a, const float * b,
                                                  const float * __restrict__ w,
                                                  float * da, float * __restrict__ dm,
                                                  int64_t n4, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    a  += blockIdx.x * n4 * 4;
    b  += blockIdx.x * n4 * 4;
    da += blockIdx.x * n4 * 4;
    dm += blockIdx.x * n4 * 4;

    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 va = ((const float4 *) a)[i];
        const float4 vb = ((const float4 *) b)[i];
        const float4 s  = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
        ((float4 *) da)[i] = s;
        sumsq += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
    }
    const float total = gk_cu_block_sum(sumsq, scratch);
    const float scale = rsqrtf(total / (float) (n4 * 4) + eps);

    for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 s  = ((const float4 *) da)[i];
        const float4 wv = ((const float4 *) w)[i];
        ((float4 *) dm)[i] = make_float4(s.x * scale * wv.x, s.y * scale * wv.y,
                                         s.z * scale * wv.z, s.w * scale * wv.w);
    }
}

void gk_cuda_fused_add_rms_mul(gkStream_t stream, const struct gk_tensor * add,
                               const struct gk_tensor * norm, const struct gk_tensor * mul) {
    const struct gk_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];

    const int64_t rows = mul->ne[1] * mul->ne[2] * mul->ne[3];

    if (rows <= 0 || mul->ne[0] <= 0) {
        return; // zero-extent graphs are legal; zero-block launches are not
    }

    gk_cu_k_add_rms_norm_mul_f<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
        (const float *) add->src[0]->data, (const float *) add->src[1]->data,
        (const float *) w->data,
        (float *) add->data, (float *) mul->data,
        mul->ne[0] / 4, gk_get_op_params_f32(norm, 0));
}

// The backend's launch loop calls this for a (rms_norm, mul) pair its fusion
// plan approved; the plan already checked ops, adjacency, single use and
// shapes, so this only has to launch. `norm` supplies the input and eps,
// `mul` supplies the weight and the destination; the norm's own output is
// never written.
void gk_cuda_fused_rms_mul(gkStream_t stream, const struct gk_tensor * norm,
                           const struct gk_tensor * mul) {
    const struct gk_tensor * a = norm->src[0];
    const struct gk_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];

    const int64_t rows = mul->ne[1] * mul->ne[2] * mul->ne[3];

    // A zero-extent pair is a legal graph (the final norm over zero selected
    // output rows, during warmup) and on a GPU an illegal launch, not a
    // no-op; the per-op path has the same guard in its dispatcher.
    if (rows <= 0 || mul->ne[0] <= 0) {
        return;
    }

    gk_cu_k_rms_norm_mul_f<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
        (const float *) a->data, (const float *) w->data, (float *) mul->data,
        mul->ne[0] / 4, gk_get_op_params_f32(norm, 0));
}

// --------------------------------------------------------------------------
// The tail fusions: the DiT patterns the adjacent pairs above cannot see.
//
// A DiT block separates a norm from the mul that consumes it by the handful
// of tiny nodes that build the weight, gates its branches with per-token
// vectors, and hand-builds rope out of repeat/mul/add - so its elementwise
// cost is not launch latency but *traffic*: every intermediate is a 25-68 MB
// tensor written and read straight back. The kernels here are the collapsed
// forms. They are deliberately bit-exact against the chains they replace -
// same per-element operations in the same order, same reduction geometry -
// so an A/B against the unfused graph must produce the identical image; the
// win is the elided round trips, not the arithmetic.
//
// The plan side (gk_cu_fuse_plan) is what proves a chain safe: single-use
// intermediates, and no node between a chain's parts whose output was
// allocated over storage the fused kernel still has to read.
// --------------------------------------------------------------------------

// rms_norm and a trailing weight mul, bit-exact: the sum is accumulated in
// the same scalar strided order as gk_cu_k_norm, and the product is rounded
// twice - (x*scale) then *w - exactly as the unfused pair rounds it.
static __global__ void gk_cu_k_rms_norm_mul_x(const float * __restrict__ a,
                                              const float * __restrict__ w,
                                              float * __restrict__ d,
                                              int64_t n, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    a += blockIdx.x * n;
    d += blockIdx.x * n;

    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = a[i];
        sumsq += v * v;
    }
    const float scale = rsqrtf(gk_cu_block_sum(sumsq, scratch) / (float) n + eps);

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        d[i] = __fmul_rn(__fmul_rn(a[i], scale), w[i]);
    }
}

// The residual add as well, still bit-exact; the sum is written - it is the
// residual stream - and read back for the second pass, as in the _f kernel.
static __global__ void gk_cu_k_add_rms_norm_mul_x(const float * a, const float * b,
                                                  const float * __restrict__ w,
                                                  float * da, float * __restrict__ dm,
                                                  int64_t n, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    a  += blockIdx.x * n;
    b  += blockIdx.x * n;
    da += blockIdx.x * n;
    dm += blockIdx.x * n;

    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float s = __fadd_rn(a[i], b[i]);
        da[i] = s;
        sumsq += s * s;
    }
    const float scale = rsqrtf(gk_cu_block_sum(sumsq, scratch) / (float) n + eps);

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dm[i] = __fmul_rn(__fmul_rn(da[i], scale), w[i]);
    }
}

// out = c + y*g (+ t): the gate/modulate cluster. g and t are one row,
// broadcast down the tensor - the DiT's per-block modulation vectors.
static __global__ void gk_cu_k_madd_row(const float * __restrict__ c,
                                        const float * __restrict__ y,
                                        const float * __restrict__ g,
                                        const float * __restrict__ t,
                                        float * __restrict__ dst, int64_t ne0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne0) {
        return;
    }
    const int64_t off = (int64_t) blockIdx.y * ne0 + i;

    float v = __fadd_rn(c[off], __fmul_rn(y[off], g[i]));
    if (t != NULL) {
        v = __fadd_rn(v, t[i]);
    }
    dst[off] = v;
}

// out = unary(x) * y: the gate activations - attention gates, and the MLP's
// geglu when the allocator left the gate projection readable.
static __global__ void gk_cu_k_unary_mul(const float * __restrict__ x,
                                         const float * __restrict__ y,
                                         float * __restrict__ dst, int64_t n,
                                         int uop, float p1, float p2, float p3, float p4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = __fmul_rn(gk_cu_unary(uop, x[i], p1, p2, p3, p4), y[i]);
}

// The hand-built rope pair: add(mul(repeat(x1), f1), mul(repeat(x2), f2)),
// where each x is [1, K, T, H], each f is [2, K, T, 1] and the output is
// [2, K, T, H]. The repeats materialize x twice at full width and the two
// products are read straight back by the add; collapsed, each x element is
// read once and each output pair written once. The grid is (K*T, H) with
// no per-element division - a two-element row is exactly the shape the flat
// elementwise path handles worst.
static __global__ void gk_cu_k_rope_pair(const float * __restrict__ x1,
                                         const float * __restrict__ x2,
                                         const float * __restrict__ f1,
                                         const float * __restrict__ f2,
                                         float2 * __restrict__ dst, int64_t nkt) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nkt) {
        return;
    }
    const int64_t off = (int64_t) blockIdx.y * nkt + i;

    const float2 a = *(const float2 *) (f1 + 2 * i);
    const float2 b = *(const float2 *) (f2 + 2 * i);
    const float v1 = x1[off];
    const float v2 = x2[off];

    dst[off] = make_float2(__fadd_rn(__fmul_rn(v1, a.x), __fmul_rn(v2, b.x)),
                           __fadd_rn(__fmul_rn(v1, a.y), __fmul_rn(v2, b.y)));
}

void gk_cuda_fused_rms_mul_x(gkStream_t stream, const struct gk_tensor * norm,
                             const struct gk_tensor * mul) {
    const struct gk_tensor * a = norm->src[0];
    const struct gk_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];

    const int64_t rows = mul->ne[1] * mul->ne[2] * mul->ne[3];
    if (rows <= 0 || mul->ne[0] <= 0) {
        return;
    }

    gk_cu_k_rms_norm_mul_x<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
        (const float *) a->data, (const float *) w->data, (float *) mul->data,
        mul->ne[0], gk_get_op_params_f32(norm, 0));
}

void gk_cuda_fused_add_rms_mul_x(gkStream_t stream, const struct gk_tensor * add,
                                 const struct gk_tensor * norm, const struct gk_tensor * mul) {
    const struct gk_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];

    const int64_t rows = mul->ne[1] * mul->ne[2] * mul->ne[3];
    if (rows <= 0 || mul->ne[0] <= 0) {
        return;
    }

    gk_cu_k_add_rms_norm_mul_x<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
        (const float *) add->src[0]->data, (const float *) add->src[1]->data,
        (const float *) w->data,
        (float *) add->data, (float *) mul->data,
        mul->ne[0], gk_get_op_params_f32(norm, 0));
}

// `mul` is the y*g product, `add` its consumer, `add2` the optional trailing
// shift; the plan verified there is exactly one row-vector operand in each.
void gk_cuda_fused_madd(gkStream_t stream, const struct gk_tensor * mul,
                        const struct gk_tensor * add, const struct gk_tensor * add2) {
    const struct gk_tensor * g = mul->src[1]->ne[1] == 1 && mul->src[1]->ne[2] == 1 &&
                                 mul->src[1]->ne[3] == 1 ? mul->src[1] : mul->src[0];
    const struct gk_tensor * y = g == mul->src[0] ? mul->src[1] : mul->src[0];
    const struct gk_tensor * c = add->src[0] == mul ? add->src[1] : add->src[0];
    const struct gk_tensor * t = add2 != NULL
        ? (add2->src[0] == add ? add2->src[1] : add2->src[0]) : NULL;
    const struct gk_tensor * d = add2 != NULL ? add2 : add;

    const int64_t ne0  = d->ne[0];
    const int64_t rows = d->ne[1] * d->ne[2] * d->ne[3];
    if (rows <= 0 || ne0 <= 0) {
        return;
    }

    dim3 grid((unsigned) ((ne0 + GK_CUDA_BLOCK - 1) / GK_CUDA_BLOCK), (unsigned) rows);
    gk_cu_k_madd_row<<<grid, GK_CUDA_BLOCK, 0, stream>>>(
        (const float *) c->data, (const float *) y->data, (const float *) g->data,
        t != NULL ? (const float *) t->data : NULL, (float *) d->data, ne0);
}

void gk_cuda_fused_unary_mul(gkStream_t stream, const struct gk_tensor * un,
                             const struct gk_tensor * mul) {
    const struct gk_tensor * y = mul->src[0] == un ? mul->src[1] : mul->src[0];

    const int64_t n = mul->ne[0] * mul->ne[1] * mul->ne[2] * mul->ne[3];
    if (n <= 0) {
        return;
    }

    gk_cu_k_unary_mul<<<(unsigned) ((n + GK_CUDA_BLOCK - 1) / GK_CUDA_BLOCK),
                        GK_CUDA_BLOCK, 0, stream>>>(
        (const float *) un->src[0]->data, (const float *) y->data, (float *) mul->data, n,
        (int) gk_get_unary_op(un),
        gk_get_op_params_f32(un, 1), gk_get_op_params_f32(un, 2),
        gk_get_op_params_f32(un, 3), gk_get_op_params_f32(un, 4));
}

void gk_cuda_fused_rope_pair(gkStream_t stream, const struct gk_tensor * m1,
                             const struct gk_tensor * m2, const struct gk_tensor * add) {
    const struct gk_tensor * r1 = m1->src[0]->op == GK_OP_REPEAT ? m1->src[0] : m1->src[1];
    const struct gk_tensor * f1 = r1 == m1->src[0] ? m1->src[1] : m1->src[0];
    const struct gk_tensor * r2 = m2->src[0]->op == GK_OP_REPEAT ? m2->src[0] : m2->src[1];
    const struct gk_tensor * f2 = r2 == m2->src[0] ? m2->src[1] : m2->src[0];

    const int64_t nkt = add->ne[1] * add->ne[2];   // K*T
    const int64_t nh  = add->ne[3];
    if (nkt <= 0 || nh <= 0) {
        return;
    }

    dim3 grid((unsigned) ((nkt + GK_CUDA_BLOCK - 1) / GK_CUDA_BLOCK), (unsigned) nh);
    gk_cu_k_rope_pair<<<grid, GK_CUDA_BLOCK, 0, stream>>>(
        (const float *) r1->src[0]->data, (const float *) r2->src[0]->data,
        (const float *) f1->data, (const float *) f2->data,
        (float2 *) add->data, nkt);
}

// Group norm's statistic spans a group of channels and their whole spatial
// extent, so the unit of work is a group rather than a row.
// A group of a contiguous f32 tensor is a contiguous span: the groups partition
// the channel axis and the channel axis is the outer one, so group g of batch
// i3 owns [(i3*n_groups + g)*count, +count). That removes the whole index
// decomposition the general kernel below does three times per element, and
// lets the span be read four floats at a time.
//
// The block is also four times wider than the general kernel's. One block per
// group is forced by the reduction, and a UNet normalizes 32 groups at a time -
// 32 blocks on a twenty-multiprocessor part, so a narrow block leaves most of
// the device idle for a pass that is otherwise pure bandwidth.
#define GK_CU_GNORM_BLOCK 1024

template <bool VEC>
static __global__ __launch_bounds__(GK_CU_GNORM_BLOCK, 1)
void gk_cu_k_group_norm_flat(const float * __restrict__ a, float * __restrict__ d,
                             int64_t count, float eps) {
    __shared__ float scratch[GK_CU_GNORM_BLOCK / GK_WARP_SIZE];

    const int64_t off = (int64_t) blockIdx.x * count;

    const float  * ar = a + off;
    float        * dr = d + off;
    const float4 * a4 = (const float4 *) ar;
    float4       * d4 = (float4 *)       dr;

    const int64_t n    = VEC ? count / 4 : count;
    const int64_t step = blockDim.x;

    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += step) {
        if (VEC) {
            const float4 v = a4[i];
            sum += v.x + v.y + v.z + v.w;
        } else {
            sum += ar[i];
        }
    }
    const float mean = gk_cu_block_sum(sum, scratch) / (float) count;
    __syncthreads();

    float var = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += step) {
        if (VEC) {
            const float4 v = a4[i];
            const float4 c = make_float4(v.x - mean, v.y - mean, v.z - mean, v.w - mean);
            var += c.x * c.x + c.y * c.y + c.z * c.z + c.w * c.w;
        } else {
            const float c = ar[i] - mean;
            var += c * c;
        }
    }
    const float scale = rsqrtf(gk_cu_block_sum(var, scratch) / (float) count + eps);

    for (int64_t i = threadIdx.x; i < n; i += step) {
        if (VEC) {
            const float4 v = a4[i];
            d4[i] = make_float4((v.x - mean) * scale, (v.y - mean) * scale,
                                (v.z - mean) * scale, (v.w - mean) * scale);
        } else {
            dr[i] = (ar[i] - mean) * scale;
        }
    }
}

static __global__ void gk_cu_k_group_norm(gk_tview a, gk_tview_mut d,
                                          int n_groups, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    const int64_t unit = blockIdx.x;
    const int64_t i3   = unit / n_groups;
    const int64_t g    = unit % n_groups;

    const int64_t per_group = a.ne[2] / n_groups;
    const int64_t c0 = g * per_group;
    const int64_t c1 = c0 + per_group;

    const int64_t n_in_plane = a.ne[0] * a.ne[1];
    const int64_t count = n_in_plane * per_group;

    float sum = 0.0f;
    for (int64_t t = threadIdx.x; t < count; t += blockDim.x) {
        const int64_t i2 = c0 + t / n_in_plane;
        const int64_t r  = t % n_in_plane;
        sum += gk_cu_get(a, r % a.ne[0], r / a.ne[0], i2, i3);
    }
    const float mean = gk_cu_block_sum(sum, scratch) / (float) count;
    __syncthreads();

    float var = 0.0f;
    for (int64_t t = threadIdx.x; t < count; t += blockDim.x) {
        const int64_t i2 = c0 + t / n_in_plane;
        const int64_t r  = t % n_in_plane;
        const float c = gk_cu_get(a, r % a.ne[0], r / a.ne[0], i2, i3) - mean;
        var += c * c;
    }
    const float scale = rsqrtf(gk_cu_block_sum(var, scratch) / (float) count + eps);

    for (int64_t t = threadIdx.x; t < count; t += blockDim.x) {
        const int64_t i2 = c0 + t / n_in_plane;
        const int64_t r  = t % n_in_plane;
        const int64_t i0 = r % a.ne[0];
        const int64_t i1 = r / a.ne[0];
        gk_cu_set(d, i0, i1, i2, i3, (gk_cu_get(a, i0, i1, i2, i3) - mean) * scale);
    }
}

// --------------------------------------------------------------------------
// copies, gathers and scatters
// --------------------------------------------------------------------------

// Same shape: positions line up. Different shape: the copy is defined over the
// flat element order, which is the path a reshape that cannot be a view takes.
static __global__ void gk_cu_k_copy(gk_tview a, gk_tview_mut d, bool same_shape, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const gk_cu_idx s = same_shape ? x : gk_cu_decompose(k, a.ne);

        if (a.type == GKT_I32 && d.type == GKT_I32) {
            // integers stay integers rather than crossing through the float
            // accessors, whose round trip is only exact below 2^24
            *(int32_t *) (gk_cu_row(d, x.i1, x.i2, x.i3) + x.i0 * d.nb[0]) =
                *(const int32_t *) (gk_cu_row(a, s.i1, s.i2, s.i3) + s.i0 * a.nb[0]);
        } else {
            gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, gk_cu_get(a, s.i0, s.i1, s.i2, s.i3));
        }
    }
}

static __global__ void gk_cu_k_get_rows(gk_tview a, gk_tview idx, gk_tview_mut d, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        // the index tensor is [ne1, ne2, ne3] against the destination's
        // (i1,i2,i3), so its dimensions are shifted down by one
        const int64_t r = (int64_t) *(const int32_t *) (idx.data
                + x.i1 * idx.nb[0] + x.i2 * idx.nb[1] + x.i3 * idx.nb[2]);

        if (a.type == GKT_I32 && d.type == GKT_I32) {
            // a gather of token ids stays in integers rather than crossing
            // through the float accessors, whose round trip is only exact
            // below 2^24
            *(int32_t *) (gk_cu_row(d, x.i1, x.i2, x.i3) + x.i0 * d.nb[0]) =
                *(const int32_t *) (gk_cu_row(a, r, x.i2, x.i3) + x.i0 * a.nb[0]);
        } else {
            gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, gk_cu_get(a, x.i0, r, x.i2, x.i3));
        }
    }
}

// Writes rows of src0 into the destination at the row indices src1 names -
// how a KV cache is filled. The destination keeps its own type.
static __global__ void gk_cu_k_set_rows(gk_tview b, gk_tview c, gk_tview_mut d,
                                        bool idx_is_64, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, b.ne);

        const char * pi = c.data + x.i1 * c.nb[0]
                        + (x.i2 % c.ne[1]) * c.nb[1] + (x.i3 % c.ne[2]) * c.nb[2];
        const int64_t row = idx_is_64 ? *(const int64_t *) pi : (int64_t) *(const int32_t *) pi;

        gk_cu_set(d, x.i0, row, x.i2, x.i3, gk_cu_get(b, x.i0, x.i1, x.i2, x.i3));
    }
}

static __global__ void gk_cu_k_repeat(gk_tview a, gk_tview_mut d, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  gk_cu_get(a, x.i0 % a.ne[0], x.i1 % a.ne[1], x.i2 % a.ne[2], x.i3 % a.ne[3]));
    }
}

static __global__ void gk_cu_k_concat(gk_tview a, gk_tview b, gk_tview_mut d,
                                      int dim, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        gk_cu_idx x = gk_cu_decompose(k, d.ne);

        int64_t j0 = x.i0, j1 = x.i1, j2 = x.i2, j3 = x.i3;
        bool second = false;

        switch (dim) {
            case 0: second = x.i0 >= a.ne[0]; if (second) j0 -= a.ne[0]; break;
            case 1: second = x.i1 >= a.ne[1]; if (second) j1 -= a.ne[1]; break;
            case 2: second = x.i2 >= a.ne[2]; if (second) j2 -= a.ne[2]; break;
            default:second = x.i3 >= a.ne[3]; if (second) j3 -= a.ne[3]; break;
        }

        const float v = second ? gk_cu_get(b, j0, j1, j2, j3)
                               : gk_cu_get(a, j0, j1, j2, j3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, v);
    }
}

// dst[i0,i1,i2] = a[i0,i1,i2] + b[i0, ids[i1,i2]] - the per-row bias shape a
// mixture-of-experts router produces.
static __global__ void gk_cu_k_add_id(gk_tview a, gk_tview b, gk_tview ids,
                                      gk_tview_mut d, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const int32_t row = *(const int32_t *) (ids.data + x.i1 * ids.nb[0] + x.i2 * ids.nb[1]);

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  gk_cu_get(a, x.i0, x.i1, x.i2, x.i3) + gk_cu_get(b, x.i0, row, 0, 0));
    }
}

// --------------------------------------------------------------------------
// softmax and masking
// --------------------------------------------------------------------------

#define GK_CU_SOFTMAX_BLOCK 256

static __global__ void gk_cu_k_soft_max(gk_tview a, gk_tview mask, bool has_mask,
                                        const float * sinks, gk_tview_mut d,
                                        float scale, float max_bias, int64_t n_head_log2) {
    __shared__ float scratch[GK_CU_SOFTMAX_BLOCK / GK_WARP_SIZE];

    const int64_t ir = blockIdx.x;
    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    const int64_t n = d.ne[0];
    const float slope = gk_cu_alibi_slope(max_bias, i2, n_head_log2);

    float local_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = gk_cu_get(a, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_cu_get(mask, i, i1, i2 % mask.ne[2], i3 % mask.ne[3]);
        }
        local_max = fmaxf(local_max, v);
    }

    float row_max = gk_cu_block_max(local_max, scratch);
    __syncthreads();

    // a sink is one extra virtual logit per head, normalised over but never
    // written out
    const float sink = sinks != NULL ? sinks[i2] : -INFINITY;
    if (sinks != NULL) {
        row_max = fmaxf(row_max, sink);
    }

    // A fully masked row is all -inf; exponentiating that is 0/0. Define it as
    // a uniform zero row, which is what a padded position needs.
    if (!isfinite(row_max)) {
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            gk_cu_set(d, i, i1, i2, i3, 0.0f);
        }
        return;
    }

    float local_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = gk_cu_get(a, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_cu_get(mask, i, i1, i2 % mask.ne[2], i3 % mask.ne[3]);
        }
        local_sum += expf(v - row_max);
    }

    float sum = gk_cu_block_sum(local_sum, scratch);
    if (sinks != NULL) {
        sum += expf(sink - row_max);
    }

    const float inv = 1.0f / sum;

    // The exponentials are recomputed rather than stored and read back. The
    // destination may be f16, and rounding through it before the normalisation
    // would put the answer a visible distance from the CPU's, which normalises
    // in f32 and converts once at the end.
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = gk_cu_get(a, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_cu_get(mask, i, i1, i2 % mask.ne[2], i3 % mask.ne[3]);
        }
        gk_cu_set(d, i, i1, i2, i3, expf(v - row_max) * inv);
    }
}

static __global__ void gk_cu_k_diag_mask(gk_tview a, gk_tview_mut d,
                                         int n_past, float fill, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, x.i0 > n_past + x.i1 ? fill : v);
    }
}

// --------------------------------------------------------------------------
// rotary position embedding
//
// One thread per rotated pair. The angle depends only on the position and the
// pair index, so it is recomputed per thread rather than cached per token: the
// two transcendentals are cheaper than the shared memory and the barrier a
// cache would cost, and it keeps every pair independent.
// --------------------------------------------------------------------------

struct gk_cu_rope_params {
    int   n_dims;
    int   mode;
    int   sections[4];
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    bool  neox;
    bool  mrope;
    bool  vision;
    bool  imrope;
};

static __device__ __forceinline__ float gk_cu_rope_ramp(float low, float high, int i) {
    const float y = ((float) (i / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ __forceinline__ void gk_cu_rope_angle(float theta_base, const gk_cu_rope_params & p,
                                                        int i, float * cos_out, float * sin_out) {
    float theta  = theta_base * p.freq_scale;
    float mscale = p.attn_factor;

    if (p.ext_factor != 0.0f) {
        const float ramp = gk_cu_rope_ramp(p.corr_dims[0], p.corr_dims[1], i) * p.ext_factor;

        // blend the scaled and unscaled angles across the correction band
        theta = theta * (1.0f - ramp) + theta_base * ramp;

        // YaRN's temperature correction for the entropy a stretched context adds
        mscale *= 1.0f + 0.1f * logf(1.0f / p.freq_scale);
    }

    *cos_out = cosf(theta) * mscale;
    *sin_out = sinf(theta) * mscale;
}

static __global__ void gk_cu_k_rope(gk_tview a, const int32_t * pos, const float * freq_factors,
                                    gk_tview_mut d, gk_cu_rope_params p, int64_t n_pairs) {
    const int64_t n_rows = d.ne[1] * d.ne[2] * d.ne[3];
    const int64_t n_rot  = p.vision ? d.ne[0] : p.n_dims;
    const int64_t pairs_per_row = n_rot / 2;

    GK_CU_FLAT_LOOP(n_pairs) {
        const int64_t ir = k / pairs_per_row;
        const int64_t ip = k % pairs_per_row;
        if (ir >= n_rows) {
            continue;
        }

        int64_t i1, i2, i3;
        gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

        const int64_t i0 = ip * 2;

        // Which angle this pair turns through. A single-axis rope advances one
        // angle geometrically; the multi-axis ropes carry four positions and
        // pick per pair by which section the pair falls in.
        float theta_base;
        if (!p.mrope) {
            theta_base = (float) pos[i2] * powf(p.theta_scale, (float) ip);
        } else {
            const int64_t n_pos = d.ne[2];
            const int sect_dims = p.sections[0] + p.sections[1] + p.sections[2] + p.sections[3];
            const int sec_w = p.sections[1] + p.sections[0];
            const int sec_e = p.sections[2] + sec_w;

            const int sector = (int) (ip % sect_dims);

            int axis;
            if (p.imrope) {
                if (sector % 3 == 1 && sector < 3 * p.sections[1])      axis = 1;
                else if (sector % 3 == 2 && sector < 3 * p.sections[2]) axis = 2;
                else if (sector % 3 == 0 && sector < 3 * p.sections[0]) axis = 0;
                else                                                    axis = 3;
            } else {
                if (sector < p.sections[0])                       axis = 0;
                else if (sector < sec_w)                          axis = 1;
                else if (sector < sec_w + p.sections[2])          axis = 2;
                else                                              axis = 3;
            }

            const float p_axis = (float) pos[i2 + n_pos * axis];

            // The vision rope restarts each axis's angle at its section, so a
            // pair's exponent counts from the start of its section rather than
            // from the start of the row.
            int step = (int) ip;
            if (p.vision) {
                const int base = axis == 0 ? 0
                               : axis == 1 ? p.sections[0]
                               : axis == 2 ? sec_w : sec_e;
                step = (int) ip - base;
            }

            theta_base = p_axis * powf(p.theta_scale, (float) step);
        }

        const float ff = freq_factors != NULL ? freq_factors[ip] : 1.0f;

        float cos_t, sin_t;
        gk_cu_rope_angle(theta_base / ff, p, (int) i0, &cos_t, &sin_t);

        // three pairings share the rotation: normal pairs (2i, 2i+1), neox and
        // the multi-axis ropes pair (i, i + n_dims/2), and the vision rope
        // pairs (i, i + n_dims) across the full width
        const int64_t offset = p.vision ? p.n_dims
                             : (p.neox || p.mrope) ? p.n_dims / 2 : 1;
        const int64_t ic = (p.neox || p.mrope) ? ip : i0;

        const float x0 = gk_cu_get(a, ic, i1, i2, i3);
        const float x1 = gk_cu_get(a, ic + offset, i1, i2, i3);

        gk_cu_set(d, ic,          i1, i2, i3, x0 * cos_t - x1 * sin_t);
        gk_cu_set(d, ic + offset, i1, i2, i3, x0 * sin_t + x1 * cos_t);
    }
}

// Channels at or past n_dims are not rotated and pass through. A separate
// kernel rather than a branch in the one above, so the rotation kernel stays
// one thread per pair with no idle lanes.
static __global__ void gk_cu_k_rope_passthrough(gk_tview a, gk_tview_mut d,
                                                int n_dims, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const int64_t width = d.ne[0] - n_dims;
        const int64_t ir = k / width;
        const int64_t i0 = n_dims + k % width;

        int64_t i1, i2, i3;
        gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

        gk_cu_set(d, i0, i1, i2, i3, gk_cu_get(a, i0, i1, i2, i3));
    }
}

// --------------------------------------------------------------------------
// reductions along a row
// --------------------------------------------------------------------------

static __global__ void gk_cu_k_sum_rows(gk_tview a, gk_tview_mut d, bool mean) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    const int64_t ir = blockIdx.x;
    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    float local = 0.0f;
    for (int64_t i = threadIdx.x; i < a.ne[0]; i += blockDim.x) {
        local += gk_cu_get(a, i, i1, i2, i3);
    }

    const float total = gk_cu_block_sum(local, scratch);

    if (threadIdx.x == 0) {
        gk_cu_set(d, 0, i1, i2, i3, mean ? total / (float) a.ne[0] : total);
    }
}

// Every element into one scalar. A single block is enough: the callers are
// the samplers reducing a vocab-sized mask, a few hundred loop iterations
// per thread, and one block spares the cross-block combine.
static __global__ void gk_cu_k_sum(gk_tview a, gk_tview_mut d, int64_t n) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    float local = 0.0f;
    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
        const gk_cu_idx x = gk_cu_decompose(k, a.ne);
        local += gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
    }

    const float total = gk_cu_block_sum(local, scratch);

    if (threadIdx.x == 0) {
        gk_cu_set(d, 0, 0, 0, 0, total);
    }
}

// --------------------------------------------------------------------------
// sorting and selection along a row
//
// Both ops answer with indices, so the destination is i32 and nothing here
// goes through gk_cu_set.
//
// The order is on the pair (value, index), not on the value alone: equal
// values keep their index order, which makes the result the same stable
// ordering the CPU's insertion sort produces. A mixture-of-experts router is
// the caller that matters, and two backends that break a tie differently
// route a token to different experts - a divergence that looks like a
// numerical bug and is not one.
//
// A padded slot is given the value that sorts last, and its index is >= n and
// therefore larger than any real one, so it loses the tie against a real slot
// holding the same value. -inf rows are the reason this needs saying: a fully
// masked router logit row is all -inf, and descending order would otherwise
// have padding displace real indices.
// --------------------------------------------------------------------------

#define GK_CU_SORT_MAX_BLOCK 256

// The widest row the in-block network takes. It holds one value and one index
// per padded slot, so this is 32 KB of shared memory - inside the 48 KB every
// target gives a block without opting in to more.
#define GK_CU_SORT_MAX_PAD 4096

struct gk_cu_sort_smem {
    float   * val;
    int32_t * idx;
};

// True if slot p sorts before slot q.
static __device__ __forceinline__ bool gk_cu_sort_before(const gk_cu_sort_smem & s,
                                                         int p, int q, bool desc) {
    const float vp = s.val[p];
    const float vq = s.val[q];

    if (vp != vq) {
        return desc ? vp > vq : vp < vq;
    }
    return s.idx[p] < s.idx[q];
}

// The network itself, over slots already staged in shared memory. Factored out
// because three callers want it: the whole-row sort below, and the two halves
// of the chunked selection further down.
//
// Every compare-exchange within a stage is independent, which is the shape a
// block wants; the stages are log2(n_pad)*(log2(n_pad)+1)/2 of them, each
// ending in a barrier.
static __device__ __forceinline__ void gk_cu_sort_network(const gk_cu_sort_smem & s,
                                                          int n_pad, bool desc) {
    for (int stage = 2; stage <= n_pad; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step >>= 1) {
            for (int i = threadIdx.x; i < n_pad; i += blockDim.x) {
                const int j = i ^ step;
                if (j <= i) {
                    continue; // the other half of the pair does this one
                }

                // Halves of a stage are built in opposite directions so the
                // next stage sees a bitonic sequence; only the last stage's
                // direction is the answer's.
                const bool up   = (i & stage) == 0;
                const bool swap = up ? gk_cu_sort_before(s, j, i, desc)
                                     : gk_cu_sort_before(s, i, j, desc);
                if (swap) {
                    const float   tv = s.val[i]; s.val[i] = s.val[j]; s.val[j] = tv;
                    const int32_t ti = s.idx[i]; s.idx[i] = s.idx[j]; s.idx[j] = ti;
                }
            }
            __syncthreads();
        }
    }
}

// A bitonic network in shared memory: log2(n_pad)*(log2(n_pad)+1)/2 stages,
// every compare-exchange within a stage independent, which is the shape a
// block wants. One block owns a row for the whole network, so the rows are
// walked with a grid stride rather than one block each - a grid dimension is
// capped and a token count is not.
static __global__ void gk_cu_k_argsort(gk_tview a, gk_tview_mut d,
                                       int64_t n, int n_pad, int64_t k_out,
                                       bool desc, bool scramble, int64_t n_rows) {
    extern __shared__ char gk_cu_sort_buf[];

    gk_cu_sort_smem s;
    s.val = (float *)   gk_cu_sort_buf;
    s.idx = (int32_t *) (s.val + n_pad);

    const float sentinel = desc ? -INFINITY : INFINITY;

    for (int64_t ir = blockIdx.x; ir < n_rows; ir += gridDim.x) {
        int64_t i1, i2, i3;
        gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

        // the previous row's write-out is still reading the buffer
        __syncthreads();

        for (int i = threadIdx.x; i < n_pad; i += blockDim.x) {
            s.idx[i] = i;
            s.val[i] = i < n ? gk_cu_get(a, i, i1, i2, i3) : sentinel;
        }
        __syncthreads();

        gk_cu_sort_network(s, n_pad, desc);

        // top_k promises no order, and says so by breaking the one it happens
        // to have - the same swap the CPU pass makes, so the two agree.
        if (scramble && k_out > 1 && threadIdx.x == 0) {
            const int32_t t = s.idx[0]; s.idx[0] = s.idx[1]; s.idx[1] = t;
        }
        __syncthreads();

        for (int64_t i = threadIdx.x; i < k_out; i += blockDim.x) {
            *(int32_t *) (gk_cu_row(d, i1, i2, i3) + i * d.nb[0]) = s.idx[i];
        }
    }
}

// Rows too wide for the network fall back to counting: a slot's position is
// the number of slots that sort before it, which each thread can work out
// alone with no shared state. Quadratic in the row width and only reachable
// above GK_CU_SORT_MAX_PAD, which no router row comes near - it is here so
// that a wide row is slow rather than unsupported, because the single-backend
// caller has no CPU to fall back to.
static __global__ void gk_cu_k_argsort_rank(gk_tview a, gk_tview_mut d,
                                            int64_t n, int64_t k_out,
                                            bool desc, bool scramble, int64_t n_rows) {
    for (int64_t ir = blockIdx.x; ir < n_rows; ir += gridDim.x) {
        int64_t i1, i2, i3;
        gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float vi = gk_cu_get(a, i, i1, i2, i3);

            int64_t rank = 0;
            for (int64_t j = 0; j < n; ++j) {
                if (j == i) {
                    continue;
                }
                const float vj = gk_cu_get(a, j, i1, i2, i3);
                const bool before = vj != vi ? (desc ? vj > vi : vj < vi) : j < i;
                rank += before ? 1 : 0;
            }

            if (rank < k_out) {
                int64_t slot = rank;
                if (scramble && k_out > 1 && rank < 2) {
                    slot = 1 - rank;
                }
                *(int32_t *) (gk_cu_row(d, i1, i2, i3) + slot * d.nb[0]) = (int32_t) i;
            }
        }
    }
}

// --------------------------------------------------------------------------
// reflected padding, argmax, and the prefix sum
// --------------------------------------------------------------------------

// The reflection: the pad on each side mirrors the row about its end element,
// which is not repeated. Written as a map from the output position back to the
// input one, so it stays one thread per output element.
static __global__ void gk_cu_k_pad_reflect_1d(gk_tview a, gk_tview_mut d,
                                              int p0, int p1, int64_t n, int64_t total) {
    GK_CU_FLAT_LOOP(total) {
        const gk_cu_idx o = gk_cu_decompose(k, d.ne);

        int64_t src;
        if (o.i0 < p0) {
            src = p0 - o.i0;              // left pad mirrors forwards
        } else if (o.i0 < p0 + n) {
            src = o.i0 - p0;              // the row itself
        } else {
            src = 2 * n - 2 - (o.i0 - p0); // right pad mirrors backwards
        }

        gk_cu_set(d, o.i0, o.i1, o.i2, o.i3, gk_cu_get(a, src, o.i1, o.i2, o.i3));
    }
}

// One block per row. Ties keep the lowest index, which is what the CPU pass's
// strictly-greater comparison gives, so the reduction has to break them the
// same way rather than take whichever half it saw first.
static __global__ void gk_cu_k_argmax(gk_tview a, int32_t * d, int64_t n) {
    extern __shared__ char gk_cu_argmax_buf[];

    float   * s_val = (float *)   gk_cu_argmax_buf;
    int32_t * s_idx = (int32_t *) (s_val + blockDim.x);

    const int64_t i1 = blockIdx.x;

    float   best  = -INFINITY;
    int32_t bestx = 0;

    // A thread walks its own indices in increasing order, so a strict
    // comparison already keeps the lowest of any it sees.
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = gk_cu_get(a, i, i1, 0, 0);
        if (v > best) {
            best  = v;
            bestx = (int32_t) i;
        }
    }

    s_val[threadIdx.x] = best;
    s_idx[threadIdx.x] = bestx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float   ov = s_val[threadIdx.x + stride];
            const int32_t ox = s_idx[threadIdx.x + stride];
            const float   cv = s_val[threadIdx.x];
            const int32_t cx = s_idx[threadIdx.x];

            if (ov > cv || (ov == cv && ox < cx)) {
                s_val[threadIdx.x] = ov;
                s_idx[threadIdx.x] = ox;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d[i1] = s_idx[0];
    }
}

// The prefix sum, in three passes.
//
// A prefix sum is sequential by definition, and doing it that way - one block
// walking a row - would leave a vocabulary row to a single multiprocessor. So
// the row is cut into chunks: pass one totals each chunk, pass two scans those
// totals, pass three scans each chunk starting from its total's prefix. Only
// pass two is serial, and it is over the chunk count rather than the row.
#define GK_CU_SCAN_BLOCK 256
#define GK_CU_SCAN_CHUNK (GK_CU_SCAN_BLOCK * 16)

// An inclusive scan of one value per thread, across the block. Returns the
// running total of the whole block in `total` for the caller to carry.
static __device__ __forceinline__ float gk_cu_block_scan(float v, float * sh, float * total) {
    sh[threadIdx.x] = v;
    __syncthreads();

    for (int off = 1; off < blockDim.x; off <<= 1) {
        const float add = threadIdx.x >= (unsigned) off ? sh[threadIdx.x - off] : 0.0f;
        __syncthreads();
        sh[threadIdx.x] += add;
        __syncthreads();
    }

    *total = sh[blockDim.x - 1];
    return sh[threadIdx.x];
}

static __global__ void gk_cu_k_cumsum_totals(gk_tview a, float * sums,
                                             int64_t n, int n_chunks) {
    extern __shared__ float gk_cu_scan_sh[];

    const int64_t ir = blockIdx.y;
    const int64_t ic = blockIdx.x;

    int64_t i1, i2, i3;
    gk_cu_unrow(ir, a.ne, &i1, &i2, &i3);

    const int64_t base = ic * (int64_t) GK_CU_SCAN_CHUNK;

    float acc = 0.0f;
    for (int64_t o = threadIdx.x; o < GK_CU_SCAN_CHUNK; o += blockDim.x) {
        const int64_t i = base + o;
        if (i < n) {
            acc += gk_cu_get(a, i, i1, i2, i3);
        }
    }

    const float t = gk_cu_block_sum(acc, gk_cu_scan_sh);

    if (threadIdx.x == 0) {
        sums[ir * n_chunks + ic] = t;
    }
}

// The serial pass, over chunk totals rather than elements. One thread: the
// count is small and a fixed order is what keeps two runs identical.
static __global__ void gk_cu_k_cumsum_offsets(float * sums, int n_chunks) {
    const int64_t ir = blockIdx.x;

    float run = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float v = sums[ir * n_chunks + c];
        sums[ir * n_chunks + c] = run;  // exclusive: what precedes this chunk
        run += v;
    }
}

static __global__ void gk_cu_k_cumsum_apply(gk_tview a, gk_tview_mut d,
                                            const float * sums, int64_t n, int n_chunks) {
    extern __shared__ float gk_cu_scan_sh[];

    const int64_t ir = blockIdx.y;
    const int64_t ic = blockIdx.x;

    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    const int64_t base = ic * (int64_t) GK_CU_SCAN_CHUNK;

    float carry = sums[ir * n_chunks + ic];

    for (int64_t o = 0; o < GK_CU_SCAN_CHUNK; o += blockDim.x) {
        const int64_t i = base + o + threadIdx.x;
        const float v = i < n ? gk_cu_get(a, i, i1, i2, i3) : 0.0f;

        float total = 0.0f;
        const float scanned = gk_cu_block_scan(v, gk_cu_scan_sh, &total);

        if (i < n) {
            gk_cu_set(d, i, i1, i2, i3, carry + scanned);
        }

        carry += total;
        __syncthreads(); // the next sub-chunk reuses the scan buffer
    }
}

// --------------------------------------------------------------------------
// argsort on a row too wide for one network
//
// top_k below gets to work in rounds because a selection composes: what is not
// in the top k of its chunk is in nobody's. A sort does not - every element's
// final position depends on every other - so the whole row has to go through
// one network, and above 4096 slots that network does not fit in shared
// memory.
//
// So it runs out of global memory instead. The same bitonic network, the same
// compare-exchanges in the same order, but each step is a kernel launch over a
// scratch copy of the row rather than a loop over shared memory.
//
// The saving grace is the tail. A step exchanges slot i with slot i^step, so
// once `step` is smaller than half a block's span every exchange that remains
// in the stage is inside one block - and all of them can be done in shared
// memory in a single launch. That collapses a quadratic-looking count of
// launches into a manageable one: a 262144-wide row is 18 stages and 171
// steps, of which 28 need a global pass and the rest ride along in 18 local
// ones.
// --------------------------------------------------------------------------

// Elements one block owns during the local tail. 2048 slots is 16 KB of
// shared memory, which leaves room for several blocks per multiprocessor.
#define GK_CU_SORT_SPAN 2048

// One compare-exchange step over the whole row, in global memory. Used only
// while `step` is too wide for a block to own both sides of the pair.
static __global__ void gk_cu_k_sort_step(float * val, int32_t * idx,
                                         int n_pad, int stage, int step,
                                         bool desc, int64_t n_rows) {
    const int64_t pairs = (int64_t) n_pad / 2;
    const int64_t t     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= pairs * n_rows) {
        return;
    }

    const int64_t ir = t / pairs;
    const int64_t p  = t % pairs;

    // Dense pair index to the low slot of the pair: `step` consecutive slots
    // belong to one side, then `step` to the other.
    const int64_t i = (p / step) * 2 * step + (p % step);
    const int64_t j = i + step;

    float   * v = val + ir * n_pad;
    int32_t * x = idx + ir * n_pad;

    const float vi = v[i], vj = v[j];
    const int32_t xi = x[i], xj = x[j];

    // Halves of a stage are built in opposite directions so the next stage
    // sees a bitonic sequence. Same rule and same tie-break as the in-block
    // network, because this is the same sort.
    const bool up = (i & stage) == 0;

    bool before; // does the value at j sort before the one at i?
    if (vi != vj) {
        before = desc ? vj > vi : vj < vi;
    } else {
        before = xj < xi;
    }

    if (before == up) {
        v[i] = vj; v[j] = vi;
        x[i] = xj; x[j] = xi;
    }
}

// The rest of a stage, once every remaining exchange is inside one block's
// span. Runs all steps from `step` down to 1 without leaving shared memory.
static __global__ void gk_cu_k_sort_tail(float * val, int32_t * idx,
                                         int n_pad, int stage, int step, bool desc) {
    __shared__ float   s_val[GK_CU_SORT_SPAN];
    __shared__ int32_t s_idx[GK_CU_SORT_SPAN];

    const int64_t ir   = blockIdx.y;
    const int64_t base = (int64_t) blockIdx.x * GK_CU_SORT_SPAN;

    float   * v = val + ir * n_pad;
    int32_t * x = idx + ir * n_pad;

    for (int e = threadIdx.x; e < GK_CU_SORT_SPAN; e += blockDim.x) {
        s_val[e] = v[base + e];
        s_idx[e] = x[base + e];
    }
    __syncthreads();

    for (int st = step; st > 0; st >>= 1) {
        for (int e = threadIdx.x; e < GK_CU_SORT_SPAN / 2; e += blockDim.x) {
            const int local = (e / st) * 2 * st + (e % st);
            const int other = local + st;

            // The direction is a property of the slot's position in the whole
            // row, not in this block, so it is taken from the global index.
            const bool up = (((int64_t) base + local) & stage) == 0;

            const float vi = s_val[local], vj = s_val[other];
            const int32_t xi = s_idx[local], xj = s_idx[other];

            bool before;
            if (vi != vj) {
                before = desc ? vj > vi : vj < vi;
            } else {
                before = xj < xi;
            }

            if (before == up) {
                s_val[local] = vj; s_val[other] = vi;
                s_idx[local] = xj; s_idx[other] = xi;
            }
        }
        __syncthreads();
    }

    for (int e = threadIdx.x; e < GK_CU_SORT_SPAN; e += blockDim.x) {
        v[base + e] = s_val[e];
        x[base + e] = s_idx[e];
    }
}

// Stages the row into scratch, padded to a power of two with sentinels that
// lose every comparison so they land past the real elements.
static __global__ void gk_cu_k_sort_stage(gk_tview a, float * val, int32_t * idx,
                                          int64_t n, int n_pad, bool desc, int64_t n_rows) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t) n_pad * n_rows) {
        return;
    }

    const int64_t ir = t / n_pad;
    const int64_t i  = t % n_pad;

    int64_t i1, i2, i3;
    gk_cu_unrow(ir, a.ne, &i1, &i2, &i3);

    const bool real = i < n;
    val[t] = real ? gk_cu_get(a, i, i1, i2, i3) : (desc ? -INFINITY : INFINITY);
    idx[t] = real ? (int32_t) i : INT32_MAX;
}

// The sorted indices back out, dropping the padding.
static __global__ void gk_cu_k_sort_emit(const int32_t * idx, gk_tview_mut d,
                                         int n_pad, int64_t k_out, int64_t n_rows) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= k_out * n_rows) {
        return;
    }

    const int64_t ir = t / k_out;
    const int64_t i  = t % k_out;

    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    *(int32_t *) (gk_cu_row(d, i1, i2, i3) + i * d.nb[0]) = idx[ir * n_pad + i];
}

// --------------------------------------------------------------------------
// top_k on a row too wide for one network
//
// The network above holds 4096 slots. A router row is 128 wide and fits with
// room to spare; a vocabulary row is 262144 and does not, and what it used to
// fall to was the ranking kernel - every element compared against every other,
// in one block. Measured, that is 31 ms at a row of 8192 and grows with the
// square: a real vocabulary row would have taken something like thirty
// seconds, per token.
//
// But top_k is a selection, not a sort, and a selection composes. The k
// largest of a row are the k largest of {the k largest of each chunk of it},
// whatever the chunks are - anything not in the top k of its own chunk cannot
// be in the top k overall, because the k elements that beat it in its chunk
// beat it everywhere. So the row is cut into chunks the network does fit, each
// block reduces its chunk to k candidates, and the round repeats on those
// until what is left fits in one network.
//
// One round takes 262144 down to 2560 at k of 40, so in practice this is two
// launches and never a third.
//
// The candidates travel as (value, index) pairs: the value because the next
// round has to compare them, the index because it is what the op returns and
// the position in the candidate array is not it.
// --------------------------------------------------------------------------

// Chunks are the widest the network takes, so that a round sheds as much as it
// can. The reduction per round is k/chunk, so a small k converges at once.
#define GK_CU_TOPK_CHUNK GK_CU_SORT_MAX_PAD

// A round has to shrink the candidate list or the loop does not terminate.
// Emitting k per chunk out of GK_CU_TOPK_CHUNK shrinks by k/chunk, so a k at
// least half the chunk makes no useful progress and the caller keeps the old
// path. No sampler or router comes near this - it is a guard, not a case.
#define GK_CU_TOPK_MAX_K (GK_CU_TOPK_CHUNK / 2)

// The sentinel a short chunk pads with. Every chunk emits the same count so
// the next round can index its input without a per-chunk length: -inf loses
// every comparison on value, and an index above any real one loses the tie
// against a real element that is also -inf - which a fully masked row is made
// of.
#define GK_CU_TOPK_PAD_IDX INT32_MAX

// One round. Reads either the source row (round 0, `in_val` NULL) or the
// candidates a previous round left, and writes `m` candidates per chunk.
static __global__ void gk_cu_k_top_k_chunk(gk_tview a,
                                           const float * in_val, const int32_t * in_idx,
                                           int64_t cur_n,
                                           float * out_val, int32_t * out_idx,
                                           int n_pad, int m, int n_chunks) {
    extern __shared__ char gk_cu_topk_buf[];

    gk_cu_sort_smem s;
    s.val = (float *)   gk_cu_topk_buf;
    s.idx = (int32_t *) (s.val + n_pad);

    const int64_t ir = blockIdx.x;  // which row
    const int64_t ic = blockIdx.y;  // which chunk of it

    int64_t i1, i2, i3;
    gk_cu_unrow(ir, a.ne, &i1, &i2, &i3);

    const int64_t base = ic * GK_CU_TOPK_CHUNK;
    const int64_t len  = cur_n - base < GK_CU_TOPK_CHUNK ? cur_n - base : GK_CU_TOPK_CHUNK;

    for (int i = threadIdx.x; i < n_pad; i += blockDim.x) {
        if (i < len) {
            if (in_val == NULL) {
                s.val[i] = gk_cu_get(a, base + i, i1, i2, i3);
                s.idx[i] = (int32_t) (base + i);
            } else {
                s.val[i] = in_val[ir * cur_n + base + i];
                s.idx[i] = in_idx[ir * cur_n + base + i];
            }
        } else {
            s.val[i] = -INFINITY;
            s.idx[i] = GK_CU_TOPK_PAD_IDX;
        }
    }
    __syncthreads();

    gk_cu_sort_network(s, n_pad, true);

    // The top m of this chunk, at this chunk's slot in the next round's input.
    const int64_t out_base = ir * ((int64_t) n_chunks * m) + ic * m;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        out_val[out_base + i] = s.val[i];
        out_idx[out_base + i] = s.idx[i];
    }
}

// The last round: what is left fits one network, so this sorts it and writes
// the answer. Same tie-break and same deliberate scrambling of the first two
// slots as the whole-row kernel, because this is the same op.
static __global__ void gk_cu_k_top_k_final(const float * in_val, const int32_t * in_idx,
                                           int64_t cur_n, gk_tview_mut d,
                                           int n_pad, int64_t k_out) {
    extern __shared__ char gk_cu_topk_buf[];

    gk_cu_sort_smem s;
    s.val = (float *)   gk_cu_topk_buf;
    s.idx = (int32_t *) (s.val + n_pad);

    const int64_t ir = blockIdx.x;

    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    for (int i = threadIdx.x; i < n_pad; i += blockDim.x) {
        const bool real = i < cur_n;
        s.val[i] = real ? in_val[ir * cur_n + i] : -INFINITY;
        s.idx[i] = real ? in_idx[ir * cur_n + i] : GK_CU_TOPK_PAD_IDX;
    }
    __syncthreads();

    gk_cu_sort_network(s, n_pad, true);

    if (k_out > 1 && threadIdx.x == 0) {
        const int32_t t = s.idx[0]; s.idx[0] = s.idx[1]; s.idx[1] = t;
    }
    __syncthreads();

    for (int64_t i = threadIdx.x; i < k_out; i += blockDim.x) {
        *(int32_t *) (gk_cu_row(d, i1, i2, i3) + i * d.nb[0]) = s.idx[i];
    }
}

// Drives the out-of-core network. Returns false if the scratch it needs cannot
// be had, which leaves the caller to fall back rather than fail.
static bool gk_cu_argsort_wide(gkStream_t stream, struct gk_cuda_scratch * scratch,
                               const struct gk_tensor * src0, struct gk_tensor * node,
                               int64_t rows, int64_t n, int64_t k_out, bool desc) {
    if (scratch == NULL) {
        return false;
    }

    int64_t n_pad = 1;
    while (n_pad < n) {
        n_pad <<= 1;
    }
    // The tail kernel gives every block a full span, so the row has to hold a
    // whole number of them.
    if (n_pad < GK_CU_SORT_SPAN) {
        n_pad = GK_CU_SORT_SPAN;
    }

    const size_t needed = (size_t) rows * n_pad * (sizeof(float) + sizeof(int32_t));

    char * buf = (char *) gk_cu_scratch_get(scratch, needed, stream);
    if (buf == NULL) {
        return false;
    }

    float   * val = (float *) buf;
    int32_t * idx = (int32_t *) (val + (size_t) rows * n_pad);

    const int block = GK_CU_SORT_MAX_BLOCK;

    {
        const int64_t total = (int64_t) n_pad * rows;
        gk_cu_k_sort_stage<<<(unsigned) ((total + block - 1) / block), block, 0, stream>>>(
            gk_cu_view(src0), val, idx, n, (int) n_pad, desc, rows);
    }

    const int64_t pairs = (int64_t) n_pad / 2 * rows;
    const unsigned pair_grid = (unsigned) ((pairs + block - 1) / block);

    dim3 tail_grid;
    tail_grid.x = (unsigned) (n_pad / GK_CU_SORT_SPAN);
    tail_grid.y = (unsigned) rows;
    tail_grid.z = 1;

    for (int stage = 2; stage <= n_pad; stage <<= 1) {
        int step = stage >> 1;

        // Wide steps reach outside any one block and have to go through
        // global memory, one launch each.
        while (step > GK_CU_SORT_SPAN / 2) {
            gk_cu_k_sort_step<<<pair_grid, block, 0, stream>>>(
                val, idx, (int) n_pad, stage, step, desc, rows);
            step >>= 1;
        }

        // Everything left in this stage is inside a span, so it is one launch
        // however many steps remain.
        gk_cu_k_sort_tail<<<tail_grid, block, 0, stream>>>(
            val, idx, (int) n_pad, stage, step, desc);
    }

    {
        const int64_t total = k_out * rows;
        gk_cu_k_sort_emit<<<(unsigned) ((total + block - 1) / block), block, 0, stream>>>(
            idx, gk_cu_view_mut(node), (int) n_pad, k_out, rows);
    }

    return true;
}

// Rounds until what is left fits one network. Returns false if the scratch it
// needs cannot be had, which leaves the caller to fall back rather than fail.
static bool gk_cu_top_k_wide(gkStream_t stream, struct gk_cuda_scratch * scratch,
                             const struct gk_tensor * src0, struct gk_tensor * node,
                             int64_t rows, int64_t n, int64_t k_out) {
    if (scratch == NULL) {
        return false;
    }

    const int m = (int) k_out;

    // Two buffers, alternating: a round reads one and writes the other. The
    // first round's output is the largest, so both are sized to it.
    const int64_t n_chunks0 = (n + GK_CU_TOPK_CHUNK - 1) / GK_CU_TOPK_CHUNK;
    const int64_t cap       = rows * n_chunks0 * m;

    const size_t pair   = sizeof(float) + sizeof(int32_t);
    const size_t needed = 2 * (size_t) cap * pair;

    char * buf = (char *) gk_cu_scratch_get(scratch, needed, stream);
    if (buf == NULL) {
        return false;
    }

    float   * val[2] = { (float *) buf, (float *) (buf + (size_t) cap * pair) };
    int32_t * idx[2] = { (int32_t *) (val[0] + cap), (int32_t *) (val[1] + cap) };

    // Threads enough for half the network's slots, since each pair is handled
    // once, and no more than a chunk can use.
    const int block = GK_CU_SORT_MAX_BLOCK;
    const size_t smem_chunk = (size_t) GK_CU_TOPK_CHUNK * pair;

    int64_t cur_n = n;
    int     cur   = 0;   // which buffer the next round writes
    bool    first = true;

    while (cur_n > GK_CU_SORT_MAX_PAD) {
        const int64_t n_chunks = (cur_n + GK_CU_TOPK_CHUNK - 1) / GK_CU_TOPK_CHUNK;

        dim3 grid;
        grid.x = (unsigned) rows;
        grid.y = (unsigned) n_chunks;
        grid.z = 1;

        gk_cu_k_top_k_chunk<<<grid, block, smem_chunk, stream>>>(
            gk_cu_view(src0),
            first ? NULL : val[1 - cur], first ? NULL : idx[1 - cur],
            cur_n, val[cur], idx[cur],
            GK_CU_TOPK_CHUNK, m, (int) n_chunks);

        cur_n = n_chunks * m;
        cur   = 1 - cur;
        first = false;
    }

    int n_pad = 1;
    while (n_pad < cur_n) {
        n_pad <<= 1;
    }

    gk_cu_k_top_k_final<<<(unsigned) rows, block,
                          (size_t) n_pad * pair, stream>>>(
        val[1 - cur], idx[1 - cur], cur_n, gk_cu_view_mut(node), n_pad, k_out);

    return true;
}

// --------------------------------------------------------------------------
// padding, resampling and the diffusion helpers
// --------------------------------------------------------------------------

static __device__ __forceinline__ int64_t gk_cu_wrap(int64_t i, int64_t n) {
    return ((i % n) + n) % n;
}

static __global__ void gk_cu_k_pad(gk_tview a, gk_tview_mut d,
                                   int lp0, int rp0, int lp1, int rp1,
                                   int lp2, int rp2, int lp3, int rp3,
                                   bool circular, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        if (circular) {
            gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                      gk_cu_get(a, gk_cu_wrap(x.i0 - lp0, a.ne[0]),
                                   gk_cu_wrap(x.i1 - lp1, a.ne[1]),
                                   gk_cu_wrap(x.i2 - lp2, a.ne[2]),
                                   gk_cu_wrap(x.i3 - lp3, a.ne[3])));
            continue;
        }

        const bool inside =
            x.i0 >= lp0 && x.i0 < d.ne[0] - rp0 &&
            x.i1 >= lp1 && x.i1 < d.ne[1] - rp1 &&
            x.i2 >= lp2 && x.i2 < d.ne[2] - rp2 &&
            x.i3 >= lp3 && x.i3 < d.ne[3] - rp3;

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  inside ? gk_cu_get(a, x.i0 - lp0, x.i1 - lp1, x.i2 - lp2, x.i3 - lp3) : 0.0f);
    }
}

// Nearest and bilinear resampling. The index arithmetic - the truncation, the
// half-pixel offset, the border clamp - is part of the op's meaning: a
// projector was trained against exactly these positions.
static __global__ void gk_cu_k_upscale(gk_tview a, gk_tview_mut d, int mode,
                                       float sf0, float sf1, float sf2, float sf3,
                                       float pixel_offset, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const int64_t i02 = (int64_t) ((float) x.i2 / sf2);
        const int64_t i03 = (int64_t) ((float) x.i3 / sf3);

        if (mode == 0) { // nearest
            gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                      gk_cu_get(a, (int64_t) ((float) x.i0 / sf0),
                                   (int64_t) ((float) x.i1 / sf1), i02, i03));
            continue;
        }

        const float fy = ((float) x.i1 + pixel_offset) / sf1 - pixel_offset;
        int64_t y0 = (int64_t) floorf(fy);
        int64_t y1 = y0 + 1;
        float dy = fmaxf(0.0f, fminf(fy - (float) y0, 1.0f));
        y0 = max((int64_t) 0, min(y0, a.ne[1] - 1));
        y1 = max((int64_t) 0, min(y1, a.ne[1] - 1));

        const float fx = ((float) x.i0 + pixel_offset) / sf0 - pixel_offset;
        int64_t x0 = (int64_t) floorf(fx);
        int64_t x1 = x0 + 1;
        float dx = fmaxf(0.0f, fminf(fx - (float) x0, 1.0f));
        x0 = max((int64_t) 0, min(x0, a.ne[0] - 1));
        x1 = max((int64_t) 0, min(x1, a.ne[0] - 1));

        const float p00 = gk_cu_get(a, x0, y0, i02, i03);
        const float p10 = gk_cu_get(a, x1, y0, i02, i03);
        const float p01 = gk_cu_get(a, x0, y1, i02, i03);
        const float p11 = gk_cu_get(a, x1, y1, i02, i03);

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  p00 * (1 - dx) * (1 - dy) + p10 * dx * (1 - dy)
                + p01 * (1 - dx) * dy       + p11 * dx * dy);
    }
}

static __global__ void gk_cu_k_timestep_embedding(const float * timesteps, gk_tview_mut d,
                                                  int dim, int max_period, int64_t n) {
    const int half = dim / 2;

    GK_CU_FLAT_LOOP(n) {
        const int64_t i = k / half; // which timestep
        const int64_t j = k % half;

        const float t    = timesteps[i];
        const float freq = expf(-logf((float) max_period) * (float) j / (float) half);
        const float arg  = t * freq;

        gk_cu_set(d, j,        i, 0, 0, cosf(arg));
        gk_cu_set(d, j + half, i, 0, 0, sinf(arg));
    }
}

static __global__ void gk_cu_k_arange(float * dst, float start, float step, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        dst[k] = start + step * (float) k;
    }
}

// im2col: unroll each input patch into a row so a convolution becomes a
// matmul. One thread per output cell.
// The same decomposition with the divisions replaced by the multiply-shifts
// the host worked out for these extents.
//
// This kernel is most of a convolution. Every conv in a UNet or a VAE lowers to
// im2col plus a matmul, the buffer it writes is the kernel area larger than the
// image it reads - nine times, for the 3x3 that both are built from - and the
// destination is contiguous, so the stores are already coalesced and the whole
// thing should run at the speed of writing that buffer. It did not: five 64-bit
// divisions per output element put several hundred instructions between one
// store and the next, and measured against the bandwidth it moves, a VAE-sized
// im2col was four times slower than its own memory traffic.
//
// Nothing else changes. The traversal order is the one below, for the same
// reason: `at` varies fastest, so a warp writes 32 consecutive elements of the
// patch dimension, and the reads it scatters across the image are the ones a
// neighbouring warp is reading too.
// One output element: where it comes from, and whether it comes from anywhere.
struct gk_cu_im2col_geom {
    struct gk_cu_fastdiv patch, ow, oh, kw, kh;
    int64_t IH, IW, ofs_n, ofs_c, ofs_h;
    int     s0, s1, p0, p1, d0, d1;
};

static __device__ __forceinline__ float gk_cu_im2col_gather(const gk_tview & img,
                                                            const struct gk_cu_im2col_geom & g,
                                                            uint32_t k) {
    uint32_t at, iow, ioh, ikw, ikh;

    const uint32_t cell = gk_cu_fastdiv_qr(g.patch, k,    &at);
    const uint32_t ohn  = gk_cu_fastdiv_qr(g.ow,    cell, &iow);
    const uint32_t in   = gk_cu_fastdiv_qr(g.oh,    ohn,  &ioh);

    const uint32_t khc  = gk_cu_fastdiv_qr(g.kw,    at,   &ikw);
    const uint32_t iic  = gk_cu_fastdiv_qr(g.kh,    khc,  &ikh);

    const int64_t iiw = (int64_t) iow * g.s0 + (int64_t) ikw * g.d0 - g.p0;
    const int64_t iih = (int64_t) ioh * g.s1 + (int64_t) ikh * g.d1 - g.p1;

    if (iih < 0 || iih >= g.IH || iiw < 0 || iiw >= g.IW) {
        return 0.0f;
    }

    // __ldg because this is a gather the compiler cannot prove disjoint from
    // the destination, and the read-only path is the one that suits a load
    // every lane of a warp takes from a different row.
    return __ldg((const float *) (img.data + in * g.ofs_n + iic * g.ofs_c
                                  + iih * g.ofs_h + iiw * img.nb[0]));
}

static __device__ __forceinline__ void gk_cu_im2col_store(const gk_tview_mut & d,
                                                          uint32_t k, float v) {
    if (d.type == GKT_F16) {
        ((__half *) d.data)[k] = __float2half(v);
    } else {
        ((float *) d.data)[k] = v;
    }
}

// What is left, once the arithmetic is gone, is a gather.
//
// A warp writes 32 consecutive elements of the patch dimension, and those 32
// come from about eleven different image rows a channel plane apart - eleven
// memory transactions where a coalesced read would take four. That is now what
// the kernel costs: it lands around 70 GB/s of useful traffic on a part that
// does 192, with the arithmetic at an eighth of issue and the loads not
// latency-starved either (batching four independent gathers per thread before
// storing any of them measured as exactly no change).
//
// Closing it means staging the rows in shared memory: a block reads the
// (channel, kernel row) segments it needs coalesced, and every thread then
// takes its element from shared. That reverses the ratio - reads become about
// seven tenths of the writes instead of five times them - and is what would
// take this to its roofline. It is a much larger kernel than this one, and at
// SD's shapes it is worth about three percent of a generation, so it has not
// been written.
static __global__ void gk_cu_k_im2col_fast(gk_tview img, gk_tview_mut d,
                                           struct gk_cu_im2col_geom g,
                                           uint32_t n) {
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += gridDim.x * blockDim.x) {
        gk_cu_im2col_store(d, k, gk_cu_im2col_gather(img, g, k));
    }
}

// The general form, for a buffer with more elements than a 32-bit index holds.
// Unreachable on any card that cannot hold a four-gigabyte im2col buffer, and
// kept because "unreachable here" is not "unreachable".
static __global__ void gk_cu_k_im2col(gk_tview img, gk_tview_mut d,
                                      int64_t IC, int64_t IH, int64_t IW,
                                      int64_t KH, int64_t KW, int64_t OH, int64_t OW,
                                      int64_t ofs_n, int64_t ofs_c, int64_t ofs_h,
                                      int s0, int s1, int p0, int p1, int d0, int d1,
                                      int64_t n) {
    const int64_t patch = IC * KH * KW;

    GK_CU_FLAT_LOOP(n) {
        const int64_t at   = k % patch;
        const int64_t cell = k / patch;

        const int64_t iow = cell % OW;
        const int64_t ioh = (cell / OW) % OH;
        const int64_t in  = cell / (OW * OH);

        const int64_t ikw = at % KW;
        const int64_t ikh = (at / KW) % KH;
        const int64_t iic = at / (KW * KH);

        const int64_t iiw = iow * s0 + ikw * d0 - p0;
        const int64_t iih = ioh * s1 + ikh * d1 - p1;

        float v = 0.0f;
        if (iih >= 0 && iih < IH && iiw >= 0 && iiw < IW) {
            v = *(const float *) (img.data + in * ofs_n + iic * ofs_c
                                  + iih * ofs_h + iiw * img.nb[0]);
        }

        // the destination is contiguous by the builder's contract, so one flat
        // index addresses it
        if (d.type == GKT_F16) {
            ((__half *) d.data)[k] = __float2half(v);
        } else {
            ((float *) d.data)[k] = v;
        }
    }
}

// roll: every axis shifts cyclically by its own amount. One thread per output
// element, each reading the one input element that lands on it - the wrap is a
// cheap index computation, so nothing is gained by the CPU pass's two-piece
// row copy here.
static __global__ void gk_cu_k_roll(gk_tview a, gk_tview_mut d,
                                    int s0, int s1, int s2, int s3, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const float v = gk_cu_get(a,
                                  gk_cu_wrap(x.i0 - s0, a.ne[0]),
                                  gk_cu_wrap(x.i1 - s1, a.ne[1]),
                                  gk_cu_wrap(x.i2 - s2, a.ne[2]),
                                  gk_cu_wrap(x.i3 - s3, a.ne[3]));

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, v);
    }
}

// The state-space convolution: a short depthwise filter slid along the time
// axis of a padded input. `sx` is [d_conv-1+n_t, d_inner, n_seq] and the
// destination is [d_inner, n_t, n_seq] - the inner dimension moves to the
// front, which is why this cannot be an im2col plus a matmul.
static __global__ void gk_cu_k_ssm_conv(gk_tview sx, gk_tview c, gk_tview_mut d,
                                        int64_t nc, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        float sum = 0.0f;
        for (int64_t i0 = 0; i0 < nc; ++i0) {
            sum += gk_cu_get(sx, x.i1 + i0, x.i0, x.i2, 0) * gk_cu_get(c, i0, x.i0, 0, 0);
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, sum);
    }
}

// max / average pooling over a 2D window. One thread per output cell; the
// window is small enough that a thread walking it serially is the right shape.
static __global__ void gk_cu_k_pool_2d(gk_tview a, gk_tview_mut d, int pool_op,
                                       int k0, int k1, int s0, int s1, int p0, int p1,
                                       int64_t IW, int64_t IH, int64_t n) {
    const float ka = (float) (k0 * k1);

    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const int64_t ix = x.i0 * s0 - p0;
        const int64_t iy = x.i1 * s1 - p1;

        float res = pool_op == GK_OP_POOL_MAX ? -FLT_MAX : 0.0f;

        for (int ky = 0; ky < k1; ++ky) {
            const int64_t j1 = iy + ky;
            if (j1 < 0 || j1 >= IH) {
                continue;
            }
            for (int kx = 0; kx < k0; ++kx) {
                const int64_t j0 = ix + kx;
                if (j0 < 0 || j0 >= IW) {
                    continue;
                }
                const float v = gk_cu_get(a, j0, j1, x.i2, x.i3);
                res = pool_op == GK_OP_POOL_MAX ? fmaxf(res, v) : res + v;
            }
        }

        // avg divides by the whole kernel area, padding included: the same
        // asymmetry with pool_1d the CPU pass carries, because it is what the
        // models were trained against
        if (pool_op == GK_OP_POOL_AVG) {
            res /= ka;
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, res);
    }
}

// The direct 2D convolution: no im2col buffer, one thread per output cell
// walking the input channels and the kernel window. The composite gk_conv_2d
// lowers to im2col plus a matmul and already runs here through those two
// kernels; this is the path taken when the intermediate would be too large to
// materialise.
static __global__ void gk_cu_k_conv_2d(gk_tview a, gk_tview b, gk_tview_mut d,
                                       int s0, int s1, int p0, int p1, int d0, int d1,
                                       int64_t IC, int64_t IW, int64_t IH,
                                       int64_t KW, int64_t KH, bool round_src, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx o = gk_cu_decompose(k, d.ne);

        float acc = 0.0f;

        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t ky = 0; ky < KH; ++ky) {
                const int64_t sy = o.i1 * s1 + ky * d1 - p1;
                if (sy < 0 || sy >= IH) {
                    continue;
                }
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const int64_t sx = o.i0 * s0 + kx * d0 - p0;
                    if (sx < 0 || sx >= IW) {
                        continue;
                    }

                    float sv = gk_cu_get(b, sx, sy, ic, o.i3);

                    // A half-precision kernel rounds the input to its own
                    // precision before multiplying. That is not an accident of
                    // the CPU pass - it is what the im2col path does, whose
                    // buffer is f16 - and dropping it here would make the two
                    // spellings of the same convolution disagree.
                    if (round_src) {
                        sv = __half2float(__float2half(sv));
                    }

                    acc += gk_cu_get(a, kx, ky, ic, o.i2) * sv;
                }
            }
        }

        gk_cu_set(d, o.i0, o.i1, o.i2, o.i3, acc);
    }
}

// Transposed 1-D convolution: kernel [K, Cout, Cin], input [L, Cin, N],
// result [(L-1)*s0 + K, Cout, N].
//
// The CPU pass *scatters*: it walks (input position, kernel tap) and adds each
// product into `out[il*s0 + ik]`, so several taps land on the same output. On
// a device that shape wants either atomics or a serialisation over `il`, and
// it needs neither, because the mapping inverts exactly. `il*s0 + ik = ox` has
// at most one solution per tap - `il = (ox - ik)/s0`, when `s0` divides
// `ox - ik` and the quotient is a real input position - so gathering gives one
// thread per output element with no contention at all.
//
// The taps are walked from the top down. That is not cosmetic: `il` falls as
// `ik` rises, so descending taps visit the contributions in ascending `il`,
// which is the order the CPU pass adds them in. Float addition is not
// associative and these two passes are checked against each other.
//
// `p0` and `d0` are read from the op params and ignored, exactly as the CPU
// pass ignores them - `gk_conv_transpose_1d` computes its output length for
// p=0, d=1 and says so. Declining those cases here would only route them to a
// pass that makes the same assumption.
static __global__ void gk_cu_k_conv_transpose_1d(gk_tview a, gk_tview b, gk_tview_mut d,
                                                 int s0, int64_t K, int64_t Cin, int64_t L,
                                                 bool round_src, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx o = gk_cu_decompose(k, d.ne);

        float acc = 0.0f;

        for (int64_t ik = K - 1; ik >= 0; --ik) {
            const int64_t rel = o.i0 - ik;
            if (rel < 0 || rel % s0 != 0) {
                continue;
            }

            const int64_t il = rel / s0;
            if (il >= L) {
                continue;
            }

            for (int64_t ic = 0; ic < Cin; ++ic) {
                float sv = gk_cu_get(b, il, ic, o.i2, 0);

                // A half-precision kernel rounds the input to its own
                // precision before multiplying, the same as gk_cu_k_conv_2d
                // and for the same reason: it is what the CPU pass does.
                if (round_src) {
                    sv = __half2float(__float2half(sv));
                }

                acc += gk_cu_get(a, ik, o.i1, ic, 0) * sv;
            }
        }

        gk_cu_set(d, o.i0, o.i1, o.i2, o.i3, acc);
    }
}

// The 3-D unrolling. One thread per output element, which means undoing the
// destination's flat layout twice: once to find which output cell the element
// belongs to, and once to find which of the cell's (channel, kernel position)
// slots it is. The destination is contiguous - the op asserts it - so the two
// decompositions are plain divisions rather than stride arithmetic.
// The volume form of the same thing, and the same fix: seven divisions per
// output element rather than five.
static __global__ void gk_cu_k_im2col_3d_fast(gk_tview b, gk_tview_mut d,
                                              struct gk_cu_fastdiv fd_cell,
                                              struct gk_cu_fastdiv fd_ow,
                                              struct gk_cu_fastdiv fd_oh,
                                              struct gk_cu_fastdiv fd_od,
                                              struct gk_cu_fastdiv fd_kw,
                                              struct gk_cu_fastdiv fd_kh,
                                              struct gk_cu_fastdiv fd_kd,
                                              int s0, int s1, int s2,
                                              int p0, int p1, int p2,
                                              int d0, int d1, int d2,
                                              int64_t IC, int64_t IW, int64_t IH, int64_t ID,
                                              uint32_t total) {
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < total;
         k += gridDim.x * blockDim.x) {
        uint32_t at, iow, ioh, iod, ikw, ikh, ikd;

        const uint32_t cell = gk_cu_fastdiv_qr(fd_cell, k,    &at);
        const uint32_t ohn  = gk_cu_fastdiv_qr(fd_ow,   cell, &iow);
        const uint32_t iodn = gk_cu_fastdiv_qr(fd_oh,   ohn,  &ioh);
        const uint32_t in   = gk_cu_fastdiv_qr(fd_od,   iodn, &iod);

        const uint32_t khc  = gk_cu_fastdiv_qr(fd_kw,   at,   &ikw);
        const uint32_t kdc  = gk_cu_fastdiv_qr(fd_kh,   khc,  &ikh);
        const uint32_t iic  = gk_cu_fastdiv_qr(fd_kd,   kdc,  &ikd);

        const int64_t iid = (int64_t) iod * s2 + (int64_t) ikd * d2 - p2;
        const int64_t iih = (int64_t) ioh * s1 + (int64_t) ikh * d1 - p1;
        const int64_t iiw = (int64_t) iow * s0 + (int64_t) ikw * d0 - p0;

        float v = 0.0f;
        if (iid >= 0 && iid < ID && iih >= 0 && iih < IH && iiw >= 0 && iiw < IW) {
            // the volume's outermost axis carries the image and the channel
            // together, image-major
            v = gk_cu_get(b, iiw, iih, iid, (int64_t) in * IC + iic);
        }

        gk_cu_set(d, at, iow, ioh, iodn, v);
    }
}

static __global__ void gk_cu_k_im2col_3d(gk_tview b, gk_tview_mut d,
                                         int s0, int s1, int s2,
                                         int p0, int p1, int p2,
                                         int d0, int d1, int d2,
                                         int64_t IC, int64_t IW, int64_t IH, int64_t ID,
                                         int64_t KW, int64_t KH, int64_t KD,
                                         int64_t OW, int64_t OH, int64_t OD,
                                         int64_t total) {
    const int64_t KH_KW    = KH * KW;
    const int64_t KD_KH_KW = KD * KH_KW;
    const int64_t cell_n   = IC * KD_KH_KW;

    GK_CU_FLAT_LOOP(total) {
        const int64_t at   = k % cell_n;      // slot within the cell
        const int64_t cell = k / cell_n;

        const int64_t iow  = cell % OW;
        const int64_t ioh  = (cell / OW) % OH;
        const int64_t iodn = cell / (OW * OH); // in * OD + iod

        const int64_t iod = iodn % OD;
        const int64_t in  = iodn / OD;

        const int64_t ikw = at % KW;
        const int64_t ikh = (at / KW) % KH;
        const int64_t ikd = (at / KH_KW) % KD;
        const int64_t iic = at / KD_KH_KW;

        const int64_t iid = iod * s2 + ikd * d2 - p2;
        const int64_t iih = ioh * s1 + ikh * d1 - p1;
        const int64_t iiw = iow * s0 + ikw * d0 - p0;

        float v = 0.0f;
        if (iid >= 0 && iid < ID && iih >= 0 && iih < IH && iiw >= 0 && iiw < IW) {
            // the volume's outermost axis carries the image and the channel
            // together, image-major
            v = gk_cu_get(b, iiw, iih, iid, in * IC + iic);
        }

        gk_cu_set(d, at, iow, ioh, iodn, v);
    }
}

// The direct depthwise convolution: one kernel plane per channel, so there is
// no input-channel loop and no reduction across channels at all. Both layouts
// the builder emits - WHCN and the channels-fastest CWHN - fall out of reading
// the operands through their strides rather than assuming a packing.
//
// Note what is absent: the f16 rounding of the input that gk_cu_k_conv_2d does.
// That is not an oversight, it is the CPU pass's behaviour, and this kernel
// exists to agree with the CPU pass. It does mean the two spellings of a
// depthwise convolution disagree - the composite gk_conv_2d_dw lowers through
// an f16 im2col buffer and so rounds, gk_conv_2d_dw_direct does not - but that
// is a question about the CPU pass, not something to settle by having the
// device answer differently from it.
static __global__ void gk_cu_k_conv_2d_dw(gk_tview a, gk_tview b, gk_tview_mut d,
                                          int s0, int s1, int p0, int p1, int d0, int d1,
                                          int64_t IW, int64_t IH,
                                          int64_t KW, int64_t KH, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx o = gk_cu_decompose(k, d.ne);

        float acc = 0.0f;

        for (int64_t ky = 0; ky < KH; ++ky) {
            const int64_t sy = o.i1 * s1 + ky * d1 - p1;
            if (sy < 0 || sy >= IH) {
                continue;
            }
            for (int64_t kx = 0; kx < KW; ++kx) {
                const int64_t sx = o.i0 * s0 + kx * d0 - p0;
                if (sx < 0 || sx >= IW) {
                    continue;
                }

                // the kernel is [KW, KH, 1, C]: one plane per channel, indexed
                // by the output's channel
                acc += gk_cu_get(a, kx, ky, 0, o.i2) * gk_cu_get(b, sx, sy, o.i2, o.i3);
            }
        }

        gk_cu_set(d, o.i0, o.i1, o.i2, o.i3, acc);
    }
}

// --------------------------------------------------------------------------
// the linear-attention recurrences
//
// RWKV's two kernels and the gated delta rule share a shape: a state matrix
// per head that a token loop walks forward, one token at a time. Time cannot
// be parallelised - that is what makes them recurrences - so the threads go
// everywhere else, and the useful observation in all three is that a thread
// can be given a slice of the state that nobody else reads or writes.
//
// That is what removes the barriers. A thread owning one row (or column) of
// the state matrix carries it across the whole token loop without ever
// synchronizing with its neighbours, because the recurrence for that slice
// depends only on that slice and on per-token vectors every thread reads
// alike.
// --------------------------------------------------------------------------

// RWKV-6. One block per (head, sequence); thread j owns column j of the state.
//
// The CPU pass accumulates the output over the i loop; here thread j runs that
// loop itself and keeps the running sum in a register, which is the same sum
// in the same order.
static __global__ void gk_cu_k_rwkv_wkv6(const float * k_in, const float * v_in,
                                         const float * r_in, const float * tf,
                                         const float * td, const float * s_in,
                                         float * out, float * state,
                                         int64_t T, int64_t C, int64_t S, int64_t T_per) {
    const int64_t h    = blockIdx.x;
    const int64_t iseq = blockIdx.y;
    const int64_t j    = threadIdx.x;

    if (j >= S) {
        return;
    }

    const int64_t h_off = h * S;
    const int64_t h2d   = h * S * S;
    const int64_t state_off = S * C * iseq;

    float * state_cur = state + state_off;

    for (int64_t t = iseq * T_per; t < (iseq + 1) * T_per; ++t) {
        const int64_t th_off = t * C + h_off;

        // the first token of a sequence reads the state it was handed; the
        // rest read what the previous token left
        const float * state_prev = (t % T_per) ? state_cur : s_in + state_off;

        const float vj = v_in[th_off + j];

        float acc = 0.0f;
        for (int64_t i = 0; i < S; ++i) {
            const float kv  = k_in[th_off + i];
            const float rv  = r_in[th_off + i];
            const float tfv = tf[h_off + i];
            const float tdv = td[th_off + i];

            const float kvv  = vj * kv;
            const float prev = state_prev[h2d + i * S + j];

            acc += (kvv * tfv + prev) * rv;
            state_cur[h2d + i * S + j] = prev * tdv + kvv;
        }

        out[th_off + j] = acc;
    }
}

// RWKV-7. Transposed from the above: this recurrence reduces along j twice -
// once for the in-context learning rate, once for the output - so a row of the
// state is the unit of work.
//
// A thread per row would put consecutive threads S floats apart and cost a
// separate memory transaction each; measured that way this kernel lost to the
// CPU. So a *warp* owns a row instead, its lanes walking j together, which
// makes every access contiguous and turns both reductions into shuffles. The
// butterfly leaves the sum in all lanes, so there is nothing to broadcast
// afterwards and no barrier anywhere in the loop.
static __global__ void gk_cu_k_rwkv_wkv7(const float * r_in, const float * w_in,
                                         const float * k_in, const float * v_in,
                                         const float * a_in, const float * b_in,
                                         const float * s_in,
                                         float * out, float * state,
                                         int64_t T, int64_t C, int64_t S, int64_t T_per) {
    const int64_t h    = blockIdx.x;
    const int64_t iseq = blockIdx.y;

    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = blockDim.x / GK_WARP_SIZE;

    const int64_t h_off = h * S;
    const int64_t h2d   = h * S * S;
    const int64_t state_off = S * C * iseq;

    float * state_cur = state + state_off;

    for (int64_t t = iseq * T_per; t < (iseq + 1) * T_per; ++t) {
        const int64_t th_off = t * C + h_off;

        const float * state_prev = (t % T_per) ? state_cur : s_in + state_off;

        for (int64_t i = warp; i < S; i += n_warps) {
            const float vv = v_in[th_off + i];

            // project the previous state row onto a
            float sa_part = 0.0f;
            for (int64_t j = lane; j < S; j += GK_WARP_SIZE) {
                sa_part += a_in[th_off + j] * state_prev[h2d + i * S + j];
            }
            // Every lane leaves with the whole sum, and with its reads to this
            // row finished - which is what lets the loop below overwrite the
            // row the loop above was reading.
            const float sa = gk_cu_warp_sum(sa_part);

            float res_part = 0.0f;
            for (int64_t j = lane; j < S; j += GK_WARP_SIZE) {
                const float kvv  = vv * k_in[th_off + j];
                const float prev = state_prev[h2d + i * S + j];
                const float cur  = prev * w_in[th_off + j] + kvv + sa * b_in[th_off + j];

                state_cur[h2d + i * S + j] = cur;
                res_part += cur * r_in[th_off + j];
            }
            const float res = gk_cu_warp_sum(res_part);

            if (lane == 0) {
                out[th_off + i] = res;
            }
        }
    }
}

// The gated delta rule, register-resident form for the head widths models
// actually ship (16..128, powers of two). A warp owns one column of the state
// and keeps its S values in registers across the whole token loop, so the
// only global traffic per token is the five input rows and the output - the
// generic kernel below instead walks the working state through global memory
// four times per token, which is where its time goes.
//
// The state layout stays transposed, as everywhere else in this op: row col
// of the stored state holds column col of the mathematical one, so a warp's
// lanes spread along i and both reductions are shuffles. For S below the
// hardware warp the shuffles are width-limited so two columns sharing a
// hardware warp cannot mix.
//
// Grid (H, n_seqs, S / GK_CU_GDN_WARPS); block (min(S, warp), GK_CU_GDN_WARPS).
#define GK_CU_GDN_WARPS 4

template <int width>
static __device__ __forceinline__ float gk_cu_warp_sum_w(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
#if defined(GK_USE_HIP)
        x += __shfl_xor(x, offset, width);
#else
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
#endif
    }
    return x;
}

template <int S_v, bool kda>
static __global__ void
__launch_bounds__((S_v < GK_WARP_SIZE ? S_v : GK_WARP_SIZE) * GK_CU_GDN_WARPS, 2)
gk_cu_k_gated_delta_net_col(gk_tview q, gk_tview k_t, gk_tview v,
                            gk_tview g, gk_tview beta,
                            const float * s_in, int64_t s_in_stride3,
                            float * attn_base, float * state_base,
                            int64_t H, int64_t n_tokens,
                            int64_t snap_elems, int64_t K,
                            int64_t rq3, int64_t rk3, float scale) {
    constexpr int WS  = S_v < GK_WARP_SIZE ? S_v : GK_WARP_SIZE;
    constexpr int RPL = S_v / WS; // rows of the column per lane

    const int64_t iv1 = blockIdx.x; // head
    const int64_t iv3 = blockIdx.y; // sequence
    const int    lane = threadIdx.x;
    const int     col = blockIdx.z * GK_CU_GDN_WARPS + threadIdx.y;

    const int64_t iq1 = iv1 % q.ne[1];
    const int64_t ik1 = iv1 % k_t.ne[1];
    const int64_t iq3 = iv3 / rq3;
    const int64_t ik3 = iv3 / rk3;

    // The handed-in state, indexed the way the CPU pass indexes it: flat
    // within a sequence, with only the sequence axis carrying a stride.
    const float * s0    = s_in + iv3 * s_in_stride3 + (iv1 * S_v + col) * S_v;
    float *       snap0 = state_base + (iv3 * H + iv1) * S_v * S_v + (int64_t) col * S_v;
    float *       attn  = attn_base + (iv3 * n_tokens * H + iv1) * S_v;

    float s_reg[RPL];
#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        s_reg[r] = s0[r * WS + lane];
    }

    if (n_tokens == 0) {
        // the generic kernel's copy-through: slot 0 carries s0 unchanged
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            snap0[r * WS + lane] = s_reg[r];
        }
        return;
    }

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * q_d = (const float *) (q.data   + iq3 * q.nb[3]    + t * q.nb[2]    + iq1 * q.nb[1]);
        const float * k_d = (const float *) (k_t.data + ik3 * k_t.nb[3]  + t * k_t.nb[2]  + ik1 * k_t.nb[1]);
        const float * v_d = (const float *) (v.data   + iv3 * v.nb[3]    + t * v.nb[2]    + iv1 * v.nb[1]);
        const float * g_d = (const float *) (g.data   + iv3 * g.nb[3]    + t * g.nb[2]    + iv1 * g.nb[1]);

        const float beta_v = *(const float *) (beta.data + iv3 * beta.nb[3] + t * beta.nb[2] + iv1 * beta.nb[1]);

        float k_reg[RPL];
        float q_reg[RPL];
        float w_reg[RPL];
        const float dg = kda ? 0.0f : expf(g_d[0]);
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            const int i = r * WS + lane;
            k_reg[r] = k_d[i];
            q_reg[r] = q_d[i];
            w_reg[r] = kda ? expf(g_d[i]) : dg;
        }

        // decay, then how far the state's prediction of v misses
        float part = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            s_reg[r] *= w_reg[r];
            part += s_reg[r] * k_reg[r];
        }
        const float dj = (v_d[col] - gk_cu_warp_sum_w<WS>(part)) * beta_v;

        // the rank-one update that closes the gap, read out against q
        float acc = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            s_reg[r] += k_reg[r] * dj;
            acc += s_reg[r] * q_reg[r];
        }
        const float outv = gk_cu_warp_sum_w<WS>(acc);

        if (lane == 0) {
            attn[t * S_v * H + col] = outv * scale;
        }

        // snapshot slot s holds the state s tokens back; slot 0, written on
        // the last token, is the final state every caller reads
        const int64_t slot = n_tokens - 1 - t;
        if (slot < K) {
            float * snap = snap0 + slot * snap_elems;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                snap[r * WS + lane] = s_reg[r];
            }
        }
    }
}

// The generic fallback for head widths the register form does not cover. One
// block per (head, sequence); a warp owns row j of the working state, which
// the layout stores transposed - row j holds column j of the state.
//
// A warp rather than a thread for the same reason as RWKV-7 above: every one
// of the four passes a token makes over a row walks it along i, so lanes
// spread along i keep the accesses contiguous and the two reductions become
// shuffles. A thread per row instead put consecutive threads S floats apart
// and ran at less than half the CPU's speed.
//
// Every step is per-row except the per-channel decay, whose factors are
// indexed by the other axis and so are shared across rows. That is the only
// barrier in the loop, and the scalar-gate form does not need even that.
static __global__ void gk_cu_k_gated_delta_net(gk_tview q, gk_tview k_t, gk_tview v,
                                               gk_tview g, gk_tview beta,
                                               const float * s_in, int64_t s_in_stride3,
                                               float * attn_base, float * state_base,
                                               int64_t S, int64_t H, int64_t n_tokens,
                                               int64_t snap_elems, int64_t K,
                                               int64_t rq3, int64_t rk3, bool kda,
                                               float scale) {
    extern __shared__ float gk_cu_gdn_decay[]; // S floats, only used when kda

    const int64_t iv1 = blockIdx.x; // head
    const int64_t iv3 = blockIdx.y; // sequence

    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = blockDim.x / GK_WARP_SIZE;

    const int64_t iq1 = iv1 % q.ne[1];
    const int64_t ik1 = iv1 % k_t.ne[1];
    const int64_t iq3 = iv3 / rq3;
    const int64_t ik3 = iv3 / rk3;

    // slot 0 of the output doubles as the working state
    float * s_work = state_base + (iv3 * H + iv1) * S * S;

    // The handed-in state, indexed the way the CPU pass indexes it: flat
    // within a sequence, with only the sequence axis carrying a stride.
    const float * s0 = s_in + iv3 * s_in_stride3 + iv1 * S * S;
    for (int64_t e = threadIdx.x; e < S * S; e += blockDim.x) {
        s_work[e] = s0[e];
    }
    __syncthreads();

    for (int64_t t = 0; t < n_tokens; ++t) {
        const int64_t q_off = iq3 * q.nb[3]    + t * q.nb[2]    + iq1 * q.nb[1];
        const int64_t k_off = ik3 * k_t.nb[3]  + t * k_t.nb[2]  + ik1 * k_t.nb[1];
        const int64_t v_off = iv3 * v.nb[3]    + t * v.nb[2]    + iv1 * v.nb[1];
        const int64_t g_off = iv3 * g.nb[3]    + t * g.nb[2]    + iv1 * g.nb[1];
        const int64_t b_off = iv3 * beta.nb[3] + t * beta.nb[2] + iv1 * beta.nb[1];

        const float * q_d = (const float *) (q.data   + q_off);
        const float * k_d = (const float *) (k_t.data + k_off);
        const float * v_d = (const float *) (v.data   + v_off);
        const float * g_d = (const float *) (g.data   + g_off);

        const float beta_v = *(const float *) (beta.data + b_off);

        if (kda) {
            for (int64_t i = threadIdx.x; i < S; i += blockDim.x) {
                gk_cu_gdn_decay[i] = expf(g_d[i]);
            }
            __syncthreads();
        }

        for (int64_t j = warp; j < S; j += n_warps) {
            float * row = s_work + j * S;

            // decay: per channel for KDA, one scalar otherwise
            if (kda) {
                for (int64_t i = lane; i < S; i += GK_WARP_SIZE) {
                    row[i] *= gk_cu_gdn_decay[i];
                }
            } else {
                const float dg = expf(g_d[0]);
                for (int64_t i = lane; i < S; i += GK_WARP_SIZE) {
                    row[i] *= dg;
                }
            }

            // how far the state's prediction of v misses, then the rank-one
            // update that closes the gap
            float part = 0.0f;
            for (int64_t i = lane; i < S; i += GK_WARP_SIZE) {
                part += row[i] * k_d[i];
            }
            const float dj = (v_d[j] - gk_cu_warp_sum(part)) * beta_v;

            for (int64_t i = lane; i < S; i += GK_WARP_SIZE) {
                row[i] += k_d[i] * dj;
            }

            // read the state out against q
            float acc = 0.0f;
            for (int64_t i = lane; i < S; i += GK_WARP_SIZE) {
                acc += row[i] * q_d[i];
            }
            const float outv = gk_cu_warp_sum(acc);

            if (lane == 0) {
                attn_base[(iv3 * n_tokens * H + iv1) * S + t * S * H + j] = outv * scale;
            }

            if (K > 1) {
                const int64_t slot = n_tokens - 1 - t;
                if (slot > 0 && slot < K) {
                    float * snap = state_base + slot * snap_elems + (iv3 * H + iv1) * S * S;
                    for (int64_t i = lane; i < S; i += GK_WARP_SIZE) {
                        snap[j * S + i] = row[i];
                    }
                }
            }
        }

        // The decay buffer is rewritten at the top of the next token, and the
        // rows above are still reading this one's.
        if (kda) {
            __syncthreads();
        }
    }
}

// host-side dispatch for the register-resident form
template <int S_v>
static void gk_cu_gdn_col_launch(bool kda, int64_t H, int64_t n_seqs, cudaStream_t stream,
                                 gk_tview q, gk_tview k, gk_tview v,
                                 gk_tview g, gk_tview beta,
                                 const float * s_in, int64_t s_in_stride3,
                                 float * attn_base, float * state_base,
                                 int64_t n_tokens, int64_t snap_elems, int64_t K,
                                 int64_t rq3, int64_t rk3, float scale) {
    constexpr int WS = S_v < GK_WARP_SIZE ? S_v : GK_WARP_SIZE;
    dim3 grid((unsigned) H, (unsigned) n_seqs, (unsigned) (S_v / GK_CU_GDN_WARPS));
    dim3 block(WS, GK_CU_GDN_WARPS, 1);
    if (kda) {
        gk_cu_k_gated_delta_net_col<S_v, true><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, s_in, s_in_stride3, attn_base, state_base,
            H, n_tokens, snap_elems, K, rq3, rk3, scale);
    } else {
        gk_cu_k_gated_delta_net_col<S_v, false><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, s_in, s_in_stride3, attn_base, state_base,
            H, n_tokens, snap_elems, K, rq3, rk3, scale);
    }
}

// The Mamba selective scan.
//
// The recurrence runs along time and cannot be parallelised across it, so the
// threads are spread over everything else: one per (sequence, head, channel).
// Each such thread owns d_state consecutive state values that no other thread
// touches, which is what makes the whole scan race-free without a barrier -
// the state chains through a thread's own registers and its own slice of the
// result, never through a neighbour's.
//
// Strides arrive as byte offsets because that is how the tensors describe
// themselves; within a row the CPU pass indexes flat floats and this matches
// it element for element, softplus included, so a graph split across the two
// devices does not drift at the seam.
static __global__ void gk_cu_k_ssm_scan(const char * s_in, const char * x, const char * dt,
                                        const float * A, const char * B, const char * C,
                                        const int32_t * ids, char * dst,
                                        int64_t nc, int64_t nr, int64_t nh, int64_t ng,
                                        int64_t nt, int64_t nheads_per_group,
                                        int64_t s_nb3, int64_t x_nb2, int64_t x_nb3,
                                        int64_t dt_nb1, int64_t dt_nb2,
                                        int64_t bc_nb2, int64_t bc_nb3,
                                        int64_t s_off, bool one_decay_per_head, int64_t n) {
    GK_UNUSED(ng);

    GK_CU_FLAT_LOOP(n) {
        const int64_t i1 = k % nr;              // channel within the head
        const int64_t h  = (k / nr) % nh;       // head
        const int64_t i3 = k / (nr * nh);       // sequence

        const int64_t ii = i1 + h * nr;
        const int64_t g  = h / nheads_per_group;

        // The state this thread carries: d_state values, read from the cache
        // on the first token and from its own output afterwards.
        const float * s0 = (const float *) (s_in + (int64_t) ids[i3] * s_nb3) + ii * nc;
        float * s = (float *) (dst + s_off + i3 * s_nb3) + ii * nc;

        for (int64_t i2 = 0; i2 < nt; ++i2) {
            const float * xt  = (const float *) (x  + i2 * x_nb2  + i3 * x_nb3);
            const float * dtt = (const float *) (dt + i2 * dt_nb1 + i3 * dt_nb2);
            const float * Bt  = (const float *) (B  + i2 * bc_nb2 + i3 * bc_nb3);
            const float * Ct  = (const float *) (C  + i2 * bc_nb2 + i3 * bc_nb3);
            float * y = (float *) dst + i2 * nh * nr + i3 * nt * nh * nr;

            // softplus, with the large-input shortcut the CPU pass takes so
            // the two agree bit for bit above the threshold
            const float dtv = dtt[h];
            const float dts = dtv > 20.0f ? dtv : logf(1.0f + expf(dtv));

            const float x_dt = xt[ii] * dts;

            float sum = 0.0f;

            if (one_decay_per_head) {
                // Mamba-2: A is [1, n_head], so the decay is a single scalar
                // for the whole state vector and is computed once.
                const float dA = expf(dts * A[h]);

                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    const int64_t ig = i0 + g * nc;
                    const float state = s0[i0] * dA + Bt[ig] * x_dt;
                    sum += state * Ct[ig];
                    s[i0] = state;
                }
            } else {
                // Mamba-1: A is [d_state, n_head], one decay per state element
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    const int64_t ig = i0 + g * nc;
                    const float state = s0[i0] * expf(dts * A[i0 + h * nc]) + Bt[ig] * x_dt;
                    sum += state * Ct[ig];
                    s[i0] = state;
                }
            }

            y[ii] = sum;

            s0 = s; // from here on the state chains through the result
        }
    }
}

// --------------------------------------------------------------------------
// dispatch
// --------------------------------------------------------------------------

// The type codes in gk_cuda_dequant.cuh are a copy, made because that header
// has to stand on its own in device code. A copy that drifts would decode
// weights as the wrong format and produce noise, so it is held against the
// original here, where both are in scope.
static_assert((int) GK_TYPE_F32  == GKT_F32,  "type enum drifted: F32");
static_assert((int) GK_TYPE_F16  == GKT_F16,  "type enum drifted: F16");
static_assert((int) GK_TYPE_BF16 == GKT_BF16, "type enum drifted: BF16");
static_assert((int) GK_TYPE_Q4_0 == GKT_Q4_0, "type enum drifted: Q4_0");
static_assert((int) GK_TYPE_Q4_1 == GKT_Q4_1, "type enum drifted: Q4_1");
static_assert((int) GK_TYPE_Q5_0 == GKT_Q5_0, "type enum drifted: Q5_0");
static_assert((int) GK_TYPE_Q5_1 == GKT_Q5_1, "type enum drifted: Q5_1");
static_assert((int) GK_TYPE_Q8_0 == GKT_Q8_0, "type enum drifted: Q8_0");
static_assert((int) GK_TYPE_Q2_K == GKT_Q2_K, "type enum drifted: Q2_K");
static_assert((int) GK_TYPE_Q3_K == GKT_Q3_K, "type enum drifted: Q3_K");
static_assert((int) GK_TYPE_Q4_K == GKT_Q4_K, "type enum drifted: Q4_K");
static_assert((int) GK_TYPE_Q5_K == GKT_Q5_K, "type enum drifted: Q5_K");
static_assert((int) GK_TYPE_Q6_K == GKT_Q6_K, "type enum drifted: Q6_K");
static_assert((int) GK_TYPE_IQ4_NL == GKT_IQ4_NL, "type enum drifted: IQ4_NL");
static_assert((int) GK_TYPE_IQ4_XS == GKT_IQ4_XS, "type enum drifted: IQ4_XS");
static_assert((int) GK_TYPE_TQ1_0  == GKT_TQ1_0,  "type enum drifted: TQ1_0");
static_assert((int) GK_TYPE_TQ2_0  == GKT_TQ2_0,  "type enum drifted: TQ2_0");
static_assert((int) GK_TYPE_MXFP4  == GKT_MXFP4,  "type enum drifted: MXFP4");
static_assert((int) GK_TYPE_NVFP4  == GKT_NVFP4,  "type enum drifted: NVFP4");
static_assert((int) GK_TYPE_Q1_0   == GKT_Q1_0,   "type enum drifted: Q1_0");
static_assert((int) GK_TYPE_Q2_0   == GKT_Q2_0,   "type enum drifted: Q2_0");
static_assert((int) GK_TYPE_I32    == GKT_I32,    "type enum drifted: I32");
static_assert((int) GK_TYPE_I64    == GKT_I64,    "type enum drifted: I64");
static_assert((int) GK_TYPE_IQ1_S   == GKT_IQ1_S,   "type enum drifted: IQ1_S");
static_assert((int) GK_TYPE_IQ1_M   == GKT_IQ1_M,   "type enum drifted: IQ1_M");
static_assert((int) GK_TYPE_IQ2_XXS == GKT_IQ2_XXS, "type enum drifted: IQ2_XXS");
static_assert((int) GK_TYPE_IQ2_XS  == GKT_IQ2_XS,  "type enum drifted: IQ2_XS");
static_assert((int) GK_TYPE_IQ2_S   == GKT_IQ2_S,   "type enum drifted: IQ2_S");
static_assert((int) GK_TYPE_IQ3_XXS == GKT_IQ3_XXS, "type enum drifted: IQ3_XXS");
static_assert((int) GK_TYPE_IQ3_S   == GKT_IQ3_S,   "type enum drifted: IQ3_S");

// The float types a generic kernel can read and write. Quantized destinations
// are not among them: writing one means encoding, and encoding needs the whole
// block, which is a different kernel shape than any of these.
static bool gk_cu_is_float_type(int type) {
    return type == GKT_F32 || type == GKT_F16 || type == GKT_BF16;
}

static bool gk_cu_readable(const struct gk_tensor * t) {
    if (t == NULL) {
        return true;
    }
    const int type = (int) t->type;
    if (gk_cu_is_float_type(type) || type == GKT_I32 || type == GKT_I64) {
        return true;
    }
    // a quantized operand is only readable where it is packed, which is what
    // gk_cu_row_elem assumes
    return gk_cu_type_supported(type) && t->nb[0] == (size_t) gk_cu_type_size(type);
}

bool gk_cuda_supports_op(const struct gk_tensor * op) {
    const struct gk_tensor * s0 = op->src[0];
    const struct gk_tensor * s1 = op->src[1];

    switch ((int) op->op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return true;

        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            // the weight may be quantized, the activations may not, and the
            // result is always f32
            return op->type == GKT_F32 && gk_cu_type_supported((int) s0->type) &&
                   gk_cu_readable(s0) && gk_cu_is_float_type((int) s1->type);

        case GK_OP_ARGSORT:
        case GK_OP_TOP_K:
            // positions out, not values, so the destination is i32 - which is
            // why these do not join the group below, whose tail demands a
            // float destination
            return op->type == GKT_I32 && gk_cu_readable(s0);

        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV:
        case GK_OP_ADD_ID:
        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS:
        case GK_OP_UNARY: case GK_OP_GLU: case GK_OP_LEAKY_RELU:
        case GK_OP_SCALE: case GK_OP_CLAMP: case GK_OP_FILL:
        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM: case GK_OP_GROUP_NORM:
        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
        case GK_OP_GET_ROWS: case GK_OP_REPEAT: case GK_OP_CONCAT:
        case GK_OP_SOFT_MAX: case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
        case GK_OP_ROPE: case GK_OP_SUM_ROWS: case GK_OP_MEAN: case GK_OP_SUM:
        case GK_OP_PAD: case GK_OP_TIMESTEP_EMBEDDING: case GK_OP_ARANGE:
        case GK_OP_FLASH_ATTN_EXT: case GK_OP_IM2COL: case GK_OP_UPSCALE:
        case GK_OP_SET_ROWS:
        case GK_OP_ROLL: case GK_OP_SSM_CONV: case GK_OP_POOL_2D:
        case GK_OP_CONV_2D: case GK_OP_SSM_SCAN:
        case GK_OP_CONV_2D_DW: case GK_OP_CONV_TRANSPOSE_1D:
        case GK_OP_RWKV_WKV6: case GK_OP_RWKV_WKV7: case GK_OP_GATED_DELTA_NET:
        case GK_OP_PAD_REFLECT_1D: case GK_OP_CUMSUM:
            break;

        case GK_OP_ARGMAX:
            // positions out, not values, like argsort and top_k above
            return op->type == GKT_I32 && gk_cu_readable(s0) &&
                   s0->type == GKT_F32 && s0->ne[2] == 1 && s0->ne[3] == 1;

        case GK_OP_IM2COL_3D:
            // The unrolled buffer is the operand of a matmul, so it is f32 or
            // f16; the volume is always f32. The destination is written by
            // flat index, which the op's own contiguity assertion guarantees.
            return (op->type == GKT_F32 || op->type == GKT_F16) &&
                   s1 != NULL && s1->type == GKT_F32 &&
                   gk_is_contiguous(op) && gk_cu_readable(s1);

        default:
            return false;
    }

    // Destination types: everything above writes through gk_cu_set, except
    // set_rows (whose destination is the cache's own type) and im2col.
    if ((int) op->op == GK_OP_SET_ROWS) {
        return gk_cu_is_float_type((int) op->type) &&
               s0 != NULL && gk_cu_is_float_type((int) s0->type);
    }
    if ((int) op->op == GK_OP_IM2COL) {
        return (op->type == GKT_F32 || op->type == GKT_F16) &&
               s1 != NULL && s1->type == GKT_F32;
    }
    if ((int) op->op == GK_OP_FLASH_ATTN_EXT) {
        // K and V may be quantized; Q and the result are f32. The head widths
        // are bounded because the query row and the value accumulator live in
        // shared memory - a wider head falls back to the CPU rather than
        // being quietly truncated.
        return op->type == GKT_F32 && s0->type == GKT_F32 &&
               gk_cu_type_supported((int) op->src[1]->type) &&
               gk_cu_type_supported((int) op->src[2]->type) &&
               op->src[1]->ne[0] <= GK_CUDA_FA_MAX_DK &&
               op->src[2]->ne[0] <= GK_CUDA_FA_MAX_DV;
    }
    if ((int) op->op == GK_OP_CONV_2D) {
        // The image and the result are f32; the kernel may be half, which also
        // decides whether the input is rounded on the way in.
        return op->type == GKT_F32 && s1 != NULL && s1->type == GKT_F32 &&
               (s0->type == GKT_F32 || s0->type == GKT_F16);
    }
    if ((int) op->op == GK_OP_CONV_TRANSPOSE_1D) {
        // Same operand rules as conv_2d, and the same reason for the half
        // case: an f16 kernel rounds the input on the way in.
        //
        // The stride is divided by per output element, so a zero would be a
        // device-side fault rather than a wrong answer. It cannot come from
        // gk_conv_transpose_1d, but supports_op is the wrong place to find
        // that out.
        return op->type == GKT_F32 && s1 != NULL && s1->type == GKT_F32 &&
               (s0->type == GKT_F32 || s0->type == GKT_F16) &&
               gk_get_op_params_i32(op, 0) > 0;
    }
    if ((int) op->op == GK_OP_SSM_SCAN) {
        // The scan indexes x, B and C as flat rows across their first two
        // dimensions, exactly as the CPU pass does, so a tensor whose second
        // dimension is not packed against its first would be read wrongly.
        // That shape never reaches here from the graph builders, and declining
        // it costs a CPU fallback rather than a wrong answer.
        const struct gk_tensor * packed[] = { op->src[1], op->src[4], op->src[5] };
        for (int i = 0; i < 3; ++i) {
            const struct gk_tensor * t = packed[i];
            if (t->type != GKT_F32 || t->nb[0] != sizeof(float) ||
                t->nb[1] != (size_t) t->ne[0] * t->nb[0]) {
                return false;
            }
        }

        return op->type == GKT_F32 && s0->type == GKT_F32 &&
               gk_is_contiguous(s0) && gk_is_contiguous(op->src[2]) &&
               gk_is_contiguous(op->src[3]) &&
               op->src[2]->type == GKT_F32 && op->src[3]->type == GKT_F32 &&
               op->src[6]->type == GKT_I32;
    }
    if ((int) op->op == GK_OP_CONV_2D_DW) {
        // The image and the result are f32; the kernel may be half. Unlike
        // conv_2d there is no rounding of the input either way, which matches
        // the CPU pass - see the kernel.
        return op->type == GKT_F32 && s1 != NULL && s1->type == GKT_F32 &&
               (s0->type == GKT_F32 || s0->type == GKT_F16);
    }
    if ((int) op->op == GK_OP_GATED_DELTA_NET) {
        // The kernels walk q, k, v, g and beta by their own nb[] per token -
        // the graph hands v in as a strided slice of the fused qkv projection,
        // and declining that sent the whole recurrence to the CPU with the
        // state crossing the bus both ways every step. Only the row itself
        // must be packed floats. The incoming state is read flat within a
        // sequence, so its inner axes must be packed; the sequence axis alone
        // may carry a stride.
        const int64_t S = op->src[2]->ne[0];
        if (S > GK_CUDA_RECURRENT_MAX_S) {
            return false;
        }
        for (int i = 0; i < 5; ++i) {
            const struct gk_tensor * t = op->src[i];
            if (t == NULL || t->type != GKT_F32 || t->nb[0] != sizeof(float)) {
                return false;
            }
        }
        const struct gk_tensor * s = op->src[5];
        if (s == NULL || s->type != GKT_F32 || s->nb[0] != sizeof(float) ||
            s->nb[1] != (size_t) s->ne[0] * sizeof(float) ||
            s->nb[2] != (size_t) (s->ne[0] * s->ne[1]) * sizeof(float)) {
            return false;
        }
        return op->type == GKT_F32;
    }
    if ((int) op->op == GK_OP_RWKV_WKV6 || (int) op->op == GK_OP_RWKV_WKV7) {
        // RWKV-6 gives a head's state one slot per thread, so a head wider
        // than a block would go partly uncomputed. RWKV-7 strides whole rows
        // over warps and has no such limit, but the bound is applied to both:
        // it is above anything published - RWKV heads are 64 - and one rule
        // is easier to keep true than two. A wider head falls back to the CPU.
        const int64_t S = op->ne[0] / op->src[1]->ne[1];

        if (S > GK_CUDA_RECURRENT_MAX_S) {
            return false;
        }

        // Every operand is read as a flat float array with only the outermost
        // axis strided, exactly as the CPU pass reads them, so a non-contiguous
        // one would be read wrongly rather than slowly.
        const int n_src = (int) op->op == GK_OP_RWKV_WKV7 ? 7 : 6;
        for (int i = 0; i < n_src; ++i) {
            const struct gk_tensor * t = op->src[i];
            if (t == NULL || t->type != GKT_F32 || !gk_is_contiguous(t)) {
                return false;
            }
        }
        return op->type == GKT_F32;
    }
    if ((int) op->op == GK_OP_UPSCALE) {
        // only nearest and plain bilinear; the antialiased and bicubic filters
        // stay on the CPU rather than be approximated here
        const int mode_flags = gk_get_op_params_i32(op, 0);
        const int mode = mode_flags & 0xFF;
        if (mode > 1 || (mode_flags & GK_SCALE_FLAG_ANTIALIAS)) {
            return false;
        }
    }

    // The samplers run on the backend too, and their graphs gather token ids
    // (an i32 get_rows over an i32 table) and cast a computed index to i32
    // (cpy). The same-type gather and copy move integers directly; the f32
    // to i32 cast goes through gk_cu_set, whose C cast matches the CPU codec.
    const bool i32_dst_ok = op->type == GKT_I32 && s0 != NULL &&
        (((int) op->op == GK_OP_GET_ROWS && s0->type == GKT_I32) ||
         (((int) op->op == GK_OP_CPY || (int) op->op == GK_OP_DUP ||
           (int) op->op == GK_OP_CONT) &&
          (s0->type == GKT_F32 || s0->type == GKT_I32)));

    if (!gk_cu_is_float_type((int) op->type) && (int) op->op != GK_OP_ARANGE &&
        !i32_dst_ok) {
        return false;
    }

    for (int i = 0; i < GK_MAX_SRC; ++i) {
        if (!gk_cu_readable(op->src[i])) {
            return false;
        }
    }

    return true;
}

// --------------------------------------------------------------------------

bool gk_cuda_compute_op(gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * node) {
    const int op = (int) node->op;

    struct gk_tensor * src0 = node->src[0];
    struct gk_tensor * src1 = node->src[1];
    struct gk_tensor * src2 = node->src[2];

    const int64_t ne = gk_cu_nelements(node);
    const int     nb = gk_cu_blocks(ne, GK_CUDA_BLOCK);

    // A node with no elements is a legal part of a graph - the output matmul of
    // a batch that produces no logits has zero columns, and a view of an empty
    // range has zero rows - but there is no launch geometry that means "no
    // work": a grid dimension of zero is rejected by the driver, not ignored.
    // So the emptiness is answered here, where it is one comparison, rather
    // than at each of the launches below.
    //
    // The op is still complete after this: everything below writes only into
    // the destination, so an empty destination has nothing owed to it. The one
    // op that writes elsewhere - set_rows, whose destination is the cache - is
    // driven by its source's element count and is handled at its own launch.
    if (ne == 0) {
        return true;
    }

    switch (op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return true; // pure reinterpretations; the memory is already right

        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV: {
            const int kind = op == GK_OP_ADD ? 0 : op == GK_OP_SUB ? 1 : op == GK_OP_MUL ? 2 : 3;

            // Everything contiguous, and src1 broadcasting in one of the four
            // patterns a graph actually uses: no index arithmetic at all.
            if (gk_cu_flat_ok(node) && gk_cu_flat_ok(src0) && gk_cu_flat_ok(src1)) {
                const int bc = gk_cu_bcast_kind(node, src1);

                // The divisor the broadcast needs is taken in float4 units by
                // the vector kernel, so a wrap length - or a plane, for a
                // per-channel operand - that is not a whole number of them keeps
                // the scalar one, where the divisor is in elements and always
                // exact.
                const int64_t ne_b = gk_cu_nelements(src1);

                const bool vec = gk_cu_flat_vec(node, ne) && gk_cu_flat_vec(src0, ne) &&
                                 (bc != GK_CU_BCAST_NONE || gk_cu_flat_vec(src1, ne)) &&
                                 (bc != GK_CU_BCAST_WRAP || (ne_b % 4 == 0 &&
                                                             (uintptr_t) src1->data % 16 == 0)) &&
                                 (bc != GK_CU_BCAST_CH   || node->ne[0] * node->ne[1] % 4 == 0);

                const int64_t n     = vec ? ne / 4 : ne;
                const int64_t plane = bc != GK_CU_BCAST_CH ? 1
                                    : (vec ? node->ne[0] * node->ne[1] / 4
                                           : node->ne[0] * node->ne[1]);

                // Only the broadcast that uses a divisor computes one. The
                // guard below rejects a divisor of zero, and a wrap length of
                // one or two elements rounds to that in float4s - which would
                // then decline the whole flat path for the broadcasts that do
                // not want a divisor at all. That is not hypothetical: this
                // used to be `ne0 / 4`, so every tensor two elements wide -
                // rope's interleaved layout, whatever its second operand did -
                // was handed to the row-mapped kernel by a divisor it was never
                // going to use, and ran 20x slower for it.
                const int64_t wrap  = bc != GK_CU_BCAST_WRAP ? 1
                                                            : (vec ? ne_b / 4 : ne_b);

                // The flat index is taken as 32 bits inside the kernel, and a
                // fast division needs a divisor of at least one.
                if (bc >= 0 && n <= GK_CU_FASTDIV_MAX && wrap >= 1 && plane >= 1 &&
                    wrap <= GK_CU_FASTDIV_MAX && plane <= GK_CU_FASTDIV_MAX) {
                    const struct gk_cu_fastdiv fr = gk_cu_fastdiv_make((uint32_t) wrap);
                    const struct gk_cu_fastdiv fp = gk_cu_fastdiv_make((uint32_t) plane);
                    const int nblk = GK_CU_FLAT_BLOCKS(n);

                    if (gk_cu_ew_dump_on()) {
                        char p[32];
                        snprintf(p, sizeof(p), "flat%s bc=%d", vec ? "4" : "", bc);
                        gk_cu_ew_dump(gk_op_name((enum gk_op) op), p, node, src0, src1);
                    }

#define GK_CU_LAUNCH_BIN_FLAT(K, B)                                                    \
                    do {                                                               \
                        if (vec) {                                                     \
                            gk_cu_k_binary_flat4<K, B><<<nblk, GK_CUDA_BLOCK, 0, stream>>>( \
                                (const float4 *) src0->data, (const float *) src1->data,    \
                                (float4 *) node->data, n, fr, fp);                     \
                        } else {                                                       \
                            gk_cu_k_binary_flat<K, B><<<nblk, GK_CUDA_BLOCK, 0, stream>>>(  \
                                (const float *) src0->data, (const float *) src1->data,     \
                                (float *) node->data, n, fr, fp);                      \
                        }                                                              \
                    } while (0)

#define GK_CU_LAUNCH_BIN_FLAT_BC(K)                                                    \
                    do {                                                               \
                        switch (bc) {                                                  \
                            case GK_CU_BCAST_NONE: GK_CU_LAUNCH_BIN_FLAT(K, GK_CU_BCAST_NONE); break; \
                            case GK_CU_BCAST_WRAP: GK_CU_LAUNCH_BIN_FLAT(K, GK_CU_BCAST_WRAP); break; \
                            case GK_CU_BCAST_CH:   GK_CU_LAUNCH_BIN_FLAT(K, GK_CU_BCAST_CH);   break; \
                            default:               GK_CU_LAUNCH_BIN_FLAT(K, GK_CU_BCAST_ONE);  break; \
                        }                                                              \
                    } while (0)

                    switch (kind) {
                        case 0:  GK_CU_LAUNCH_BIN_FLAT_BC(0); break;
                        case 1:  GK_CU_LAUNCH_BIN_FLAT_BC(1); break;
                        case 2:  GK_CU_LAUNCH_BIN_FLAT_BC(2); break;
                        default: GK_CU_LAUNCH_BIN_FLAT_BC(3); break;
                    }
#undef GK_CU_LAUNCH_BIN_FLAT_BC
#undef GK_CU_LAUNCH_BIN_FLAT
                    return true;
                }
            }

            const struct gk_cu_rows_plan p = gk_cu_rows_plan_make(node, src0, src1);

            gk_cu_ew_dump(gk_op_name((enum gk_op) op), p.ok ? (p.vec ? "rows4" : "rows") : "generic",
                          node, src0, src1);

            if (p.ok) {
#define GK_CU_LAUNCH_BIN_ROWS(K)                                                  \
                do {                                                                   \
                    if (p.vec) {                                                       \
                        if (p.bcast0) {                                                \
                            gk_cu_k_binary_rows4<K, true><<<p.grid, p.block, 0, stream>>>( \
                                (const float *) src0->data, (const float *) src1->data, \
                                (float *) node->data, node->ne[0] / 4,                  \
                                p.as[1], p.as[2], p.as[3],                              \
                                p.bs[1], p.bs[2], p.bs[3],                              \
                                p.ds[1], p.ds[2], p.ds[3], p.ne2);                      \
                        } else {                                                        \
                            gk_cu_k_binary_rows4<K, false><<<p.grid, p.block, 0, stream>>>( \
                                (const float *) src0->data, (const float *) src1->data, \
                                (float *) node->data, node->ne[0] / 4,                  \
                                p.as[1], p.as[2], p.as[3],                              \
                                p.bs[1], p.bs[2], p.bs[3],                              \
                                p.ds[1], p.ds[2], p.ds[3], p.ne2);                      \
                        }                                                               \
                    } else {                                                            \
                        if (p.bcast0) {                                                 \
                            gk_cu_k_binary_rows<K, true><<<p.grid, p.block, 0, stream>>>( \
                                (const float *) src0->data, (const float *) src1->data, \
                                (float *) node->data, node->ne[0],                      \
                                p.as[1], p.as[2], p.as[3],                              \
                                p.bs[0], p.bs[1], p.bs[2], p.bs[3],                     \
                                p.ds[1], p.ds[2], p.ds[3], p.ne2);                      \
                        } else {                                                        \
                            gk_cu_k_binary_rows<K, false><<<p.grid, p.block, 0, stream>>>( \
                                (const float *) src0->data, (const float *) src1->data, \
                                (float *) node->data, node->ne[0],                      \
                                p.as[1], p.as[2], p.as[3],                              \
                                p.bs[0], p.bs[1], p.bs[2], p.bs[3],                     \
                                p.ds[1], p.ds[2], p.ds[3], p.ne2);                      \
                        }                                                               \
                    }                                                                   \
                } while (0)

                switch (kind) {
                    case 0:  GK_CU_LAUNCH_BIN_ROWS(0); break;
                    case 1:  GK_CU_LAUNCH_BIN_ROWS(1); break;
                    case 2:  GK_CU_LAUNCH_BIN_ROWS(2); break;
                    default: GK_CU_LAUNCH_BIN_ROWS(3); break;
                }
#undef GK_CU_LAUNCH_BIN_ROWS
                return true;
            }

            gk_cu_k_binary<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node), kind, ne);
            return true;
        }

        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS: {
            const int which = op == GK_OP_SQR ? 0 : op == GK_OP_SQRT ? 1 :
                              op == GK_OP_LOG ? 2 : op == GK_OP_SIN ? 3 : 4;
            gk_cu_k_simple<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), which, ne);
            return true;
        }

        case GK_OP_UNARY: {
            const int uop = (int) gk_get_unary_op(node);

            if (gk_cu_flat_ok(node) && gk_cu_flat_ok(src0)) {
                const bool  vec  = gk_cu_flat_vec(node, ne) && gk_cu_flat_vec(src0, ne);
                const int64_t n  = vec ? ne / 4 : ne;
                const int   nblk = GK_CU_FLAT_BLOCKS(n);

                const float p1 = gk_get_op_params_f32(node, 1);
                const float p2 = gk_get_op_params_f32(node, 2);
                const float p3 = gk_get_op_params_f32(node, 3);
                const float p4 = gk_get_op_params_f32(node, 4);

                gk_cu_ew_dump("unary", vec ? "flat4" : "flat", node, src0, NULL);

                if (vec) {
                    gk_cu_k_unary_flat4<<<nblk, GK_CUDA_BLOCK, 0, stream>>>(
                        (const float4 *) src0->data, (float4 *) node->data, n,
                        uop, p1, p2, p3, p4);
                } else {
                    gk_cu_k_unary_flat<<<nblk, GK_CUDA_BLOCK, 0, stream>>>(
                        (const float *) src0->data, (float *) node->data, n,
                        uop, p1, p2, p3, p4);
                }
                return true;
            }

            const struct gk_cu_rows_plan p = gk_cu_rows_plan_make(node, src0, NULL);

            gk_cu_ew_dump("unary", p.ok ? (p.vec ? "rows4" : "rows") : "generic",
                          node, src0, NULL);

            if (p.ok) {
                const float p1 = gk_get_op_params_f32(node, 1);
                const float p2 = gk_get_op_params_f32(node, 2);
                const float p3 = gk_get_op_params_f32(node, 3);
                const float p4 = gk_get_op_params_f32(node, 4);

                if (p.vec) {
                    gk_cu_k_unary_rows4<<<p.grid, p.block, 0, stream>>>(
                        (const float *) src0->data, (float *) node->data, node->ne[0] / 4,
                        p.as[1], p.as[2], p.as[3], p.ds[1], p.ds[2], p.ds[3],
                        uop, p1, p2, p3, p4, p.ne2);
                } else {
                    gk_cu_k_unary_rows<<<p.grid, p.block, 0, stream>>>(
                        (const float *) src0->data, (float *) node->data, node->ne[0],
                        p.as[1], p.as[2], p.as[3], p.ds[1], p.ds[2], p.ds[3],
                        uop, p1, p2, p3, p4, p.ne2);
                }
                return true;
            }

            gk_cu_k_unary<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), uop,
                gk_get_op_params_f32(node, 1), gk_get_op_params_f32(node, 2),
                gk_get_op_params_f32(node, 3), gk_get_op_params_f32(node, 4), ne);
            return true;
        }

        case GK_OP_LEAKY_RELU:
            gk_cu_k_leaky_relu<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), gk_get_op_params_f32(node, 0), ne);
            return true;

        case GK_OP_GLU: {
            const int gop = (int) gk_get_glu_op(node);
            const bool swapped = gk_get_op_params_i32(node, 1) != 0;
            gk_cu_k_glu<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), src1 ? gk_cu_view(src1) : gk_cu_view(src0), src1 != NULL,
                gk_cu_view_mut(node), gop, swapped,
                gk_get_op_params_f32(node, 2), gk_get_op_params_f32(node, 3), ne);
            return true;
        }

        case GK_OP_SCALE:
            gk_cu_k_scale<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_f32(node, 0), gk_get_op_params_f32(node, 1), ne);
            return true;

        case GK_OP_CLAMP:
            gk_cu_k_clamp<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_f32(node, 0), gk_get_op_params_f32(node, 1), ne);
            return true;

        case GK_OP_FILL:
            gk_cu_k_fill<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view_mut(node), gk_get_op_params_f32(node, 0), ne);
            return true;

        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM: {
            const int kind = op == GK_OP_RMS_NORM ? 0 : op == GK_OP_NORM ? 1 : 2;
            const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];
            gk_cu_k_norm<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), kind, gk_get_op_params_f32(node, 0));
            return true;
        }

        case GK_OP_GROUP_NORM: {
            const int n_groups = gk_get_op_params_i32(node, 0);

            if (gk_cu_flat_ok(node) && gk_cu_flat_ok(src0) && src0->ne[2] % n_groups == 0) {
                const int64_t count = src0->ne[0] * src0->ne[1] * (src0->ne[2] / n_groups);
                const int     nblk  = (int) (node->ne[3] * n_groups);
                const bool    vec   = count % 4 == 0 &&
                                      (uintptr_t) src0->data % 16 == 0 &&
                                      (uintptr_t) node->data % 16 == 0;

                if (vec) {
                    gk_cu_k_group_norm_flat<true><<<nblk, GK_CU_GNORM_BLOCK, 0, stream>>>(
                        (const float *) src0->data, (float *) node->data, count,
                        gk_get_op_params_f32(node, 1));
                } else {
                    gk_cu_k_group_norm_flat<false><<<nblk, GK_CU_GNORM_BLOCK, 0, stream>>>(
                        (const float *) src0->data, (float *) node->data, count,
                        gk_get_op_params_f32(node, 1));
                }
                return true;
            }

            gk_cu_k_group_norm<<<(int) (node->ne[3] * n_groups), GK_CU_NORM_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), n_groups, gk_get_op_params_f32(node, 1));
            return true;
        }

        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
            gk_cu_k_copy<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_are_same_shape(src0, node), ne);
            return true;

        case GK_OP_GET_ROWS:
            gk_cu_k_get_rows<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node), ne);
            return true;

        case GK_OP_SET_ROWS: {
            const int64_t n = gk_cu_nelements(src0);
            gk_cu_k_set_rows<<<gk_cu_blocks(n, GK_CUDA_BLOCK), GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                src1->type == GKT_I64, n);
            return true;
        }

        case GK_OP_REPEAT:
            gk_cu_k_repeat<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), ne);
            return true;

        case GK_OP_CONCAT:
            gk_cu_k_concat<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), ne);
            return true;

        case GK_OP_ADD_ID:
            gk_cu_k_add_id<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view(src2),
                gk_cu_view_mut(node), ne);
            return true;

        case GK_OP_SOFT_MAX: {
            const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];

            int64_t n_head_log2 = 1;
            while (n_head_log2 * 2 <= src0->ne[2]) {
                n_head_log2 *= 2;
            }

            gk_cu_k_soft_max<<<(int) rows, GK_CU_SOFTMAX_BLOCK, 0, stream>>>(
                gk_cu_view(src0),
                src1 ? gk_cu_view(src1) : gk_cu_view(src0), src1 != NULL,
                src2 ? (const float *) src2->data : NULL,
                gk_cu_view_mut(node),
                gk_get_op_params_f32(node, 0), gk_get_op_params_f32(node, 1), n_head_log2);
            return true;
        }

        case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
            gk_cu_k_diag_mask<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0),
                op == GK_OP_DIAG_MASK_INF ? -INFINITY : 0.0f, ne);
            return true;

        case GK_OP_ROPE: {
            gk_cu_rope_params p;
            p.n_dims      = gk_get_op_params_i32(node, 1);
            p.mode        = gk_get_op_params_i32(node, 2);
            p.freq_base   = gk_get_op_params_f32(node, 5);
            p.freq_scale  = gk_get_op_params_f32(node, 6);
            p.ext_factor  = gk_get_op_params_f32(node, 7);
            p.attn_factor = gk_get_op_params_f32(node, 8);

            for (int i = 0; i < 4; ++i) {
                p.sections[i] = gk_get_op_params_i32(node, 11 + i);
            }

            gk_rope_corr_dims(p.n_dims, gk_get_op_params_i32(node, 4), p.freq_base,
                              gk_get_op_params_f32(node, 9), gk_get_op_params_f32(node, 10),
                              p.corr_dims);

            p.theta_scale = powf(p.freq_base, -2.0f / (float) p.n_dims);
            p.neox   = (p.mode & GK_ROPE_TYPE_NEOX) != 0;
            p.mrope  = (p.mode & GK_ROPE_TYPE_MROPE) != 0;
            p.vision = p.mode == GK_ROPE_TYPE_VISION;
            p.imrope = p.mode == GK_ROPE_TYPE_IMROPE;

            const int64_t rows  = node->ne[1] * node->ne[2] * node->ne[3];
            const int64_t n_rot = p.vision ? node->ne[0] : p.n_dims;
            const int64_t pairs = rows * (n_rot / 2);

            gk_cu_k_rope<<<gk_cu_blocks(pairs, GK_CUDA_BLOCK), GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), (const int32_t *) src1->data,
                src2 ? (const float *) src2->data : NULL,
                gk_cu_view_mut(node), p, pairs);

            if (!p.vision && node->ne[0] > p.n_dims) {
                const int64_t rest = rows * (node->ne[0] - p.n_dims);
                gk_cu_k_rope_passthrough<<<gk_cu_blocks(rest, GK_CUDA_BLOCK),
                                           GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src0), gk_cu_view_mut(node), p.n_dims, rest);
            }
            return true;
        }

        case GK_OP_SUM_ROWS: case GK_OP_MEAN: {
            const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];
            gk_cu_k_sum_rows<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), op == GK_OP_MEAN);
            return true;
        }

        case GK_OP_SUM:
            gk_cu_k_sum<<<1, GK_CU_NORM_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), gk_cu_nelements(src0));
            return true;

        case GK_OP_ARGSORT: case GK_OP_TOP_K: {
            const int64_t rows  = node->ne[1] * node->ne[2] * node->ne[3];
            const int64_t n     = src0->ne[0];
            const int64_t k_out = node->ne[0];

            // top_k is a selection of the largest, so it has no order
            // parameter of its own; argsort carries one.
            const bool desc = op == GK_OP_TOP_K ||
                (enum gk_sort_order) gk_get_op_params_i32(node, 0) == GK_SORT_ORDER_DESC;

            int64_t n_pad = 1;
            while (n_pad < n) {
                n_pad <<= 1;
            }

            const int grid = gk_cu_blocks(rows, 1);

            if (n_pad <= GK_CU_SORT_MAX_PAD) {
                // Enough threads for half the network's slots - each pair is
                // handled once - and no more than the row can use.
                int block = GK_WARP_SIZE;
                while (block < n_pad / 2 && block < GK_CU_SORT_MAX_BLOCK) {
                    block <<= 1;
                }

                const size_t smem = (size_t) n_pad * (sizeof(float) + sizeof(int32_t));

                gk_cu_k_argsort<<<grid, block, smem, stream>>>(
                    gk_cu_view(src0), gk_cu_view_mut(node),
                    n, (int) n_pad, k_out, desc, op == GK_OP_TOP_K, rows);
                return true;
            }

            // Too wide for one network. A selection composes, so top_k can be
            // done in rounds over chunks; a sort cannot, so argsort runs the
            // same network out of global memory instead.
            if (op == GK_OP_TOP_K && k_out <= GK_CU_TOPK_MAX_K) {
                if (gk_cu_top_k_wide(stream, scratch, src0, node, rows, n, k_out)) {
                    return true;
                }
            } else if (gk_cu_argsort_wide(stream, scratch, src0, node,
                                          rows, n, k_out, desc)) {
                return true;
            }

            // What is left: a row so wide that even the scratch for it could
            // not be had, or a top_k with a k so large that rounds would not
            // converge. Every element against every other, which is slow but
            // finite - and the single-backend caller has no CPU to fall back
            // to.
            gk_cu_k_argsort_rank<<<grid, GK_CU_SORT_MAX_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                n, k_out, desc, op == GK_OP_TOP_K, rows);
            return true;
        }

        case GK_OP_PAD:
            gk_cu_k_pad<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                gk_get_op_params_i32(node, 2), gk_get_op_params_i32(node, 3),
                gk_get_op_params_i32(node, 4), gk_get_op_params_i32(node, 5),
                gk_get_op_params_i32(node, 6), gk_get_op_params_i32(node, 7),
                gk_get_op_params_i32(node, 8) != 0, ne);
            return true;

        case GK_OP_UPSCALE: {
            const int mode_flags = gk_get_op_params_i32(node, 0);
            const int mode = mode_flags & 0xFF;

            float sf0 = (float) node->ne[0] / src0->ne[0];
            float sf1 = (float) node->ne[1] / src0->ne[1];
            const float sf2 = (float) node->ne[2] / src0->ne[2];
            const float sf3 = (float) node->ne[3] / src0->ne[3];
            float pixel_offset = 0.5f;

            if (mode_flags & GK_SCALE_FLAG_ALIGN_CORNERS) {
                pixel_offset = 0.0f;
                if (node->ne[0] > 1 && src0->ne[0] > 1) {
                    sf0 = (float) (node->ne[0] - 1) / (src0->ne[0] - 1);
                }
                if (node->ne[1] > 1 && src0->ne[1] > 1) {
                    sf1 = (float) (node->ne[1] - 1) / (src0->ne[1] - 1);
                }
            }

            gk_cu_k_upscale<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), mode,
                sf0, sf1, sf2, sf3, pixel_offset, ne);
            return true;
        }

        case GK_OP_TIMESTEP_EMBEDDING: {
            const int dim = gk_get_op_params_i32(node, 0);
            const int64_t n = node->ne[1] * (dim / 2);
            gk_cu_k_timestep_embedding<<<gk_cu_blocks(n, GK_CUDA_BLOCK),
                                         GK_CUDA_BLOCK, 0, stream>>>(
                (const float *) src0->data, gk_cu_view_mut(node),
                dim, gk_get_op_params_i32(node, 1), n);
            return true;
        }

        case GK_OP_ARANGE:
            gk_cu_k_arange<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                (float *) node->data, gk_get_op_params_f32(node, 0),
                gk_get_op_params_f32(node, 2), ne);
            return true;

        case GK_OP_IM2COL: {
            const int s0 = gk_get_op_params_i32(node, 0);
            const int s1 = gk_get_op_params_i32(node, 1);
            const int p0 = gk_get_op_params_i32(node, 2);
            const int p1 = gk_get_op_params_i32(node, 3);
            const int d0 = gk_get_op_params_i32(node, 4);
            const int d1 = gk_get_op_params_i32(node, 5);
            const bool is_2D = gk_get_op_params_i32(node, 6) != 0;

            const int64_t IC = is_2D ? src1->ne[2] : src1->ne[1];
            const int64_t IH = is_2D ? src1->ne[1] : 1;
            const int64_t IW = src1->ne[0];
            const int64_t KH = is_2D ? src0->ne[1] : 1;
            const int64_t KW = src0->ne[0];
            const int64_t OH = is_2D ? node->ne[2] : 1;
            const int64_t OW = node->ne[1];

            const int64_t ofs_n = (int64_t) (is_2D ? src1->nb[3] : src1->nb[2]);
            const int64_t ofs_c = (int64_t) (is_2D ? src1->nb[2] : src1->nb[1]);
            const int64_t ofs_h = (int64_t) (is_2D ? src1->nb[1] : 0);

            // Every divisor divides the element count, so one bound covers
            // all of them.
            if (ne <= (int64_t) GK_CU_FASTDIV_MAX) {
                struct gk_cu_im2col_geom g;

                g.patch = gk_cu_fastdiv_make((uint32_t) (IC * KH * KW));
                g.ow    = gk_cu_fastdiv_make((uint32_t) OW);
                g.oh    = gk_cu_fastdiv_make((uint32_t) OH);
                g.kw    = gk_cu_fastdiv_make((uint32_t) KW);
                g.kh    = gk_cu_fastdiv_make((uint32_t) KH);

                g.IH    = IH;
                g.IW    = IW;
                g.ofs_n = ofs_n;
                g.ofs_c = ofs_c;
                g.ofs_h = ofs_h;

                g.s0 = s0; g.s1 = s1;
                g.p0 = p0; g.p1 = p1;
                g.d0 = d0; g.d1 = d1;

                gk_cu_k_im2col_fast<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src1), gk_cu_view_mut(node), g, (uint32_t) ne);
                return true;
            }

            gk_cu_k_im2col<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src1), gk_cu_view_mut(node),
                IC, IH, IW, KH, KW, OH, OW, ofs_n, ofs_c, ofs_h,
                s0, s1, p0, p1, d0, d1, ne);
            return true;
        }

        case GK_OP_ROLL:
            gk_cu_k_roll<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                gk_get_op_params_i32(node, 2), gk_get_op_params_i32(node, 3), ne);
            return true;

        case GK_OP_SSM_CONV:
            gk_cu_k_ssm_conv<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                src1->ne[0], ne);
            return true;

        case GK_OP_POOL_2D:
            gk_cu_k_pool_2d<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0),
                gk_get_op_params_i32(node, 1), gk_get_op_params_i32(node, 2),
                gk_get_op_params_i32(node, 3), gk_get_op_params_i32(node, 4),
                gk_get_op_params_i32(node, 5), gk_get_op_params_i32(node, 6),
                src0->ne[0], src0->ne[1], ne);
            return true;

        case GK_OP_PAD_REFLECT_1D:
            gk_cu_k_pad_reflect_1d<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                src0->ne[0], ne);
            return true;

        case GK_OP_ARGMAX: {
            const size_t smem = (size_t) GK_CUDA_BLOCK * (sizeof(float) + sizeof(int32_t));
            gk_cu_k_argmax<<<(unsigned) src0->ne[1], GK_CUDA_BLOCK, smem, stream>>>(
                gk_cu_view(src0), (int32_t *) node->data, src0->ne[0]);
            return true;
        }

        case GK_OP_CUMSUM: {
            const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];
            const int64_t n    = node->ne[0];
            const int     n_chunks =
                (int) ((n + GK_CU_SCAN_CHUNK - 1) / GK_CU_SCAN_CHUNK);

            float * sums = (float *) gk_cu_scratch_get(
                scratch, (size_t) rows * n_chunks * sizeof(float), stream);
            if (sums == NULL) {
                return false;
            }

            dim3 grid;
            grid.x = (unsigned) n_chunks;
            grid.y = (unsigned) rows;
            grid.z = 1;

            const int n_warps = GK_CU_SCAN_BLOCK / GK_WARP_SIZE;

            gk_cu_k_cumsum_totals<<<grid, GK_CU_SCAN_BLOCK,
                                    n_warps * sizeof(float), stream>>>(
                gk_cu_view(src0), sums, n, n_chunks);

            gk_cu_k_cumsum_offsets<<<(unsigned) rows, 1, 0, stream>>>(sums, n_chunks);

            gk_cu_k_cumsum_apply<<<grid, GK_CU_SCAN_BLOCK,
                                   GK_CU_SCAN_BLOCK * sizeof(float), stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), sums, n, n_chunks);
            return true;
        }

        case GK_OP_IM2COL_3D: {
            const int64_t IC = gk_get_op_params_i32(node, 9);
            const int64_t N  = src1->ne[3] / IC;

            if (ne <= (int64_t) GK_CU_FASTDIV_MAX) {
                const int64_t KW = src0->ne[0];
                const int64_t KH = src0->ne[1];
                const int64_t KD = src0->ne[2];

                gk_cu_k_im2col_3d_fast<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src1), gk_cu_view_mut(node),
                    gk_cu_fastdiv_make((uint32_t) (IC * KD * KH * KW)),
                    gk_cu_fastdiv_make((uint32_t) node->ne[1]),
                    gk_cu_fastdiv_make((uint32_t) node->ne[2]),
                    gk_cu_fastdiv_make((uint32_t) (node->ne[3] / N)),
                    gk_cu_fastdiv_make((uint32_t) KW),
                    gk_cu_fastdiv_make((uint32_t) KH),
                    gk_cu_fastdiv_make((uint32_t) KD),
                    gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                    gk_get_op_params_i32(node, 2), gk_get_op_params_i32(node, 3),
                    gk_get_op_params_i32(node, 4), gk_get_op_params_i32(node, 5),
                    gk_get_op_params_i32(node, 6), gk_get_op_params_i32(node, 7),
                    gk_get_op_params_i32(node, 8),
                    IC, src1->ne[0], src1->ne[1], src1->ne[2], (uint32_t) ne);
                return true;
            }

            gk_cu_k_im2col_3d<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src1), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                gk_get_op_params_i32(node, 2), gk_get_op_params_i32(node, 3),
                gk_get_op_params_i32(node, 4), gk_get_op_params_i32(node, 5),
                gk_get_op_params_i32(node, 6), gk_get_op_params_i32(node, 7),
                gk_get_op_params_i32(node, 8),
                IC, src1->ne[0], src1->ne[1], src1->ne[2],
                src0->ne[0], src0->ne[1], src0->ne[2],
                node->ne[1], node->ne[2], node->ne[3] / N, ne);
            return true;
        }

        case GK_OP_CONV_2D_DW:
            gk_cu_k_conv_2d_dw<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                gk_get_op_params_i32(node, 2), gk_get_op_params_i32(node, 3),
                gk_get_op_params_i32(node, 4), gk_get_op_params_i32(node, 5),
                src1->ne[0], src1->ne[1], src0->ne[0], src0->ne[1], ne);
            return true;

        case GK_OP_RWKV_WKV6: {
            const int64_t T = src1->ne[2];
            const int64_t C = node->ne[0];
            const int64_t H = src1->ne[1];
            const int64_t n_seqs = node->src[5]->ne[1];
            const int64_t S = C / H;

            dim3 grid;
            grid.x = (unsigned) H;
            grid.y = (unsigned) n_seqs;
            grid.z = 1;

            gk_cu_k_rwkv_wkv6<<<grid, (unsigned) gk_cu_round_warp(S), 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) node->src[2]->data, (const float *) node->src[3]->data,
                (const float *) node->src[4]->data, (const float *) node->src[5]->data,
                (float *) node->data, (float *) node->data + C * T,
                T, C, S, T / n_seqs);
            return true;
        }

        case GK_OP_RWKV_WKV7: {
            const int64_t T = src1->ne[2];
            const int64_t C = node->ne[0];
            const int64_t H = src1->ne[1];
            const int64_t n_seqs = node->src[6]->ne[1];
            const int64_t S = C / H;

            dim3 grid;
            grid.x = (unsigned) H;
            grid.y = (unsigned) n_seqs;
            grid.z = 1;

            gk_cu_k_rwkv_wkv7<<<grid, GK_CU_RECURRENT_BLOCK, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) src2->data, (const float *) node->src[3]->data,
                (const float *) node->src[4]->data, (const float *) node->src[5]->data,
                (const float *) node->src[6]->data,
                (float *) node->data, (float *) node->data + C * T,
                T, C, S, T / n_seqs);
            return true;
        }

        case GK_OP_GATED_DELTA_NET: {
            const struct gk_tensor * v    = node->src[2];
            const struct gk_tensor * g    = node->src[3];
            const struct gk_tensor * s_in = node->src[5];

            const int64_t S = v->ne[0];
            const int64_t H = v->ne[1];
            const int64_t n_tokens = v->ne[2];
            const int64_t n_seqs   = v->ne[3];

            const int64_t K = gk_get_op_params_i32(node, 0);
            const bool  kda = g->ne[0] == S;

            const int64_t attn_elems = S * H * n_tokens * n_seqs;
            const int64_t snap_elems = S * S * H * n_seqs;

            const int64_t rq3 = v->ne[3] / src0->ne[3];
            const int64_t rk3 = v->ne[3] / src1->ne[3];
            const float scale = 1.0f / sqrtf((float) S);

            // published head widths take the register-resident form; anything
            // else falls through to the generic kernel
            switch (S) {
                case 16:
                    gk_cu_gdn_col_launch<16>(kda, H, n_seqs, stream,
                        gk_cu_view(src0), gk_cu_view(src1), gk_cu_view(v),
                        gk_cu_view(g), gk_cu_view(node->src[4]),
                        (const float *) s_in->data, (int64_t) (s_in->nb[3] / sizeof(float)),
                        (float *) node->data, (float *) node->data + attn_elems,
                        n_tokens, snap_elems, K, rq3, rk3, scale);
                    return true;
                case 32:
                    gk_cu_gdn_col_launch<32>(kda, H, n_seqs, stream,
                        gk_cu_view(src0), gk_cu_view(src1), gk_cu_view(v),
                        gk_cu_view(g), gk_cu_view(node->src[4]),
                        (const float *) s_in->data, (int64_t) (s_in->nb[3] / sizeof(float)),
                        (float *) node->data, (float *) node->data + attn_elems,
                        n_tokens, snap_elems, K, rq3, rk3, scale);
                    return true;
                case 64:
                    gk_cu_gdn_col_launch<64>(kda, H, n_seqs, stream,
                        gk_cu_view(src0), gk_cu_view(src1), gk_cu_view(v),
                        gk_cu_view(g), gk_cu_view(node->src[4]),
                        (const float *) s_in->data, (int64_t) (s_in->nb[3] / sizeof(float)),
                        (float *) node->data, (float *) node->data + attn_elems,
                        n_tokens, snap_elems, K, rq3, rk3, scale);
                    return true;
                case 128:
                    gk_cu_gdn_col_launch<128>(kda, H, n_seqs, stream,
                        gk_cu_view(src0), gk_cu_view(src1), gk_cu_view(v),
                        gk_cu_view(g), gk_cu_view(node->src[4]),
                        (const float *) s_in->data, (int64_t) (s_in->nb[3] / sizeof(float)),
                        (float *) node->data, (float *) node->data + attn_elems,
                        n_tokens, snap_elems, K, rq3, rk3, scale);
                    return true;
                default:
                    break;
            }

            dim3 grid;
            grid.x = (unsigned) H;
            grid.y = (unsigned) n_seqs;
            grid.z = 1;

            gk_cu_k_gated_delta_net<<<grid, GK_CU_RECURRENT_BLOCK,
                                      kda ? (size_t) S * sizeof(float) : 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view(v),
                gk_cu_view(g), gk_cu_view(node->src[4]),
                (const float *) s_in->data, (int64_t) (s_in->nb[3] / sizeof(float)),
                (float *) node->data, (float *) node->data + attn_elems,
                S, H, n_tokens, snap_elems, K,
                rq3, rk3, kda, scale);
            return true;
        }

        case GK_OP_CONV_TRANSPOSE_1D:
            gk_cu_k_conv_transpose_1d<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0),
                src0->ne[0], src0->ne[2], src1->ne[0],
                src0->type == GKT_F16, ne);
            return true;

        case GK_OP_CONV_2D:
            gk_cu_k_conv_2d<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                gk_get_op_params_i32(node, 2), gk_get_op_params_i32(node, 3),
                gk_get_op_params_i32(node, 4), gk_get_op_params_i32(node, 5),
                src1->ne[2], src1->ne[0], src1->ne[1], src0->ne[0], src0->ne[1],
                src0->type == GKT_F16, ne);
            return true;

        case GK_OP_SSM_SCAN: {
            const struct gk_tensor * s_in = node->src[0];
            const struct gk_tensor * dt   = node->src[2];
            const struct gk_tensor * A    = node->src[3];
            const struct gk_tensor * B    = node->src[4];

            const int64_t nc = s_in->ne[0];  // d_state
            const int64_t nr = s_in->ne[1];  // channels per head
            const int64_t nh = src1->ne[1];  // heads
            const int64_t ng = B->ne[1];     // groups sharing a B/C pair
            const int64_t nt = src1->ne[2];  // tokens
            const int64_t ns = src1->ne[3];  // sequences

            // One thread per (sequence, head, channel); the token loop is
            // inside the thread because the recurrence is sequential in it.
            const int64_t threads = ns * nh * nr;

            gk_cu_k_ssm_scan<<<gk_cu_blocks(threads, GK_CUDA_BLOCK),
                               GK_CUDA_BLOCK, 0, stream>>>(
                (const char *) s_in->data, (const char *) src1->data,
                (const char *) dt->data, (const float *) A->data,
                (const char *) B->data, (const char *) node->src[5]->data,
                (const int32_t *) node->src[6]->data, (char *) node->data,
                nc, nr, nh, ng, nt, nh / ng,
                (int64_t) s_in->nb[3], (int64_t) src1->nb[2], (int64_t) src1->nb[3],
                (int64_t) dt->nb[1], (int64_t) dt->nb[2],
                (int64_t) B->nb[2], (int64_t) B->nb[3],
                gk_cu_nelements(src1) * (int64_t) sizeof(float),
                A->ne[0] == 1, threads);
            return true;
        }

        case GK_OP_MUL_MAT:
            gk_cuda_mul_mat(stream, scratch, node);
            return true;

        case GK_OP_MUL_MAT_ID:
            gk_cuda_mul_mat_id(stream, node);
            return true;

        case GK_OP_FLASH_ATTN_EXT:
            gk_cuda_flash_attn(stream, scratch, node);
            return true;

        default:
            return false;
    }
}
