// The matmuls and fused attention.
//
// These are separated from the other kernels because they are the only ones
// whose shape is chosen for speed rather than for clarity. Everything else in
// this backend is one thread per output element; that is the wrong shape here,
// because a matmul's output element is a whole reduction over k and a weight
// row read once per output element is the entire memory traffic of inference.
//
// So the unit of work is a block per output element (or per small group of
// them), the reduction is a block reduction, and the weight row is read once
// and used for several activation columns. The quantized formats are decoded
// on the fly rather than expanded into a temporary: a 4096x4096 q4_K weight is
// 9 MB packed and 64 MB as f32, and materialising that per matmul would spend
// more bandwidth than the decode costs arithmetic.
//
// The tensor-core paths came last and in that order: the clear implementation
// first, checked against the CPU, and the fast one after, checked against the
// clear one. There are two of them - one for nvfp4 through the integer
// instruction, one for f16 - and both are gated so that any shape, type or
// device they do not cover falls through to the float tile, which still
// computes the same thing.

#include "gk_cuda_ops.cuh"

#include <chrono>
#include <mutex>

#include <float.h>
#include <stdlib.h>

// GK_MM_FP4_STATS: what the nvfp4 activation quantizer is actually doing to
// this model's numbers. Three counters, summed over every block it writes:
// the energy of the input, the energy of what it reconstructs to, and how many
// groups of sixteen come out identically zero because their amax fell under
// the smallest ue4m3 scale. Together they say whether a bad result is
// four-bit rounding or a group being deleted.
__device__ double g_gk_fp4_sq_err;
__device__ double g_gk_fp4_sq_ref;
__device__ unsigned long long g_gk_fp4_zero_groups;
__device__ unsigned long long g_gk_fp4_groups;

#define GK_CU_MM_BLOCK 128

// Blocks to cover n items, uncapped: the activation quantizer indexes a flat
// range and needs the whole grid, not gk_cu_blocks' first 65535 of it.
static __host__ __forceinline__ unsigned gk_cu_blocks_1d(int64_t n, int block) {
    return (unsigned) ((n + block - 1) / block);
}

// Output rows below which the integer path is not worth taking. It costs one
// extra launch to quantize the activations, and that is amortized over the
// rows: at a few thousand it disappears, at a couple of hundred the matmul is
// too small to fill the device either way and the extra launch is most of the
// difference. Measured, the crossover is a few hundred rows; attn_k and attn_v
// are the shapes that sit below it.
#define GK_CU_MM_Q8_MIN_ROWS 512

static int gk_cu_env_int(const char * name, int def);

// Whether the integer dot path applies: the format has one, the row cuts into
// whole 32-element groups so a group never straddles two of them, and there
// are enough rows to pay for the quantize pass.
//
// The row floor applies only to the formats whose float decode is cheap - for
// those, below a few hundred rows the extra quantize launch is most of the
// difference. The k-quants, the codebook formats and the fp4 pair decode at a
// twentieth of that on the float path (a q5_K lm_head measured 110 GFLOP/s
// against the integer path's thousands), so for them the integer path wins at
// any row count a matmul actually has.
static __host__ __forceinline__ bool gk_cuda_mm_q8_supported(int type, int64_t k_len,
                                                             int64_t n_rows) {
    if (k_len % 32 != 0) {
        return false;
    }
    switch (type) {
        case GK_TYPE_Q4_0: case GK_TYPE_Q4_1: case GK_TYPE_Q8_0:
            return n_rows >= GK_CU_MM_Q8_MIN_ROWS;
        case GK_TYPE_Q4_K: case GK_TYPE_Q5_K: case GK_TYPE_Q2_K:
        case GK_TYPE_Q3_K: case GK_TYPE_IQ4_XS:
        case GK_TYPE_MXFP4: case GK_TYPE_NVFP4:
        case GK_TYPE_IQ2_XXS: case GK_TYPE_IQ3_XXS: case GK_TYPE_IQ3_S:
        case GK_TYPE_IQ2_XS: case GK_TYPE_IQ1_M:
            return true;
        // `GK_MM_Q8_SPLIT=0` puts the split-scale formats back on the float
        // decoder, which is what they used before they had an integer path.
        // Same answer, and a tenth of the speed - it is here to take the newer
        // kernel out of a bisect rather than to be set.
        case GK_TYPE_Q6_K: case GK_TYPE_IQ2_S:
            return gk_cu_env_int("GK_MM_Q8_SPLIT", 1) != 0;
        default:
            return false;
    }
}

// gk_cu_has_split_scale, on the host, for the one decision that has to be made
// before the type becomes a template parameter.
static __host__ __forceinline__ bool gk_cuda_mm_split_scale(int type) {
    return type == GK_TYPE_Q6_K || type == GK_TYPE_IQ2_XS ||
           type == GK_TYPE_IQ2_S || type == GK_TYPE_IQ1_M ||
           type == GK_TYPE_Q2_K  || type == GK_TYPE_Q3_K  ||
           type == GK_TYPE_NVFP4;
}

// This list and gk_cu_has_dp4a's are the same list seen from the two sides -
// here the host decides to take the integer path, there the device knows how
// to stage a group for it - so they are checked against each other rather than
// kept in step by hand.
static_assert(gk_cu_has_dp4a<GKT_Q4_0>()    && gk_cu_has_dp4a<GKT_Q4_1>() &&
              gk_cu_has_dp4a<GKT_Q8_0>()    && gk_cu_has_dp4a<GKT_Q4_K>() &&
              gk_cu_has_dp4a<GKT_IQ2_XXS>() && gk_cu_has_dp4a<GKT_IQ3_XXS>() &&
              gk_cu_has_dp4a<GKT_IQ3_S>()   && gk_cu_has_dp4a<GKT_Q6_K>() &&
              gk_cu_has_dp4a<GKT_IQ2_S>()   && gk_cu_has_dp4a<GKT_Q5_K>() &&
              gk_cu_has_dp4a<GKT_Q2_K>()    && gk_cu_has_dp4a<GKT_MXFP4>() &&
              gk_cu_has_dp4a<GKT_NVFP4>()   && gk_cu_has_dp4a<GKT_Q3_K>() &&
              gk_cu_has_dp4a<GKT_IQ4_XS>()  && gk_cu_has_dp4a<GKT_IQ2_XS>() &&
              gk_cu_has_dp4a<GKT_IQ1_M>(),
              "gk_cuda_mm_q8_supported offers a type gk_cu_wblk32 cannot stage");

// Activation columns one pass over a weight row serves. Each column costs a
// register accumulator and a shared-memory slot in the reduction; four is
// where the reuse stops paying for the occupancy it costs.
#define GK_CU_MM_NC 4

// Consecutive weight elements a lane takes at a time. Every quantized block
// size in the format set - 32, 64, 128, 256 - is a multiple of the largest
// value here, so an aligned run always sits inside one block and that block's
// header is decoded once for the whole run rather than once per element.
//
// The trade is against coalescing: a longer run spreads a warp's reads further
// apart, and where the header is cheap that costs more than it saves. Four is
// where the two balance for every format measured on an Ada part except q6_K,
// whose 210-byte block comes out ahead with no run at all - 0.18 ms against
// 0.21. The type is already a template parameter, so saying so per format
// costs nothing.
//
// These are measured, not derived. bench-cuda's decoder-cost group is the
// measurement to retune them against: one matmul shape in every format, where
// the only thing that varies between rows is the decode.
#define GK_CU_MM_RUN_MAX 4

template <int TYPE>
static __device__ __host__ __forceinline__ int gk_cu_mm_run() {
    return TYPE == GKT_Q6_K ? 1 : GK_CU_MM_RUN_MAX;
}

// The tiled path below. A block computes a TILE_M x TILE_N patch of the
// result, marching over k in TILE_K slices staged in shared memory, with each
// of its 16x16 threads holding a 4x4 patch in registers.
//
// The point is the decode. In the mat-vec kernel a weight element is decoded
// once and used GK_CU_MM_NC times; here it is decoded once and used TILE_N
// times, and an activation element once and used TILE_M times. For a diffusion
// transformer - hundreds of tokens per matmul rather than the one token
// generation has - that difference is the whole cost of the pass.
#define GK_CU_MM_TILE_M 64
#define GK_CU_MM_TILE_N 64
#define GK_CU_MM_TILE_K 16
#define GK_CU_MM_TILE_T 4   // results per thread per axis (TILE_M / 16)

// Below this many columns the tile is mostly padding and the mat-vec kernel,
// which splits k across the block instead, keeps the device busier.
#define GK_CU_MM_TILE_MIN_N 24

// A block reduction over `NC` accumulators at once. Written out rather than
// calling the single-value reduction NC times so the barriers are shared.
template <int NC>
static __device__ __forceinline__ void gk_cu_reduce_n(float (&acc)[NC], float * shared) {
    const int n_warps = blockDim.x / GK_WARP_SIZE;
    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = gk_cu_warp_sum(acc[j]);
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            shared[j * n_warps + warp] = acc[j];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float total = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                total += shared[j * n_warps + w];
            }
            acc[j] = total;
        }
    }
}

// --------------------------------------------------------------------------
// dst[i0,i1,i2,i3] = dot( a_row(i0, i2/r2, i3/r3), b_row(i1,i2,i3) )
//
// `a` is the weight, read along its fastest dimension, which is what lets a
// quantized weight be used without transposing it. The higher dimensions of
// `a` broadcast onto `b`'s: grouped-query attention has fewer key heads than
// query heads and each key head serves a group.
// --------------------------------------------------------------------------

// A *warp* owns an output row, not a block.
//
// The block-per-row shape this replaced spent most of a short row's time in
// its own reduction: a 1536-wide q4_0 row is 864 packed bytes, which 128
// threads read in a few instructions each and then pay two barriers and a
// shared-memory round trip to add up. At 42 GB/s against a card that does 183
// the bottleneck was never the decode.
//
// A warp needs no barriers and no shared memory at all - the reduction is
// shuffles - and a block of several warps retires several rows for one launch.
// The activations are read by every warp in the block at the same k, so they
// stay resident in L1 rather than being staged by hand; staging them through
// shared memory was tried and was slower than not.
//
// Within a warp, a lane takes gk_cu_mm_run<ATYPE>() consecutive elements at a
// time, aligned so the run never crosses a quantized block's boundary. That is
// what lets the block be found once and its header decoded once for the whole
// run instead of once per element.
template <int NC, int ATYPE>
static __global__ void gk_cu_k_mul_mat(gk_tview a, gk_tview b, gk_tview_mut d,
                                       int64_t k_len, int64_t r2, int64_t r3) {
    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = blockDim.x / GK_WARP_SIZE;

    const int64_t i0  = (int64_t) blockIdx.x * n_warps + warp; // weight row
    const int64_t c0  = (int64_t) blockIdx.y * NC;             // first column
    const int64_t i23 = blockIdx.z;

    if (i0 >= d.ne[0]) {
        return; // whole-warp uniform, so the shuffles below stay well formed
    }

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }

    const char * a_row = gk_cu_row(a, i0, a2, a3);

    const int run = gk_cu_mm_run<ATYPE>();

    for (int64_t k0 = (int64_t) lane * run; k0 < k_len;
         k0 += GK_WARP_SIZE * run) {

        const uint8_t * blk = NULL;
        int j0 = 0;
        if (gk_cu_is_blocked<ATYPE>()) {
            // k0 is a multiple of the run and every block size is a multiple
            // of it too, so the whole run lives in this one block.
            const int blck = gk_cu_blck_size(ATYPE);
            const int tsz  = gk_cu_type_size(ATYPE);
            blk = (const uint8_t *) a_row + (k0 / blck) * tsz;
            j0  = (int) (k0 % blck);
        }

        // The run's weights, decoded together, so the block's header is paid
        // for once instead of once per element. Only for the formats where
        // that is worth an array to hold them; the rest decode in place below.
        const bool use_run = gk_cu_is_blocked<ATYPE>() && gk_cu_has_run_path<ATYPE>();

        float wv[GK_CU_MM_RUN_MAX];
        if (use_run) {
            const int64_t left = k_len - k0;
            gk_cu_blk_run_t<ATYPE, GK_CU_MM_RUN_MAX>(
                blk, j0, (int) (left < run ? left : run), wv);
        }

#pragma unroll
        for (int e = 0; e < GK_CU_MM_RUN_MAX; ++e) {
            if (e >= run) {
                break;
            }
            const int64_t kk = k0 + e;
            if (kk >= k_len) {
                break;
            }

            // the reuse this kernel exists for: one decode of the weight
            // element, NC multiplies against it
            const float av = use_run ? wv[e]
                : gk_cu_is_blocked<ATYPE>() ? gk_cu_blk_elem_t<ATYPE>(blk, j0 + e)
                : gk_cu_get_t<ATYPE>(a, kk, i0, a2, a3);

#pragma unroll
            // Resolving the NC row addresses once outside this loop, so that
            // gk_cu_get's three multiply-adds per element are not repaid per
            // column, was measured and is not here: it left attn_k unchanged
            // and cost attn_v 20%, because these shapes are 256 output rows -
            // 64 blocks - and are waiting on latency rather than on address
            // arithmetic. The extra live pointers cost occupancy the indexing
            // never cost time. What this kernel wants at that width is more
            // parallelism per row, not a cheaper index.
            for (int j = 0; j < NC; ++j) {
                const int64_t col = c0 + j;
                if (col < d.ne[1]) {
                    acc[j] += av * gk_cu_get(b, kk, col, i2, i3);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = gk_cu_warp_sum(acc[j]);
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                gk_cu_set(d, i0, col, i2, i3, acc[j]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The integer path: the same mat-vec, dotted as 8-bit integers.
//
// Two kernels. The first quantizes the activation columns once per matmul -
// cheap, because there are at most a couple of dozen of them against thousands
// of weight rows. The second is the mat-vec, shaped exactly like the float one
// above (a warp per output row, no barriers, shuffles for the reduction) but
// with the inner loop replaced by an integer dot.
// --------------------------------------------------------------------------

// One thread per 32-element group. The absolute maximum sets the scale, and
// the sum of the codes is carried alongside because the asymmetric formats
// need it.
// `planes`, when non-NULL, receives the block's three scalars transposed to
// column-major - [i23][group][d, d*s, d*sl][column] - which is the layout the
// 128x128 integer tile reads them in: a warp of consecutive columns lands on
// consecutive words. Read from the q8blk array itself they are three loads
// 36*n_grp bytes apart per lane, and ncu billed that divergence at a third of
// the tile's remaining global sectors.
static __global__ void gk_cu_k_quantize_act(gk_tview b, gk_cu_q8blk * out, float * planes,
                                            int64_t n_grp, int64_t n_cols, int64_t total) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }

    const int64_t g   = t % n_grp;
    const int64_t col = (t / n_grp) % n_cols;
    const int64_t i23 = t / (n_grp * n_cols);

    const int64_t i2 = i23 % b.ne[2];
    const int64_t i3 = i23 / b.ne[2];

    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 32; ++e) {
        v[e] = gk_cu_get(b, g * 32 + e, col, i2, i3);
        amax = fmaxf(amax, fabsf(v[e]));
    }

    gk_cu_q8blk blk;
    blk.d = amax / 127.0f;

    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

    int sum    = 0;
    int sum_lo = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int packed = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = (int) rintf(v[i * 4 + e] * inv);
            sum += q;
            packed |= (q & 0xff) << (8 * e);
        }
        // the first four words are the group's first sixteen values, which is
        // the span a q6_K or iq2_s scale covers
        if (i < 4) {
            sum_lo = sum;
        }
        blk.q[i] = packed;
    }
    blk.s  = (float) sum;
    blk.sl = (float) sum_lo;

    out[t] = blk;

    if (planes != NULL) {
        float * p = planes + ((i23 * n_grp + g) * 3) * n_cols + col;
        p[0]          = blk.d;
        p[n_cols]     = blk.d * blk.s;
        p[2 * n_cols] = blk.d * blk.sl;
    }
}

// The column scale that puts a column's group scales in the middle of ue4m3's
// range, as a power of two.
//
// ue4m3 spans 2^-9 to 448, and a group scale is `amax/6`, so a group whose
// amax is below about 2^-6 has no scale that fits and one below 2^-11 cannot
// be represented at all. That is not a hypothetical: measured on this DiT the
// activation quantizer's round trip was **74.9% of signal**, because most of
// its groups live down there. A per-16 ue4m3 scale on its own is simply not a
// float; NVFP4 is a two-level format, and this is the level gk was missing.
//
// A power of two rather than `amax/T`, so that dividing by it and multiplying
// back in the epilogue are both exact and add no rounding of their own. The
// target puts the column's own amax in [512, 1024), which leaves the largest
// group scale at about 170 against ue4m3's 448 - room for the search below to
// try a code or two above it - and gives a group 2^13 times smaller than the
// column's peak a scale that is still normal.
static __device__ __forceinline__ void gk_cu_fp4_col_scale(float amax, float * scale,
                                                           float * inv) {
    if (!(amax > 0.0f)) {
        *scale = 1.0f;
        *inv   = 0.0f;
        return;
    }

    int e;
    frexpf(amax, &e);   // amax = frac * 2^e, frac in [0.5, 1)

    // Kept inside the exponent range where both directions stay normal: the
    // scale multiplies the accumulator and its inverse multiplies activations,
    // so a subnormal either way would lose what it is here to protect.
    e = min(max(e, -110), 110);

    *scale = ldexpf(1.0f, e - 10);
    *inv   = ldexpf(1.0f, 10 - e);
}

// The same pass, quantizing to nvfp4 instead of to int8, for the FP4
// tensor-core tile.
//
// One block is 64 activations against the int8 pass's 32, because 64 is what
// one `m16n8k64` covers and what one weight block holds. Within it the scale
// changes every sixteen - so this finds four amaxes, not one.
//
// Two things here are chosen to make the tile's inner loop free of work:
//
//   * The scale is quantized to ue4m3 *first* and the codes are made against
//     the quantized scale, not against the amax. The hardware will multiply by
//     the ue4m3 value, so that is the divisor the codes have to be consistent
//     with; dividing by the amax and letting the scale round afterwards would
//     put the rounding error into every one of the sixteen values instead of
//     into the scale alone.
//   * The codes are written in gk's own nibble order, eight to a word. That is
//     a permutation of k, and the tile applies the same one to the weights, so
//     the two cancel inside the dot product and neither side ever repacks.
//
// One block of threads per column rather than one thread per group, because
// the column scale above needs the whole column before any of it can be
// quantized. The column is therefore read twice - once to reduce, once to
// encode - and the second read is out of cache, since a column is at most
// 48 KB and the block that wants it has just walked it.
static __global__ void gk_cu_k_quantize_act_fp4(gk_tview b, gk_cu_fp4blk * out,
                                                float * col_scale, int64_t n_grp,
                                                int64_t n_cols, bool stats) {
    const int64_t cid = blockIdx.x;             // one block per column
    const int64_t col = cid % n_cols;
    const int64_t i23 = cid / n_cols;

    const int64_t i2 = i23 % b.ne[2];
    const int64_t i3 = i23 / b.ne[2];

    const int64_t k_len = n_grp * 64;

    // Pass one: the column's amax, reduced across the block.
    float cmax = 0.0f;
    for (int64_t e = threadIdx.x; e < k_len; e += blockDim.x) {
        cmax = fmaxf(cmax, fabsf(gk_cu_get(b, e, col, i2, i3)));
    }

    __shared__ float s_warp[GK_CUDA_BLOCK / GK_WARP_SIZE];
    __shared__ float s_cmax;

#pragma unroll
    for (int off = GK_WARP_SIZE / 2; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffff, cmax, off));
    }

    if (threadIdx.x % GK_WARP_SIZE == 0) {
        s_warp[threadIdx.x / GK_WARP_SIZE] = cmax;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float m = 0.0f;
        for (int i = 0; i < (int) (blockDim.x / GK_WARP_SIZE); ++i) {
            m = fmaxf(m, s_warp[i]);
        }
        s_cmax = m;
    }
    __syncthreads();

    float cs, cs_inv;
    gk_cu_fp4_col_scale(s_cmax, &cs, &cs_inv);

    if (threadIdx.x == 0) {
        col_scale[cid] = cs;
    }

    double sq_err = 0.0;
    double sq_ref = 0.0;
    int    zeroed = 0;
    int    groups = 0;

    // Pass two: one group of 64 per thread, strided over the column.
    for (int64_t g = threadIdx.x; g < n_grp; g += blockDim.x) {
        gk_cu_fp4blk blk;

        uint32_t sc     = 0;
        groups += 4;

        // A sub-group of sixteen at a time, because that is the unit the scale
        // covers and the search below needs all sixteen values in registers.
        for (int sub = 0; sub < 4; ++sub) {
            float v[16];
            float amax = 0.0f;

    #pragma unroll
            for (int e = 0; e < 16; ++e) {
                v[e] = gk_cu_get(b, g * 64 + sub * 16 + e, col, i2, i3) * cs_inv;
                amax = fmaxf(amax, fabsf(v[e]));
            }

            // The first candidate: the smallest scale that keeps the group inside
            // the codebook, 6 being the largest e2m1 magnitude.
            //
            // Clamped to the smallest non-zero ue4m3 rather than allowed to flush.
            // Flushing looks harmless - a group whose amax is under 2^-9 is tiny -
            // but it deletes the group outright, and on a real DiT that happened to
            // **24% of all groups**, which is not tiny in aggregate and was the
            // difference between a picture and noise. A clamped scale represents
            // such a group coarsely; a zero scale represents it not at all.
            //
            // With the column scale above in front of it this is now the rare tail
            // rather than the common case: it takes a group 2^16 below the column's
            // own peak to reach the clamp.
            int first = (int) gk_cu_f2ue4m3(amax * (1.0f / 6.0f));
            if (first == 0 && amax > 0.0f) {
                first = 1;
            }

            // Then four neighbours either side of it. The first candidate is the
            // smallest scale that does not clip, which is not the same as the one
            // that reproduces the group best: a larger scale clips the extreme
            // value but rounds the other fifteen more finely, and with only eight
            // codes that trade is often worth taking. Five squared-error
            // evaluations of sixteen values each, against a matmul that will use
            // this block thousands of times.
            int   best_code = first;
            float best_err  = FLT_MAX;

    #pragma unroll
            for (int i = 0; i < 5; ++i) {
                static const int offs[5] = { 0, -1, 1, -2, 2 };
                const int code = first + offs[i];

                if (code < 1 || code > 0x7e) {
                    continue;
                }

                const float sv  = 2.0f * gk_cu_ue4m3((uint8_t) code);
                const float inv = sv > 0.0f ? 1.0f / sv : 0.0f;

                float err = 0.0f;
    #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const float r = sv * 0.5f * (float) gk_cu_e2m1_value(gk_cu_f2e2m1(v[e] * inv));
                    const float d = r - v[e];

                    err = fmaf(d, d, err);
                }

                if (err < best_err) {
                    best_err  = err;
                    best_code = code;
                }
            }

            if (amax <= 0.0f) {
                best_code = 0;
                zeroed++;
            }

            const float sv  = 2.0f * gk_cu_ue4m3((uint8_t) best_code);
            const float inv = sv > 0.0f ? 1.0f / sv : 0.0f;

            sc |= (uint32_t) best_code << (8 * sub);

            // Into gk's own nvfp4 nibble order, not k order: a byte holds elements
            // `j` and `j + 8` of its sixteen, low nibble first. That is how the
            // weight side is stored and how the tile stages it, and the tile
            // relies on the two agreeing - see the staging comment in
            // `gk_cu_k_mul_mat_mma_fp4`.
    #pragma unroll
            for (int half = 0; half < 2; ++half) {
                uint32_t packed = 0;
    #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int q_lo = gk_cu_f2e2m1(v[half * 4 + e]     * inv);
                    const int q_hi = gk_cu_f2e2m1(v[half * 4 + e + 8] * inv);

                    packed |= (uint32_t) q_lo << (8 * e);
                    packed |= (uint32_t) q_hi << (8 * e + 4);
                }
                blk.q[sub * 2 + half] = packed;
            }

            if (stats) {
    #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const float r = sv * 0.5f * (float) gk_cu_e2m1_value(gk_cu_f2e2m1(v[e] * inv));
                    const double d = (double) (r - v[e]);

                    sq_err += d * d;
                    sq_ref += (double) v[e] * (double) v[e];
                }
            }
        }

        blk.sc = sc;
        out[cid * n_grp + g] = blk;
    }

    if (stats) {
        // Reported in the column's own units, not the rescaled ones. The ratio
        // is what the printer shows and a common factor cancels out of it, but
        // the two sums are also summed across columns of very different
        // magnitudes, and there the factor does not cancel.
        const double cs2 = (double) cs * (double) cs;

        atomicAdd(&g_gk_fp4_sq_err, sq_err * cs2);
        atomicAdd(&g_gk_fp4_sq_ref, sq_ref * cs2);
        atomicAdd(&g_gk_fp4_zero_groups, (unsigned long long) zeroed);
        atomicAdd(&g_gk_fp4_groups,      (unsigned long long) groups);
    }
}

template <int NC, int ATYPE>
static __global__ void gk_cu_k_mul_mat_q8(gk_tview a, gk_tview_mut d,
                                          const gk_cu_q8blk * aq, int64_t n_grp,
                                          int64_t r2, int64_t r3) {
    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = blockDim.x / GK_WARP_SIZE;

    const int64_t i0  = (int64_t) blockIdx.x * n_warps + warp;
    const int64_t c0  = (int64_t) blockIdx.y * NC;
    const int64_t i23 = blockIdx.z;

    if (i0 >= d.ne[0]) {
        return; // whole-warp uniform, so the shuffles below stay well formed
    }

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }

    const uint8_t * a_row = (const uint8_t *) gk_cu_row(a, i0, a2, a3);

    // The quantized activations for this (i2, i3) slice, one run of groups per
    // column.
    const gk_cu_q8blk * aq23 = aq + i23 * d.ne[1] * n_grp;

    for (int64_t g = lane; g < n_grp; g += GK_WARP_SIZE) {
        // Decode the weight group once, then dot it against each column.
        //
        // The reuse this kernel exists for, the same way the float one above
        // spells it: the weight is read and unpacked once and NC accumulators
        // take a turn against it. Calling gk_cu_vecdot32 per column instead
        // hands the decode the same (row, g) NC times, and for the formats
        // whose decode is a codebook gather that is most of the work - at four
        // columns it cost between 1.7x and 2.8x the single-column pass, which
        // is the pass speculative decoding replaces every matmul with.
        int   codes[8];
        float scale[2], offset[2];
        gk_cu_wblk32<ATYPE>(a_row, g, codes, scale, offset);

#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                acc[j] += gk_cu_vecdot32_pre<ATYPE>(codes, scale, offset,
                                                    aq23[col * n_grp + g]);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = gk_cu_warp_sum(acc[j]);
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                gk_cu_set(d, i0, col, i2, i3, acc[j]);
            }
        }
    }
}

// The same dot with a whole block on one row, for the outputs too narrow to
// fill the device a warp at a time. An attention k/v projection is 256 rows:
// warp-per-row is 64 blocks on a card with 170 multiprocessors, and each of
// those warps walks its groups in one long dependent chain. A block per row
// is four times the blocks and a quarter the chain, and the price - one block
// reduction per output instead of one warp shuffle - is paid once per row,
// not once per group.
template <int NC, int ATYPE>
static __global__ void gk_cu_k_mul_mat_q8_row(gk_tview a, gk_tview_mut d,
                                              const gk_cu_q8blk * aq, int64_t n_grp,
                                              int64_t r2, int64_t r3) {
    __shared__ float red[GK_CU_MM_BLOCK / GK_WARP_SIZE];

    const int64_t i0  = blockIdx.x;
    const int64_t c0  = (int64_t) blockIdx.y * NC;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }

    const uint8_t * a_row = (const uint8_t *) gk_cu_row(a, i0, a2, a3);
    const gk_cu_q8blk * aq23 = aq + i23 * d.ne[1] * n_grp;

    for (int64_t g = threadIdx.x; g < n_grp; g += blockDim.x) {
        int   codes[8];
        float scale[2], offset[2];
        gk_cu_wblk32<ATYPE>(a_row, g, codes, scale, offset);

#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                acc[j] += gk_cu_vecdot32_pre<ATYPE>(codes, scale, offset,
                                                    aq23[col * n_grp + g]);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = gk_cu_block_sum(acc[j], red);
        // The reduction reads `red` after its barrier; the next one writes it.
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                gk_cu_set(d, i0, col, i2, i3, acc[j]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The tensor-core path, for nvfp4.
//
// A pilot rather than a general kernel: one format, to find out what
// `mma.sync` is worth here before the same is done for the rest.
//
// nvfp4 is the awkward case to start from and that is deliberate - it is the
// format the diffusion models use. Its scale changes every 16 elements, and an
// integer mma accumulates in s32, so the accumulator has to be drained to
// float and rescaled every 16 of k. That is one `m16n8k16` (2048 multiply-
// accumulates in a single warp instruction) against roughly eight instructions
// of rescaling, where a format with a scale per 32 could use `m16n8k32` and
// halve the rescaling per unit of work. So this measures the *floor* of what
// tensor cores give, not the ceiling.
//
// What makes it fit at all: nvfp4's values are `sub_scale * e2m1[code]`, and
// every entry of the e2m1 codebook is a small integer, so the codebook lookup
// can be folded into the staging - once per weight row per 16 of k, reused
// across the whole column tile - and what reaches the tensor core is plain
// int8. There is no offset term, so unlike q4_1 or q4_K this needs nothing
// from the activation block but its codes and its scale.
// --------------------------------------------------------------------------

#if !defined(GK_USE_HIP) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define GK_CU_HAVE_MMA 1
#endif

// The block-scaled FP4 mma, which is not simply "Blackwell or later".
//
// It lives behind the *architecture-specific* target, `sm_120a` rather than
// `sm_120`, and nvcc marks that with `__CUDA_ARCH_FEAT_SM120_ALL`. Keying on
// `__CUDA_ARCH__ >= 1200` instead would be wrong in the one case that matters:
// this build also embeds `compute_120` PTX for forward compatibility, and that
// PTX must not contain an instruction the assembler will reject. Verified both
// ways - the instruction reaches the PTX for `compute_120a` and does not for
// `compute_120`.
//
// The host side of the gate cannot read this macro, since it is device-only.
// `gk_cuda_mm_mma_fp4_available` below settles that by asking the *build* what
// it contains rather than asking the device what it can do.
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define GK_CU_HAVE_MMA_FP4 1
#endif

// Whether this build has FP4 device code at all, as a host-visible answer.
//
// A host function cannot see `__CUDA_ARCH_FEAT_SM120_ALL` - it is defined only
// during a device pass - so this is a constant folded out of the *device*
// passes and read back. `gk_cu_k_fp4_probe` writes 1 from a pass that has the
// instruction and 0 from one that does not, so the answer is whatever the
// device actually running the kernel was compiled for. That is the honest
// question: a fat binary can contain sm_89 without the instruction and
// sm_120a with it, and the compute capability alone does not distinguish a
// 12.0 device whose image was built as `120` from one built as `120a`.
static __global__ void gk_cu_k_fp4_probe(int * out) {
#if defined(GK_CU_HAVE_MMA_FP4)
    *out = 1;
#else
    *out = 0;
#endif
}

// D += A*B, for a 16x8 tile of s32 over 16 of k, warp-wide.
//
// The fragment layouts are the ones PTX fixes for this shape: with
// `group = lane/4` and `tig = lane%4`, a thread holds A rows `group` and
// `group+8` at columns `4*tig..+3`, B column `group` at rows `4*tig..+3`, and
// D at rows `group`/`group+8`, columns `2*tig` and `2*tig+1`.
static __device__ __forceinline__ void gk_cu_mma_s8(int (&d)[4], const int (&a)[2], int b) {
#if defined(GK_CU_HAVE_MMA)
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    // No such instruction on this target. The host gate below is what keeps
    // this kernel from being launched; this branch exists so a build for an
    // older part still compiles.
    (void) a; (void) b; (void) d;
#endif
}

// D += A*B, for a 16x8 tile of s32 over *thirty-two* of k, warp-wide.
//
// The wider window costs nothing to reach from the narrow one: its fragments
// are exactly two of the k16 fragments side by side. With the same `group` and
// `tig`, a thread holds A rows `group`/`group+8` at columns `4*tig..+3` and at
// those columns again sixteen further along, and B column `group` at the
// matching pairs of rows. D is unchanged - still rows `group`/`group+8`,
// columns `2*tig` and `2*tig+1`.
//
// What it is for is the drain. An integer accumulator has to be emptied to
// float whenever the scale it would be multiplied by changes, and that is the
// dominant cost of a quantized tensor-core tile - so a format whose scale
// covers thirty-two elements should pay it once over thirty-two, not twice
// over sixteen. nvfp4 cannot: its sub-scale changes every sixteen. Every
// format in `gk_cu_has_dp4a` can.
static __device__ __forceinline__ void gk_cu_mma_s8_k32(int (&d)[4], const int (&a)[4],
                                                        const int (&b)[2]) {
#if defined(GK_CU_HAVE_MMA)
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) a; (void) b; (void) d;
#endif
}

// --------------------------------------------------------------------------
// D += A*B for a 16x8 tile over *sixty-four* of k, with both operands in fp4
// and the block scales applied by the hardware.
//
// This is a different machine from the two above, not a wider version of them.
// `mma.sync...s8` multiplies integers and hands back an integer that gk then
// has to scale itself, once per window, in CUDA-core arithmetic - which is
// what makes an nvfp4 tile expensive, because nvfp4's scale changes every
// sixteen elements and the drain therefore runs four times per sixty-four.
// This instruction takes the e2m1 codes and the ue4m3 scales as themselves,
// applies the scales internally, and accumulates in f32. There is no drain,
// nothing to convert, and nothing to rescale.
//
// The scale operands are the unusual part. `scale_vec::4X` means four scales
// per sixty-four elements - one per sixteen, exactly nvfp4's own geometry -
// packed one per byte into a single register. They are supplied per *warp*
// rather than per lane: with the `{byte-id, thread-id}` selectors both zero,
// the hardware reads `a_scale` from threads 0 and 1 of each quad and
// `b_scale` from thread 0, so a lane must hold
//
//   a_scale: the four scales of A row `lane/4 + (lane%2)*8`
//   b_scale: the four scales of B column `lane/4`
//
// Lanes the hardware does not read still have to hold *something* defined;
// the tile below simply computes the same expression in all four lanes of a
// quad, which costs nothing and avoids a branch.
//
// A and B are the ordinary m16n8 fragments with eight codes to a register
// rather than four bytes: `a[0]` and `a[1]` are rows `group` and `group+8` at
// k `8*tig..+7`, `a[2]` and `a[3]` the same rows thirty-two further along, and
// `b[0]`/`b[1]` column `group` at those same two k windows.
static __device__ __forceinline__ void gk_cu_mma_fp4(float (&d)[4], const int (&a)[4],
                                                     const int (&b)[2],
                                                     unsigned sa, unsigned sb) {
#if defined(GK_CU_HAVE_MMA_FP4)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col."
        "f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sa), "r"(sb));
#else
    (void) a; (void) b; (void) d; (void) sa; (void) sb;
#endif
}

// Whether the integer tile above can run on tensor cores on this device.
//
// Keyed on `GK_CU_HAVE_MMA` rather than on the compute capability alone,
// because that macro is also what decides whether `gk_cu_mma_s8_k32` compiles
// to an instruction or to nothing: on a target without it the kernel would
// still launch and would still write, but it would write zeros. Host and
// device have to agree, so they read the same macro. (During host compilation
// `__CUDA_ARCH__` is undefined, so this is exactly "not HIP"; the runtime
// check below is what covers a pre-Ampere part in a build that has the
// instruction.)
static __host__ __forceinline__ bool gk_cuda_mm_mma_q8_available(
        const struct gk_cuda_scratch * scratch) {
#if defined(GK_CU_HAVE_MMA)
    return scratch != NULL && scratch->cc >= 80;
#else
    (void) scratch;
    return false;
#endif
}

// Four consecutive codes of a packed nibble word, through the e2m1 codebook
// and into four int8 lanes. `shift` picks the low or high nibble of each byte.
// Four codes to four packed int8 values. `gk_cu_e2m1_value` is arithmetic
// rather than a codebook read, which is what makes this affordable at the rate
// the tile below calls it - thirty-two times per thread per k-step.
static __device__ __forceinline__ int gk_cu_e2m1_quad(int w, int shift) {
    int out = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int code = (w >> (8 * e + shift)) & 0xf;

        out |= (gk_cu_e2m1_value(code) & 0xff) << (8 * e);
    }
    return out;
}

// The block's four warps, as two by two rather than four by one.
//
// What that changes is how much shared memory the arithmetic reads per
// instruction, which is what this kernel turned out to be spending itself on.
// A warp that owns a single sixteen-row tile has to fetch a B fragment for
// every mma it issues: the A fragment is reused across the column tiles, but
// nothing reuses B. Measured against the f16 tile at the same shapes - which
// gives a warp two row tiles - nvfp4 was reading 3.5 shared words per mma
// where f16 read 1.5, and was 1.2x to 2.4x slower for it.
//
// Two row tiles per warp make each B fragment feed two instructions and each A
// fragment four.
//
// How wide the block's tile is, on the other hand, is what decides how many
// times each operand is read from memory rather than from shared: a TILE_M x
// TILE_N tile re-streams the weights `n_cols / TILE_N` times and the
// activations `n_rows / TILE_M` times, so doubling both halves both. At
// MiniMax-H3's 28672x8742x5376 those multipliers are 137 and 448 at 64x64 -
// 34 GB of requests against 140 MB of actual operands - which is what makes
// the tile, not the instruction, the thing worth changing. The integer tile
// for the k-quants learned this the expensive way (see `mma-q8` below, where
// the instruction was worth 16% and the width the other 1.5x), and this is the
// same lever on the format that pilot was written for.
//
// So the geometry is a template parameter with two instantiations. A wide tile
// buys its reuse by making blocks scarce - a quarter as many for the same
// output - and the shapes this kernel serves range from a MiniMax-H3
// projection, thousands of blocks at either width, down to a UNet's deepest
// level at 5120x1280 with 256 columns, which is eighty blocks narrow and
// twenty wide on a part with 170 multiprocessors. Measured, that shape is 1.9x
// *slower* wide. `gk_cu_mma_nvfp4_wide` below is where the choice is made.
#define GK_CU_MMA_WARPS_N 2   // columns of warps, both widths
#define GK_CU_MMA_WM      32  // rows a warp owns, both widths
#define GK_CU_MMA_WMT     (GK_CU_MMA_WM / 16)  // 2 mma row tiles
#define GK_CU_MMA_K       32  // one activation block; two mma windows of 16

// The narrow tile: 64x64 over 4 warps. The wide one: 128x128 over 8.
#define GK_CU_MMA_WARPS_M_NARROW 2
#define GK_CU_MMA_WN_NARROW      32
#define GK_CU_MMA_WARPS_M_WIDE   4
#define GK_CU_MMA_WN_WIDE        64

// A third width, for the FP4 tile only: sixteen warps and a 128x256 tile.
//
// The FP4 GEMM's time is A's re-read traffic. Every column tile reads the
// whole of A, so A crosses DRAM n/TILE_N times - 68 times for the qkv shape,
// 5.9 GB of it - and B does not, because the resident blocks share a column
// tile and its few hundred KB sit in L2. **Only TILE_N moves that number.**
//
// 256 columns is bought with WARPS_N rather than WN, because WN sets WNT and
// WNT sets the 64-register accumulator, which is already half of what this
// kernel is allowed. Four warp columns of 64 leave the accumulator alone.
// Sixteen warps of 512 threads at 128 registers is exactly the register file,
// so this variant runs one block per multiprocessor and keeps the same sixteen
// warps that two blocks of eight were giving.
//
// The 256x128 arrangement - the same sixteen warps stacked in M instead - was
// measured first and is *worse* (qkv 7.70 ms against 7.24). That is the
// asymmetry above showing up: doubling TILE_M only halves the B traffic that
// L2 was already absorbing.
#define GK_CU_MMA_WARPS_M_XWIDE  4
#define GK_CU_MMA_WARPS_N_XWIDE  4
#define GK_CU_MMA_WN_XWIDE       64

// 256 threads can hold two blocks per multiprocessor at <=128 registers; 512
// threads at the same register count fill it with one.
#define GK_CU_MMA_FP4_BLOCKS(warps_m) (GK_CU_MMA_THREADS(warps_m) <= 256 ? 2 : 1)

#define GK_CU_MMA_THREADS(warps_m) ((warps_m) * GK_CU_MMA_WARPS_N * GK_WARP_SIZE)
#define GK_CU_MMA_TILE_M_OF(warps_m) ((warps_m) * GK_CU_MMA_WM)
#define GK_CU_MMA_TILE_N_OF(wn)      (GK_CU_MMA_WARPS_N * (wn))

template <int WARPS_M, int WN>
static __global__ __launch_bounds__(GK_CU_MMA_THREADS(WARPS_M), 2)
void gk_cu_k_mul_mat_mma_nvfp4(gk_tview a, gk_tview_mut d,
                               const gk_cu_q8blk * aq, int64_t n_grp,
                               int64_t r2, int64_t r3) {
    constexpr int TILE_M = GK_CU_MMA_TILE_M_OF(WARPS_M);
    constexpr int TILE_N = GK_CU_MMA_TILE_N_OF(WN);
    constexpr int WNT    = WN / 8;   // mma column tiles a warp owns

    // Staging is one whole nvfp4 block of k - 64 elements - per round rather
    // than the 32 the mma windows walk, which halves the barriers and reads
    // each weight row's 36-byte block in one go. Every thread stages half a
    // row *and* half a column: the row half carries the e2m1 decode and the
    // column half is a plain copy, so splitting both across all threads keeps
    // the critical path balanced instead of making the row threads the wait.
    static_assert(TILE_M == TILE_N && 2 * TILE_M == GK_CU_MMA_THREADS(WARPS_M),
                  "staging assumes two threads per staged row and per column");

    // Staged as ints rather than bytes so that the fragment reads below are
    // plain aligned loads: word `kk*8 + h*4 + tig` is exactly the four codes a
    // lane wants for half `h` of 32-window `kk`.
    // Row strides of 16 and 4 words put every fragment lane of a `group` on
    // one of eight banks - a four-way conflict on the hottest loads in the
    // kernel - so each row carries one word of padding to spread the groups.
    // (An `ldmatrix` variant of the fragment reads - which needs stride 20
    // for its 16-byte row alignment - was measured 10% *slower* here: this
    // loop's indexed loads interleave with the drain arithmetic, and there
    // is no long serial phase for the bunched loads to hide behind, unlike
    // flash attention's, where the same change paid.)
    __shared__ int   As[TILE_M][17];
    __shared__ int   Bs[TILE_N][17];
    __shared__ float Ws[TILE_M][5];
    __shared__ float Ad[TILE_N][2];

    const int lane  = threadIdx.x % GK_WARP_SIZE;
    const int warp  = threadIdx.x / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int warp_m = warp / GK_CU_MMA_WARPS_N;
    const int warp_n = warp % GK_CU_MMA_WARPS_N;

    const int64_t m0  = (int64_t) blockIdx.x * TILE_M;
    const int64_t n0  = (int64_t) blockIdx.y * TILE_N;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_q8blk * aq23 = aq + i23 * n_cols * n_grp;

    float acc[GK_CU_MMA_WMT][WNT][4];
#pragma unroll
    for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[wt][ct][i] = 0.0f;
            }
        }
    }

    // Which half of a 64-wide k-round this thread stages, of both its row and
    // its column.
    const int st_i = (int) threadIdx.x / 2;
    const int st_h = (int) threadIdx.x % 2;

    // The weight stream, software-pipelined one round ahead: the raw words of
    // round g+1 are requested while round g computes, so the DRAM latency of
    // the larger operand is paid behind the mma work instead of on the
    // critical path between the barriers. A 64-element block and its stride
    // are both a whole number of words, so the five loads are plain word
    // loads; they stay raw here and are decoded only at the store into
    // shared. The activation side is not worth pipelining: it is the smaller
    // stream by 4x to 8x and mostly lands in L2 - measured, a cp.async
    // double buffer for it cost 5% rather than paying, between the extra
    // shared memory and a barrier that no longer covered the weight decode.
    const int64_t n_rnd = n_grp / 2;

    const uint8_t * a_row  = NULL;
    int             a_pw[4];
    int             a_sc   = 0;

    {
        const int64_t m = m0 + st_i;
        if (m < n_rows) {
            a_row = (const uint8_t *) gk_cu_row(a, m, a2, a3);
        }
    }

#define GK_CU_NVFP4_FETCH(gg)                                                  \
    do {                                                                       \
        if (a_row != NULL && (gg) < n_rnd) {                                   \
            const uint8_t * blk = a_row + (gg) * 36;                           \
            const int *     w   = (const int *) (blk + 4) + st_h * 4;          \
            a_sc    = *(const int *) blk;                                      \
            a_pw[0] = w[0];                                                    \
            a_pw[1] = w[1];                                                    \
            a_pw[2] = w[2];                                                    \
            a_pw[3] = w[3];                                                    \
        }                                                                      \
    } while (0)

    GK_CU_NVFP4_FETCH(0);

    for (int64_t g = 0; g < n_rnd; ++g) {
        {
            const int r    = st_i;
            const int sub0 = st_h * 2;

            if (a_row != NULL) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int w0 = a_pw[h * 2 + 0]; // sub-block elements 0..3, 8..11
                    const int w1 = a_pw[h * 2 + 1]; // and 4..7, 12..15

                    As[r][st_h * 8 + h * 4 + 0] = gk_cu_e2m1_quad(w0, 0);
                    As[r][st_h * 8 + h * 4 + 1] = gk_cu_e2m1_quad(w1, 0);
                    As[r][st_h * 8 + h * 4 + 2] = gk_cu_e2m1_quad(w0, 4);
                    As[r][st_h * 8 + h * 4 + 3] = gk_cu_e2m1_quad(w1, 4);

                    Ws[r][sub0 + h] = gk_cu_ue4m3((uint8_t) (a_sc >> (8 * (sub0 + h))));
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    As[r][st_h * 8 + i] = 0;
                }
                Ws[r][st_h * 2 + 0] = 0.0f;
                Ws[r][st_h * 2 + 1] = 0.0f;
            }
        }

        {
            const int     c = st_i;
            const int64_t n = n0 + c;

            if (n < n_cols) {
                const gk_cu_q8blk & ab = aq23[n * n_grp + g * 2 + st_h];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Bs[c][st_h * 8 + i] = ab.q[i];
                }
                Ad[c][st_h] = ab.d;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Bs[c][st_h * 8 + i] = 0;
                }
                Ad[c][st_h] = 0.0f;
            }
        }

        __syncthreads();

        GK_CU_NVFP4_FETCH(g + 1);

        // Two 32-wide windows per staged round, each two mma sub-windows -
        // one per sub-scale - drained together. The accumulator has to leave
        // s32 whenever the scale changes - the whole cost of this format on
        // tensor cores - but the two sub-windows' drains share every factor
        // except the sub-scale, so paying for them in one expression is five
        // float instructions per element instead of eight:
        //   acc += adv * (ws0*df0 + ws1*df1)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            // The activation scale belongs to the whole 32-element block, so
            // it is the same for both mma sub-windows - read once per window
            // rather than once per sub-window.
            float adv[WNT][2];
#pragma unroll
            for (int ct = 0; ct < WNT; ++ct) {
                const int c = warp_n * WN + ct * 8 + tig * 2;
                adv[ct][0] = Ad[c + 0][kk];
                adv[ct][1] = Ad[c + 1][kk];
            }

            int   af[2][GK_CU_MMA_WMT][2];
            float ws[2][GK_CU_MMA_WMT][2];

#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
                    const int r_lo = warp_m * GK_CU_MMA_WM + wt * 16 + group;

                    af[h][wt][0] = As[r_lo    ][kk * 8 + h * 4 + tig];
                    af[h][wt][1] = As[r_lo + 8][kk * 8 + h * 4 + tig];
                    ws[h][wt][0] = Ws[r_lo    ][kk * 2 + h];
                    ws[h][wt][1] = Ws[r_lo + 8][kk * 2 + h];
                }
            }

#pragma unroll
            for (int ct = 0; ct < WNT; ++ct) {
                // One B fragment per sub-window, both row tiles - which is
                // what giving a warp two of them buys.
                const int bf0 = Bs[warp_n * WN + ct * 8 + group][kk * 8 +     tig];
                const int bf1 = Bs[warp_n * WN + ct * 8 + group][kk * 8 + 4 + tig];

#pragma unroll
                for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
                    int df0[4] = { 0, 0, 0, 0 };
                    int df1[4] = { 0, 0, 0, 0 };

                    gk_cu_mma_s8(df0, af[0][wt], bf0);
                    gk_cu_mma_s8(df1, af[1][wt], bf1);

                    acc[wt][ct][0] += adv[ct][0] * (ws[0][wt][0] * (float) df0[0] + ws[1][wt][0] * (float) df1[0]);
                    acc[wt][ct][1] += adv[ct][1] * (ws[0][wt][0] * (float) df0[1] + ws[1][wt][0] * (float) df1[1]);
                    acc[wt][ct][2] += adv[ct][0] * (ws[0][wt][1] * (float) df0[2] + ws[1][wt][1] * (float) df1[2]);
                    acc[wt][ct][3] += adv[ct][1] * (ws[0][wt][1] * (float) df0[3] + ws[1][wt][1] * (float) df1[3]);
                }
            }
        }

        __syncthreads();
    }

#undef GK_CU_NVFP4_FETCH

#pragma unroll
    for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t m = m0 + warp_m * GK_CU_MMA_WM + wt * 16
                                + group + (i >= 2 ? 8 : 0);
                const int64_t n = n0 + warp_n * WN + ct * 8
                                + tig * 2 + (i & 1);

                if (m < n_rows && n < n_cols) {
                    gk_cu_set(d, m, n, i2, i3, acc[wt][ct][i]);
                }
            }
        }
    }
}

// Whether the FP4 tensor-core tile can run on the device this stream is on.
//
// Asked of the *binary*, by launching a kernel that reports which device pass
// it was compiled from, and cached per device. Two reasons it is not a compute
// capability test:
//
//   * a fat binary can hold sm_120a code with the instruction and sm_120 code
//     without it, and the capability is 12.0 either way;
//   * `__CUDA_ARCH_FEAT_SM120_ALL` is device-only, so the host has no other
//     way to find out.
//
// The cost is one launch and one 4-byte copy the first time a device is used.
// A failure of any kind answers "no", which is always safe: the integer tile
// computes the same thing.
static bool gk_cuda_mm_mma_fp4_available(const struct gk_cuda_scratch * s,
                                         gkStream_t stream) {
#if defined(GK_USE_HIP)
    (void) s; (void) stream;
    return false;
#else
    if (s == NULL || s->cc < 120) {
        return false;
    }

    static int         cache[GK_CUDA_MAX_DEVICES];
    static bool        known[GK_CUDA_MAX_DEVICES] = { false };
    static std::mutex  lock;

    int dev = 0;
    if (gkGetDevice(&dev) != gkSuccess || dev < 0 || dev >= GK_CUDA_MAX_DEVICES) {
        return false;
    }

    std::lock_guard<std::mutex> guard(lock);

    if (known[dev]) {
        return cache[dev] != 0;
    }

    known[dev] = true;
    cache[dev] = 0;

    int * flag = NULL;
    if (gkMalloc((void **) &flag, sizeof(int)) != gkSuccess) {
        return false;
    }

    int host = 0;
    gk_cu_k_fp4_probe<<<1, 1, 0, stream>>>(flag);

    if (gkGetLastError() == gkSuccess &&
        gkMemcpyAsync(&host, flag, sizeof(int), gkMemcpyDeviceToHost, stream) == gkSuccess &&
        gkStreamSynchronize(stream) == gkSuccess) {
        cache[dev] = host;
    }

    gkFree(flag);

    return cache[dev] != 0;
#endif
}

// An integer knob read from the environment, once. `def` when it is unset or
// unparseable - a typo should leave the default in place rather than select a
// third thing.
static int gk_cu_env_int(const char * name, int def) {
    const char * e = getenv(name);

    if (e == NULL || e[0] == '\0') {
        return def;
    }

    char *    end = NULL;
    const long v  = strtol(e, &end, 10);

    return (end != NULL && *end == '\0') ? (int) v : def;
}

// --------------------------------------------------------------------------
// The FP4 tensor-core tile.
//
// Geometry, staging and epilogue are the integer tile above; the inner loop is
// `gk_cu_mma_fp4`. Almost everything that makes this faster follows from the
// instruction covering sixty-four of k rather than sixteen:
//
//   * No drain. The integer tile spends roughly sixteen CUDA-core
//     instructions per tensor-core instruction emptying and rescaling an s32
//     accumulator, four times per sixty-four of k, because that is how often
//     an nvfp4 sub-scale changes. Here the scales are an operand.
//   * Half the staging. A row's sixty-four codes are eight words of packed
//     nibbles, where the integer tile stages eight words per *thirty-two*
//     elements - so the same shared memory and the same barrier count carry
//     twice the k.
//   * Two registers of scale per lane instead of four floats plus the
//     temporaries the drain needs.
//
// The one thing that is *not* free is the weight repack. gk stores an nvfp4
// sub-block as "low nibbles are elements 0..7, high nibbles are 8..15", and a
// fragment register wants eight consecutive elements. Those turn out to be the
// same set: element `8*tig..+7` of a sub-block is entirely low nibbles when
// `tig` is even and entirely high when it is odd, so a register is one
// nibble-gather over eight bytes and never a cross-byte shuffle. It is done
// once per weight row per k-step and used TILE_N times.
// --------------------------------------------------------------------------

// One A fragment - 16 rows by 64 fp4 codes - straight out of shared memory.
//
// `ldmatrix` rather than four indexed loads, and that is not a tuning choice
// either. The per-lane register layout an mma expects for its A operand is
// fixed by the hardware and is not the obvious one; `ldmatrix` is the
// instruction that produces it from a row-major tile, so using it means the
// layout never has to be written down, only the tile does. Four hand-rolled
// loads have to guess it, and a wrong guess is a kernel that computes
// something plausible-looking and wrong.
//
// `xs` is the lane's own row of the tile: row `lane % 16`, and the second half
// of the k range for the upper sixteen lanes. Rows are 8 ints, so both halves
// are 16-byte aligned, which the instruction requires.
static __device__ __forceinline__ void gk_cu_ldmatrix_x4(int (&d)[4], const int * xs) {
#if defined(GK_CU_HAVE_MMA)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "l"(xs));
#else
    (void) xs;
    d[0] = d[1] = d[2] = d[3] = 0;
#endif
}

#define GK_CU_MMA_FP4_K 64   // one nvfp4 block; one mma window

// How many nvfp4 groups the tile stages per round.
//
// One group is 64 elements of k, and every round costs two `__syncthreads()`.
// At k=5376 that was 84 rounds and 168 barriers to do 84 groups of work, with
// the global load latency exposed on each of them because nothing else was in
// flight. Staging several groups at once divides the barrier count by KSTEP and
// multiplies the mma work between barriers by it, which is what gives the
// scheduler something to hide the next round's loads behind.
//
// Shared cost is (TILE_M + TILE_N) * (8*KSTEP + PAD) * 4 bytes plus the scales.
// KSTEP 2 on the wide tile is ~20 KB, so two blocks per multiprocessor still
// fit in the 99 KB without the opt-in.
// k-step per tile width, in nvfp4 groups of 64.
//
// Nsight says why this matters: at a k-step of one group the kernel's largest
// single stall is `barrier`, at **16.0 cycles per issued instruction against
// ggml's 0.37**, because two `__syncthreads()` buy only 64 elements of k.
// ggml's NVFP4 path runs `MMQ_ITER_K_FP4 = 512` - eight groups a round - and
// its dominant stall is `math_pipe_throttle`, which is the tensor cores being
// the limit and is where this kernel wants to be.
//
// Eight groups only fit the 128x128 tile. Shared is
// (TILE_M + TILE_N) * (8*KSTEP + PAD) * 4 plus the scales, so 128x128 at eight
// is ~76 KB of the 99 available while 128x256 at eight would be ~114 KB. The
// wide tile therefore goes deep and the extra-wide one stays shallower.
#define GK_CU_MMA_FP4_KSTEP_WIDE  4
#define GK_CU_MMA_FP4_KSTEP_XWIDE 4

// Words of padding on each staged row, to keep rows off the same shared banks.
//
// A row of KSTEP groups is 8*KSTEP words, and both 8 and 16 divide the 32-bank
// width - so without padding every second (KSTEP=2) or fourth (KSTEP=1) row
// starts on the same bank and the ldmatrix reads collide. Four words moves the
// stride off a power of two while keeping it a multiple of 16 bytes, which
// `ldmatrix` requires of every address it is given.
#define GK_CU_MMA_FP4_PAD 4

// Sixteen warps per multiprocessor, however they are packaged.
//
// Asking for one block let the compiler spend up to 255 registers, and on
// sm_120a it settled on 130 - two over the 128 that a second block of 256
// threads needs to fit in the 65536-register file, so the kernel ran at 8
// warps of a possible 48. This kernel is not the flash attention one: its
// 64-register accumulator is the only large thing it holds, so the two
// registers were free.
//
// `GK_CU_MMA_FP4_BLOCKS` reads that constraint the other way round for the
// 512-thread tile: 512 threads at 128 registers *are* the register file, so
// that one asks for a single block and gets the same sixteen warps.
// WARPS_N is a template parameter here, unlike the other two mma kernels.
//
// The tile has to grow in N rather than in M, and the reason is which operand
// actually reaches DRAM. Every column tile reads the whole of A, and every row
// tile reads the whole of B, so nominally the traffic is
// |A|*(n/TILE_N) + |B|*(m/TILE_M) and the two look symmetric. They are not:
// `blockIdx.x` varies fastest, so the blocks resident at any moment share a
// column tile, B's few hundred KB of it stay in L2, and only A is genuinely
// streamed. So A's re-read count - and only that - is the traffic, and it is
// set by TILE_N alone.
//
// Growing TILE_N through WN would double WNT and with it the 64-register
// accumulator; growing it through WARPS_N does not, which is why this is a
// warp count and not a wider warp.
// Words of dynamic shared a configuration needs. Host and device both compute
// it from this, so the launch and the kernel cannot disagree about the layout.
#define GK_CU_FP4_SMEM_WORDS(tile_m, tile_n, ks)                    \
    (((tile_m) + (tile_n)) * (8 * (ks) + GK_CU_MMA_FP4_PAD)         \
     + (ks) * ((tile_m) + (tile_n)))

// Blocks per multiprocessor to ask the compiler for. A deep k-step puts the
// staging in shared memory rather than registers, and past 48 KB only one
// block fits anyway - so asking for two would cap registers at 128 for nothing.
#define GK_CU_FP4_BPSM(warps_m, warps_n, ks) \
    ((((warps_m) * (warps_n) * GK_WARP_SIZE) <= 256 && (ks) <= 2) ? 2 : 1)

template <int WARPS_M, int WN, int WARPS_N, int KSTEP>
static __global__
__launch_bounds__(WARPS_M * WARPS_N * GK_WARP_SIZE,
                  GK_CU_FP4_BPSM(WARPS_M, WARPS_N, KSTEP))
void gk_cu_k_mul_mat_mma_fp4(gk_tview a, gk_tview_mut d,
                             const gk_cu_fp4blk * aq, const float * col_scale,
                             int64_t n_grp, int64_t r2, int64_t r3) {
    constexpr int THREADS = WARPS_M * WARPS_N * GK_WARP_SIZE;
    constexpr int TILE_M  = GK_CU_MMA_TILE_M_OF(WARPS_M);
    constexpr int TILE_N  = WARPS_N * WN;
    constexpr int WNT     = WN / 8;

    // One thread per staged row or column, and beyond 256 threads there are
    // more threads than there are of either - the tail of the block simply
    // does not stage. Equalising that costs an extra index computation per
    // thread and buys nothing: staging is not what this kernel is short of.
    static_assert(TILE_M + TILE_N <= THREADS,
                  "staging assumes at most one thread per staged row and column");

    constexpr int KS  = KSTEP;
    constexpr int ROW = 8 * KS + GK_CU_MMA_FP4_PAD;   // words per staged row

    static_assert(GK_CU_MMA_FP4_PAD % 4 == 0,
                  "ldmatrix needs every address it is handed 16-byte aligned, "
                  "so the row stride must stay a multiple of four words");

    static_assert(ROW % 2 == 0,
                  "the B fragment pair is read as one 8-byte load at an even "
                  "word offset, so an odd row stride would misalign it on "
                  "every odd column");

    // Dynamic, not static, because a deep k-step needs more than the 48 KB a
    // block gets without asking. The caller raises the cap with
    // `gkFuncAttributeMaxDynamicSharedMemorySize` and passes the size, and
    // both sides size it with GK_CU_FP4_SMEM_WORDS so the layout cannot drift.
    //
    // 16-byte aligned because `ldmatrix` reads As in 16-byte pieces. Each
    // following block starts a multiple of ROW (a multiple of four) words in,
    // so all four stay aligned.
    extern __shared__ __align__(16) uint32_t gk_fp4_smem[];

    uint32_t * const As = gk_fp4_smem;                   // TILE_M rows of ROW
    uint32_t * const Bs = As + (size_t) TILE_M * ROW;    // TILE_N rows of ROW
    uint32_t * const Ws = Bs + (size_t) TILE_N * ROW;    // KS x TILE_M scales
    uint32_t * const Ad = Ws + (size_t) KS * TILE_M;     // KS x TILE_N scales

    const int lane  = threadIdx.x % GK_WARP_SIZE;
    const int warp  = threadIdx.x / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    const int64_t m0  = (int64_t) blockIdx.x * TILE_M;
    const int64_t n0  = (int64_t) blockIdx.y * TILE_N;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_fp4blk * aq23 = aq + i23 * n_cols * n_grp;
    const float *        cs23 = col_scale + i23 * n_cols;

    // The scale registers the hardware will read from this lane: A row
    // `group + (lane%2)*8`, B column `group`. Computed in every lane of a quad
    // rather than only in the two that are read - see `gk_cu_mma_fp4`.
    const int sc_row = group + (lane % 2) * 8;

    float acc[GK_CU_MMA_WMT][WNT][4];
#pragma unroll
    for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[wt][ct][i] = 0.0f;
            }
        }
    }

    for (int64_t g0 = 0; g0 < n_grp; g0 += KS) {
        if (threadIdx.x < TILE_M) {
            const int     r = (int) threadIdx.x;
            const int64_t m = m0 + r;

            // Hoisted: the row is the same for every group this round, and
            // only the 36-byte block offset moves.
            const uint8_t * row = m < n_rows
                                ? (const uint8_t *) gk_cu_row(a, m, a2, a3)
                                : NULL;

#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                const int64_t g = g0 + kk;

                if (row != NULL && g < n_grp) {
                    // 36 bytes a block: four ue4m3 scales, then sixty-four codes.
                    //
                    // Read as words, not as `gk_cu_int_b2`. That helper pairs
                    // two 16-bit loads because it cannot assume alignment, and
                    // here it costs **18 `LDG.E.U16` where 9 `LDG.E` would
                    // do** - which Nsight attributes as `lg_throttle`, 9.4 of
                    // this kernel's 53.7 stall cycles per issued instruction.
                    // A block is 36 bytes and no view can begin part-way
                    // through one, so a row is a multiple of 36 from a base
                    // the allocator gives at least 16-byte alignment: every
                    // block, and every word inside it, is 4-byte aligned.
                    const uint32_t * blk = (const uint32_t *) (row + g * 36);

                    // Verbatim. gk stores an nvfp4 sub-block as "low nibbles are
                    // elements 0..7, high nibbles are 8..15", which is not the
                    // order of k - but a matmul sums over k, so a permutation of
                    // it costs nothing as long as *both* operands carry the same
                    // one. The activation quantizer writes this same order for
                    // exactly that reason, and the two cancel. What must survive
                    // the permutation is the scale grouping, and it does: a
                    // sub-block is two whole words either way.
#pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        As[r * ROW + kk * 8 + w] = blk[1 + w];   // word 0 is the scales
                    }

                    Ws[kk * TILE_M + r] = blk[0];
                } else {
                    // Past the end of the matrix, or past the end of k on the
                    // last round. A zero scale with zero codes contributes
                    // nothing to the accumulator, which is what makes the tail
                    // need no separate path.
#pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        As[r * ROW + kk * 8 + w] = 0;
                    }
                    Ws[kk * TILE_M + r] = 0;
                }
            }
        } else if (threadIdx.x < TILE_M + TILE_N) {
            const int     c = (int) threadIdx.x - TILE_M;
            const int64_t n = n0 + c;

            const gk_cu_fp4blk * col = n < n_cols ? aq23 + n * n_grp : NULL;

#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                const int64_t g = g0 + kk;

                if (col != NULL && g < n_grp) {
                    const gk_cu_fp4blk & ab = col[g];

                    // Interleaved, not verbatim: word w goes to 2w for the
                    // low half and 2(w-4)+1 for the high half.
                    //
                    // A lane's two B fragment registers are words `tig` and
                    // `tig + 4`, which are four apart and so are two separate
                    // `LDS`. Stored this way they are adjacent and the pair is
                    // one 8-byte load, which halves the shared-memory
                    // instruction count of the whole B side - 16 loads per
                    // warp per group become 8, against the 16 `mma` they feed.
                    //
                    // This is a permutation of *storage* only. The read below
                    // undoes it exactly, so each fragment register receives the
                    // same word it always did and the mma cannot tell.
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        Bs[c * ROW + kk * 8 + 2 * w    ] = ab.q[w];
                        Bs[c * ROW + kk * 8 + 2 * w + 1] = ab.q[w + 4];
                    }
                    Ad[kk * TILE_N + c] = ab.sc;
                } else {
#pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        Bs[c * ROW + kk * 8 + w] = 0;
                    }
                    Ad[kk * TILE_N + c] = 0;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            int      af[GK_CU_MMA_WMT][4];
            unsigned as[GK_CU_MMA_WMT];

#pragma unroll
            for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
                const int r0 = warp_m * GK_CU_MMA_WM + wt * 16;

                gk_cu_ldmatrix_x4(
                    af[wt],
                    (const int *) &As[(r0 + (lane % 16)) * ROW + kk * 8 + (lane / 16) * 4]);

                as[wt] = Ws[kk * TILE_M + r0 + sc_row];
            }

#pragma unroll
            for (int ct = 0; ct < WNT; ++ct) {
                const int c = warp_n * WN + ct * 8 + group;

                // One 8-byte load where there were two 4-byte ones. See the
                // interleaved store above: `2*tig` holds what word `tig` held
                // and `2*tig + 1` what word `tig + 4` held, so the pair lands
                // in the fragment registers in the order the mma expects.
                const uint2 b2 = *(const uint2 *) &Bs[c * ROW + kk * 8 + 2 * tig];

                int bf[2];
                bf[0] = (int) b2.x;
                bf[1] = (int) b2.y;

                const unsigned bs = Ad[kk * TILE_N + c];

#pragma unroll
                for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
                    gk_cu_mma_fp4(acc[wt][ct], af[wt], bf, as[wt], bs);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int wt = 0; wt < GK_CU_MMA_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t m = m0 + warp_m * GK_CU_MMA_WM + wt * 16
                                + group + (i >= 2 ? 8 : 0);
                const int64_t n = n0 + warp_n * WN + ct * 8
                                + tig * 2 + (i & 1);

                if (m < n_rows && n < n_cols) {
                    // The column's second-level scale, undone here. It is a
                    // constant over the whole k reduction, so it factors out of
                    // the sum entirely and costs one multiply per output rather
                    // than anything in the inner loop - which is the reason the
                    // scale is per column and not per block.
                    gk_cu_set(d, m, n, i2, i3, acc[wt][ct][i] * cs23[n]);
                }
            }
        }
    }
}

// Whether a shape should take the wide nvfp4 tile.
//
// The wide tile halves both re-read multipliers, so it is what you want
// whenever it can be kept fed - and it is paid for in blocks, four of the
// narrow tile's for every one of its own. The test is therefore not about the
// shape in the abstract but about this device: a grid that no longer covers
// the multiprocessors has multiprocessors standing idle, and no amount of
// reuse inside a block makes up for that.
//
// One block per multiprocessor is the floor rather than a target. Above it the
// wide tile wins on every diffusion shape measured; below it the narrow one is
// strictly better, because the device is short of work rather than short of
// bandwidth.
//
// `GK_MM_NVFP4_TILE=64` or `=128` overrides the heuristic. That exists for two
// reasons and both are worth more than the knob: it is how the two widths are
// A/B'd at a fixed shape, and it is how the wide tile gets *tested* at all -
// every nvfp4 case in `test_cuda.c` is far too small to select it, so without
// a way to force the choice the widening would ship covered by nothing.
static __host__ __forceinline__ bool gk_cu_mma_nvfp4_wide(const struct gk_tensor * dst,
                                                          const struct gk_cuda_scratch * s) {
    static const int forced = gk_cu_env_int("GK_MM_NVFP4_TILE", 0);

    // 256 is the FP4 tile's third width and this kernel has no such variant;
    // give it the widest it does have rather than silently dropping to the
    // narrow one, which would make an A/B at =256 compare two things at once.
    if (forced != 0) {
        return forced != 64;
    }

    const int64_t tm = GK_CU_MMA_TILE_M_OF(GK_CU_MMA_WARPS_M_WIDE);
    const int64_t tn = GK_CU_MMA_TILE_N_OF(GK_CU_MMA_WN_WIDE);

    const int64_t blocks = ((dst->ne[0] + tm - 1) / tm)
                         * ((dst->ne[1] + tn - 1) / tn)
                         * dst->ne[2] * dst->ne[3];

    return s != NULL && s->n_sm > 0 && blocks >= (int64_t) s->n_sm;
}

// Launch one FP4 tile configuration: size its dynamic shared memory, raise the
// per-kernel cap the first time if it wants more than the 48 KB a block gets
// for free, and launch.
//
// The cap is a property of the kernel *on a device* rather than of the launch,
// so it is raised once per instantiation per device - `raised` is a
// function-local static array, and each macro expansion is its own scope and so
// its own flags. A single flag would leave every card but the first at the
// 48 KB default, where the launch fails with "invalid argument".
#define GK_CU_FP4_LAUNCH(warps_m, wn, warps_n, kstep)                              \
    do {                                                                           \
        constexpr int  tm_    = GK_CU_MMA_TILE_M_OF(warps_m);                      \
        constexpr int  tn_    = (warps_n) * (wn);                                  \
        constexpr int  thr_   = (warps_m) * (warps_n) * GK_WARP_SIZE;              \
        constexpr size_t smem_ =                                                   \
            (size_t) GK_CU_FP4_SMEM_WORDS(tm_, tn_, kstep) * sizeof(uint32_t);     \
        if (smem_ > 48u * 1024u) {                                                 \
            static bool raised[GK_CUDA_MAX_DEVICES] = { false };                   \
            int dev_ = 0;                                                          \
            if (gkGetDevice(&dev_) == gkSuccess && dev_ >= 0 &&                    \
                dev_ < GK_CUDA_MAX_DEVICES && !raised[dev_]) {                     \
                raised[dev_] = true;                                               \
                GK_CUDA_CHECK(gkFuncSetAttribute(                                  \
                    (const void *) gk_cu_k_mul_mat_mma_fp4<warps_m, wn,            \
                                                           warps_n, kstep>,        \
                    gkFuncAttributeMaxDynamicSharedMemorySize, (int) smem_));      \
            }                                                                      \
        }                                                                          \
        gk_cu_k_mul_mat_mma_fp4<warps_m, wn, warps_n, kstep>                       \
            <<<mgrid, thr_, smem_, stream>>>(                                      \
                gk_cu_view(src0), gk_cu_view_mut(dst), aq, acs, n_grp, r2, r3);    \
    } while (0)

// Whether a shape should take the 256x128 FP4 tile instead of the 128x128 one.
//
// Same argument as the width above, one step further along, and for the same
// reason: this kernel's time is the operands' re-read traffic, so the tile
// wants to be as large as the register file allows and no larger than the grid
// can afford. Doubling M halves how many times A is read and doubles how many
// times the grid must cover the multiprocessors to stay busy, so the test is
// again a floor on blocks rather than anything about the shape.
//
// The floor is two blocks per multiprocessor rather than one. At 512 threads
// this variant is a single block of sixteen warps, so a grid that only just
// covers the device leaves no second block to overlap the tail with.
//
// `GK_MM_NVFP4_TILE=256` forces it, and as with the other widths that is how
// it gets tested at all - `test_cuda.c`'s shapes are far too small to select it.
static __host__ __forceinline__ bool gk_cu_mma_nvfp4_xwide(const struct gk_tensor * dst,
                                                           const struct gk_cuda_scratch * s) {
    static const int forced = gk_cu_env_int("GK_MM_NVFP4_TILE", 0);

    if (forced != 0) {
        return forced == 256;
    }

    const int64_t tm = GK_CU_MMA_TILE_M_OF(GK_CU_MMA_WARPS_M_XWIDE);
    const int64_t tn = (int64_t) GK_CU_MMA_WARPS_N_XWIDE * GK_CU_MMA_WN_XWIDE;

    const int64_t blocks = ((dst->ne[0] + tm - 1) / tm)
                         * ((dst->ne[1] + tn - 1) / tn)
                         * dst->ne[2] * dst->ne[3];

    return s != NULL && s->n_sm > 0 && blocks >= 2 * (int64_t) s->n_sm;
}

// --------------------------------------------------------------------------
// The tensor-core path, for f16 activations.
//
// The nvfp4 pilot above answered what `mma.sync` is worth; this is the same
// instruction where the shapes that dominate a diffusion graph actually live.
// Every convolution in a UNet or a VAE lowers to `im2col` plus a matmul whose
// src0 is that f16 im2col buffer - so this one kernel is most of the time in
// both, and until it existed all of it ran on the float tile below at roughly
// a tenth of what the part can do.
//
// Three things make this simpler than the nvfp4 case:
//
//   * The operands are already f16, so nothing is decoded on the way to the
//     fragment and nothing is rescaled inside the k loop. The accumulator is
//     f32 and stays in the tensor core across the whole reduction, which is
//     what the integer path could not do.
//
//   * gk's matmul is `dst[m,n] = sum_k a[k,m] * b[k,n]` with both operands
//     k-major, and `mma.row.col` wants exactly that: A row-major with k along
//     the row, B column-major with k along the column. So the fragments are
//     the operands' own layout, not a transpose of it.
//
//   * src1 is read through the runtime accessor rather than a template. It is
//     touched once per tile and used TILE_M times, so what it costs is a
//     rounding error on the total, and not templating it keeps this to two
//     instantiations instead of two per weight format.
//
// The tile is sized by memory, not by arithmetic. At these shapes the GEMM is
// bandwidth-bound - src0 is an im2col buffer of tens of megabytes and gets
// re-read once per column tile - so what sets the ceiling is TILE_N, and 128
// is where a 512-channel VAE convolution stops being limited by DRAM. TILE_M
// only picks how many blocks there are, and drops to 64 when there are not
// enough rows to fill 128.
// --------------------------------------------------------------------------

// D += A*B for a 16x8 tile of f32 over 16 of k, warp-wide, f16 operands. The
// fragment geometry is the s8 instruction's at half the k per register: with
// `group = lane/4` and `tig = lane%4`, a lane holds A rows `group` and
// `group+8` at columns `2*tig..+1` and `2*tig+8..+9`, B column `group` at
// those same rows, and D at rows `group`/`group+8`, columns `2*tig` and
// `2*tig+1` - the layout the epilogue below and the nvfp4 kernel's share.
// The same instruction accumulating in half rather than float, and on a
// consumer part that is not a rounding decision - it is the whole throughput.
// GeForce Ada runs f16 mma with an f32 accumulator at half the rate of the
// same instruction with an f16 one (24 against 48 TFLOP/s on this card), which
// is why cuBLAS is 2.2-2.8x faster than this kernel at these shapes and 2.2x
// *slower* when `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` takes the same choice away
// from it. The D fragment is two registers rather than four, packed as
// (row, col) pairs in the same lane order.
//
// It is not used alone: the caller sums a bounded run of k in half and then
// promotes into a float accumulator, so the error grows with the run and not
// with k. See GK_CU_MMA_F16_ACC_CHUNK.
static __device__ __forceinline__ void gk_cu_mma_f16_h(int (&d)[2], const int (&a)[4],
                                                       const int (&b)[2]) {
#if defined(GK_CU_HAVE_MMA)
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) a; (void) b; (void) d;
#endif
}

static __device__ __forceinline__ void gk_cu_mma_f16(float (&d)[4], const int (&a)[4],
                                                     const int (&b)[2]) {
#if defined(GK_CU_HAVE_MMA)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    // No such instruction on this target; the host gate keeps the kernel from
    // being launched. This branch is here so the build still compiles.
    (void) a; (void) b; (void) d;
#endif
}

#define GK_CU_MMA_F16_K   32   // k staged per pass: two mma windows of sixteen
#define GK_CU_MMA_F16_WM  32   // rows a warp owns: two mma row tiles
#define GK_CU_MMA_F16_WN  32   // columns a warp owns: four mma column tiles
#define GK_CU_MMA_F16_WNT (GK_CU_MMA_F16_WN / 8)
#define GK_CU_MMA_F16_WARPS_N 4
#define GK_CU_MMA_F16_TILE_N  (GK_CU_MMA_F16_WARPS_N * GK_CU_MMA_F16_WN)

// The staged rows are padded past the k they hold. A fragment read has the
// eight lanes of a quadrant on eight different rows at the same k, so an
// unpadded row of 32 halves - 8 banks apart - would put four of those eight on
// the banks of another four. At a stride of 40 halves the eight land on eight
// disjoint quads and a warp's fragment read is conflict-free.
#define GK_CU_MMA_F16_SK  (GK_CU_MMA_F16_K + 8)

// Columns below which the tile is mostly padding and the paths below, which
// split k across a block instead of tiling n, keep the device busier.
#define GK_CU_MMA_F16_MIN_N 32

// The shortest piece of k worth giving a block of its own. Below this the
// block spends more of itself on its prologue and its share of the combine
// pass than on the reduction it was split off to do.
#define GK_CU_MMA_F16_SPLIT_MIN_K 1024

// Warps a shape has to put in flight before splitting k stops paying: one
// wide-tile block's worth per multiprocessor.
#define GK_CU_MMA_F16_SPLIT_WARPS 8

// Staged k-rounds accumulated in half before the running total is promoted to
// float, when the half accumulator is in use.
//
// Half has an eleven-bit mantissa, so a run of n products drifts by roughly
// sqrt(n) * 2^-11: 4.5% over a k of 8640 done entirely in half, which is what
// cuBLAS does here by default, against 0.8% over a run of 256 - and the runs
// themselves are then summed in float, so the total is better than either.
// Eight rounds of 32 is that 256, and it costs one conversion and one add per
// accumulator register per 128 mma, which does not show up in the measurement.
#define GK_CU_MMA_F16_ACC_CHUNK 8

// Eight elements of a float-typed operand, widened to half and packed into one
// 16-byte word - the unit the tile is staged in.
//
// Staging one element per instruction is what the float tile does, and there
// it is affordable because each staged element feeds 16 multiply-accumulates.
// Here it feeds 128, so an instruction spent per element is an instruction
// spent against a tensor-core instruction rather than against sixteen FFMAs,
// and it dominates. Whether the eight can be read as one word is decided on
// the host, once per launch, and passed in.
static __device__ __forceinline__ int4 gk_cu_pack8_half(const char * p, int type) {
    if (type == GKT_F16) {
        return *(const int4 *) p;
    }

    __align__(16) __half h[8];

    if (type == GKT_F32) {
        const float4 lo = *(const float4 *) p;
        const float4 hi = *(const float4 *) (p + 16);

        h[0] = __float2half(lo.x); h[1] = __float2half(lo.y);
        h[2] = __float2half(lo.z); h[3] = __float2half(lo.w);
        h[4] = __float2half(hi.x); h[5] = __float2half(hi.y);
        h[6] = __float2half(hi.z); h[7] = __float2half(hi.w);
    } else {
        const int4       w = *(const int4 *) p;
        const uint16_t * u = (const uint16_t *) &w;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            h[i] = __float2half(gk_cu_bf2f(u[i]));
        }
    }

    return *(const int4 *) h;
}

// Whether an operand can be staged eight at a time: contiguous along k, every
// row start 16-byte aligned, and a k that a run of eight either fits inside or
// starts past the end of, so the guard stays a comparison rather than a
// per-element mask.
static __host__ __forceinline__ bool gk_cuda_mma_f16_vec(const struct gk_tensor * t, int64_t k_len) {
    if ((size_t) t->nb[0] != gk_type_size(t->type) || k_len % 8 != 0) {
        return false;
    }
    const uintptr_t bits = (uintptr_t) t->data | (uintptr_t) t->nb[1]
                         | (uintptr_t) t->nb[2] | (uintptr_t) t->nb[3];
    return bits % 16 == 0;
}

// Whether this node may accumulate in half. `GK_MM_F16_ACC=1` to allow it;
// off unless asked.
//
// ggml takes the opposite default - `compute_type = GGML_TYPE_F16` whenever
// src0 is f16 and the device has fast half arithmetic - and on a consumer part
// that is the difference between half and full tensor rate, which is the whole
// of what looked like a 1.8-2.1x tiling deficit in this kernel. Measured here
// it is worth 1.35x on the convolutions and ~3% on a generation, and it costs
// about 1% relative error per matmul against the float accumulator's 1e-7:
// one denoising step and a VAE decode come out at 39.6 dB against the same run
// with the wide accumulator, where a pure rounding-order change is 60+.
//
// Three per cent is not worth that here, and gk is one library behind a
// language-model server as well as a diffuser, so the default cannot be set by
// what suits a UNet. A caller that wants ggml's arithmetic asks for it with
// the variable; a caller that needs the wide accumulator regardless says so
// per node with GK_PREC_F32, exactly as it would to ggml, and
// stable-diffusion.cpp already does on every attention score matrix.
static __host__ bool gk_cuda_mm_acc16(const struct gk_tensor * dst) {
    static int allowed = -1;

    if (allowed < 0) {
        const char * e = getenv("GK_MM_F16_ACC");
        allowed = e != NULL && e[0] != '0';
    }

    return allowed != 0 && (enum gk_prec) gk_get_op_params_i32(dst, 0) != GK_PREC_F32;
}

template <int WARPS_M, bool ACC16>
static __global__ __launch_bounds__(WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE, 1)
void gk_cu_k_mul_mat_mma_f16(gk_tview a, gk_tview b, gk_tview_mut d,
                             int64_t k_len, int64_t r2, int64_t r3,
                             bool a_vec, bool b_vec,
                             float * part, int64_t k_split, int64_t n_23) {
    const int TILE_M = WARPS_M * GK_CU_MMA_F16_WM;

    // Sixteen-byte aligned, and the padded row stride is a multiple of sixteen
    // too, because `ldmatrix` requires that of every address it is handed.
    // (The padding was chosen for the four-byte reads this loop used to do; it
    // happens to serve the wider ones as well - at 80 bytes a row the eight
    // rows of a matrix land on eight distinct sixteen-byte phases of the 128
    // byte bank row, so a fragment read is still conflict-free.)
    static_assert(GK_CU_MMA_F16_SK % 8 == 0,
                  "ldmatrix needs 16-byte aligned rows; the padded stride must be "
                  "a whole number of 8-half words");

    __shared__ __align__(16) __half As[2][WARPS_M * GK_CU_MMA_F16_WM][GK_CU_MMA_F16_SK];
    __shared__ __align__(16) __half Bs[2][GK_CU_MMA_F16_TILE_N]     [GK_CU_MMA_F16_SK];

    const int tid    = (int) threadIdx.x;
    const int lane   = tid % GK_WARP_SIZE;
    const int warp   = tid / GK_WARP_SIZE;
    const int warp_m = warp / GK_CU_MMA_F16_WARPS_N;
    const int warp_n = warp % GK_CU_MMA_F16_WARPS_N;
    const int group  = lane / 4;
    const int tig    = lane % 4;

    const int n_threads = WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE;

    // x is the *column* tile and y the row tile, which is the opposite of the
    // obvious assignment and is deliberate. Blocks are dispatched x fastest, so
    // whichever axis x carries is the one whose blocks are co-resident - and
    // the blocks that should be co-resident are the ones that share an operand.
    // Here every block covering a row tile reads the same TILE_M rows of A,
    // which for a UNet's convolutions is the big operand: 4096x8640 halves, 71
    // MB, against a 5 MB weight. With x on the row axis those blocks are
    // separated by the whole grid and A is re-read from DRAM once per column
    // tile; with x on the column axis they run together and the re-reads come
    // out of L2.
    const int64_t m0  = (int64_t) blockIdx.y * TILE_M;
    const int64_t n0  = (int64_t) blockIdx.x * GK_CU_MMA_F16_TILE_N;

    // z carries the slice and, when k is split, which piece of k this block
    // reduces. A split block owns the same output patch as its siblings and
    // differs only in the range it sums, so it cannot write the destination -
    // it writes its own plane of `part` and the pass below adds them up.
    const int64_t i23   = blockIdx.z % n_23;
    const int64_t split = blockIdx.z / n_23;

    const int64_t k_beg = split * k_split;
    const int64_t k_end = part != NULL && k_beg + k_split < k_len ? k_beg + k_split : k_len;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    float acc[2][GK_CU_MMA_F16_WNT][4];
#pragma unroll
    for (int wt = 0; wt < 2; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[wt][ct][i] = 0.0f;
            }
        }
    }

    // The half accumulator, when it is in use: the same tile of results in the
    // instruction's packed form, emptied into `acc` every GK_CU_MMA_F16_ACC_CHUNK
    // rounds. Declared unconditionally because a zero-length array is not a
    // thing; the compiler removes it when ACC16 is false and nothing reads it.
    int acch[2][GK_CU_MMA_F16_WNT][2];

#define GK_CU_MMA_F16_ACC_CLEAR()                                              \
    _Pragma("unroll")                                                          \
    for (int wt = 0; wt < 2; ++wt) {                                           \
        _Pragma("unroll")                                                      \
        for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ++ct) {                       \
            acch[wt][ct][0] = 0;                                               \
            acch[wt][ct][1] = 0;                                               \
        }                                                                      \
    }

    // Two halves per register, in the same (row, column) order the float
    // fragment uses, so the promotion is a widen and an add per element.
#define GK_CU_MMA_F16_ACC_DRAIN()                                              \
    _Pragma("unroll")                                                          \
    for (int wt = 0; wt < 2; ++wt) {                                           \
        _Pragma("unroll")                                                      \
        for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ++ct) {                       \
            const __half2 lo = *(const __half2 *) &acch[wt][ct][0];            \
            const __half2 hi = *(const __half2 *) &acch[wt][ct][1];            \
            acc[wt][ct][0] += __low2float(lo);                                 \
            acc[wt][ct][1] += __high2float(lo);                                \
            acc[wt][ct][2] += __low2float(hi);                                 \
            acc[wt][ct][3] += __high2float(hi);                                \
        }                                                                      \
    }                                                                          \
    GK_CU_MMA_F16_ACC_CLEAR()

    if (ACC16) {
        GK_CU_MMA_F16_ACC_CLEAR();
    }

    // A thread's share of one k-slice, held in registers between being read
    // from memory and being written to the tile. Both are known at compile
    // time: the slice is a fixed size and so is the block.
    int4 pa[TILE_M * (GK_CU_MMA_F16_K / 8) / (WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE)];
    int4 pb[GK_CU_MMA_F16_TILE_N * (GK_CU_MMA_F16_K / 8) / (WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE)];

    const int n_runs_a = (int) (sizeof(pa) / sizeof(pa[0]));
    const int n_runs_b = (int) (sizeof(pb) / sizeof(pb[0]));

    // Reading a k-slice into those registers. A run of eight per thread:
    // consecutive threads take consecutive k of one row, which is the
    // direction both operands are contiguous in, so a run is one 16-byte
    // load. The scalar arms are what an operand that is permuted, misaligned
    // or ragged in k falls to - the same answer, an instruction at a time.
#define GK_CU_MMA_F16_LOAD(k0)                                                       \
    do {                                                                             \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_a; ++i) {                                         \
            const int     e  = tid + i * n_threads;                                  \
            const int     r  = e / (GK_CU_MMA_F16_K / 8);                            \
            const int     kr = e % (GK_CU_MMA_F16_K / 8);                            \
            const int64_t m  = m0 + r;                                               \
            const int64_t k  = (k0) + kr * 8;                                        \
                                                                                     \
            pa[i] = make_int4(0, 0, 0, 0);                                           \
                                                                                     \
            if (m < n_rows) {                                                        \
                const char * row = gk_cu_row(a, m, a2, a3);                          \
                if (a_vec && k + 8 <= k_end) {                                       \
                    pa[i] = *(const int4 *) (row + k * a.nb[0]);                     \
                } else {                                                             \
                    __align__(16) __half h[8];                                       \
                    _Pragma("unroll")                                                \
                    for (int j = 0; j < 8; ++j) {                                    \
                        h[j] = (k + j < k_end)                                       \
                             ? *(const __half *) (row + (k + j) * a.nb[0])           \
                             : __float2half(0.0f);                                   \
                    }                                                                \
                    pa[i] = *(const int4 *) h;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
                                                                                     \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_b; ++i) {                                         \
            const int     e  = tid + i * n_threads;                                  \
            const int     c  = e / (GK_CU_MMA_F16_K / 8);                            \
            const int     kr = e % (GK_CU_MMA_F16_K / 8);                            \
            const int64_t n  = n0 + c;                                               \
            const int64_t k  = (k0) + kr * 8;                                        \
                                                                                     \
            pb[i] = make_int4(0, 0, 0, 0);                                           \
                                                                                     \
            if (n < n_cols) {                                                        \
                if (b_vec && k + 8 <= k_end) {                                       \
                    pb[i] = gk_cu_pack8_half(gk_cu_row(b, n, i2, i3) + k * b.nb[0],  \
                                             b.type);                                \
                } else {                                                             \
                    __align__(16) __half h[8];                                       \
                    _Pragma("unroll")                                                \
                    for (int j = 0; j < 8; ++j) {                                    \
                        h[j] = __float2half((k + j < k_end)                          \
                                            ? gk_cu_get(b, k + j, n, i2, i3)         \
                                            : 0.0f);                                 \
                    }                                                                \
                    pb[i] = *(const int4 *) h;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
    } while (0)

#define GK_CU_MMA_F16_STORE(buf)                                                     \
    do {                                                                             \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_a; ++i) {                                         \
            const int e = tid + i * n_threads;                                       \
            *(int4 *) &As[buf][e / (GK_CU_MMA_F16_K / 8)]                            \
                            [(e % (GK_CU_MMA_F16_K / 8)) * 8] = pa[i];               \
        }                                                                            \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_b; ++i) {                                         \
            const int e = tid + i * n_threads;                                       \
            *(int4 *) &Bs[buf][e / (GK_CU_MMA_F16_K / 8)]                            \
                            [(e % (GK_CU_MMA_F16_K / 8)) * 8] = pb[i];               \
        }                                                                            \
    } while (0)

    // The tile is double-buffered and the next slice is read while the current
    // one is being multiplied. That ordering is the point: a global read costs
    // several hundred cycles and a slice's worth of mma covers it, where a
    // kernel that loads and then waits at a barrier pays the two in series.
    // What it costs is one more copy of the tile in shared memory and the
    // registers to hold a slice in flight.
    GK_CU_MMA_F16_LOAD(k_beg);
    GK_CU_MMA_F16_STORE(0);
    __syncthreads();

    int buf   = 0;
    int chunk = 0;

    for (int64_t k0 = k_beg; k0 < k_end; k0 += GK_CU_MMA_F16_K) {
        const bool more = k0 + GK_CU_MMA_F16_K < k_end;

        if (more) {
            GK_CU_MMA_F16_LOAD(k0 + GK_CU_MMA_F16_K);
        }

        // Two mma windows over the staged k. The A fragments are read once
        // and used for every column tile - that reuse is the whole point of
        // giving a warp 64 columns rather than 8.
        //
        // Both operands come out of shared through `ldmatrix`, which is worth
        // spelling out because the hand-indexed form it replaced was correct:
        // an A fragment is four 4-byte loads and a B fragment two, so a k-step
        // of 32 was 48 shared-load instructions against 32 mma. At roughly a
        // cycle each that is the load/store pipe issuing as much as the tensor
        // pipe, on a kernel that is register-capped to eight warps per
        // multiprocessor and has nothing else to hide it behind. `ldmatrix`
        // fetches a whole 16x16 tile per instruction and brings the same 48
        // down to 12.
        //
        // The B tiles are taken two at a time for the same reason: one x4 over
        // sixteen columns is two column tiles' worth of fragments, with the
        // even tile in registers 0 and 2 and the odd one in 1 and 3.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kk = h * 16;

            int af[2][4];
#pragma unroll
            for (int wt = 0; wt < 2; ++wt) {
                const int r0 = warp_m * GK_CU_MMA_F16_WM + wt * 16;

                gk_cu_ldmatrix_x4(
                    af[wt], (const int *) &As[buf][r0 + (lane % 16)][kk + (lane / 16) * 8]);
            }

#pragma unroll
            for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ct += 2) {
                const int c0 = warp_n * GK_CU_MMA_F16_WN + ct * 8;

                int bq[4];
                gk_cu_ldmatrix_x4(
                    bq, (const int *) &Bs[buf][c0 + (lane % 16)][kk + (lane / 16) * 8]);

                const int bf0[2] = { bq[0], bq[2] };
                const int bf1[2] = { bq[1], bq[3] };

#pragma unroll
                for (int wt = 0; wt < 2; ++wt) {
                    if (ACC16) {
                        gk_cu_mma_f16_h(acch[wt][ct],     af[wt], bf0);
                        gk_cu_mma_f16_h(acch[wt][ct + 1], af[wt], bf1);
                    } else {
                        gk_cu_mma_f16(acc[wt][ct],     af[wt], bf0);
                        gk_cu_mma_f16(acc[wt][ct + 1], af[wt], bf1);
                    }
                }
            }
        }

        // The half accumulator only holds a bounded run of k; empty it into
        // the float one before the drift can grow past what the chunk size
        // was chosen for.
        if (ACC16) {
            if (++chunk == GK_CU_MMA_F16_ACC_CHUNK) {
                GK_CU_MMA_F16_ACC_DRAIN();
                chunk = 0;
            }
        }

        // Into the other half, which nothing is reading, so the only barrier
        // in the loop is the one that publishes it.
        if (more) {
            GK_CU_MMA_F16_STORE(1 - buf);
        }

        __syncthreads();
        buf = 1 - buf;
    }

    if (ACC16 && chunk != 0) {
        GK_CU_MMA_F16_ACC_DRAIN();
    }

#undef GK_CU_MMA_F16_ACC_DRAIN
#undef GK_CU_MMA_F16_ACC_CLEAR
#undef GK_CU_MMA_F16_LOAD
#undef GK_CU_MMA_F16_STORE

#pragma unroll
    for (int wt = 0; wt < 2; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t m = m0 + warp_m * GK_CU_MMA_F16_WM + wt * 16
                                + group + (i >= 2 ? 8 : 0);
                const int64_t n = n0 + warp_n * GK_CU_MMA_F16_WN + ct * 8
                                + tig * 2 + (i & 1);

                if (m < n_rows && n < n_cols) {
                    if (part != NULL) {
                        part[((split * n_23 + i23) * n_cols + n) * n_rows + m] = acc[wt][ct][i];
                    } else {
                        gk_cu_set(d, m, n, i2, i3, acc[wt][ct][i]);
                    }
                }
            }
        }
    }
}

// Adding the split blocks' planes back together. Summed in slice order rather
// than by whichever block finished first, so a split matmul returns the same
// bits every run - which an atomic accumulation into the destination, the
// other way to write this, would not.
static __global__ void gk_cu_k_mma_f16_combine(const float * part, gk_tview_mut d,
                                               int64_t n_splits, int64_t n_23,
                                               int64_t n_rows, int64_t n_cols) {
    const int64_t n_out = n_rows * n_cols * n_23;

    for (int64_t e = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
         e < n_out; e += (int64_t) gridDim.x * blockDim.x) {
        const int64_t m   = e % n_rows;
        const int64_t n   = (e / n_rows) % n_cols;
        const int64_t i23 = e / (n_rows * n_cols);

        float sum = 0.0f;
        for (int64_t s = 0; s < n_splits; ++s) {
            sum += part[((s * n_23 + i23) * n_cols + n) * n_rows + m];
        }

        gk_cu_set(d, m, n, i23 % d.ne[2], i23 / d.ne[2], sum);
    }
}

// Whether this matmul goes to the tensor core. The instruction is Ampere and
// later and has no HIP spelling here, the operand has to already be f16 for
// the fragments to be its own bytes, and a caller that asked for f32 precision
// asked for the float path - the accumulator is f32 either way, but the
// products are not.
static __host__ __forceinline__ bool gk_cuda_mma_f16_supported(const struct gk_tensor * dst,
                                                               const struct gk_cuda_scratch * s) {
#if defined(GK_USE_HIP)
    GK_UNUSED(dst);
    GK_UNUSED(s);
    return false;
#else
    if (s == NULL || s->cc < 80) {
        return false;
    }
    if ((int) dst->src[0]->type != GK_TYPE_F16) {
        return false;
    }
    if (gk_get_op_params_i32(dst, 0) == GK_PREC_F32) {
        return false;
    }
    return dst->ne[1] >= GK_CU_MMA_F16_MIN_N;
#endif
}

// --------------------------------------------------------------------------
// The tiled path, dotted as integers.
//
// The float tiled kernel below already decodes each weight element once and
// uses it TILE_N times, so decoding is not what it spends its time on -
// measured, f16 with no decode at all and q4_K are within 14% of each other
// there. What it spends its time on is the multiply-accumulate itself, one
// FFMA per element pair.
//
// So the win here is not avoiding the decode, it is `__dp4a`: four
// multiply-accumulates in one instruction instead of one. The weights are
// staged as their integer codes rather than as floats, which also makes the
// tile a quarter of the size in shared memory, and each group's two scaling
// constants are applied once per 32 elements instead of once per element.
//
// A block owns a TILE_M x TILE_N patch and marches over k a group at a time.
// --------------------------------------------------------------------------

#define GK_CU_MMQ_TILE_M 64
#define GK_CU_MMQ_TILE_N 64
#define GK_CU_MMQ_T      4   // results per thread per axis (TILE / 16)

template <int ATYPE>
static __global__ void gk_cu_k_mul_mat_tiled_q8(gk_tview a, gk_tview_mut d,
                                                const gk_cu_q8blk * aq, int64_t n_grp,
                                                int64_t r2, int64_t r3) {
    // Transposed - [element word][row] - so that a warp walking rows or
    // columns for a fixed word reads consecutive shared addresses.
    __shared__ int   Wc[8][GK_CU_MMQ_TILE_M];
    __shared__ int   Ac[8][GK_CU_MMQ_TILE_N];
    // Two of each where the format's scale covers only sixteen elements; the
    // second slot is dead code for every other format, so it costs them
    // nothing but the shared memory it does not allocate.
    __shared__ float Wscale[gk_cu_has_split_scale<ATYPE>() ? 2 : 1][GK_CU_MMQ_TILE_M];
    __shared__ float Woff  [gk_cu_has_split_scale<ATYPE>() ? 2 : 1][GK_CU_MMQ_TILE_M];
    __shared__ float Ad    [GK_CU_MMQ_TILE_N];
    __shared__ float Asum  [GK_CU_MMQ_TILE_N];
    __shared__ float Asuml [gk_cu_has_split_scale<ATYPE>() ? GK_CU_MMQ_TILE_N : 1];

    const int tx  = threadIdx.x;          // 0..15, column group
    const int ty  = threadIdx.y;          // 0..15, row group
    const int tid = ty * 16 + tx;         // 0..255

    const int64_t m0  = (int64_t) blockIdx.x * GK_CU_MMQ_TILE_M;
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MMQ_TILE_N;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_q8blk * aq23 = aq + i23 * n_cols * n_grp;

    float acc[GK_CU_MMQ_T][GK_CU_MMQ_T];
#pragma unroll
    for (int i = 0; i < GK_CU_MMQ_T; ++i) {
#pragma unroll
        for (int j = 0; j < GK_CU_MMQ_T; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t g = 0; g < n_grp; ++g) {
        // Staging. The first 64 threads take a weight row each, the next 64 an
        // activation column each; a row or column past the end stages zeros so
        // that the arithmetic below needs no bounds test.
        if (tid < GK_CU_MMQ_TILE_M) {
            const int64_t m = m0 + tid;

            int   codes[8];
            float sc[2] = { 0.0f, 0.0f };
            float off[2] = { 0.0f, 0.0f };

            if (m < n_rows) {
                gk_cu_wblk32<ATYPE>((const uint8_t *) gk_cu_row(a, m, a2, a3),
                                    g, codes, sc, off);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    codes[i] = 0;
                }
            }

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                Wc[i][tid] = codes[i];
            }
            Wscale[0][tid] = sc[0];
            Woff  [0][tid] = off[0];
            if (gk_cu_has_split_scale<ATYPE>()) {
                Wscale[1][tid] = sc[1];
                Woff  [1][tid] = off[1];
            }
        } else if (tid < GK_CU_MMQ_TILE_M + GK_CU_MMQ_TILE_N) {
            const int     slot = tid - GK_CU_MMQ_TILE_M;
            const int64_t n    = n0 + slot;

            if (n < n_cols) {
                const gk_cu_q8blk & ab = aq23[n * n_grp + g];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Ac[i][slot] = ab.q[i];
                }
                Ad  [slot] = ab.d;
                Asum[slot] = ab.s;
                if (gk_cu_has_split_scale<ATYPE>()) {
                    Asuml[slot] = ab.sl;
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Ac[i][slot] = 0;
                }
                Ad  [slot] = 0.0f;
                Asum[slot] = 0.0f;
                if (gk_cu_has_split_scale<ATYPE>()) {
                    Asuml[slot] = 0.0f;
                }
            }
        }

        __syncthreads();

        // The integer dot: eight instructions per output pair, against the
        // thirty-two a float pass would take.
        //
        // A split-scale format keeps two accumulators and drains both, because
        // its scale changes halfway through the group. `HALVES` is a constant,
        // so the other formats compile to exactly the single-accumulator loop
        // that was here before.
        constexpr int HALVES = gk_cu_has_split_scale<ATYPE>() ? 2 : 1;
        constexpr int PER    = 8 / HALVES;

        int sumi[HALVES][GK_CU_MMQ_T][GK_CU_MMQ_T];
#pragma unroll
        for (int h = 0; h < HALVES; ++h) {
#pragma unroll
            for (int i = 0; i < GK_CU_MMQ_T; ++i) {
#pragma unroll
                for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                    sumi[h][i][j] = 0;
                }
            }
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            int wv[GK_CU_MMQ_T];
            int av[GK_CU_MMQ_T];
#pragma unroll
            for (int i = 0; i < GK_CU_MMQ_T; ++i) {
                wv[i] = Wc[e][ty * GK_CU_MMQ_T + i];
            }
#pragma unroll
            for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                av[j] = Ac[e][tx * GK_CU_MMQ_T + j];
            }
            const int h = e / PER;
#pragma unroll
            for (int i = 0; i < GK_CU_MMQ_T; ++i) {
#pragma unroll
                for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                    sumi[h][i][j] = gk_cu_dp4a(wv[i], av[j], sumi[h][i][j]);
                }
            }
        }

        // The scales, once per group rather than once per element.
#pragma unroll
        for (int i = 0; i < GK_CU_MMQ_T; ++i) {
            const int mi = ty * GK_CU_MMQ_T + i;
#pragma unroll
            for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                const int nj = tx * GK_CU_MMQ_T + j;

                float v = Wscale[0][mi] * (float) sumi[0][i][j];
                if (gk_cu_has_split_scale<ATYPE>()) {
                    v += Woff  [0][mi] * Asuml[nj];
                    v += Wscale[1][mi] * (float) sumi[HALVES - 1][i][j];
                    v += Woff  [1][mi] * (Asum[nj] - Asuml[nj]);
                } else {
                    v += Woff[0][mi] * Asum[nj];
                }

                acc[i][j] += Ad[nj] * v;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < GK_CU_MMQ_T; ++i) {
        const int64_t m = m0 + ty * GK_CU_MMQ_T + i;
        if (m >= n_rows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < GK_CU_MMQ_T; ++j) {
            const int64_t n = n0 + tx * GK_CU_MMQ_T + j;
            if (n < n_cols) {
                gk_cu_set(d, m, n, i2, i3, acc[i][j]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The same integer dot, on tensor cores.
//
// This is the kernel above with `__dp4a` replaced by `mma.sync`, and it is the
// path every k-quant matmul in a diffusion model takes. The one above stays
// for the parts and shapes this declines.
//
// The two kernels agree on everything that is not the multiply, which is what
// made this cheap to write and is worth stating because it is also what makes
// it cheap to verify:
//
//   * the activation side is byte-identical - the same `gk_cu_k_quantize_act`
//     pass into the same `gk_cu_q8blk` scratch, the same `n_grp = k/32`.
//   * `gk_cu_wblk32<ATYPE>` already hands back exactly the fragment a tensor
//     core wants. Its contract is that `codes[i]` holds elements `4i..4i+3` in
//     natural order, and the `m16n8k32` A fragment wants columns
//     `4*tig..+3` and those columns again sixteen on - which is `codes[tig]`
//     and `codes[4 + tig]`. No repacking, no transpose, no swizzle.
//   * the epilogue is the same identity the float path documents: a value is
//     `scale*code + offset`, so a group contributes
//     `a.d * (scale * sum_j code_j*c_j + offset * sum_j c_j)`.
//
// That last term is the whole difference from the nvfp4 pilot, which had no
// zero point and so needed nothing from the activation block but its codes and
// scale. Here `offset * sum_j c_j` is the q4_1/q4_K minimum, and both halves of
// it are already computed and already staged: `Woff` from `gk_cu_wblk32`, and
// the code sum in `gk_cu_q8blk::s`.
//
// The other difference from the pilot is the drain. A whole-group format's
// scale covers the whole thirty-two, so one `m16n8k32` spans a group and the
// accumulator is drained once per group - halving what the pilot's own notes
// call the whole cost of the format on tensor cores. A split-scale format -
// q2_K, q3_K, q6_K, the split-scale lattice codes - changes scale at element
// sixteen, so its group runs as two `m16n8k16` windows drained where the
// scale changes, the same identity the narrow tile below documents. Twice
// the drain, but still tensor cores: on the shapes that sent q2_K here it
// measures ~1.7x the dp4a tile it used to fall back to, with ggml's mmq at
// the same shapes still ~2x further on - the drain is now the wall.
//
// ## The tile is 128x128, and that matters more than the instruction
//
// Swapping `__dp4a` for `mma.sync` at the dp4a tile's 64x64 was worth 16%.
// That is the measurement that says what this kernel is actually bound by, so
// it is worth writing down rather than rediscovering: at 64x64 and these
// shapes the weight matrix is re-read `n/64` times per GEMM and the
// activations `m/64` times, which for 28672x8742x5376 is 137 and 448 passes
// respectively - 38 GB of traffic for a matmul whose operands are 140 MB. It
// ran at 1.22 TB/s of a 1.79 TB/s part. **The tile was memory bound, and no
// choice of multiply can help a kernel that is waiting for operands.**
//
// Doubling the tile in both directions halves both multipliers. It also
// happens to keep the staging trivial: 128 rows and 128 columns against a
// 256-thread block is still one thread per row and one per column, so this
// needs no strided staging loop - the thing that made the same change look
// expensive when it was contemplated for the nvfp4 pilot at 128 threads.
// --------------------------------------------------------------------------

#define GK_CU_MMAQ_TILE_M  128   // rows a block owns
#define GK_CU_MMAQ_TILE_N  128   // columns a block owns
#define GK_CU_MMAQ_WARPS_M 4
#define GK_CU_MMAQ_WARPS_N 2
// k-groups staged per barrier - the narrow tile's round, at this tile's
// thread count. 256 threads against 128 rows and 128 columns means exactly
// two groups make one decode and one column load per thread per round, so
// the barriers halve and every thread has two independent loads in flight.
//
// Four was measured (2026-08-28, with the two-block occupancy in place) and
// is 4-6% WORSE at every krea2 shape: the extra 22 KB of shared per block
// comes straight out of the unified L1, and this tile's staging reads are
// row-strided - L1 is what was absorbing them.
#define GK_CU_MMAQ_KSTEP   2
#define GK_CU_MMAQ_WM      (GK_CU_MMAQ_TILE_M / GK_CU_MMAQ_WARPS_M)  // 32 rows a warp owns
#define GK_CU_MMAQ_WN      (GK_CU_MMAQ_TILE_N / GK_CU_MMAQ_WARPS_N)  // 64 columns
#define GK_CU_MMAQ_WMT     (GK_CU_MMAQ_WM / 16)                      // 2 mma row tiles
#define GK_CU_MMAQ_WNT     (GK_CU_MMAQ_WN /  8)                      // 8 mma column tiles
#define GK_CU_MMAQ_THREADS (GK_CU_MMAQ_WARPS_M * GK_CU_MMAQ_WARPS_N * GK_WARP_SIZE)

// Two blocks per SM, not one. At 151 registers the first cut of this tile sat
// at a single block - 8 warps on an SM that holds 48 - and measured
// latency-bound everywhere: halving the drain FLOPs (the q2_K fold) and
// halving the barriers (KSTEP) each moved it single digits. minBlocks=2 pins
// the allocator to 128 registers, and the column constants are re-read from
// shared per column tile below instead of cached across the k-step, which is
// what made that budget fit. The split-scale instantiations carry a second
// scale plane and two mma windows; they stay at one block rather than spill.
template <int ATYPE>
static __global__ __launch_bounds__(GK_CU_MMAQ_THREADS,
                                    gk_cu_has_split_scale<ATYPE>() &&
                                    !gk_cu_fold_subscale<ATYPE>() ? 1 : 2)
void gk_cu_k_mul_mat_mma_q8(gk_tview a, gk_tview_mut d,
                            const gk_cu_q8blk * aq, const float * ap,
                            int64_t n_grp, int64_t r2, int64_t r3) {
    // Same staging as the nvfp4 tile: ints rather than bytes, so a fragment
    // read is a plain aligned load of the word a lane already wants.
    //
    // A split-scale format changes scale at element sixteen, so its rows
    // carry two scale/offset planes and its group runs as two k16 windows
    // below - the same shape the narrow tile gives these formats. Whole-group
    // formats keep the single plane and the single k32 window. A format whose
    // sub-scale folds into its codes (q2_K) sits between the two: the folded
    // stage is uniform-scale, so it keeps the single k32 window and the single
    // scale plane, and only the offset stays split.
    constexpr bool FOLD  = gk_cu_fold_subscale<ATYPE>();
    constexpr bool SPLIT = gk_cu_has_split_scale<ATYPE>() && !FOLD;
    constexpr bool OFF2  = gk_cu_has_split_scale<ATYPE>();

    // The code rows are eight words plus one of padding. At eight, a
    // warp-wide staging store walks banks (8r+i) mod 32 - four distinct
    // banks, an eight-way conflict on every store - and ncu put this kernel
    // at 85% of L1TEX peak with mio_throttle its top stall. Nine is coprime
    // with 32, so the same store touches 32 banks.
    __shared__ int   As[GK_CU_MMAQ_KSTEP][GK_CU_MMAQ_TILE_M][9];
    __shared__ int   Bs[GK_CU_MMAQ_KSTEP][GK_CU_MMAQ_TILE_N][9];
    __shared__ float Wsc[GK_CU_MMAQ_KSTEP][SPLIT ? 2 : 1][GK_CU_MMAQ_TILE_M];
    __shared__ float Wof[GK_CU_MMAQ_KSTEP][OFF2  ? 2 : 1][GK_CU_MMAQ_TILE_M];
    __shared__ float Ad [GK_CU_MMAQ_KSTEP][GK_CU_MMAQ_TILE_N];
    // The activation scale times its code sum. The offset term needs the two
    // only as a product, and folding them here turns a multiply per output per
    // group into one per column per group. `Adl` is the same product over the
    // first sixteen codes, which is what the split drain's low window needs;
    // the high window's sum is one subtract, as in the narrow tile.
    __shared__ float Ads[GK_CU_MMAQ_KSTEP][GK_CU_MMAQ_TILE_N];
    __shared__ float Adl[GK_CU_MMAQ_KSTEP][OFF2 ? GK_CU_MMAQ_TILE_N : 1];

    // The high planes' indices, kept in bounds when an array has one plane.
    constexpr int HI  = SPLIT ? 1 : 0;
    constexpr int HIO = OFF2  ? 1 : 0;

    const int lane  = threadIdx.x % GK_WARP_SIZE;
    const int warp  = threadIdx.x / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int warp_m = warp / GK_CU_MMAQ_WARPS_N;
    const int warp_n = warp % GK_CU_MMAQ_WARPS_N;

    const int64_t m0  = (int64_t) blockIdx.x * GK_CU_MMAQ_TILE_M;
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MMAQ_TILE_N;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_q8blk * aq23 = aq + i23 * n_cols * n_grp;
    const float *       ap23 = ap + i23 * n_cols * n_grp * 3;

    // The cooperative stages below hand a thread the same two rows and the
    // same two columns on every k step, so their base pointers hoist out of
    // the k loop entirely - the per-item 64-bit row multiply they replace
    // was itself a measurable slice of the stage.
    static_assert(GK_CU_MMAQ_KSTEP == 2 && GK_CU_MMAQ_TILE_M == 128 &&
                  GK_CU_MMAQ_TILE_N == 128 && GK_CU_MMAQ_THREADS == 256,
                  "the cooperative stages assume this geometry");

    const int stage_w = 2 * ((int) threadIdx.x % 4);   // first of the lane's two words
    const int stage_c = (int) threadIdx.x / 4;         // cluster: row/column base

    const uint8_t * frow[2] = { NULL, NULL };
    if constexpr (FOLD) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int64_t m = m0 + stage_c + j * 64;
            if (m < n_rows) {
                frow[j] = (const uint8_t *) gk_cu_row(a, m, a2, a3);
            }
        }
    }

    const gk_cu_q8blk * bblk[2] = { NULL, NULL };
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int64_t n = n0 + stage_c + j * 64;
        if (n < n_cols) {
            bblk[j] = aq23 + n * n_grp;
        }
    }

    float acc[GK_CU_MMAQ_WMT][GK_CU_MMAQ_WNT][4];
#pragma unroll
    for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMAQ_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[wt][ct][i] = 0.0f;
            }
        }
    }

    for (int64_t g = 0; g < n_grp; g += GK_CU_MMAQ_KSTEP) {
        // The weight stage. ncu on the per-thread version of this loop is
        // worth reading before touching it: at the krea2 shapes the kernel
        // sat at 85% of L1TEX peak with 31.8 sectors per global load - every
        // lane of a warp was decoding a different row two kilobytes from its
        // neighbor, so every load instruction touched thirty-two sectors.
        // DRAM sat at 1.6%; the cost was never the memory, it was the L1
        // pipe processing the divergence.
        if constexpr (FOLD) {
            // The folded formats stage cooperatively: four lanes take one
            // row-group, two adjacent code words each, so a warp's code
            // loads walk eight consecutive 32-byte runs instead of
            // thirty-two scattered rows. q2_K makes this possible - its
            // word decode needs nothing from the rest of the group, only
            // the sub-scale, which lane zero of the cluster reads and
            // shuffles across. A lane's word pair never straddles the
            // half-group boundary, so each pair folds one sub-scale.
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int j  = it & 1;             // which of the lane's rows
                const int sg = it >> 1;
                const int r  = stage_c + j * 64;
                const int64_t gk = g + sg;

                const uint8_t * rp   = frow[j];
                const bool      live = rp != NULL && gk < n_grp;

                const uint8_t * blk = NULL;
                int             sub = 0;
                uint32_t        s01 = 0;
                if (live) {
                    blk = rp + (gk >> 3) * (GK_QK / 16 + GK_QK / 4 + 4);
                    sub = (int) (gk & 7);
                    if (stage_w == 0) {
                        s01 = *(const uint16_t *) (blk + 2 * sub);
                    }
                }
                // warp-uniform, so it sits outside the liveness test
                s01 = __shfl_sync(0xffffffffu, s01, (threadIdx.x & 31u) & ~3u);

                int code0 = 0;
                int code1 = 0;
                if (live) {
                    const int * qs =
                        (const int *) (blk + GK_QK / 16 + (sub / 4) * 32);
                    const int      shift = 2 * (sub & 3);
                    const uint32_t sc    = (s01 >> (stage_w < 4 ? 0 : 8)) & 0xf;
                    code0 = (int) ((uint32_t) ((qs[stage_w    ] >> shift) & 0x03030303) * sc);
                    code1 = (int) ((uint32_t) ((qs[stage_w + 1] >> shift) & 0x03030303) * sc);
                }
                As[sg][r][stage_w    ] = code0;
                As[sg][r][stage_w + 1] = code1;

                if (stage_w == 0) {
                    float d = 0.0f, dmin = 0.0f;
                    if (live) {
                        const float2 dd = gk_cu_h2f2(blk + GK_QK / 16 + GK_QK / 4);
                        d    = dd.x;
                        dmin = dd.y;
                    }
                    Wsc[sg][0][r]   = d;
                    Wof[sg][0][r]   = -dmin * (float) ((s01 >>  4) & 0xf);
                    Wof[sg][HIO][r] = -dmin * (float) ((s01 >> 12) & 0xf);
                }
            }
        } else {
            // Every thread decodes one weight row-group: with 256 threads,
            // 128 rows and two staged groups the units land one decode per
            // thread. A row or group past the end stages zeros, so the
            // arithmetic below needs no bounds test.
            const int r   = (int) threadIdx.x % GK_CU_MMAQ_TILE_M;
            const int sg0 = (int) threadIdx.x / GK_CU_MMAQ_TILE_M;
            const int64_t m = m0 + r;

#pragma unroll
            for (int sg = sg0; sg < GK_CU_MMAQ_KSTEP;
                 sg += GK_CU_MMAQ_THREADS / GK_CU_MMAQ_TILE_M) {
                const int64_t gk = g + sg;

                int   codes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                float sc[2] = { 0.0f, 0.0f };
                float off[2] = { 0.0f, 0.0f };

                if (m < n_rows && gk < n_grp) {
                    gk_cu_wblk32<ATYPE>((const uint8_t *) gk_cu_row(a, m, a2, a3),
                                        gk, codes, sc, off);
                }

#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    As[sg][r][i] = codes[i];
                }
                Wsc[sg][0][r] = sc[0];
                Wof[sg][0][r] = off[0];
                if (SPLIT) {
                    Wsc[sg][HI][r] = sc[1];
                }
                if (OFF2) {
                    Wof[sg][HIO][r] = off[1];
                }
            }
        }
        {
            // The activation codes stage cooperatively for every format -
            // a q8blk's eight words are the same eight words whatever the
            // weights are. Four lanes copy one block, two words each, so a
            // warp's loads walk eight 32-byte code runs instead of
            // thirty-two blocks 36*n_grp bytes apart (see the ncu note on
            // the weight stage above).
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int j  = it & 1;         // which of the lane's columns
                const int sg = it >> 1;
                const int c  = stage_c + j * 64;
                const int64_t gk = g + sg;

                const gk_cu_q8blk * bp = bblk[j];

                int q0 = 0;
                int q1 = 0;
                if (bp != NULL && gk < n_grp) {
                    q0 = bp[gk].q[stage_w    ];
                    q1 = bp[gk].q[stage_w + 1];
                }
                Bs[sg][c][stage_w    ] = q0;
                Bs[sg][c][stage_w + 1] = q1;
            }
        }
        {
            // The scalar planes, from the transposed copy the quantize pass
            // wrote: a warp of consecutive columns reads consecutive words,
            // where the same three scalars read out of the q8blk array were
            // three loads 36*n_grp bytes apart per lane.
            const int c   = (int) threadIdx.x % GK_CU_MMAQ_TILE_N;
            const int sg0 = (int) threadIdx.x / GK_CU_MMAQ_TILE_N;
            const int64_t n = n0 + c;

#pragma unroll
            for (int sg = sg0; sg < GK_CU_MMAQ_KSTEP;
                 sg += GK_CU_MMAQ_THREADS / GK_CU_MMAQ_TILE_N) {
                const int64_t gk = g + sg;

                if (n < n_cols && gk < n_grp) {
                    const float * p = ap23 + gk * 3 * n_cols + n;
                    Ad [sg][c] = p[0];
                    Ads[sg][c] = p[n_cols];
                    if (OFF2) {
                        Adl[sg][c] = p[2 * n_cols];
                    }
                } else {
                    Ad [sg][c] = 0.0f;
                    Ads[sg][c] = 0.0f;
                    if (OFF2) {
                        Adl[sg][c] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int gg = 0; gg < GK_CU_MMAQ_KSTEP; ++gg) {

        int   af[GK_CU_MMAQ_WMT][4];
        float ws[GK_CU_MMAQ_WMT][2];
        float wo[GK_CU_MMAQ_WMT][2];
        float wsh[GK_CU_MMAQ_WMT][2];
        float woh[GK_CU_MMAQ_WMT][2];

#pragma unroll
        for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
            const int r_lo = warp_m * GK_CU_MMAQ_WM + wt * 16 + group;

            af[wt][0] = As[gg][r_lo    ][tig];
            af[wt][1] = As[gg][r_lo + 8][tig];
            af[wt][2] = As[gg][r_lo    ][4 + tig];
            af[wt][3] = As[gg][r_lo + 8][4 + tig];

            ws[wt][0] = Wsc[gg][0][r_lo    ];
            ws[wt][1] = Wsc[gg][0][r_lo + 8];
            wo[wt][0] = Wof[gg][0][r_lo    ];
            wo[wt][1] = Wof[gg][0][r_lo + 8];
            if (SPLIT) {
                wsh[wt][0] = Wsc[gg][HI][r_lo    ];
                wsh[wt][1] = Wsc[gg][HI][r_lo + 8];
            }
            if (OFF2) {
                woh[wt][0] = Wof[gg][HIO][r_lo    ];
                woh[wt][1] = Wof[gg][HIO][r_lo + 8];
            }
        }

#pragma unroll
        for (int ct = 0; ct < GK_CU_MMAQ_WNT; ++ct) {
            // One B fragment feeds both row tiles - which is what giving a
            // warp two of them buys, exactly as in the nvfp4 tile.
            const int c = warp_n * GK_CU_MMAQ_WN + ct * 8 + group;

            int bf[2];
            bf[0] = Bs[gg][c][tig];
            bf[1] = Bs[gg][c][4 + tig];

            // The lane's column constants, re-read per column tile rather
            // than cached across the whole k-step: cached they were six
            // two-wide register arrays, and that cache is what held the
            // kernel at one block per SM. Loaded here they live one loop
            // iteration, the loads overlap the mma issue, and the doubled
            // occupancy pays for the re-read many times over.
            const int cd = warp_n * GK_CU_MMAQ_WN + ct * 8 + tig * 2;

            float adv[2], asv[2], alv[2], ahv[2];
            adv[0] = Ad[gg][cd + 0];
            adv[1] = Ad[gg][cd + 1];
            if (OFF2) {
                alv[0] = Adl[gg][cd + 0];
                alv[1] = Adl[gg][cd + 1];
                // the high window's product, one subtract here rather than
                // one per output element in the drain
                ahv[0] = Ads[gg][cd + 0] - alv[0];
                ahv[1] = Ads[gg][cd + 1] - alv[1];
                asv[0] = asv[1] = 0.0f;
            } else {
                asv[0] = Ads[gg][cd + 0];
                asv[1] = Ads[gg][cd + 1];
                alv[0] = alv[1] = 0.0f;
                ahv[0] = ahv[1] = 0.0f;
            }

#pragma unroll
            for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
                if (SPLIT) {
                    // The scale changes at element sixteen, so the group is
                    // two k16 windows drained where it changes - the narrow
                    // tile's identity, at this tile's width.
                    const int af_lo[2] = { af[wt][0], af[wt][1] };
                    const int af_hi[2] = { af[wt][2], af[wt][3] };

                    int df0[4] = { 0, 0, 0, 0 };
                    int df1[4] = { 0, 0, 0, 0 };
                    gk_cu_mma_s8(df0, af_lo, bf[0]);
                    gk_cu_mma_s8(df1, af_hi, bf[1]);

#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int rh = i >> 1;      // row half of the mma tile
                        const int ch = i & 1;       // which of the lane's columns
                        acc[wt][ct][i] += adv[ch] * (ws[wt][rh]  * (float) df0[i]
                                                   + wsh[wt][rh] * (float) df1[i])
                                        + wo[wt][rh]  * alv[ch]
                                        + woh[wt][rh] * ahv[ch];
                    }
                } else if (FOLD) {
                    // The sub-scale is already in the codes, so one k32
                    // window spans the group and the scale drain is the
                    // whole-group formats'; only the offset stays split.
                    int df[4] = { 0, 0, 0, 0 };

                    gk_cu_mma_s8_k32(df, af[wt], bf);

#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int rh = i >> 1;      // row half of the mma tile
                        const int ch = i & 1;       // which of the lane's columns
                        acc[wt][ct][i] += ws[wt][rh]  * adv[ch] * (float) df[i]
                                        + wo[wt][rh]  * alv[ch]
                                        + woh[wt][rh] * ahv[ch];
                    }
                } else {
                    int df[4] = { 0, 0, 0, 0 };

                    gk_cu_mma_s8_k32(df, af[wt], bf);

                    acc[wt][ct][0] += ws[wt][0] * adv[0] * (float) df[0] + wo[wt][0] * asv[0];
                    acc[wt][ct][1] += ws[wt][0] * adv[1] * (float) df[1] + wo[wt][0] * asv[1];
                    acc[wt][ct][2] += ws[wt][1] * adv[0] * (float) df[2] + wo[wt][1] * asv[0];
                    acc[wt][ct][3] += ws[wt][1] * adv[1] * (float) df[3] + wo[wt][1] * asv[1];
                }
            }
        }
        }

        __syncthreads();
    }

#pragma unroll
    for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMAQ_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t m = m0 + warp_m * GK_CU_MMAQ_WM + wt * 16
                                + group + (i >= 2 ? 8 : 0);
                const int64_t n = n0 + warp_n * GK_CU_MMAQ_WN + ct * 8
                                + tig * 2 + (i & 1);

                if (m < n_rows && n < n_cols) {
                    gk_cu_set(d, m, n, i2, i3, acc[wt][ct][i]);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// The q2_K pipeline tile: the 128x128 integer tile above, its staging
// double-buffered through cp.async.
//
// Investigation 10 left that tile bound on the *latency* of its own staging
// loads: post-fix ncu read long_scoreboard 32%, issue 49%, no pipe
// saturated. Its round is stage -> sync -> compute -> sync, so inside a
// block nothing overlaps anything; the second block per SM was all the
// overlap it had. This kernel overlaps a whole round: while round g
// computes, every staging word of round g+1 is already in flight - and
// `cp.async` is what makes that free, because the copy never lands in a
// register, so there is no scoreboard for the issuing warp to wait on.
//
// Three q2_K facts make the stage smaller as well as asynchronous, and they
// are why this is a q2_K kernel rather than a template:
//
//   * the k loop steps by two groups and a superblock holds eight, so a
//     round's pair of groups always shares one 32-byte code run (sub and
//     sub+1 sit in the same half, with shifts s and s+2). The codes are
//     staged *raw*, once per pair, and folded at fragment load - where the
//     synchronous tile stages decoded codes once per group;
//   * the pair's four scale/minimum bytes are one *aligned* word
//     (`blk + 2*sub` with sub even), and d/dmin at byte 80 another - so the
//     whole weight header is two cp.async words per row per round;
//   * the sub-scale fold (gk_cu_fold_subscale) keeps the fragment-time
//     decode to a shift, a mask and one byte-parallel multiply per word.
//
// Ten words per row per round against the synchronous tile's eighteen, all
// asynchronous, and the double buffer of everything still sits at ~35 KB -
// under the 48 KB static limit, and two blocks per SM fit in Ada's 99.
//
// Rows and columns past the edge are zeroed once in the prologue and never
// staged again; with zero codes, zero d and zero dmin every term they
// contribute is zero, so neither the stage nor the compute carries bounds
// arithmetic. A round whose second group is past n_grp skips that group's
// compute block-uniformly.
//
// `GK_MM_MMA_PIPE=0` sends q2_K back to the synchronous tile - a bisect
// lever, not a setting.
// --------------------------------------------------------------------------

// A 4-byte global->shared copy that never touches a register. Guarded like
// the s8 mma helpers: where that instruction compiles (sm_80+), so does
// cp.async. 4 bytes rather than 16 because q2_K's 84-byte blocks and the
// 36-byte activation structs guarantee word alignment and nothing more.
static __device__ __forceinline__ void gk_cu_cp_async4(void * dst, const void * src) {
#if defined(GK_CU_HAVE_MMA)
    const unsigned s = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(src));
#else
    *(int *) dst = *(const int *) src;
#endif
}

static __device__ __forceinline__ void gk_cu_cp_async_commit() {
#if defined(GK_CU_HAVE_MMA)
    asm volatile("cp.async.commit_group;");
#endif
}

// Wait until at most N commit groups are still in flight.
template <int N>
static __device__ __forceinline__ void gk_cu_cp_async_wait() {
#if defined(GK_CU_HAVE_MMA)
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif
}

// WM_ is the tile's height in warps. 4 is the 128-row tile: 256 threads,
// two blocks per SM. 8 is a 256-row tile: 512 threads, one block per SM -
// the same sixteen warps either way, but a row of activations now serves
// twice the weight rows, so the larger operand stream halves. The
// activations, not the weights, dominate this kernel's traffic (36 bytes of
// codes and 12 of planes per column-group against the weights' amortized
// ten per row-round), which is why the extra height pays where the smaller
// half-based weight round above already took the weight stream to its
// floor. Tall runs where the rows exist to fill it; the modulation
// projection keeps the short tile.
template <int WM_>
static __global__ __launch_bounds__(WM_ * 64, WM_ == 4 ? 2 : 1)
void gk_cu_k_mul_mat_mma_q2k_pipe(gk_tview a, gk_tview_mut d,
                                  const gk_cu_q8blk * aq, const float * ap,
                                  int64_t n_grp, int64_t r2, int64_t r3,
                                  int act, float4 actp) {
    constexpr int TM      = WM_ * GK_CU_MMAQ_WM;       // rows a block owns
    constexpr int THREADS = TM * 2;
    constexpr int APAD    = TM == 128 ? 9 : 8;         // tall trims the pad to fit 48 KB
    constexpr int NCOL    = THREADS / 4;               // rows/columns one stage pass covers
    constexpr int NB      = GK_CU_MMAQ_TILE_N / NCOL;  // a thread's distinct columns

    static_assert(WM_ == 4 || WM_ == 8, "the two shapes this kernel is tuned at");
    // Everything the compute phase reads is double-buffered; the weight side
    // is raw (see above), the activation side is the same codes and
    // transposed planes the synchronous tile stages.
    //
    // The two sides run on different cadences. A round is two groups - the
    // mma's k32 window - but a q2_K superblock *half* is four: one 32-byte
    // code run, two scale/minimum words and one d/dmin word serve four
    // groups, only the shift changes. So the weight buffers are keyed by
    // half and staged every other round, one round ahead, which halves the
    // weight stream against the round-keyed version of this kernel - each
    // raw byte is now fetched exactly once per pass, which is the floor.
    __shared__ int      Ar [2][TM][APAD];                   // the half's qs words (8 + pad)
    __shared__ uint32_t Ahd[2][TM][2];                      // the half's two s0,s1 word pairs
    __shared__ uint32_t Add[2][TM];                         // d, dmin as raw halves
    __shared__ int      Bs [2][2][GK_CU_MMAQ_TILE_N][9];    // [buf][sg][col][word + pad]
    // 16-aligned so a lane's even-indexed pair of column constants is one
    // 64-bit load - three loads per column tile instead of six.
    __shared__ __align__(16) float Apl[2][2][3][GK_CU_MMAQ_TILE_N]; // d, d*s, d*sl planes

    const int lane  = threadIdx.x % GK_WARP_SIZE;
    const int warp  = threadIdx.x / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int warp_m = warp / GK_CU_MMAQ_WARPS_N;
    const int warp_n = warp % GK_CU_MMAQ_WARPS_N;

    const int64_t m0  = (int64_t) blockIdx.x * TM;
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MMAQ_TILE_N;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_q8blk * aq23 = aq + i23 * n_cols * n_grp;
    const float *       ap23 = ap + i23 * n_cols * n_grp * 3;

    static_assert(GK_CU_MMAQ_KSTEP == 2 && GK_CU_MMAQ_TILE_N == 128,
                  "the pipeline stage assumes this geometry");

    // Stage geometry, as in the synchronous tile: four lanes to a cluster,
    // two adjacent words each, and every base pointer hoisted out of the k
    // loop because a thread only ever touches the same rows and columns.
    const int stage_w = 2 * ((int) threadIdx.x % 4);
    const int stage_c = (int) threadIdx.x / 4;

    const uint8_t * frow[2] = { NULL, NULL };
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int64_t m = m0 + stage_c + j * NCOL;
        if (m < n_rows) {
            frow[j] = (const uint8_t *) gk_cu_row(a, m, a2, a3);
        }
    }

    // The header stage: the low half of the block fetches the scale/min
    // words, the high half d/dmin, each for row threadIdx.x % TM.
    const int       hdr_r   = (int) threadIdx.x % TM;
    const bool      hdr_dd  = (int) threadIdx.x >= TM;
    const uint8_t * hdr_row = m0 + hdr_r < n_rows
        ? (const uint8_t *) gk_cu_row(a, m0 + hdr_r, a2, a3) : NULL;

    const gk_cu_q8blk * bblk[NB] = {};
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        const int64_t n = n0 + stage_c + j * NCOL;
        if (n < n_cols) {
            bblk[j] = aq23 + n * n_grp;
        }
    }

    // The scalar planes: thread (col, sg) copies its three words; the tall
    // tile's upper half sits this stage out.
    const int     pl_c = (int) threadIdx.x % GK_CU_MMAQ_TILE_N;
    const int     pl_s = (int) threadIdx.x / GK_CU_MMAQ_TILE_N;
    const bool    pl_ok = pl_s < 2 && n0 + pl_c < n_cols;

    // Rows and columns past the edge, zeroed once in both buffers. The
    // stage never writes them again, so they stay zero and contribute zero.
    {
        const int x = (int) threadIdx.x % TM;
        const int h = (int) threadIdx.x / TM;

        if (m0 + x >= n_rows) {
            if (h == 0) {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { Ar[b][x][i] = 0; }
                }
            } else {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    Ahd[b][x][0] = 0;
                    Ahd[b][x][1] = 0;
                    Add[b][x] = 0;
                }
            }
        }

        const int c  = (int) threadIdx.x % GK_CU_MMAQ_TILE_N;
        const int h2 = (int) threadIdx.x / GK_CU_MMAQ_TILE_N;
        if (n0 + c >= n_cols && h2 < 2) {
            if (h2 == 0) {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
#pragma unroll
                    for (int sg = 0; sg < 2; ++sg) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) { Bs[b][sg][c][i] = 0; }
                    }
                }
            } else {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
#pragma unroll
                    for (int sg = 0; sg < 2; ++sg) {
#pragma unroll
                        for (int q = 0; q < 3; ++q) { Apl[b][sg][q][c] = 0.0f; }
                    }
                }
            }
        }
    }

    // The weight stage, once per superblock half (four groups): the half's
    // eight code words, its two scale/minimum words, and d/dmin.
    const auto stage_a = [&](int64_t h, int ab) {
        const int64_t sb = h >> 1;          // two halves to a superblock
        const int     hh = (int) (h & 1);   // which 32-byte code run

#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint8_t * rp = frow[j];
            if (rp != NULL) {
                const uint8_t * qs = rp + sb * (GK_QK / 16 + GK_QK / 4 + 4)
                                   + GK_QK / 16 + hh * 32;
                const int r = stage_c + j * NCOL;
                gk_cu_cp_async4(&Ar[ab][r][stage_w    ], qs + 4 * stage_w);
                gk_cu_cp_async4(&Ar[ab][r][stage_w + 1], qs + 4 * stage_w + 4);
            }
        }

        if (hdr_row != NULL) {
            const uint8_t * blk = hdr_row + sb * (GK_QK / 16 + GK_QK / 4 + 4);
            if (!hdr_dd) {
                gk_cu_cp_async4(&Ahd[ab][hdr_r][0], blk + 8 * hh);
                gk_cu_cp_async4(&Add[ab][hdr_r], blk + GK_QK / 16 + GK_QK / 4);
            } else {
                gk_cu_cp_async4(&Ahd[ab][hdr_r][1], blk + 8 * hh + 4);
            }
        }
    };

    // The activation stage, once per round (two groups).
    const auto stage_b = [&](int64_t it, int buf) {
        const int64_t g = it * 2;

#pragma unroll
        for (int it2 = 0; it2 < 2 * NB; ++it2) {
            const int jj = it2 % NB;
            const int sg = it2 / NB;
            const int c  = stage_c + jj * NCOL;
            const int64_t gk = g + sg;

            const gk_cu_q8blk * bp = bblk[jj];
            if (bp != NULL && gk < n_grp) {
                gk_cu_cp_async4(&Bs[buf][sg][c][stage_w    ], &bp[gk].q[stage_w    ]);
                gk_cu_cp_async4(&Bs[buf][sg][c][stage_w + 1], &bp[gk].q[stage_w + 1]);
            }
        }

        // the scalar planes
        {
            const int64_t gk = g + pl_s;
            if (pl_ok && gk < n_grp) {
                const float * p = ap23 + gk * 3 * n_cols + (n0 + pl_c);
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    gk_cu_cp_async4(&Apl[buf][pl_s][q][pl_c], p + q * n_cols);
                }
            }
        }
    };

    float acc[GK_CU_MMAQ_WMT][GK_CU_MMAQ_WNT][4];
#pragma unroll
    for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMAQ_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[wt][ct][i] = 0.0f;
            }
        }
    }

    const int64_t n_iter = (n_grp + 1) / 2;
    const int64_t n_half = (n_iter + 1) / 2;

    stage_a(0, 0);
    stage_b(0, 0);
    gk_cu_cp_async_commit();

    for (int64_t it = 0; it < n_iter; ++it) {
        const int buf = (int) (it & 1);
        const int ab  = (int) ((it >> 1) & 1);
        const int hq  = (int) (it & 1);       // which pair of the half

        // Round it+1's activations go into the other buffer before this
        // round computes, and every second round the *next half's* weights
        // go with them - a round ahead of first use, behind two barriers of
        // last use. The trailing barrier of round it-1 is what makes the
        // writes safe. One commit per round keeps the wait arithmetic
        // uniform: with one group still in flight everything older is
        // complete, which covers both this round's activations and this
        // half's weights.
        if (it + 1 < n_iter) {
            stage_b(it + 1, buf ^ 1);
            if (hq == 1 && (it >> 1) + 1 < n_half) {
                stage_a((it >> 1) + 1, ab ^ 1);
            }
            gk_cu_cp_async_commit();
            gk_cu_cp_async_wait<1>();
        } else {
            gk_cu_cp_async_wait<0>();
        }
        __syncthreads();

        const int64_t g   = it * 2;
        const int     sh0 = hq * 4;   // the half's second pair starts at shift 4

        // The raw words and headers serve all four of the half's groups, so
        // they are read once per round and decoded per group.
        int      aw [GK_CU_MMAQ_WMT][4];
        uint32_t s01[GK_CU_MMAQ_WMT][2];
        float2   dd [GK_CU_MMAQ_WMT][2];

#pragma unroll
        for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
            const int r_lo = warp_m * GK_CU_MMAQ_WM + wt * 16 + group;

            aw[wt][0] = Ar[ab][r_lo    ][tig];
            aw[wt][1] = Ar[ab][r_lo + 8][tig];
            aw[wt][2] = Ar[ab][r_lo    ][4 + tig];
            aw[wt][3] = Ar[ab][r_lo + 8][4 + tig];

            s01[wt][0] = Ahd[ab][r_lo    ][hq];
            s01[wt][1] = Ahd[ab][r_lo + 8][hq];
            dd [wt][0] = gk_cu_h2f2_w(Add[ab][r_lo    ]);
            dd [wt][1] = gk_cu_h2f2_w(Add[ab][r_lo + 8]);
        }

#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            if (g + gg >= n_grp) {
                break;
            }

            const int sh   = sh0 + 2 * gg;   // this group's code shift
            const int sb16 = 16 * gg;        // where its bytes sit in the header word

            // The fold, at fragment time: shift, mask, one byte-parallel
            // multiply per word - and the header's minima become the offset
            // planes the drain wants.
            int   af [GK_CU_MMAQ_WMT][4];
            float ws [GK_CU_MMAQ_WMT][2];
            float wo [GK_CU_MMAQ_WMT][2];
            float woh[GK_CU_MMAQ_WMT][2];

#pragma unroll
            for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
#pragma unroll
                for (int rh = 0; rh < 2; ++rh) {
                    const uint32_t s = s01[wt][rh] >> sb16;

                    af[wt][rh    ] = (int) ((uint32_t) ((aw[wt][rh    ] >> sh) & 0x03030303)
                                            * (s & 0xf));
                    af[wt][rh + 2] = (int) ((uint32_t) ((aw[wt][rh + 2] >> sh) & 0x03030303)
                                            * ((s >> 8) & 0xf));

                    ws [wt][rh] = dd[wt][rh].x;
                    wo [wt][rh] = -dd[wt][rh].y * (float) ((s >>  4) & 0xf);
                    woh[wt][rh] = -dd[wt][rh].y * (float) ((s >> 12) & 0xf);
                }
            }

#pragma unroll
            for (int ct = 0; ct < GK_CU_MMAQ_WNT; ++ct) {
                const int c = warp_n * GK_CU_MMAQ_WN + ct * 8 + group;

                int bf[2];
                bf[0] = Bs[buf][gg][c][tig];
                bf[1] = Bs[buf][gg][c][4 + tig];

                const int cd = warp_n * GK_CU_MMAQ_WN + ct * 8 + tig * 2;

                // cd is even, so each pair is one aligned 64-bit load
                const float2 advv = *(const float2 *) &Apl[buf][gg][0][cd];
                const float2 adsv = *(const float2 *) &Apl[buf][gg][1][cd];
                const float2 alvv = *(const float2 *) &Apl[buf][gg][2][cd];

                const float adv[2] = { advv.x, advv.y };
                const float alv[2] = { alvv.x, alvv.y };
                const float ahv[2] = { adsv.x - alvv.x, adsv.y - alvv.y };

#pragma unroll
                for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
                    int df[4] = { 0, 0, 0, 0 };

                    gk_cu_mma_s8_k32(df, af[wt], bf);

#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int rh = i >> 1;      // row half of the mma tile
                        const int ch = i & 1;       // which of the lane's columns
                        acc[wt][ct][i] += ws[wt][rh]  * adv[ch] * (float) df[i]
                                        + wo[wt][rh]  * alv[ch]
                                        + woh[wt][rh] * ahv[ch];
                    }
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int wt = 0; wt < GK_CU_MMAQ_WMT; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMAQ_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t m = m0 + warp_m * GK_CU_MMAQ_WM + wt * 16
                                + group + (i >= 2 ? 8 : 0);
                const int64_t n = n0 + warp_n * GK_CU_MMAQ_WN + ct * 8
                                + tig * 2 + (i & 1);

                if (m < n_rows && n < n_cols) {
                    float v = acc[wt][ct][i];
                    // The fused activation epilogue: the unary that always
                    // follows a gate projection, applied to the finished
                    // accumulator instead of in a pass of its own - the
                    // same gk_cu_unary the standalone kernel would run, so
                    // the value is bit-identical.
                    if (act >= 0) {
                        v = gk_cu_unary(act, v, actp.x, actp.y, actp.z, actp.w);
                    }
                    gk_cu_set(d, m, n, i2, i3, v);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// The narrow integer tile: 2..23 columns, the shape of a speculative verify.
//
// The NC dp4a mat-vec covers this range, and at four columns it is issue
// bound: an instruction-budget estimate put its decode+4-dot loop at ~100%
// issue rate, costing ~2x the single-column pass for what should be the same
// memory traffic. One `mma.sync` retires the work of 128 dp4a instructions,
// so the multiply drops out of the budget and the pass goes back to being
// paid for in decodes - which a batch amortizes across every column at once,
// where the NC path re-reads and re-decodes the weights once per group of
// four columns.
//
// A block is eight warps stacked along m: a warp owns 32 rows, the block 256,
// and the whole of n - at most three 8-wide mma tiles. Columns come from the
// same quantized-activation scratch as the mat-vec path. k is cut across
// blocks exactly as the f16 tile cuts it, with the same partial planes and
// the same combine pass, because a projection matrix is short of rows in
// exactly the way that tile's UNet shapes are.
//
// The split-scale formats are admitted here, as everywhere on the integer
// tensor-core paths: a verify's dominant weights are whatever format the
// model shipped in, and q2_K excluded is most of a q2_K model excluded.
// Their group is two `m16n8k16` windows drained separately - the accumulator
// has to be emptied where the scale changes - against the whole-group
// formats' one k32 window. The activation block's `sl` is what makes the
// second drain's offset term affordable: the low half's code sum is
// precomputed, so the high half's is one subtract. The 128x128 tile above
// runs the same two-window identity for these formats.
// --------------------------------------------------------------------------

#define GK_CU_MMAN_TILE_M  128   // rows a block owns; a warp owns 16
#define GK_CU_MMAN_TILE_N  8     // columns: one mma tile
#define GK_CU_MMAN_WARPS   8
#define GK_CU_MMAN_THREADS (GK_CU_MMAN_WARPS * GK_WARP_SIZE)
// k-groups staged per barrier. The first cut of this kernel staged one and
// paid two barriers per ~66 scattered bytes per thread - it lost to the dp4a
// mat-vec by 2-3x on pure latency. Four gives each thread two independent
// decodes in flight and divides the barriers by four.
#define GK_CU_MMAN_KSTEP   4
#define GK_CU_MMAN_MIN_N   2     // below this the plain mat-vec keeps the win
#define GK_CU_MMAN_SPLIT_MIN_GRP 8   // a k piece keeps at least this many groups

template <int ATYPE>
static __global__ __launch_bounds__(GK_CU_MMAN_THREADS, 2)
void gk_cu_k_mul_mat_mma_q8n(gk_tview a, gk_tview_mut d,
                             const gk_cu_q8blk * aq, int64_t n_grp,
                             int64_t r2, int64_t r3,
                             float * part, int64_t n_splits, int64_t n_23) {
    __shared__ int   As[GK_CU_MMAN_KSTEP][GK_CU_MMAN_TILE_M][8];
    __shared__ float Wsc[GK_CU_MMAN_KSTEP][GK_CU_MMAN_TILE_M][2];
    __shared__ float Wof[GK_CU_MMAN_KSTEP][GK_CU_MMAN_TILE_M][2];
    __shared__ int   Bs[GK_CU_MMAN_KSTEP][GK_CU_MMAN_TILE_N][8];
    __shared__ float Ad [GK_CU_MMAN_KSTEP][GK_CU_MMAN_TILE_N];
    __shared__ float Ads[GK_CU_MMAN_KSTEP][GK_CU_MMAN_TILE_N];  // d * (sum of the 32)
    __shared__ float Adl[GK_CU_MMAN_KSTEP][GK_CU_MMAN_TILE_N];  // d * (sum of the first 16)

    const int lane  = threadIdx.x % GK_WARP_SIZE;
    const int warp  = threadIdx.x / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int64_t m0    = (int64_t) blockIdx.x * GK_CU_MMAN_TILE_M;
    const int64_t n0    = (int64_t) blockIdx.y * GK_CU_MMAN_TILE_N;
    const int64_t i23   = blockIdx.z / n_splits;
    const int64_t split = blockIdx.z % n_splits;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_q8blk * aq23 = aq + i23 * n_cols * n_grp;

    const int64_t g_per = (n_grp + n_splits - 1) / n_splits;
    const int64_t g0    = split * g_per;
    const int64_t g1    = g0 + g_per < n_grp ? g0 + g_per : n_grp;

    // The staging assignment: thread t owns row t % TILE_M at staged groups
    // 2*(t / TILE_M) and 2*(t / TILE_M) + 1 - two decodes with nothing between
    // them, which is the memory-level parallelism the whole redesign is for.
    const int st_r  = (int) threadIdx.x % GK_CU_MMAN_TILE_M;
    const int st_g  = 2 * ((int) threadIdx.x / GK_CU_MMAN_TILE_M);

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    const int r_lo = warp * 16 + group;

    for (int64_t g = g0; g < g1; g += GK_CU_MMAN_KSTEP) {
        // A row past the end - or a staged group past this split's slice -
        // stages zero codes and zero scales, so the arithmetic below needs
        // no bound.
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int     gg = st_g + s;
            const int64_t gk = g + gg;
            const int64_t m  = m0 + st_r;

            int   codes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            float sc[2]  = { 0.0f, 0.0f };
            float off[2] = { 0.0f, 0.0f };

            if (m < n_rows && gk < g1) {
                gk_cu_wblk32<ATYPE>((const uint8_t *) gk_cu_row(a, m, a2, a3),
                                    gk, codes, sc, off);
            }

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                As[gg][st_r][i] = codes[i];
            }
            Wsc[gg][st_r][0] = sc[0];
            Wsc[gg][st_r][1] = sc[1];
            Wof[gg][st_r][0] = off[0];
            Wof[gg][st_r][1] = off[1];
        }

        if (threadIdx.x < GK_CU_MMAN_KSTEP * GK_CU_MMAN_TILE_N) {
            const int     gg = (int) threadIdx.x / GK_CU_MMAN_TILE_N;
            const int     c  = (int) threadIdx.x % GK_CU_MMAN_TILE_N;
            const int64_t n  = n0 + c;
            const int64_t gk = g + gg;

            if (n < n_cols && gk < g1) {
                const gk_cu_q8blk & ab = aq23[n * n_grp + gk];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Bs[gg][c][i] = ab.q[i];
                }
                Ad [gg][c] = ab.d;
                Ads[gg][c] = ab.d * ab.s;
                Adl[gg][c] = ab.d * ab.sl;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Bs[gg][c][i] = 0;
                }
                Ad [gg][c] = 0.0f;
                Ads[gg][c] = 0.0f;
                Adl[gg][c] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int gg = 0; gg < GK_CU_MMAN_KSTEP; ++gg) {
            // Column constants for this lane's two output columns.
            float adv[2], asv[2], alv[2];
            adv[0] = Ad [gg][tig * 2];  adv[1] = Ad [gg][tig * 2 + 1];
            asv[0] = Ads[gg][tig * 2];  asv[1] = Ads[gg][tig * 2 + 1];
            alv[0] = Adl[gg][tig * 2];  alv[1] = Adl[gg][tig * 2 + 1];

            int bf[2];
            bf[0] = Bs[gg][group][tig];
            bf[1] = Bs[gg][group][4 + tig];

            if (gk_cu_has_split_scale<ATYPE>()) {
                // The scale changes at element sixteen, so the group is two
                // k16 windows drained where it changes.
                int af_lo[2] = { As[gg][r_lo][tig],     As[gg][r_lo + 8][tig]     };
                int af_hi[2] = { As[gg][r_lo][4 + tig], As[gg][r_lo + 8][4 + tig] };

                int df0[4] = { 0, 0, 0, 0 };
                int df1[4] = { 0, 0, 0, 0 };
                gk_cu_mma_s8(df0, af_lo, bf[0]);
                gk_cu_mma_s8(df1, af_hi, bf[1]);

                const float ws0[2] = { Wsc[gg][r_lo][0], Wsc[gg][r_lo + 8][0] };
                const float ws1[2] = { Wsc[gg][r_lo][1], Wsc[gg][r_lo + 8][1] };
                const float wo0[2] = { Wof[gg][r_lo][0], Wof[gg][r_lo + 8][0] };
                const float wo1[2] = { Wof[gg][r_lo][1], Wof[gg][r_lo + 8][1] };

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int rh = i >> 1;      // 0: row group, 1: row group+8
                    const int ch = i & 1;       // which of the lane's columns
                    acc[i] += ws0[rh] * adv[ch] * (float) df0[i] + wo0[rh] * alv[ch]
                            + ws1[rh] * adv[ch] * (float) df1[i] + wo1[rh] * (asv[ch] - alv[ch]);
                }
            } else {
                int af[4] = { As[gg][r_lo][tig],     As[gg][r_lo + 8][tig],
                              As[gg][r_lo][4 + tig], As[gg][r_lo + 8][4 + tig] };

                int df[4] = { 0, 0, 0, 0 };
                gk_cu_mma_s8_k32(df, af, bf);

                const float ws[2] = { Wsc[gg][r_lo][0], Wsc[gg][r_lo + 8][0] };
                const float wo[2] = { Wof[gg][r_lo][0], Wof[gg][r_lo + 8][0] };

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int rh = i >> 1;
                    const int ch = i & 1;
                    acc[i] += ws[rh] * adv[ch] * (float) df[i] + wo[rh] * asv[ch];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t m = m0 + warp * 16 + group + ((i >> 1) != 0 ? 8 : 0);
        const int64_t n = n0 + tig * 2 + (i & 1);

        if (m < n_rows && n < n_cols) {
            if (part != NULL) {
                part[((split * n_23 + i23) * n_cols + n) * n_rows + m] = acc[i];
            } else {
                gk_cu_set(d, m, n, i2, i3, acc[i]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The tiled path: same result as gk_cu_k_mul_mat, arranged for reuse.
//
// Each block owns a TILE_M x TILE_N patch of `d` and walks the whole of k,
// which is the opposite of the mat-vec kernel's split-k. That trade is what
// buys the reuse: k stays in one block, so a staged tile serves every output
// in the patch and no cross-block reduction is needed.
// --------------------------------------------------------------------------

template <int ATYPE>
static __global__ void gk_cu_k_mul_mat_tiled(gk_tview a, gk_tview b, gk_tview_mut d,
                                             int64_t k_len, int64_t r2, int64_t r3,
                                             int act, float4 actp) {
    __shared__ float As[GK_CU_MM_TILE_K][GK_CU_MM_TILE_M];
    __shared__ float Bs[GK_CU_MM_TILE_K][GK_CU_MM_TILE_N];

    const int tx  = threadIdx.x;              // 0..15, column group
    const int ty  = threadIdx.y;              // 0..15, row group
    const int tid = ty * 16 + tx;             // 0..255

    const int64_t m0  = (int64_t) blockIdx.x * GK_CU_MM_TILE_M;  // first weight row
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MM_TILE_N;  // first activation column
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    float acc[GK_CU_MM_TILE_T][GK_CU_MM_TILE_T];
#pragma unroll
    for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
#pragma unroll
        for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < k_len; k0 += GK_CU_MM_TILE_K) {
        // Stage both tiles. The index split puts consecutive threads on
        // consecutive k, which is the direction both operands are contiguous
        // in - a quantized row is packed along k, and a permuted activation
        // still has its k stride in nb[0].
#pragma unroll
        for (int e = tid; e < GK_CU_MM_TILE_M * GK_CU_MM_TILE_K; e += 256) {
            const int     mm = e / GK_CU_MM_TILE_K;
            const int     kk = e % GK_CU_MM_TILE_K;
            const int64_t k  = k0 + kk;
            const int64_t m  = m0 + mm;
            As[kk][mm] = (k < k_len && m < n_rows) ? gk_cu_get_t<ATYPE>(a, k, m, a2, a3) : 0.0f;
        }
#pragma unroll
        for (int e = tid; e < GK_CU_MM_TILE_N * GK_CU_MM_TILE_K; e += 256) {
            const int     nn = e / GK_CU_MM_TILE_K;
            const int     kk = e % GK_CU_MM_TILE_K;
            const int64_t k  = k0 + kk;
            const int64_t n  = n0 + nn;
            Bs[kk][nn] = (k < k_len && n < n_cols) ? gk_cu_get(b, k, n, i2, i3) : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < GK_CU_MM_TILE_K; ++kk) {
            float av[GK_CU_MM_TILE_T];
            float bv[GK_CU_MM_TILE_T];
#pragma unroll
            for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
                av[i] = As[kk][ty * GK_CU_MM_TILE_T + i];
            }
#pragma unroll
            for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
                bv[j] = Bs[kk][tx * GK_CU_MM_TILE_T + j];
            }
#pragma unroll
            for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
#pragma unroll
                for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
        const int64_t m = m0 + ty * GK_CU_MM_TILE_T + i;
        if (m >= n_rows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
            const int64_t n = n0 + tx * GK_CU_MM_TILE_T + j;
            if (n < n_cols) {
                float v = acc[i][j];
                // the fused activation epilogue, as in the pipe tile: this
                // fallback has to honor it too, because it is where a shape
                // lands when the scratch allocation fails mid-flight
                if (act >= 0) {
                    v = gk_cu_unary(act, v, actp.x, actp.y, actp.z, actp.w);
                }
                gk_cu_set(d, m, n, i2, i3, v);
            }
        }
    }
}

// The kernel the last gk_cuda_mul_mat picked. Set on the way into each launch
// rather than derived afterwards from the shape, because the conditions that
// send a shape down a slower path - a failed scratch allocation, a tile that
// would leave the device half idle - are not visible in the shape at all.
static const char * g_gk_mm_path = "-";

double g_gk_mm_quant_ms = 0.0;
double g_gk_mm_tile_ms  = 0.0;

// The fused activation epilogue. The launch loop's fusion plan pairs a gate
// projection with the unary that always follows it; the launcher parks the
// unary here, the dispatch below writes the *unary's* destination with the
// activation applied to the finished accumulator, and the standalone unary
// launch is skipped. Host-side and single-threaded, like the launch loop
// that drives it. `gk_cuda_mm_act_fusable` is the plan's side of the
// contract: it mirrors the dispatch conditions exactly, so a fused matmul
// can only land on a kernel that carries the epilogue - the pipe tiles, or
// the float fallback they degrade to when the scratch allocation fails.
static int                g_gk_mm_act     = -1;
static float4             g_gk_mm_actp    = { 0.0f, 0.0f, 0.0f, 0.0f };
static struct gk_tensor * g_gk_mm_act_dst = NULL;

bool gk_cuda_mm_act_fusable(const struct gk_cuda_scratch * scratch,
                            const struct gk_tensor * mm) {
    return scratch != NULL && mm->src[0] != NULL &&
           (int) mm->src[0]->type == GK_TYPE_Q2_K &&
           mm->src[0]->ne[0] % 32 == 0 &&
           mm->ne[1] >= GK_CU_MM_TILE_MIN_N &&
           gk_cuda_mm_mma_q8_available(scratch) &&
           gk_cu_env_int("GK_MM_MMA_PIPE", 1) != 0;
}

void gk_cuda_mul_mat_act(gkStream_t stream, struct gk_cuda_scratch * scratch,
                         struct gk_tensor * mm, struct gk_tensor * un) {
    g_gk_mm_act  = (int) gk_get_unary_op(un);
    g_gk_mm_actp = make_float4(gk_get_op_params_f32(un, 1), gk_get_op_params_f32(un, 2),
                               gk_get_op_params_f32(un, 3), gk_get_op_params_f32(un, 4));
    g_gk_mm_act_dst = un;
    gk_cuda_mul_mat(stream, scratch, mm);
    g_gk_mm_act     = -1;
    g_gk_mm_act_dst = NULL;
}


// Both diagnostics below are off in every run that is not being investigated,
// so the environment is read once rather than once per matmul.
static bool gk_cu_env_on(const char * name) {
    const char * e = getenv(name);
    return e != NULL && e[0] != '0';
}

static bool gk_cu_mm_dump_on(void) {
    static const bool on = gk_cu_env_on("GK_MM_DUMP");
    return on;
}

static bool gk_cu_mm_split_on(void) {
    static const bool on = gk_cu_env_on("GK_MM_SPLIT");
    return on;
}

// `GK_MM_NVFP4_FP4=1` takes the FP4 tensor-core tile on a device that has the
// instruction; the integer tile is the default. It is also how the two are
// A/B'd, and how a numerical question about fp4 activations gets a same-binary
// control - the two paths quantize the activation side differently, so they are
// the one pair here that does *not* agree bit for bit.
//
// Off by default because that difference is not small and what it buys is.
// The FP4 tile takes both operands as e2m1, so the activation side is 4 bits
// against the integer tile's 8, and measured on MageFlow at 512x512 that is
// 21.7 dB against the integer tile's 52.0 dB - the reference image with visible
// noise in the flat areas, not a different image. What it wins is the matmul
// phase, 212 ms against 325 ms over four steps, which is 1.53x of a part of a
// step and about 6% of the step: 1.69 s of sampling against 1.80 s. Six percent
// is not worth thirty decibels, and the tile stays here, correct and one
// environment variable away, because the instruction is the right one to build
// on once the activation side can be something wider than e2m1.
static bool gk_cu_mm_fp4_stats_on(void) {
    static const bool on = gk_cu_env_on("GK_MM_FP4_STATS");
    return on;
}

static bool gk_cu_mm_fp4_enabled(void) {
    static const int on = gk_cu_env_int("GK_MM_NVFP4_FP4", 0);
    return on != 0;
}

// The FP4 activation counters, read in the translation unit that owns them.
//
// Not read directly from the profile printer, and that is not tidiness: gk
// builds with separable compilation off, so each .cu is its own device module
// and an `extern __device__` in another file does not resolve to these. It
// silently reads a different, permanently zero copy - which is exactly what it
// did before this function existed.
void gk_cuda_fp4_stats(double * sq_err, double * sq_ref,
                       unsigned long long * zero_groups, unsigned long long * groups) {
    *sq_err = 0.0; *sq_ref = 0.0; *zero_groups = 0; *groups = 0;

    if (gkMemcpyFromSymbol(sq_err,      g_gk_fp4_sq_err,      sizeof(*sq_err))      != gkSuccess ||
        gkMemcpyFromSymbol(sq_ref,      g_gk_fp4_sq_ref,      sizeof(*sq_ref))      != gkSuccess ||
        gkMemcpyFromSymbol(zero_groups, g_gk_fp4_zero_groups, sizeof(*zero_groups)) != gkSuccess ||
        gkMemcpyFromSymbol(groups,      g_gk_fp4_groups,      sizeof(*groups))      != gkSuccess) {
        *groups = 0;
    }
}

const char * gk_cuda_mm_last_path(void) {
    return g_gk_mm_path;
}

void gk_cuda_mul_mat(gkStream_t stream, struct gk_cuda_scratch * scratch,
                     struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    const int64_t k_len = src0->ne[0];
    const int64_t r2    = src1->ne[2] / src0->ne[2];
    const int64_t r3    = src1->ne[3] / src0->ne[3];

    const int n_warps = GK_CU_MM_BLOCK / GK_WARP_SIZE;

    // GK_MM_DUMP: the operand geometry behind each distinct matmul shape, once
    // per shape. A shape alone does not say whether an operand is contiguous,
    // how it is strided, or how it broadcasts, and those are exactly the things
    // that separate a matmul that runs at the rate a microbenchmark says it
    // should from the same shape in a real graph.
    if (gk_cu_mm_dump_on()) {
        static char seen[64][128];
        static int  n_seen = 0;

        char key[128];
        snprintf(key, sizeof(key), "%s %lldx%lldx%lld",
                 gk_type_name(src0->type), (long long) dst->ne[0],
                 (long long) dst->ne[1], (long long) k_len);

        bool dup = false;
        for (int i = 0; i < n_seen; ++i) {
            if (strcmp(seen[i], key) == 0) { dup = true; break; }
        }

        if (!dup && n_seen < 64) {
            snprintf(seen[n_seen++], sizeof(seen[0]), "%s", key);
            gk_logf("mm %-28s\n", key);
            const struct gk_tensor * ts[3] = { src0, src1, dst };
            const char * nm[3] = { "src0", "src1", "dst " };
            for (int t = 0; t < 3; ++t) {
                gk_logf("   %s %-6s ne=[%lld %lld %lld %lld] nb=[%zu %zu %zu %zu] cont=%d\n",
                        nm[t], gk_type_name(ts[t]->type),
                        (long long) ts[t]->ne[0], (long long) ts[t]->ne[1],
                        (long long) ts[t]->ne[2], (long long) ts[t]->ne[3],
                        ts[t]->nb[0], ts[t]->nb[1], ts[t]->nb[2], ts[t]->nb[3],
                        (int) gk_is_contiguous(ts[t]));
            }
        }
    }

    // Enough columns for a tile to be mostly real work: reuse across the tile
    // beats splitting k across the block, by a wide margin on a batch.
    if (dst->ne[1] >= GK_CU_MM_TILE_MIN_N) {
        // The tensor-core pilot. nvfp4 only, and only where the instruction
        // exists: `mma.sync` with integer operands is Ampere and later. A
        // 64-element nvfp4 block and a 32-element activation block both have
        // to divide k, so 64 does.
        // The FP4 tensor-core tile, where the hardware has the instruction.
        // Same shapes and the same gate as the integer nvfp4 path below, so
        // this sits in front of it rather than beside it; `GK_MM_NVFP4_FP4=0`
        // puts a run back on the integer tile for comparison.
        if ((int) src0->type == GK_TYPE_NVFP4 && k_len % 64 == 0 &&
            gk_cu_mm_fp4_enabled() && gk_cuda_mm_mma_fp4_available(scratch, stream)) {
            const int64_t n_grp  = k_len / 64;
            const int64_t n_cols = dst->ne[1];
            const int64_t n_23   = dst->ne[2] * dst->ne[3];
            const int64_t n_blk  = n_grp * n_cols * n_23;
            const int64_t n_col2 = n_cols * n_23;

            // One scratch allocation for both: the blocks, then the per-column
            // scales behind them. Two allocations would be two grow paths and
            // the scratch keeps only one buffer.
            const size_t aq_bytes = (size_t) n_blk * sizeof(gk_cu_fp4blk);

            gk_cu_fp4blk * aq = (gk_cu_fp4blk *) gk_cu_scratch_get(
                scratch, aq_bytes + (size_t) n_col2 * sizeof(float), stream);

            float * acs = aq != NULL ? (float *) ((char *) aq + aq_bytes) : NULL;

            if (aq != NULL) {
                const bool split = gk_cu_mm_split_on();
                if (split) { GK_CUDA_CHECK(gkStreamSynchronize(stream)); }
                const std::chrono::steady_clock::time_point q0 =
                    std::chrono::steady_clock::now();

                // One block per column: the column scale is a reduction over
                // the whole column, so a thread that owns only one group of
                // sixty-four cannot compute it.
                gk_cu_k_quantize_act_fp4<<<(unsigned) n_col2, GK_CUDA_BLOCK,
                                           0, stream>>>(
                    gk_cu_view(src1), aq, acs, n_grp, n_cols,
                    gk_cu_mm_fp4_stats_on());

                if (split) {
                    GK_CUDA_CHECK(gkStreamSynchronize(stream));
                    g_gk_mm_quant_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - q0).count();
                }

                const bool xwide = gk_cu_mma_nvfp4_xwide(dst, scratch);
                const bool wide  = !xwide && gk_cu_mma_nvfp4_wide(dst, scratch);

                const int64_t tile_m = xwide ? GK_CU_MMA_TILE_M_OF(GK_CU_MMA_WARPS_M_XWIDE)
                                     : wide  ? GK_CU_MMA_TILE_M_OF(GK_CU_MMA_WARPS_M_WIDE)
                                             : GK_CU_MMA_TILE_M_OF(GK_CU_MMA_WARPS_M_NARROW);
                const int64_t tile_n = xwide ? (int64_t) GK_CU_MMA_WARPS_N_XWIDE * GK_CU_MMA_WN_XWIDE
                                     : wide  ? GK_CU_MMA_TILE_N_OF(GK_CU_MMA_WN_WIDE)
                                             : GK_CU_MMA_TILE_N_OF(GK_CU_MMA_WN_NARROW);

                dim3 mgrid;
                mgrid.x = (unsigned) ((dst->ne[0] + tile_m - 1) / tile_m);
                mgrid.y = (unsigned) ((n_cols     + tile_n - 1) / tile_n);
                mgrid.z = (unsigned) n_23;

                g_gk_mm_path = xwide ? "mma-fp4-256" : wide ? "mma-fp4-128" : "mma-fp4-64";
                const std::chrono::steady_clock::time_point m0 =
                    std::chrono::steady_clock::now();

                if (xwide) {
                    GK_CU_FP4_LAUNCH(GK_CU_MMA_WARPS_M_XWIDE, GK_CU_MMA_WN_XWIDE,
                                     GK_CU_MMA_WARPS_N_XWIDE, GK_CU_MMA_FP4_KSTEP_XWIDE);
                } else if (wide) {
                    GK_CU_FP4_LAUNCH(GK_CU_MMA_WARPS_M_WIDE, GK_CU_MMA_WN_WIDE,
                                     GK_CU_MMA_WARPS_N, GK_CU_MMA_FP4_KSTEP_WIDE);
                } else {
                    GK_CU_FP4_LAUNCH(GK_CU_MMA_WARPS_M_NARROW, GK_CU_MMA_WN_NARROW,
                                     GK_CU_MMA_WARPS_N, 1);
                }

                if (split) {
                    GK_CUDA_CHECK(gkStreamSynchronize(stream));
                    g_gk_mm_tile_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - m0).count();
                }
                return;
            }
        }

        if ((int) src0->type == GK_TYPE_NVFP4 && k_len % 64 == 0 &&
            scratch != NULL && scratch->cc >= 80) {
            const int64_t n_grp  = k_len / 32;
            const int64_t n_cols = dst->ne[1];
            const int64_t n_23   = dst->ne[2] * dst->ne[3];
            const int64_t n_blk  = n_grp * n_cols * n_23;

            gk_cu_q8blk * aq = (gk_cu_q8blk *) gk_cu_scratch_get(
                scratch, (size_t) n_blk * sizeof(gk_cu_q8blk), stream);

            if (aq != NULL) {
                // GK_MM_SPLIT: the quantize pass and the tile timed apart.
                // A profile attributes both launches to the node, and the two
                // have entirely different cures, so a number that mixes them
                // cannot be acted on.
                const bool split = gk_cu_mm_split_on();
                if (split) { GK_CUDA_CHECK(gkStreamSynchronize(stream)); }
                const std::chrono::steady_clock::time_point q0 =
                    std::chrono::steady_clock::now();

                gk_cu_k_quantize_act<<<gk_cu_blocks_1d(n_blk, GK_CUDA_BLOCK),
                                       GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src1), aq, NULL, n_grp, n_cols, n_blk);

                if (split) {
                    GK_CUDA_CHECK(gkStreamSynchronize(stream));
                    g_gk_mm_quant_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - q0).count();
                }

                const bool wide = gk_cu_mma_nvfp4_wide(dst, scratch);

                const int64_t tile_m = wide ? GK_CU_MMA_TILE_M_OF(GK_CU_MMA_WARPS_M_WIDE)
                                            : GK_CU_MMA_TILE_M_OF(GK_CU_MMA_WARPS_M_NARROW);
                const int64_t tile_n = wide ? GK_CU_MMA_TILE_N_OF(GK_CU_MMA_WN_WIDE)
                                            : GK_CU_MMA_TILE_N_OF(GK_CU_MMA_WN_NARROW);

                dim3 mgrid;
                mgrid.x = (unsigned) ((dst->ne[0] + tile_m - 1) / tile_m);
                mgrid.y = (unsigned) ((n_cols     + tile_n - 1) / tile_n);
                mgrid.z = (unsigned) n_23;

                // The width is in the path name because it is the thing most
                // worth knowing about a run of this kernel, and a profile that
                // said only "mma-nvfp4" would hide the heuristic above going
                // the wrong way on a shape.
                g_gk_mm_path = wide ? "mma-nvfp4-128" : "mma-nvfp4-64";
                const std::chrono::steady_clock::time_point m0 =
                    std::chrono::steady_clock::now();

                if (wide) {
                    gk_cu_k_mul_mat_mma_nvfp4<GK_CU_MMA_WARPS_M_WIDE, GK_CU_MMA_WN_WIDE>
                        <<<mgrid, GK_CU_MMA_THREADS(GK_CU_MMA_WARPS_M_WIDE), 0, stream>>>(
                            gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3);
                } else {
                    gk_cu_k_mul_mat_mma_nvfp4<GK_CU_MMA_WARPS_M_NARROW, GK_CU_MMA_WN_NARROW>
                        <<<mgrid, GK_CU_MMA_THREADS(GK_CU_MMA_WARPS_M_NARROW), 0, stream>>>(
                            gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3);
                }

                if (split) {
                    GK_CUDA_CHECK(gkStreamSynchronize(stream));
                    g_gk_mm_tile_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - m0).count();
                }
                return;
            }
        }

        // The f16 tensor-core tile - the convolution path of every diffusion
        // model, and the one shape class where the float tile below is an
        // order of magnitude off what the part can do.
        if (gk_cuda_mma_f16_supported(dst, scratch)) {
            // Rows only pick how many blocks there are, so a shape with too
            // few of them for the wide tile takes the narrow one rather than
            // launching half-empty blocks.
            const int warps_m = dst->ne[0] >= 4 * GK_CU_MMA_F16_WM ? 4 : 2;
            const int tile_m  = warps_m * GK_CU_MMA_F16_WM;

            const int64_t n_23   = dst->ne[2] * dst->ne[3];
            const int64_t n_rows = dst->ne[0];
            const int64_t n_cols = dst->ne[1];

            const int64_t grid_m = (n_rows + tile_m - 1) / tile_m;
            const int64_t grid_n = (n_cols + GK_CU_MMA_F16_TILE_N - 1) / GK_CU_MMA_F16_TILE_N;

            // A tile this wide buys its reuse by making blocks scarce, and an
            // output too small to cover the device leaves multiprocessors
            // standing still no matter how much arithmetic each block does.
            // A UNet's deepest levels are exactly that shape - 64 pixels
            // against 1280 channels is ten blocks on a twenty multiprocessor
            // part - and the only axis left to cut is k, because it is the one
            // dimension the output does not have.
            //
            // What decides it is warps rather than blocks, and the difference
            // is not pedantic: the narrow tile has half the warps of the wide
            // one, so counting blocks says the same thing about a shape that
            // fills the device and one that half-fills it. Measured, 20 blocks
            // of the wide tile want no split and splitting them costs half the
            // throughput, while 10 blocks of the narrow tile - the same block
            // count, half the warps - gain three times over.
            //
            // A split costs a plane of f32 per piece and a pass to add them
            // up, so it is also capped by k being long enough that a piece of
            // it is still a reasonable unit of work.
            const int64_t n_warps = warps_m * GK_CU_MMA_F16_WARPS_N;
            const int64_t want    = (int64_t) scratch->n_sm * GK_CU_MMA_F16_SPLIT_WARPS;
            const int64_t have    = grid_m * grid_n * n_23 * n_warps;

            int64_t n_splits = 1;

            if (have < want) {
                n_splits = (want + have - 1) / have;
                if (n_splits > k_len / GK_CU_MMA_F16_SPLIT_MIN_K) {
                    n_splits = k_len / GK_CU_MMA_F16_SPLIT_MIN_K;
                }
                if (n_splits < 1) {
                    n_splits = 1;
                }
            }

            // Rounded up to the staged k so that every split starts on a
            // boundary the vectorized staging can still read from.
            const int64_t k_split =
                ((k_len + n_splits - 1) / n_splits + GK_CU_MMA_F16_K - 1)
                / GK_CU_MMA_F16_K * GK_CU_MMA_F16_K;

            n_splits = (k_len + k_split - 1) / k_split;

            float * part = NULL;
            if (n_splits > 1) {
                part = (float *) gk_cu_scratch_get(
                    scratch, (size_t) n_splits * n_23 * n_cols * n_rows * sizeof(float), stream);
                if (part == NULL) {
                    n_splits = 1;   // no room; one block per patch still works
                }
            }

            // Still short of warps with k already cut up as far as it goes:
            // the output is simply too small for a tile this wide, and the
            // paths below - narrower tiles, k split across a block rather than
            // across blocks - keep more of the device busy.
            if (have * n_splits >= want / 2) {
                const bool a_vec = gk_cuda_mma_f16_vec(src0, k_len);
                const bool b_vec = gk_cuda_mma_f16_vec(src1, k_len);

                dim3 fgrid;
                fgrid.x = (unsigned) grid_n;   // column tiles: see the kernel
                fgrid.y = (unsigned) grid_m;
                fgrid.z = (unsigned) (n_23 * n_splits);

                // Which accumulator. The graph asks for one - a caller that
                // needs the wide one says so with GK_PREC_F32, the way an
                // attention score matrix does - and the environment can take
                // the choice away in either direction for an A/B.
                const bool acc16 = gk_cuda_mm_acc16(dst);

                g_gk_mm_path = acc16 ? "mma-f16-a16" : "mma-f16";

#define GK_CU_LAUNCH_MMA_F16(WM, A16)                                                       \
                gk_cu_k_mul_mat_mma_f16<WM, A16><<<fgrid, (WM) * GK_CU_MMA_F16_WARPS_N *    \
                                                   GK_WARP_SIZE, 0, stream>>>(              \
                    gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst),                \
                    k_len, r2, r3, a_vec, b_vec,                                            \
                    n_splits > 1 ? part : NULL, k_split, n_23)

                if (warps_m == 4) {
                    if (acc16) { GK_CU_LAUNCH_MMA_F16(4, true); }
                    else       { GK_CU_LAUNCH_MMA_F16(4, false); }
                } else {
                    if (acc16) { GK_CU_LAUNCH_MMA_F16(2, true); }
                    else       { GK_CU_LAUNCH_MMA_F16(2, false); }
                }
#undef GK_CU_LAUNCH_MMA_F16

                if (n_splits > 1) {
                    const int64_t n_out = n_rows * n_cols * n_23;
                    gk_cu_k_mma_f16_combine<<<gk_cu_blocks_1d(n_out, GK_CUDA_BLOCK),
                                              GK_CUDA_BLOCK, 0, stream>>>(
                        part, gk_cu_view_mut(dst), n_splits, n_23, n_rows, n_cols);
                }
                return;
            }
        }

        // The integer tile, where the format has one. Same quantized
        // activations as the mat-vec path, so the same scratch and the same
        // one-off quantize pass; a batch amortizes it even better.
        //
        // Two kernels behind one gate: the tensor-core tile where the
        // instruction exists, the dp4a tile everywhere else. They take the
        // same arguments and compute the same thing, so the choice is purely
        // which multiply the part has.
        if (gk_cuda_mm_q8_supported((int) src0->type, k_len, dst->ne[0]) &&
            scratch != NULL) {
            const int64_t n_grp  = k_len / 32;
            const int64_t n_cols = dst->ne[1];
            const int64_t n_23   = dst->ne[2] * dst->ne[3];
            const int64_t n_blk  = n_grp * n_cols * n_23;

            // Whether the tensor-core tile takes this, decided before the
            // quantize pass because that tile wants the block scalars
            // transposed (see gk_cu_k_quantize_act) and the planes ride the
            // same scratch allocation. `GK_MM_MMA_SPLIT=0` puts the
            // split-scale formats back on the dp4a tile - a bisect lever,
            // not a setting.
            const bool use_mma = gk_cuda_mm_mma_q8_available(scratch) &&
                (!gk_cuda_mm_split_scale((int) src0->type) ||
                 gk_cu_env_int("GK_MM_MMA_SPLIT", 1) != 0);

            const size_t aq_bytes = (size_t) n_blk * sizeof(gk_cu_q8blk);
            const size_t ap_bytes = use_mma ? (size_t) n_blk * 3 * sizeof(float) : 0;

            // A DiT block dots its q, k, v and gate weights against the
            // *same* activation; without the mat-vec path's claim this
            // quantized it four times. Same claim, same rules (the tensor
            // is named as well as the address, the pass and the generation
            // bound it - see gk_cuda_common.cuh), plus one of this path's
            // own: the tile reads the transposed planes, so a claim staked
            // without them (the mat-vec's) is not reusable here.
            const void *   aq_src_prev  = scratch->aq_src;
            const void *   aq_tsr_prev  = scratch->aq_tensor;
            const int64_t  aq_blk_prev  = scratch->aq_blk;
            const int64_t  aq_grp_prev  = scratch->aq_grp;
            const int64_t  aq_pln_prev  = scratch->aq_planes;
            const uint64_t aq_pass_prev = scratch->aq_pass;
            const uint64_t aq_gen_prev  = scratch->gen;

            gk_cu_q8blk * aq = (gk_cu_q8blk *) gk_cu_scratch_get(
                scratch, aq_bytes + ap_bytes, stream);
            float * ap = use_mma && aq != NULL ? (float *) (aq + n_blk) : NULL;

            if (aq != NULL) {
                const bool aq_reuse = aq_tsr_prev == (const void *) src1 &&
                                      aq_src_prev == src1->data &&
                                      aq_blk_prev == n_blk && aq_grp_prev == n_grp &&
                                      aq_pass_prev == scratch->pass &&
                                      aq_gen_prev  == scratch->gen &&
                                      (!use_mma || aq_pln_prev != 0);

                if (!aq_reuse) {
                    gk_cu_k_quantize_act<<<gk_cu_blocks_1d(n_blk, GK_CUDA_BLOCK),
                                           GK_CUDA_BLOCK, 0, stream>>>(
                        gk_cu_view(src1), aq, ap, n_grp, n_cols, n_blk);
                }
                scratch->aq_src    = src1->data;
                scratch->aq_tensor = src1;
                scratch->aq_blk    = n_blk;
                scratch->aq_grp    = n_grp;
                scratch->aq_planes = aq_reuse ? aq_pln_prev : (use_mma ? 1 : 0);
                scratch->aq_pass   = scratch->pass;

                // A whole-group format runs one `mma.sync...k32` per group; a
                // split-scale format runs the group as two k16 windows
                // drained where the scale changes, exactly as the narrow tile
                // does. Both live in the same kernel, so every integer-path
                // format takes the tensor cores when the device has them.
                if (use_mma) {
                    dim3 mgrid;
                    mgrid.x = (unsigned) ((dst->ne[0] + GK_CU_MMAQ_TILE_M - 1) / GK_CU_MMAQ_TILE_M);
                    mgrid.y = (unsigned) ((n_cols     + GK_CU_MMAQ_TILE_N - 1) / GK_CU_MMAQ_TILE_N);
                    mgrid.z = (unsigned) n_23;

                    // q2_K takes the cp.async pipeline tile - the tall
                    // (256-row) shape where the rows fill it, the short one
                    // elsewhere. `GK_MM_MMA_PIPE=0` puts it back on the
                    // synchronous tile and `GK_MM_PIPE_TALL=0` pins the
                    // short shape - bisect levers, not settings.
                    if ((int) src0->type == GK_TYPE_Q2_K &&
                        gk_cu_env_int("GK_MM_MMA_PIPE", 1) != 0) {
                        // Tall wants rows to fill it and loses ~2% on the
                        // deep-k shape (measured: 16384-k ff-down prefers
                        // the short tile's two blocks per SM), so the gate
                        // is per-shape, as every tile width here has been.
                        struct gk_tensor * fdst =
                            g_gk_mm_act_dst != NULL ? g_gk_mm_act_dst : dst;

                        if (dst->ne[0] >= 4096 && n_grp <= 256 &&
                            gk_cu_env_int("GK_MM_PIPE_TALL", 1) != 0) {
                            dim3 tgrid2;
                            tgrid2.x = (unsigned) ((dst->ne[0] + 255) / 256);
                            tgrid2.y = mgrid.y;
                            tgrid2.z = mgrid.z;
                            g_gk_mm_path = "mma-q8pt";
                            gk_cu_k_mul_mat_mma_q2k_pipe<8><<<tgrid2, 512, 0, stream>>>(
                                gk_cu_view(src0), gk_cu_view_mut(fdst), aq, ap, n_grp, r2, r3,
                                g_gk_mm_act, g_gk_mm_actp);
                        } else {
                            g_gk_mm_path = "mma-q8p";
                            gk_cu_k_mul_mat_mma_q2k_pipe<4><<<mgrid, GK_CU_MMAQ_THREADS, 0, stream>>>(
                                gk_cu_view(src0), gk_cu_view_mut(fdst), aq, ap, n_grp, r2, r3,
                                g_gk_mm_act, g_gk_mm_actp);
                        }
                        return;
                    }

#define GK_CU_LAUNCH_MMA_Q8(T)                                                            \
                    gk_cu_k_mul_mat_mma_q8<T><<<mgrid, GK_CU_MMAQ_THREADS, 0, stream>>>(   \
                        gk_cu_view(src0), gk_cu_view_mut(dst), aq, ap, n_grp, r2, r3)

                    g_gk_mm_path = "mma-q8";
                    GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_MMA_Q8);
#undef GK_CU_LAUNCH_MMA_Q8
                    return;
                }

                dim3 qgrid;
                qgrid.x = (unsigned) ((dst->ne[0] + GK_CU_MMQ_TILE_M - 1) / GK_CU_MMQ_TILE_M);
                qgrid.y = (unsigned) ((n_cols     + GK_CU_MMQ_TILE_N - 1) / GK_CU_MMQ_TILE_N);
                qgrid.z = (unsigned) n_23;

#define GK_CU_LAUNCH_TILED_Q8(T)                                           \
                gk_cu_k_mul_mat_tiled_q8<T><<<qgrid, dim3(16, 16), 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)

                g_gk_mm_path = "tile-q8";
                GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_TILED_Q8);
#undef GK_CU_LAUNCH_TILED_Q8
                return;
            }
        }

        dim3 tgrid;
        tgrid.x = (unsigned) ((dst->ne[0] + GK_CU_MM_TILE_M - 1) / GK_CU_MM_TILE_M);
        tgrid.y = (unsigned) ((dst->ne[1] + GK_CU_MM_TILE_N - 1) / GK_CU_MM_TILE_N);
        tgrid.z = (unsigned) (dst->ne[2] * dst->ne[3]);

        struct gk_tensor * tdst = g_gk_mm_act_dst != NULL ? g_gk_mm_act_dst : dst;

#define GK_CU_LAUNCH_TILED(T)                                              \
        gk_cu_k_mul_mat_tiled<T><<<tgrid, dim3(16, 16), 0, stream>>>(      \
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(tdst), k_len, r2, r3, \
            g_gk_mm_act, g_gk_mm_actp)

        g_gk_mm_path = "tile-f32";
        GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_TILED);
#undef GK_CU_LAUNCH_TILED
        return;
    }

    // A warp per output row, so a block of GK_CU_MM_BLOCK threads retires
    // n_warps of them and the grid is that much shorter.
    dim3 grid;
    grid.x = (unsigned) ((dst->ne[0] + n_warps - 1) / n_warps);
    grid.z = (unsigned) (dst->ne[2] * dst->ne[3]);

    // The integer path, where the format has one and the row divides into
    // whole 32-element groups. It quantizes the activations first, which needs
    // somewhere to put them; without that scratch it simply does not run and
    // the float path below gives the same answer more slowly.
    if (gk_cuda_mm_q8_supported((int) src0->type, k_len, dst->ne[0]) && scratch != NULL) {
        const int64_t n_grp  = k_len / 32;
        const int64_t n_cols = dst->ne[1];
        const int64_t n_23   = dst->ne[2] * dst->ne[3];
        const int64_t n_blk  = n_grp * n_cols * n_23;

        // Whether the narrow tensor-core tile takes this shape, decided up
        // front because its k-split planes ride the same scratch allocation
        // as the activation blocks. Zero means the mat-vec keeps it.
        //
        // Off by default: on an Ada part it measured 1.3-3x BEHIND the NC
        // dp4a mat-vec on every verify shape (the mat-vec's lane-per-group
        // walk is coalesced and barrier-free; this tile's staging is not).
        // It exists because the same mat-vec measured issue-bound on
        // Blackwell, where the mma should give back the dot's issue slots -
        // measure there before flipping the default.
        int64_t mman_splits = 0;
        if (gk_cuda_mm_mma_q8_available(scratch) && n_cols >= GK_CU_MMAN_MIN_N &&
            gk_cu_env_int("GK_MM_MMA_NARROW", 0) != 0) {
            const int64_t grid_m = (dst->ne[0] + GK_CU_MMAN_TILE_M - 1) / GK_CU_MMAN_TILE_M;
            const int64_t grid_n = (n_cols + GK_CU_MMAN_TILE_N - 1) / GK_CU_MMAN_TILE_N;
            const int64_t have   = grid_m * grid_n * n_23 * GK_CU_MMAN_WARPS;
            const int64_t want   = (int64_t) scratch->n_sm * 8;

            int64_t ns = 1;
            if (have < want) {
                ns = (want + have - 1) / have;
                const int64_t ns_max = n_grp / GK_CU_MMAN_SPLIT_MIN_GRP;
                if (ns > ns_max) { ns = ns_max; }
                if (ns < 1)      { ns = 1; }
            }

            // Still short of warps with k cut as far as it goes: the mat-vec,
            // which splits k across its block, keeps the device busier.
            if (have * ns >= want / 2) {
                mman_splits = ns;
            }
        }

        const size_t aq_bytes   = (size_t) n_blk * sizeof(gk_cu_q8blk);
        const size_t part_bytes = mman_splits > 1
            ? (size_t) mman_splits * n_23 * n_cols * dst->ne[0] * sizeof(float)
            : 0;

        // Read before scratch_get: it clears the claim on entry. The
        // generation too - a grow moves the buffer, and blocks quantized
        // into the old one are not in the new one whatever the other fields
        // still say.
        const void *   aq_src_prev  = scratch->aq_src;
        const void *   aq_tsr_prev  = scratch->aq_tensor;
        const int64_t  aq_blk_prev  = scratch->aq_blk;
        const int64_t  aq_grp_prev  = scratch->aq_grp;
        const uint64_t aq_pass_prev = scratch->aq_pass;
        const uint64_t aq_gen_prev  = scratch->gen;

        gk_cu_q8blk * aq = (gk_cu_q8blk *) gk_cu_scratch_get(
            scratch, aq_bytes + part_bytes, stream);

        if (aq != NULL) {
            // The q, k and v projections dot different weights against the
            // same activation; if the scratch still holds this exact tensor's
            // blocks from this exact graph pass, the quantize is already done.
            //
            // "This exact tensor" is the tensor, not its address. The graph
            // allocator reuses a dead tensor's storage, so an address that
            // held the activation of an earlier matmul can hold an unrelated
            // result of the same size later in the same pass - and then every
            // other field matches too and the stale blocks are used. See the
            // note on `aq_tensor` in gk_cuda_common.cuh.
            const bool aq_reuse = aq_tsr_prev == (const void *) src1 &&
                                  aq_src_prev == src1->data &&
                                  aq_blk_prev == n_blk && aq_grp_prev == n_grp &&
                                  aq_pass_prev == scratch->pass &&
                                  aq_gen_prev  == scratch->gen;

            if (!aq_reuse) {
                gk_cu_k_quantize_act<<<gk_cu_blocks_1d(n_blk, GK_CUDA_BLOCK),
                                       GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src1), aq, NULL, n_grp, n_cols, n_blk);
            }
            scratch->aq_src    = src1->data;
            scratch->aq_tensor = src1;
            scratch->aq_blk    = n_blk;
            scratch->aq_grp    = n_grp;
            // a reused claim keeps whatever planes it already carried; a
            // fresh mat-vec quantize wrote none
            scratch->aq_planes = aq_reuse ? scratch->aq_planes : 0;
            scratch->aq_pass   = scratch->pass;

            if (mman_splits > 0) {
                float * part = mman_splits > 1
                    ? (float *) ((char *) aq + aq_bytes)
                    : NULL;

                dim3 ngrid;
                ngrid.x = (unsigned) ((dst->ne[0] + GK_CU_MMAN_TILE_M - 1) / GK_CU_MMAN_TILE_M);
                ngrid.y = (unsigned) ((n_cols + GK_CU_MMAN_TILE_N - 1) / GK_CU_MMAN_TILE_N);
                ngrid.z = (unsigned) (n_23 * mman_splits);

#define GK_CU_LAUNCH_MMA_Q8N(T)                                                     \
                gk_cu_k_mul_mat_mma_q8n<T><<<ngrid, GK_CU_MMAN_THREADS, 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3,       \
                    part, mman_splits, n_23)

                g_gk_mm_path = "mma-q8n";
                GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_MMA_Q8N);
#undef GK_CU_LAUNCH_MMA_Q8N

                if (mman_splits > 1) {
                    const int64_t n_out = dst->ne[0] * n_cols * n_23;
                    gk_cu_k_mma_f16_combine<<<gk_cu_blocks_1d(n_out, GK_CUDA_BLOCK),
                                              GK_CUDA_BLOCK, 0, stream>>>(
                        part, gk_cu_view_mut(dst), mman_splits, n_23,
                        dst->ne[0], n_cols);
                }
                return;
            }

            // A warp per row wants at least a few blocks per multiprocessor
            // out of rows alone; an output narrower than that - an attention
            // k/v projection is 256 rows - runs a whole block per row
            // instead, trading a per-row reduction for four times the blocks
            // and a quarter of the dependent-load chain. Only when the row is
            // long enough to feed four warps, though: with fewer groups than
            // threads the reduction is the whole kernel, and measured on a
            // 2048-row, 96-group projection it gave back half the win.
            const bool rowwise = dst->ne[0] < (int64_t) scratch->n_sm * 8 &&
                                 n_grp >= GK_CU_MM_BLOCK;

            // Only a single column takes the one-column kernel. Two or three
            // columns used to as well - grid.y column blocks of NC=1 - which
            // read and decoded the whole weight matrix once *per column*. A
            // speculative verify at draft depth two is exactly three columns,
            // so it paid 3x the traffic of the batched kernel for the same
            // answer; the NC kernel's idle accumulators cost registers, not
            // reads.
            if (n_cols == 1) {
                grid.y = (unsigned) n_cols;

#define GK_CU_LAUNCH_Q8_1(T)                                               \
                gk_cu_k_mul_mat_q8<1, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)
#define GK_CU_LAUNCH_Q8R_1(T)                                              \
                gk_cu_k_mul_mat_q8_row<1, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)

                if (rowwise) {
                    grid.x = (unsigned) dst->ne[0];
                    g_gk_mm_path = "mv-q8r";
                    GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_Q8R_1);
                } else {
                    g_gk_mm_path = "mv-q8";
                    GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_Q8_1);
                }
#undef GK_CU_LAUNCH_Q8_1
#undef GK_CU_LAUNCH_Q8R_1
            } else {
                grid.y = (unsigned) ((n_cols + GK_CU_MM_NC - 1) / GK_CU_MM_NC);

#define GK_CU_LAUNCH_Q8_N(T)                                               \
                gk_cu_k_mul_mat_q8<GK_CU_MM_NC, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)
#define GK_CU_LAUNCH_Q8R_N(T)                                              \
                gk_cu_k_mul_mat_q8_row<GK_CU_MM_NC, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)

                // No rowwise variant here: with four columns per pass the
                // block pays four sequential reductions per row, and at one
                // or two groups per thread that is the whole kernel - the
                // speculative verify measured ~10% slower end to end on it.
                GK_UNUSED(rowwise);
                g_gk_mm_path = "mv-q8";
                GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_Q8_N);
#undef GK_CU_LAUNCH_Q8_N
#undef GK_CU_LAUNCH_Q8R_N
            }
            return;
        }
    }

    // One column at a time when there is only one - the decode case, every
    // token of generation - and blocks of NC for anything more: the NC
    // kernel's column guard makes a partial block correct, and its idle
    // accumulators are cheaper than reading the weights once per column,
    // which is what NC=1 with a column grid did to a 2-3 column verify.
    if (dst->ne[1] == 1) {
        grid.y = (unsigned) dst->ne[1];

#define GK_CU_LAUNCH_MV1(T)                                                \
        gk_cu_k_mul_mat<1, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>(            \
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3)

        g_gk_mm_path = "mv-f32";
        GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_MV1);
#undef GK_CU_LAUNCH_MV1
    } else {
        grid.y = (unsigned) ((dst->ne[1] + GK_CU_MM_NC - 1) / GK_CU_MM_NC);

#define GK_CU_LAUNCH_MVN(T)                                                \
        gk_cu_k_mul_mat<GK_CU_MM_NC, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>(  \
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3)

        g_gk_mm_path = "mv-f32";
        GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_MVN);
#undef GK_CU_LAUNCH_MVN
    }
}

// --------------------------------------------------------------------------
// mixture of experts
//
// `ids` holds, per token, which experts it routes to; every (token, slot) pair
// picks one expert's rows out of `as`. One block per output element again, but
// the expert index is read per block rather than per element.
// --------------------------------------------------------------------------

template <int ATYPE>
static __global__ void gk_cu_k_mul_mat_id(gk_tview as, gk_tview b, gk_tview ids,
                                          gk_tview_mut d, int64_t k_len) {
    extern __shared__ float shared[];

    const int64_t i0 = blockIdx.x; // row within the expert
    const int64_t is = blockIdx.y; // which of the token's expert slots
    const int64_t it = blockIdx.z; // which token

    const int32_t expert = *(const int32_t *) (ids.data + it * ids.nb[1] + is * ids.nb[0]);

    float acc[1] = { 0.0f };

    for (int64_t kk = threadIdx.x; kk < k_len; kk += blockDim.x) {
        acc[0] += gk_cu_get_t<ATYPE>(as, kk, i0, expert, 0) * gk_cu_get(b, kk, 0, it, 0);
    }

    gk_cu_reduce_n<1>(acc, shared);

    if (threadIdx.x == 0) {
        gk_cu_set(d, i0, is, it, 0, acc[0]);
    }
}

void gk_cuda_mul_mat_id(gkStream_t stream, struct gk_tensor * dst) {
    const struct gk_tensor * as  = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];
    const struct gk_tensor * ids = dst->src[2];

    dim3 grid;
    grid.x = (unsigned) dst->ne[0];
    grid.y = (unsigned) dst->ne[1];
    grid.z = (unsigned) dst->ne[2];

    const int n_warps = GK_CU_MM_BLOCK / GK_WARP_SIZE;

#define GK_CU_LAUNCH_ID(T)                                                 \
    gk_cu_k_mul_mat_id<T><<<grid, GK_CU_MM_BLOCK,                          \
                            n_warps * sizeof(float), stream>>>(            \
        gk_cu_view(as), gk_cu_view(src1), gk_cu_view(ids), gk_cu_view_mut(dst), as->ne[0])

    GK_CU_MM_DISPATCH((int) as->type, GK_CU_LAUNCH_ID);
#undef GK_CU_LAUNCH_ID
}

// --------------------------------------------------------------------------
// fused attention
//
// One pass over the keys with an online softmax: keep the running maximum M
// and normaliser S, and rescale the value accumulator whenever the maximum
// moves. The accumulator is f32 whatever the value type is, which is what the
// CPU pass does and what keeps a graph split across devices from drifting.
//
// The query row and the value accumulator live in shared memory, which is what
// bounds the head sizes this kernel accepts; larger ones fall back to the CPU
// rather than being silently truncated.
//
// There are two ways to spread that pass over the device, and which one is
// right depends entirely on the batch.
//
// A prompt pass has a query row per token per head - thousands of them - so
// one block each already fills the card, and each block walking the whole
// cache is the cheapest thing to do: the query row is read once and the cache
// is read once.
//
// Generation has one token. Eight heads, one query row each, is eight blocks,
// and a card with twenty multiprocessors runs them in a single wave with most
// of itself idle - and then that handful of blocks walks the entire cache one
// position at a time. Measured on a 4050, that path ran at 0.8 GB/s and lost
// to the CPU by a factor of three.
//
// So when the row count alone will not fill the device, the cache is cut into
// slices and a block takes one. Each produces a partial result: its own
// accumulator, its own M, its own S, each relative to the slice it saw. A
// second pass merges them, which is possible because the online softmax
// composes - a partial from a slice is exactly what that slice would have
// contributed had it been seen first, and rescaling it to a common maximum is
// the same arithmetic the single-block path already does whenever its running
// maximum moves.
//
// Both paths run the same accumulation loop, below, told apart only by whether
// it was handed somewhere to put partials.
// --------------------------------------------------------------------------

#define GK_CU_FA_BLOCK   128

// Splitting only pays once there is enough cache to divide; below this the
// launch and the merge cost more than the parallelism is worth.
#define GK_CU_FA_MIN_SPLIT_KV 256

// What the split is aiming at: enough blocks to give every multiprocessor
// several, and slices short enough that no block is left walking a long tail
// alone. Both are targets rather than limits - the clamps below decide.
#define GK_CU_FA_BLOCKS_PER_SM  8
#define GK_CU_FA_TARGET_SLICE 256

// A slice shorter than this is mostly launch overhead, and more splits than
// this stop paying for the merge they add.
#define GK_CU_FA_MIN_SLICE  64
#define GK_CU_FA_MAX_SPLIT  64

// --------------------------------------------------------------------------
// The tiled path: a block of query rows against the cache a tile at a time.
//
// The kernel below walks the cache one position per iteration, and for a
// generation batch that is fine - the cache is short and the split across
// blocks covers it. For a diffusion transformer it is ruinous. Every token
// attends to every other, so a 4096-token layer has 65536 query rows and each
// one of them re-reads the whole of K and V: about 64 GB of requests per
// attention layer, and a block reduction with two barriers at every one of the
// 4096 steps. Measured, 1.15 seconds for a single layer.
//
// Two changes fix it, and they are the two FlashAttention makes:
//
//   * A block owns several query rows, not one, and stages each K/V tile into
//     shared memory once for all of them. That is where the factor of eight in
//     memory traffic comes from.
//
//   * A lane owns a whole key position rather than a slice of one. The dot
//     product for that key is then entirely within the lane - no reduction at
//     all - and the only cross-lane work left is the softmax's maximum and
//     sum, which happen once per tile of 32 keys instead of once per key.
//
// The output accumulator is distributed the other way, a lane per value
// dimension, so the probabilities have to cross between the two layouts. That
// is what the small per-warp `Ps` array is for.
// --------------------------------------------------------------------------

#define GK_CU_FAT_WARPS   8    // warps per block
#define GK_CU_FAT_QR      4    // query rows per warp
#define GK_CU_FAT_QROWS   (GK_CU_FAT_WARPS * GK_CU_FAT_QR)
#define GK_CU_FAT_BC      32   // key positions per tile, one per lane
#define GK_CU_FAT_MAX_D   128  // head width the shared tiles are budgeted for
#define GK_CU_FAT_DV_LANE (GK_CU_FAT_MAX_D / GK_WARP_SIZE)

static __global__ void gk_cu_k_flash_attn_tiled(gk_tview q, gk_tview k, gk_tview v,
                                                gk_tview mask, bool has_mask,
                                                const float * sinks, gk_tview_mut d,
                                                float scale, float max_bias,
                                                float logit_softcap, int64_t n_head_log2,
                                                int64_t rk2, int64_t rk3,
                                                int64_t rv2, int64_t rv3) {
    extern __shared__ float fat_smem[];

    const int64_t DK   = k.ne[0];
    const int64_t DV   = v.ne[0];
    const int64_t n_kv = k.ne[1];

    // Ks and Vs are padded by one so that a lane walking a key's row and a lane
    // walking a value's column both stride across banks rather than into one.
    float * Ks = fat_smem;
    float * Vs = Ks + GK_CU_FAT_BC * (DK + 1);
    float * Qs = Vs + GK_CU_FAT_BC * (DV + 1);
    float * Ps = Qs + GK_CU_FAT_QROWS * DK;

    const int lane = threadIdx.x % GK_WARP_SIZE;
    const int warp = threadIdx.x / GK_WARP_SIZE;

    const int64_t q1_0 = (int64_t) blockIdx.x * GK_CU_FAT_QROWS + warp * GK_CU_FAT_QR;
    const int64_t iq2  = blockIdx.y;
    const int64_t iq3  = blockIdx.z;

    const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
    const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

    const float slope = gk_cu_alibi_slope(max_bias, iq2, n_head_log2);

    bool live[GK_CU_FAT_QR];
#pragma unroll
    for (int r = 0; r < GK_CU_FAT_QR; ++r) {
        live[r] = q1_0 + r < q.ne[1];
        if (live[r]) {
            for (int64_t i = lane; i < DK; i += GK_WARP_SIZE) {
                Qs[(warp * GK_CU_FAT_QR + r) * DK + i] = gk_cu_get(q, i, q1_0 + r, iq2, iq3);
            }
        }
    }

    float M[GK_CU_FAT_QR];
    float S[GK_CU_FAT_QR];
    float O[GK_CU_FAT_QR][GK_CU_FAT_DV_LANE];
#pragma unroll
    for (int r = 0; r < GK_CU_FAT_QR; ++r) {
        M[r] = -INFINITY;
        S[r] = 0.0f;
#pragma unroll
        for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
            O[r][t] = 0.0f;
        }
    }

    for (int64_t c0 = 0; c0 < n_kv; c0 += GK_CU_FAT_BC) {
        // The whole block stages the tile and every warp reads it back. Four
        // query rows per warp is what makes this pay: the tile is staged once
        // for thirty-two of them, and each staged element is then used four
        // times in the arithmetic below rather than once.
        for (int64_t e = threadIdx.x; e < GK_CU_FAT_BC * DK; e += blockDim.x) {
            const int64_t c = e / DK, i = e % DK;
            Ks[c * (DK + 1) + i] = c0 + c < n_kv ? gk_cu_get(k, i, c0 + c, ik2, ik3) : 0.0f;
        }
        for (int64_t e = threadIdx.x; e < GK_CU_FAT_BC * DV; e += blockDim.x) {
            const int64_t c = e / DV, i = e % DV;
            Vs[c * (DV + 1) + i] = c0 + c < n_kv ? gk_cu_get(v, i, c0 + c, iv2, iv3) : 0.0f;
        }
        __syncthreads();

        // This lane's key. The dot products never leave the lane, and the key
        // is read from shared once for all four query rows.
        const int64_t ic = c0 + lane;

        float sc[GK_CU_FAT_QR];
#pragma unroll
        for (int r = 0; r < GK_CU_FAT_QR; ++r) {
            sc[r] = 0.0f;
        }

        if (ic < n_kv) {
            for (int64_t i = 0; i < DK; ++i) {
                const float kv = Ks[lane * (DK + 1) + i];
#pragma unroll
                for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                    sc[r] += Qs[(warp * GK_CU_FAT_QR + r) * DK + i] * kv;
                }
            }
        }

        float corr[GK_CU_FAT_QR];

#pragma unroll
        for (int r = 0; r < GK_CU_FAT_QR; ++r) {
            float s = -INFINITY;

            if (ic < n_kv && live[r]) {
                float mv = 0.0f;
                bool  skip = false;
                if (has_mask) {
                    mv = slope * gk_cu_get(mask, ic, q1_0 + r,
                                           iq2 % mask.ne[2], iq3 % mask.ne[3]);
                    skip = mv == -INFINITY;
                }
                if (!skip) {
                    s = sc[r] * scale;
                    if (logit_softcap != 0.0f) {
                        s = logit_softcap * tanhf(s);
                    }
                    s += mv;
                }
            }

            // The online softmax, once for the tile rather than once per key.
            const float m_tile = gk_cu_warp_max(s);
            const float m_new  = fmaxf(M[r], m_tile);

            // A tile that was entirely masked leaves the running maximum where
            // it was; expf(-inf - -inf) is not a number, so it is not asked.
            corr[r] = m_new == -INFINITY ? 1.0f : expf(M[r] - m_new);
            const float p = s == -INFINITY || m_new == -INFINITY
                          ? 0.0f : expf(s - m_new);

            S[r] = S[r] * corr[r] + gk_cu_warp_sum(p);
            M[r] = m_new;

            // The probabilities are held one per key, the accumulator one per
            // value dimension; this is where the two layouts meet.
            Ps[(warp * GK_CU_FAT_QR + r) * GK_CU_FAT_BC + lane] = p;
        }
        __syncwarp();

#pragma unroll
        for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
            const int64_t dd = lane + (int64_t) t * GK_WARP_SIZE;
            if (dd >= DV) {
                continue;
            }

            float a[GK_CU_FAT_QR];
#pragma unroll
            for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                a[r] = 0.0f;
            }

            // One value read serves all four rows, which is the other half of
            // what the four-row warp buys.
            for (int c = 0; c < GK_CU_FAT_BC; ++c) {
                const float vv = Vs[c * (DV + 1) + dd];
#pragma unroll
                for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                    a[r] += Ps[(warp * GK_CU_FAT_QR + r) * GK_CU_FAT_BC + c] * vv;
                }
            }

#pragma unroll
            for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                O[r][t] = O[r][t] * corr[r] + a[r];
            }
        }
        __syncwarp();

        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < GK_CU_FAT_QR; ++r) {
        if (!live[r]) {
            continue;
        }

        // the sink is one more virtual position, with a logit but no value row
        if (sinks != NULL) {
            const float sv = sinks[iq2];
            if (sv > M[r]) {
                const float c = M[r] == -INFINITY ? 0.0f : expf(M[r] - sv);
                M[r] = sv;
                S[r] = S[r] * c + 1.0f;
#pragma unroll
                for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
                    O[r][t] *= c;
                }
            } else {
                S[r] += expf(sv - M[r]);
            }
        }

        const float inv = S[r] == 0.0f ? 0.0f : 1.0f / S[r];

#pragma unroll
        for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
            const int64_t dd = lane + (int64_t) t * GK_WARP_SIZE;
            if (dd < DV) {
                gk_cu_set(d, dd, iq2, q1_0 + r, iq3, O[r][t] * inv);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The tensor-core path.
//
// The kernel above is the same algorithm as this one and runs at about a
// tenth of the speed, for a reason that is not the algorithm. Its two inner
// loops each read five values out of shared memory to feed four FFMAs - one
// K element and four Q elements, then one V element and four probabilities -
// and shared memory issues one warp-wide load per multiprocessor-cycle where
// the FFMA pipe issues four. So five loads cost five cycles to feed one
// cycle of arithmetic, and the kernel spends five sixths of itself waiting on
// the memory it is reading operands out of. Measured at the shape a 64x64
// UNet layer runs - 4096 queries against 4096 keys, eight heads, d_head 40 -
// that is 934 GFLOP/s where the same card's f16 GEMM does 12 to 16 TFLOP/s.
//
// The fix is to stop moving operands one element at a time. `mma.sync`
// consumes a whole 16x16-by-16x8 tile per instruction from registers, so the
// ratio inverts: a warp reads its operands once into fragments and gets 2048
// multiply-accumulates out of them.
//
// Three things make the mapping work out better here than a GEMM's:
//
//   * Q is read once, into registers, and stays there for the whole cache.
//     It is the operand a query block reuses across every key tile, so it
//     never belongs in shared memory at all - which is also what frees the
//     room for K and V to be staged at a 64-key tile instead of 32.
//
//   * The S fragment that comes out of the first mma is bit-for-bit the A
//     fragment the second one wants. `mma.m16n8k16` leaves a lane holding D
//     rows `group`/`group+8` at columns `2*tig`/`2*tig+1`, and wants A rows
//     `group`/`group+8` at columns `2*tig`/`2*tig+1` of each 16-wide window.
//     Those are the same elements in the same lanes, so the probabilities go
//     from the softmax straight into the next instruction as registers. No
//     staging, no barrier, and no shared memory for P.
//
//   * The softmax's row reductions shrink from a warp to a quadrant. A row of
//     S lives in the four lanes that share a `group`, so the maximum and the
//     sum are two shuffles over the low two lane bits instead of the five a
//     whole-warp reduction costs.
//
// What the tile costs in exchange is that the products are f16. The
// accumulator is f32 across the whole reduction, and K and V arrive f16 from
// the caller already, so what is new is Q and the probabilities being rounded
// before they are multiplied - the same trade the f16 GEMM makes, and the
// reason the host gate below insists on f16 K and V rather than converting
// an f32 cache down to reach this path.
// --------------------------------------------------------------------------

// Warps per block, and with it the query rows a block owns.
//
// This is the one number that decides how many times the cache is read. A
// block walks the whole of K and V whatever its height, so a head's cache is
// re-read once per query block: at 64 rows a 8742-token layer re-reads it 137
// times, which for d=128 h=56 is 34.6 GB of requests against 4.5 MB of actual
// K and V. Measured there at 1.11 TB/s of a 1.79 TB/s part - the same disease
// the quantized GEMM had, found the same way, by counting the re-reads rather
// than the operands.
//
// Doubling it is close to free, which is why it is the first thing to try:
// every warp owns its own 16 query rows with its own Q fragments, accumulator
// and running softmax, all in registers, and the only thing warps share is the
// staged K/V tile and the barrier protecting it. So twice the warps means
// twice the arithmetic per staged tile, at identical per-warp cost.
//
// But that argument is about *traffic*, and it only binds when the re-reads
// reach DRAM. A short cache - a 1024-pixel image's 1056 tokens is half a
// megabyte of K and V per head - sits in L2 however many blocks walk it, and
// there the same registers are better spent the other way: two blocks of
// four warps per multiprocessor instead of one of eight. Same warps, but two
// independent softmax chains whose barriers only wait on half as much, where
// the single wide block serializes every tile behind one barrier. So the
// warp count is a template parameter and the host picks by cache size:
// 4 warps when a head's K+V fits comfortably in L2, 8 when it cannot.
#define GK_CU_FAM_WARPS   8                                  // warps per block, long caches
#define GK_CU_FAM_BR(W)      ((W) * 16)                      // query rows a block owns
#define GK_CU_FAM_THREADS(W) ((W) * GK_WARP_SIZE)

// Key positions per tile. A wide head already spends its registers on the
// output accumulator - a lane holds DV/2 floats of it - and the S tile is
// what it can give back: halving the tile halves S and the probability
// fragments both, and halves the staged K and V with them, which is the
// difference between two blocks resident per multiprocessor and five.
// Narrow heads have the registers to spare and would rather sync half as
// often.
#define GK_CU_FAM_BC(D)   ((D) <= 80 ? 64 : 32)
#define GK_CU_FAM_NTC(D)  (GK_CU_FAM_BC(D) /  8)             // 8-wide S tiles a warp holds
#define GK_CU_FAM_NKW(D)  (GK_CU_FAM_BC(D) / 16)             // 16-wide PV windows

// The head width is a template parameter because the accumulator it sizes is
// registers, which have to be counted at compile time. The buckets are the
// widths that exist: 40 and 80 and 160 are SD's three levels, 64 and 128 are
// what a language model and a diffusion transformer run. A head is rounded up
// to the next bucket and the slack is zero-padded, so 40 costs a 48-wide
// reduction and nothing else.
#define GK_CU_FAM_NTK(D)  (((D) + 15) / 16)                  // 16-wide k windows
#define GK_CU_FAM_NTV(D)  (((D) +  7) /  8)                  // 8-wide value tiles

// Staged rows are padded past the width they hold, for the same reason the
// GEMM's are: a fragment read puts the eight lanes of a quadrant on eight
// different rows at one k, and eight rows an exact power of two apart would
// land on one bank. Every bucket's padded stride is a multiple of 8 halves
// past a multiple of 16, which spreads those eight over eight disjoint quads.
#define GK_CU_FAM_SK(D)   (GK_CU_FAM_NTK(D) * 16 + 8)
#define GK_CU_FAM_SC(D)   (GK_CU_FAM_BC(D) + 8)

// Two floats into the half2 an mma fragment register is. The low half is the
// lower k, which is the order both operands are read out of shared memory in.
static __device__ __forceinline__ int gk_cu_pack2_half(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const int *) &h;
}

// Eight f16 out of a row, as one 16-byte word, with the tail of a head that is
// not a multiple of eight zero-filled.
//
// Staging is why this matters. A tile is 64 keys of K and 64 of V, so a block
// moves several thousand elements into shared memory per iteration and issues
// a few hundred mma against them - which means an instruction spent per staged
// element costs several times what the arithmetic does, and the kernel becomes
// a memcpy with a tensor core attached. Whether the eight can be taken as one
// word is decided on the host, once per launch.
template <bool VEC>
static __device__ __forceinline__ int4 gk_cu_fam_run8(const gk_tview & t,
                                                      int64_t i0, int64_t i1,
                                                      int64_t i2, int64_t i3,
                                                      int64_t n0) {
    const char * row = gk_cu_row(t, i1, i2, i3);

    if (VEC && i0 + 8 <= n0) {
        return *(const int4 *) (row + i0 * 2);
    }

    __align__(16) __half h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        h[j] = i0 + j < n0 ? *(const __half *) (row + (i0 + j) * t.nb[0])
                           : __float2half(0.0f);
    }
    return *(const int4 *) h;
}

// One Q element, for the three float types Q can actually arrive as.
//
// Same reason as the mask below: `gk_cu_get` resolves the type at run time
// across every format gk supports, and inlining that switch into the 32
// unrolled reads that fill a warp's Q fragments is a jump table each. Three
// arms of one load each is what this is instead. The host gate restricts Q to
// these types so the fourth case cannot happen.
static __device__ __forceinline__ float gk_cu_fam_qval(const gk_tview & q,
                                                       int64_t k0, int64_t r,
                                                       int64_t i2, int64_t i3) {
    const char * p = gk_cu_row(q, r, i2, i3) + k0 * q.nb[0];

    if (q.type == GKT_F32) {
        return *(const float *) p;
    }
    if (q.type == GKT_F16) {
        return __half2float(*(const __half *) p);
    }
    return gk_cu_bf2f(*(const uint16_t *) p);
}

// One mask element, read as the f16 it is required to be.
//
// Not `gk_cu_get`. That resolves the tensor's type at run time, and a type
// switch is a jump table - which, inlined into a loop that reads sixteen mask
// elements per key tile, is most of what this kernel's inner loop compiles to.
// The host gate below requires an f16 mask precisely so that this can be a
// load.
static __device__ __forceinline__ float gk_cu_fam_mask(const gk_tview & m,
                                                       int64_t ic, int64_t r,
                                                       int64_t i2, int64_t i3) {
    return __half2float(*(const __half *) (gk_cu_row(m, r, i2, i3) + ic * m.nb[0]));
}

template <int D_PAD, bool VEC, bool HAS_MASK, int WARPS>
static __global__ __launch_bounds__(GK_CU_FAM_THREADS(WARPS))
void gk_cu_k_flash_attn_mma(gk_tview q, gk_tview k, gk_tview v,
                            gk_tview mask,
                            const float * sinks, gk_tview_mut d,
                            float scale, float max_bias,
                            float logit_softcap, int64_t n_head_log2,
                            int64_t rk2, int64_t rk3,
                            int64_t rv2, int64_t rv3) {
    const int NT_K = GK_CU_FAM_NTK(D_PAD);
    const int NT_V = GK_CU_FAM_NTV(D_PAD);

    // Two of each: the tile being computed and the one being fetched. See the
    // loop below for why. Shared memory is what pays for it, and shared memory
    // is the one resource this kernel has to spare - it is registers that cap
    // its occupancy.
    // 16-byte aligned because the fragment loads below are `ldmatrix`, which
    // requires that of every address it is handed; both padded strides are
    // whole numbers of 16-byte words already.
    __align__(16) __shared__ __half Ks[2][GK_CU_FAM_BC(D_PAD)]     [GK_CU_FAM_SK(D_PAD)];
    __align__(16) __shared__ __half Vt[2][GK_CU_FAM_NTV(D_PAD)*8][GK_CU_FAM_SC(D_PAD)];

    const int64_t DK   = k.ne[0];
    const int64_t DV   = v.ne[0];
    const int64_t n_kv = k.ne[1];
    const int64_t n_q  = q.ne[1];

    const int tid   = (int) threadIdx.x;
    const int lane  = tid % GK_WARP_SIZE;
    const int warp  = tid / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int64_t iq2 = blockIdx.y;
    const int64_t iq3 = blockIdx.z;

    const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
    const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

    const float slope = gk_cu_alibi_slope(max_bias, iq2, n_head_log2);

    // The two query rows this lane owns, in the warp's 16-row tile.
    const int64_t q1_0 = (int64_t) blockIdx.x * GK_CU_FAM_BR(WARPS) + warp * 16;
    const int64_t row0 = q1_0 + group;
    const int64_t row1 = q1_0 + group + 8;

    const bool live0 = row0 < n_q;
    const bool live1 = row1 < n_q;

    // Q, once, as fragments. Rows past the end and k past the head read zero,
    // which contributes nothing to a dot product - so the ragged tail needs no
    // branch anywhere below this point.
    // The logit scale rides in with Q: `(sQ)K = s(QK)`, so multiplying the
    // fragments once here deletes the per-element multiply the softmax loop
    // paid on every tile - sixteen FMULs a lane a tile. The softcap still
    // sees `s * scale` exactly as before; only where the multiply happens
    // moved.
    int qf[GK_CU_FAM_NTK(D_PAD)][4];
#pragma unroll
    for (int kt = 0; kt < NT_K; ++kt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t r  = (j & 1) ? row1 : row0;
            const int64_t k0 = kt * 16 + (j >= 2 ? 8 : 0) + 2 * tig;

            const float a = r < n_q && k0     < DK ? gk_cu_fam_qval(q, k0,     r, iq2, iq3) : 0.0f;
            const float b = r < n_q && k0 + 1 < DK ? gk_cu_fam_qval(q, k0 + 1, r, iq2, iq3) : 0.0f;

            qf[kt][j] = gk_cu_pack2_half(a * scale, b * scale);
        }
    }

    float acc[GK_CU_FAM_NTV(D_PAD)][4];
#pragma unroll
    for (int ct = 0; ct < NT_V; ++ct) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[ct][i] = 0.0f;
        }
    }

    // The running softmax, one pair per lane: row0 in slot 0, row1 in slot 1.
    // Both are replicated across the four lanes of a quadrant, because that is
    // what the reductions below leave behind.
    float M[2] = { -INFINITY, -INFINITY };
    float S[2] = { 0.0f, 0.0f };

    // One key tile, fetched and committed as two separate steps.
    //
    // K as [key][k] and V transposed to [value dim][key], which is the layout
    // `mma.row.col` wants for the B operand of each of the two products. V's
    // transpose is paid here, on the staging store, rather than in the inner
    // loop - the read side of it is what runs 64 times per tile in the
    // arithmetic below, and the write side once.
    //
    // Both are staged eight elements to the instruction. The divisors are
    // template constants, so the index arithmetic is shifts rather than the
    // 64-bit division a runtime divisor would compile to.
    //
    // Fetch and commit are split because a store that follows its load
    // consumes the whole global latency on the spot - profiled here, the
    // scoreboard wait on those loads was 30% of every stall cycle in the
    // kernel, double buffer or no, because the buffer only moved the barrier
    // and not the dependency. Fetching into registers before the tile's
    // arithmetic and committing to shared after it puts the whole mma block
    // inside the load shadow instead.
    constexpr int K_IT = (GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTK(D_PAD) * 2
                          + GK_CU_FAM_THREADS(WARPS) - 1) / GK_CU_FAM_THREADS(WARPS);
    constexpr int V_IT = (GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTV(D_PAD)
                          + GK_CU_FAM_THREADS(WARPS) - 1) / GK_CU_FAM_THREADS(WARPS);

    int4 kreg[K_IT];
    int4 vreg[V_IT];

    auto fetch_tile = [&](int64_t c_base) {
#pragma unroll
        for (int it = 0; it < K_IT; ++it) {
            const int e = tid + it * GK_CU_FAM_THREADS(WARPS);
            if (e < GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTK(D_PAD) * 2) {
                const int     c  = e / (NT_K * 2);
                const int     i  = (e - c * (NT_K * 2)) * 8;
                const int64_t ic = c_base + c;

                kreg[it] = ic < n_kv ? gk_cu_fam_run8<VEC>(k, i, ic, ik2, ik3, DK)
                                     : make_int4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int it = 0; it < V_IT; ++it) {
            const int e = tid + it * GK_CU_FAM_THREADS(WARPS);
            if (e < GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTV(D_PAD)) {
                const int     c  = e / NT_V;
                const int     i  = (e - c * NT_V) * 8;
                const int64_t ic = c_base + c;

                vreg[it] = ic < n_kv ? gk_cu_fam_run8<VEC>(v, i, ic, iv2, iv3, DV)
                                     : make_int4(0, 0, 0, 0);
            }
        }
    };

    auto commit_tile = [&](int buf) {
#pragma unroll
        for (int it = 0; it < K_IT; ++it) {
            const int e = tid + it * GK_CU_FAM_THREADS(WARPS);
            if (e < GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTK(D_PAD) * 2) {
                const int c = e / (NT_K * 2);
                const int i = (e - c * (NT_K * 2)) * 8;

                *(int4 *) &Ks[buf][c][i] = kreg[it];
            }
        }
#pragma unroll
        for (int it = 0; it < V_IT; ++it) {
            const int e = tid + it * GK_CU_FAM_THREADS(WARPS);
            if (e >= GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTV(D_PAD)) {
                continue;
            }
            const int  c = e / NT_V;
            const int  i = (e - c * NT_V) * 8;
            const int4 w = vreg[it];

            // The one scattered write in the kernel: eight value dimensions of
            // one key, which land a row apart in the transposed tile.
            //
            // Rotated by the thread's dim-block index, because in address
            // order it is the worst store this kernel can make: lanes on the
            // same key column write rows eight apart, the padded stride is a
            // multiple of four words, and eight rows times any multiple of
            // four words is zero mod 32 - every lane of a half-warp on one
            // bank, a measured 14.8-way conflict that was 91% of the kernel's
            // shared-store wavefronts. The stride cannot fix it: `ldmatrix`
            // needs rows 16-byte aligned, which is exactly what pins them to
            // one bank. Rotating the *order* changes no address, but puts
            // eight consecutive lanes on eight distinct banks per step.
            const int      t = e - c * NT_V;
            const __half * h = (const __half *) &w;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int j = (jj + t) & 7;
                Vt[buf][i + j][c] = h[j];
            }
        }
    };

    // Software pipeline, two-deep: while tile g is computed on out of one
    // buffer, tile g+1 is already fetched into registers, and its commit into
    // the other buffer happens after the arithmetic - so both the global
    // latency of the fetch and the shared stores of the commit sit inside
    // the mma work, and the one barrier per iteration covers the commit
    // becoming visible together with the reads having finished.
    fetch_tile(0);
    commit_tile(0);
    __syncthreads();

    int buf = 0;

    for (int64_t c0 = 0; c0 < n_kv; c0 += GK_CU_FAM_BC(D_PAD), buf ^= 1) {
        const int64_t c_next  = c0 + GK_CU_FAM_BC(D_PAD);
        const bool    prefetch = c_next < n_kv;
        if (prefetch) {
            fetch_tile(c_next);
        }

        // S = Q K^T, for this warp's 16 query rows against all 64 keys.
        float s[GK_CU_FAM_NTC(D_PAD)][4];
#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[nt][i] = 0.0f;
            }
        }
        // Fragments come out of shared by `ldmatrix.x4` - one instruction for
        // the four 8x8 pieces two S tiles want, where the indexed loads this
        // loop used to make were four per mma against one mma of arithmetic,
        // and the load-store pipe was the pipe this kernel starved on. Lanes
        // 0-7 address the low-k rows of the first tile, 8-15 the high-k, and
        // the upper half of the warp the second tile; the fragment each lane
        // ends up holding is byte-identical to what the indexed loads gave it.
        const int ldm_r = (lane / 16) * 8 + (lane % 8);   // row within a 16-row pair
        const int ldm_c = ((lane / 8) & 1) * 8;           // low or high k half

#pragma unroll
        for (int kt = 0; kt < NT_K; ++kt) {
#pragma unroll
            for (int np = 0; np < GK_CU_FAM_NTC(D_PAD) / 2; ++np) {
                int kf[4];
                gk_cu_ldmatrix_x4(kf,
                    (const int *) &Ks[buf][np * 16 + ldm_r][kt * 16 + ldm_c]);

                gk_cu_mma_f16(s[2 * np + 0], qf[kt], *(const int (*)[2]) &kf[0]);
                gk_cu_mma_f16(s[2 * np + 1], qf[kt], *(const int (*)[2]) &kf[2]);
            }
        }

        // Scale, softcap and mask, and mark everything that is not a real
        // (query, key) pair as -inf so the softmax gives it no weight. A key
        // past the end of the cache was staged as zero, and a zero logit is
        // not a zero probability - this is the guard that matters.
        //
        // Only the last tile of the cache can be ragged, and whether a query
        // row exists does not change across the cache at all, so both bounds
        // are decided once rather than per element. Without that, the 32
        // elements a lane holds each carry two comparisons against a 64-bit
        // extent into what is otherwise a multiply and an exponential.
        const bool whole = c0 + GK_CU_FAM_BC(D_PAD) <= n_kv;

#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t r  = (i >= 2) ? row1 : row0;
                const int64_t ic = c0 + nt * 8 + 2 * tig + (i & 1);

                float x = -INFINITY;
                if (((i >= 2) ? live1 : live0) && (whole || ic < n_kv)) {
                    x = s[nt][i];
                    if (logit_softcap != 0.0f) {
                        x = logit_softcap * tanhf(x);
                    }
                    if (HAS_MASK) {
                        const float mv = slope * gk_cu_fam_mask(mask, ic, r,
                                                                iq2 % mask.ne[2],
                                                                iq3 % mask.ne[3]);
                        x = mv == -INFINITY ? -INFINITY : x + mv;
                    }
                }
                s[nt][i] = x;
            }
        }

        // The row maximum. A row lives in the four lanes that share a group,
        // so the reduction is over the low two lane bits and nothing else.
        float m[2] = { -INFINITY, -INFINITY };
#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
            m[0] = fmaxf(m[0], fmaxf(s[nt][0], s[nt][1]));
            m[1] = fmaxf(m[1], fmaxf(s[nt][2], s[nt][3]));
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 1));
            m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 2));
        }

        float corr[2];
        float sum[2] = { 0.0f, 0.0f };
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float m_new = fmaxf(M[r], m[r]);
            // A tile that was entirely masked leaves the running maximum
            // where it was; expf(-inf - -inf) is not a number, so it is not
            // asked.
            corr[r] = m_new == -INFINITY ? 1.0f : __expf(M[r] - m_new);
            M[r]    = m_new;
        }

#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int   r = (i >= 2) ? 1 : 0;
                const float p = s[nt][i] == -INFINITY || M[r] == -INFINITY
                              ? 0.0f : __expf(s[nt][i] - M[r]);
                s[nt][i] = p;
                sum[r]  += p;
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 1);
            sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 2);
            S[r]    = S[r] * corr[r] + sum[r];
        }

        // O = O * corr + P V. The probabilities are already in the lanes and
        // the registers the A operand wants, so they are packed rather than
        // staged: window `kw` covers keys `kw*16 .. +15`, which is exactly the
        // pair of 8-wide S tiles `2*kw` and `2*kw+1`.
        int pf[GK_CU_FAM_NKW(D_PAD)][4];
#pragma unroll
        for (int kw = 0; kw < GK_CU_FAM_NKW(D_PAD); ++kw) {
            pf[kw][0] = gk_cu_pack2_half(s[2 * kw + 0][0], s[2 * kw + 0][1]);
            pf[kw][1] = gk_cu_pack2_half(s[2 * kw + 0][2], s[2 * kw + 0][3]);
            pf[kw][2] = gk_cu_pack2_half(s[2 * kw + 1][0], s[2 * kw + 1][1]);
            pf[kw][3] = gk_cu_pack2_half(s[2 * kw + 1][2], s[2 * kw + 1][3]);
        }

        // The rescale runs only when this tile raised some row's running
        // maximum. `corr` is exactly 1.0f whenever it did not - `fmaxf(M, m)`
        // hands M back bit-for-bit and `expf(0.0f)` is 1.0f - and past the
        // first tiles of the cache that is most of them, so these sixty-four
        // FMULs a lane are usually dead weight. The vote makes the branch
        // warp-uniform; a divergent skip would predicate the multiplies and
        // save nothing.
        if (!__all_sync(0xffffffffu, corr[0] == 1.0f && corr[1] == 1.0f)) {
#pragma unroll
            for (int ct = 0; ct < NT_V; ++ct) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[ct][i] *= corr[(i >= 2) ? 1 : 0];
                }
            }
        }

        // Same `ldmatrix` pairing over the value tiles. d=40's five tiles do
        // not pair evenly; its odd tail keeps the indexed loads, and the
        // constant comparison folds the branch away everywhere else.
#pragma unroll
        for (int kw = 0; kw < GK_CU_FAM_NKW(D_PAD); ++kw) {
#pragma unroll
            for (int cp = 0; cp < NT_V / 2; ++cp) {
                int vf[4];
                gk_cu_ldmatrix_x4(vf,
                    (const int *) &Vt[buf][cp * 16 + ldm_r][kw * 16 + ldm_c]);

                gk_cu_mma_f16(acc[2 * cp + 0], pf[kw], *(const int (*)[2]) &vf[0]);
                gk_cu_mma_f16(acc[2 * cp + 1], pf[kw], *(const int (*)[2]) &vf[2]);
            }

            if (NT_V % 2 != 0) {
                const int ct = NT_V - 1;
                int bf[2];
                bf[0] = *(const int *) &Vt[buf][ct * 8 + group][kw * 16 + 2 * tig];
                bf[1] = *(const int *) &Vt[buf][ct * 8 + group][kw * 16 + 8 + 2 * tig];

                gk_cu_mma_f16(acc[ct], pf[kw], bf);
            }
        }

        if (prefetch) {
            commit_tile(buf ^ 1);
        }

        // The only barrier in the loop, and it carries both obligations: the
        // tile just committed into the other buffer becomes visible, and every
        // warp is known to have finished reading this one before the next
        // iteration commits over it.
        __syncthreads();
    }

    // The sink is one more virtual position, with a logit but no value row.
    if (sinks != NULL) {
        const float sv = sinks[iq2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (sv > M[r]) {
                const float c = M[r] == -INFINITY ? 0.0f : __expf(M[r] - sv);
                M[r] = sv;
                S[r] = S[r] * c + 1.0f;
#pragma unroll
                for (int ct = 0; ct < NT_V; ++ct) {
                    acc[ct][2 * r + 0] *= c;
                    acc[ct][2 * r + 1] *= c;
                }
            } else {
                S[r] += __expf(sv - M[r]);
            }
        }
    }

    const float inv[2] = { S[0] == 0.0f ? 0.0f : 1.0f / S[0],
                           S[1] == 0.0f ? 0.0f : 1.0f / S[1] };

#pragma unroll
    for (int ct = 0; ct < NT_V; ++ct) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t r  = (i >= 2) ? row1 : row0;
            const int64_t dv = ct * 8 + 2 * tig + (i & 1);

            if (r < n_q && dv < DV) {
                gk_cu_set(d, dv, iq2, r, iq3, acc[ct][i] * inv[(i >= 2) ? 1 : 0]);
            }
        }
    }
}

// Which instantiation a head goes to, or zero for one that has none. Both
// widths have to fit: the k windows are sized from the bucket and so are the
// value tiles, so a head that is narrow in V and wide in K - which is what
// multi-head latent attention is - has no bucket here and takes the float
// path instead of a bucket that would silently drop the rest of K.
// Whether a cache can be staged eight elements at a time: packed along the
// head, and every row start 16-byte aligned so the run is one word. A cache
// that is neither still works, an element at a time, through the scalar arm of
// gk_cu_fam_run8 - which is what a permuted or offset view of one falls to.
static __host__ __forceinline__ bool gk_cuda_fam_vec(const struct gk_tensor * t) {
    if ((size_t) t->nb[0] != gk_type_size(t->type)) {
        return false;
    }
    const uintptr_t bits = (uintptr_t) t->data | (uintptr_t) t->nb[1]
                         | (uintptr_t) t->nb[2] | (uintptr_t) t->nb[3];
    return bits % 16 == 0;
}

static __host__ __forceinline__ int gk_cuda_fam_bucket(int64_t dk, int64_t dv) {
#if defined(GK_USE_HIP)
    GK_UNUSED(dk);
    GK_UNUSED(dv);
    return 0;   // `mma.sync` is PTX; there is no HIP spelling of it here
#else
    const int64_t w = dk > dv ? dk : dv;

    if (w <=  40) { return  40; }
    if (w <=  64) { return  64; }
    if (w <=  80) { return  80; }
    if (w <= 128) { return 128; }
    if (w <= 160) { return 160; }

    return 0;
#endif
}

// `part_vkq` and `part_ms`, when given, are where a slice leaves its partial
// result instead of writing an answer: n_split accumulators of DV floats, and
// n_split (M, S) pairs, indexed by row then slice. When they are NULL this is
// the whole-cache path and the tail below finishes the softmax itself.
static __global__ void gk_cu_k_flash_attn(gk_tview q, gk_tview k, gk_tview v,
                                          gk_tview mask, bool has_mask,
                                          const float * sinks, gk_tview_mut d,
                                          float scale, float max_bias, float logit_softcap,
                                          int64_t n_head_log2,
                                          int64_t rk2, int64_t rk3, int64_t rv2, int64_t rv3,
                                          float * part_vkq, float * part_ms, int n_split) {
    __shared__ float sq[GK_CUDA_FA_MAX_DK];
    __shared__ float vkq[GK_CUDA_FA_MAX_DV];
    __shared__ float reduce[GK_CU_FA_BLOCK / GK_WARP_SIZE];
    __shared__ float s_shared;
    __shared__ float ms_shared;
    __shared__ float vs_shared;

    const int64_t DK = k.ne[0];
    const int64_t DV = v.ne[0];

    const int64_t ir  = blockIdx.x;
    const int64_t iq1 = ir % q.ne[1];
    const int64_t iq2 = (ir / q.ne[1]) % q.ne[2];
    const int64_t iq3 = ir / (q.ne[1] * q.ne[2]);

    // The slice of the cache this block owns. Unsplit, that is all of it.
    const int64_t n_kv = k.ne[1];
    const int64_t per  = (n_kv + n_split - 1) / n_split;
    const int64_t ic0  = (int64_t) blockIdx.y * per;
    const int64_t ic1  = ic0 + per < n_kv ? ic0 + per : n_kv;

    // A slice past the end of the cache - the last one, when the split does
    // not divide evenly. It still has to leave a partial behind, because the
    // merge reads every slot rather than checking which were written.
    if (ic0 >= n_kv) {
        if (part_ms != NULL && threadIdx.x == 0) {
            const int64_t slot = (ir * n_split + blockIdx.y) * 2;
            part_ms[slot + 0] = -INFINITY;
            part_ms[slot + 1] = 0.0f;
        }
        return;
    }

    const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
    const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

    const float slope = gk_cu_alibi_slope(max_bias, iq2, n_head_log2);

    for (int64_t i = threadIdx.x; i < DK; i += blockDim.x) {
        sq[i] = gk_cu_get(q, i, iq1, iq2, iq3);
    }
    for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
        vkq[i] = 0.0f;
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    for (int64_t ic = ic0; ic < ic1; ++ic) {
        // A fully masked position contributes nothing and is skipped before
        // the dot product rather than after it.
        float mv = 0.0f;
        if (has_mask) {
            mv = slope * gk_cu_get(mask, ic, iq1, iq2 % mask.ne[2], iq3 % mask.ne[3]);
            if (mv == -INFINITY) {
                continue;
            }
        }

        float part = 0.0f;
        for (int64_t i = threadIdx.x; i < DK; i += blockDim.x) {
            part += sq[i] * gk_cu_get(k, i, ic, ik2, ik3);
        }
        const float dot = gk_cu_block_sum(part, reduce);

        if (threadIdx.x == 0) {
            float s = dot * scale;
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            s += mv;
            s_shared = s;
        }
        __syncthreads();

        const float s = s_shared;

        // The rescale: if this logit is the new maximum, everything already
        // accumulated is scaled down to match it; otherwise this value is
        // scaled down to the running maximum.
        float ms = 1.0f;
        float vs = 1.0f;
        if (s > M) {
            ms = expf(M - s);
            M  = s;
            S  = S * ms + 1.0f;
        } else {
            vs = expf(s - M);
            S += vs;
        }

        if (threadIdx.x == 0) {
            ms_shared = ms;
            vs_shared = vs;
        }
        __syncthreads();

        const float ms_b = ms_shared;
        const float vs_b = vs_shared;

        for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
            vkq[i] = vkq[i] * ms_b + gk_cu_get(v, i, ic, iv2, iv3) * vs_b;
        }
        __syncthreads();
    }

    // A slice hands its accumulator over unnormalised and unsunk. Dividing by
    // its own S here would be wrong - S is only this slice's share of the
    // total - and the sink belongs to the row rather than to any one slice, so
    // applying it per slice would count it n_split times. Both are the merge's
    // job.
    if (part_vkq != NULL) {
        const int64_t slot = ir * n_split + blockIdx.y;

        for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
            part_vkq[slot * DV + i] = vkq[i];
        }
        if (threadIdx.x == 0) {
            part_ms[slot * 2 + 0] = M;
            part_ms[slot * 2 + 1] = S;
        }
        return;
    }

    // the sink is one more virtual position, with a logit but no value row
    if (sinks != NULL) {
        const float s = sinks[iq2];
        if (s > M) {
            const float ms = expf(M - s);
            M = s;
            S = S * ms + 1.0f;

            for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
                vkq[i] *= ms;
            }
            __syncthreads();
        } else {
            S += expf(s - M);
        }
    }

    const float inv = S == 0.0f ? 0.0f : 1.0f / S;

    // heads and batch swap on the way out: dst is [DV, n_head, n_batch, ne3]
    for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
        gk_cu_set(d, i, iq2, iq1, iq3, vkq[i] * inv);
    }
}

// The same computation with the synchronization turned inside out. The kernel
// above pays a block reduction and three barriers at every cache position,
// which for a decode shape - a handful of query rows, a block per row - makes
// the whole launch latency: a 512-position slice is fifteen hundred barriers
// deep before a single byte of V has moved. Measured on the speculative-verify
// shape (4 rows, 32 heads, 512 positions) it ran at 240 GFLOP/s on a part
// whose scalar units alone do fifty times that.
//
// Here a *lane owns a whole position*: each thread computes one full q.k dot
// by itself, a tile of 128 scores lands in shared memory at once, and the
// block synchronizes three times per 128 positions instead of three times per
// position. The V pass then walks the tile with every thread owning one output
// dimension, which makes the V reads coalesced - consecutive lanes touch
// consecutive elements of the same value row.
//
// The accumulator lives in registers, striped across the block (thread t owns
// dimensions t, t+128, ...), so there is no shared-memory accumulator to
// rescale under a barrier; each thread rescales its own stripe when the
// running maximum moves, once per tile.
//
// F16 lifts the K and V reads out of the per-element type switch: the
// dispatch only sets it when both operands really are contiguous f16, which
// every f16 KV cache is. Anything else - a quantized cache, an exotic stride -
// takes the generic accessor and still keeps the barrier structure.
template <bool F16>
static __global__ void gk_cu_k_flash_attn_vec(gk_tview q, gk_tview k, gk_tview v,
                                              gk_tview mask, bool has_mask,
                                              const float * sinks, gk_tview_mut d,
                                              float scale, float max_bias, float logit_softcap,
                                              int64_t n_head_log2,
                                              int64_t rk2, int64_t rk3, int64_t rv2, int64_t rv3,
                                              float * part_vkq, float * part_ms, int n_split) {
    constexpr int NACC = GK_CUDA_FA_MAX_DV / GK_CU_FA_BLOCK;

    __shared__ float sq[GK_CUDA_FA_MAX_DK];
    __shared__ float stile[GK_CU_FA_BLOCK];
    __shared__ float reduce[GK_CU_FA_BLOCK / GK_WARP_SIZE];

    const int64_t DK = k.ne[0];
    const int64_t DV = v.ne[0];

    const int64_t ir  = blockIdx.x;
    const int64_t iq1 = ir % q.ne[1];
    const int64_t iq2 = (ir / q.ne[1]) % q.ne[2];
    const int64_t iq3 = ir / (q.ne[1] * q.ne[2]);

    const int64_t n_kv = k.ne[1];
    const int64_t per  = (n_kv + n_split - 1) / n_split;
    const int64_t ic0  = (int64_t) blockIdx.y * per;
    const int64_t ic1  = ic0 + per < n_kv ? ic0 + per : n_kv;

    if (ic0 >= n_kv) {
        if (part_ms != NULL && threadIdx.x == 0) {
            const int64_t slot = (ir * n_split + blockIdx.y) * 2;
            part_ms[slot + 0] = -INFINITY;
            part_ms[slot + 1] = 0.0f;
        }
        return;
    }

    const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
    const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

    const float slope = gk_cu_alibi_slope(max_bias, iq2, n_head_log2);

    const char * kbase = k.data + ik2 * k.nb[2] + ik3 * k.nb[3];
    const char * vbase = v.data + iv2 * v.nb[2] + iv3 * v.nb[3];

    for (int64_t i = threadIdx.x; i < DK; i += blockDim.x) {
        sq[i] = gk_cu_get(q, i, iq1, iq2, iq3);
    }

    float acc[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a) {
        acc[a] = 0.0f;
    }

    float M = -INFINITY;
    float S = 0.0f;

    for (int64_t t0 = ic0; t0 < ic1; t0 += GK_CU_FA_BLOCK) {
        __syncthreads(); // sq on the first pass, last tile's stile after that

        const int64_t ic = t0 + threadIdx.x;

        float s = -INFINITY;
        if (ic < ic1) {
            float mv   = 0.0f;
            bool  dead = false;
            if (has_mask) {
                mv   = slope * gk_cu_get(mask, ic, iq1, iq2 % mask.ne[2], iq3 % mask.ne[3]);
                dead = mv == -INFINITY;
            }
            if (!dead) {
                float part = 0.0f;
                if (F16) {
                    const __half * krow = (const __half *) (kbase + ic * k.nb[1]);
                    int64_t i = 0;
                    for (; i + 1 < DK; i += 2) {
                        const float2 k2 = __half22float2(*(const __half2 *) (krow + i));
                        part += sq[i] * k2.x + sq[i + 1] * k2.y;
                    }
                    for (; i < DK; ++i) {
                        part += sq[i] * __half2float(krow[i]);
                    }
                } else {
                    for (int64_t i = 0; i < DK; ++i) {
                        part += sq[i] * gk_cu_get(k, i, ic, ik2, ik3);
                    }
                }
                s = part * scale;
                if (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;
            }
        }
        stile[threadIdx.x] = s;
        __syncthreads();

        const float tile_max = gk_cu_block_max(s, reduce);
        if (tile_max == -INFINITY) {
            continue; // every position masked or past the slice: nothing to add
        }

        const float newM = fmaxf(M, tile_max);
        const float ms   = M == -INFINITY ? 0.0f : __expf(M - newM);
        S *= ms;
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
            acc[a] *= ms;
        }
        M = newM;

        // Each thread exponentiates its own score in place; the barrier
        // publishes the whole tile of weights at once.
        stile[threadIdx.x] = s == -INFINITY ? 0.0f : __expf(s - M);
        __syncthreads();

        const int tlen = (int) (ic1 - t0 < GK_CU_FA_BLOCK ? ic1 - t0 : GK_CU_FA_BLOCK);
        for (int j = 0; j < tlen; ++j) {
            const float pj = stile[j];
            if (pj == 0.0f) {
                continue; // same value in every thread, so the branch is uniform
            }
            S += pj;

            const int64_t jc = t0 + j;
            if (F16) {
                const __half * vrow = (const __half *) (vbase + jc * v.nb[1]);
#pragma unroll
                for (int a = 0; a < NACC; ++a) {
                    const int64_t dim = threadIdx.x + a * GK_CU_FA_BLOCK;
                    if (dim < DV) {
                        acc[a] += pj * __half2float(vrow[dim]);
                    }
                }
            } else {
#pragma unroll
                for (int a = 0; a < NACC; ++a) {
                    const int64_t dim = threadIdx.x + a * GK_CU_FA_BLOCK;
                    if (dim < DV) {
                        acc[a] += pj * gk_cu_get(v, dim, jc, iv2, iv3);
                    }
                }
            }
        }
    }

    // From here the shape of the hand-off matches the kernel above exactly:
    // a slice leaves (M, S) and its unnormalised stripe for the merge; the
    // unsplit case folds the sink in and normalizes itself.
    if (part_vkq != NULL) {
        const int64_t slot = ir * n_split + blockIdx.y;
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
            const int64_t dim = threadIdx.x + a * GK_CU_FA_BLOCK;
            if (dim < DV) {
                part_vkq[slot * DV + dim] = acc[a];
            }
        }
        if (threadIdx.x == 0) {
            part_ms[slot * 2 + 0] = M;
            part_ms[slot * 2 + 1] = S;
        }
        return;
    }

    if (sinks != NULL) {
        const float sk = sinks[iq2];
        if (sk > M) {
            const float ms = M == -INFINITY ? 0.0f : __expf(M - sk);
            M = sk;
            S = S * ms + 1.0f;
#pragma unroll
            for (int a = 0; a < NACC; ++a) {
                acc[a] *= ms;
            }
        } else {
            S += __expf(sk - M);
        }
    }

    const float inv = S == 0.0f ? 0.0f : 1.0f / S;

#pragma unroll
    for (int a = 0; a < NACC; ++a) {
        const int64_t dim = threadIdx.x + a * GK_CU_FA_BLOCK;
        if (dim < DV) {
            gk_cu_set(d, dim, iq2, iq1, iq3, acc[a] * inv);
        }
    }
}

// Merges the slices of one query row.
//
// Each slice arrived with its accumulator relative to its own maximum, so the
// merge picks the maximum of those maxima and rescales every slice to it - the
// same `exp(M_slice - M)` factor the accumulation loop applies to itself
// whenever its running maximum moves. The sink, which belongs to the row and
// not to any slice, joins here as one more virtual position.
//
// The slices are walked in index order rather than reduced in a tree, so the
// sum is the same sum every run. That costs a little parallelism at n_split of
// 64 and buys back the property that two runs of one prompt agree.
static __global__ void gk_cu_k_flash_attn_combine(const float * part_vkq,
                                                  const float * part_ms,
                                                  const float * sinks, gk_tview_mut d,
                                                  int64_t DV, int n_split,
                                                  int64_t nq1, int64_t nq2) {
    const int64_t ir  = blockIdx.x;
    const int64_t iq1 = ir % nq1;
    const int64_t iq2 = (ir / nq1) % nq2;
    const int64_t iq3 = ir / (nq1 * nq2);

    const float * ms = part_ms + ir * n_split * 2;

    // The common maximum. A slice that saw nothing - every position masked, or
    // a slice off the end of the cache - reports S of zero and is skipped
    // rather than allowed to drag the maximum to -inf.
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        if (ms[s * 2 + 1] > 0.0f && ms[s * 2 + 0] > M) {
            M = ms[s * 2 + 0];
        }
    }
    if (sinks != NULL && sinks[iq2] > M) {
        M = sinks[iq2];
    }

    // Every thread works this out for itself. It is n_split adds against a
    // block-wide reduction and two barriers, and n_split is at most 64.
    float S = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        if (ms[s * 2 + 1] > 0.0f) {
            S += ms[s * 2 + 1] * expf(ms[s * 2 + 0] - M);
        }
    }
    if (sinks != NULL) {
        S += expf(sinks[iq2] - M);
    }

    const float inv = S == 0.0f ? 0.0f : 1.0f / S;

    for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            if (ms[s * 2 + 1] > 0.0f) {
                acc += part_vkq[((ir * n_split) + s) * DV + i] * expf(ms[s * 2 + 0] - M);
            }
        }
        gk_cu_set(d, i, iq2, iq1, iq3, acc * inv);
    }
}

// How many slices to cut the cache into, or 1 to walk it whole. Zero means the
// split was wanted but its scratch could not be had, which the caller turns
// back into the unsplit path rather than a failure.
static int gk_cu_fa_n_split(int n_sm, int64_t rows, int64_t n_kv) {
    if (n_sm <= 0 || n_kv < GK_CU_FA_MIN_SPLIT_KV) {
        return 1;
    }

    // The row count alone already fills the device: splitting would add a
    // merge and buy nothing. This is every prompt pass.
    const int64_t want = (int64_t) n_sm * GK_CU_FA_BLOCKS_PER_SM;
    if (rows >= want) {
        return 1;
    }

    int64_t n_split = (want + rows - 1) / rows;

    // A long cache wants cutting further than the block count asks for: ten
    // blocks each walking eight hundred positions is still ten long walks.
    const int64_t by_length = (n_kv + GK_CU_FA_TARGET_SLICE - 1) / GK_CU_FA_TARGET_SLICE;
    if (by_length > n_split) {
        n_split = by_length;
    }

    const int64_t by_kv = n_kv / GK_CU_FA_MIN_SLICE;
    if (n_split > by_kv) {
        n_split = by_kv;
    }
    if (n_split > GK_CU_FA_MAX_SPLIT) {
        n_split = GK_CU_FA_MAX_SPLIT;
    }

    return n_split < 1 ? 1 : (int) n_split;
}

// GK_FA_DUMP: the operand geometry behind each distinct attention shape and
// which of the three kernels took it, once per shape. Same reason as
// GK_EW_DUMP: the three paths differ by an order of magnitude and by which
// bounds they check, and a node name plus a destination extent does not say
// which one ran.
static __host__ void gk_cu_fa_dump(const char * path, const struct gk_tensor * dst) {
    static const bool on = getenv("GK_FA_DUMP") != NULL && getenv("GK_FA_DUMP")[0] != '0';
    if (!on) {
        return;
    }

    static char seen[64][160];
    static int  n_seen = 0;

    char key[160];
    int  off = snprintf(key, sizeof(key), "%-6s", path);

    for (int s = 0; s < 5 && off < (int) sizeof(key); ++s) {
        const struct gk_tensor * t = dst->src[s];
        if (t == NULL) {
            off += snprintf(key + off, sizeof(key) - off, " -");
            continue;
        }
        off += snprintf(key + off, sizeof(key) - off,
                        " %s[%lld %lld %lld %lld/%lld %lld %lld %lld]",
                        gk_type_name(t->type),
                        (long long) t->ne[0], (long long) t->ne[1],
                        (long long) t->ne[2], (long long) t->ne[3],
                        (long long) t->nb[0], (long long) t->nb[1],
                        (long long) t->nb[2], (long long) t->nb[3]);
    }

    for (int i = 0; i < n_seen; ++i) {
        if (strcmp(seen[i], key) == 0) {
            return;
        }
    }
    if (n_seen >= 64) {
        return;
    }
    snprintf(seen[n_seen++], sizeof(seen[0]), "%s", key);

    gk_logf("fa %s\n", key);
}

void gk_cuda_flash_attn(gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * dst) {
    const struct gk_tensor * q     = dst->src[0];
    const struct gk_tensor * k     = dst->src[1];
    const struct gk_tensor * v     = dst->src[2];
    const struct gk_tensor * mask  = dst->src[3];
    const struct gk_tensor * sinks = dst->src[4];

    float scale               = gk_get_op_params_f32(dst, 0);
    const float max_bias      = gk_get_op_params_f32(dst, 1);
    const float logit_softcap = gk_get_op_params_f32(dst, 2);

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    int64_t n_head_log2 = 1;
    while (n_head_log2 * 2 <= q->ne[2]) {
        n_head_log2 *= 2;
    }

    const int64_t rows = q->ne[1] * q->ne[2] * q->ne[3];
    const int64_t DK   = k->ne[0];
    const int64_t DV   = v->ne[0];
    const int64_t n_kv = k->ne[1];

    // The tensor-core path, when the head fits one of the widths its
    // accumulator is compiled for and the cache is already f16. A cache that
    // is not gets the float kernel rather than a conversion: reaching the
    // tensor core by rounding an operand the caller chose to keep in f32
    // would be trading their precision for our speed.
    //
    // Query rows are the other condition. A block owns 64 of them and does
    // not split the cache, so a handful of rows against a long cache - which
    // is generation - would leave the card idle. That shape is what the split
    // path below is for.
    const int fam_d = gk_cuda_fam_bucket(DK, DV);

    // The same escape hatch GK_FA_VEC gives the split path, for the two paths
    // above it: with three kernels answering one op, "which kernel is wrong"
    // is otherwise only answerable by rebuilding.
    static const char * fa_mma_env   = getenv("GK_FA_MMA");
    static const char * fa_tiled_env = getenv("GK_FA_TILED");
    const bool fa_mma   = !(fa_mma_env   != NULL && fa_mma_env[0]   == '0');
    const bool fa_tiled = !(fa_tiled_env != NULL && fa_tiled_env[0] == '0');

    if (fa_mma && fam_d != 0 && q->ne[1] >= 16 &&
        (int) k->type == GK_TYPE_F16 && (int) v->type == GK_TYPE_F16 &&
        (mask == NULL || (int) mask->type == GK_TYPE_F16) &&
        ((int) q->type == GK_TYPE_F32 || (int) q->type == GK_TYPE_F16 ||
         (int) q->type == GK_TYPE_BF16) &&
        scratch != NULL && scratch->cc >= 80) {

        // Four warps a block when a head's K and V sit in L2 anyway - two
        // independent blocks per multiprocessor instead of one barrier over
        // eight warps - and the full eight when the cache is long enough that
        // halving a block's query rows would double real DRAM re-reads.
        const bool fam_short = n_kv * (DK + DV) * 2 <= (int64_t) 2 << 20;
        const int  fam_warps = fam_short ? 4 : GK_CU_FAM_WARPS;

        dim3 mgrid;
        mgrid.x = (unsigned) ((q->ne[1] + GK_CU_FAM_BR(fam_warps) - 1) / GK_CU_FAM_BR(fam_warps));
        mgrid.y = (unsigned) q->ne[2];
        mgrid.z = (unsigned) q->ne[3];

        // Whether the eight halves of a run can be taken as one 16-byte load
        // is a *template* parameter, not an argument, and the difference is
        // not small. As an argument both paths are compiled into every kernel,
        // and the scalar one - eight separate loads, each with its own 64-bit
        // stride arithmetic, inside unrolled staging loops - dominated the
        // result: the d=128 kernel was 73k SASS instructions around 64 `HMMA`,
        // with 8131 `LDG` and 7227 branches for a tile it stages with sixteen
        // vector loads. Splitting the instantiation leaves the fast kernel
        // with only the fast path in it.
#define GK_CU_FAM_LAUNCH_VMW(D, V, M, W)                                            \
        gk_cu_k_flash_attn_mma<D, V, M, W>                                          \
            <<<mgrid, GK_CU_FAM_THREADS(W), 0, stream>>>(                           \
            gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),                            \
            mask ? gk_cu_view(mask) : gk_cu_view(q),                                \
            sinks ? (const float *) sinks->data : NULL,                             \
            gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,       \
            q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],                               \
            q->ne[2] / v->ne[2], q->ne[3] / v->ne[3])

#define GK_CU_FAM_LAUNCH_VM(D, V, M)                                                \
        do {                                                                        \
            if (fam_short) { GK_CU_FAM_LAUNCH_VMW(D, V, M, 4);               }      \
            else           { GK_CU_FAM_LAUNCH_VMW(D, V, M, GK_CU_FAM_WARPS); }      \
        } while (0)

        // Whether there is a mask at all is the other template parameter, and
        // it is worth more than the contiguity one. The mask read is the only
        // per-element access in the inner loop that goes through a tensor
        // view, and a diffusion transformer's self-attention has no mask - so
        // for the shape this kernel spends its life on, this deletes the code
        // rather than predicating it.
#define GK_CU_FAM_LAUNCH_V(D, V)                                    \
        do {                                                        \
            if (mask) { GK_CU_FAM_LAUNCH_VM(D, V, true);  }         \
            else      { GK_CU_FAM_LAUNCH_VM(D, V, false); }         \
        } while (0)

        // One flag for both operands rather than two: K and V come from the
        // same cache and are contiguous or not together, so the mixed case is
        // not worth a third and fourth instantiation of a kernel this size. If
        // either is strided, both take the scalar path.
        const bool fam_vec = gk_cuda_fam_vec(k) && gk_cuda_fam_vec(v);

        gk_cu_fa_dump(fam_vec ? (fam_short ? "mma/v w4" : "mma/v")
                              : (fam_short ? "mma w4"   : "mma"), dst);

#define GK_CU_FAM_LAUNCH(D)                                    \
        do {                                                   \
            if (fam_vec) { GK_CU_FAM_LAUNCH_V(D, true);  }     \
            else         { GK_CU_FAM_LAUNCH_V(D, false); }     \
            return;                                            \
        } while (0)

        switch (fam_d) {
            case  40: GK_CU_FAM_LAUNCH( 40);
            case  64: GK_CU_FAM_LAUNCH( 64);
            case  80: GK_CU_FAM_LAUNCH( 80);
            case 128: GK_CU_FAM_LAUNCH(128);
            default:  GK_CU_FAM_LAUNCH(160);
        }
#undef GK_CU_FAM_LAUNCH_V
#undef GK_CU_FAM_LAUNCH_VM
#undef GK_CU_FAM_LAUNCH_VMW

#undef GK_CU_FAM_LAUNCH
    }

    // The tiled path, when there are enough query rows to fill a block's warps
    // and the head is narrow enough for two tiles of it to sit in shared
    // memory. A single query row would leave seven of eight warps idle, so
    // generation keeps the split path below instead.
    const size_t fat_smem = ((size_t) GK_CU_FAT_BC * (DK + 1)
                           + (size_t) GK_CU_FAT_BC * (DV + 1)
                           + (size_t) GK_CU_FAT_QROWS * DK
                           + (size_t) GK_CU_FAT_QROWS * GK_CU_FAT_BC) * sizeof(float);

    // The tiled path needs its whole working set in shared memory, and how much
    // a block may have is a property of the device, not a constant: 48 KB
    // without asking, and on Ampere and later most of the multiprocessor's
    // store on request. A head of 128 - which is what a diffusion transformer
    // usually runs - needs 52 KB, so the request is not optional.
    //
    // Asking for more than the device will give is a launch failure, not a
    // slow kernel, so the limit is checked rather than assumed.
    // DV is bounded separately from the memory: a lane holds the accumulator
    // for value dimensions `lane, lane+32, ...`, and that array is sized at
    // compile time. A wider DV would fit in shared memory long before it fit
    // in the registers, and would quietly drop every dimension past the end.
    // At least half the block's query rows, not merely one warp's worth: a
    // block always walks the whole cache, so a speculative-decode verify pass
    // (four rows) taken here runs one block per head with seven of eight
    // warps idle - 32 blocks on a 170-SM part - where the split path below
    // cuts the same cache into hundreds of blocks. Measured on a 4-row,
    // 32-head, 256-position shape, tiled was 0.21 ms against split's tens of
    // microseconds. Above sixteen rows the mma path has already taken every
    // f16 cache, so this floor only routes the small-q tail.
    if (fa_tiled && q->ne[1] >= GK_CU_FAT_QROWS / 2 && DV <= GK_CU_FAT_MAX_D && scratch != NULL &&
        fat_smem <= (size_t) scratch->smem_max) {

        if (fat_smem > 48u * 1024u) {
            // Raising the cap is per-kernel and sticky, but it is a property of
            // the function *on the current device*, not of the function. One
            // global flag raises it on whichever card happens to run the first
            // wide head and leaves every other card at the 48 KB default, where
            // the launch fails outright - "invalid argument", not a slow kernel.
            static bool raised[GK_CUDA_MAX_DEVICES] = { false };
            int dev = 0;
            if (gkGetDevice(&dev) == gkSuccess && dev >= 0 && dev < GK_CUDA_MAX_DEVICES && !raised[dev]) {
                raised[dev] = true;
                GK_CUDA_CHECK(gkFuncSetAttribute(
                    (const void *) gk_cu_k_flash_attn_tiled,
                    gkFuncAttributeMaxDynamicSharedMemorySize, scratch->smem_max));
            }
        }

        const size_t smem = fat_smem;

        gk_cu_fa_dump("tiled", dst);

        dim3 tgrid;
        tgrid.x = (unsigned) ((q->ne[1] + GK_CU_FAT_QROWS - 1) / GK_CU_FAT_QROWS);
        tgrid.y = (unsigned) q->ne[2];
        tgrid.z = (unsigned) q->ne[3];

        gk_cu_k_flash_attn_tiled<<<tgrid, GK_CU_FAT_WARPS * GK_WARP_SIZE, smem, stream>>>(
            gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),
            mask ? gk_cu_view(mask) : gk_cu_view(q), mask != NULL,
            sinks ? (const float *) sinks->data : NULL,
            gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,
            q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],
            q->ne[2] / v->ne[2], q->ne[3] / v->ne[3]);
        return;
    }

    int n_split = gk_cu_fa_n_split(scratch != NULL ? scratch->n_sm : 0, rows, n_kv);

    float * part_vkq = NULL;
    float * part_ms  = NULL;

    if (n_split > 1) {
        // The accumulators and the (M, S) pairs share one allocation, the
        // pairs after the accumulators, so a grow moves one buffer.
        const size_t n_vkq   = (size_t) rows * n_split * DV;
        const size_t n_ms    = (size_t) rows * n_split * 2;
        const size_t needed  = (n_vkq + n_ms) * sizeof(float);

        float * buf = (float *) gk_cu_scratch_get(scratch, needed, stream);
        if (buf != NULL) {
            part_vkq = buf;
            part_ms  = buf + n_vkq;
        } else {
            // No room for the partials. The whole-cache path needs none and
            // gives the same answer, so it is the fallback rather than an
            // error - slower is better than refusing a graph the scheduler
            // has already placed here.
            n_split = 1;
        }
    }

    dim3 grid;
    grid.x = (unsigned) rows;
    grid.y = (unsigned) n_split;
    grid.z = 1;

    // The lane-per-position kernel unless the environment asks for the old
    // one; its f16 instantiation additionally wants both cache operands
    // element-contiguous, which is how every f16 KV cache is laid out.
    static const char * fa_vec_env = getenv("GK_FA_VEC");
    const bool fa_vec = !(fa_vec_env != NULL && fa_vec_env[0] == '0');
    const bool fa_f16 = (int) k->type == GK_TYPE_F16 && (int) v->type == GK_TYPE_F16 &&
                        k->nb[0] == sizeof(__half) && v->nb[0] == sizeof(__half);

    if (fa_vec) {
        gk_cu_fa_dump(n_split > 1 ? (fa_f16 ? "split-vec/f16" : "split-vec")
                                  : (fa_f16 ? "flat-vec/f16"  : "flat-vec"), dst);

#define GK_CU_FA_VEC_LAUNCH(F16)                                                    \
        gk_cu_k_flash_attn_vec<F16><<<grid, GK_CU_FA_BLOCK, 0, stream>>>(           \
            gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),                            \
            mask ? gk_cu_view(mask) : gk_cu_view(q), mask != NULL,                  \
            sinks ? (const float *) sinks->data : NULL,                             \
            gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,       \
            q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],                               \
            q->ne[2] / v->ne[2], q->ne[3] / v->ne[3],                               \
            part_vkq, part_ms, n_split)

        if (fa_f16) { GK_CU_FA_VEC_LAUNCH(true);  }
        else        { GK_CU_FA_VEC_LAUNCH(false); }
#undef GK_CU_FA_VEC_LAUNCH
    } else {
        gk_cu_fa_dump(n_split > 1 ? "split" : "flat", dst);

        gk_cu_k_flash_attn<<<grid, GK_CU_FA_BLOCK, 0, stream>>>(
            gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),
            mask ? gk_cu_view(mask) : gk_cu_view(q), mask != NULL,
            sinks ? (const float *) sinks->data : NULL,
            gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,
            q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],
            q->ne[2] / v->ne[2], q->ne[3] / v->ne[3],
            part_vkq, part_ms, n_split);
    }

    if (n_split > 1) {
        // One block per row again, and the merge is the only thing that writes
        // the destination. Same stream, so it cannot start before the slices
        // that feed it have finished.
        const int block = DV < GK_CU_FA_BLOCK ? (int) DV : GK_CU_FA_BLOCK;

        gk_cu_k_flash_attn_combine<<<(int) rows, block, 0, stream>>>(
            part_vkq, part_ms,
            sinks ? (const float *) sinks->data : NULL,
            gk_cu_view_mut(dst), DV, n_split, q->ne[1], q->ne[2]);
    }

}
