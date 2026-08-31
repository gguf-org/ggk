#pragma once

// A thin portable vector layer.
//
// Just enough to express the handful of things the hot kernels do: load a run
// of floats, multiply-accumulate, and reduce to a scalar. Everything above
// this header is written once and compiles to AVX-512, AVX2, NEON or plain
// scalar depending on what the compiler was told it may use.
//
// Two decisions worth stating, because they shape every kernel that uses this:
//
// **Multiple accumulators.** An FMA has a latency of four or five cycles and a
// throughput of one or two per cycle, so a loop with a single accumulator
// stalls on its own dependency chain and reaches a fraction of the achievable
// rate. Every dot below keeps `GK_SIMD_ACC` independent accumulators and sums
// them at the end. That is also why the results differ from a plain left-to-
// right scalar sum - see below.
//
// **Accumulation order is part of the answer.** Summing the same values in a
// different order gives a different float. The scalar reference in gk_traits.c
// accumulates in double, left to right; these accumulate in float, in a fixed
// interleaved order. Both are correct; they are not identical, and the
// difference is a few parts in 10^7 over a long row.
//
// What matters is that the order here is *fixed* - it depends only on the row
// length, never on the thread count or on which thread ran it. So the
// bit-identical-across-thread-counts property the pool relies on survives, and
// results stay reproducible. Anything added here must keep that: no
// accumulation order that varies with how the work was split.

#include <stdint.h>

// How many independent accumulator chains the dots keep.
#define GK_SIMD_ACC 4

#if defined(__AVX512F__)

#include <immintrin.h>

#define GK_SIMD_NAME     "AVX-512"
#define GK_SIMD_F32_STEP 16

typedef __m512 gk_f32x;

static inline gk_f32x gk_f32x_zero(void)             { return _mm512_setzero_ps(); }
static inline gk_f32x gk_f32x_load(const float * p)  { return _mm512_loadu_ps(p); }
static inline void    gk_f32x_store(float * p, gk_f32x v) { _mm512_storeu_ps(p, v); }
static inline gk_f32x gk_f32x_set1(float v)          { return _mm512_set1_ps(v); }
static inline gk_f32x gk_f32x_add(gk_f32x a, gk_f32x b) { return _mm512_add_ps(a, b); }
static inline gk_f32x gk_f32x_sub(gk_f32x a, gk_f32x b) { return _mm512_sub_ps(a, b); }
static inline gk_f32x gk_f32x_mul(gk_f32x a, gk_f32x b) { return _mm512_mul_ps(a, b); }
static inline gk_f32x gk_f32x_fma(gk_f32x acc, gk_f32x a, gk_f32x b) {
    return _mm512_fmadd_ps(a, b, acc);
}
static inline float gk_f32x_reduce(gk_f32x v) { return _mm512_reduce_add_ps(v); }

#elif defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

#define GK_SIMD_NAME     "AVX2"
#define GK_SIMD_F32_STEP 8

typedef __m256 gk_f32x;

static inline gk_f32x gk_f32x_zero(void)             { return _mm256_setzero_ps(); }
static inline gk_f32x gk_f32x_load(const float * p)  { return _mm256_loadu_ps(p); }
static inline void    gk_f32x_store(float * p, gk_f32x v) { _mm256_storeu_ps(p, v); }
static inline gk_f32x gk_f32x_set1(float v)          { return _mm256_set1_ps(v); }
static inline gk_f32x gk_f32x_add(gk_f32x a, gk_f32x b) { return _mm256_add_ps(a, b); }
static inline gk_f32x gk_f32x_sub(gk_f32x a, gk_f32x b) { return _mm256_sub_ps(a, b); }
static inline gk_f32x gk_f32x_mul(gk_f32x a, gk_f32x b) { return _mm256_mul_ps(a, b); }
static inline gk_f32x gk_f32x_fma(gk_f32x acc, gk_f32x a, gk_f32x b) {
    return _mm256_fmadd_ps(a, b, acc);
}

static inline float gk_f32x_reduce(gk_f32x v) {
    // fold 256 -> 128 -> 64 -> 32
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

#elif defined(__ARM_NEON)

#include <arm_neon.h>

#define GK_SIMD_NAME     "NEON"
#define GK_SIMD_F32_STEP 4

typedef float32x4_t gk_f32x;

static inline gk_f32x gk_f32x_zero(void)             { return vdupq_n_f32(0.0f); }
static inline gk_f32x gk_f32x_load(const float * p)  { return vld1q_f32(p); }
static inline void    gk_f32x_store(float * p, gk_f32x v) { vst1q_f32(p, v); }
static inline gk_f32x gk_f32x_set1(float v)          { return vdupq_n_f32(v); }
static inline gk_f32x gk_f32x_add(gk_f32x a, gk_f32x b) { return vaddq_f32(a, b); }
static inline gk_f32x gk_f32x_sub(gk_f32x a, gk_f32x b) { return vsubq_f32(a, b); }
static inline gk_f32x gk_f32x_mul(gk_f32x a, gk_f32x b) { return vmulq_f32(a, b); }
static inline gk_f32x gk_f32x_fma(gk_f32x acc, gk_f32x a, gk_f32x b) {
    return vfmaq_f32(acc, a, b);
}
static inline float gk_f32x_reduce(gk_f32x v) { return vaddvq_f32(v); }

#else

#define GK_SIMD_NAME     "scalar"
#define GK_SIMD_F32_STEP 1

typedef float gk_f32x;

static inline gk_f32x gk_f32x_zero(void)             { return 0.0f; }
static inline gk_f32x gk_f32x_load(const float * p)  { return *p; }
static inline void    gk_f32x_store(float * p, gk_f32x v) { *p = v; }
static inline gk_f32x gk_f32x_set1(float v)          { return v; }
static inline gk_f32x gk_f32x_add(gk_f32x a, gk_f32x b) { return a + b; }
static inline gk_f32x gk_f32x_sub(gk_f32x a, gk_f32x b) { return a - b; }
static inline gk_f32x gk_f32x_mul(gk_f32x a, gk_f32x b) { return a * b; }
static inline gk_f32x gk_f32x_fma(gk_f32x acc, gk_f32x a, gk_f32x b) { return acc + a * b; }
static inline float   gk_f32x_reduce(gk_f32x v) { return v; }

#endif

// Floats consumed per unrolled iteration.
#define GK_SIMD_BLOCK (GK_SIMD_F32_STEP * GK_SIMD_ACC)

// --------------------------------------------------------------------------
// f16 conversion
//
// x86 with F16C and ARM with the fp16 extension both convert a whole vector in
// one instruction, which is the difference between f16 weights being free to
// read and being the bottleneck. Without either, this falls through to the
// scalar path in gk_traits.c.
// --------------------------------------------------------------------------

#if defined(__F16C__)
   // F16C is selected independently of AVX2, so the includes above may not have
   // happened. Including it again is harmless.
#  include <immintrin.h>
#endif

// This loader has to produce a whole `gk_f32x`, so it is gated on the vector
// width and not on F16C alone: `-mf16c` can be selected without AVX2, and then
// `gk_f32x` is a scalar while `_mm256_cvtph_ps` still compiles. The scalar
// converter in gk_vecdot.c is the one that needs only F16C.
#if defined(__F16C__) && defined(__AVX512F__)
#  define GK_SIMD_HAVE_F16 1
   // 16 halves -> 16 floats
   static inline gk_f32x gk_f32x_load_f16(const uint16_t * p) {
       return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) p));
   }
#elif defined(__F16C__) && defined(__AVX2__) && defined(__FMA__)
#  define GK_SIMD_HAVE_F16 1
   static inline gk_f32x gk_f32x_load_f16(const uint16_t * p) {
       return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) p));
   }
#elif defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
#  define GK_SIMD_HAVE_F16 1
   static inline gk_f32x gk_f32x_load_f16(const uint16_t * p) {
       return vcvt_f32_f16(vld1_f16((const __fp16 *) p));
   }
#else
#  define GK_SIMD_HAVE_F16 0
#endif

// --------------------------------------------------------------------------
// the primitives
// --------------------------------------------------------------------------

// Dot product of two f32 runs. Fixed accumulation order, independent of any
// work split.
static inline float gk_dot_f32(int64_t n, const float * x, const float * y) {
#if GK_SIMD_F32_STEP > 1
    gk_f32x acc[GK_SIMD_ACC];
    for (int a = 0; a < GK_SIMD_ACC; ++a) {
        acc[a] = gk_f32x_zero();
    }

    int64_t i = 0;
    for (; i + GK_SIMD_BLOCK <= n; i += GK_SIMD_BLOCK) {
        for (int a = 0; a < GK_SIMD_ACC; ++a) {
            const int64_t off = i + (int64_t) a * GK_SIMD_F32_STEP;
            acc[a] = gk_f32x_fma(acc[a], gk_f32x_load(x + off), gk_f32x_load(y + off));
        }
    }

    for (; i + GK_SIMD_F32_STEP <= n; i += GK_SIMD_F32_STEP) {
        acc[0] = gk_f32x_fma(acc[0], gk_f32x_load(x + i), gk_f32x_load(y + i));
    }

    // pairwise fold, so the reduction order is also fixed
    for (int stride = GK_SIMD_ACC / 2; stride > 0; stride /= 2) {
        for (int a = 0; a < stride; ++a) {
            acc[a] = gk_f32x_add(acc[a], acc[a + stride]);
        }
    }

    float sum = gk_f32x_reduce(acc[0]);

    for (; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += (double) x[i] * y[i];
    }
    return (float) sum;
#endif
}

// y += a * x
static inline void gk_axpy_f32(int64_t n, float a, const float * x, float * y) {
#if GK_SIMD_F32_STEP > 1
    const gk_f32x va = gk_f32x_set1(a);

    int64_t i = 0;
    for (; i + GK_SIMD_F32_STEP <= n; i += GK_SIMD_F32_STEP) {
        gk_f32x_store(y + i, gk_f32x_fma(gk_f32x_load(y + i), va, gk_f32x_load(x + i)));
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
#endif
}

// dst = src * scale + bias
static inline void gk_scale_f32(int64_t n, const float * src, float * dst,
                                float scale, float bias) {
#if GK_SIMD_F32_STEP > 1
    const gk_f32x vs = gk_f32x_set1(scale);
    const gk_f32x vb = gk_f32x_set1(bias);

    int64_t i = 0;
    for (; i + GK_SIMD_F32_STEP <= n; i += GK_SIMD_F32_STEP) {
        gk_f32x_store(dst + i, gk_f32x_fma(vb, gk_f32x_load(src + i), vs));
    }
    for (; i < n; ++i) {
        dst[i] = src[i] * scale + bias;
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = src[i] * scale + bias;
    }
#endif
}

// Sum of squares - what rms_norm needs. Accumulated in float with the same
// fixed interleave as the dot; over a row of a few thousand that is well
// inside what the epsilon already absorbs.
static inline float gk_sumsq_f32(int64_t n, const float * x) {
    return gk_dot_f32(n, x, x);
}

// Plain sum, same structure.
static inline float gk_sum_f32(int64_t n, const float * x) {
#if GK_SIMD_F32_STEP > 1
    gk_f32x acc[GK_SIMD_ACC];
    for (int a = 0; a < GK_SIMD_ACC; ++a) {
        acc[a] = gk_f32x_zero();
    }

    int64_t i = 0;
    for (; i + GK_SIMD_BLOCK <= n; i += GK_SIMD_BLOCK) {
        for (int a = 0; a < GK_SIMD_ACC; ++a) {
            acc[a] = gk_f32x_add(acc[a], gk_f32x_load(x + i + (int64_t) a * GK_SIMD_F32_STEP));
        }
    }
    for (; i + GK_SIMD_F32_STEP <= n; i += GK_SIMD_F32_STEP) {
        acc[0] = gk_f32x_add(acc[0], gk_f32x_load(x + i));
    }

    for (int stride = GK_SIMD_ACC / 2; stride > 0; stride /= 2) {
        for (int a = 0; a < stride; ++a) {
            acc[a] = gk_f32x_add(acc[a], acc[a + stride]);
        }
    }

    float sum = gk_f32x_reduce(acc[0]);
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    return (float) sum;
#endif
}

// Elementwise binary over equal-length runs.
#define GK_SIMD_BINARY(name, scalar_op, vec_op) \
    static inline void name(int64_t n, const float * a, const float * b, float * dst) { \
        int64_t i = 0; \
        GK_SIMD_BINARY_BODY(vec_op) \
        for (; i < n; ++i) { \
            dst[i] = scalar_op; \
        } \
    }

#if GK_SIMD_F32_STEP > 1
#  define GK_SIMD_BINARY_BODY(vec_op) \
    for (; i + GK_SIMD_F32_STEP <= n; i += GK_SIMD_F32_STEP) { \
        gk_f32x_store(dst + i, vec_op(gk_f32x_load(a + i), gk_f32x_load(b + i))); \
    }
#else
#  define GK_SIMD_BINARY_BODY(vec_op) (void) 0;
#endif

GK_SIMD_BINARY(gk_vec_add_f32, a[i] + b[i], gk_f32x_add)
GK_SIMD_BINARY(gk_vec_sub_f32, a[i] - b[i], gk_f32x_sub)
GK_SIMD_BINARY(gk_vec_mul_f32, a[i] * b[i], gk_f32x_mul)

#undef GK_SIMD_BINARY_BODY
#undef GK_SIMD_BINARY

// The f16 dot lives in gk_traits.c rather than here: its tail needs the scalar
// f16 converter, which is part of the shared codec. This header supplies
// `gk_f32x_load_f16` and `GK_SIMD_HAVE_F16`; the loop is built on top there.

// --------------------------------------------------------------------------
// integer dot
//
// The inner product of two int8 runs, accumulated in int32. This is the
// primitive every quantized dot is built on: a block's weights and the
// matching activations are both integers, and the block's float scales factor
// out of the sum entirely, so the arithmetic that dominates a matmul is
// integer and exact.
//
// Exact is the operative word - there is no rounding inside the accumulation,
// so unlike the float dots this one has no accumulation-order question at all.
// Any order gives the same int32, provided it does not overflow, and it cannot:
// the widest case is 256 products of two int8 values, bounded by
// 256 * 127 * 128, comfortably inside int32.
// --------------------------------------------------------------------------

// Short runs matter here as much as long ones: nvfp4's sub-group is sixteen
// elements and the lattice formats' is eight.
//
// This used to run the 256-bit fold unconditionally, so a call shorter than 32
// paid for a horizontal reduction of an accumulator nothing had been added to
// and then did the whole dot scalar anyway. Both the fold and the 128-bit path
// are now entered only when there is something in them, which took nvfp4 from
// 3.46 to 4.11 GFLOP/s.
//
// A wider dot is not automatically better. Restructuring the iq2_xs kernel to
// feed this 16 elements at a time instead of 8 made it 1.9x *slower* - the
// buffer is filled a byte at a time and then read back as a vector, and at that
// size the round trip costs more than the vectorisation saves. The lattice
// kernels therefore still dot eight.
static inline int32_t gk_dot_i8(int64_t n, const int8_t * a, const int8_t * b) {
#if defined(__AVX2__)
    int64_t i = 0;
    int32_t sum = 0;

    if (n >= 32) {
        __m256i acc = _mm256_setzero_si256();

        for (; i + 32 <= n; i += 32) {
            const __m256i va = _mm256_loadu_si256((const __m256i *) (a + i));
            const __m256i vb = _mm256_loadu_si256((const __m256i *) (b + i));

            // maddubs takes the first operand as unsigned, so the sign of `a`
            // is moved onto `b` and `a` is made magnitude-only. That keeps the
            // whole product in range and lets one instruction do 32 multiplies
            // and 16 adds.
            const __m256i sa = _mm256_sign_epi8(va, va);          // |a|
            const __m256i sb = _mm256_sign_epi8(vb, va);          // b * sign(a)

            const __m256i p16 = _mm256_maddubs_epi16(sa, sb);     // 16 x int16
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
        }

        // fold 8 lanes to one
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4e));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xb1));

        sum = _mm_cvtsi128_si32(lo);
    }

    // the same idea 128 bits wide, for the 16-element runs the lattice formats
    // and the K-quants' groups produce
    if (i + 16 <= n) {
        const __m128i va = _mm_loadu_si128((const __m128i *) (a + i));
        const __m128i vb = _mm_loadu_si128((const __m128i *) (b + i));

        const __m128i sa = _mm_sign_epi8(va, va);
        const __m128i sb = _mm_sign_epi8(vb, va);

        __m128i p = _mm_madd_epi16(_mm_maddubs_epi16(sa, sb), _mm_set1_epi16(1));
        p = _mm_add_epi32(p, _mm_shuffle_epi32(p, 0x4e));
        p = _mm_add_epi32(p, _mm_shuffle_epi32(p, 0xb1));

        sum += _mm_cvtsi128_si32(p);
        i += 16;
    }

    for (; i < n; ++i) {
        sum += (int32_t) a[i] * b[i];
    }
    return sum;
#elif defined(__ARM_NEON)
    int32x4_t acc = vdupq_n_s32(0);

    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const int8x16_t va = vld1q_s8(a + i);
        const int8x16_t vb = vld1q_s8(b + i);

        const int16x8_t lo = vmull_s8(vget_low_s8(va),  vget_low_s8(vb));
        const int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
    }

    int32_t sum = vaddvq_s32(acc);
    for (; i < n; ++i) {
        sum += (int32_t) a[i] * b[i];
    }
    return sum;
#else
    int32_t sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += (int32_t) a[i] * b[i];
    }
    return sum;
#endif
}

// Sum of an int8 run - the block sums a format with an offset or a minimum
// needs, where the offset factors out as (offset * sum of activations).
static inline int32_t gk_sum_i8(int64_t n, const int8_t * a) {
    int32_t sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}
