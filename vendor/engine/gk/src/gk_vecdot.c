// Integer dot products against quantized weights.
//
// This is what makes a quantized matmul fast, and the reason is worth stating
// because it is not obvious from the shapes.
//
// The obvious way to multiply an f32 activation by a quantized weight is to
// turn the weight back into f32 and do a float dot. That is what
// `vec_dot_quant` in gk_traits.c does, and it is correct for every format the
// codec can decode - but it spends most of its time unpacking bits into floats
// that are immediately multiplied and thrown away.
//
// The alternative is to notice that a block's weights are *integers* with one
// float scale factored out of the whole block:
//
//     w_i = d * q_i          (plus, for some formats, a per-block minimum)
//
// So if the activations are quantized the same way - integers a_i with their
// own scale da - the block's contribution to the dot is
//
//     sum_i (d * q_i) * (da * a_i)  =  d * da * sum_i (q_i * a_i)
//
// and the inner sum is an integer dot of two int8 runs. The floats appear once
// per block instead of once per element, and the inner loop becomes integer
// arithmetic that vectorises far better than the unpacking it replaces.
//
// Formats with a minimum (`q4_1`, `q4_K`) need one extra term: the minimum is
// constant across the block, so it contributes m * sum(activations), which is
// why the activation formats below carry a precomputed block sum.
//
// This costs accuracy. The activations are quantized to 8 bits, which the
// float path never did, so results move by roughly a part in 10^3 rather than
// 10^8. That is the same trade every fast implementation makes, and the float
// path stays available as the reference each of these is checked against.
//
// A format opts in by changing two fields in its traits entry - `vec_dot_type`
// and `vec_dot`. Anything that has not opted in keeps the float path and stays
// correct, which is what let these land one at a time.

#include "gk_impl.h"
#include "gk_simd.h"

#include "qz_codebook.h"
#include "qz_format.h"
#include "qz_fp.h"
#include "qz_impl.h"
#include "qz_quant.h"

#include <math.h>

// --------------------------------------------------------------------------
// activation encoders
//
// The right-hand side of a matmul is converted to one of these once per row
// and then reused across every weight row, so their cost is amortised over the
// whole output column.
// --------------------------------------------------------------------------

// q8_0: a scale per 32, symmetric.
void gk_quantize_row_q8_0(const float * src, void * dst, int64_t k) {
    GK_ASSERT(k % QZ_QK8_0 == 0);

    qz_blk_q8_0 * y = (qz_blk_q8_0 *) dst;

    for (int64_t b = 0; b < k / QZ_QK8_0; ++b) {
        const float * x = src + b * QZ_QK8_0;

        float amax = 0.0f;
        for (int i = 0; i < QZ_QK8_0; ++i) {
            const float a = fabsf(x[i]);
            if (a > amax) {
                amax = a;
            }
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        y[b].d = qz_fp32_to_fp16(d);
        for (int i = 0; i < QZ_QK8_0; ++i) {
            y[b].qs[i] = (int8_t) lrintf(x[i] * id);
        }
    }
}

// q8_1: q8_0 plus the block sum already multiplied by the scale, which is the
// term a weight format with a minimum needs.
void gk_quantize_row_q8_1(const float * src, void * dst, int64_t k) {
    GK_ASSERT(k % QZ_QK8_1 == 0);

    qz_blk_q8_1 * y = (qz_blk_q8_1 *) dst;

    for (int64_t b = 0; b < k / QZ_QK8_1; ++b) {
        const float * x = src + b * QZ_QK8_1;

        float amax = 0.0f;
        for (int i = 0; i < QZ_QK8_1; ++i) {
            const float a = fabsf(x[i]);
            if (a > amax) {
                amax = a;
            }
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        int32_t isum = 0;
        for (int i = 0; i < QZ_QK8_1; ++i) {
            const int8_t q = (int8_t) lrintf(x[i] * id);
            y[b].qs[i] = q;
            isum += q;
        }

        y[b].d = qz_fp32_to_fp16(d);
        y[b].s = qz_fp32_to_fp16(d * (float) isum);
    }
}

// q8_K: one scale for a whole 256-element super-block, plus the sum over each
// group of 16 - the granularity the K-quants' per-group minima need.
void gk_quantize_row_q8_K(const float * src, void * dst, int64_t k) {
    GK_ASSERT(k % QZ_K == 0);

    qz_blk_q8_k * y = (qz_blk_q8_k *) dst;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float * x = src + b * QZ_K;

        float amax = 0.0f;
        for (int i = 0; i < QZ_K; ++i) {
            const float a = fabsf(x[i]);
            if (a > amax) {
                amax = a;
            }
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        y[b].d = d;

        for (int i = 0; i < QZ_K; ++i) {
            y[b].qs[i] = (int8_t) lrintf(x[i] * id);
        }
        for (int g = 0; g < QZ_K / 16; ++g) {
            y[b].bsums[g] = (int16_t) gk_sum_i8(16, y[b].qs + g * 16);
        }
    }
}

// --------------------------------------------------------------------------
// block scales
//
// Every kernel below reads one or two f16 scales per block and does nothing
// else with floats until the end. That makes the conversion, not the vector
// work, the thing to watch: the codec's `qz_fp16_to_fp32` lives in another
// translation unit, so each call is a real call, and its body is a dozen
// integer ops around two branches. Measured on a 4096x4096 q4_0 matmul,
// routing these through the header instead took it from 22.9 to 28.7 GFLOP/s -
// and q4_0 then matched q8_0 exactly, which is the tell that the unpacking was
// never what these kernels were spending their time on.
//
// F16C does the same conversion in one instruction. Both paths are exact
// (every f16 is representable in f32), so this changes speed only.
// --------------------------------------------------------------------------

static inline float gk_h2f(qz_fp16_t h) {
#if defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
#else
    return qz_h2f(h);
#endif
}

// --------------------------------------------------------------------------
// nibble unpacking
//
// The 4- and 5-bit formats interleave a block's two halves in one byte:
// element j and element j + n/2 share byte j. Unpacking into a flat int8 run
// first, then handing that to one shared integer dot, keeps each format's
// bit-twiddling in one short loop and puts all the vectorised work in
// `gk_dot_i8` where it is written once.
// --------------------------------------------------------------------------

// q4_0: value = d * (q - 8)
static inline void unpack_q4_0(const qz_blk_q4_0 * b, int8_t * out) {
    for (int i = 0; i < QZ_QK4_0 / 2; ++i) {
        out[i]                = (int8_t) ((b->qs[i] & 0xf) - 8);
        out[i + QZ_QK4_0 / 2] = (int8_t) ((b->qs[i] >> 4)  - 8);
    }
}

// q4_1: value = d * q + m, so the integers stay unsigned and the minimum is
// carried separately.
static inline void unpack_q4_1(const qz_blk_q4_1 * b, int8_t * out) {
    for (int i = 0; i < QZ_QK4_1 / 2; ++i) {
        out[i]                = (int8_t) (b->qs[i] & 0xf);
        out[i + QZ_QK4_1 / 2] = (int8_t) (b->qs[i] >> 4);
    }
}

// q5_0: the fifth bit of element j is bit j of the little-endian word qh.
static inline void unpack_q5_0(const qz_blk_q5_0 * b, int8_t * out) {
    uint32_t qh;
    memcpy(&qh, b->qh, sizeof(qh));

    for (int i = 0; i < QZ_QK5_0 / 2; ++i) {
        const int q0 = (b->qs[i] & 0xf) | (int) (((qh >> i) & 1u) << 4);
        const int q1 = (b->qs[i] >> 4)  | (int) (((qh >> (i + QZ_QK5_0 / 2)) & 1u) << 4);
        out[i]                = (int8_t) (q0 - 16);
        out[i + QZ_QK5_0 / 2] = (int8_t) (q1 - 16);
    }
}

static inline void unpack_q5_1(const qz_blk_q5_1 * b, int8_t * out) {
    uint32_t qh;
    memcpy(&qh, b->qh, sizeof(qh));

    for (int i = 0; i < QZ_QK5_1 / 2; ++i) {
        out[i]                = (int8_t) ((b->qs[i] & 0xf) | (int) (((qh >> i) & 1u) << 4));
        out[i + QZ_QK5_1 / 2] = (int8_t) ((b->qs[i] >> 4)  |
                                          (int) (((qh >> (i + QZ_QK5_1 / 2)) & 1u) << 4));
    }
}

// The non-linear 4-bit formats - iq4_nl, iq4_xs, mxfp4, nvfp4 - store a nibble
// that indexes a fixed 16-entry codebook rather than a plain integer. That
// looks like it would rule out an integer dot, and does not: every one of these
// codebooks holds *int8 values*. iq4's are the non-linear levels themselves,
// and the E2M1 table is stored pre-doubled with the scale converters returning
// a correspondingly halved scale, so the product is exact either way.
//
// So the only difference from a plain 4-bit format is one table lookup, and the
// dot stays integer.
static inline void unpack_codebook(const uint8_t * qs, int n, const int8_t * tab,
                                   int8_t * out) {
    for (int i = 0; i < n / 2; ++i) {
        out[i]         = tab[qs[i] & 0xf];
        out[i + n / 2] = tab[qs[i] >> 4];
    }
}

// The sixteen 6-bit scales of a q3_K super-block, stored offset by 32: the low
// nibbles of the first eight bytes, the high two bits packed four to a byte in
// the last four.
static inline void gk_q3_k_scales(const uint8_t * src, int8_t * sc) {
    for (int g = 0; g < 16; ++g) {
        const int low  = g < 8 ? (src[g] & 0xf) : (src[g - 8] >> 4);
        const int high = (src[8 + (g % 4)] >> (2 * (g / 4))) & 3;
        sc[g] = (int8_t) ((low | (high << 4)) - 32);
    }
}

// --------------------------------------------------------------------------
// AVX2 block kernels
//
// The scalar versions above unpack a block into an int8 buffer and then call
// the shared integer dot. That is fine for a 256-element super-block, where the
// unpacking is amortised, and it is *worse than not bothering* for a 32-element
// block: the buffer round trip and the call cost more than the dequantise-to-
// float path they were meant to replace. Measured, q4_0 went from 5.4 to 2.6
// GFLOP/s that way.
//
// These keep everything in registers. Two things make that work:
//
//   * the whole 32-element block is one __m256i, so unpacking is three
//     instructions and never touches memory;
//
//   * the per-block integer sums are *not* reduced to a scalar per block.
//     `madd_epi16` leaves eight int32 partial sums; those are converted to
//     float, scaled by the block's d*d, and accumulated into a running vector.
//     One horizontal reduction happens at the very end instead of one per
//     block, which is what removes the serial dependency between blocks.
// --------------------------------------------------------------------------

#if defined(__AVX2__)

static inline float gk_hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

static inline int32_t gk_hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4e));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xb1));
    return _mm_cvtsi128_si32(lo);
}

// 16 packed bytes -> 32 nibbles, low half first, matching the layout where
// element j and element j+16 share byte j.
static inline __m256i gk_unpack_nibbles(const uint8_t * qs) {
    const __m128i b = _mm_loadu_si128((const __m128i *) qs);
    const __m256i both = _mm256_set_m128i(_mm_srli_epi16(b, 4), b);
    return _mm256_and_si256(both, _mm256_set1_epi8(0xF));
}

// Eight int32 partial products of two int8 vectors. maddubs wants its first
// operand unsigned, so the sign of `q` is moved onto `a` and `q` is made
// magnitude-only - the product is unchanged and stays in range.
static inline __m256i gk_mul_i8(__m256i q, __m256i a) {
    const __m256i sq = _mm256_sign_epi8(q, q);
    const __m256i sa = _mm256_sign_epi8(a, q);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(sq, sa), _mm256_set1_epi16(1));
}

// The fifth bit of element j of a q5_x block is bit j of the little-endian
// word qh. Scattering 32 such bits into 32 byte lanes has no single
// instruction, so it goes through a shuffle and a mask.
static inline __m256i gk_unpack_qh_bits(const uint8_t * qh) {
    uint32_t bits;
    memcpy(&bits, qh, sizeof(bits));

    // broadcast the word, then select the byte holding each lane's bit
    __m256i v = _mm256_set1_epi32((int) bits);
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1,
        2,2,2,2, 2,2,2,2, 3,3,3,3, 3,3,3,3);
    v = _mm256_shuffle_epi8(v, shuf);

    const __m256i mask = _mm256_setr_epi8(
        1,2,4,8, 16,32,64,-128, 1,2,4,8, 16,32,64,-128,
        1,2,4,8, 16,32,64,-128, 1,2,4,8, 16,32,64,-128);
    v = _mm256_and_si256(v, mask);

    // a lane is 0xFF where its bit was set
    return _mm256_cmpeq_epi8(v, mask);
}

// The symmetric formats need the same product of two scales, but their two
// halves sit in different blocks, so there is no pair to load. Packing them
// into one dword first still converts both in a single instruction, and the
// product never leaves the vector unit.
static inline __m256 gk_scale2(qz_fp16_t a, qz_fp16_t b) {
#if defined(__F16C__)
    const __m128 v = _mm_cvtph_ps(
        _mm_cvtsi32_si128((int) ((uint32_t) a | ((uint32_t) b << 16))));
    return _mm256_broadcastss_ps(_mm_mul_ss(v, _mm_movehdup_ps(v)));
#else
    return _mm256_set1_ps(qz_h2f(a) * qz_h2f(b));
#endif
}

#define GK_AVX2_SYMMETRIC(name, blk_t, blk, unpack_expr) \
static void name(int n, float * s, const void * vx, const void * vy) { \
    const blk_t       * x = (const blk_t *) vx; \
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy; \
    __m256 acc = _mm256_setzero_ps(); \
    for (int b = 0; b < n / (blk); ++b) { \
        const __m256i q = (unpack_expr); \
        const __m256i a = _mm256_loadu_si256((const __m256i *) y[b].qs); \
        const __m256i p = gk_mul_i8(q, a); \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), \
                              gk_scale2(x[b].d, y[b].d), acc); \
    } \
    *s = gk_hsum_ps(acc); \
}

// q4_0: value = d * (q - 8)
GK_AVX2_SYMMETRIC(dot_q4_0_avx2, qz_blk_q4_0, QZ_QK4_0,
    _mm256_sub_epi8(gk_unpack_nibbles(x[b].qs), _mm256_set1_epi8(8)))

// q8_0: already int8
GK_AVX2_SYMMETRIC(dot_q8_0_avx2, qz_blk_q8_0, QZ_QK8_0,
    _mm256_loadu_si256((const __m256i *) x[b].qs))

// q5_0: value = d * (q - 16), fifth bit from qh. The comparison mask is all
// ones where the bit is set, so subtracting it adds one - hence the 16 lands
// as a subtraction of 16 after the high bit is added in.
static void dot_q5_0_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_q5_0 * x = (const qz_blk_q5_0 *) vx;
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy;

    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < n / QZ_QK5_0; ++b) {
        const __m256i lo = gk_unpack_nibbles(x[b].qs);
        const __m256i hi = _mm256_and_si256(gk_unpack_qh_bits(x[b].qh),
                                            _mm256_set1_epi8(16));
        const __m256i q  = _mm256_sub_epi8(_mm256_or_si256(lo, hi),
                                           _mm256_set1_epi8(16));

        const __m256i a = _mm256_loadu_si256((const __m256i *) y[b].qs);
        const __m256i p = gk_mul_i8(q, a);

        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p),
                              gk_scale2(x[b].d, y[b].d), acc);
    }
    *s = gk_hsum_ps(acc);
}

// Two adjacent f16 -> the low two lanes of an f32 vector, one instruction.
// This is the layout the offset formats already have: a block stores `d` and
// `m` side by side, and so does q8_1's `d` and `s`, so a block's whole float
// state is one 32-bit load.
static inline __m128 gk_h2f_pair(const qz_fp16_t * p) {
#if defined(__F16C__)
    uint32_t w;
    memcpy(&w, p, sizeof(w));
    return _mm_cvtph_ps(_mm_cvtsi32_si128((int) w));
#else
    // AVX2 without F16C is not a machine that exists, but the flags allow it
    return _mm_set_ps(0.0f, 0.0f, qz_h2f(p[1]), qz_h2f(p[0]));
#endif
}

// The formats with a minimum. The integers stay unsigned here, and the minimum
// contributes m * (the activation block sum), which q8_1 already carries.
//
// That second term looked cheap and was not. Deleting it - wrong, but
// informative - put the kernel at exactly the rate of its symmetric sibling:
// 28.7 against q4_0's 28.7, where keeping it gave 17.4. (Both measured
// mid-change, before the conversions below; the table in the README is the
// finished numbers.) One multiply and one add cannot cost 40% of a block. The
// two extra f16 conversions could, and did.
//
// So the pairs go through F16C, and the two products come out of a single
// multiply: [dx, mx] * [dy, sy] = [dx*dy, mx*sy]. Lane 0 broadcasts to scale
// the integer sums, lane 1 is the minimum term. Per block that is two loads,
// two converts and a multiply, in place of four conversions.
#define GK_AVX2_OFFSET(name, blk_t, blk, unpack_expr) \
static void name(int n, float * s, const void * vx, const void * vy) { \
    const blk_t       * x = (const blk_t *) vx; \
    const qz_blk_q8_1 * y = (const qz_blk_q8_1 *) vy; \
    __m256 acc  = _mm256_setzero_ps(); \
    __m128 tail = _mm_setzero_ps(); \
    for (int b = 0; b < n / (blk); ++b) { \
        const __m256i q = (unpack_expr); \
        const __m256i a = _mm256_loadu_si256((const __m256i *) y[b].qs); \
        const __m256i p = gk_mul_i8(q, a); \
        const __m128 dm = _mm_mul_ps(gk_h2f_pair(&x[b].d), gk_h2f_pair(&y[b].d)); \
        acc  = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), \
                               _mm256_broadcastss_ps(dm), acc); \
        tail = _mm_add_ss(tail, _mm_movehdup_ps(dm)); \
    } \
    *s = gk_hsum_ps(acc) + _mm_cvtss_f32(tail); \
}

GK_AVX2_OFFSET(dot_q4_1_avx2, qz_blk_q4_1, QZ_QK4_1, gk_unpack_nibbles(x[b].qs))

GK_AVX2_OFFSET(dot_q5_1_avx2, qz_blk_q5_1, QZ_QK5_1,
    _mm256_or_si256(gk_unpack_nibbles(x[b].qs),
                    _mm256_and_si256(gk_unpack_qh_bits(x[b].qh), _mm256_set1_epi8(16))))

// --------------------------------------------------------------------------
// AVX2 K-quant kernels
//
// A super-block has a second level of scaling - a small integer scale per
// group of 16 or 32 - which is why the scalar versions further down unpack a
// group into a buffer and call the shared integer dot: the group scale has to
// be applied to that group's sum and nothing else.
//
// Two observations remove the buffer:
//
//   * `maddubs` already emits int16 pair-sums, and `madd_epi16` multiplies
//     int16 by int16 while widening. Feeding the *group scale* as the second
//     operand of that `madd` folds the scaling into the widening step - no
//     separate multiply, and the accumulator can stay int32 for the whole
//     super-block. Since integer addition is associative and nothing here
//     overflows, the total is bit-identical to the scalar path.
//
//   * `maddubs` wants an unsigned first operand, and both formats' raw quants
//     already are; it is the *offset* (q - 32 for q6_K) that makes them
//     signed. Hoisting that offset out - it contributes -32 * scale_g *
//     sum(group's activations), and q8_K stores exactly those group sums -
//     lets the quants stay unsigned and skips the sign shuffle entirely.
//
// So the group scale costs nothing beyond the instruction that was already
// there, and the offset costs one scalar pass over 16 precomputed sums.
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// per-group scales, without per-group scalar work
//
// q2_K and q3_K carry a small integer scale per *16* elements, so a 32-byte
// vector spans two groups and needs two different scales - one per 128-bit
// half of the int32 accumulator. Building that vector the obvious way, a
// broadcast per half joined by `insertf128`, is eight scalar-fed constructions
// per super-block, and it dominated both kernels: replacing it with a constant
// took q2_K from 49.04 to 73.99 and q3_K from 32.48 to 42.66.
//
// Instead the sixteen scales are widened once per super-block into one
// register, evens in the low lane and odds in the high one. `shuffle_epi8` is
// an in-lane byte shuffle, so a single one then broadcasts group 2j across the
// low half and group 2j+1 across the high half together - the pairing the
// accumulator wants falls out of the lane structure for free.
// --------------------------------------------------------------------------

static inline __m128i gk_deinterleave_u8(__m128i v) {
    return _mm_shuffle_epi8(v, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                             1, 3, 5, 7, 9, 11, 13, 15));
}

static inline __m256i gk_group_scales(__m256i pairs, int j) {
    return _mm256_shuffle_epi8(pairs,
        _mm256_set1_epi16((short) ((2 * j) | ((2 * j + 1) << 8))));
}

// q3_K's sixteen 6-bit scales, unpacked into a register rather than a stack
// buffer. The buffer was not incidental: writing it a byte at a time and then
// reading it back is what made this kernel slow three separate ways - the
// unpack itself, the scale vectors built from it, and the offset term that
// scans it - and a wide load of a freshly byte-written buffer cannot
// store-forward, so the obvious vectorisation of any one of them made things
// worse. Nothing here touches memory.
//
// Low nibbles come from the first eight bytes (high nibbles giving groups
// 8-15); the top two bits of each scale are packed four to a byte in the last
// four, which is why the same broadcast word is masked at four shifts and
// blended by position. Stored offset by 32. `gk_q3_k_scales` above is the
// definition; test-foundation checks the two agree.
static inline __m128i gk_q3_k_scales_v(const uint8_t * src) {
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i mf = _mm_set1_epi8(0xF);

    const __m128i v8 = _mm_loadl_epi64((const __m128i *) src);
    const __m128i lo = _mm_and_si128(v8, mf);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(v8, 4), mf);
    const __m128i low16 = _mm_or_si128(lo, _mm_slli_si128(hi, 8));

    uint32_t w;
    memcpy(&w, src + 8, sizeof(w));
    const __m128i h  = _mm_set1_epi32((int) w);
    const __m128i h0 = _mm_and_si128(h, m3);
    const __m128i h1 = _mm_and_si128(_mm_srli_epi16(h, 2), m3);
    const __m128i h2 = _mm_and_si128(_mm_srli_epi16(h, 4), m3);
    const __m128i h3 = _mm_and_si128(_mm_srli_epi16(h, 6), m3);

    const __m128i odd4 = _mm_setr_epi8(0, 0, 0, 0, -1, -1, -1, -1,
                                       0, 0, 0, 0, -1, -1, -1, -1);
    const __m128i top8 = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i high2 = _mm_blendv_epi8(_mm_blendv_epi8(h0, h1, odd4),
                                          _mm_blendv_epi8(h2, h3, odd4), top8);

    return _mm_sub_epi8(_mm_or_si128(low16, _mm_slli_epi16(high2, 4)),
                        _mm_set1_epi8(32));
}

// value = d * scale_g * q - dmin * min_g, over 8 groups of 32.
//
// Group 2j and group 2j+1 are the low and high nibbles of the *same* 32 bytes,
// so one load feeds two groups.
static void dot_q4_K_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_q4_k * x = (const qz_blk_q4_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    const __m256i m4 = _mm256_set1_epi8(0xF);

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        uint8_t sc[QZ_K / 32], mn[QZ_K / 32];
        for (int g = 0; g < QZ_K / 32; ++g) {
            qz_unpack_scale_min_6bit_inline(x[b].scales, g, &sc[g], &mn[g]);
        }

        __m256i acc = _mm256_setzero_si256();

        for (int j = 0; j < QZ_K / 64; ++j) {
            const __m256i packed = _mm256_loadu_si256((const __m256i *) (x[b].qs + j * 32));

            const __m256i q0 = _mm256_and_si256(packed, m4);
            const __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), m4);

            const __m256i a0 = _mm256_loadu_si256((const __m256i *) (y[b].qs + (2 * j)     * 32));
            const __m256i a1 = _mm256_loadu_si256((const __m256i *) (y[b].qs + (2 * j + 1) * 32));

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q0, a0),
                                                         _mm256_set1_epi16(sc[2 * j])));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q1, a1),
                                                         _mm256_set1_epi16(sc[2 * j + 1])));
        }

        // the minima, from the two 16-wide activation sums each group spans
        int32_t sum_mins = 0;
        for (int g = 0; g < QZ_K / 32; ++g) {
            sum_mins += (int32_t) mn[g] *
                ((int32_t) y[b].bsums[2 * g] + (int32_t) y[b].bsums[2 * g + 1]);
        }

        sum += y[b].d * (gk_h2f(x[b].d)    * (float) gk_hsum_i32(acc)
                       - gk_h2f(x[b].dmin) * (float) sum_mins);
    }
    *s = sum;
}

// q5_K is q4_K with a fifth bit: element l of group g takes bit g of qh[l].
// Everything else - the 6-bit scale/min pairs, the group size, the minimum
// term - is identical, so this is the q4_K kernel with one extra OR.
static void dot_q5_K_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_q5_k * x = (const qz_blk_q5_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    const __m256i m4  = _mm256_set1_epi8(0xF);
    const __m256i m16 = _mm256_set1_epi8(16);

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        uint8_t sc[QZ_K / 32], mn[QZ_K / 32];
        for (int g = 0; g < QZ_K / 32; ++g) {
            qz_unpack_scale_min_6bit_inline(x[b].scales, g, &sc[g], &mn[g]);
        }

        // one byte per element position, reused by all eight groups
        const __m256i qh = _mm256_loadu_si256((const __m256i *) x[b].qh);

        __m256i acc = _mm256_setzero_si256();

        for (int j = 0; j < QZ_K / 64; ++j) {
            const __m256i packed = _mm256_loadu_si256((const __m256i *) (x[b].qs + j * 32));

            for (int half = 0; half < 2; ++half) {
                const int g = 2 * j + half;

                const __m256i nib = half == 0 ? _mm256_and_si256(packed, m4)
                                              : _mm256_and_si256(_mm256_srli_epi16(packed, 4), m4);

                // 0xFF where this group's bit is set, masked down to the value 16
                const __m256i bit = _mm256_set1_epi8((char) (1 << g));
                const __m256i hi  = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit), m16);

                const __m256i q = _mm256_or_si256(nib, hi);
                const __m256i a = _mm256_loadu_si256((const __m256i *) (y[b].qs + g * 32));

                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q, a),
                                                             _mm256_set1_epi16(sc[g])));
            }
        }

        int32_t sum_mins = 0;
        for (int g = 0; g < QZ_K / 32; ++g) {
            sum_mins += (int32_t) mn[g] *
                ((int32_t) y[b].bsums[2 * g] + (int32_t) y[b].bsums[2 * g + 1]);
        }

        sum += y[b].d * (gk_h2f(x[b].d)    * (float) gk_hsum_i32(acc)
                       - gk_h2f(x[b].dmin) * (float) sum_mins);
    }
    *s = sum;
}

// value = d * scale_g * (q - 32), over 16 groups of 16.
//
// Each half of the super-block is four 32-element chunks drawn from two 32-byte
// nibble loads and one 32-byte load of high bit-pairs. A chunk spans two groups,
// so the scale vector is two broadcasts: `maddubs` lane k covers elements 2k and
// 2k+1, which puts elements 0-15 in the low 128 bits and 16-31 in the high.
static void dot_q6_K_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_q6_k * x = (const qz_blk_q6_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m3 = _mm256_set1_epi8(3);

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        __m256i acc = _mm256_setzero_si256();

        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = x[b].ql      + h * 64;
            const uint8_t * qh = x[b].qh      + h * 32;
            const int8_t  * sc = x[b].scales  + h * 8;
            const int8_t  * a  = y[b].qs      + h * 128;

            const __m256i l0 = _mm256_loadu_si256((const __m256i *) ql);
            const __m256i l1 = _mm256_loadu_si256((const __m256i *) (ql + 32));
            const __m256i hb = _mm256_loadu_si256((const __m256i *) qh);

            // the two high bits of each element, already shifted into place
            #define GK_Q6_HI(shift) \
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, (shift)), m3), 4)

            const __m256i q[4] = {
                _mm256_or_si256(_mm256_and_si256(l0, m4),                    GK_Q6_HI(0)),
                _mm256_or_si256(_mm256_and_si256(l1, m4),                    GK_Q6_HI(2)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), GK_Q6_HI(4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), GK_Q6_HI(6)),
            };

            #undef GK_Q6_HI

            for (int c = 0; c < 4; ++c) {
                const __m256i av = _mm256_loadu_si256((const __m256i *) (a + c * 32));
                const __m256i sv = _mm256_set_m128i(_mm_set1_epi16(sc[2 * c + 1]),
                                                    _mm_set1_epi16(sc[2 * c]));

                acc = _mm256_add_epi32(acc,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(q[c], av), sv));
            }
        }

        // the -32 offset, hoisted out of every group
        int32_t bias = 0;
        for (int g = 0; g < QZ_K / 16; ++g) {
            bias += (int32_t) x[b].scales[g] * (int32_t) y[b].bsums[g];
        }

        sum += y[b].d * gk_h2f(x[b].d) * (float) (gk_hsum_i32(acc) - 32 * bias);
    }
    *s = sum;
}

// q2_K and q3_K share q6_K's shape - sixteen groups of sixteen - and its
// packing trick: the 2-bit payload of a 128-element half sits in one 32-byte
// span, so a single load covers eight groups, and each shift of it yields two
// consecutive groups in one register. `maddubs` lane k covers elements 2k and
// 2k+1, which puts the first group in the low 128 bits and the second in the
// high, so the scale vector is two broadcasts exactly as it is for q6_K.
//
// Both keep their quants unsigned and hoist the offset (the minimum for q2_K,
// the -4 for q3_K) into the activation block sums, which q8_K stores at the
// 16-element granularity these formats need.

// value = d * scale_g * q - dmin * min_g, q in [0,3]
static void dot_q2_K_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_q2_k * x = (const qz_blk_q2_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    const __m256i m3 = _mm256_set1_epi8(3);

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        const __m256i sc_pairs = _mm256_cvtepu8_epi16(gk_deinterleave_u8(
            _mm_and_si128(_mm_loadu_si128((const __m128i *) x[b].scales),
                          _mm_set1_epi8(0xF))));

        __m256i acc = _mm256_setzero_si256();

        for (int half = 0; half < 2; ++half) {
            const __m256i v = _mm256_loadu_si256((const __m256i *) (x[b].qs + half * 32));

            for (int sh = 0; sh < 4; ++sh) {
                const int jg = half * 4 + sh;   // group pair, covering 2jg and 2jg+1

                const __m256i q = _mm256_and_si256(_mm256_srli_epi16(v, 2 * sh), m3);
                const __m256i a = _mm256_loadu_si256(
                    (const __m256i *) (y[b].qs + jg * 32));

                acc = _mm256_add_epi32(acc,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(q, a),
                                      gk_group_scales(sc_pairs, jg)));
            }
        }

        // the minima, at the 16-element granularity q8_K's block sums use
        int32_t sum_mins = 0;
        for (int g = 0; g < QZ_K / 16; ++g) {
            sum_mins += (int32_t) (x[b].scales[g] >> 4) * (int32_t) y[b].bsums[g];
        }

        sum += y[b].d * (gk_h2f(x[b].d)    * (float) gk_hsum_i32(acc)
                       - gk_h2f(x[b].dmin) * (float) sum_mins);
    }
    *s = sum;
}

// value = d * scale_g * (q - 4), q in [0,7]. The third bit comes from hmask and
// is stored inverted - a set bit means "do not subtract 4" - which is the same
// as saying the raw quant is (low two bits) + 4 * (mask bit).
static void dot_q3_K_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_q3_k * x = (const qz_blk_q3_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m1 = _mm256_set1_epi8(1);

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        const __m128i sc_v     = gk_q3_k_scales_v(x[b].scales);
        const __m256i sc_pairs = _mm256_cvtepi8_epi16(gk_deinterleave_u8(sc_v));

        const __m256i hm = _mm256_loadu_si256((const __m256i *) x[b].hmask);

        __m256i acc = _mm256_setzero_si256();

        for (int half = 0; half < 2; ++half) {
            const __m256i v = _mm256_loadu_si256((const __m256i *) (x[b].qs + half * 32));

            for (int sh = 0; sh < 4; ++sh) {
                const int jg  = half * 4 + sh;   // group pair, covering 2jg and 2jg+1
                const int bit = half * 4 + sh;

                const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(v, 2 * sh), m3);
                const __m256i hb = _mm256_and_si256(_mm256_srli_epi16(hm, bit), m1);
                const __m256i q  = _mm256_or_si256(lo, _mm256_slli_epi16(hb, 2));

                const __m256i a = _mm256_loadu_si256(
                    (const __m256i *) (y[b].qs + jg * 32));

                acc = _mm256_add_epi32(acc,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(q, a),
                                      gk_group_scales(sc_pairs, jg)));
            }
        }

        // the -4 offset, hoisted out of every group. The scales are already in
        // a register, so this is one madd against the block sums rather than a
        // scan of a stack buffer.
        const __m256i bias = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(sc_v),
            _mm256_loadu_si256((const __m256i *) y[b].bsums));

        sum += y[b].d * gk_h2f(x[b].d) * (float) gk_hsum_i32(
            _mm256_sub_epi32(acc, _mm256_slli_epi32(bias, 2)));
    }
    *s = sum;
}

// A 16-entry int8 codebook applied to 32 nibble indices. `shuffle_epi8` is a
// 16-way byte lookup per 128-bit lane, so broadcasting the table to both lanes
// turns the whole codebook indirection into one instruction.
static inline __m256i gk_lut16(const int8_t * tab, __m256i idx) {
    const __m128i t = _mm_loadu_si128((const __m128i *) tab);
    return _mm256_shuffle_epi8(_mm256_set_m128i(t, t), idx);
}

// iq4_nl and mxfp4 are the same shape - 32 elements, one block scale, nibbles
// through a codebook - and differ only in the table and how the scale is
// spelled. The codebook values are signed, so the sign has to move onto the
// activations the way it does for q8_0.
#define GK_AVX2_CODEBOOK(name, blk_t, blk, table, scale_expr) \
static void name(int n, float * s, const void * vx, const void * vy) { \
    const blk_t       * x = (const blk_t *) vx; \
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy; \
    __m256 acc = _mm256_setzero_ps(); \
    for (int b = 0; b < n / (blk); ++b) { \
        const __m256i q = gk_lut16((table), gk_unpack_nibbles(x[b].qs)); \
        const __m256i a = _mm256_loadu_si256((const __m256i *) y[b].qs); \
        const __m256i p = gk_mul_i8(q, a); \
        const float d = (scale_expr) * gk_h2f(y[b].d); \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d), acc); \
    } \
    *s = gk_hsum_ps(acc); \
}

GK_AVX2_CODEBOOK(dot_iq4_nl_avx2, qz_blk_iq4_nl, QZ_QK4_NL,
    qz_iq4_values, gk_h2f(x[b].d))

GK_AVX2_CODEBOOK(dot_mxfp4_avx2, qz_blk_mxfp4, QZ_QK_MXFP4,
    qz_e2m1_values, qz_e8m0_to_fp32_half(x[b].e))

// One 32-element pass: two groups, one activation block. The permute puts the
// nibbles back in order (see below), and each group's scale is broadcast into
// the half of the accumulator its four int32 lanes land in.
//
// The scales arrive already decoded, as plain floats. nvfp4 needs two per pass
// and decoding them arithmetically was most of this kernel: 9.59 GFLOP/s with
// the codec's converter, 10.00 with a branchless one - the cost is per-scale
// scalar work at all, not the branches - 29.65 decoding all four of a block's
// scales together in a register, and 40.56 reading them from a table. The
// table lives in the codec beside the other value tables, so this is a lookup
// and not a second definition of the format; `test_ue4m3_vs_codec` checks
// every entry against `qz_ue4m3_to_fp32`.
static inline __m256 gk_nvfp4_pass(__m256 acc, const uint8_t * qs,
                                   const qz_blk_q8_0 * yb,
                                   float s_lo, float s_hi) {
    const __m256i q = _mm256_permute4x64_epi64(
        gk_lut16(qz_e2m1_values, gk_unpack_nibbles(qs)), _MM_SHUFFLE(3, 1, 2, 0));

    const __m256i a = _mm256_loadu_si256((const __m256i *) yb->qs);
    const __m256i p = gk_mul_i8(q, a);

    const float da = gk_h2f(yb->d);
    const __m256 sv = _mm256_set_m128(_mm_set1_ps(s_hi * da),
                                      _mm_set1_ps(s_lo * da));

    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), sv, acc);
}

static void dot_nvfp4_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_nvfp4 * x = (const qz_blk_nvfp4 *) vx;
    const qz_blk_q8_0  * y = (const qz_blk_q8_0 *) vy;

    // one q8_0 activation block spans two groups, so a pass is 32 elements and
    // a block is exactly two passes
    _Static_assert(QZ_QK8_0 / QZ_QK_NVFP4_SUB == 2, "nvfp4 pass covers two groups");
    _Static_assert(QZ_QK_NVFP4 / QZ_QK8_0 == 2, "nvfp4 block is two passes");

    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < n / QZ_QK_NVFP4; ++b) {
        const uint8_t * d = x[b].d;

        acc = gk_nvfp4_pass(acc, x[b].qs, &y[2 * b],
                            qz_ue4m3_values[d[0]], qz_ue4m3_values[d[1]]);
        acc = gk_nvfp4_pass(acc, x[b].qs + QZ_QK8_0 / 2, &y[2 * b + 1],
                            qz_ue4m3_values[d[2]], qz_ue4m3_values[d[3]]);
    }
    *s = gk_hsum_ps(acc);
}

// iq4_xs: eight groups of 32 in a super-block, each with a 6-bit scale stored
// offset by 32. The scale is an integer, so it folds into the same `madd_epi16`
// that widens - the whole super-block accumulates in int32.
//
// Unlike iq4_nl the nibbles of a group sit in its own 16 bytes, so the halves
// interleave within the group rather than across the block.
static void dot_iq4_xs_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq4_xs * x = (const qz_blk_iq4_xs *) vx;
    const qz_blk_q8_k   * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            const int ls = ((x[b].scales_l[g / 2] >> (4 * (g % 2))) & 0xf) |
                           (int) (((x[b].scales_h >> (2 * g)) & 3) << 4);

            const __m256i q = gk_lut16(qz_iq4_values,
                                       gk_unpack_nibbles(x[b].qs + g * 16));
            const __m256i a = _mm256_loadu_si256((const __m256i *) (y[b].qs + g * 32));

            // maddubs wants the first operand unsigned; move the sign across
            const __m256i sq = _mm256_sign_epi8(q, q);
            const __m256i sa = _mm256_sign_epi8(a, q);

            acc = _mm256_add_epi32(acc,
                _mm256_madd_epi16(_mm256_maddubs_epi16(sq, sa),
                                  _mm256_set1_epi16((short) (ls - 32))));
        }

        sum += y[b].d * gk_h2f(x[b].d) * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

// --------------------------------------------------------------------------
// AVX2 lattice kernels
//
// The scalar versions expand each sub-group's magnitudes and signs into an int8
// buffer and dot that. The buffer is what costs: eight bytes written one at a
// time, per eight elements.
//
// It is avoidable, because of how the codebooks are stored. A grid entry holds
// *unsigned magnitudes* and the sign lives in a separate mask - so rather than
// negating the weights to match the activations, the sign can be moved onto the
// activations instead:
//
//     sum_i (+/-mag_i) * a_i  ==  sum_i mag_i * (+/-a_i)
//
// and `maddubs` wants exactly that shape: an unsigned first operand and a signed
// second. The magnitudes go in untouched, straight from four 64-bit grid loads
// into one register, and the sign mask becomes one `sign_epi8` on the
// activations. Nothing is written to memory.
//
// The group scale is the odd integer (2s + 1), so it folds into the same
// `madd_epi16` that widens, and the accumulator stays int32 for the super-block.
// --------------------------------------------------------------------------

// Four 8-magnitude grid entries as one 32-byte register.
static inline __m256i gk_grid4_u64(const uint64_t * grid,
                                   int i0, int i1, int i2, int i3) {
    return _mm256_set_epi64x((long long) grid[i3], (long long) grid[i2],
                             (long long) grid[i1], (long long) grid[i0]);
}

// 32 sign bits, four bytes of them, as +1/-1 per lane. `gk_unpack_qh_bits`
// already scatters bit i of byte j to lane 8j+i, which is the layout every one
// of these formats stores its sub-group masks in.
static inline __m256i gk_signs32(const uint8_t * bits) {
    return _mm256_or_si256(_mm256_set1_epi8(1), gk_unpack_qh_bits(bits));
}

// One group of 32: magnitudes, the four sign-mask bytes, the activations, and
// an int16 scale vector (one broadcast, or two when the scale changes at 16).
static inline __m256i gk_lattice_group(__m256i mag, const uint8_t * sgn,
                                       const int8_t * a, __m256i scale) {
    const __m256i av = _mm256_loadu_si256((const __m256i *) a);
    const __m256i sa = _mm256_sign_epi8(av, gk_signs32(sgn));

    return _mm256_madd_epi16(_mm256_maddubs_epi16(mag, sa), scale);
}

static inline __m256i gk_scale16x2(int s0, int s1) {
    return _mm256_set_m128i(_mm_set1_epi16((short) s1), _mm_set1_epi16((short) s0));
}

static void dot_iq2_xxs_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq2_xxs * x = (const qz_blk_iq2_xxs *) vx;
    const qz_blk_q8_k    * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint32_t w[2];
            memcpy(w, x[b].qs + 4 * g, sizeof(w));

            const uint8_t * idx = (const uint8_t *) w;

            uint8_t sgn[4];
            for (int t = 0; t < 4; ++t) {
                sgn[t] = qz_sign_mask_from_bits((uint8_t) ((w[1] >> (7 * t)) & 127));
            }

            const __m256i mag = gk_grid4_u64(qz_grid_iq2_xxs,
                                             idx[0], idx[1], idx[2], idx[3]);

            acc = _mm256_add_epi32(acc,
                gk_lattice_group(mag, sgn, y[b].qs + g * 32,
                                 _mm256_set1_epi16((short) (2 * (w[1] >> 28) + 1))));
        }

        sum += y[b].d * gk_h2f(x[b].d) * 0.125f * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

static void dot_iq2_xs_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq2_xs * x = (const qz_blk_iq2_xs *) vx;
    const qz_blk_q8_k   * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            const uint16_t * q = x[b].qs + 4 * g;

            uint8_t sgn[4];
            for (int t = 0; t < 4; ++t) {
                sgn[t] = qz_sign_mask_from_bits((uint8_t) (q[t] >> 9));
            }

            const __m256i mag = gk_grid4_u64(qz_grid_iq2_xs,
                                             q[0] & 511, q[1] & 511,
                                             q[2] & 511, q[3] & 511);

            // one scale per half-group
            acc = _mm256_add_epi32(acc,
                gk_lattice_group(mag, sgn, y[b].qs + g * 32,
                                 gk_scale16x2(2 * (x[b].scales[g] & 0xf) + 1,
                                              2 * (x[b].scales[g] >> 4)  + 1)));
        }

        sum += y[b].d * gk_h2f(x[b].d) * 0.125f * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

static void dot_iq2_s_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq2_s * x = (const qz_blk_iq2_s *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        const uint8_t * qs    = x[b].qs;
        const uint8_t * signs = x[b].qs + QZ_K / 8;

        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            const uint8_t qh = x[b].qh[g];

            const __m256i mag = gk_grid4_u64(qz_grid_iq2_s,
                qs[0] | (((qh >> 0) & 3) << 8), qs[1] | (((qh >> 2) & 3) << 8),
                qs[2] | (((qh >> 4) & 3) << 8), qs[3] | (((qh >> 6) & 3) << 8));

            // the sign masks are already four consecutive bytes
            acc = _mm256_add_epi32(acc,
                gk_lattice_group(mag, signs, y[b].qs + g * 32,
                                 gk_scale16x2(2 * (x[b].scales[g] & 0xf) + 1,
                                              2 * (x[b].scales[g] >> 4)  + 1)));

            qs    += 4;
            signs += 4;
        }

        sum += y[b].d * gk_h2f(x[b].d) * 0.125f * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

// The iq3 grids hold four magnitudes per entry, so a group of 32 is eight
// entries rather than four.
static inline __m256i gk_grid8_u32(const uint32_t * grid, const int * idx) {
    return _mm256_setr_epi32((int) grid[idx[0]], (int) grid[idx[1]],
                             (int) grid[idx[2]], (int) grid[idx[3]],
                             (int) grid[idx[4]], (int) grid[idx[5]],
                             (int) grid[idx[6]], (int) grid[idx[7]]);
}

static void dot_iq3_xxs_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq3_xxs * x = (const qz_blk_iq3_xxs *) vx;
    const qz_blk_q8_k    * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        const uint8_t * qs   = x[b].qs;
        const uint8_t * tail = x[b].qs + QZ_K / 4;

        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint32_t w;
            memcpy(&w, tail + 4 * g, sizeof(w));

            uint8_t sgn[4];
            int     idx[8];
            for (int t = 0; t < 4; ++t) {
                sgn[t] = qz_sign_mask_from_bits((uint8_t) ((w >> (7 * t)) & 127));
                idx[2 * t + 0] = qs[2 * t + 0];
                idx[2 * t + 1] = qs[2 * t + 1];
            }

            acc = _mm256_add_epi32(acc,
                gk_lattice_group(gk_grid8_u32(qz_grid_iq3_xxs, idx), sgn,
                                 y[b].qs + g * 32,
                                 _mm256_set1_epi16((short) (2 * (w >> 28) + 1))));

            qs += 8;
        }

        sum += y[b].d * gk_h2f(x[b].d) * 0.25f * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

static void dot_iq3_s_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq3_s * x = (const qz_blk_iq3_s *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        const uint8_t * qs    = x[b].qs;
        const uint8_t * signs = x[b].signs;

        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            const uint8_t qh = x[b].qh[g];

            int idx[8];
            for (int j = 0; j < 8; ++j) {
                idx[j] = qs[j] | (((qh >> j) & 1) << 8);
            }

            const int sc = (x[b].scales[g / 2] >> (4 * (g % 2))) & 0xf;

            acc = _mm256_add_epi32(acc,
                gk_lattice_group(gk_grid8_u32(qz_grid_iq3_s, idx), signs,
                                 y[b].qs + g * 32,
                                 _mm256_set1_epi16((short) (1 + 2 * sc))));

            qs    += 8;
            signs += 4;
        }

        sum += y[b].d * gk_h2f(x[b].d) * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

// iq1 is the one family whose grid entries are signed - ternary, with no
// separate sign mask - and whose values carry the +/-1/8 offset. Working in
// eighths, the weight is 8v +/- 1, and both halves of that come out of one pair
// of `maddubs`: v.a with the sign moved onto the activations as usual, and the
// plain activation sum for the offset. 8*p1 + p2 stays inside int16.
static inline __m256i gk_iq1_group(__m256i v, __m256i a, __m256i offs, __m256i scale) {
    const __m256i p1 = _mm256_maddubs_epi16(_mm256_sign_epi8(v, v),
                                            _mm256_sign_epi8(a, v));
    const __m256i p2 = _mm256_maddubs_epi16(_mm256_set1_epi8(1),
                                            _mm256_sign_epi8(a, offs));

    return _mm256_madd_epi16(
        _mm256_add_epi16(_mm256_slli_epi16(p1, 3), p2), scale);
}

static void dot_iq1_s_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq1_s * x = (const qz_blk_iq1_s *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            const uint16_t qh = x[b].qh[g];
            const uint8_t * q = x[b].qs + 4 * g;

            const __m256i v = gk_grid4_u64(qz_grid_iq1,
                q[0] | (int) (((qh >> 0) & 7) << 8), q[1] | (int) (((qh >> 3) & 7) << 8),
                q[2] | (int) (((qh >> 6) & 7) << 8), q[3] | (int) (((qh >> 9) & 7) << 8));

            // the offset's sign is constant over the whole group
            const __m256i offs = _mm256_set1_epi8((qh & 0x8000) ? -1 : 1);

            acc = _mm256_add_epi32(acc,
                gk_iq1_group(v, _mm256_loadu_si256((const __m256i *) (y[b].qs + g * 32)),
                             offs,
                             _mm256_set1_epi16((short) (2 * ((qh >> 12) & 7) + 1))));
        }

        sum += y[b].d * gk_h2f(x[b].d) * 0.125f * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

static void dot_iq1_m_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_iq1_m * x = (const qz_blk_iq1_m *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        uint16_t sc[4];
        memcpy(sc, x[b].scales, sizeof(sc));

        const qz_fp16_t dh = (qz_fp16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                          ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));

        const uint8_t * qs = x[b].qs;
        const uint8_t * qh = x[b].qh;

        __m256i acc = _mm256_setzero_si256();

        for (int g = 0; g < QZ_K / 32; ++g) {
            int idx[4];
            long long off[4];

            for (int t = 0; t < 4; ++t) {
                idx[t] = qs[t] | (int) (((qh[t / 2] >> (4 * (t % 2))) & 7) << 8);

                // unlike iq1_s the offset sign changes every eight elements, so
                // it is built per 64-bit lane rather than broadcast
                off[t] = (qh[t / 2] & (0x08u << (4 * (t % 2))))
                       ? (long long) 0xffffffffffffffffULL
                       : (long long) 0x0101010101010101ULL;
            }

            const __m256i v    = gk_grid4_u64(qz_grid_iq1, idx[0], idx[1], idx[2], idx[3]);
            const __m256i offs = _mm256_set_epi64x(off[3], off[2], off[1], off[0]);

            const int s0 = (sc[g / 2] >> (6 * (g % 2) + 0)) & 7;
            const int s1 = (sc[g / 2] >> (6 * (g % 2) + 3)) & 7;

            acc = _mm256_add_epi32(acc,
                gk_iq1_group(v, _mm256_loadu_si256((const __m256i *) (y[b].qs + g * 32)),
                             offs, gk_scale16x2(2 * s0 + 1, 2 * s1 + 1)));

            qs += 4;
            qh += 2;
        }

        sum += y[b].d * gk_h2f(dh) * 0.125f * (float) gk_hsum_i32(acc);
    }
    *s = sum;
}

// tq2_0 is the simplest super-block in the set and had the least done to it:
// two bits per weight, one f16 scale for all 256, no second level of scaling
// at all. Its shape is q2_K's - four 2-bit fields of the same 32-byte span,
// each field one group of 32 - minus the group scales, so `madd_epi16` folds
// in a constant 1 rather than a scale vector.
//
// The `- 1` every weight carries hoists out the same way q3_K's `- 4` does:
// sum((q-1)*a) = sum(q*a) - sum(a), and q8_K stores the activation sums. So
// the quants stay unsigned and the offset costs one `madd`, not a scalar pass.
_Static_assert(sizeof(((qz_blk_q8_k *) 0)->bsums) == 32,
               "tq2_0 reads a super-block's bsums as one 256-bit register");

static void dot_tq2_0_avx2(int n, float * s, const void * vx, const void * vy) {
    const qz_blk_tq2_0 * x = (const qz_blk_tq2_0 *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi16(1);

    float sum = 0.0f;

    for (int b = 0; b < n / QZ_K; ++b) {
        // With no group scale there is nothing to fold into `madd_epi16`, so
        // the widening is pure overhead per group - and it can wait. A
        // `maddubs` lane here is at most 2*127*2 = 508, so all eight groups
        // accumulate in int16 without coming close to overflow, and the whole
        // super-block widens once.
        __m256i acc16 = _mm256_setzero_si256();

        for (int half = 0; half < 2; ++half) {
            const __m256i v = _mm256_loadu_si256((const __m256i *) (x[b].qs + half * 32));

            for (int p = 0; p < 4; ++p) {
                const int g = half * 4 + p;

                const __m256i q = _mm256_and_si256(_mm256_srli_epi16(v, 2 * p), m3);
                const __m256i a = _mm256_loadu_si256((const __m256i *) (y[b].qs + g * 32));

                acc16 = _mm256_add_epi16(acc16, _mm256_maddubs_epi16(q, a));
            }
        }

        // sixteen int16 block sums are exactly one register, so the offset
        // term is a `madd` against ones rather than a scalar loop
        const __m256i bs = _mm256_loadu_si256((const __m256i *) y[b].bsums);

        const int32_t isum = gk_hsum_i32(
            _mm256_sub_epi32(_mm256_madd_epi16(acc16, one),
                             _mm256_madd_epi16(bs, one)));

        sum += y[b].d * gk_h2f(x[b].d) * (float) isum;
    }
    *s = sum;
}

#endif // __AVX2__

void gk_q3_k_unpack_scalar(const uint8_t * src, int8_t * out) {
    gk_q3_k_scales(src, out);
}

void gk_q3_k_unpack_vector(const uint8_t * src, int8_t * out) {
#if defined(__AVX2__)
    _mm_storeu_si128((__m128i *) out, gk_q3_k_scales_v(src));
#else
    gk_q3_k_scales(src, out);
#endif
}

// --------------------------------------------------------------------------
// the dots
// --------------------------------------------------------------------------

#define GK_DOT_ARGS \
    int n, float * s, size_t bs, const void * vx, size_t bx, \
    const void * vy, size_t by, int nrc

#define GK_DOT_PROLOGUE \
    GK_ASSERT(nrc == 1); \
    GK_UNUSED(bs); GK_UNUSED(bx); GK_UNUSED(by)

void gk_vec_dot_q8_0_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK8_0 == 0);

#if defined(__AVX2__)
    dot_q8_0_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q8_0 * x = (const qz_blk_q8_0 *) vx;
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy;

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK8_0; ++b) {
        const int32_t isum = gk_dot_i8(QZ_QK8_0, x[b].qs, y[b].qs);
        sum += gk_h2f(x[b].d) * gk_h2f(y[b].d) * (float) isum;
    }
    *s = sum;
#endif
}

void gk_vec_dot_q4_0_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK4_0 == 0);

#if defined(__AVX2__)
    dot_q4_0_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q4_0 * x = (const qz_blk_q4_0 *) vx;
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy;

    int8_t q[QZ_QK4_0];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK4_0; ++b) {
        unpack_q4_0(&x[b], q);
        const int32_t isum = gk_dot_i8(QZ_QK4_0, q, y[b].qs);
        sum += gk_h2f(x[b].d) * gk_h2f(y[b].d) * (float) isum;
    }
    *s = sum;
#endif
}

void gk_vec_dot_q5_0_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK5_0 == 0);

#if defined(__AVX2__)
    dot_q5_0_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q5_0 * x = (const qz_blk_q5_0 *) vx;
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy;

    int8_t q[QZ_QK5_0];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK5_0; ++b) {
        unpack_q5_0(&x[b], q);
        const int32_t isum = gk_dot_i8(QZ_QK5_0, q, y[b].qs);
        sum += gk_h2f(x[b].d) * gk_h2f(y[b].d) * (float) isum;
    }
    *s = sum;
#endif
}

// The formats with a minimum. The minimum is constant over the block, so it
// contributes m * sum(activations) - and q8_1 carries that sum already scaled,
// which is the whole reason it exists as a separate activation format.
void gk_vec_dot_q4_1_q8_1(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK4_1 == 0);

#if defined(__AVX2__)
    dot_q4_1_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q4_1 * x = (const qz_blk_q4_1 *) vx;
    const qz_blk_q8_1 * y = (const qz_blk_q8_1 *) vy;

    int8_t q[QZ_QK4_1];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK4_1; ++b) {
        unpack_q4_1(&x[b], q);
        const int32_t isum = gk_dot_i8(QZ_QK4_1, q, y[b].qs);

        sum += gk_h2f(x[b].d) * gk_h2f(y[b].d) * (float) isum
             + gk_h2f(x[b].m) * gk_h2f(y[b].s);
    }
    *s = sum;
#endif
}

void gk_vec_dot_q5_1_q8_1(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK5_1 == 0);

#if defined(__AVX2__)
    dot_q5_1_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q5_1 * x = (const qz_blk_q5_1 *) vx;
    const qz_blk_q8_1 * y = (const qz_blk_q8_1 *) vy;

    int8_t q[QZ_QK5_1];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK5_1; ++b) {
        unpack_q5_1(&x[b], q);
        const int32_t isum = gk_dot_i8(QZ_QK5_1, q, y[b].qs);

        sum += gk_h2f(x[b].d) * gk_h2f(y[b].d) * (float) isum
             + gk_h2f(x[b].m) * gk_h2f(y[b].s);
    }
    *s = sum;
#endif
}

// --------------------------------------------------------------------------
// K-quants
//
// A super-block of 256 carries per-group scales on top of the block scale, so
// the integer sums are accumulated per group and weighted by that group's
// scale before the block's float scale is applied once at the end.
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// codebook and micro-scaling formats
// --------------------------------------------------------------------------

// value = d * table[nibble], 32 per block
void gk_vec_dot_iq4_nl_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK4_NL == 0);

#if defined(__AVX2__)
    dot_iq4_nl_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq4_nl * x = (const qz_blk_iq4_nl *) vx;
    const qz_blk_q8_0   * y = (const qz_blk_q8_0 *) vy;

    int8_t q[QZ_QK4_NL];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK4_NL; ++b) {
        unpack_codebook(x[b].qs, QZ_QK4_NL, qz_iq4_values, q);
        sum += gk_h2f(x[b].d) * gk_h2f(y[b].d) *
               (float) gk_dot_i8(QZ_QK4_NL, q, y[b].qs);
    }
    *s = sum;
#endif
}

// MXFP4: one E8M0 exponent per 32 elements. The codebook is stored doubled and
// the exponent converter returns half the scale, so the product is exact.
void gk_vec_dot_mxfp4_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK_MXFP4 == 0);

#if defined(__AVX2__)
    dot_mxfp4_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_mxfp4 * x = (const qz_blk_mxfp4 *) vx;
    const qz_blk_q8_0  * y = (const qz_blk_q8_0 *) vy;

    int8_t q[QZ_QK_MXFP4];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK_MXFP4; ++b) {
        unpack_codebook(x[b].qs, QZ_QK_MXFP4, qz_e2m1_values, q);
        sum += qz_e8m0_to_fp32_half(x[b].e) * gk_h2f(y[b].d) *
               (float) gk_dot_i8(QZ_QK_MXFP4, q, y[b].qs);
    }
    *s = sum;
#endif
}

// NVFP4: 64 elements in four groups of 16, each group with its own UE4M3 scale.
// Those scales are floats rather than integers, so the integer sum is per group
// and only the 16-element inner product stays exact - still far less float work
// than decoding every element. Two q8_0 activation blocks cover one nvfp4 block.
void gk_vec_dot_nvfp4_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK_NVFP4 == 0);

#if defined(__AVX2__)
    dot_nvfp4_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_nvfp4 * x = (const qz_blk_nvfp4 *) vx;
    const qz_blk_q8_0  * y = (const qz_blk_q8_0 *) vy;

    const int sub  = QZ_QK_NVFP4_SUB;
    const int nsub = QZ_QK_NVFP4 / QZ_QK_NVFP4_SUB;
    const int per  = QZ_QK8_0 / QZ_QK_NVFP4_SUB;   // sub-groups per activation block

    int8_t q[QZ_QK_NVFP4_SUB];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK_NVFP4; ++b) {
        for (int j = 0; j < nsub / per; ++j) {
            const qz_blk_q8_0 * yb = &y[b * (nsub / per) + j];

            float part = 0.0f;
            for (int t = 0; t < per; ++t) {
                const int g = j * per + t;

                unpack_codebook(x[b].qs + g * (sub / 2), sub, qz_e2m1_values, q);
                part += qz_ue4m3_to_fp32(x[b].d[g]) *
                        (float) gk_dot_i8(sub, q, yb->qs + t * sub);
            }
            sum += gk_h2f(yb->d) * part;
        }
    }
    *s = sum;
#endif
}

// iq4_xs: eight groups of 32 with a 6-bit integer scale each, offset by 32.
void gk_vec_dot_iq4_xs_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq4_xs_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq4_xs * x = (const qz_blk_iq4_xs *) vx;
    const qz_blk_q8_k   * y = (const qz_blk_q8_k *) vy;

    int8_t q[32];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            const int ls = ((x[b].scales_l[g / 2] >> (4 * (g % 2))) & 0xf) |
                           (int) (((x[b].scales_h >> (2 * g)) & 3) << 4);

            unpack_codebook(x[b].qs + g * 16, 32, qz_iq4_values, q);
            acc += (int32_t) (ls - 32) * gk_dot_i8(32, q, y[b].qs + g * 32);
        }

        sum += y[b].d * gk_h2f(x[b].d) * (float) acc;
    }
    *s = sum;
#endif
}

// --------------------------------------------------------------------------
// ternary and the very low bit-width formats
//
// These have no per-group scale at all - one delta for the whole block - so the
// entire block is one integer dot and the float appears once.
// --------------------------------------------------------------------------

// TQ1_0 packs five ternary digits per byte as a base-3 numeral, with the tail
// of the block held four digits deep in qh. Unpacking follows the decoder's
// order exactly, which is what makes the group index the element index.
static void unpack_tq1_0(const qz_blk_tq1_0 * b, int8_t * out) {
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };

    int o = 0;
    size_t j = 0;

    for (int span = 32; span >= 16; span >>= 1) {
        for (int p = 0; p < 5; ++p) {
            for (int m = 0; m < span; ++m) {
                out[o++] = (int8_t) qz_unpack_trit(b->qs[j + m], pow3[p]);
            }
        }
        j += span;
    }

    for (int p = 0; p < 4; ++p) {
        for (size_t i = 0; i < sizeof(b->qh); ++i) {
            out[o++] = (int8_t) qz_unpack_trit(b->qh[i], pow3[p]);
        }
    }
}

void gk_vec_dot_tq1_0_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

    const qz_blk_tq1_0 * x = (const qz_blk_tq1_0 *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    int8_t q[QZ_K];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        unpack_tq1_0(&x[b], q);
        sum += y[b].d * gk_h2f(x[b].d) *
               (float) gk_dot_i8(QZ_K, q, y[b].qs);
    }
    *s = sum;
}

// TQ2_0: two bits per weight, q - 1 over four passes of a 32-byte span.
void gk_vec_dot_tq2_0_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_tq2_0_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_tq2_0 * x = (const qz_blk_tq2_0 *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    int8_t q[QZ_K];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int o = 0;
        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (int p = 0; p < 4; ++p) {
                for (int m = 0; m < 32; ++m) {
                    q[o++] = (int8_t) (((x[b].qs[j + m] >> (2 * p)) & 3) - 1);
                }
            }
        }

        sum += y[b].d * gk_h2f(x[b].d) *
               (float) gk_dot_i8(QZ_K, q, y[b].qs);
    }
    *s = sum;
#endif
}

// q1_0: only the sign survives, so every weight is +d or -d.
void gk_vec_dot_q1_0_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK1_0 == 0);

    const qz_blk_q1_0 * x = (const qz_blk_q1_0 *) vx;
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy;

    const int per = QZ_QK1_0 / QZ_QK8_0;   // activation blocks per weight block

    int8_t q[QZ_QK1_0];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK1_0; ++b) {
        for (int i = 0; i < QZ_QK1_0; ++i) {
            q[i] = ((x[b].qs[i >> 3] >> (i & 7)) & 1) ? 1 : -1;
        }

        const float d = gk_h2f(x[b].d);
        for (int j = 0; j < per; ++j) {
            const qz_blk_q8_0 * yb = &y[b * per + j];
            sum += d * gk_h2f(yb->d) *
                   (float) gk_dot_i8(QZ_QK8_0, q + j * QZ_QK8_0, yb->qs);
        }
    }
    *s = sum;
}

// q2_0: value = d * (q - 1), four elements per byte, lowest bits first.
void gk_vec_dot_q2_0_q8_0(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_QK2_0 == 0);

    const qz_blk_q2_0 * x = (const qz_blk_q2_0 *) vx;
    const qz_blk_q8_0 * y = (const qz_blk_q8_0 *) vy;

    const int per = QZ_QK2_0 / QZ_QK8_0;

    int8_t q[QZ_QK2_0];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_QK2_0; ++b) {
        for (int i = 0; i < QZ_QK2_0; ++i) {
            q[i] = (int8_t) (((x[b].qs[i / 4] >> ((i % 4) * 2)) & 3) - 1);
        }

        const float d = gk_h2f(x[b].d);
        for (int j = 0; j < per; ++j) {
            const qz_blk_q8_0 * yb = &y[b * per + j];
            sum += d * gk_h2f(yb->d) *
                   (float) gk_dot_i8(QZ_QK8_0, q + j * QZ_QK8_0, yb->qs);
        }
    }
    *s = sum;
}

// --------------------------------------------------------------------------
// lattice ("IQ") formats
//
// These look the least like an integer dot and turn out to be one anyway. Two
// facts do it:
//
//   * a group scale is always d * (0.5 + s) * 0.25 or d * (1 + 2s), and both
//     are `(2s + 1)` - an odd integer - times a factor constant over the whole
//     super-block. So the per-group scale is an integer and the float is one
//     multiply per super-block, exactly as for the K-quants.
//
//   * the codebooks hold small integer magnitudes (at most 62), which a stored
//     sign mask negates. iq1's entries are ternary and carry an extra offset of
//     QZ_IQ1_DELTA, and that is exactly 1/8 - so 8*(v + delta) is 8v +/- 1,
//     an integer, and the 1/8 folds into the same super-block factor.
//
// The sub-group is eight elements, so these unpack a group into a buffer and
// call the shared integer dot rather than staying in registers. At this size
// that is the right structure; see the note on q4_0 above for where it is not.
// --------------------------------------------------------------------------

static inline void expand_signs_i8(const uint8_t * mag, uint8_t signs, int n, int8_t * out) {
    for (int i = 0; i < n; ++i) {
        out[i] = (signs & (1u << i)) ? (int8_t) -(int) mag[i] : (int8_t) mag[i];
    }
}

// The scaled integer accumulator every one of these shares: `acc` sums
// scale_g * (group dot), and the block factor multiplies in once at the end.
#define GK_IQ_FINISH(block_factor) \
    sum += y[b].d * (block_factor) * (float) acc

void gk_vec_dot_iq2_xxs_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq2_xxs_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq2_xxs * x = (const qz_blk_iq2_xxs *) vx;
    const qz_blk_q8_k    * y = (const qz_blk_q8_k *) vy;

    int8_t v[32];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint32_t w[2];
            memcpy(w, x[b].qs + 4 * g, sizeof(w));

            const uint8_t * idx = (const uint8_t *) w;

            for (int s = 0; s < 4; ++s) {
                const uint8_t * mag = (const uint8_t *) (qz_grid_iq2_xxs + idx[s]);
                expand_signs_i8(mag,
                    qz_sign_mask_from_bits((uint8_t) ((w[1] >> (7 * s)) & 127)),
                    8, v + 8 * s);
            }

            // the group scale is the top nibble of the second word
            acc += (int32_t) (2 * (w[1] >> 28) + 1) * gk_dot_i8(32, v, y[b].qs + g * 32);
        }

        GK_IQ_FINISH(gk_h2f(x[b].d) * 0.125f);
    }
    *s = sum;
#endif
}

void gk_vec_dot_iq2_xs_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq2_xs_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq2_xs * x = (const qz_blk_iq2_xs *) vx;
    const qz_blk_q8_k   * y = (const qz_blk_q8_k *) vy;

    int8_t v[8];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            // one scale per half-group, so per sixteen elements
            const int sc[2] = { x[b].scales[g] & 0xf, x[b].scales[g] >> 4 };

            for (int t = 0; t < 4; ++t) {
                const uint16_t q = x[b].qs[4 * g + t];

                expand_signs_i8((const uint8_t *) (qz_grid_iq2_xs + (q & 511)),
                                qz_sign_mask_from_bits((uint8_t) (q >> 9)), 8, v);

                acc += (int32_t) (2 * sc[t / 2] + 1) *
                       gk_dot_i8(8, v, y[b].qs + g * 32 + t * 8);
            }
        }

        GK_IQ_FINISH(gk_h2f(x[b].d) * 0.125f);
    }
    *s = sum;
#endif
}

void gk_vec_dot_iq2_s_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq2_s_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq2_s * x = (const qz_blk_iq2_s *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    int8_t v[8];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        const uint8_t * qs    = x[b].qs;
        const uint8_t * signs = x[b].qs + QZ_K / 8;

        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            const int sc[2] = { x[b].scales[g] & 0xf, x[b].scales[g] >> 4 };

            for (int t = 0; t < 4; ++t) {
                // two extra index bits per sub-group live in qh
                const int idx = qs[t] | (((x[b].qh[g] >> (2 * t)) & 3) << 8);

                expand_signs_i8((const uint8_t *) (qz_grid_iq2_s + idx),
                                signs[t], 8, v);

                acc += (int32_t) (2 * sc[t / 2] + 1) *
                       gk_dot_i8(8, v, y[b].qs + g * 32 + t * 8);
            }

            qs    += 4;
            signs += 4;
        }

        GK_IQ_FINISH(gk_h2f(x[b].d) * 0.125f);
    }
    *s = sum;
#endif
}

void gk_vec_dot_iq3_xxs_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq3_xxs_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq3_xxs * x = (const qz_blk_iq3_xxs *) vx;
    const qz_blk_q8_k    * y = (const qz_blk_q8_k *) vy;

    int8_t v[32];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        const uint8_t * qs   = x[b].qs;
        const uint8_t * tail = x[b].qs + QZ_K / 4;

        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint32_t w;
            memcpy(&w, tail + 4 * g, sizeof(w));

            for (int t = 0; t < 4; ++t) {
                const uint8_t sg =
                    qz_sign_mask_from_bits((uint8_t) ((w >> (7 * t)) & 127));

                // two grid entries of four magnitudes each, the sign mask split
                // across them
                const uint8_t * m0 = (const uint8_t *) (qz_grid_iq3_xxs + qs[2 * t + 0]);
                const uint8_t * m1 = (const uint8_t *) (qz_grid_iq3_xxs + qs[2 * t + 1]);

                expand_signs_i8(m0, sg, 4, v + 8 * t);
                expand_signs_i8(m1, (uint8_t) (sg >> 4), 4, v + 8 * t + 4);
            }

            acc += (int32_t) (2 * (w >> 28) + 1) *
                   gk_dot_i8(32, v, y[b].qs + g * 32);

            qs += 8;
        }

        GK_IQ_FINISH(gk_h2f(x[b].d) * 0.25f);
    }
    *s = sum;
#endif
}

void gk_vec_dot_iq3_s_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq3_s_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq3_s * x = (const qz_blk_iq3_s *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    int8_t v[32];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        const uint8_t * qs    = x[b].qs;
        const uint8_t * signs = x[b].signs;

        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            const int sc = (x[b].scales[g / 2] >> (4 * (g % 2))) & 0xf;
            const uint8_t qh = x[b].qh[g];

            for (int t = 0; t < 4; ++t) {
                const int i0 = qs[2 * t + 0] | (((qh >> (2 * t + 0)) & 1) << 8);
                const int i1 = qs[2 * t + 1] | (((qh >> (2 * t + 1)) & 1) << 8);

                expand_signs_i8((const uint8_t *) (qz_grid_iq3_s + i0),
                                signs[t], 4, v + 8 * t);
                expand_signs_i8((const uint8_t *) (qz_grid_iq3_s + i1),
                                (uint8_t) (signs[t] >> 4), 4, v + 8 * t + 4);
            }

            acc += (int32_t) (1 + 2 * sc) * gk_dot_i8(32, v, y[b].qs + g * 32);

            qs    += 8;
            signs += 4;
        }

        GK_IQ_FINISH(gk_h2f(x[b].d));
    }
    *s = sum;
#endif
}

// The iq1 codebook is ternary and every value carries an offset of
// QZ_IQ1_DELTA = 1/8, whose sign the block stores. Working in eighths turns
// (v + delta) into the integer 8v +/- 1 and moves the 1/8 into the block factor.
static inline void expand_iq1_i8(const int8_t * v, int sgn, int8_t * out) {
    for (int i = 0; i < 8; ++i) {
        out[i] = (int8_t) (8 * (int) v[i] + sgn);
    }
}

void gk_vec_dot_iq1_s_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq1_s_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq1_s * x = (const qz_blk_iq1_s *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    int8_t v[32];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            const uint16_t qh  = x[b].qh[g];
            const int      sc  = (qh >> 12) & 7;
            const int      sgn = (qh & 0x8000) ? -1 : 1;

            for (int t = 0; t < 4; ++t) {
                const int idx = x[b].qs[4 * g + t] | (int) (((qh >> (3 * t)) & 7) << 8);
                expand_iq1_i8((const int8_t *) (qz_grid_iq1 + idx), sgn, v + 8 * t);
            }

            acc += (int32_t) (2 * sc + 1) * gk_dot_i8(32, v, y[b].qs + g * 32);
        }

        GK_IQ_FINISH(gk_h2f(x[b].d) * 0.125f);
    }
    *s = sum;
#endif
}

void gk_vec_dot_iq1_m_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_iq1_m_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_iq1_m * x = (const qz_blk_iq1_m *) vx;
    const qz_blk_q8_k  * y = (const qz_blk_q8_k *) vy;

    int8_t v[8];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        uint16_t sc[4];
        memcpy(sc, x[b].scales, sizeof(sc));

        // iq1_m has no d field: the block delta is spread over the top nibble
        // of each scale word
        const qz_fp16_t dh = (qz_fp16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                          ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));

        const uint8_t * qs = x[b].qs;
        const uint8_t * qh = x[b].qh;

        int32_t acc = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            // two 3-bit scales per group of 32, one per half
            const int s0 = (sc[g / 2] >> (6 * (g % 2) + 0)) & 7;
            const int s1 = (sc[g / 2] >> (6 * (g % 2) + 3)) & 7;

            for (int t = 0; t < 4; ++t) {
                const int idx = qs[t] | (int) (((qh[t / 2] >> (4 * (t % 2))) & 7) << 8);
                const int sgn = (qh[t / 2] & (0x08u << (4 * (t % 2)))) ? -1 : 1;

                expand_iq1_i8((const int8_t *) (qz_grid_iq1 + idx), sgn, v);

                acc += (int32_t) (2 * (t < 2 ? s0 : s1) + 1) *
                       gk_dot_i8(8, v, y[b].qs + g * 32 + t * 8);
            }

            qs += 4;
            qh += 2;
        }

        GK_IQ_FINISH(gk_h2f(dh) * 0.125f);
    }
    *s = sum;
#endif
}

#undef GK_IQ_FINISH

// The 2-bit payload of a 128-element half lives in one 32-byte span; group g
// reads byte (g%2)*16 + i of that span, taking the two bits at shift
// 2*((g%8)/2). Group g covers elements [16g, 16g+16), so it lines up with one
// of q8_K's block sums exactly.
static inline const uint8_t * gk_k16_group(const uint8_t * qs, int g, int * shift) {
    const int half = g / 8;
    const int rem  = g % 8;
    *shift = (rem / 2) * 2;
    return qs + half * 32 + (rem % 2) * 16;
}

// value = d * scale_g * q - dmin * min_g, q in [0,3], over 16 groups of 16.
void gk_vec_dot_q2_K_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_q2_K_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q2_k * x = (const qz_blk_q2_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int32_t sum_scaled = 0;
        int32_t sum_mins   = 0;

        for (int g = 0; g < QZ_K / 16; ++g) {
            int shift;
            const uint8_t * q = gk_k16_group(x[b].qs, g, &shift);
            const int8_t  * a = y[b].qs + g * 16;

            int32_t isum = 0;
            for (int i = 0; i < 16; ++i) {
                isum += (int32_t) ((q[i] >> shift) & 3) * a[i];
            }

            sum_scaled += (int32_t) (x[b].scales[g] & 0xf) * isum;
            sum_mins   += (int32_t) (x[b].scales[g] >> 4) * (int32_t) y[b].bsums[g];
        }

        sum += y[b].d * (gk_h2f(x[b].d)    * (float) sum_scaled
                       - gk_h2f(x[b].dmin) * (float) sum_mins);
    }
    *s = sum;
#endif
}

// value = d * scale_g * (q - 4), q in [0,7], over 16 groups of 16.
void gk_vec_dot_q3_K_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_q3_K_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q3_k * x = (const qz_blk_q3_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int8_t sc[16];
        gk_q3_k_scales(x[b].scales, sc);

        int32_t acc  = 0;
        int32_t bias = 0;

        for (int g = 0; g < QZ_K / 16; ++g) {
            int shift;
            const uint8_t * q = gk_k16_group(x[b].qs, g, &shift);
            const int8_t  * a = y[b].qs + g * 16;

            // hmask is indexed by the position within the 32-byte span; which
            // half it belongs to is carried by the bit index instead.
            const int bit = (g / 8) * 4 + (g % 8) / 2;
            const uint8_t * hm = x[b].hmask + (g % 2) * 16;

            int32_t isum = 0;
            for (int i = 0; i < 16; ++i) {
                const int raw = ((q[i] >> shift) & 3) | (((hm[i] >> bit) & 1) << 2);
                isum += (int32_t) raw * a[i];
            }

            acc  += (int32_t) sc[g] * isum;
            bias += (int32_t) sc[g] * (int32_t) y[b].bsums[g];
        }

        sum += y[b].d * gk_h2f(x[b].d) * (float) (acc - 4 * bias);
    }
    *s = sum;
#endif
}

// value = d * scale_g * q - dmin * min_g, over 8 groups of 32.
void gk_vec_dot_q4_K_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_q4_K_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q4_k * x = (const qz_blk_q4_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    int8_t q[32];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        const float d    = gk_h2f(x[b].d);
        const float dmin = gk_h2f(x[b].dmin);

        int32_t sum_scaled = 0; // sum over groups of scale_g * (q . a)
        int32_t sum_mins   = 0; // sum over groups of min_g * (sum of a)

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint8_t sc, mn;
            qz_unpack_scale_min_6bit(x[b].scales, g, &sc, &mn);

            const uint8_t * src = x[b].qs + (g / 2) * 32;
            const int shift = (g % 2) * 4;

            for (int i = 0; i < 32; ++i) {
                q[i] = (int8_t) ((src[i] >> shift) & 0xf);
            }

            sum_scaled += (int32_t) sc * gk_dot_i8(32, q, y[b].qs + g * 32);

            // the group's activations, from the two 16-wide sums q8_K stores
            sum_mins += (int32_t) mn *
                ((int32_t) y[b].bsums[2 * g] + (int32_t) y[b].bsums[2 * g + 1]);
        }

        sum += y[b].d * (d * (float) sum_scaled - dmin * (float) sum_mins);
    }
    *s = sum;
#endif
}

// q5_K: q4_K plus a fifth bit, element l of group g taking bit g of qh[l].
void gk_vec_dot_q5_K_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_q5_K_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q5_k * x = (const qz_blk_q5_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    int8_t q[32];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        int32_t sum_scaled = 0;
        int32_t sum_mins   = 0;

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint8_t sc, mn;
            qz_unpack_scale_min_6bit(x[b].scales, g, &sc, &mn);

            const uint8_t * src   = x[b].qs + (g / 2) * 32;
            const int       shift = (g % 2) * 4;
            const uint8_t   hmask = (uint8_t) (1u << g);

            for (int i = 0; i < 32; ++i) {
                q[i] = (int8_t) (((src[i] >> shift) & 0xf) | ((x[b].qh[i] & hmask) ? 16 : 0));
            }

            sum_scaled += (int32_t) sc * gk_dot_i8(32, q, y[b].qs + g * 32);
            sum_mins   += (int32_t) mn *
                ((int32_t) y[b].bsums[2 * g] + (int32_t) y[b].bsums[2 * g + 1]);
        }

        sum += y[b].d * (gk_h2f(x[b].d)    * (float) sum_scaled
                       - gk_h2f(x[b].dmin) * (float) sum_mins);
    }
    *s = sum;
#endif
}

// value = d * scale_g * (q - 32), over 16 groups of 16. The -32 factors out of
// each group as 32 * (sum of that group's activations), which is exactly what
// q8_K's bsums hold.
void gk_vec_dot_q6_K_q8_K(GK_DOT_ARGS) {
    GK_DOT_PROLOGUE;
    GK_ASSERT(n % QZ_K == 0);

#if defined(__AVX2__)
    dot_q6_K_avx2(n, s, vx, vy);
    return;
#else

    const qz_blk_q6_k * x = (const qz_blk_q6_k *) vx;
    const qz_blk_q8_k * y = (const qz_blk_q8_k *) vy;

    int8_t q[QZ_K];

    float sum = 0.0f;
    for (int b = 0; b < n / QZ_K; ++b) {
        const float d = gk_h2f(x[b].d);

        // Unpack the whole super-block once, in the order the decoder defines,
        // so the group index is simply the element index over 16.
        for (int half = 0; half < 2; ++half) {
            const uint8_t * ql = x[b].ql + half * 64;
            const uint8_t * qh = x[b].qh + half * 32;
            int8_t * o = q + half * 128;

            for (int i = 0; i < 32; ++i) {
                o[i]      = (int8_t) (((ql[i]      & 0xf) | (((qh[i] >> 0) & 3) << 4)) - 32);
                o[i + 32] = (int8_t) (((ql[i + 32] & 0xf) | (((qh[i] >> 2) & 3) << 4)) - 32);
                o[i + 64] = (int8_t) (((ql[i]      >> 4)  | (((qh[i] >> 4) & 3) << 4)) - 32);
                o[i + 96] = (int8_t) (((ql[i + 32] >> 4)  | (((qh[i] >> 6) & 3) << 4)) - 32);
            }
        }

        int32_t acc = 0;
        for (int g = 0; g < QZ_K / 16; ++g) {
            acc += (int32_t) x[b].scales[g] * gk_dot_i8(16, q + g * 16, y[b].qs + g * 16);
        }

        sum += y[b].d * d * (float) acc;
    }
    *s = sum;
#endif
}
