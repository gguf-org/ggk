#pragma once

// On-disk block layouts of the GGUF tensor formats.
//
// Every struct here mirrors bytes that already exist in published model files,
// so the field order, widths and packing are fixed by the format. The comments
// spell out how the bits decode; the encoders and decoders in this directory
// are written against those descriptions.
//
// Two conventions run through the formats:
//   * "delta" (d) is an f16 multiplier; a stored integer q reconstructs as
//     d*q, optionally offset by a stored minimum m.
//   * 4-bit payloads pack the first half of a block into the low nibbles and
//     the second half into the high nibbles of the same bytes, so element j
//     and element j + n/2 share a byte.

#include "qz_quant.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__cplusplus)
#  define QZ_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define QZ_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#  define QZ_STATIC_ASSERT(cond, msg)
#endif

// super-block size shared by the K-quants, the ternary and the lattice formats
#define QZ_K 256

// --------------------------------------------------------------------------
// sign/magnitude and plain block-scale formats
// --------------------------------------------------------------------------

// 1 bit per weight: only the sign survives, all magnitudes collapse to d.
#define QZ_QK1_0 128
typedef struct {
    qz_fp16_t d;                 // magnitude shared by the whole block
    uint8_t   qs[QZ_QK1_0 / 8];  // bit j of byte j/8: 1 = +d, 0 = -d
} qz_blk_q1_0;
QZ_STATIC_ASSERT(sizeof(qz_blk_q1_0) == 2 + QZ_QK1_0 / 8, "q1_0 block layout");

// 2 bits per weight over 64 elements: value = d * (q - 1), q in [0,3], so the
// levels are -d, 0, +d and +2d - deliberately lopsided, which is what lets a
// two-bit format keep a little headroom above the mean magnitude.
#define QZ_QK2_0 64
typedef struct {
    qz_fp16_t d;
    uint8_t   qs[QZ_QK2_0 / 4];  // four elements per byte, lowest bits first
} qz_blk_q2_0;
QZ_STATIC_ASSERT(sizeof(qz_blk_q2_0) == 2 + QZ_QK2_0 / 4, "q2_0 block layout");

// 4 bits per weight, symmetric: value = d * (q - 8), q in [0,15].
#define QZ_QK4_0 32
typedef struct {
    qz_fp16_t d;
    uint8_t   qs[QZ_QK4_0 / 2];
} qz_blk_q4_0;
QZ_STATIC_ASSERT(sizeof(qz_blk_q4_0) == 2 + QZ_QK4_0 / 2, "q4_0 block layout");

// 4 bits per weight, asymmetric: value = d * q + m, q in [0,15].
#define QZ_QK4_1 32
typedef struct {
    qz_fp16_t d;
    qz_fp16_t m;
    uint8_t   qs[QZ_QK4_1 / 2];
} qz_blk_q4_1;
QZ_STATIC_ASSERT(sizeof(qz_blk_q4_1) == 4 + QZ_QK4_1 / 2, "q4_1 block layout");

// 5 bits per weight, symmetric: value = d * (q - 16). The fifth bit of element
// j lives in bit j of the little-endian word qh, the low four in qs.
#define QZ_QK5_0 32
typedef struct {
    qz_fp16_t d;
    uint8_t   qh[4];
    uint8_t   qs[QZ_QK5_0 / 2];
} qz_blk_q5_0;
QZ_STATIC_ASSERT(sizeof(qz_blk_q5_0) == 6 + QZ_QK5_0 / 2, "q5_0 block layout");

// 5 bits per weight, asymmetric: value = d * q + m.
#define QZ_QK5_1 32
typedef struct {
    qz_fp16_t d;
    qz_fp16_t m;
    uint8_t   qh[4];
    uint8_t   qs[QZ_QK5_1 / 2];
} qz_blk_q5_1;
QZ_STATIC_ASSERT(sizeof(qz_blk_q5_1) == 8 + QZ_QK5_1 / 2, "q5_1 block layout");

// 8 bits per weight, symmetric: value = d * q, q in [-127,127] (in practice
// [-128,127], the encoder keeps to the symmetric range).
#define QZ_QK8_0 32
typedef struct {
    qz_fp16_t d;
    int8_t    qs[QZ_QK8_0];
} qz_blk_q8_0;
QZ_STATIC_ASSERT(sizeof(qz_blk_q8_0) == 2 + QZ_QK8_0, "q8_0 block layout");

// q8_0 plus the block sum, used as a dot-product intermediate rather than a
// storage format.
#define QZ_QK8_1 32
typedef struct {
    qz_fp16_t d;
    qz_fp16_t s;  // d * sum(qs)
    int8_t    qs[QZ_QK8_1];
} qz_blk_q8_1;
QZ_STATIC_ASSERT(sizeof(qz_blk_q8_1) == 4 + QZ_QK8_1, "q8_1 block layout");

// --------------------------------------------------------------------------
// micro-scaling float formats
// --------------------------------------------------------------------------

// MXFP4: one E8M0 exponent per 32 elements, each element a 4-bit E2M1 code.
#define QZ_QK_MXFP4 32
typedef struct {
    uint8_t e;                     // shared exponent
    uint8_t qs[QZ_QK_MXFP4 / 2];   // E2M1 codes, low/high nibble halves
} qz_blk_mxfp4;
QZ_STATIC_ASSERT(sizeof(qz_blk_mxfp4) == 1 + QZ_QK_MXFP4 / 2, "mxfp4 block layout");

// NVFP4: 64 elements split into four groups of 16, each group with its own
// UE4M3 scale. The nibble halving is per group, not per block.
#define QZ_QK_NVFP4     64
#define QZ_QK_NVFP4_SUB 16
typedef struct {
    uint8_t d[QZ_QK_NVFP4 / QZ_QK_NVFP4_SUB];
    uint8_t qs[QZ_QK_NVFP4 / 2];
} qz_blk_nvfp4;
QZ_STATIC_ASSERT(sizeof(qz_blk_nvfp4) == 4 + QZ_QK_NVFP4 / 2, "nvfp4 block layout");

// --------------------------------------------------------------------------
// ternary formats: every weight is -1, 0 or +1 times a block delta
// --------------------------------------------------------------------------

// TQ1_0 packs five ternary digits into one byte as a base-3 numeral
// (3^5 = 243 fits). The tail of the block that does not divide evenly is held
// in qh at four digits per byte.
typedef struct {
    uint8_t   qs[(QZ_K - 4 * QZ_K / 64) / 5];  // 32 elements per 32 bytes, five digits deep
    uint8_t   qh[QZ_K / 64];                   // remaining 16 elements, four digits deep
    qz_fp16_t d;
} qz_blk_tq1_0;
QZ_STATIC_ASSERT(sizeof(qz_blk_tq1_0) == 2 + QZ_K / 64 + (QZ_K - 4 * QZ_K / 64) / 5, "tq1_0 block layout");

// TQ2_0 spends two bits per weight: q in [0,2] decodes as q - 1.
typedef struct {
    uint8_t   qs[QZ_K / 4];
    qz_fp16_t d;
} qz_blk_tq2_0;
QZ_STATIC_ASSERT(sizeof(qz_blk_tq2_0) == 2 + QZ_K / 4, "tq2_0 block layout");

// --------------------------------------------------------------------------
// super-block ("K") formats: 256 elements sharing an f16 scale, subdivided
// into groups that each carry their own small integer scale
// --------------------------------------------------------------------------

#define QZ_K_SCALE_SIZE 12  // packed 6-bit scale+min pairs of q4_K / q5_K

// 2 bits per weight over 16 groups of 16. Each group has a 4-bit scale (low
// nibble) and a 4-bit min (high nibble): value = d*scale*q - dmin*min.
// The quant bits of a 128-element half sit in one 32-byte span, two bits at a
// time, group g using shift 2*(g/2) of byte (g&1)*16 + l.
typedef struct {
    uint8_t   scales[QZ_K / 16];
    uint8_t   qs[QZ_K / 4];
    qz_fp16_t d;
    qz_fp16_t dmin;
} qz_blk_q2_k;
QZ_STATIC_ASSERT(sizeof(qz_blk_q2_k) == 4 + QZ_K / 16 + QZ_K / 4, "q2_K block layout");

// 3 bits per weight over 16 groups of 16, symmetric: value = d*scale*(q - 4),
// where the low two bits of q come from qs and the third from hmask - inverted,
// a set mask bit means the high bit is zero. Scales are 6-bit signed-by-offset
// (stored value minus 32) packed into 12 bytes: the low nibbles of the first
// eight bytes hold bits 0-3 of scales 0-7, the last four bytes hold bits 4-5 of
// all sixteen, two bits at a time.
typedef struct {
    uint8_t   hmask[QZ_K / 8];
    uint8_t   qs[QZ_K / 4];
    uint8_t   scales[12];
    qz_fp16_t d;
} qz_blk_q3_k;
QZ_STATIC_ASSERT(sizeof(qz_blk_q3_k) == 2 + QZ_K / 4 + QZ_K / 8 + 12, "q3_K block layout");

// 4 bits per weight over 8 groups of 32: value = d*scale*q - dmin*min. Scales
// and mins are 6-bit, packed into 12 bytes (see qz_pack_scale_min_6bit()).
typedef struct {
    qz_fp16_t d;
    qz_fp16_t dmin;
    uint8_t   scales[QZ_K_SCALE_SIZE];
    uint8_t   qs[QZ_K / 2];
} qz_blk_q4_k;
QZ_STATIC_ASSERT(sizeof(qz_blk_q4_k) == 4 + QZ_K_SCALE_SIZE + QZ_K / 2, "q4_K block layout");

// 5 bits per weight over 8 groups of 32, otherwise identical to q4_K; the
// fifth bit of element i is bit (i/32) of qh[i%32]... more precisely, group g
// uses mask 1<<g of qh[l] for its l-th element.
typedef struct {
    qz_fp16_t d;
    qz_fp16_t dmin;
    uint8_t   scales[QZ_K_SCALE_SIZE];
    uint8_t   qh[QZ_K / 8];
    uint8_t   qs[QZ_K / 2];
} qz_blk_q5_k;
QZ_STATIC_ASSERT(sizeof(qz_blk_q5_k) == 4 + QZ_K_SCALE_SIZE + QZ_K / 2 + QZ_K / 8, "q5_K block layout");

// 6 bits per weight over 16 groups of 16, symmetric: value = d*scale*(q - 32)
// with an int8 scale per group. Four bits come from ql, two from qh.
typedef struct {
    uint8_t   ql[QZ_K / 2];
    uint8_t   qh[QZ_K / 4];
    int8_t    scales[QZ_K / 16];
    qz_fp16_t d;
} qz_blk_q6_k;
QZ_STATIC_ASSERT(sizeof(qz_blk_q6_k) == 2 + QZ_K / 16 + 3 * QZ_K / 4, "q6_K block layout");

// 8-bit intermediate used when a K-quant needs an exact integer copy of a row.
typedef struct {
    float   d;
    int8_t  qs[QZ_K];
    int16_t bsums[QZ_K / 16];
} qz_blk_q8_k;
QZ_STATIC_ASSERT(sizeof(qz_blk_q8_k) == 4 + QZ_K + QZ_K / 16 * 2, "q8_K block layout");

// --------------------------------------------------------------------------
// lattice ("IQ") formats: groups of 8 weights are stored as an index into a
// fixed codebook of sign-free patterns plus a sign mask
// --------------------------------------------------------------------------

// 2.06 bpw. Each 32-element group packs into two little-endian words: the
// first holds four 8-bit codebook indices, the second four 7-bit sign masks
// and a 4-bit group scale in its top nibble.
typedef struct {
    qz_fp16_t d;
    uint16_t  qs[QZ_K / 8];
} qz_blk_iq2_xxs;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq2_xxs) == 2 + QZ_K / 8 * 2, "iq2_xxs block layout");

// 2.31 bpw. Codebook index (9 bits) and sign mask (7 bits) share a 16-bit
// word; scales move into their own nibble-packed array, one nibble per 16
// elements.
typedef struct {
    qz_fp16_t d;
    uint16_t  qs[QZ_K / 8];
    uint8_t   scales[QZ_K / 32];
} qz_blk_iq2_xs;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq2_xs) == 2 + QZ_K / 8 * 2 + QZ_K / 32, "iq2_xs block layout");

// 2.56 bpw. A 1024-entry codebook: 8 index bits in qs, two more in qh, and a
// full byte of signs per group of 8 in the second half of qs.
typedef struct {
    qz_fp16_t d;
    uint8_t   qs[QZ_K / 4];      // first half indices, second half sign masks
    uint8_t   qh[QZ_K / 32];
    uint8_t   scales[QZ_K / 32];
} qz_blk_iq2_s;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq2_s) == 2 + QZ_K / 4 + QZ_K / 16, "iq2_s block layout");

// 3.06 bpw. Two 4-element codebook entries per 8 weights, then a trailing
// array of 32-bit words holding four 7-bit sign masks and a 4-bit scale.
typedef struct {
    qz_fp16_t d;
    uint8_t   qs[3 * QZ_K / 8];  // first QZ_K/4 bytes indices, rest signs+scales
} qz_blk_iq3_xxs;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq3_xxs) == 2 + 3 * (QZ_K / 8), "iq3_xxs block layout");

// 3.44 bpw. 512-entry codebook of 4-element patterns: 8 index bits in qs, one
// in qh; signs get a byte per 8 weights; scales are 4-bit, one per 64.
#define QZ_IQ3S_N_SCALE (QZ_K / 64)
typedef struct {
    qz_fp16_t d;
    uint8_t   qs[QZ_K / 4];
    uint8_t   qh[QZ_K / 32];
    uint8_t   signs[QZ_K / 8];
    uint8_t   scales[QZ_IQ3S_N_SCALE];
} qz_blk_iq3_s;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq3_s) == 2 + 13 * (QZ_K / 32) + QZ_IQ3S_N_SCALE, "iq3_s block layout");

// 1.56 bpw. Codebook of 8-element ternary patterns, 11 index bits (8 in qs,
// 3 in qh), a 3-bit scale and a shared sign in the top bits of qh, and a fixed
// offset applied to every decoded value.
typedef struct {
    qz_fp16_t d;
    uint8_t   qs[QZ_K / 8];
    uint16_t  qh[QZ_K / 32];
} qz_blk_iq1_s;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq1_s) == 2 + QZ_K / 8 + QZ_K / 16, "iq1_s block layout");

// 1.75 bpw. Same codebook as iq1_s but with a per-16 offset sign and no f16
// delta of its own: the block delta is assembled from the top nibbles of the
// four scale words.
typedef struct {
    uint8_t qs[QZ_K / 8];
    uint8_t qh[QZ_K / 16];
    uint8_t scales[QZ_K / 32];
} qz_blk_iq1_m;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq1_m) == QZ_K / 8 + QZ_K / 16 + QZ_K / 32, "iq1_m block layout");

// the offset added to every iq1 codebook value before scaling
#define QZ_IQ1_DELTA 0.125f

// --------------------------------------------------------------------------
// non-linear 4-bit formats: the nibble is an index into a fixed 16-entry
// codebook rather than a plain integer
// --------------------------------------------------------------------------

#define QZ_QK4_NL 32
typedef struct {
    qz_fp16_t d;
    uint8_t   qs[QZ_QK4_NL / 2];
} qz_blk_iq4_nl;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq4_nl) == 2 + QZ_QK4_NL / 2, "iq4_nl block layout");

// super-block variant: 8 groups of 32 with a 6-bit scale each, split between a
// nibble in scales_l and two bits in scales_h; the stored scale is offset by 32.
typedef struct {
    qz_fp16_t d;
    uint16_t  scales_h;
    uint8_t   scales_l[QZ_K / 64];
    uint8_t   qs[QZ_K / 2];
} qz_blk_iq4_xs;
QZ_STATIC_ASSERT(sizeof(qz_blk_iq4_xs) == 2 + 2 + QZ_K / 64 + QZ_K / 2, "iq4_xs block layout");

#ifdef __cplusplus
}
#endif
