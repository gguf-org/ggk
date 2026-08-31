#pragma once

// Internal encoder/decoder entry points, one pair per format.
//
// Encoders take a whole row at a time: `imatrix`, when non-NULL, holds one
// importance value per column of the row. They return the number of bytes
// written. Decoders expand `k` consecutive elements back to f32.

#include "qz_format.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QZ_ENCODER(name) \
    size_t name(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix)

// block-scale formats (qz_pack.c)
QZ_ENCODER(qz_encode_q1_0);
QZ_ENCODER(qz_encode_q2_0);
QZ_ENCODER(qz_encode_q4_0);
QZ_ENCODER(qz_encode_q4_1);
QZ_ENCODER(qz_encode_q5_0);
QZ_ENCODER(qz_encode_q5_1);
QZ_ENCODER(qz_encode_q8_0);
QZ_ENCODER(qz_encode_q8_1);
QZ_ENCODER(qz_encode_q8_k);
QZ_ENCODER(qz_encode_mxfp4);
QZ_ENCODER(qz_encode_nvfp4);
QZ_ENCODER(qz_encode_tq1_0);
QZ_ENCODER(qz_encode_tq2_0);
QZ_ENCODER(qz_encode_iq4_nl);
QZ_ENCODER(qz_encode_iq4_xs);

// super-block formats (qz_super.c)
QZ_ENCODER(qz_encode_q2_k);
QZ_ENCODER(qz_encode_q3_k);
QZ_ENCODER(qz_encode_q4_k);
QZ_ENCODER(qz_encode_q5_k);
QZ_ENCODER(qz_encode_q6_k);

// lattice formats (qz_lattice.c)
QZ_ENCODER(qz_encode_iq1_s);
QZ_ENCODER(qz_encode_iq1_m);
QZ_ENCODER(qz_encode_iq2_xxs);
QZ_ENCODER(qz_encode_iq2_xs);
QZ_ENCODER(qz_encode_iq2_s);
QZ_ENCODER(qz_encode_iq3_xxs);
QZ_ENCODER(qz_encode_iq3_s);

// single-block helpers used across files
void qz_encode_q8_k_block(const float * x, qz_blk_q8_k * blk);

// lattice codebook lookup tables, built once (qz_lattice.c)
void qz_lattice_init(qz_type type);
void qz_lattice_free(void);

// decoders (qz_decode.c)
void qz_decode_q1_0   (const void * src, float * dst, int64_t k);
void qz_decode_q2_0   (const void * src, float * dst, int64_t k);
void qz_decode_q4_0   (const void * src, float * dst, int64_t k);
void qz_decode_q4_1   (const void * src, float * dst, int64_t k);
void qz_decode_q5_0   (const void * src, float * dst, int64_t k);
void qz_decode_q5_1   (const void * src, float * dst, int64_t k);
void qz_decode_q8_0   (const void * src, float * dst, int64_t k);
void qz_decode_q8_1   (const void * src, float * dst, int64_t k);
void qz_decode_mxfp4  (const void * src, float * dst, int64_t k);
void qz_decode_nvfp4  (const void * src, float * dst, int64_t k);
void qz_decode_tq1_0  (const void * src, float * dst, int64_t k);
void qz_decode_tq2_0  (const void * src, float * dst, int64_t k);
void qz_decode_q2_k   (const void * src, float * dst, int64_t k);
void qz_decode_q3_k   (const void * src, float * dst, int64_t k);
void qz_decode_q4_k   (const void * src, float * dst, int64_t k);
void qz_decode_q5_k   (const void * src, float * dst, int64_t k);
void qz_decode_q6_k   (const void * src, float * dst, int64_t k);
void qz_decode_q8_k   (const void * src, float * dst, int64_t k);
void qz_decode_iq1_s  (const void * src, float * dst, int64_t k);
void qz_decode_iq1_m  (const void * src, float * dst, int64_t k);
void qz_decode_iq2_xxs(const void * src, float * dst, int64_t k);
void qz_decode_iq2_xs (const void * src, float * dst, int64_t k);
void qz_decode_iq2_s  (const void * src, float * dst, int64_t k);
void qz_decode_iq3_xxs(const void * src, float * dst, int64_t k);
void qz_decode_iq3_s  (const void * src, float * dst, int64_t k);
void qz_decode_iq4_nl (const void * src, float * dst, int64_t k);
void qz_decode_iq4_xs (const void * src, float * dst, int64_t k);

// scale/min packing shared by q4_K and q5_K: eight 6-bit scales and eight
// 6-bit mins into twelve bytes
void qz_pack_scale_min_6bit(const uint8_t * scales, const uint8_t * mins, uint8_t * dst);
void qz_unpack_scale_min_6bit(const uint8_t * src, int group, uint8_t * scale, uint8_t * min);

// The same unpack as a header body. An encoder touches it once per group, so
// the call is free there and the out-of-line version stays the definition; a
// decoder that pulls all eight groups out ahead of a vectorised inner loop
// wants it inlined, and a call per group is measurable at that rate.
static inline void qz_unpack_scale_min_6bit_inline(const uint8_t * src, int group,
                                                   uint8_t * scale, uint8_t * min) {
    if (group < 4) {
        *scale = src[group] & 63;
        *min   = src[group + 4] & 63;
    } else {
        *scale = (uint8_t) ((src[group + 4] & 0xf) | ((src[group - 4] >> 6) << 4));
        *min   = (uint8_t) ((src[group + 4] >> 4)  | ((src[group]     >> 6) << 4));
    }
}

// A packed TQ1_0 byte holds its trits scaled up by 256/3^5; peeling digit n is
// a multiply by 3^n in 8-bit arithmetic followed by a multiply-and-shift.
static inline int qz_unpack_trit(uint8_t byte, int pow3) {
    const uint8_t v = (uint8_t) (byte * pow3);
    return (int) (((uint16_t) v * 3) >> 8) - 1;
}

#ifdef __cplusplus
}
#endif
