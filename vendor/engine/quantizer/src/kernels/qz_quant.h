#pragma once

// Public interface of the quantization kernels.
//
// The kernels in this directory are an independent implementation written
// against the GGUF on-disk tensor formats. Encoded blocks are byte-compatible
// with any GGUF reader; the numeric type ids below are part of the file format
// and are fixed. Nothing here is shared with, or derived from, another
// quantization library.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// a GGUF tensor carries at most four dimensions
#define QZ_MAX_DIMS 4

// --------------------------------------------------------------------------
// tensor element types
//
// The values are the type ids stored in GGUF files and must never change.
// Gaps are ids that exist in the format but that this library does not handle.
// --------------------------------------------------------------------------
typedef enum qz_type {
    QZ_TYPE_F32     = 0,
    QZ_TYPE_F16     = 1,
    QZ_TYPE_Q4_0    = 2,
    QZ_TYPE_Q4_1    = 3,
    QZ_TYPE_Q5_0    = 6,
    QZ_TYPE_Q5_1    = 7,
    QZ_TYPE_Q8_0    = 8,
    QZ_TYPE_Q8_1    = 9,
    QZ_TYPE_Q2_K    = 10,
    QZ_TYPE_Q3_K    = 11,
    QZ_TYPE_Q4_K    = 12,
    QZ_TYPE_Q5_K    = 13,
    QZ_TYPE_Q6_K    = 14,
    QZ_TYPE_Q8_K    = 15,
    QZ_TYPE_IQ2_XXS = 16,
    QZ_TYPE_IQ2_XS  = 17,
    QZ_TYPE_IQ3_XXS = 18,
    QZ_TYPE_IQ1_S   = 19,
    QZ_TYPE_IQ4_NL  = 20,
    QZ_TYPE_IQ3_S   = 21,
    QZ_TYPE_IQ2_S   = 22,
    QZ_TYPE_IQ4_XS  = 23,
    QZ_TYPE_I8      = 24,
    QZ_TYPE_I16     = 25,
    QZ_TYPE_I32     = 26,
    QZ_TYPE_I64     = 27,
    QZ_TYPE_F64     = 28,
    QZ_TYPE_IQ1_M   = 29,
    QZ_TYPE_BF16    = 30,
    QZ_TYPE_TQ1_0   = 34,
    QZ_TYPE_TQ2_0   = 35,
    QZ_TYPE_MXFP4   = 39,
    QZ_TYPE_NVFP4   = 40,
    QZ_TYPE_Q1_0    = 41,
    QZ_TYPE_Q2_0    = 42,
    QZ_TYPE_COUNT   = 43,
} qz_type;

// --------------------------------------------------------------------------
// type traits
// --------------------------------------------------------------------------

// human-readable name as used on the command line and in GGUF metadata;
// NULL for ids this library does not know
const char * qz_type_name(qz_type type);

// number of elements encoded by one block (1 for the plain float/int types)
int64_t qz_blck_size(qz_type type);

// encoded size of one block, in bytes
size_t qz_type_size(qz_type type);

// encoded size of a row of `ne` elements; `ne` must be a multiple of the block size
size_t qz_row_size(qz_type type, int64_t ne);

// true for the block-compressed types
bool qz_is_quantized(qz_type type);

// true if `type` can be produced by qz_quantize_chunk()
bool qz_is_quantize_target(qz_type type);

// true for types whose encoder needs per-column importances to produce
// usable output
bool qz_quantize_requires_imatrix(qz_type type);

// --------------------------------------------------------------------------
// scalar conversions
// --------------------------------------------------------------------------

typedef uint16_t qz_fp16_t;

typedef struct {
    uint16_t bits;
} qz_bf16_t;

float     qz_fp16_to_fp32(qz_fp16_t x);
qz_fp16_t qz_fp32_to_fp16(float x);

void qz_fp16_to_fp32_row(const qz_fp16_t * x, float * y, int64_t n);
void qz_fp32_to_fp16_row(const float * x, qz_fp16_t * y, int64_t n);
void qz_bf16_to_fp32_row(const qz_bf16_t * x, float * y, int64_t n);
void qz_fp32_to_bf16_row(const float * x, qz_bf16_t * y, int64_t n);

// --------------------------------------------------------------------------
// encode / decode
// --------------------------------------------------------------------------

// Prepares any lookup structures `type` needs. Called automatically by
// qz_quantize_chunk(); safe to call from several threads and to call twice.
void qz_quantize_init(qz_type type);

// Releases what qz_quantize_init() allocated. Not thread-safe against
// concurrent encoding.
void qz_quantize_free(void);

// Encodes `nrows` rows of `n_per_row` elements starting at element `start` of
// `src` into `dst`. `start` must be a whole number of rows. `imatrix`, when
// given, holds one importance value per column (n_per_row entries) and steers
// the encoders towards the columns that matter. Returns the number of bytes
// written.
size_t qz_quantize_chunk(
        qz_type       type,
        const float * src,
        void        * dst,
        int64_t       start,
        int64_t       nrows,
        int64_t       n_per_row,
        const float * imatrix);

// Decodes `k` consecutive elements to f32. Returns false only for type ids
// this library does not handle.
bool qz_dequantize(qz_type type, const void * src, float * dst, int64_t k);

// Sanity-checks a run of encoded bytes: rejects blocks whose scales are NaN or
// infinite, and whose codebook indices are out of range.
bool qz_validate_row_data(qz_type type, const void * data, size_t nbytes);

#ifdef __cplusplus
}
#endif
