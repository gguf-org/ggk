// Type traits, row conversions and the encode/decode dispatchers.

#include "qz_common.h"
#include "qz_impl.h"

#include <stdio.h>

// --------------------------------------------------------------------------
// traits
// --------------------------------------------------------------------------

struct qz_traits {
    const char * name;
    int64_t      blck;
    size_t       size;
    bool         quantized;
};

static const struct qz_traits g_traits[QZ_TYPE_COUNT] = {
    [QZ_TYPE_F32]     = { "f32",     1,            sizeof(float),               false },
    [QZ_TYPE_F16]     = { "f16",     1,            sizeof(qz_fp16_t),           false },
    [QZ_TYPE_BF16]    = { "bf16",    1,            sizeof(qz_bf16_t),           false },
    [QZ_TYPE_F64]     = { "f64",     1,            sizeof(double),              false },
    [QZ_TYPE_I8]      = { "i8",      1,            sizeof(int8_t),              false },
    [QZ_TYPE_I16]     = { "i16",     1,            sizeof(int16_t),             false },
    [QZ_TYPE_I32]     = { "i32",     1,            sizeof(int32_t),             false },
    [QZ_TYPE_I64]     = { "i64",     1,            sizeof(int64_t),             false },

    [QZ_TYPE_Q1_0]    = { "q1_0",    QZ_QK1_0,     sizeof(qz_blk_q1_0),         true  },
    [QZ_TYPE_Q2_0]    = { "q2_0",    QZ_QK2_0,     sizeof(qz_blk_q2_0),         true  },
    [QZ_TYPE_Q4_0]    = { "q4_0",    QZ_QK4_0,     sizeof(qz_blk_q4_0),         true  },
    [QZ_TYPE_Q4_1]    = { "q4_1",    QZ_QK4_1,     sizeof(qz_blk_q4_1),         true  },
    [QZ_TYPE_Q5_0]    = { "q5_0",    QZ_QK5_0,     sizeof(qz_blk_q5_0),         true  },
    [QZ_TYPE_Q5_1]    = { "q5_1",    QZ_QK5_1,     sizeof(qz_blk_q5_1),         true  },
    [QZ_TYPE_Q8_0]    = { "q8_0",    QZ_QK8_0,     sizeof(qz_blk_q8_0),         true  },
    [QZ_TYPE_Q8_1]    = { "q8_1",    QZ_QK8_1,     sizeof(qz_blk_q8_1),         true  },

    [QZ_TYPE_Q2_K]    = { "q2_K",    QZ_K,         sizeof(qz_blk_q2_k),         true  },
    [QZ_TYPE_Q3_K]    = { "q3_K",    QZ_K,         sizeof(qz_blk_q3_k),         true  },
    [QZ_TYPE_Q4_K]    = { "q4_K",    QZ_K,         sizeof(qz_blk_q4_k),         true  },
    [QZ_TYPE_Q5_K]    = { "q5_K",    QZ_K,         sizeof(qz_blk_q5_k),         true  },
    [QZ_TYPE_Q6_K]    = { "q6_K",    QZ_K,         sizeof(qz_blk_q6_k),         true  },
    [QZ_TYPE_Q8_K]    = { "q8_K",    QZ_K,         sizeof(qz_blk_q8_k),         true  },

    [QZ_TYPE_IQ2_XXS] = { "iq2_xxs", QZ_K,         sizeof(qz_blk_iq2_xxs),      true  },
    [QZ_TYPE_IQ2_XS]  = { "iq2_xs",  QZ_K,         sizeof(qz_blk_iq2_xs),       true  },
    [QZ_TYPE_IQ2_S]   = { "iq2_s",   QZ_K,         sizeof(qz_blk_iq2_s),        true  },
    [QZ_TYPE_IQ3_XXS] = { "iq3_xxs", QZ_K,         sizeof(qz_blk_iq3_xxs),      true  },
    [QZ_TYPE_IQ3_S]   = { "iq3_s",   QZ_K,         sizeof(qz_blk_iq3_s),        true  },
    [QZ_TYPE_IQ1_S]   = { "iq1_s",   QZ_K,         sizeof(qz_blk_iq1_s),        true  },
    [QZ_TYPE_IQ1_M]   = { "iq1_m",   QZ_K,         sizeof(qz_blk_iq1_m),        true  },
    [QZ_TYPE_IQ4_NL]  = { "iq4_nl",  QZ_QK4_NL,    sizeof(qz_blk_iq4_nl),       true  },
    [QZ_TYPE_IQ4_XS]  = { "iq4_xs",  QZ_K,         sizeof(qz_blk_iq4_xs),       true  },

    [QZ_TYPE_TQ1_0]   = { "tq1_0",   QZ_K,         sizeof(qz_blk_tq1_0),        true  },
    [QZ_TYPE_TQ2_0]   = { "tq2_0",   QZ_K,         sizeof(qz_blk_tq2_0),        true  },
    [QZ_TYPE_MXFP4]   = { "mxfp4",   QZ_QK_MXFP4,  sizeof(qz_blk_mxfp4),        true  },
    [QZ_TYPE_NVFP4]   = { "nvfp4",   QZ_QK_NVFP4,  sizeof(qz_blk_nvfp4),        true  },
};

static bool qz_type_known(qz_type type) {
    return (int) type >= 0 && (int) type < QZ_TYPE_COUNT && g_traits[type].name != NULL;
}

const char * qz_type_name(qz_type type) {
    return qz_type_known(type) ? g_traits[type].name : NULL;
}

int64_t qz_blck_size(qz_type type) {
    return qz_type_known(type) ? g_traits[type].blck : 0;
}

size_t qz_type_size(qz_type type) {
    return qz_type_known(type) ? g_traits[type].size : 0;
}

size_t qz_row_size(qz_type type, int64_t ne) {
    const int64_t blck = qz_blck_size(type);
    if (blck <= 0 || ne % blck != 0) {
        return 0;
    }
    return g_traits[type].size * (size_t) (ne / blck);
}

bool qz_is_quantized(qz_type type) {
    return qz_type_known(type) && g_traits[type].quantized;
}

bool qz_quantize_requires_imatrix(qz_type type) {
    // the 2-bit lattice formats and the 1.5-bit one have too little room to
    // absorb a bad scale choice; without importances their output is not worth
    // shipping, so the caller has to supply them
    return type == QZ_TYPE_IQ2_XXS || type == QZ_TYPE_IQ2_XS || type == QZ_TYPE_IQ1_S;
}

bool qz_is_quantize_target(qz_type type) {
    switch (type) {
        case QZ_TYPE_F32:
        case QZ_TYPE_F16:
        case QZ_TYPE_BF16:
        case QZ_TYPE_Q1_0:
        case QZ_TYPE_Q2_0:
        case QZ_TYPE_Q4_0:
        case QZ_TYPE_Q4_1:
        case QZ_TYPE_Q5_0:
        case QZ_TYPE_Q5_1:
        case QZ_TYPE_Q8_0:
        case QZ_TYPE_MXFP4:
        case QZ_TYPE_NVFP4:
        case QZ_TYPE_Q2_K:
        case QZ_TYPE_Q3_K:
        case QZ_TYPE_Q4_K:
        case QZ_TYPE_Q5_K:
        case QZ_TYPE_Q6_K:
        case QZ_TYPE_TQ1_0:
        case QZ_TYPE_TQ2_0:
        case QZ_TYPE_IQ2_XXS:
        case QZ_TYPE_IQ2_XS:
        case QZ_TYPE_IQ2_S:
        case QZ_TYPE_IQ3_XXS:
        case QZ_TYPE_IQ3_S:
        case QZ_TYPE_IQ1_S:
        case QZ_TYPE_IQ1_M:
        case QZ_TYPE_IQ4_NL:
        case QZ_TYPE_IQ4_XS:
        case QZ_TYPE_Q8_1:
        case QZ_TYPE_Q8_K:
        case QZ_TYPE_I8:
        case QZ_TYPE_I16:
        case QZ_TYPE_I32:
        case QZ_TYPE_I64:
        case QZ_TYPE_F64:
            return true;
        default:
            return false;
    }
}

// --------------------------------------------------------------------------
// scalar and row conversions
// --------------------------------------------------------------------------

float qz_fp16_to_fp32(qz_fp16_t x) {
    return qz_h2f(x);
}

qz_fp16_t qz_fp32_to_fp16(float x) {
    return qz_f2h(x);
}

void qz_fp16_to_fp32_row(const qz_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = qz_h2f(x[i]);
    }
}

void qz_fp32_to_fp16_row(const float * x, qz_fp16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = qz_f2h(x[i]);
    }
}

void qz_bf16_to_fp32_row(const qz_bf16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = qz_bf2f(x[i]);
    }
}

void qz_fp32_to_bf16_row(const float * x, qz_bf16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = qz_f2bf(x[i]);
    }
}

// float <-> raw integer / f64 rows, used when a plain integer tensor type is
// the source or the target: round to nearest and clamp to the type's range
static void qz_fp32_to_i8_row(const float * x, int8_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (int8_t) qz_clampi(qz_lround(x[i]), INT8_MIN, INT8_MAX);
    }
}

static void qz_fp32_to_i16_row(const float * x, int16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (int16_t) qz_clampi(qz_lround(x[i]), INT16_MIN, INT16_MAX);
    }
}

static void qz_fp32_to_i32_row(const float * x, int32_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        // 2147483647.0f rounds up to 2^31, so >= catches everything that
        // would overflow the cast
        if (v >= 2147483647.0f) {
            y[i] = INT32_MAX;
        } else if (v <= -2147483648.0f) {
            y[i] = INT32_MIN;
        } else {
            y[i] = (int32_t) lrintf(v);
        }
    }
}

static void qz_fp32_to_i64_row(const float * x, int64_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (v >= 9223372036854775807.0f) {
            y[i] = INT64_MAX;
        } else if (v <= -9223372036854775808.0f) {
            y[i] = INT64_MIN;
        } else {
            y[i] = llrintf(v);
        }
    }
}

static void qz_fp32_to_fp64_row(const float * x, double * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (double) x[i];
    }
}

static void qz_i8_to_fp32_row(const int8_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (float) x[i];
    }
}

static void qz_i16_to_fp32_row(const int16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (float) x[i];
    }
}

static void qz_i32_to_fp32_row(const int32_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (float) x[i];
    }
}

static void qz_i64_to_fp32_row(const int64_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (float) x[i];
    }
}

static void qz_fp64_to_fp32_row(const double * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (float) x[i];
    }
}

// --------------------------------------------------------------------------
// encode
// --------------------------------------------------------------------------

typedef size_t (*qz_encode_fn)(const float *, void *, int64_t, int64_t, const float *);

static qz_encode_fn qz_encoder_for(qz_type type) {
    switch (type) {
        case QZ_TYPE_Q1_0:    return qz_encode_q1_0;
        case QZ_TYPE_Q2_0:    return qz_encode_q2_0;
        case QZ_TYPE_Q4_0:    return qz_encode_q4_0;
        case QZ_TYPE_Q4_1:    return qz_encode_q4_1;
        case QZ_TYPE_Q5_0:    return qz_encode_q5_0;
        case QZ_TYPE_Q5_1:    return qz_encode_q5_1;
        case QZ_TYPE_Q8_0:    return qz_encode_q8_0;
        case QZ_TYPE_Q8_1:    return qz_encode_q8_1;
        case QZ_TYPE_Q8_K:    return qz_encode_q8_k;
        case QZ_TYPE_MXFP4:   return qz_encode_mxfp4;
        case QZ_TYPE_NVFP4:   return qz_encode_nvfp4;
        case QZ_TYPE_TQ1_0:   return qz_encode_tq1_0;
        case QZ_TYPE_TQ2_0:   return qz_encode_tq2_0;
        case QZ_TYPE_Q2_K:    return qz_encode_q2_k;
        case QZ_TYPE_Q3_K:    return qz_encode_q3_k;
        case QZ_TYPE_Q4_K:    return qz_encode_q4_k;
        case QZ_TYPE_Q5_K:    return qz_encode_q5_k;
        case QZ_TYPE_Q6_K:    return qz_encode_q6_k;
        case QZ_TYPE_IQ1_S:   return qz_encode_iq1_s;
        case QZ_TYPE_IQ1_M:   return qz_encode_iq1_m;
        case QZ_TYPE_IQ2_XXS: return qz_encode_iq2_xxs;
        case QZ_TYPE_IQ2_XS:  return qz_encode_iq2_xs;
        case QZ_TYPE_IQ2_S:   return qz_encode_iq2_s;
        case QZ_TYPE_IQ3_XXS: return qz_encode_iq3_xxs;
        case QZ_TYPE_IQ3_S:   return qz_encode_iq3_s;
        case QZ_TYPE_IQ4_NL:  return qz_encode_iq4_nl;
        case QZ_TYPE_IQ4_XS:  return qz_encode_iq4_xs;
        default:              return NULL;
    }
}

size_t qz_quantize_chunk(qz_type type, const float * src, void * dst, int64_t start, int64_t nrows,
                         int64_t n_per_row, const float * imatrix) {
    const int64_t blck = qz_blck_size(type);
    if (blck <= 0 || n_per_row % blck != 0 || start % n_per_row != 0) {
        return 0;
    }
    if (qz_quantize_requires_imatrix(type) && imatrix == NULL) {
        return 0;
    }

    const int64_t n = nrows * n_per_row;

    switch (type) {
        case QZ_TYPE_F32:
            memcpy((float *) dst + start, src + start, (size_t) n * sizeof(float));
            return (size_t) n * sizeof(float);
        case QZ_TYPE_F16:
            qz_fp32_to_fp16_row(src + start, (qz_fp16_t *) dst + start, n);
            return (size_t) n * sizeof(qz_fp16_t);
        case QZ_TYPE_BF16:
            qz_fp32_to_bf16_row(src + start, (qz_bf16_t *) dst + start, n);
            return (size_t) n * sizeof(qz_bf16_t);
        case QZ_TYPE_F64:
            qz_fp32_to_fp64_row(src + start, (double *) dst + start, n);
            return (size_t) n * sizeof(double);
        case QZ_TYPE_I8:
            qz_fp32_to_i8_row(src + start, (int8_t *) dst + start, n);
            return (size_t) n * sizeof(int8_t);
        case QZ_TYPE_I16:
            qz_fp32_to_i16_row(src + start, (int16_t *) dst + start, n);
            return (size_t) n * sizeof(int16_t);
        case QZ_TYPE_I32:
            qz_fp32_to_i32_row(src + start, (int32_t *) dst + start, n);
            return (size_t) n * sizeof(int32_t);
        case QZ_TYPE_I64:
            qz_fp32_to_i64_row(src + start, (int64_t *) dst + start, n);
            return (size_t) n * sizeof(int64_t);
        default:
            break;
    }

    const qz_encode_fn fn = qz_encoder_for(type);
    if (!fn) {
        return 0;
    }

    qz_quantize_init(type);

    const size_t row_size  = qz_row_size(type, n_per_row);
    const size_t start_row = (size_t) (start / n_per_row);

    return fn(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
}

// --------------------------------------------------------------------------
// decode
// --------------------------------------------------------------------------

bool qz_dequantize(qz_type type, const void * src, float * dst, int64_t k) {
    switch (type) {
        case QZ_TYPE_F32:     memcpy(dst, src, (size_t) k * sizeof(float));              break;
        case QZ_TYPE_F16:     qz_fp16_to_fp32_row((const qz_fp16_t *) src, dst, k);      break;
        case QZ_TYPE_BF16:    qz_bf16_to_fp32_row((const qz_bf16_t *) src, dst, k);      break;
        case QZ_TYPE_F64:     qz_fp64_to_fp32_row((const double *) src, dst, k);         break;
        case QZ_TYPE_I8:      qz_i8_to_fp32_row((const int8_t *) src, dst, k);           break;
        case QZ_TYPE_I16:     qz_i16_to_fp32_row((const int16_t *) src, dst, k);         break;
        case QZ_TYPE_I32:     qz_i32_to_fp32_row((const int32_t *) src, dst, k);         break;
        case QZ_TYPE_I64:     qz_i64_to_fp32_row((const int64_t *) src, dst, k);         break;
        case QZ_TYPE_Q1_0:    qz_decode_q1_0(src, dst, k);                               break;
        case QZ_TYPE_Q2_0:    qz_decode_q2_0(src, dst, k);                               break;
        case QZ_TYPE_Q4_0:    qz_decode_q4_0(src, dst, k);                               break;
        case QZ_TYPE_Q4_1:    qz_decode_q4_1(src, dst, k);                               break;
        case QZ_TYPE_Q5_0:    qz_decode_q5_0(src, dst, k);                               break;
        case QZ_TYPE_Q5_1:    qz_decode_q5_1(src, dst, k);                               break;
        case QZ_TYPE_Q8_0:    qz_decode_q8_0(src, dst, k);                               break;
        case QZ_TYPE_Q8_1:    qz_decode_q8_1(src, dst, k);                               break;
        case QZ_TYPE_MXFP4:   qz_decode_mxfp4(src, dst, k);                              break;
        case QZ_TYPE_NVFP4:   qz_decode_nvfp4(src, dst, k);                              break;
        case QZ_TYPE_TQ1_0:   qz_decode_tq1_0(src, dst, k);                              break;
        case QZ_TYPE_TQ2_0:   qz_decode_tq2_0(src, dst, k);                              break;
        case QZ_TYPE_Q2_K:    qz_decode_q2_k(src, dst, k);                               break;
        case QZ_TYPE_Q3_K:    qz_decode_q3_k(src, dst, k);                               break;
        case QZ_TYPE_Q4_K:    qz_decode_q4_k(src, dst, k);                               break;
        case QZ_TYPE_Q5_K:    qz_decode_q5_k(src, dst, k);                               break;
        case QZ_TYPE_Q6_K:    qz_decode_q6_k(src, dst, k);                               break;
        case QZ_TYPE_Q8_K:    qz_decode_q8_k(src, dst, k);                               break;
        case QZ_TYPE_IQ1_S:   qz_decode_iq1_s(src, dst, k);                              break;
        case QZ_TYPE_IQ1_M:   qz_decode_iq1_m(src, dst, k);                              break;
        case QZ_TYPE_IQ2_XXS: qz_decode_iq2_xxs(src, dst, k);                            break;
        case QZ_TYPE_IQ2_XS:  qz_decode_iq2_xs(src, dst, k);                             break;
        case QZ_TYPE_IQ2_S:   qz_decode_iq2_s(src, dst, k);                              break;
        case QZ_TYPE_IQ3_XXS: qz_decode_iq3_xxs(src, dst, k);                            break;
        case QZ_TYPE_IQ3_S:   qz_decode_iq3_s(src, dst, k);                              break;
        case QZ_TYPE_IQ4_NL:  qz_decode_iq4_nl(src, dst, k);                             break;
        case QZ_TYPE_IQ4_XS:  qz_decode_iq4_xs(src, dst, k);                             break;
        default:
            return false;  // type ids this library does not handle
    }
    return true;
}

// --------------------------------------------------------------------------
// validation
//
// Cheap structural check for tensor data read from a file: a block whose scale
// is not a finite number will poison everything downstream, and a codebook
// index past the end of its grid would read out of bounds during decoding.
// (The lattice indices are range-limited by their bit widths, so only the
// scales need checking there.)
// --------------------------------------------------------------------------

static bool qz_half_ok(qz_fp16_t h) {
    return (h & 0x7c00u) != 0x7c00u;  // not inf, not nan
}

bool qz_validate_row_data(qz_type type, const void * data, size_t nbytes) {
    const size_t bsize = qz_type_size(type);
    if (bsize == 0 || nbytes % bsize != 0) {
        return false;
    }
    const size_t nblk = nbytes / bsize;

    switch (type) {
        case QZ_TYPE_F16: {
            const qz_fp16_t * h = (const qz_fp16_t *) data;
            for (size_t i = 0; i < nbytes / sizeof(qz_fp16_t); ++i) {
                if (!qz_half_ok(h[i])) {
                    return false;
                }
            }
            return true;
        }
        case QZ_TYPE_F32: {
            const float * f = (const float *) data;
            for (size_t i = 0; i < nbytes / sizeof(float); ++i) {
                if (!isfinite(f[i])) {
                    return false;
                }
            }
            return true;
        }
        case QZ_TYPE_BF16: {
            const qz_bf16_t * h = (const qz_bf16_t *) data;
            for (size_t i = 0; i < nbytes / sizeof(qz_bf16_t); ++i) {
                if ((h[i].bits & 0x7f80u) == 0x7f80u) {  // inf or nan
                    return false;
                }
            }
            return true;
        }
        case QZ_TYPE_F64: {
            const double * f = (const double *) data;
            for (size_t i = 0; i < nbytes / sizeof(double); ++i) {
                if (!isfinite(f[i])) {
                    return false;
                }
            }
            return true;
        }
        case QZ_TYPE_I8:
        case QZ_TYPE_I16:
        case QZ_TYPE_I32:
        case QZ_TYPE_I64:
            return true;  // every bit pattern is a valid integer
        default:
            break;
    }

    // every remaining quantized type starts or ends its block with one or two
    // f16 scales; locate them by type rather than by guessing
    for (size_t i = 0; i < nblk; ++i) {
        const uint8_t * blk = (const uint8_t *) data + i * bsize;
        const qz_fp16_t * h;

        switch (type) {
            case QZ_TYPE_Q1_0: case QZ_TYPE_Q2_0: case QZ_TYPE_Q4_0: case QZ_TYPE_Q5_0: case QZ_TYPE_Q8_0:
            case QZ_TYPE_IQ4_NL: case QZ_TYPE_IQ4_XS: case QZ_TYPE_IQ2_XXS: case QZ_TYPE_IQ2_XS:
            case QZ_TYPE_IQ2_S: case QZ_TYPE_IQ3_XXS: case QZ_TYPE_IQ3_S: case QZ_TYPE_IQ1_S:
            case QZ_TYPE_Q4_K: case QZ_TYPE_Q5_K:
                h = (const qz_fp16_t *) blk;
                if (!qz_half_ok(h[0])) {
                    return false;
                }
                if ((type == QZ_TYPE_Q4_K || type == QZ_TYPE_Q5_K) && !qz_half_ok(h[1])) {
                    return false;
                }
                break;
            case QZ_TYPE_Q4_1: case QZ_TYPE_Q5_1: case QZ_TYPE_Q8_1:
                h = (const qz_fp16_t *) blk;
                if (!qz_half_ok(h[0]) || !qz_half_ok(h[1])) {
                    return false;
                }
                break;
            case QZ_TYPE_Q2_K:
                h = (const qz_fp16_t *) (blk + offsetof(qz_blk_q2_k, d));
                if (!qz_half_ok(h[0]) || !qz_half_ok(h[1])) {
                    return false;
                }
                break;
            case QZ_TYPE_Q3_K:
                h = (const qz_fp16_t *) (blk + offsetof(qz_blk_q3_k, d));
                if (!qz_half_ok(h[0])) {
                    return false;
                }
                break;
            case QZ_TYPE_Q6_K:
                h = (const qz_fp16_t *) (blk + offsetof(qz_blk_q6_k, d));
                if (!qz_half_ok(h[0])) {
                    return false;
                }
                break;
            case QZ_TYPE_TQ1_0:
                h = (const qz_fp16_t *) (blk + offsetof(qz_blk_tq1_0, d));
                if (!qz_half_ok(h[0])) {
                    return false;
                }
                break;
            case QZ_TYPE_TQ2_0:
                h = (const qz_fp16_t *) (blk + offsetof(qz_blk_tq2_0, d));
                if (!qz_half_ok(h[0])) {
                    return false;
                }
                break;
            case QZ_TYPE_Q8_K: {
                float df;
                memcpy(&df, blk + offsetof(qz_blk_q8_k, d), sizeof(df));
                if (!isfinite(df)) {
                    return false;
                }
                break;
            }
            case QZ_TYPE_IQ1_M: {
                // the delta is spread over the top nibbles of the scale words
                const qz_blk_iq1_m * b = (const qz_blk_iq1_m *) blk;
                uint16_t sc[4];
                memcpy(sc, b->scales, sizeof(sc));
                const qz_fp16_t dh = (qz_fp16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                                  ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
                if (!qz_half_ok(dh)) {
                    return false;
                }
                break;
            }
            case QZ_TYPE_MXFP4:
            case QZ_TYPE_NVFP4:
                break;  // exponent bytes cannot encode a non-finite scale
            default:
                return false;
        }
    }
    return true;
}
