// Encoders for the formats that scale a whole block by a single delta: the
// legacy 1-8 bit integer blocks, the ternary blocks, the micro-scaling float
// blocks and the non-linear 4-bit blocks.
//
// Where a format fixes its own rounding rule - "map the largest magnitude onto
// the end of the range, then round to nearest" - the encoder follows that rule
// literally, because any other choice would be a different (and worse)
// encoder, not a different implementation of the same one. Where the format
// leaves the scale open, the encoder fits it: see qz_common.h.

#include "qz_common.h"
#include "qz_codebook.h"
#include "qz_impl.h"

// --------------------------------------------------------------------------
// Q1_0 - sign only, magnitude shared by the block
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_q1_0) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_QK1_0;
    qz_blk_q1_0 * y = (qz_blk_q1_0 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK1_0;

            // the magnitude that minimises the squared error against +-d is
            // the mean absolute value
            float sum = 0.0f;
            for (int i = 0; i < QZ_QK1_0; ++i) {
                sum += fabsf(x[i]);
            }
            y->d = qz_f2h(sum / (float) QZ_QK1_0);

            memset(y->qs, 0, sizeof(y->qs));
            for (int i = 0; i < QZ_QK1_0; ++i) {
                if (x[i] >= 0.0f) {
                    y->qs[i >> 3] |= (uint8_t) (1u << (i & 7));
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q1_0);
}

// --------------------------------------------------------------------------
// Q2_0 - two bits per weight over 64, levels -d, 0, +d, +2d
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_q2_0) {
    const int64_t nblk = n_per_row / QZ_QK2_0;
    qz_blk_q2_0 * y = (qz_blk_q2_0 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK2_0;
            uint8_t q[QZ_QK2_0];
            float d;

            if (imatrix) {
                float  w[QZ_QK2_0];
                int8_t qi[QZ_QK2_0];
                qz_weights(x, imatrix + b * QZ_QK2_0, QZ_QK2_0, w);
                d = qz_fit_sym(x, w, QZ_QK2_0, -1, 2, qi);
                for (int i = 0; i < QZ_QK2_0; ++i) {
                    q[i] = (uint8_t) (qi[i] + 1);
                }
            } else {
                // the format's own rule: the block delta is its largest
                // magnitude, so every weight lands on -1, 0 or +1 of it
                float amax = 0.0f;
                for (int i = 0; i < QZ_QK2_0; ++i) {
                    amax = QZ_MAX(amax, fabsf(x[i]));
                }
                d = amax;
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                for (int i = 0; i < QZ_QK2_0; ++i) {
                    q[i] = (uint8_t) qz_clampi(qz_lround(x[i] * id) + 1, 0, 3);
                }
            }

            y->d = qz_f2h(d);
            memset(y->qs, 0, sizeof(y->qs));
            for (int i = 0; i < QZ_QK2_0; ++i) {
                y->qs[i / 4] |= (uint8_t) (q[i] << ((i % 4) * 2));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q2_0);
}

// --------------------------------------------------------------------------
// Q4_0 / Q5_0 - symmetric integer range, one delta
// --------------------------------------------------------------------------

// Shared body: `bits` is 4 or 5, so the integer range is [-2^(bits-1),
// 2^(bits-1)-1] and the stored value is q + 2^(bits-1).
static void qz_encode_sym_block(const float * x, const float * imp, int bits, uint8_t * qout, float * dout) {
    const int qlo = -(1 << (bits - 1));
    const int qhi = (1 << (bits - 1)) - 1;

    if (imp) {
        float w[QZ_QK4_0];
        int8_t q[QZ_QK4_0];
        qz_weights(x, imp, QZ_QK4_0, w);
        const float s = qz_fit_sym(x, w, QZ_QK4_0, qlo, qhi, q);
        *dout = s;
        for (int i = 0; i < QZ_QK4_0; ++i) {
            qout[i] = (uint8_t) (q[i] - qlo);
        }
        return;
    }

    // no importances: the format's own rule - the extreme element lands on the
    // low end of the range, everything else rounds to nearest
    float amax = 0.0f, xext = 0.0f;
    for (int i = 0; i < QZ_QK4_0; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) {
            amax = a;
            xext = x[i];
        }
    }
    const float d  = xext / (float) qlo;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *dout = d;

    for (int i = 0; i < QZ_QK4_0; ++i) {
        qout[i] = (uint8_t) qz_clampi((int) (x[i] * id - (float) qlo + 0.5f), 0, qhi - qlo);
    }
}

QZ_ENCODER(qz_encode_q4_0) {
    const int64_t nblk = n_per_row / QZ_QK4_0;
    qz_blk_q4_0 * y = (qz_blk_q4_0 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK4_0;
            uint8_t q[QZ_QK4_0];
            float d;
            qz_encode_sym_block(x, imatrix ? imatrix + b * QZ_QK4_0 : NULL, 4, q, &d);

            y->d = qz_f2h(d);
            for (int i = 0; i < QZ_QK4_0 / 2; ++i) {
                y->qs[i] = (uint8_t) (q[i] | (q[i + QZ_QK4_0 / 2] << 4));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q4_0);
}

QZ_ENCODER(qz_encode_q5_0) {
    const int64_t nblk = n_per_row / QZ_QK5_0;
    qz_blk_q5_0 * y = (qz_blk_q5_0 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK5_0;
            uint8_t q[QZ_QK5_0];
            float d;
            qz_encode_sym_block(x, imatrix ? imatrix + b * QZ_QK5_0 : NULL, 5, q, &d);

            y->d = qz_f2h(d);

            uint32_t qh = 0;
            for (int i = 0; i < QZ_QK5_0 / 2; ++i) {
                const uint8_t q0 = q[i];
                const uint8_t q1 = q[i + QZ_QK5_0 / 2];
                y->qs[i] = (uint8_t) ((q0 & 0xf) | ((q1 & 0xf) << 4));
                qh |= (uint32_t) ((q0 >> 4) & 1) << i;
                qh |= (uint32_t) ((q1 >> 4) & 1) << (i + QZ_QK5_0 / 2);
            }
            memcpy(y->qh, &qh, sizeof(y->qh));
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q5_0);
}

// --------------------------------------------------------------------------
// Q4_1 / Q5_1 - integer range with a stored minimum
// --------------------------------------------------------------------------

static void qz_encode_off_block(const float * x, const float * imp, int qmax, uint8_t * qout, float * dout,
                                float * mout) {
    if (imp) {
        float w[QZ_QK4_1];
        float o;
        qz_weights(x, imp, QZ_QK4_1, w);
        const float s = qz_fit_min(x, w, QZ_QK4_1, qmax, qout, &o);
        *dout = s;
        *mout = -o;
        return;
    }

    float lo = x[0], hi = x[0];
    for (int i = 1; i < QZ_QK4_1; ++i) {
        lo = QZ_MIN(lo, x[i]);
        hi = QZ_MAX(hi, x[i]);
    }
    const float d  = (hi - lo) / (float) qmax;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *dout = d;
    *mout = lo;

    for (int i = 0; i < QZ_QK4_1; ++i) {
        qout[i] = (uint8_t) qz_clampi((int) ((x[i] - lo) * id + 0.5f), 0, qmax);
    }
}

QZ_ENCODER(qz_encode_q4_1) {
    const int64_t nblk = n_per_row / QZ_QK4_1;
    qz_blk_q4_1 * y = (qz_blk_q4_1 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK4_1;
            uint8_t q[QZ_QK4_1];
            float d, m;
            qz_encode_off_block(x, imatrix ? imatrix + b * QZ_QK4_1 : NULL, 15, q, &d, &m);

            y->d = qz_f2h(d);
            y->m = qz_f2h(m);
            for (int i = 0; i < QZ_QK4_1 / 2; ++i) {
                y->qs[i] = (uint8_t) (q[i] | (q[i + QZ_QK4_1 / 2] << 4));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q4_1);
}

QZ_ENCODER(qz_encode_q5_1) {
    const int64_t nblk = n_per_row / QZ_QK5_1;
    qz_blk_q5_1 * y = (qz_blk_q5_1 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK5_1;
            uint8_t q[QZ_QK5_1];
            float d, m;
            qz_encode_off_block(x, imatrix ? imatrix + b * QZ_QK5_1 : NULL, 31, q, &d, &m);

            y->d = qz_f2h(d);
            y->m = qz_f2h(m);

            uint32_t qh = 0;
            for (int i = 0; i < QZ_QK5_1 / 2; ++i) {
                const uint8_t q0 = q[i];
                const uint8_t q1 = q[i + QZ_QK5_1 / 2];
                y->qs[i] = (uint8_t) ((q0 & 0xf) | ((q1 & 0xf) << 4));
                qh |= (uint32_t) ((q0 >> 4) & 1) << i;
                qh |= (uint32_t) ((q1 >> 4) & 1) << (i + QZ_QK5_1 / 2);
            }
            memcpy(y->qh, &qh, sizeof(y->qh));
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q5_1);
}

// --------------------------------------------------------------------------
// Q8_0 - 8 bits, symmetric, no search worth doing
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_q8_0) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_QK8_0;
    qz_blk_q8_0 * y = (qz_blk_q8_0 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK8_0;

            float amax = 0.0f;
            for (int i = 0; i < QZ_QK8_0; ++i) {
                amax = QZ_MAX(amax, fabsf(x[i]));
            }
            const float d  = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;

            y->d = qz_f2h(d);
            for (int i = 0; i < QZ_QK8_0; ++i) {
                y->qs[i] = (int8_t) qz_lround(x[i] * id);
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q8_0);
}

// q8_0 plus the block sum d * sum(qs), kept alongside the quants so a dot
// product against an offset format can fold the offset into one multiply.
QZ_ENCODER(qz_encode_q8_1) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_QK8_1;
    qz_blk_q8_1 * y = (qz_blk_q8_1 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK8_1;

            float amax = 0.0f;
            for (int i = 0; i < QZ_QK8_1; ++i) {
                amax = QZ_MAX(amax, fabsf(x[i]));
            }
            const float d  = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;

            y->d = qz_f2h(d);
            int sum = 0;
            for (int i = 0; i < QZ_QK8_1; ++i) {
                y->qs[i] = (int8_t) qz_lround(x[i] * id);
                sum += y->qs[i];
            }
            y->s = qz_f2h(d * (float) sum);
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q8_1);
}

QZ_ENCODER(qz_encode_q8_k) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_K;
    qz_blk_q8_k * y = (qz_blk_q8_k *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            qz_encode_q8_k_block(src + row * n_per_row + b * QZ_K, y);
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q8_k);
}

// Q8_K is an intermediate rather than a storage format: exact 8-bit copy of a
// super-block plus the group sums its consumers need.
void qz_encode_q8_k_block(const float * x, qz_blk_q8_k * blk) {
    float amax = 0.0f;
    for (int i = 0; i < QZ_K; ++i) {
        amax = QZ_MAX(amax, fabsf(x[i]));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    blk->d = d;
    for (int i = 0; i < QZ_K; ++i) {
        blk->qs[i] = (int8_t) qz_clampi(qz_lround(x[i] * id), -128, 127);
    }
    for (int g = 0; g < QZ_K / 16; ++g) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) {
            sum += blk->qs[g * 16 + i];
        }
        blk->bsums[g] = (int16_t) sum;
    }
}

// --------------------------------------------------------------------------
// MXFP4 / NVFP4 - 4-bit floats with a shared exponent
// --------------------------------------------------------------------------

// nearest entry of the (doubled) E2M1 codebook for x/scale
static inline int qz_nearest_e2m1(float x, float scale) {
    int best = 0;
    float best_err = fabsf(x);

    for (int k = 1; k < 16; ++k) {
        const float err = fabsf((float) qz_e2m1_values[k] * scale - x);
        if (err < best_err) {
            best_err = err;
            best = k;
        }
    }
    return best;
}

QZ_ENCODER(qz_encode_mxfp4) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_QK_MXFP4;
    qz_blk_mxfp4 * y = (qz_blk_mxfp4 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK_MXFP4;

            float amax = 0.0f;
            for (int i = 0; i < QZ_QK_MXFP4; ++i) {
                amax = QZ_MAX(amax, fabsf(x[i]));
            }

            // shared exponent: the power of two that puts the largest element
            // inside the codebook's top entry
            y->e = qz_e8m0_from_amax(amax);
            const float d = qz_e8m0_to_fp32_half(y->e);

            for (int i = 0; i < QZ_QK_MXFP4 / 2; ++i) {
                const int k0 = qz_nearest_e2m1(x[i], d);
                const int k1 = qz_nearest_e2m1(x[i + QZ_QK_MXFP4 / 2], d);
                y->qs[i] = (uint8_t) (k0 | (k1 << 4));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_mxfp4);
}

QZ_ENCODER(qz_encode_nvfp4) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_QK_NVFP4;
    const int nsub = QZ_QK_NVFP4 / QZ_QK_NVFP4_SUB;
    qz_blk_nvfp4 * y = (qz_blk_nvfp4 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_QK_NVFP4;

            for (int s = 0; s < nsub; ++s) {
                const float * x = xb + s * QZ_QK_NVFP4_SUB;

                float amax = 0.0f;
                for (int i = 0; i < QZ_QK_NVFP4_SUB; ++i) {
                    amax = QZ_MAX(amax, fabsf(x[i]));
                }

                // 6.0 is the largest magnitude the E2M1 codebook can express
                y->d[s] = qz_fp32_to_ue4m3(amax / 6.0f);
                const float d = qz_ue4m3_to_fp32(y->d[s]);

                for (int i = 0; i < QZ_QK_NVFP4_SUB / 2; ++i) {
                    const int k0 = qz_nearest_e2m1(x[i], d);
                    const int k1 = qz_nearest_e2m1(x[i + QZ_QK_NVFP4_SUB / 2], d);
                    y->qs[s * (QZ_QK_NVFP4_SUB / 2) + i] = (uint8_t) (k0 | (k1 << 4));
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_nvfp4);
}

// --------------------------------------------------------------------------
// TQ1_0 / TQ2_0 - ternary
//
// Both scale the block by its largest magnitude and round every weight to
// -1, 0 or +1. They differ only in how the trits are packed.
// --------------------------------------------------------------------------

// TQ1_0 stores five trits per byte as a base-3 numeral, but scaled into the
// full byte range so a decoder can peel digits off with a multiply and a
// shift: byte = ceil(numeral * 256 / 3^digits).
static inline uint8_t qz_pack_trits(const int * t, int ndigits, int pow3_total) {
    int v = 0;
    for (int i = 0; i < ndigits; ++i) {
        v = v * 3 + t[i];
    }
    // left-align the digits that were written, so a short group occupies the
    // most significant trits
    for (int i = ndigits; i < 5; ++i) {
        v *= 3;
    }
    return (uint8_t) ((v * 256 + pow3_total - 1) / pow3_total);
}

QZ_ENCODER(qz_encode_tq1_0) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_K;
    qz_blk_tq1_0 * y = (qz_blk_tq1_0 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_K;

            float amax = 0.0f;
            for (int i = 0; i < QZ_K; ++i) {
                amax = QZ_MAX(amax, fabsf(x[i]));
            }
            const float id = amax != 0.0f ? 1.0f / amax : 0.0f;
            y->d = qz_f2h(amax);

            // 160 elements as five trits per byte over 32 bytes, then 80 more
            // over 16 bytes, then the last 16 as four trits per byte
            const float * xp = x;
            size_t j = 0;
            for (int span = 32; span >= 16; span >>= 1) {
                for (int m = 0; m < span; ++m) {
                    int t[5];
                    for (int n = 0; n < 5; ++n) {
                        t[n] = qz_clampi(qz_lround(xp[m + n * span] * id), -1, 1) + 1;
                    }
                    y->qs[j + m] = qz_pack_trits(t, 5, 243);
                }
                j += span;
                xp += 5 * span;
            }

            for (size_t i = 0; i < sizeof(y->qh); ++i) {
                int t[4];
                for (int m = 0; m < 4; ++m) {
                    t[m] = qz_clampi(qz_lround(xp[i + m * sizeof(y->qh)] * id), -1, 1) + 1;
                }
                y->qh[i] = qz_pack_trits(t, 4, 243);
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_tq1_0);
}

QZ_ENCODER(qz_encode_tq2_0) {
    (void) imatrix;
    const int64_t nblk = n_per_row / QZ_K;
    qz_blk_tq2_0 * y = (qz_blk_tq2_0 *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_K;

            float amax = 0.0f;
            for (int i = 0; i < QZ_K; ++i) {
                amax = QZ_MAX(amax, fabsf(x[i]));
            }
            const float id = amax != 0.0f ? 1.0f / amax : 0.0f;
            y->d = qz_f2h(amax);

            const float * xp = x;
            for (size_t j = 0; j < sizeof(y->qs); j += 32) {
                for (int m = 0; m < 32; ++m) {
                    uint8_t q = 0;
                    for (int n = 0; n < 4; ++n) {
                        const int t = qz_clampi(qz_lround(xp[m + n * 32] * id), -1, 1) + 1;
                        q |= (uint8_t) (t << (2 * n));
                    }
                    y->qs[j + m] = q;
                }
                xp += 4 * 32;
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_tq2_0);
}

// --------------------------------------------------------------------------
// IQ4_NL / IQ4_XS - nibbles index a fixed non-linear codebook
//
// The codebook is not symmetric, so the scale that best fits a group has to be
// searched rather than derived. Both variants use the same group fit; they
// differ in how the resulting scales are stored.
// --------------------------------------------------------------------------

// Fits x[0..n) to s * qz_iq4_values[k]. Returns s, fills k.
static float qz_fit_iq4(const float * x, const float * w, int n, uint8_t * k) {
    float amax = 0.0f, xext = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) {
            amax = a;
            xext = x[i];
        }
    }
    if (!(amax > 0.0f)) {
        for (int i = 0; i < n; ++i) {
            k[i] = 8; // codebook entry closest to zero
        }
        return 0.0f;
    }

    // two starts: the extreme element pinned to either end of the codebook
    const float starts[2] = { xext / (float) qz_iq4_values[0], xext / (float) qz_iq4_values[15] };

    uint8_t cur[32];
    float best_s   = 0.0f;
    float best_sse = -1.0f;

    for (int t = 0; t < 2; ++t) {
        float s = starts[t];
        if (!(fabsf(s) > 0.0f) || !isfinite(s)) {
            continue;
        }

        for (int it = 0; it < QZ_FIT_ROUNDS; ++it) {
            const float is = 1.0f / s;

            float sse = 0.0f;
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < n; ++i) {
                const int idx = qz_nearest_sorted(qz_iq4_values, 16, x[i] * is);
                const float v = (float) qz_iq4_values[idx];
                const float e = x[i] - s * v;
                cur[i] = (uint8_t) idx;
                sse += w[i] * e * e;
                num += w[i] * x[i] * v;
                den += w[i] * v * v;
            }

            if (best_sse < 0.0f || sse < best_sse) {
                best_sse = sse;
                best_s   = s;
                memcpy(k, cur, (size_t) n);
            }

            if (!(den > 0.0f)) {
                break;
            }
            const float ns = num / den;
            if (!isfinite(ns) || ns == s) {
                break;
            }
            s = ns;
        }
    }

    return best_s;
}

QZ_ENCODER(qz_encode_iq4_nl) {
    const int64_t nblk = n_per_row / QZ_QK4_NL;
    qz_blk_iq4_nl * y = (qz_blk_iq4_nl *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * x = src + row * n_per_row + b * QZ_QK4_NL;
            float   w[QZ_QK4_NL];
            uint8_t k[QZ_QK4_NL];

            qz_weights(x, imatrix ? imatrix + b * QZ_QK4_NL : NULL, QZ_QK4_NL, w);
            const float d = qz_fit_iq4(x, w, QZ_QK4_NL, k);

            y->d = qz_f2h(d);
            for (int i = 0; i < QZ_QK4_NL / 2; ++i) {
                y->qs[i] = (uint8_t) (k[i] | (k[i + QZ_QK4_NL / 2] << 4));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq4_nl);
}

QZ_ENCODER(qz_encode_iq4_xs) {
    const int64_t nblk = n_per_row / QZ_K;
    const int ngroup = QZ_K / 32;
    qz_blk_iq4_xs * y = (qz_blk_iq4_xs *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float   w[QZ_K];
            uint8_t k[QZ_K];
            float   gs[QZ_K / 32];  // per-group scale
            float   gw[QZ_K / 32];  // how much that group's scale matters

            for (int g = 0; g < ngroup; ++g) {
                const float * x = xb + g * 32;
                qz_weights(x, ib ? ib + g * 32 : NULL, 32, w + g * 32);
                gs[g] = qz_fit_iq4(x, w + g * 32, 32, k + g * 32);

                float sens = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const float v = (float) qz_iq4_values[k[g * 32 + i]];
                    sens += w[g * 32 + i] * v * v;
                }
                gw[g] = sens;
            }

            // the group scales themselves are stored as 6-bit integers offset
            // by 32, sharing one f16
            float smax = 0.0f;
            for (int g = 0; g < ngroup; ++g) {
                smax = QZ_MAX(smax, fabsf(gs[g]));
            }

            int ls[QZ_K / 32];
            float d = smax / 32.0f;

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int g = 0; g < ngroup; ++g) {
                    ls[g] = qz_clampi(qz_lround(gs[g] / d), -32, 31);
                    num += gw[g] * gs[g] * (float) ls[g];
                    den += gw[g] * (float) ls[g] * (float) ls[g];
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d != 0.0f)) {
                d = 0.0f;
                memset(ls, 0, sizeof(ls));
            } else {
                for (int g = 0; g < ngroup; ++g) {
                    ls[g] = qz_clampi(qz_lround(gs[g] / d), -32, 31);
                }
            }

            y->d = qz_f2h(d);
            const float dq = qz_h2f(y->d);

            // re-pick the codebook entries against the scale that will
            // actually be used when decoding
            for (int g = 0; g < ngroup; ++g) {
                const float dl = dq * (float) ls[g];
                const float * x = xb + g * 32;
                if (dl != 0.0f) {
                    const float idl = 1.0f / dl;
                    for (int i = 0; i < 32; ++i) {
                        k[g * 32 + i] = (uint8_t) qz_nearest_sorted(qz_iq4_values, 16, x[i] * idl);
                    }
                } else {
                    for (int i = 0; i < 32; ++i) {
                        k[g * 32 + i] = 8;
                    }
                }
            }

            y->scales_h = 0;
            memset(y->scales_l, 0, sizeof(y->scales_l));
            for (int g = 0; g < ngroup; ++g) {
                const int stored = ls[g] + 32;  // 6 bits, offset encoded
                y->scales_l[g / 2] |= (uint8_t) ((stored & 0xf) << (4 * (g % 2)));
                y->scales_h |= (uint16_t) (((stored >> 4) & 3) << (2 * g));
            }

            for (int g = 0; g < ngroup; ++g) {
                for (int i = 0; i < 16; ++i) {
                    y->qs[g * 16 + i] = (uint8_t) (k[g * 32 + i] | (k[g * 32 + i + 16] << 4));
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq4_xs);
}
