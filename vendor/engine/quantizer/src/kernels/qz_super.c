// Encoders for the super-block ("K") formats.
//
// All five share one shape: a super-block of 256 weights is cut into groups
// (sixteen of 16, or eight of 32), each group gets its own small integer scale
// - and, for the asymmetric formats, its own integer minimum - and those
// per-group values are themselves quantized against one f16 multiplier for the
// whole super-block.
//
// The encoders therefore run in three stages:
//   1. fit each group on its own (qz_common.h does the least-squares work),
//   2. quantize the resulting per-group scales/mins, weighting each group by
//      how much its error actually costs,
//   3. re-round the weights against the scales that survived stage 2, so the
//      payload matches what a decoder will really see rather than the ideal
//      scale that was never storable.
//
// Stage 3 is what keeps the error close to the group-wise optimum: without it
// the rounding error of the scale quantization lands on every weight.

#include "qz_common.h"
#include "qz_impl.h"

// --------------------------------------------------------------------------
// Q2_K - sixteen groups of 16, 4-bit scale and 4-bit min per group
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_q2_k) {
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 16;
    qz_blk_q2_k * y = (qz_blk_q2_k *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float   w[QZ_K];
            uint8_t q[QZ_K];
            float   gs[QZ_K / 16], gm[QZ_K / 16];   // group scale and min
            float   ws[QZ_K / 16], wm[QZ_K / 16];   // how much each of them matters

            for (int g = 0; g < ng; ++g) {
                const float * x = xb + g * 16;
                float * wg = w + g * 16;
                qz_weights(x, ib ? ib + g * 16 : NULL, 16, wg);

                float o;
                gs[g] = qz_fit_min(x, wg, 16, 3, q + g * 16, &o);
                gm[g] = o;

                float sens = 0.0f, tot = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const float qf = (float) q[g * 16 + i];
                    sens += wg[i] * qf * qf;
                    tot  += wg[i];
                }
                ws[g] = sens;
                wm[g] = tot;
            }

            uint8_t ls[QZ_K / 16], lm[QZ_K / 16];
            float d    = qz_fit_group_scale(gs, ws, ng, 15, ls);
            float dmin = qz_fit_group_scale(gm, wm, ng, 15, lm);

            // four bits of scale and four of min per sixteen weights is coarse
            // enough that the rounded pair is often beatable by a neighbour
            {
                const float dq0 = qz_h2f(qz_f2h(d));
                const float mq0 = qz_h2f(qz_f2h(dmin));
                for (int g = 0; g < ng; ++g) {
                    int s = ls[g], m = lm[g];
                    qz_polish_group(xb + g * 16, w + g * 16, 16, 3, dq0, mq0, 15, &s, &m);
                    ls[g] = (uint8_t) s;
                    lm[g] = (uint8_t) m;
                }
            }

            // round the weights against the stored scales, then re-solve the
            // two multipliers for the rounding that resulted
            uint8_t qq[QZ_K];
            float   sq[QZ_K], sm[QZ_K];

            for (int pass = 0; pass < 2; ++pass) {
                const qz_fp16_t dh = qz_f2h(d);
                const qz_fp16_t mh = qz_f2h(dmin);
                const float dq = qz_h2f(dh);
                const float mq = qz_h2f(mh);

                for (int g = 0; g < ng; ++g) {
                    const float dl = dq * (float) ls[g];
                    const float ml = mq * (float) lm[g];
                    const float * x = xb + g * 16;

                    for (int i = 0; i < 16; ++i) {
                        const int k = dl != 0.0f ? qz_clampi(qz_lround((x[i] + ml) / dl), 0, 3) : 0;
                        qq[g * 16 + i] = (uint8_t) k;
                        sq[g * 16 + i] = (float) ls[g] * (float) k;
                        sm[g * 16 + i] = (float) lm[g];
                    }
                }

                if (pass == 0) {
                    qz_refit_dm(xb, w, sq, sm, QZ_K, &d, &dmin);
                } else {
                    y->d    = dh;
                    y->dmin = mh;
                }
            }

            memset(y->qs, 0, sizeof(y->qs));
            for (int g = 0; g < ng; ++g) {
                y->scales[g] = (uint8_t) (ls[g] | (lm[g] << 4));

                const int half  = g / 8;
                const int r     = g % 8;
                const int part  = r % 2;
                const int shift = (r / 2) * 2;
                uint8_t * qs = y->qs + half * 32 + part * 16;

                for (int i = 0; i < 16; ++i) {
                    qs[i] |= (uint8_t) (qq[g * 16 + i] << shift);
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q2_k);
}

// --------------------------------------------------------------------------
// Q3_K - sixteen groups of 16, symmetric, 6-bit signed scales
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_q3_k) {
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 16;
    qz_blk_q3_k * y = (qz_blk_q3_k *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float  w[QZ_K];
            int8_t q[QZ_K];
            float  gs[QZ_K / 16], ws[QZ_K / 16];

            for (int g = 0; g < ng; ++g) {
                const float * x = xb + g * 16;
                float * wg = w + g * 16;
                qz_weights(x, ib ? ib + g * 16 : NULL, 16, wg);

                gs[g] = qz_fit_sym(x, wg, 16, -4, 3, q + g * 16);

                float sens = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const float qf = (float) q[g * 16 + i];
                    sens += wg[i] * qf * qf;
                }
                ws[g] = sens;
            }

            // scales are stored as 6 bits offset by 32, i.e. -32..31
            float smax = 0.0f;
            for (int g = 0; g < ng; ++g) {
                smax = QZ_MAX(smax, fabsf(gs[g]));
            }

            int ls[QZ_K / 16];
            float d = smax / 32.0f;
            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int g = 0; g < ng; ++g) {
                    ls[g] = qz_clampi(qz_lround(gs[g] / d), -32, 31);
                    num += ws[g] * gs[g] * (float) ls[g];
                    den += ws[g] * (float) ls[g] * (float) ls[g];
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
            if (d != 0.0f) {
                for (int g = 0; g < ng; ++g) {
                    ls[g] = qz_clampi(qz_lround(gs[g] / d), -32, 31);
                }
            } else {
                memset(ls, 0, sizeof(ls));
            }

            // round against the stored scales, then re-solve the shared f16
            int8_t qq[QZ_K];
            float  sq[QZ_K];

            for (int pass = 0; pass < 2; ++pass) {
                const qz_fp16_t dh = qz_f2h(d);
                const float dq_p = qz_h2f(dh);

                for (int g = 0; g < ng; ++g) {
                    const float dl = dq_p * (float) ls[g];
                    const float * x = xb + g * 16;
                    for (int i = 0; i < 16; ++i) {
                        const int k = dl != 0.0f ? qz_clampi(qz_lround(x[i] / dl), -4, 3) : 0;
                        qq[g * 16 + i] = (int8_t) k;
                        sq[g * 16 + i] = (float) ls[g] * (float) k;
                    }
                }

                if (pass == 0) {
                    qz_refit_d(xb, w, sq, QZ_K, &d);
                } else {
                    y->d = dh;
                }
            }

            memset(y->qs, 0, sizeof(y->qs));
            memset(y->hmask, 0, sizeof(y->hmask));
            memset(y->scales, 0, sizeof(y->scales));

            for (int g = 0; g < ng; ++g) {
                const int stored = ls[g] + 32;  // 6 bits
                if (g < 8) {
                    y->scales[g] |= (uint8_t) (stored & 0xf);
                } else {
                    y->scales[g - 8] |= (uint8_t) ((stored & 0xf) << 4);
                }
                y->scales[8 + (g % 4)] |= (uint8_t) (((stored >> 4) & 3) << (2 * (g / 4)));

                const int half  = g / 8;
                const int r     = g % 8;
                const int part  = r % 2;
                const int shift = (r / 2) * 2;
                const int bit   = half * 4 + r / 2;

                for (int i = 0; i < 16; ++i) {
                    const int stored3 = qq[g * 16 + i] + 4;
                    const int idx = part * 16 + i;

                    y->qs[half * 32 + idx] |= (uint8_t) ((stored3 & 3) << shift);
                    if (stored3 & 4) {
                        y->hmask[idx] |= (uint8_t) (1u << bit);
                    }
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q3_k);
}

// --------------------------------------------------------------------------
// Q4_K / Q5_K - eight groups of 32, 6-bit scale and 6-bit min per group
// --------------------------------------------------------------------------

// Shared body: `qmax` is 15 for q4_K and 31 for q5_K. Writes the per-element
// integers to `qout` and the packed scale bytes to `scales`.
static void qz_encode_k32(const float * xb, const float * ib, int qmax, uint8_t * qout, uint8_t * scale_bytes,
                          qz_fp16_t * d_out, qz_fp16_t * dmin_out) {
    const int ng = QZ_K / 32;

    float   w[QZ_K];
    uint8_t q[QZ_K];
    float   gs[QZ_K / 32], gm[QZ_K / 32];
    float   ws[QZ_K / 32], wm[QZ_K / 32];

    for (int g = 0; g < ng; ++g) {
        const float * x = xb + g * 32;
        float * wg = w + g * 32;
        qz_weights(x, ib ? ib + g * 32 : NULL, 32, wg);

        float o;
        gs[g] = qz_fit_min(x, wg, 32, qmax, q + g * 32, &o);
        gm[g] = o;

        float sens = 0.0f, tot = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float qf = (float) q[g * 32 + i];
            sens += wg[i] * qf * qf;
            tot  += wg[i];
        }
        ws[g] = sens;
        wm[g] = tot;
    }

    uint8_t ls[QZ_K / 32], lm[QZ_K / 32];
    float d    = qz_fit_group_scale(gs, ws, ng, 63, ls);
    float dmin = qz_fit_group_scale(gm, wm, ng, 63, lm);

    // round against the stored scales, then re-solve the shared multipliers
    // for the rounding that resulted
    float sq[QZ_K], sm[QZ_K];

    for (int pass = 0; pass < 2; ++pass) {
        const qz_fp16_t dh = qz_f2h(d);
        const qz_fp16_t mh = qz_f2h(dmin);
        const float dq = qz_h2f(dh);
        const float mq = qz_h2f(mh);

        for (int g = 0; g < ng; ++g) {
            const float dl = dq * (float) ls[g];
            const float ml = mq * (float) lm[g];
            const float * x = xb + g * 32;

            for (int i = 0; i < 32; ++i) {
                const int k = dl != 0.0f ? qz_clampi(qz_lround((x[i] + ml) / dl), 0, qmax) : 0;
                qout[g * 32 + i] = (uint8_t) k;
                sq[g * 32 + i] = (float) ls[g] * (float) k;
                sm[g * 32 + i] = (float) lm[g];
            }
        }

        if (pass == 0) {
            qz_refit_dm(xb, w, sq, sm, QZ_K, &d, &dmin);
        } else {
            *d_out    = dh;
            *dmin_out = mh;
        }
    }

    qz_pack_scale_min_6bit(ls, lm, scale_bytes);
}

QZ_ENCODER(qz_encode_q4_k) {
    const int64_t nblk = n_per_row / QZ_K;
    qz_blk_q4_k * y = (qz_blk_q4_k *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            uint8_t q[QZ_K];
            qz_encode_k32(src + row * n_per_row + b * QZ_K, imatrix ? imatrix + b * QZ_K : NULL,
                          15, q, y->scales, &y->d, &y->dmin);

            for (int g = 0; g < QZ_K / 32; ++g) {
                uint8_t * qs = y->qs + (g / 2) * 32;
                const int shift = (g % 2) * 4;
                for (int i = 0; i < 32; ++i) {
                    if (shift == 0) {
                        qs[i] = (uint8_t) (q[g * 32 + i] & 0xf);
                    } else {
                        qs[i] |= (uint8_t) ((q[g * 32 + i] & 0xf) << 4);
                    }
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q4_k);
}

QZ_ENCODER(qz_encode_q5_k) {
    const int64_t nblk = n_per_row / QZ_K;
    qz_blk_q5_k * y = (qz_blk_q5_k *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            uint8_t q[QZ_K];
            qz_encode_k32(src + row * n_per_row + b * QZ_K, imatrix ? imatrix + b * QZ_K : NULL,
                          31, q, y->scales, &y->d, &y->dmin);

            memset(y->qh, 0, sizeof(y->qh));
            for (int g = 0; g < QZ_K / 32; ++g) {
                uint8_t * qs = y->qs + (g / 2) * 32;
                const int shift = (g % 2) * 4;
                for (int i = 0; i < 32; ++i) {
                    const uint8_t v = q[g * 32 + i];
                    if (shift == 0) {
                        qs[i] = (uint8_t) (v & 0xf);
                    } else {
                        qs[i] |= (uint8_t) ((v & 0xf) << 4);
                    }
                    if (v & 0x10) {
                        y->qh[i] |= (uint8_t) (1u << g);
                    }
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q5_k);
}

// --------------------------------------------------------------------------
// Q6_K - sixteen groups of 16, symmetric, int8 scales
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_q6_k) {
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 16;
    qz_blk_q6_k * y = (qz_blk_q6_k *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float  w[QZ_K];
            int8_t q[QZ_K];
            float  gs[QZ_K / 16], ws[QZ_K / 16];

            for (int g = 0; g < ng; ++g) {
                const float * x = xb + g * 16;
                float * wg = w + g * 16;
                qz_weights(x, ib ? ib + g * 16 : NULL, 16, wg);

                gs[g] = qz_fit_sym(x, wg, 16, -32, 31, q + g * 16);

                float sens = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const float qf = (float) q[g * 16 + i];
                    sens += wg[i] * qf * qf;
                }
                ws[g] = sens;
            }

            float d = qz_fit_group_scale_sym(gs, ws, ng, 127, y->scales);

            // round against the stored scales, then re-solve the shared f16
            int8_t qq[QZ_K];
            float  sq[QZ_K];

            for (int pass = 0; pass < 2; ++pass) {
                const qz_fp16_t dh = qz_f2h(d);
                const float dq_p = qz_h2f(dh);

                for (int g = 0; g < ng; ++g) {
                    const float dl = dq_p * (float) y->scales[g];
                    const float * x = xb + g * 16;
                    for (int i = 0; i < 16; ++i) {
                        const int k = dl != 0.0f ? qz_clampi(qz_lround(x[i] / dl), -32, 31) : 0;
                        qq[g * 16 + i] = (int8_t) k;
                        sq[g * 16 + i] = (float) y->scales[g] * (float) k;
                    }
                }

                if (pass == 0) {
                    qz_refit_d(xb, w, sq, QZ_K, &d);
                } else {
                    y->d = dh;
                }
            }

            memset(y->ql, 0, sizeof(y->ql));
            memset(y->qh, 0, sizeof(y->qh));

            for (int g = 0; g < ng; ++g) {
                const int half = g / 8;          // 128 elements per half
                const int r    = g % 8;          // group within the half
                const int quad = r / 2;          // which 32-element quarter
                const int lane = (r % 2) * 16;   // offset inside the quarter

                uint8_t * ql = y->ql + half * 64;
                uint8_t * qh = y->qh + half * 32;

                for (int i = 0; i < 16; ++i) {
                    const int stored = qq[g * 16 + i] + 32;   // 6 bits
                    const int idx = lane + i;

                    // quarters 0 and 2 share the low nibbles of ql[0..31] and
                    // ql[32..63]; quarters 1 and 3 the high nibbles
                    ql[(quad % 2) * 32 + idx] |= (uint8_t) ((stored & 0xf) << ((quad / 2) * 4));
                    qh[idx] |= (uint8_t) (((stored >> 4) & 3) << (2 * quad));
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_q6_k);
}
