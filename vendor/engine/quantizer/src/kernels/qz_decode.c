// Decoders: encoded blocks back to f32.
//
// These are the reference for what every encoder in this directory has to
// produce, and they are what the requantize path uses when it reads an already
// quantized tensor. Each one walks the layout documented in qz_format.h.

#include "qz_codebook.h"
#include "qz_common.h"
#include "qz_impl.h"

// --------------------------------------------------------------------------
// block-scale formats
// --------------------------------------------------------------------------

void qz_decode_q1_0(const void * src, float * dst, int64_t k) {
    const qz_blk_q1_0 * x = (const qz_blk_q1_0 *) src;

    for (int64_t b = 0; b < k / QZ_QK1_0; ++b) {
        const float d = qz_h2f(x[b].d);
        for (int i = 0; i < QZ_QK1_0; ++i) {
            const int bit = (x[b].qs[i >> 3] >> (i & 7)) & 1;
            *dst++ = bit ? d : -d;
        }
    }
}

void qz_decode_q2_0(const void * src, float * dst, int64_t k) {
    const qz_blk_q2_0 * x = (const qz_blk_q2_0 *) src;

    for (int64_t b = 0; b < k / QZ_QK2_0; ++b) {
        const float d = qz_h2f(x[b].d);
        for (int i = 0; i < QZ_QK2_0; ++i) {
            const int q = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
            *dst++ = d * (float) (q - 1);
        }
    }
}

void qz_decode_q4_0(const void * src, float * dst, int64_t k) {
    const qz_blk_q4_0 * x = (const qz_blk_q4_0 *) src;

    for (int64_t b = 0; b < k / QZ_QK4_0; ++b) {
        const float d = qz_h2f(x[b].d);
        float * y = dst + b * QZ_QK4_0;

        for (int i = 0; i < QZ_QK4_0 / 2; ++i) {
            y[i]                  = d * (float) ((x[b].qs[i] & 0xf) - 8);
            y[i + QZ_QK4_0 / 2]   = d * (float) ((x[b].qs[i] >> 4) - 8);
        }
    }
}

void qz_decode_q4_1(const void * src, float * dst, int64_t k) {
    const qz_blk_q4_1 * x = (const qz_blk_q4_1 *) src;

    for (int64_t b = 0; b < k / QZ_QK4_1; ++b) {
        const float d = qz_h2f(x[b].d);
        const float m = qz_h2f(x[b].m);
        float * y = dst + b * QZ_QK4_1;

        for (int i = 0; i < QZ_QK4_1 / 2; ++i) {
            y[i]                = d * (float) (x[b].qs[i] & 0xf) + m;
            y[i + QZ_QK4_1 / 2] = d * (float) (x[b].qs[i] >> 4)  + m;
        }
    }
}

void qz_decode_q5_0(const void * src, float * dst, int64_t k) {
    const qz_blk_q5_0 * x = (const qz_blk_q5_0 *) src;

    for (int64_t b = 0; b < k / QZ_QK5_0; ++b) {
        const float d = qz_h2f(x[b].d);
        uint32_t qh;
        memcpy(&qh, x[b].qh, sizeof(qh));
        float * y = dst + b * QZ_QK5_0;

        for (int i = 0; i < QZ_QK5_0 / 2; ++i) {
            const int q0 = (x[b].qs[i] & 0xf) | (int) (((qh >> i) & 1u) << 4);
            const int q1 = (x[b].qs[i] >> 4)  | (int) (((qh >> (i + QZ_QK5_0 / 2)) & 1u) << 4);
            y[i]                = d * (float) (q0 - 16);
            y[i + QZ_QK5_0 / 2] = d * (float) (q1 - 16);
        }
    }
}

void qz_decode_q5_1(const void * src, float * dst, int64_t k) {
    const qz_blk_q5_1 * x = (const qz_blk_q5_1 *) src;

    for (int64_t b = 0; b < k / QZ_QK5_1; ++b) {
        const float d = qz_h2f(x[b].d);
        const float m = qz_h2f(x[b].m);
        uint32_t qh;
        memcpy(&qh, x[b].qh, sizeof(qh));
        float * y = dst + b * QZ_QK5_1;

        for (int i = 0; i < QZ_QK5_1 / 2; ++i) {
            const int q0 = (x[b].qs[i] & 0xf) | (int) (((qh >> i) & 1u) << 4);
            const int q1 = (x[b].qs[i] >> 4)  | (int) (((qh >> (i + QZ_QK5_1 / 2)) & 1u) << 4);
            y[i]                = d * (float) q0 + m;
            y[i + QZ_QK5_1 / 2] = d * (float) q1 + m;
        }
    }
}

void qz_decode_q8_0(const void * src, float * dst, int64_t k) {
    const qz_blk_q8_0 * x = (const qz_blk_q8_0 *) src;

    for (int64_t b = 0; b < k / QZ_QK8_0; ++b) {
        const float d = qz_h2f(x[b].d);
        for (int i = 0; i < QZ_QK8_0; ++i) {
            dst[b * QZ_QK8_0 + i] = d * (float) x[b].qs[i];
        }
    }
}

void qz_decode_q8_1(const void * src, float * dst, int64_t k) {
    const qz_blk_q8_1 * x = (const qz_blk_q8_1 *) src;

    for (int64_t b = 0; b < k / QZ_QK8_1; ++b) {
        const float d = qz_h2f(x[b].d);
        for (int i = 0; i < QZ_QK8_1; ++i) {
            dst[b * QZ_QK8_1 + i] = d * (float) x[b].qs[i];
        }
    }
}

void qz_decode_mxfp4(const void * src, float * dst, int64_t k) {
    const qz_blk_mxfp4 * x = (const qz_blk_mxfp4 *) src;

    for (int64_t b = 0; b < k / QZ_QK_MXFP4; ++b) {
        const float d = qz_e8m0_to_fp32_half(x[b].e);
        float * y = dst + b * QZ_QK_MXFP4;

        for (int i = 0; i < QZ_QK_MXFP4 / 2; ++i) {
            y[i]                    = d * (float) qz_e2m1_values[x[b].qs[i] & 0xf];
            y[i + QZ_QK_MXFP4 / 2]  = d * (float) qz_e2m1_values[x[b].qs[i] >> 4];
        }
    }
}

void qz_decode_nvfp4(const void * src, float * dst, int64_t k) {
    const qz_blk_nvfp4 * x = (const qz_blk_nvfp4 *) src;
    const int nsub = QZ_QK_NVFP4 / QZ_QK_NVFP4_SUB;

    for (int64_t b = 0; b < k / QZ_QK_NVFP4; ++b) {
        for (int s = 0; s < nsub; ++s) {
            const float d = qz_ue4m3_to_fp32(x[b].d[s]);
            float * y = dst + b * QZ_QK_NVFP4 + s * QZ_QK_NVFP4_SUB;
            const uint8_t * q = x[b].qs + s * (QZ_QK_NVFP4_SUB / 2);

            for (int i = 0; i < QZ_QK_NVFP4_SUB / 2; ++i) {
                y[i]                        = d * (float) qz_e2m1_values[q[i] & 0xf];
                y[i + QZ_QK_NVFP4_SUB / 2]  = d * (float) qz_e2m1_values[q[i] >> 4];
            }
        }
    }
}

// --------------------------------------------------------------------------
// ternary
// --------------------------------------------------------------------------

// qz_unpack_trit, which peels one trit out of a packed byte, is in qz_impl.h:
// a decoder that vectorises this format needs it too.

void qz_decode_tq1_0(const void * src, float * dst, int64_t k) {
    const qz_blk_tq1_0 * x = (const qz_blk_tq1_0 *) src;
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        size_t j = 0;
        for (int span = 32; span >= 16; span >>= 1) {
            for (int n = 0; n < 5; ++n) {
                for (int m = 0; m < span; ++m) {
                    *dst++ = d * (float) qz_unpack_trit(x[b].qs[j + m], pow3[n]);
                }
            }
            j += span;
        }

        for (int n = 0; n < 4; ++n) {
            for (size_t i = 0; i < sizeof(x->qh); ++i) {
                *dst++ = d * (float) qz_unpack_trit(x[b].qh[i], pow3[n]);
            }
        }
    }
}

void qz_decode_tq2_0(const void * src, float * dst, int64_t k) {
    const qz_blk_tq2_0 * x = (const qz_blk_tq2_0 *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (int n = 0; n < 4; ++n) {
                for (int m = 0; m < 32; ++m) {
                    *dst++ = d * (float) (((x[b].qs[j + m] >> (2 * n)) & 3) - 1);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// super-block formats
// --------------------------------------------------------------------------

// q4_K / q5_K keep eight 6-bit scales and eight 6-bit mins in twelve bytes:
// the first four bytes hold scales 0-3, the next four mins 0-3, and the last
// four carry the low nibbles of scales 4-7 and mins 4-7 with their top two
// bits borrowed from the upper bits of the first eight bytes.
void qz_unpack_scale_min_6bit(const uint8_t * src, int group, uint8_t * scale, uint8_t * min) {
    if (group < 4) {
        *scale = src[group] & 63;
        *min   = src[group + 4] & 63;
    } else {
        *scale = (uint8_t) ((src[group + 4] & 0xf) | ((src[group - 4] >> 6) << 4));
        *min   = (uint8_t) ((src[group + 4] >> 4)  | ((src[group]     >> 6) << 4));
    }
}

void qz_pack_scale_min_6bit(const uint8_t * scales, const uint8_t * mins, uint8_t * dst) {
    memset(dst, 0, 12);

    for (int g = 0; g < 4; ++g) {
        dst[g]     = (uint8_t) (scales[g] & 63);
        dst[g + 4] = (uint8_t) (mins[g] & 63);
    }
    for (int g = 4; g < 8; ++g) {
        dst[g + 4]  = (uint8_t) ((scales[g] & 0xf) | ((mins[g] & 0xf) << 4));
        dst[g - 4] |= (uint8_t) ((scales[g] >> 4) << 6);
        dst[g]     |= (uint8_t) ((mins[g] >> 4) << 6);
    }
}

void qz_decode_q2_k(const void * src, float * dst, int64_t k) {
    const qz_blk_q2_k * x = (const qz_blk_q2_k *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d    = qz_h2f(x[b].d);
        const float dmin = qz_h2f(x[b].dmin);

        // the 2-bit payload of a 128-element half lives in one 32-byte span,
        // two bits at a time
        for (int half = 0; half < 2; ++half) {
            const uint8_t * q = x[b].qs + half * 32;

            for (int shift = 0; shift < 8; shift += 2) {
                for (int part = 0; part < 2; ++part) {
                    const int g = half * 8 + (shift / 2) * 2 + part;
                    const uint8_t sc = x[b].scales[g];
                    const float dl = d * (float) (sc & 0xf);
                    const float ml = dmin * (float) (sc >> 4);

                    for (int i = 0; i < 16; ++i) {
                        *dst++ = dl * (float) ((q[part * 16 + i] >> shift) & 3) - ml;
                    }
                }
            }
        }
    }
}

void qz_decode_q3_k(const void * src, float * dst, int64_t k) {
    const qz_blk_q3_k * x = (const qz_blk_q3_k *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        // unpack the sixteen 6-bit scales: low nibbles from the first eight
        // bytes, high two bits from the last four
        int8_t sc[16];
        for (int g = 0; g < 16; ++g) {
            const int low  = g < 8 ? (x[b].scales[g] & 0xf) : (x[b].scales[g - 8] >> 4);
            const int high = (x[b].scales[8 + (g % 4)] >> (2 * (g / 4))) & 3;
            sc[g] = (int8_t) ((low | (high << 4)) - 32);
        }

        int bit = 0;
        for (int half = 0; half < 2; ++half) {
            const uint8_t * q = x[b].qs + half * 32;

            for (int shift = 0; shift < 8; shift += 2, ++bit) {
                const uint8_t mask = (uint8_t) (1u << bit);

                for (int part = 0; part < 2; ++part) {
                    const int g = half * 8 + (shift / 2) * 2 + part;
                    const float dl = d * (float) sc[g];

                    for (int i = 0; i < 16; ++i) {
                        const int idx = part * 16 + i;
                        const int lo  = (q[idx] >> shift) & 3;
                        // the high bit is stored inverted: a set mask bit
                        // means "do not subtract 4"
                        const int v = lo - ((x[b].hmask[idx] & mask) ? 0 : 4);
                        *dst++ = dl * (float) v;
                    }
                }
            }
        }
    }
}

void qz_decode_q4_k(const void * src, float * dst, int64_t k) {
    const qz_blk_q4_k * x = (const qz_blk_q4_k *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d    = qz_h2f(x[b].d);
        const float dmin = qz_h2f(x[b].dmin);

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint8_t sc, mn;
            qz_unpack_scale_min_6bit(x[b].scales, g, &sc, &mn);
            const float dl = d * (float) sc;
            const float ml = dmin * (float) mn;
            const uint8_t * q = x[b].qs + (g / 2) * 32;
            const int shift = (g % 2) * 4;

            for (int i = 0; i < 32; ++i) {
                *dst++ = dl * (float) ((q[i] >> shift) & 0xf) - ml;
            }
        }
    }
}

void qz_decode_q5_k(const void * src, float * dst, int64_t k) {
    const qz_blk_q5_k * x = (const qz_blk_q5_k *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d    = qz_h2f(x[b].d);
        const float dmin = qz_h2f(x[b].dmin);

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint8_t sc, mn;
            qz_unpack_scale_min_6bit(x[b].scales, g, &sc, &mn);
            const float dl = d * (float) sc;
            const float ml = dmin * (float) mn;
            const uint8_t * q = x[b].qs + (g / 2) * 32;
            const int shift = (g % 2) * 4;
            const uint8_t hmask = (uint8_t) (1u << g);

            for (int i = 0; i < 32; ++i) {
                const int v = ((q[i] >> shift) & 0xf) | ((x[b].qh[i] & hmask) ? 16 : 0);
                *dst++ = dl * (float) v - ml;
            }
        }
    }
}

void qz_decode_q6_k(const void * src, float * dst, int64_t k) {
    const qz_blk_q6_k * x = (const qz_blk_q6_k *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        for (int half = 0; half < 2; ++half) {
            const uint8_t * ql = x[b].ql + half * 64;
            const uint8_t * qh = x[b].qh + half * 32;
            const int8_t  * sc = x[b].scales + half * 8;
            float * y = dst + half * 128;

            for (int i = 0; i < 32; ++i) {
                const int is = i / 16;
                const int q1 = (int) ((ql[i]      & 0xf) | (((qh[i] >> 0) & 3) << 4)) - 32;
                const int q2 = (int) ((ql[i + 32] & 0xf) | (((qh[i] >> 2) & 3) << 4)) - 32;
                const int q3 = (int) ((ql[i]      >> 4)  | (((qh[i] >> 4) & 3) << 4)) - 32;
                const int q4 = (int) ((ql[i + 32] >> 4)  | (((qh[i] >> 6) & 3) << 4)) - 32;

                y[i]      = d * (float) sc[is]     * (float) q1;
                y[i + 32] = d * (float) sc[is + 2] * (float) q2;
                y[i + 64] = d * (float) sc[is + 4] * (float) q3;
                y[i + 96] = d * (float) sc[is + 6] * (float) q4;
            }
        }
        dst += QZ_K;
    }
}

void qz_decode_q8_k(const void * src, float * dst, int64_t k) {
    const qz_blk_q8_k * x = (const qz_blk_q8_k *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        for (int i = 0; i < QZ_K; ++i) {
            *dst++ = x[b].d * (float) x[b].qs[i];
        }
    }
}

// --------------------------------------------------------------------------
// lattice formats
//
// A codebook entry expands to eight (or four) magnitudes; a sign mask picks
// the sign of each. The sign bit convention is "set means negative".
// --------------------------------------------------------------------------

static inline void qz_expand_signs(const uint8_t * mag, uint8_t signs, float scale, int n, float * y) {
    for (int i = 0; i < n; ++i) {
        const float v = scale * (float) mag[i];
        y[i] = (signs & (1u << i)) ? -v : v;
    }
}

void qz_decode_iq2_xxs(const void * src, float * dst, int64_t k) {
    const qz_blk_iq2_xxs * x = (const qz_blk_iq2_xxs *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint32_t w[2];
            memcpy(w, x[b].qs + 4 * g, sizeof(w));

            // top nibble of the second word is the group scale
            const float dl = d * (0.5f + (float) (w[1] >> 28)) * 0.25f;
            const uint8_t * idx = (const uint8_t *) w;

            for (int s = 0; s < 4; ++s) {
                const uint8_t * mag = (const uint8_t *) (qz_grid_iq2_xxs + idx[s]);
                const uint8_t signs = qz_sign_mask_from_bits((uint8_t) ((w[1] >> (7 * s)) & 127));
                qz_expand_signs(mag, signs, dl, 8, dst);
                dst += 8;
            }
        }
    }
}

void qz_decode_iq2_xs(const void * src, float * dst, int64_t k) {
    const qz_blk_iq2_xs * x = (const qz_blk_iq2_xs *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        for (int g = 0; g < QZ_K / 32; ++g) {
            const float dl[2] = {
                d * (0.5f + (float) (x[b].scales[g] & 0xf)) * 0.25f,
                d * (0.5f + (float) (x[b].scales[g] >> 4))  * 0.25f,
            };

            for (int s = 0; s < 4; ++s) {
                const uint16_t q = x[b].qs[4 * g + s];
                const uint8_t * mag = (const uint8_t *) (qz_grid_iq2_xs + (q & 511));
                const uint8_t signs = qz_sign_mask_from_bits((uint8_t) (q >> 9));
                qz_expand_signs(mag, signs, dl[s / 2], 8, dst);
                dst += 8;
            }
        }
    }
}

void qz_decode_iq2_s(const void * src, float * dst, int64_t k) {
    const qz_blk_iq2_s * x = (const qz_blk_iq2_s *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);
        const uint8_t * qs    = x[b].qs;
        const uint8_t * signs = x[b].qs + QZ_K / 8;

        for (int g = 0; g < QZ_K / 32; ++g) {
            const float dl[2] = {
                d * (0.5f + (float) (x[b].scales[g] & 0xf)) * 0.25f,
                d * (0.5f + (float) (x[b].scales[g] >> 4))  * 0.25f,
            };

            for (int s = 0; s < 4; ++s) {
                // two extra index bits per sub-group live in qh
                const int idx = qs[s] | (((x[b].qh[g] >> (2 * s)) & 3) << 8);
                const uint8_t * mag = (const uint8_t *) (qz_grid_iq2_s + idx);
                qz_expand_signs(mag, signs[s], dl[s / 2], 8, dst);
                dst += 8;
            }
            qs    += 4;
            signs += 4;
        }
    }
}

void qz_decode_iq3_xxs(const void * src, float * dst, int64_t k) {
    const qz_blk_iq3_xxs * x = (const qz_blk_iq3_xxs *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);
        const uint8_t * qs   = x[b].qs;
        const uint8_t * tail = x[b].qs + QZ_K / 4;

        for (int g = 0; g < QZ_K / 32; ++g) {
            uint32_t w;
            memcpy(&w, tail + 4 * g, sizeof(w));
            const float dl = d * (0.5f + (float) (w >> 28)) * 0.5f;

            for (int s = 0; s < 4; ++s) {
                const uint8_t signs = qz_sign_mask_from_bits((uint8_t) ((w >> (7 * s)) & 127));
                const uint8_t * m0 = (const uint8_t *) (qz_grid_iq3_xxs + qs[2 * s + 0]);
                const uint8_t * m1 = (const uint8_t *) (qz_grid_iq3_xxs + qs[2 * s + 1]);

                for (int i = 0; i < 4; ++i) {
                    const float v0 = dl * (float) m0[i];
                    const float v1 = dl * (float) m1[i];
                    dst[i]     = (signs & (1u << i))       ? -v0 : v0;
                    dst[i + 4] = (signs & (1u << (i + 4))) ? -v1 : v1;
                }
                dst += 8;
            }
            qs += 8;
        }
    }
}

void qz_decode_iq3_s(const void * src, float * dst, int64_t k) {
    const qz_blk_iq3_s * x = (const qz_blk_iq3_s *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);
        const uint8_t * qs    = x[b].qs;
        const uint8_t * signs = x[b].signs;

        for (int g = 0; g < QZ_K / 32; ++g) {
            // one 4-bit scale per two groups
            const int sc = (x[b].scales[g / 2] >> (4 * (g % 2))) & 0xf;
            const float dl = d * (float) (1 + 2 * sc);
            const uint8_t qh = x[b].qh[g];

            for (int s = 0; s < 4; ++s) {
                const int i0 = qs[2 * s + 0] | (((qh >> (2 * s + 0)) & 1) << 8);
                const int i1 = qs[2 * s + 1] | (((qh >> (2 * s + 1)) & 1) << 8);
                const uint8_t * m0 = (const uint8_t *) (qz_grid_iq3_s + i0);
                const uint8_t * m1 = (const uint8_t *) (qz_grid_iq3_s + i1);

                for (int i = 0; i < 4; ++i) {
                    const float v0 = dl * (float) m0[i];
                    const float v1 = dl * (float) m1[i];
                    dst[i]     = (signs[s] & (1u << i))       ? -v0 : v0;
                    dst[i + 4] = (signs[s] & (1u << (i + 4))) ? -v1 : v1;
                }
                dst += 8;
            }
            qs    += 8;
            signs += 4;
        }
    }
}

void qz_decode_iq1_s(const void * src, float * dst, int64_t k) {
    const qz_blk_iq1_s * x = (const qz_blk_iq1_s *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        for (int g = 0; g < QZ_K / 32; ++g) {
            const uint16_t qh = x[b].qh[g];
            const float dl = d * (float) (2 * ((qh >> 12) & 7) + 1);
            const float delta = (qh & 0x8000) ? -QZ_IQ1_DELTA : QZ_IQ1_DELTA;

            for (int s = 0; s < 4; ++s) {
                const int idx = x[b].qs[4 * g + s] | (int) (((qh >> (3 * s)) & 7) << 8);
                const int8_t * v = (const int8_t *) (qz_grid_iq1 + idx);

                for (int i = 0; i < 8; ++i) {
                    *dst++ = dl * ((float) v[i] + delta);
                }
            }
        }
    }
}

void qz_decode_iq1_m(const void * src, float * dst, int64_t k) {
    const qz_blk_iq1_m * x = (const qz_blk_iq1_m *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        uint16_t sc[4];
        memcpy(sc, x[b].scales, sizeof(sc));

        // the block delta is spread over the top nibble of each scale word
        const qz_fp16_t dh = (qz_fp16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                          ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = qz_h2f(dh);

        const uint8_t * qs = x[b].qs;
        const uint8_t * qh = x[b].qh;

        for (int g = 0; g < QZ_K / 32; ++g) {
            // two 3-bit scales per group of 32, one per half
            const float dl0 = d * (float) (2 * ((sc[g / 2] >> (6 * (g % 2) + 0)) & 7) + 1);
            const float dl1 = d * (float) (2 * ((sc[g / 2] >> (6 * (g % 2) + 3)) & 7) + 1);

            for (int s = 0; s < 4; ++s) {
                const int idx = qs[s] | (int) (((qh[s / 2] >> (4 * (s % 2))) & 7) << 8);
                const float delta = (qh[s / 2] & (0x08u << (4 * (s % 2)))) ? -QZ_IQ1_DELTA : QZ_IQ1_DELTA;
                const int8_t * v = (const int8_t *) (qz_grid_iq1 + idx);
                const float dl = s < 2 ? dl0 : dl1;

                for (int i = 0; i < 8; ++i) {
                    *dst++ = dl * ((float) v[i] + delta);
                }
            }
            qs += 4;
            qh += 2;
        }
    }
}

void qz_decode_iq4_nl(const void * src, float * dst, int64_t k) {
    const qz_blk_iq4_nl * x = (const qz_blk_iq4_nl *) src;

    for (int64_t b = 0; b < k / QZ_QK4_NL; ++b) {
        const float d = qz_h2f(x[b].d);
        float * y = dst + b * QZ_QK4_NL;

        for (int i = 0; i < QZ_QK4_NL / 2; ++i) {
            y[i]                   = d * (float) qz_iq4_values[x[b].qs[i] & 0xf];
            y[i + QZ_QK4_NL / 2]   = d * (float) qz_iq4_values[x[b].qs[i] >> 4];
        }
    }
}

void qz_decode_iq4_xs(const void * src, float * dst, int64_t k) {
    const qz_blk_iq4_xs * x = (const qz_blk_iq4_xs *) src;

    for (int64_t b = 0; b < k / QZ_K; ++b) {
        const float d = qz_h2f(x[b].d);

        for (int g = 0; g < QZ_K / 32; ++g) {
            const int ls = ((x[b].scales_l[g / 2] >> (4 * (g % 2))) & 0xf) |
                           (int) (((x[b].scales_h >> (2 * g)) & 3) << 4);
            const float dl = d * (float) (ls - 32);
            const uint8_t * q = x[b].qs + g * 16;

            for (int i = 0; i < 16; ++i) {
                dst[i]      = dl * (float) qz_iq4_values[q[i] & 0xf];
                dst[i + 16] = dl * (float) qz_iq4_values[q[i] >> 4];
            }
            dst += 32;
        }
    }
}
