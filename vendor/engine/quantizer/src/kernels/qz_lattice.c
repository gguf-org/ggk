// Encoders for the lattice ("IQ") formats.
//
// These formats do not store a number per weight. They store, for each run of
// eight (or four) weights, an index into a fixed codebook of patterns, plus a
// scale and - for the unsigned codebooks - a mask of signs. Encoding is
// therefore a nearest-neighbour search in an 8-dimensional space, run once per
// run of weights, which has to be fast enough to sweep a whole model.
//
// The search works like this. Every codebook entry is built from a tiny
// alphabet of magnitudes (three values for the 2-bit grids, eight for the
// 3-bit ones, {-1,0,1} for the 1-bit one), so a pattern of eight weights maps
// onto a short key: two or three bits per position. There are only a few
// thousand such keys, so at start-up we walk the codebook once and record, for
// every key, which entry sits closest to it. At encode time the search is then
//
//   1. divide the weights by the trial scale,
//   2. round each one to the nearest alphabet symbol - that gives a key,
//   3. look up that key, and also look up the eight keys obtained by moving
//      one position to its second-choice symbol,
//   4. score those handful of candidates with the real weighted error and keep
//      the best.
//
// Step 3 matters: the rounded key is usually not in the codebook at all, and
// the entry nearest to it in plain symbol space is not always the one that
// minimises the weighted error against the actual values.
//
// Scales are fitted the same way as everywhere else in this library: start
// from the value that maps the largest magnitude onto the top of the codebook,
// alternate between "pick entries" and "re-solve the scale", keep the best.

#include "qz_codebook.h"
#include "qz_common.h"
#include "qz_impl.h"

// --------------------------------------------------------------------------
// codebook lookup structure
// --------------------------------------------------------------------------

#define QZ_GRID_MAX_POS  8
#define QZ_GRID_MAX_SYM  8
#define QZ_GRID_NO_ENTRY 0xffff

// How many codebook entries to remember per key. One is not enough: the entry
// closest in plain symbol space is frequently not the one that minimises the
// weighted error against the real values, and the runner-up costs one dot
// product to check.
#define QZ_GRID_NCAND 4

typedef struct {
    int8_t     sym[QZ_GRID_MAX_SYM]; // alphabet, ascending
    int        nsym;
    int        npos;                 // values per codebook entry
    int        bits;                 // bits per position in a key
    int        nentry;
    int8_t   * val;                  // nentry * npos expanded values
    uint16_t * map;                  // key -> QZ_GRID_NCAND closest entries
    int        nkey;
    int        ready;
} qz_grid;

static qz_grid g_iq2_xxs, g_iq2_xs, g_iq2_s, g_iq3_xxs, g_iq3_s, g_iq1;

// Expands a codebook into per-position values and builds the key -> entry map.
// `stride` is 8 for the uint64 grids and 4 for the uint32 ones.
static void qz_grid_build(qz_grid * g, const void * grid, int nentry, int npos, int stride,
                          const int8_t * sym, int nsym, int bits) {
    if (g->ready) {
        return;
    }

    memcpy(g->sym, sym, (size_t) nsym);
    g->nsym   = nsym;
    g->npos   = npos;
    g->bits   = bits;
    g->nentry = nentry;
    g->nkey   = 1 << (bits * npos);

    g->val = (int8_t *) malloc((size_t) nentry * npos);
    g->map = (uint16_t *) malloc((size_t) g->nkey * QZ_GRID_NCAND * sizeof(uint16_t));
    if (!g->val || !g->map) {
        free(g->val);
        free(g->map);
        g->val = NULL;
        g->map = NULL;
        return;
    }

    const uint8_t * raw = (const uint8_t *) grid;
    for (int e = 0; e < nentry; ++e) {
        for (int p = 0; p < npos; ++p) {
            g->val[e * npos + p] = (int8_t) raw[e * stride + p];
        }
    }

    for (int k = 0; k < g->nkey * QZ_GRID_NCAND; ++k) {
        g->map[k] = QZ_GRID_NO_ENTRY;
    }

    // walk every reachable key and record the entries closest to it; keys that
    // hold a symbol index the alphabet does not have stay unset and are never
    // produced by the encoder
    int8_t pattern[QZ_GRID_MAX_POS];
    int    digit[QZ_GRID_MAX_POS];
    memset(digit, 0, sizeof(digit));

    for (;;) {
        int key = 0;
        for (int p = 0; p < npos; ++p) {
            pattern[p] = sym[digit[p]];
            key |= digit[p] << (bits * p);
        }

        int best[QZ_GRID_NCAND];
        int best_d[QZ_GRID_NCAND];
        int nbest = 0;

        for (int e = 0; e < nentry; ++e) {
            int d = 0;
            for (int p = 0; p < npos; ++p) {
                const int diff = pattern[p] - g->val[e * npos + p];
                d += diff * diff;
            }
            if (nbest == QZ_GRID_NCAND && d >= best_d[nbest - 1]) {
                continue;
            }
            // insertion sort into the running shortlist
            int at = nbest < QZ_GRID_NCAND ? nbest : QZ_GRID_NCAND - 1;
            while (at > 0 && best_d[at - 1] > d) {
                best_d[at] = best_d[at - 1];
                best[at]   = best[at - 1];
                --at;
            }
            best_d[at] = d;
            best[at]   = e;
            if (nbest < QZ_GRID_NCAND) {
                ++nbest;
            }
        }

        for (int c = 0; c < nbest; ++c) {
            g->map[(size_t) key * QZ_GRID_NCAND + c] = (uint16_t) best[c];
        }

        int p = 0;
        while (p < npos && ++digit[p] == nsym) {
            digit[p] = 0;
            ++p;
        }
        if (p == npos) {
            break;
        }
    }

    g->ready = 1;
}

static void qz_grid_release(qz_grid * g) {
    free(g->val);
    free(g->map);
    g->val = NULL;
    g->map = NULL;
    g->ready = 0;
}

void qz_lattice_init(qz_type type) {
    static const int8_t sym_iq2[3] = { 8, 25, 43 };
    static const int8_t sym_iq3_xxs[8] = { 4, 12, 20, 28, 36, 44, 52, 62 };
    static const int8_t sym_iq3_s[8] = { 1, 3, 5, 7, 9, 11, 13, 15 };
    static const int8_t sym_iq1[3] = { -1, 0, 1 };

    switch (type) {
        case QZ_TYPE_IQ2_XXS:
            qz_grid_build(&g_iq2_xxs, qz_grid_iq2_xxs, QZ_GRID_IQ2_XXS_SIZE, 8, 8, sym_iq2, 3, 2);
            break;
        case QZ_TYPE_IQ2_XS:
            qz_grid_build(&g_iq2_xs, qz_grid_iq2_xs, QZ_GRID_IQ2_XS_SIZE, 8, 8, sym_iq2, 3, 2);
            break;
        case QZ_TYPE_IQ2_S:
            qz_grid_build(&g_iq2_s, qz_grid_iq2_s, QZ_GRID_IQ2_S_SIZE, 8, 8, sym_iq2, 3, 2);
            break;
        case QZ_TYPE_IQ3_XXS:
            qz_grid_build(&g_iq3_xxs, qz_grid_iq3_xxs, QZ_GRID_IQ3_XXS_SIZE, 4, 4, sym_iq3_xxs, 8, 3);
            break;
        case QZ_TYPE_IQ3_S:
            qz_grid_build(&g_iq3_s, qz_grid_iq3_s, QZ_GRID_IQ3_S_SIZE, 4, 4, sym_iq3_s, 8, 3);
            break;
        case QZ_TYPE_IQ1_S:
        case QZ_TYPE_IQ1_M:
            qz_grid_build(&g_iq1, qz_grid_iq1, QZ_GRID_IQ1_SIZE, 8, 8, sym_iq1, 3, 2);
            break;
        default:
            break;
    }
}

void qz_lattice_free(void) {
    qz_grid_release(&g_iq2_xxs);
    qz_grid_release(&g_iq2_xs);
    qz_grid_release(&g_iq2_s);
    qz_grid_release(&g_iq3_xxs);
    qz_grid_release(&g_iq3_s);
    qz_grid_release(&g_iq1);
}

// --------------------------------------------------------------------------
// nearest-entry search
// --------------------------------------------------------------------------

// weighted error of entry `e` against t[] (targets already divided by the scale)
static inline float qz_entry_cost(const qz_grid * g, int e, const float * t, const float * w) {
    const int8_t * v = g->val + (size_t) e * g->npos;
    float c = 0.0f;
    for (int p = 0; p < g->npos; ++p) {
        const float d = t[p] - (float) v[p];
        c += w[p] * d * d;
    }
    return c;
}

// Finds the codebook entry closest to t[] under the weights w[]. Returns the
// entry index and, in *cost, its weighted error.
static int qz_grid_search(const qz_grid * g, const float * t, const float * w, float * cost) {
    int   first[QZ_GRID_MAX_POS];
    int   second[QZ_GRID_MAX_POS];

    for (int p = 0; p < g->npos; ++p) {
        int b0 = 0, b1 = 0;
        float d0 = -1.0f, d1 = -1.0f;

        for (int s = 0; s < g->nsym; ++s) {
            const float d = fabsf(t[p] - (float) g->sym[s]);
            if (d0 < 0.0f || d < d0) {
                d1 = d0; b1 = b0;
                d0 = d;  b0 = s;
            } else if (d1 < 0.0f || d < d1) {
                d1 = d;  b1 = s;
            }
        }
        first[p]  = b0;
        second[p] = b1;
    }

    int base_key = 0;
    for (int p = 0; p < g->npos; ++p) {
        base_key |= first[p] << (g->bits * p);
    }

    int   best = -1;
    float best_c = 0.0f;

    // the rounded pattern itself, then each single-position alternative; the
    // rounded pattern contributes its whole shortlist, the alternatives only
    // their nearest entry
    for (int p = -1; p < g->npos; ++p) {
        int key = base_key;
        if (p >= 0) {
            key &= ~(((1 << g->bits) - 1) << (g->bits * p));
            key |= second[p] << (g->bits * p);
        }

        const uint16_t * cand = g->map + (size_t) key * QZ_GRID_NCAND;
        const int ncand = p < 0 ? QZ_GRID_NCAND : 1;

        for (int c = 0; c < ncand; ++c) {
            const int e = cand[c];
            if (e == QZ_GRID_NO_ENTRY) {
                break;
            }
            const float cost = qz_entry_cost(g, e, t, w);
            if (best < 0 || cost < best_c) {
                best_c = cost;
                best = e;
            }
        }
    }

    if (best < 0) {
        best = 0;
        best_c = qz_entry_cost(g, 0, t, w);
    }
    *cost = best_c;
    return best;
}

// --------------------------------------------------------------------------
// group fit
//
// One group is `nsub` runs of `npos` weights sharing a scale. The fit returns
// the scale and the entry chosen for each run.
// --------------------------------------------------------------------------

typedef struct {
    float scale;
    int   entry[8];  // at most eight runs per group (32 weights / 4)
    float sse;
} qz_group_fit;

static void qz_fit_group(const qz_grid * g, const float * a, const float * w, int nsub, qz_group_fit * out) {
    const int npos = g->npos;
    const int n    = nsub * npos;

    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        amax = QZ_MAX(amax, fabsf(a[i]));
    }

    out->sse = -1.0f;
    out->scale = 0.0f;
    memset(out->entry, 0, sizeof(out->entry));

    if (!(amax > 0.0f)) {
        return;
    }

    const float top = (float) g->sym[g->nsym - 1];

    for (int t = 0; t < QZ_FIT_STARTS; ++t) {
    float s = amax / top * QZ_FIT_SHRINK[t];

    for (int it = 0; it < 4; ++it) {
        if (!(fabsf(s) > 0.0f) || !isfinite(s)) {
            break;
        }
        const float is = 1.0f / s;

        float t[QZ_GRID_MAX_POS];
        float sse = 0.0f;
        float num = 0.0f, den = 0.0f;
        int   ent[8] = { 0 };

        for (int sub = 0; sub < nsub; ++sub) {
            const float * ai = a + sub * npos;
            const float * wi = w + sub * npos;
            for (int p = 0; p < npos; ++p) {
                t[p] = ai[p] * is;
            }

            float c;
            const int e = qz_grid_search(g, t, wi, &c);
            ent[sub] = e;

            const int8_t * v = g->val + (size_t) e * npos;
            for (int p = 0; p < npos; ++p) {
                const float gv = (float) v[p];
                const float d  = ai[p] - s * gv;
                sse += wi[p] * d * d;
                num += wi[p] * ai[p] * gv;
                den += wi[p] * gv * gv;
            }
        }

        if (out->sse < 0.0f || sse < out->sse) {
            out->sse   = sse;
            out->scale = s;
            memcpy(out->entry, ent, sizeof(ent));
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
}

// Re-picks the codebook entries for a group once the scale it will really be
// decoded with is known.
static void qz_repick_group(const qz_grid * g, const float * a, const float * w, int nsub, float s, int * entry) {
    if (!(fabsf(s) > 0.0f)) {
        for (int sub = 0; sub < nsub; ++sub) {
            entry[sub] = 0;
        }
        return;
    }
    const float is = 1.0f / s;
    float t[QZ_GRID_MAX_POS];

    for (int sub = 0; sub < nsub; ++sub) {
        for (int p = 0; p < g->npos; ++p) {
            t[p] = a[sub * g->npos + p] * is;
        }
        float c;
        entry[sub] = qz_grid_search(g, t, w + sub * g->npos, &c);
    }
}

// --------------------------------------------------------------------------
// sign handling
//
// The unsigned codebooks pair every entry with a mask of signs. Two of the
// formats spend only seven bits on eight signs and derive the eighth from the
// rule that the mask has an even number of set bits; when the natural signs
// break that rule, the cheapest single weight is flipped. Flipping weight i
// costs 4*w*|x|*s*value, so the cheapest one is simply the smallest product.
// --------------------------------------------------------------------------

static uint8_t qz_signs_for(const float * x, int n) {
    uint8_t m = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f) {
            m |= (uint8_t) (1u << i);
        }
    }
    return m;
}

static uint8_t qz_signs_even(const float * x, const float * w, const int8_t * v, int n) {
    uint8_t m = qz_signs_for(x, n);

    int ones = 0;
    for (int i = 0; i < n; ++i) {
        ones += (m >> i) & 1;
    }
    if ((ones & 1) == 0) {
        return m;
    }

    int   cheapest = 0;
    float cost = -1.0f;
    for (int i = 0; i < n; ++i) {
        const float c = w[i] * fabsf(x[i]) * (float) v[i];
        if (cost < 0.0f || c < cost) {
            cost = c;
            cheapest = i;
        }
    }
    return (uint8_t) (m ^ (1u << cheapest));
}

// --------------------------------------------------------------------------
// IQ2_XXS - four runs of eight per group of 32, 4-bit group scale, 7-bit signs
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_iq2_xxs) {
    const qz_grid * g = &g_iq2_xxs;
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 32;
    qz_blk_iq2_xxs * y = (qz_blk_iq2_xxs *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float a[QZ_K], w[QZ_K];
            for (int i = 0; i < QZ_K; ++i) {
                a[i] = fabsf(xb[i]);
            }
            qz_group_fit fit[QZ_K / 32];
            float u[QZ_K / 32], gw[QZ_K / 32];

            for (int gi = 0; gi < ng; ++gi) {
                qz_weights(xb + gi * 32, ib ? ib + gi * 32 : NULL, 32, w + gi * 32);
                qz_fit_group(g, a + gi * 32, w + gi * 32, 4, &fit[gi]);

                // the stored scale is d * (0.5 + sc) * 0.25
                u[gi] = fit[gi].scale * 4.0f;

                float sens = 0.0f;
                for (int sub = 0; sub < 4; ++sub) {
                    const int8_t * v = g->val + (size_t) fit[gi].entry[sub] * 8;
                    for (int p = 0; p < 8; ++p) {
                        const float gv = (float) v[p];
                        sens += w[gi * 32 + sub * 8 + p] * gv * gv;
                    }
                }
                gw[gi] = sens;
            }

            // one f16 for the whole super-block, 4-bit offsets per group
            float umax = 0.0f;
            for (int gi = 0; gi < ng; ++gi) {
                umax = QZ_MAX(umax, u[gi]);
            }
            float d = umax / 15.5f;
            int sc[QZ_K / 32];

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                    const float f = 0.5f + (float) sc[gi];
                    num += gw[gi] * u[gi] * f;
                    den += gw[gi] * f * f;
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || !(nd > 0.0f) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d > 0.0f)) {
                d = 0.0f;
                memset(sc, 0, sizeof(sc));
            } else {
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                }
            }

            y->d = qz_f2h(d);
            const float dq = qz_h2f(y->d);

            for (int gi = 0; gi < ng; ++gi) {
                const float s = dq * (0.5f + (float) sc[gi]) * 0.25f;
                qz_repick_group(g, a + gi * 32, w + gi * 32, 4, s, fit[gi].entry);

                uint32_t word[2] = { 0, 0 };
                for (int sub = 0; sub < 4; ++sub) {
                    const int e = fit[gi].entry[sub];
                    const int8_t * v = g->val + (size_t) e * 8;
                    const uint8_t signs = qz_signs_even(xb + gi * 32 + sub * 8, w + gi * 32 + sub * 8, v, 8);

                    word[0] |= (uint32_t) e << (8 * sub);
                    word[1] |= (uint32_t) (signs & 0x7f) << (7 * sub);
                }
                word[1] |= (uint32_t) sc[gi] << 28;
                memcpy(y->qs + 4 * gi, word, sizeof(word));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq2_xxs);
}

// --------------------------------------------------------------------------
// IQ2_XS - like IQ2_XXS but a 512-entry codebook, its own scale nibbles and a
// 9-bit index sharing a word with the signs
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_iq2_xs) {
    const qz_grid * g = &g_iq2_xs;
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 16;  // one 4-bit scale per 16 weights
    qz_blk_iq2_xs * y = (qz_blk_iq2_xs *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float a[QZ_K], w[QZ_K];
            for (int i = 0; i < QZ_K; ++i) {
                a[i] = fabsf(xb[i]);
            }
            qz_group_fit fit[QZ_K / 16];
            float u[QZ_K / 16], gw[QZ_K / 16];

            for (int gi = 0; gi < ng; ++gi) {
                qz_weights(xb + gi * 16, ib ? ib + gi * 16 : NULL, 16, w + gi * 16);
                qz_fit_group(g, a + gi * 16, w + gi * 16, 2, &fit[gi]);
                u[gi] = fit[gi].scale * 4.0f;

                float sens = 0.0f;
                for (int sub = 0; sub < 2; ++sub) {
                    const int8_t * v = g->val + (size_t) fit[gi].entry[sub] * 8;
                    for (int p = 0; p < 8; ++p) {
                        const float gv = (float) v[p];
                        sens += w[gi * 16 + sub * 8 + p] * gv * gv;
                    }
                }
                gw[gi] = sens;
            }

            float umax = 0.0f;
            for (int gi = 0; gi < ng; ++gi) {
                umax = QZ_MAX(umax, u[gi]);
            }
            float d = umax / 15.5f;
            int sc[QZ_K / 16];

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                    const float f = 0.5f + (float) sc[gi];
                    num += gw[gi] * u[gi] * f;
                    den += gw[gi] * f * f;
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || !(nd > 0.0f) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d > 0.0f)) {
                d = 0.0f;
                memset(sc, 0, sizeof(sc));
            } else {
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                }
            }

            y->d = qz_f2h(d);
            const float dq = qz_h2f(y->d);

            for (int gi = 0; gi < ng; ++gi) {
                const float s = dq * (0.5f + (float) sc[gi]) * 0.25f;
                qz_repick_group(g, a + gi * 16, w + gi * 16, 2, s, fit[gi].entry);

                for (int sub = 0; sub < 2; ++sub) {
                    const int e = fit[gi].entry[sub];
                    const int8_t * v = g->val + (size_t) e * 8;
                    const uint8_t signs = qz_signs_even(xb + gi * 16 + sub * 8, w + gi * 16 + sub * 8, v, 8);
                    y->qs[gi * 2 + sub] = (uint16_t) ((e & 511) | ((signs & 0x7f) << 9));
                }
            }

            for (int gi = 0; gi < ng; gi += 2) {
                y->scales[gi / 2] = (uint8_t) (sc[gi] | (sc[gi + 1] << 4));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq2_xs);
}

// --------------------------------------------------------------------------
// IQ2_S - 1024-entry codebook, full eight-bit sign bytes
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_iq2_s) {
    const qz_grid * g = &g_iq2_s;
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 16;
    qz_blk_iq2_s * y = (qz_blk_iq2_s *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float a[QZ_K], w[QZ_K];
            for (int i = 0; i < QZ_K; ++i) {
                a[i] = fabsf(xb[i]);
            }
            qz_group_fit fit[QZ_K / 16];
            float u[QZ_K / 16], gw[QZ_K / 16];

            for (int gi = 0; gi < ng; ++gi) {
                qz_weights(xb + gi * 16, ib ? ib + gi * 16 : NULL, 16, w + gi * 16);
                qz_fit_group(g, a + gi * 16, w + gi * 16, 2, &fit[gi]);
                u[gi] = fit[gi].scale * 4.0f;

                float sens = 0.0f;
                for (int sub = 0; sub < 2; ++sub) {
                    const int8_t * v = g->val + (size_t) fit[gi].entry[sub] * 8;
                    for (int p = 0; p < 8; ++p) {
                        const float gv = (float) v[p];
                        sens += w[gi * 16 + sub * 8 + p] * gv * gv;
                    }
                }
                gw[gi] = sens;
            }

            float umax = 0.0f;
            for (int gi = 0; gi < ng; ++gi) {
                umax = QZ_MAX(umax, u[gi]);
            }
            float d = umax / 15.5f;
            int sc[QZ_K / 16];

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                    const float f = 0.5f + (float) sc[gi];
                    num += gw[gi] * u[gi] * f;
                    den += gw[gi] * f * f;
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || !(nd > 0.0f) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d > 0.0f)) {
                d = 0.0f;
                memset(sc, 0, sizeof(sc));
            } else {
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                }
            }

            y->d = qz_f2h(d);
            const float dq = qz_h2f(y->d);

            memset(y->qh, 0, sizeof(y->qh));

            for (int gi = 0; gi < ng; ++gi) {
                const float s = dq * (0.5f + (float) sc[gi]) * 0.25f;
                qz_repick_group(g, a + gi * 16, w + gi * 16, 2, s, fit[gi].entry);

                for (int sub = 0; sub < 2; ++sub) {
                    const int e = fit[gi].entry[sub];
                    const int8_t * v = g->val + (size_t) e * 8;
                    const int slot = gi * 2 + sub;          // run of eight inside the block
                    const int grp32 = slot / 4;             // which group of 32
                    const int in32  = slot % 4;

                    y->qs[slot] = (uint8_t) (e & 0xff);
                    y->qh[grp32] |= (uint8_t) (((e >> 8) & 3) << (2 * in32));
                    y->qs[QZ_K / 8 + slot] = qz_signs_for(xb + slot * 8, 8);
                }
            }

            for (int gi = 0; gi < ng; gi += 2) {
                y->scales[gi / 2] = (uint8_t) (sc[gi] | (sc[gi + 1] << 4));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq2_s);
}

// --------------------------------------------------------------------------
// IQ3_XXS - two four-value runs per eight weights, 4-bit scale per 32
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_iq3_xxs) {
    const qz_grid * g = &g_iq3_xxs;
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 32;
    qz_blk_iq3_xxs * y = (qz_blk_iq3_xxs *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float a[QZ_K], w[QZ_K];
            for (int i = 0; i < QZ_K; ++i) {
                a[i] = fabsf(xb[i]);
            }
            qz_group_fit fit[QZ_K / 32];
            float u[QZ_K / 32], gw[QZ_K / 32];

            for (int gi = 0; gi < ng; ++gi) {
                qz_weights(xb + gi * 32, ib ? ib + gi * 32 : NULL, 32, w + gi * 32);
                qz_fit_group(g, a + gi * 32, w + gi * 32, 8, &fit[gi]);

                // the stored scale is d * (0.5 + sc) * 0.5
                u[gi] = fit[gi].scale * 2.0f;

                float sens = 0.0f;
                for (int sub = 0; sub < 8; ++sub) {
                    const int8_t * v = g->val + (size_t) fit[gi].entry[sub] * 4;
                    for (int p = 0; p < 4; ++p) {
                        const float gv = (float) v[p];
                        sens += w[gi * 32 + sub * 4 + p] * gv * gv;
                    }
                }
                gw[gi] = sens;
            }

            float umax = 0.0f;
            for (int gi = 0; gi < ng; ++gi) {
                umax = QZ_MAX(umax, u[gi]);
            }
            float d = umax / 15.5f;
            int sc[QZ_K / 32];

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                    const float f = 0.5f + (float) sc[gi];
                    num += gw[gi] * u[gi] * f;
                    den += gw[gi] * f * f;
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || !(nd > 0.0f) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d > 0.0f)) {
                d = 0.0f;
                memset(sc, 0, sizeof(sc));
            } else {
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(u[gi] / d - 0.5f), 0, 15);
                }
            }

            y->d = qz_f2h(d);
            const float dq = qz_h2f(y->d);
            uint8_t * tail = y->qs + QZ_K / 4;

            for (int gi = 0; gi < ng; ++gi) {
                const float s = dq * (0.5f + (float) sc[gi]) * 0.5f;
                qz_repick_group(g, a + gi * 32, w + gi * 32, 8, s, fit[gi].entry);

                uint32_t word = 0;
                for (int sub = 0; sub < 8; ++sub) {
                    y->qs[gi * 8 + sub] = (uint8_t) fit[gi].entry[sub];
                }
                // signs cover eight weights, i.e. two codebook runs at a time
                for (int pair = 0; pair < 4; ++pair) {
                    int8_t v[8];
                    const int8_t * v0 = g->val + (size_t) fit[gi].entry[2 * pair + 0] * 4;
                    const int8_t * v1 = g->val + (size_t) fit[gi].entry[2 * pair + 1] * 4;
                    memcpy(v, v0, 4);
                    memcpy(v + 4, v1, 4);

                    const float * xp = xb + gi * 32 + pair * 8;
                    const float * wp = w + gi * 32 + pair * 8;
                    const uint8_t signs = qz_signs_even(xp, wp, v, 8);
                    word |= (uint32_t) (signs & 0x7f) << (7 * pair);
                }
                word |= (uint32_t) sc[gi] << 28;
                memcpy(tail + 4 * gi, &word, sizeof(word));
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq3_xxs);
}

// --------------------------------------------------------------------------
// IQ3_S - 512-entry codebook, full sign bytes, 4-bit scale per 32
// --------------------------------------------------------------------------

QZ_ENCODER(qz_encode_iq3_s) {
    const qz_grid * g = &g_iq3_s;
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 32;
    qz_blk_iq3_s * y = (qz_blk_iq3_s *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float a[QZ_K], w[QZ_K];
            for (int i = 0; i < QZ_K; ++i) {
                a[i] = fabsf(xb[i]);
            }
            qz_group_fit fit[QZ_K / 32];
            float u[QZ_K / 32], gw[QZ_K / 32];

            for (int gi = 0; gi < ng; ++gi) {
                qz_weights(xb + gi * 32, ib ? ib + gi * 32 : NULL, 32, w + gi * 32);
                qz_fit_group(g, a + gi * 32, w + gi * 32, 8, &fit[gi]);

                // the stored scale is d * (1 + 2*sc)
                u[gi] = fit[gi].scale;

                float sens = 0.0f;
                for (int sub = 0; sub < 8; ++sub) {
                    const int8_t * v = g->val + (size_t) fit[gi].entry[sub] * 4;
                    for (int p = 0; p < 4; ++p) {
                        const float gv = (float) v[p];
                        sens += w[gi * 32 + sub * 4 + p] * gv * gv;
                    }
                }
                gw[gi] = sens;
            }

            float umax = 0.0f;
            for (int gi = 0; gi < ng; ++gi) {
                umax = QZ_MAX(umax, u[gi]);
            }
            float d = umax / 31.0f;  // largest representable multiplier is 1 + 2*15
            int sc[QZ_K / 32];

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(0.5f * (u[gi] / d - 1.0f)), 0, 15);
                    const float f = 1.0f + 2.0f * (float) sc[gi];
                    num += gw[gi] * u[gi] * f;
                    den += gw[gi] * f * f;
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || !(nd > 0.0f) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d > 0.0f)) {
                d = 0.0f;
                memset(sc, 0, sizeof(sc));
            } else {
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(0.5f * (u[gi] / d - 1.0f)), 0, 15);
                }
            }

            y->d = qz_f2h(d);
            const float dq = qz_h2f(y->d);

            memset(y->qh, 0, sizeof(y->qh));
            memset(y->scales, 0, sizeof(y->scales));

            for (int gi = 0; gi < ng; ++gi) {
                const float s = dq * (1.0f + 2.0f * (float) sc[gi]);
                qz_repick_group(g, a + gi * 32, w + gi * 32, 8, s, fit[gi].entry);

                y->scales[gi / 2] |= (uint8_t) (sc[gi] << (4 * (gi % 2)));

                for (int sub = 0; sub < 8; ++sub) {
                    const int e = fit[gi].entry[sub];
                    y->qs[gi * 8 + sub] = (uint8_t) (e & 0xff);
                    y->qh[gi] |= (uint8_t) (((e >> 8) & 1) << sub);
                }
                for (int pair = 0; pair < 4; ++pair) {
                    y->signs[gi * 4 + pair] = qz_signs_for(xb + gi * 32 + pair * 8, 8);
                }
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq3_s);
}

// --------------------------------------------------------------------------
// IQ1_S / IQ1_M - ternary codebook with a shared offset
//
// Here the codebook already carries signs, and every decoded value is shifted
// by a small constant whose sign is stored per run. Both offsets are tried and
// the better one kept.
// --------------------------------------------------------------------------

// Fits a run of eight weights against the ternary codebook for a given scale
// and offset. Returns the entry and its weighted error.
static int qz_fit_iq1_run(const float * x, const float * w, float s, float delta, float * cost) {
    if (!(fabsf(s) > 0.0f)) {
        *cost = 0.0f;
        for (int i = 0; i < 8; ++i) {
            *cost += w[i] * x[i] * x[i];
        }
        return 0;
    }

    float t[8];
    const float is = 1.0f / s;
    for (int i = 0; i < 8; ++i) {
        t[i] = x[i] * is - delta;
    }

    float c;
    const int e = qz_grid_search(&g_iq1, t, w, &c);
    // c is measured in units of s^2 against the shifted target, which is
    // exactly the reconstruction error divided by s^2
    *cost = c * s * s;
    return e;
}

// Fits one group of 32 weights: picks the offset sign, the scale and the four
// entries. Returns the scale; `entry` and `*neg` describe the rest.
static float qz_fit_iq1_group(const float * x, const float * w, int * entry, int * neg) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = QZ_MAX(amax, fabsf(x[i]));
    }
    *neg = 0;
    memset(entry, 0, 4 * sizeof(int));
    if (!(amax > 0.0f)) {
        return 0.0f;
    }

    float best_s = 0.0f;
    float best_c = -1.0f;

    for (int sign = 0; sign < 2; ++sign) {
        const float delta = sign ? -QZ_IQ1_DELTA : QZ_IQ1_DELTA;
        float s = amax / (1.0f + QZ_IQ1_DELTA);
        int   ent[4];

        for (int it = 0; it < 4; ++it) {
            if (!(s > 0.0f) || !isfinite(s)) {
                break;
            }
            float total = 0.0f;
            float num = 0.0f, den = 0.0f;

            for (int sub = 0; sub < 4; ++sub) {
                float c;
                const int e = qz_fit_iq1_run(x + sub * 8, w + sub * 8, s, delta, &c);
                ent[sub] = e;
                total += c;

                const int8_t * v = g_iq1.val + (size_t) e * 8;
                for (int p = 0; p < 8; ++p) {
                    const float gv = (float) v[p] + delta;
                    num += w[sub * 8 + p] * x[sub * 8 + p] * gv;
                    den += w[sub * 8 + p] * gv * gv;
                }
            }

            if (best_c < 0.0f || total < best_c) {
                best_c = total;
                best_s = s;
                *neg = sign;
                memcpy(entry, ent, sizeof(ent));
            }

            if (!(den > 0.0f)) {
                break;
            }
            const float ns = num / den;
            if (!isfinite(ns) || !(ns > 0.0f) || ns == s) {
                break;
            }
            s = ns;
        }
    }

    return best_s;
}

QZ_ENCODER(qz_encode_iq1_s) {
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 32;
    qz_blk_iq1_s * y = (qz_blk_iq1_s *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float w[QZ_K];
            float gs[QZ_K / 32], gw[QZ_K / 32];
            int   entry[QZ_K / 32][4];
            int   neg[QZ_K / 32];

            for (int gi = 0; gi < ng; ++gi) {
                qz_weights(xb + gi * 32, ib ? ib + gi * 32 : NULL, 32, w + gi * 32);
                gs[gi] = qz_fit_iq1_group(xb + gi * 32, w + gi * 32, entry[gi], &neg[gi]);

                const float delta = neg[gi] ? -QZ_IQ1_DELTA : QZ_IQ1_DELTA;
                float sens = 0.0f;
                for (int sub = 0; sub < 4; ++sub) {
                    const int8_t * v = g_iq1.val + (size_t) entry[gi][sub] * 8;
                    for (int p = 0; p < 8; ++p) {
                        const float gv = (float) v[p] + delta;
                        sens += w[gi * 32 + sub * 8 + p] * gv * gv;
                    }
                }
                gw[gi] = sens;
            }

            // group scales are odd multiples: d * (2*sc + 1), sc in [0,7]
            float smax = 0.0f;
            for (int gi = 0; gi < ng; ++gi) {
                smax = QZ_MAX(smax, gs[gi]);
            }
            float d = smax / 15.0f;
            int sc[QZ_K / 32];

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(0.5f * (gs[gi] / d - 1.0f)), 0, 7);
                    const float f = 1.0f + 2.0f * (float) sc[gi];
                    num += gw[gi] * gs[gi] * f;
                    den += gw[gi] * f * f;
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || !(nd > 0.0f) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d > 0.0f)) {
                d = 0.0f;
                memset(sc, 0, sizeof(sc));
            } else {
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(0.5f * (gs[gi] / d - 1.0f)), 0, 7);
                }
            }

            y->d = qz_f2h(d);
            const float dq = qz_h2f(y->d);

            for (int gi = 0; gi < ng; ++gi) {
                const float s = dq * (1.0f + 2.0f * (float) sc[gi]);
                const float delta = neg[gi] ? -QZ_IQ1_DELTA : QZ_IQ1_DELTA;

                uint16_t qh = (uint16_t) ((sc[gi] & 7) << 12);
                if (neg[gi]) {
                    qh |= 0x8000;
                }

                for (int sub = 0; sub < 4; ++sub) {
                    float c;
                    const int e = qz_fit_iq1_run(xb + gi * 32 + sub * 8, w + gi * 32 + sub * 8, s, delta, &c);
                    y->qs[gi * 4 + sub] = (uint8_t) (e & 0xff);
                    qh |= (uint16_t) (((e >> 8) & 7) << (3 * sub));
                }
                y->qh[gi] = qh;
            }
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq1_s);
}

QZ_ENCODER(qz_encode_iq1_m) {
    const int64_t nblk = n_per_row / QZ_K;
    const int ng = QZ_K / 16;  // one 3-bit scale per 16 weights
    qz_blk_iq1_m * y = (qz_blk_iq1_m *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t b = 0; b < nblk; ++b, ++y) {
            const float * xb = src + row * n_per_row + b * QZ_K;
            const float * ib = imatrix ? imatrix + b * QZ_K : NULL;

            float w[QZ_K];
            float gs[QZ_K / 16], gw[QZ_K / 16];
            int   entry[QZ_K / 16][2];
            int   neg[QZ_K / 16][2];

            for (int gi = 0; gi < ng; ++gi) {
                const float * x = xb + gi * 16;
                float * wg = w + gi * 16;
                qz_weights(x, ib ? ib + gi * 16 : NULL, 16, wg);

                float amax = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    amax = QZ_MAX(amax, fabsf(x[i]));
                }

                // each run of eight carries its own offset sign, but the two
                // runs of a group share the scale
                float best_c = -1.0f;
                float best_s = 0.0f;
                int   best_e[2] = { 0, 0 };
                int   best_n[2] = { 0, 0 };

                if (amax > 0.0f) {
                    float s = amax / (1.0f + QZ_IQ1_DELTA);

                    for (int it = 0; it < 4; ++it) {
                        if (!(s > 0.0f) || !isfinite(s)) {
                            break;
                        }
                        float total = 0.0f, num = 0.0f, den = 0.0f;
                        int ent[2], sgn[2];

                        for (int sub = 0; sub < 2; ++sub) {
                            float c_pos, c_neg;
                            const int e_pos = qz_fit_iq1_run(x + sub * 8, wg + sub * 8, s, QZ_IQ1_DELTA, &c_pos);
                            const int e_neg = qz_fit_iq1_run(x + sub * 8, wg + sub * 8, s, -QZ_IQ1_DELTA, &c_neg);

                            const int use_neg = c_neg < c_pos;
                            ent[sub] = use_neg ? e_neg : e_pos;
                            sgn[sub] = use_neg;
                            total += use_neg ? c_neg : c_pos;

                            const float delta = use_neg ? -QZ_IQ1_DELTA : QZ_IQ1_DELTA;
                            const int8_t * v = g_iq1.val + (size_t) ent[sub] * 8;
                            for (int p = 0; p < 8; ++p) {
                                const float gv = (float) v[p] + delta;
                                num += wg[sub * 8 + p] * x[sub * 8 + p] * gv;
                                den += wg[sub * 8 + p] * gv * gv;
                            }
                        }

                        if (best_c < 0.0f || total < best_c) {
                            best_c = total;
                            best_s = s;
                            memcpy(best_e, ent, sizeof(ent));
                            memcpy(best_n, sgn, sizeof(sgn));
                        }

                        if (!(den > 0.0f)) {
                            break;
                        }
                        const float ns = num / den;
                        if (!isfinite(ns) || !(ns > 0.0f) || ns == s) {
                            break;
                        }
                        s = ns;
                    }
                }

                gs[gi] = best_s;
                entry[gi][0] = best_e[0];
                entry[gi][1] = best_e[1];
                neg[gi][0] = best_n[0];
                neg[gi][1] = best_n[1];

                float sens = 0.0f;
                for (int sub = 0; sub < 2; ++sub) {
                    const float delta = neg[gi][sub] ? -QZ_IQ1_DELTA : QZ_IQ1_DELTA;
                    const int8_t * v = g_iq1.val + (size_t) entry[gi][sub] * 8;
                    for (int p = 0; p < 8; ++p) {
                        const float gv = (float) v[p] + delta;
                        sens += wg[sub * 8 + p] * gv * gv;
                    }
                }
                gw[gi] = sens;
            }

            float smax = 0.0f;
            for (int gi = 0; gi < ng; ++gi) {
                smax = QZ_MAX(smax, gs[gi]);
            }
            float d = smax / 15.0f;
            int sc[QZ_K / 16];

            for (int it = 0; it < 3 && d > 0.0f; ++it) {
                float num = 0.0f, den = 0.0f;
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(0.5f * (gs[gi] / d - 1.0f)), 0, 7);
                    const float f = 1.0f + 2.0f * (float) sc[gi];
                    num += gw[gi] * gs[gi] * f;
                    den += gw[gi] * f * f;
                }
                if (!(den > 0.0f)) {
                    break;
                }
                const float nd = num / den;
                if (!isfinite(nd) || !(nd > 0.0f) || nd == d) {
                    break;
                }
                d = nd;
            }
            if (!(d > 0.0f)) {
                d = 0.0f;
                memset(sc, 0, sizeof(sc));
            } else {
                for (int gi = 0; gi < ng; ++gi) {
                    sc[gi] = qz_clampi(qz_lround(0.5f * (gs[gi] / d - 1.0f)), 0, 7);
                }
            }

            // the block delta is stored as the top nibble of each scale word
            const qz_fp16_t dh = qz_f2h(d);
            uint16_t sw[4];
            memset(sw, 0, sizeof(sw));
            sw[0] = (uint16_t) ((dh & 0x000f) << 12);
            sw[1] = (uint16_t) ((dh & 0x00f0) << 8);
            sw[2] = (uint16_t) ((dh & 0x0f00) << 4);
            sw[3] = (uint16_t) (dh & 0xf000);

            const float dq = qz_h2f(dh);

            memset(y->qs, 0, sizeof(y->qs));
            memset(y->qh, 0, sizeof(y->qh));

            for (int gi = 0; gi < ng; ++gi) {
                // four 3-bit scales per 16-bit word: two groups of 32, each
                // holding one scale for its first 16 weights and one for the
                // second, with the top nibble reserved for the block delta
                sw[gi / 4] |= (uint16_t) ((sc[gi] & 7) << (6 * ((gi / 2) % 2) + 3 * (gi % 2)));

                const float s = dq * (1.0f + 2.0f * (float) sc[gi]);
                const float * x = xb + gi * 16;
                const float * wg = w + gi * 16;

                for (int sub = 0; sub < 2; ++sub) {
                    float c_pos, c_neg;
                    const int e_pos = qz_fit_iq1_run(x + sub * 8, wg + sub * 8, s, QZ_IQ1_DELTA, &c_pos);
                    const int e_neg = qz_fit_iq1_run(x + sub * 8, wg + sub * 8, s, -QZ_IQ1_DELTA, &c_neg);
                    const int use_neg = c_neg < c_pos;
                    const int e = use_neg ? e_neg : e_pos;

                    const int slot = gi * 2 + sub;   // run of eight inside the block
                    y->qs[slot] = (uint8_t) (e & 0xff);
                    y->qh[slot / 2] |= (uint8_t) (((e >> 8) & 7) << (4 * (slot % 2)));
                    if (use_neg) {
                        y->qh[slot / 2] |= (uint8_t) (0x08u << (4 * (slot % 2)));
                    }
                }
            }

            memcpy(y->scales, sw, sizeof(sw));
        }
    }
    return (size_t) nrows * nblk * sizeof(qz_blk_iq1_m);
}
