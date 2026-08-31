#pragma once

// Internal helpers shared by the encoders.
//
// Most GGUF formats boil down to the same fitting problem: pick a scale (and
// sometimes an offset) so that a group of weights, once rounded onto a small
// integer range, reconstructs as closely as possible. The routines here solve
// that problem by alternating between "round with the current scale" and
// "re-solve the scale for the current rounding", started from a couple of
// different points and keeping whichever result has the lowest weighted error.
// That is a plain least-squares alternation - no format-specific tricks live
// here, only in the callers.

#include "qz_fp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QZ_MIN(a, b) ((a) < (b) ? (a) : (b))
#define QZ_MAX(a, b) ((a) > (b) ? (a) : (b))

// a super-block never has more than sixteen groups
#define QZ_MAX_GROUPS 16

#define QZ_FIT_ROUNDS 5  // alternation steps; converges well before this

// starting scales are tried at these fractions of the value that just fits the
// group's largest element
#define QZ_FIT_STARTS 3
static const float QZ_FIT_SHRINK[QZ_FIT_STARTS] = { 1.0f, 0.94f, 0.88f };

static inline int qz_clampi(int v, int lo, int hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

static inline int qz_lround(float v) {
    return (int) lrintf(v);
}

// --------------------------------------------------------------------------
// weights
//
// Every fit is weighted, and the shape of that weight decides what the whole
// library optimizes for.
//
// An importance matrix entry is the mean square activation a column sees, so
// weighting purely by it minimizes the expected squared error of the layer's
// output - the objective that matters on paper. On its own, though, it treats
// a group's largest weight and its smallest alike, and the large ones are the
// ones a coarse format struggles to place. The default therefore multiplies
// the importance by a magnitude term, sqrt(x^2 + floor), where the floor is a
// fraction of the group's mean square so that near-zero elements keep some
// pull. The floor was swept against both objectives on synthetic weight-like
// data; half the mean square sits at the point where raising it further stops
// buying accuracy and only flattens the magnitude term away. Measured against both objectives - plain reconstruction error and
// importance-weighted error - that hedge lands within a fraction of a percent
// of either extreme, while either extreme is clearly worse on the other one.
//
// Define QZ_WEIGHT_SHAPE_FLAT to weight by importance alone, or
// QZ_WEIGHT_SHAPE_SQUARE for the x^2 end of the range.
// --------------------------------------------------------------------------

#ifndef QZ_WEIGHT_FLOOR
#  define QZ_WEIGHT_FLOOR 0.5f
#endif

static inline void qz_weights(const float * x, const float * imp, int n, float * w) {
    float sumx2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        sumx2 += x[i] * x[i];
    }
    const float floor2 = QZ_WEIGHT_FLOOR * sumx2 / (float) n;

    for (int i = 0; i < n; ++i) {
#if defined(QZ_WEIGHT_SHAPE_FLAT)
        w[i] = 1.0f;
        (void) floor2;
#elif defined(QZ_WEIGHT_SHAPE_SQUARE)
        w[i] = x[i] * x[i] + floor2;
#else
        w[i] = sqrtf(x[i] * x[i] + floor2);
#endif
        if (imp) {
            w[i] *= imp[i];
        }
    }
}

// weighted squared error of x against s*q
static inline float qz_sse_sym(const float * x, const float * w, const int8_t * q, int n, float s) {
    float e = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - s * (float) q[i];
        e += w[i] * d * d;
    }
    return e;
}

// weighted squared error of x against s*q - o
static inline float qz_sse_min(const float * x, const float * w, const uint8_t * q, int n, float s, float o) {
    float e = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - (s * (float) q[i] - o);
        e += w[i] * d * d;
    }
    return e;
}

// --------------------------------------------------------------------------
// symmetric fit: x[i] ~= s * q[i], q integer in [qlo, qhi]
//
// The integer range is deliberately lopsided in most formats (-8..7, -32..31),
// so the scale that maps the largest magnitude onto the low end and the one
// that maps it onto the high end are both worth trying; they differ by more
// than a sign once rounding enters. Returns s and fills q.
// --------------------------------------------------------------------------

static inline float qz_fit_sym(const float * x, const float * w, int n, int qlo, int qhi, int8_t * q) {
    float amax = 0.0f;
    float xext = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) {
            amax = a;
            xext = x[i];
        }
    }
    if (!(amax > 0.0f)) {
        memset(q, 0, (size_t) n);
        return 0.0f;
    }

    int8_t cur[256];
    float  best_s   = 0.0f;
    float  best_sse = -1.0f;

    const float ends[2] = { xext / (float) qlo, xext / (float) qhi };

    // The alternation below only finds the optimum nearest to where it starts,
    // and the scale that just fits the largest element is not always in the
    // right basin: pulling it in clips that element but buys resolution for
    // the rest of the group, which pays off whenever the group has an outlier.
    // So each end is also tried tightened by a few per cent.
    for (int t = 0; t < 2 * QZ_FIT_STARTS; ++t) {
        float s = ends[t % 2] * QZ_FIT_SHRINK[t / 2];
        if (!(fabsf(s) > 0.0f) || !isfinite(s)) {
            continue;
        }

        for (int it = 0; it < QZ_FIT_ROUNDS; ++it) {
            const float is = 1.0f / s;
            for (int i = 0; i < n; ++i) {
                cur[i] = (int8_t) qz_clampi(qz_lround(x[i] * is), qlo, qhi);
            }

            const float sse = qz_sse_sym(x, w, cur, n, s);
            if (best_sse < 0.0f || sse < best_sse) {
                best_sse = sse;
                best_s   = s;
                memcpy(q, cur, (size_t) n);
            }

            // re-solve the scale for this rounding
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float qf = (float) cur[i];
                num += w[i] * x[i] * qf;
                den += w[i] * qf * qf;
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

// --------------------------------------------------------------------------
// offset fit: x[i] ~= s * q[i] - o, q integer in [0, qmax], s >= 0, o >= 0
//
// The formats that carry a minimum store it as a non-negative multiple of a
// shared f16, hence the sign constraints. Returns s and writes the offset to
// *offset.
// --------------------------------------------------------------------------

static inline float qz_fit_min(const float * x, const float * w, int n, int qmax, uint8_t * q, float * offset) {
    float lo = x[0], hi = x[0];
    for (int i = 1; i < n; ++i) {
        lo = QZ_MIN(lo, x[i]);
        hi = QZ_MAX(hi, x[i]);
    }
    // the reconstruction always contains 0 (q = 0 gives -o with o >= 0), so a
    // group that never goes negative simply anchors at zero
    if (lo > 0.0f) {
        lo = 0.0f;
    }
    if (hi < 0.0f) {
        hi = 0.0f;
    }

    if (!(hi > lo)) {
        memset(q, 0, (size_t) n);
        *offset = 0.0f;
        return 0.0f;
    }

    uint8_t cur[256];
    float   best_s   = 0.0f;
    float   best_o   = -lo;
    float   best_sse = -1.0f;

    // Two things can go wrong with the obvious start (span the group exactly):
    // an outlier at either end stretches the span and coarsens every step, and
    // the two ends misbehave independently. So the ends are pulled in
    // separately, giving a small grid of starting spans; the alternation then
    // refines whichever of them landed in the best basin.
    for (int t = 0; t < QZ_FIT_STARTS * QZ_FIT_STARTS; ++t) {
        const float lo_t = lo * QZ_FIT_SHRINK[t % QZ_FIT_STARTS];
        const float hi_t = hi * QZ_FIT_SHRINK[t / QZ_FIT_STARTS];

        float s = (hi_t - lo_t) / (float) qmax;
        float o = -lo_t;
        if (o < 0.0f) {
            o = 0.0f;
        }
        if (!(s > 0.0f)) {
            continue;
        }

        for (int it = 0; it < QZ_FIT_ROUNDS; ++it) {
            const float is = 1.0f / s;
            for (int i = 0; i < n; ++i) {
                cur[i] = (uint8_t) qz_clampi(qz_lround((x[i] + o) * is), 0, qmax);
            }

            const float sse = qz_sse_min(x, w, cur, n, s, o);
            if (best_sse < 0.0f || sse < best_sse) {
                best_sse = sse;
                best_s   = s;
                best_o   = o;
                memcpy(q, cur, (size_t) n);
            }

            // re-solve (s, o) jointly for this rounding:
            //   [ Sq2 Sq ] [ s]   [ Sxq ]
            //   [ Sq  Sw ] [-o] = [ Sx  ]
            float sq2 = 0.0f, sq = 0.0f, sw = 0.0f, sxq = 0.0f, sx = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float qf = (float) cur[i];
                sq2 += w[i] * qf * qf;
                sq  += w[i] * qf;
                sw  += w[i];
                sxq += w[i] * x[i] * qf;
                sx  += w[i] * x[i];
            }

            const float det = sq2 * sw - sq * sq;
            float ns, no;
            if (fabsf(det) > 1e-30f) {
                ns = (sxq * sw - sq * sx) / det;
                no = -(sq2 * sx - sq * sxq) / det;
            } else {
                ns = sq2 > 0.0f ? sxq / sq2 : s;
                no = o;
            }
            if (no < 0.0f) {
                // offset pinned at zero: re-solve the scale alone
                no = 0.0f;
                ns = sq2 > 0.0f ? sxq / sq2 : s;
            }
            if (!(ns > 0.0f) || !isfinite(ns) || !isfinite(no)) {
                break;
            }
            if (ns == s && no == o) {
                break;
            }
            s = ns;
            o = no;
        }
    }

    *offset = best_o;
    return best_s;
}

// --------------------------------------------------------------------------
// group scale quantization
//
// The K-quants store one small integer scale per group plus a shared f16
// multiplier. Given the per-group scales this picks the multiplier that keeps
// the rounded integers closest, weighting each group by how much its error
// matters (the caller passes the group's own error sensitivity).
// --------------------------------------------------------------------------

// Core of both variants: fits v[g] ~= d * k[g] with k integer in [imin, imax],
// weighted by gw[]. Same alternation and same multi-start as the element-level
// fits, for the same reason - with only sixteen levels to hand out, the scale
// that just fits the largest group is often not the one that fits the rest.
static inline float qz_fit_scales(const float * v, const float * gw, int ng, int imin, int imax, int * k) {
    float vmax = 0.0f;
    for (int g = 0; g < ng; ++g) {
        vmax = QZ_MAX(vmax, fabsf(v[g]));
    }
    if (!(vmax > 0.0f)) {
        memset(k, 0, (size_t) ng * sizeof(int));
        return 0.0f;
    }

    int   cur[QZ_MAX_GROUPS];
    float best_d   = vmax / (float) imax;
    float best_err = -1.0f;

    for (int t = 0; t < QZ_FIT_STARTS; ++t) {
        float d = vmax / (float) imax * QZ_FIT_SHRINK[t];

        for (int it = 0; it < 4; ++it) {
            if (!(d > 0.0f) || !isfinite(d)) {
                break;
            }
            float err = 0.0f, num = 0.0f, den = 0.0f;
            for (int g = 0; g < ng; ++g) {
                cur[g] = qz_clampi(qz_lround(v[g] / d), imin, imax);
                const float e = v[g] - d * (float) cur[g];
                err += gw[g] * e * e;
                num += gw[g] * v[g] * (float) cur[g];
                den += gw[g] * (float) cur[g] * (float) cur[g];
            }

            if (best_err < 0.0f || err < best_err) {
                best_err = err;
                best_d   = d;
                memcpy(k, cur, (size_t) ng * sizeof(int));
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
    }

    return best_d;
}

// unsigned variant: values are non-negative, integers land in [0, imax]
static inline float qz_fit_group_scale(const float * v, const float * gw, int ng, int imax, uint8_t * iq) {
    int k[QZ_MAX_GROUPS];
    const float d = qz_fit_scales(v, gw, ng, 0, imax, k);
    for (int g = 0; g < ng; ++g) {
        iq[g] = (uint8_t) k[g];
    }
    return d;
}

// signed variant: integers land in [-imax, imax]
static inline float qz_fit_group_scale_sym(const float * v, const float * gw, int ng, int imax, int8_t * iq) {
    int k[QZ_MAX_GROUPS];
    const float d = qz_fit_scales(v, gw, ng, -imax, imax, k);
    for (int g = 0; g < ng; ++g) {
        iq[g] = (int8_t) k[g];
    }
    return d;
}

// --------------------------------------------------------------------------
// super-block refit
//
// After the per-group scales and mins have been squeezed into their small
// integers, the two f16 multipliers they share are no longer the best pair for
// the integers that actually got stored. Both enter the reconstruction
// linearly - x ~= d*(ls[g]*q) - dmin*lm[g] - so the pair that minimises the
// weighted error over the whole super-block is one 2x2 solve away. Recovering
// it costs nothing next to the search that produced the integers, and it takes
// back most of what the scale rounding gave away.
//
// `sq[i]` is ls[group of i] * q[i] and `sm[i]` is lm[group of i].
// --------------------------------------------------------------------------

static inline void qz_refit_dm(const float * x, const float * w, const float * sq, const float * sm, int n,
                               float * d, float * dmin) {
    float a11 = 0.0f, a12 = 0.0f, a22 = 0.0f, b1 = 0.0f, b2 = 0.0f;

    for (int i = 0; i < n; ++i) {
        a11 += w[i] * sq[i] * sq[i];
        a12 -= w[i] * sq[i] * sm[i];
        a22 += w[i] * sm[i] * sm[i];
        b1  += w[i] * x[i] * sq[i];
        b2  -= w[i] * x[i] * sm[i];
    }

    const float det = a11 * a22 - a12 * a12;
    if (!(fabsf(det) > 1e-30f)) {
        if (a11 > 0.0f) {
            *d = b1 / a11;
        }
        return;
    }

    const float nd = (b1 * a22 - a12 * b2) / det;
    const float nm = (a11 * b2 - a12 * b1) / det;

    // the stored min is a non-negative multiple of dmin, so a negative dmin
    // would flip every group's offset; leave the pair alone in that case
    if (isfinite(nd) && isfinite(nm) && nd > 0.0f && nm >= 0.0f) {
        *d    = nd;
        *dmin = nm;
    }
}

// symmetric variant: only one multiplier to re-solve
static inline void qz_refit_d(const float * x, const float * w, const float * sq, int n, float * d) {
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < n; ++i) {
        num += w[i] * x[i] * sq[i];
        den += w[i] * sq[i] * sq[i];
    }
    if (den > 0.0f) {
        const float nd = num / den;
        if (isfinite(nd) && nd != 0.0f) {
            *d = nd;
        }
    }
}

// --------------------------------------------------------------------------
// group polish
//
// The per-group scale and min are fitted as real numbers and only then rounded
// into their four or six bits. For the wider formats that rounding is a small
// perturbation, but Q2_K spends four bits on each and its groups are only
// sixteen weights long, so the rounded pair is regularly not the best pair
// available. This tries the neighbouring integers - which cost one evaluation
// of the group each - and keeps whichever reconstructs the group best.
// --------------------------------------------------------------------------

static inline float qz_group_error(const float * x, const float * w, int n, int qmax, float dl, float ml) {
    if (!(dl > 0.0f)) {
        float e = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float d = x[i] + ml;
            e += w[i] * d * d;
        }
        return e;
    }

    const float idl = 1.0f / dl;
    float e = 0.0f;
    for (int i = 0; i < n; ++i) {
        const int k = qz_clampi(qz_lround((x[i] + ml) * idl), 0, qmax);
        const float d = x[i] - (dl * (float) k - ml);
        e += w[i] * d * d;
    }
    return e;
}

static inline void qz_polish_group(const float * x, const float * w, int n, int qmax, float dq, float mq,
                                   int imax, int * ls, int * lm) {
    int   best_s = *ls, best_m = *lm;
    float best_e = qz_group_error(x, w, n, qmax, dq * (float) *ls, mq * (float) *lm);

    for (int ds = -1; ds <= 1; ++ds) {
        for (int dm = -1; dm <= 1; ++dm) {
            if (ds == 0 && dm == 0) {
                continue;
            }
            const int s = qz_clampi(*ls + ds, 0, imax);
            const int m = qz_clampi(*lm + dm, 0, imax);
            const float e = qz_group_error(x, w, n, qmax, dq * (float) s, mq * (float) m);
            if (e < best_e) {
                best_e = e;
                best_s = s;
                best_m = m;
            }
        }
    }

    *ls = best_s;
    *lm = best_m;
}

// --------------------------------------------------------------------------
// nearest entry of a sorted codebook (the non-linear 4-bit tables)
// --------------------------------------------------------------------------

static inline int qz_nearest_sorted(const int8_t * tab, int n, float v) {
    int lo = 0, hi = n - 1;
    if (v <= (float) tab[0]) {
        return 0;
    }
    if (v >= (float) tab[n - 1]) {
        return n - 1;
    }
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (v < (float) tab[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return (v - (float) tab[lo] <= (float) tab[hi] - v) ? lo : hi;
}

#ifdef __cplusplus
}
#endif
