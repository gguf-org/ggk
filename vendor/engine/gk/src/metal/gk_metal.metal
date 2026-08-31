// The Metal kernels.
//
// The shape is the same as the CUDA backend's, for the same reasons: one
// thread per destination element for the elementwise ops, one threadgroup per
// destination row for the ops with a reduction along dimension 0, and strides
// honoured everywhere so a permuted view needs no materialising.
//
// This source is compiled at load time from a copy embedded in the library,
// rather than shipped as a .metallib next to it. A library that depends on a
// file it cannot guarantee is installed is a library that fails in the field
// for reasons the user cannot act on; a few hundred kilobytes of source in
// .rodata is the cheaper promise. The compile happens once per process.
//
// The quantized formats decoded here are the block-scale and K families, plus
// the two non-linear 4-bit ones. The lattice formats (IQ1, IQ2, IQ3) need
// their codebooks resident and are deliberately absent; a matmul against one
// is reported unsupported and runs on the CPU.

#include <metal_stdlib>

using namespace metal;

// --------------------------------------------------------------------------
// types, mirrored from gk.h
// --------------------------------------------------------------------------

#define GKT_F32     0
#define GKT_F16     1
#define GKT_Q4_0    2
#define GKT_Q4_1    3
#define GKT_Q5_0    6
#define GKT_Q5_1    7
#define GKT_Q8_0    8
#define GKT_Q2_K   10
#define GKT_Q3_K   11
#define GKT_Q4_K   12
#define GKT_Q5_K   13
#define GKT_Q6_K   14
#define GKT_IQ4_NL 20
#define GKT_IQ4_XS 23
#define GKT_I32    26
#define GKT_I64    27
#define GKT_BF16   30
#define GKT_MXFP4  39
#define GKT_NVFP4  40

#define GK_QK 256

// --------------------------------------------------------------------------
// tensor views
//
// The host fills one of these per operand. `data` is an offset into the buffer
// bound at the matching index rather than a pointer, because Metal binds
// buffers rather than handing out addresses.
// --------------------------------------------------------------------------

struct gk_mtl_tview {
    long ne[4];
    long nb[4];
    int  type;
    int  pad;
};

struct gk_mtl_params {
    gk_mtl_tview src0;
    gk_mtl_tview src1;
    gk_mtl_tview src2;
    gk_mtl_tview dst;

    // whatever the op needs: scales, epsilons, modes. Named per op below
    // rather than in a union, because a union of thirty op parameter sets is
    // harder to read than four floats and four ints with comments at the use.
    float f[8];
    int   i[16];

    long  n;      // how many destination elements this launch covers
    int   flags;
    int   pad2;
};

// --------------------------------------------------------------------------
// narrow floats and codebooks
// --------------------------------------------------------------------------

static inline float gk_mtl_bf2f(ushort bits) {
    return as_type<float>((uint) bits << 16);
}

static inline float gk_mtl_e8m0_half(uchar e) {
    if (e >= 2) {
        return as_type<float>((uint) (e - 1) << 23);
    }
    return as_type<float>(0x00200000u << e);
}

// UE4M3 scale, halved like E8M0 above (the e2m1 table is doubled). 0x7f is
// the NaN slot and decodes as zero.
static inline float gk_mtl_ue4m3_half(uchar v) {
    if (v == 0 || v == 0x7f) {
        return 0.0f;
    }
    const uint e = (v >> 3) & 0xfu;
    const uint m = v & 0x7u;
    if (e == 0) {
        return (float) m * (1.0f / 512.0f) * 0.5f; // subnormal: m * 2^-9, halved
    }
    return as_type<float>(((e + 120u) << 23) | (m << 20)) * 0.5f;
}

constant char gk_mtl_iq4_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

constant char gk_mtl_e2m1_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static inline int gk_mtl_blck_size(int type) {
    switch (type) {
        case GKT_Q4_0: case GKT_Q4_1: case GKT_Q5_0: case GKT_Q5_1:
        case GKT_Q8_0: case GKT_IQ4_NL: case GKT_MXFP4:
            return 32;
        case GKT_NVFP4:
            return 64;
        case GKT_Q2_K: case GKT_Q3_K: case GKT_Q4_K: case GKT_Q5_K:
        case GKT_Q6_K: case GKT_IQ4_XS:
            return GK_QK;
        default:
            return 1;
    }
}

static inline int gk_mtl_type_size(int type) {
    switch (type) {
        case GKT_F32:    return 4;
        case GKT_F16:    return 2;
        case GKT_BF16:   return 2;
        case GKT_I32:    return 4;
        case GKT_I64:    return 8;
        case GKT_Q4_0:   return 18;
        case GKT_Q4_1:   return 20;
        case GKT_Q5_0:   return 22;
        case GKT_Q5_1:   return 24;
        case GKT_Q8_0:   return 34;
        case GKT_MXFP4:  return 17;
        case GKT_NVFP4:  return 36;
        case GKT_Q2_K:   return 4 + GK_QK / 16 + GK_QK / 4;
        case GKT_Q3_K:   return 2 + GK_QK / 4 + GK_QK / 8 + 12;
        case GKT_Q4_K:   return 4 + 12 + GK_QK / 2;
        case GKT_Q5_K:   return 4 + 12 + GK_QK / 2 + GK_QK / 8;
        case GKT_Q6_K:   return 2 + GK_QK / 16 + 3 * GK_QK / 4;
        case GKT_IQ4_NL: return 18;
        case GKT_IQ4_XS: return 2 + 2 + GK_QK / 64 + GK_QK / 2;
        default:         return 0;
    }
}

// f16 out of two bytes, without assuming the block is 2-byte aligned
static inline float gk_mtl_h2f(device const uchar * p) {
    const ushort bits = (ushort) p[0] | ((ushort) p[1] << 8);
    return (float) as_type<half>(bits);
}

// --------------------------------------------------------------------------
// block decoding
//
// One function per format, each taking the block's address and the element's
// index within it. Transcribed from the codec in ../../quantizer, which is the
// definition of these bytes.
// --------------------------------------------------------------------------

static inline void gk_mtl_scale_min_6(device const uchar * src, int g,
                                      thread int & scale, thread int & min_out) {
    if (g < 4) {
        scale   = src[g] & 63;
        min_out = src[g + 4] & 63;
    } else {
        scale   = (src[g + 4] & 0xf) | ((src[g - 4] >> 6) << 4);
        min_out = (src[g + 4] >> 4)  | ((src[g]     >> 6) << 4);
    }
}

static float gk_mtl_block_elem(device const uchar * b, int type, int j) {
    switch (type) {
        case GKT_Q4_0: {
            const float d = gk_mtl_h2f(b);
            device const uchar * qs = b + 2;
            const int q = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
            return d * (float) (q - 8);
        }
        case GKT_Q4_1: {
            const float d = gk_mtl_h2f(b);
            const float m = gk_mtl_h2f(b + 2);
            device const uchar * qs = b + 4;
            const int q = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
            return d * (float) q + m;
        }
        case GKT_Q5_0: {
            const float d = gk_mtl_h2f(b);
            const uint qh = (uint) b[2] | ((uint) b[3] << 8) | ((uint) b[4] << 16) | ((uint) b[5] << 24);
            device const uchar * qs = b + 6;
            const int nib = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
            const int q = nib | (int) (((qh >> j) & 1u) << 4);
            return d * (float) (q - 16);
        }
        case GKT_Q5_1: {
            const float d = gk_mtl_h2f(b);
            const float m = gk_mtl_h2f(b + 2);
            const uint qh = (uint) b[4] | ((uint) b[5] << 8) | ((uint) b[6] << 16) | ((uint) b[7] << 24);
            device const uchar * qs = b + 8;
            const int nib = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
            const int q = nib | (int) (((qh >> j) & 1u) << 4);
            return d * (float) q + m;
        }
        case GKT_Q8_0: {
            const float d = gk_mtl_h2f(b);
            device const char * qs = (device const char *) (b + 2);
            return d * (float) qs[j];
        }
        case GKT_MXFP4: {
            const float d = gk_mtl_e8m0_half(b[0]);
            device const uchar * qs = b + 1;
            const int code = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
            return d * (float) gk_mtl_e2m1_values[code];
        }
        case GKT_NVFP4: {
            // 64 elements in four groups of 16; one UE4M3 scale per group,
            // nibble halving per group rather than per block
            const int s = j / 16;
            const int r = j % 16;
            const float d = gk_mtl_ue4m3_half(b[s]);
            device const uchar * qs = b + 4 + s * 8;
            const int code = r < 8 ? (qs[r] & 0xf) : (qs[r - 8] >> 4);
            return d * (float) gk_mtl_e2m1_values[code];
        }
        case GKT_Q2_K: {
            device const uchar * scales = b;
            device const uchar * qs     = b + GK_QK / 16;
            const float d    = gk_mtl_h2f(b + GK_QK / 16 + GK_QK / 4);
            const float dmin = gk_mtl_h2f(b + GK_QK / 16 + GK_QK / 4 + 2);

            const int half_i = j / 128;
            const int r      = j % 128;
            const int shift  = 2 * (r / 32);
            const int part   = (r % 32) / 16;
            const int l      = r % 16;

            const int g = half_i * 8 + (shift / 2) * 2 + part;
            const uchar sc = scales[g];
            const int q = (qs[half_i * 32 + part * 16 + l] >> shift) & 3;

            return d * (float) (sc & 0xf) * (float) q - dmin * (float) (sc >> 4);
        }
        case GKT_Q3_K: {
            device const uchar * hmask  = b;
            device const uchar * qs     = b + GK_QK / 8;
            device const uchar * scales = b + GK_QK / 8 + GK_QK / 4;
            const float d = gk_mtl_h2f(b + GK_QK / 8 + GK_QK / 4 + 12);

            const int half_i = j / 128;
            const int r      = j % 128;
            const int shift  = 2 * (r / 32);
            const int bit    = half_i * 4 + (r / 32);
            const int part   = (r % 32) / 16;
            const int l      = r % 16;
            const int idx    = part * 16 + l;

            const int g = half_i * 8 + (shift / 2) * 2 + part;

            const int low  = g < 8 ? (scales[g] & 0xf) : (scales[g - 8] >> 4);
            const int high = (scales[8 + (g % 4)] >> (2 * (g / 4))) & 3;
            const int sc   = (low | (high << 4)) - 32;

            const int lo = (qs[half_i * 32 + idx] >> shift) & 3;
            // the high bit is stored inverted: a set mask bit means "do not
            // subtract 4"
            const int v = lo - ((hmask[idx] & (1u << bit)) ? 0 : 4);

            return d * (float) sc * (float) v;
        }
        case GKT_Q4_K: {
            const float d    = gk_mtl_h2f(b);
            const float dmin = gk_mtl_h2f(b + 2);
            device const uchar * scales = b + 4;
            device const uchar * qs     = b + 16;

            const int g = j / 32;
            const int l = j % 32;

            int sc, mn;
            gk_mtl_scale_min_6(scales, g, sc, mn);

            const int q = (qs[(g / 2) * 32 + l] >> ((g % 2) * 4)) & 0xf;
            return d * (float) sc * (float) q - dmin * (float) mn;
        }
        case GKT_Q5_K: {
            const float d    = gk_mtl_h2f(b);
            const float dmin = gk_mtl_h2f(b + 2);
            device const uchar * scales = b + 4;
            device const uchar * qh     = b + 16;
            device const uchar * qs     = b + 16 + GK_QK / 8;

            const int g = j / 32;
            const int l = j % 32;

            int sc, mn;
            gk_mtl_scale_min_6(scales, g, sc, mn);

            const int lo = (qs[(g / 2) * 32 + l] >> ((g % 2) * 4)) & 0xf;
            const int q  = lo | ((qh[l] & (1u << g)) ? 16 : 0);

            return d * (float) sc * (float) q - dmin * (float) mn;
        }
        case GKT_Q6_K: {
            device const uchar * ql = b;
            device const uchar * qh = b + GK_QK / 2;
            device const char  * sc = (device const char *) (b + GK_QK / 2 + GK_QK / 4);
            const float d = gk_mtl_h2f(b + GK_QK / 2 + GK_QK / 4 + GK_QK / 16);

            const int half_i = j / 128;
            const int r      = j % 128;
            const int which  = r / 32;
            const int i      = r % 32;
            const int is     = i / 16;

            device const uchar * l = ql + half_i * 64;
            device const uchar * h = qh + half_i * 32;
            device const char  * s = sc + half_i * 8;

            int q, scale;
            switch (which) {
                case 0: q = (l[i]      & 0xf) | (((h[i] >> 0) & 3) << 4); scale = s[is];     break;
                case 1: q = (l[i + 32] & 0xf) | (((h[i] >> 2) & 3) << 4); scale = s[is + 2]; break;
                case 2: q = (l[i]      >> 4)  | (((h[i] >> 4) & 3) << 4); scale = s[is + 4]; break;
                default:q = (l[i + 32] >> 4)  | (((h[i] >> 6) & 3) << 4); scale = s[is + 6]; break;
            }

            return d * (float) scale * (float) (q - 32);
        }
        case GKT_IQ4_NL: {
            const float d = gk_mtl_h2f(b);
            device const uchar * qs = b + 2;
            const int code = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
            return d * (float) gk_mtl_iq4_values[code];
        }
        case GKT_IQ4_XS: {
            const float d = gk_mtl_h2f(b);
            const ushort scales_h = (ushort) b[2] | ((ushort) b[3] << 8);
            device const uchar * scales_l = b + 4;
            device const uchar * qs       = b + 4 + GK_QK / 64;

            const int g = j / 32;
            const int r = j % 32;

            const int ls = ((scales_l[g / 2] >> (4 * (g % 2))) & 0xf) |
                           (int) (((scales_h >> (2 * g)) & 3) << 4);
            const float dl = d * (float) (ls - 32);

            device const uchar * q = qs + g * 16;
            const int code = r < 16 ? (q[r] & 0xf) : (q[r - 16] >> 4);

            return dl * (float) gk_mtl_iq4_values[code];
        }
        default:
            return 0.0f;
    }
}

// --------------------------------------------------------------------------
// element access
// --------------------------------------------------------------------------

static inline device const uchar * gk_mtl_row(device const uchar * base,
                                              constant gk_mtl_tview & t,
                                              long i1, long i2, long i3) {
    return base + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
}

static float gk_mtl_get(device const uchar * base, constant gk_mtl_tview & t,
                        long i0, long i1, long i2, long i3) {
    device const uchar * row = gk_mtl_row(base, t, i1, i2, i3);

    switch (t.type) {
        case GKT_F32:  return *(device const float *)  (row + i0 * t.nb[0]);
        case GKT_F16:  return (float) *(device const half *) (row + i0 * t.nb[0]);
        case GKT_BF16: return gk_mtl_bf2f(*(device const ushort *) (row + i0 * t.nb[0]));
        case GKT_I32:  return (float) *(device const int *) (row + i0 * t.nb[0]);
        default: break;
    }

    // every block size is a power of two, and the GPU has no integer divide
    const int shift = ctz((uint) gk_mtl_blck_size(t.type));
    const int tsz   = gk_mtl_type_size(t.type);

    return gk_mtl_block_elem(row + (i0 >> shift) * tsz, t.type, (int) (i0 & ((1 << shift) - 1)));
}

static void gk_mtl_set(device uchar * base, constant gk_mtl_tview & t,
                       long i0, long i1, long i2, long i3, float v) {
    device uchar * p = base + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3] + i0 * t.nb[0];

    switch (t.type) {
        case GKT_F32:  *(device float *) p = v; break;
        case GKT_F16:  *(device half *)  p = (half) v; break;
        case GKT_BF16: {
            const uint u = as_type<uint>(v);
            const uint rounded = u + 0x7fffu + ((u >> 16) & 1u);
            *(device ushort *) p = (ushort) (rounded >> 16);
            break;
        }
        case GKT_I32:  *(device int *) p = (int) v; break;
        default: break;
    }
}

static float gk_mtl_get_dst(device const uchar * base, constant gk_mtl_tview & t,
                            long i0, long i1, long i2, long i3) {
    device const uchar * p = base + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3] + i0 * t.nb[0];

    switch (t.type) {
        case GKT_F16:  return (float) *(device const half *) p;
        case GKT_BF16: return gk_mtl_bf2f(*(device const ushort *) p);
        default:       return *(device const float *) p;
    }
}

struct gk_mtl_idx {
    long i0, i1, i2, i3;
};

static gk_mtl_idx gk_mtl_decompose(long k, constant gk_mtl_tview & t) {
    gk_mtl_idx x;
    x.i0 = k % t.ne[0];
    x.i1 = (k / t.ne[0]) % t.ne[1];
    x.i2 = (k / (t.ne[0] * t.ne[1])) % t.ne[2];
    x.i3 = k / (t.ne[0] * t.ne[1] * t.ne[2]);
    return x;
}

// --------------------------------------------------------------------------
// activations
// --------------------------------------------------------------------------

static float gk_mtl_erf_poly(float x) {
    // Abramowitz and Stegun 7.1.26 - the same polynomial the CPU pass uses, so
    // GELU_ERF agrees to the last bits wherever a graph happens to be split.
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabs(x);

    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    const float t = 1.0f / (1.0f + p * x);
    const float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}

static float gk_mtl_gelu(float x) {
    const float c = 0.797884560802865f;
    return 0.5f * x * (1.0f + precise::tanh(c * (x + 0.044715f * x * x * x)));
}

static float gk_mtl_gelu_erf(float x) {
    return 0.5f * x * (1.0f + gk_mtl_erf_poly(x * 0.7071067811865475f));
}

static float gk_mtl_gelu_quick(float x) {
    return x * (1.0f / (1.0f + exp(-1.702f * x)));
}

static float gk_mtl_silu(float x) {
    return x / (1.0f + exp(-x));
}

// the unary op codes, in gk.h's order
static float gk_mtl_unary(int op, float x, float p1, float p2, float p3, float p4) {
    switch (op) {
        case 0:  return fabs(x);
        case 1:  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        case 2:  return -x;
        case 3:  return x > 0.0f ? 1.0f : 0.0f;
        case 4:  return precise::tanh(x);
        case 5:  return x > 0.0f ? x : (exp(x) - 1.0f);
        case 6:  return x > 0.0f ? x : 0.0f;
        case 7:  return 1.0f / (1.0f + exp(-x));
        case 8:  return gk_mtl_gelu(x);
        case 9:  return gk_mtl_gelu_quick(x);
        case 10: return gk_mtl_silu(x);
        case 11: return x * min(1.0f, max(0.0f, (x + 3.0f) / 6.0f));
        case 12: return min(1.0f, max(0.0f, (x + 3.0f) / 6.0f));
        case 13: return exp(x);
        case 14: return exp(x) - 1.0f;
        case 15: return x > 20.0f ? x : log(1.0f + exp(x));
        case 16: return gk_mtl_gelu_erf(x);
        case 17: { // xielu
            if (x > 0.0f) {
                return p2 * x * x + p3 * x;
            }
            const float mx = min(x, p4);
            return (exp(mx) - 1.0f - x) * p1 + p3 * x;
        }
        case 18: return floor(x);
        case 19: return ceil(x);
        case 20: return rint(x);
        case 21: return trunc(x);
        default: return x;
    }
}

// --------------------------------------------------------------------------
// elementwise kernels
// --------------------------------------------------------------------------

kernel void gk_mtl_binary(device const uchar * a  [[buffer(0)]],
                          device const uchar * b  [[buffer(1)]],
                          device uchar *       d  [[buffer(2)]],
                          constant gk_mtl_params & p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    const float va = gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3);
    const float vb = gk_mtl_get(b, p.src1, x.i0 % p.src1.ne[0], x.i1 % p.src1.ne[1],
                                           x.i2 % p.src1.ne[2], x.i3 % p.src1.ne[3]);

    float r;
    switch (p.i[0]) {
        case 0:  r = va + vb; break;
        case 1:  r = va - vb; break;
        case 2:  r = va * vb; break;
        default: r = va / vb; break;
    }

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, r);
}

kernel void gk_mtl_unary_op(device const uchar * a [[buffer(0)]],
                            device uchar *       d [[buffer(2)]],
                            constant gk_mtl_params & p [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);
    const float v = gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3);

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3,
               gk_mtl_unary(p.i[0], v, p.f[0], p.f[1], p.f[2], p.f[3]));
}

// sqr / sqrt / log / sin / cos
kernel void gk_mtl_simple(device const uchar * a [[buffer(0)]],
                          device uchar *       d [[buffer(2)]],
                          constant gk_mtl_params & p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);
    const float v = gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3);

    float r;
    switch (p.i[0]) {
        case 0:  r = v * v;   break;
        case 1:  r = sqrt(v); break;
        case 2:  r = log(v);  break;
        case 3:  r = sin(v);  break;
        default: r = cos(v);  break;
    }

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, r);
}

// scale (f[0] = s, f[1] = bias), clamp (f[0] = lo, f[1] = hi), fill (f[0] = c)
// and leaky_relu (f[0] = slope) share a kernel selected by i[0]
kernel void gk_mtl_affine(device const uchar * a [[buffer(0)]],
                          device uchar *       d [[buffer(2)]],
                          constant gk_mtl_params & p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    float r;
    if (p.i[0] == 2) { // fill needs no source
        r = p.f[0];
    } else {
        const float v = gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3);
        switch (p.i[0]) {
            case 0:  r = v * p.f[0] + p.f[1];      break;
            case 1:  r = min(p.f[1], max(p.f[0], v)); break;
            default: r = v > 0.0f ? v : v * p.f[0]; break;
        }
    }

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, r);
}

kernel void gk_mtl_glu(device const uchar * a [[buffer(0)]],
                       device const uchar * b [[buffer(1)]],
                       device uchar *       d [[buffer(2)]],
                       constant gk_mtl_params & p [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);
    const long half_w = p.dst.ne[0];

    float act, mul;
    if (p.flags != 0) { // a separate gate tensor
        act = gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3);
        mul = gk_mtl_get(b, p.src1, x.i0, x.i1, x.i2, x.i3);
    } else {
        const long ia = p.i[1] != 0 ? x.i0 + half_w : x.i0;
        const long im = p.i[1] != 0 ? x.i0          : x.i0 + half_w;
        act = gk_mtl_get(a, p.src0, ia, x.i1, x.i2, x.i3);
        mul = gk_mtl_get(a, p.src0, im, x.i1, x.i2, x.i3);
    }

    float r;
    switch (p.i[0]) {
        case 0:  r = (act > 0.0f ? act : 0.0f) * mul; break;
        case 1:  r = gk_mtl_gelu(act) * mul;          break;
        case 2:  r = gk_mtl_silu(act) * mul;          break;
        case 4:  r = gk_mtl_gelu_erf(act) * mul;      break;
        case 5:  r = gk_mtl_gelu_quick(act) * mul;    break;
        default: {
            // the gpt-oss variant: clamped operands, a sigmoid with a
            // temperature, +1 on the gate
            const float alpha = p.f[0];
            const float limit = p.f[1];
            const float xv = min(act, limit);
            const float yv = min(max(mul, -limit), limit);
            const float g  = xv / (1.0f + exp(alpha * (-xv)));
            r = g * (yv + 1.0f);
            break;
        }
    }

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, r);
}

kernel void gk_mtl_copy(device const uchar * a [[buffer(0)]],
                        device uchar *       d [[buffer(2)]],
                        constant gk_mtl_params & p [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    float v;
    if (p.flags != 0) { // same shape: positions line up
        v = gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3);
    } else {            // otherwise the copy is over the flat element order
        const gk_mtl_idx s = gk_mtl_decompose(k, p.src0);
        v = gk_mtl_get(a, p.src0, s.i0, s.i1, s.i2, s.i3);
    }

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, v);
}

kernel void gk_mtl_get_rows(device const uchar * a   [[buffer(0)]],
                            device const uchar * idx [[buffer(1)]],
                            device uchar *       d   [[buffer(2)]],
                            constant gk_mtl_params & p [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    device const int * r = (device const int *) (idx + x.i1 * p.src1.nb[0]
                                                     + x.i2 * p.src1.nb[1]
                                                     + x.i3 * p.src1.nb[2]);

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3,
               gk_mtl_get(a, p.src0, x.i0, (long) *r, x.i2, x.i3));
}

kernel void gk_mtl_set_rows(device const uchar * b [[buffer(0)]],
                            device const uchar * c [[buffer(1)]],
                            device uchar *       d [[buffer(2)]],
                            constant gk_mtl_params & p [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.src0);

    device const uchar * pi = c + x.i1 * p.src1.nb[0]
                                + (x.i2 % p.src1.ne[1]) * p.src1.nb[1]
                                + (x.i3 % p.src1.ne[2]) * p.src1.nb[2];

    const long row = p.flags != 0 ? (long) *(device const long *) pi
                                  : (long) *(device const int *) pi;

    gk_mtl_set(d, p.dst, x.i0, row, x.i2, x.i3,
               gk_mtl_get(b, p.src0, x.i0, x.i1, x.i2, x.i3));
}

kernel void gk_mtl_repeat(device const uchar * a [[buffer(0)]],
                          device uchar *       d [[buffer(2)]],
                          constant gk_mtl_params & p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3,
               gk_mtl_get(a, p.src0, x.i0 % p.src0.ne[0], x.i1 % p.src0.ne[1],
                                     x.i2 % p.src0.ne[2], x.i3 % p.src0.ne[3]));
}

kernel void gk_mtl_concat(device const uchar * a [[buffer(0)]],
                          device const uchar * b [[buffer(1)]],
                          device uchar *       d [[buffer(2)]],
                          constant gk_mtl_params & p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    long j0 = x.i0, j1 = x.i1, j2 = x.i2, j3 = x.i3;
    bool second = false;

    switch (p.i[0]) {
        case 0: second = x.i0 >= p.src0.ne[0]; if (second) j0 -= p.src0.ne[0]; break;
        case 1: second = x.i1 >= p.src0.ne[1]; if (second) j1 -= p.src0.ne[1]; break;
        case 2: second = x.i2 >= p.src0.ne[2]; if (second) j2 -= p.src0.ne[2]; break;
        default:second = x.i3 >= p.src0.ne[3]; if (second) j3 -= p.src0.ne[3]; break;
    }

    const float v = second ? gk_mtl_get(b, p.src1, j0, j1, j2, j3)
                           : gk_mtl_get(a, p.src0, j0, j1, j2, j3);

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, v);
}

kernel void gk_mtl_add_id(device const uchar * a   [[buffer(0)]],
                          device const uchar * b   [[buffer(1)]],
                          device const uchar * ids [[buffer(4)]],
                          device uchar *       d   [[buffer(2)]],
                          constant gk_mtl_params & p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    const int row = *(device const int *) (ids + x.i1 * p.src2.nb[0] + x.i2 * p.src2.nb[1]);

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3,
               gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3) +
               gk_mtl_get(b, p.src1, x.i0, (long) row, 0, 0));
}

kernel void gk_mtl_diag_mask(device const uchar * a [[buffer(0)]],
                             device uchar *       d [[buffer(2)]],
                             constant gk_mtl_params & p [[buffer(3)]],
                             uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);
    const float v = gk_mtl_get(a, p.src0, x.i0, x.i1, x.i2, x.i3);

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3,
               x.i0 > (long) p.i[0] + x.i1 ? p.f[0] : v);
}

kernel void gk_mtl_pad(device const uchar * a [[buffer(0)]],
                       device uchar *       d [[buffer(2)]],
                       constant gk_mtl_params & p [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    const int lp0 = p.i[0], rp0 = p.i[1];
    const int lp1 = p.i[2], rp1 = p.i[3];
    const int lp2 = p.i[4], rp2 = p.i[5];
    const int lp3 = p.i[6], rp3 = p.i[7];

    if (p.flags != 0) { // circular
        const long j0 = ((x.i0 - lp0) % p.src0.ne[0] + p.src0.ne[0]) % p.src0.ne[0];
        const long j1 = ((x.i1 - lp1) % p.src0.ne[1] + p.src0.ne[1]) % p.src0.ne[1];
        const long j2 = ((x.i2 - lp2) % p.src0.ne[2] + p.src0.ne[2]) % p.src0.ne[2];
        const long j3 = ((x.i3 - lp3) % p.src0.ne[3] + p.src0.ne[3]) % p.src0.ne[3];
        gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, gk_mtl_get(a, p.src0, j0, j1, j2, j3));
        return;
    }

    const bool inside =
        x.i0 >= lp0 && x.i0 < p.dst.ne[0] - rp0 &&
        x.i1 >= lp1 && x.i1 < p.dst.ne[1] - rp1 &&
        x.i2 >= lp2 && x.i2 < p.dst.ne[2] - rp2 &&
        x.i3 >= lp3 && x.i3 < p.dst.ne[3] - rp3;

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3,
               inside ? gk_mtl_get(a, p.src0, x.i0 - lp0, x.i1 - lp1, x.i2 - lp2, x.i3 - lp3)
                      : 0.0f);
}

// im2col: one thread per destination element. i[0..5] = s0,s1,p0,p1,d0,d1,
// i[6] = is_2D. src0 is the conv kernel and contributes only its shape; the
// image is src1 and is always f32.
kernel void gk_mtl_im2col(device const uchar * a [[buffer(0)]],
                          device const uchar * b [[buffer(1)]],
                          device uchar *       d [[buffer(2)]],
                          constant gk_mtl_params & p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const gk_mtl_idx x = gk_mtl_decompose(k, p.dst);

    const bool is_2d = p.i[6] != 0;

    const long KW = p.src0.ne[0];
    const long KH = is_2d ? p.src0.ne[1] : 1;

    // x.i0 is iic * (KH*KW) + ikh * KW + ikw, matching the CPU's cell layout
    const long iic = x.i0 / (KH * KW);
    const long r   = x.i0 % (KH * KW);
    const long ikh = r / KW;
    const long ikw = r % KW;

    const long iow = x.i1;
    const long ioh = is_2d ? x.i2 : 0;
    const long in  = is_2d ? x.i3 : x.i2;

    const long iiw = iow * p.i[0] + ikw * p.i[4] - p.i[2];
    const long iih = ioh * p.i[1] + ikh * p.i[5] - p.i[3];

    const long IW = p.src1.ne[0];
    const long IH = is_2d ? p.src1.ne[1] : 1;

    float v = 0.0f;
    if (iih >= 0 && iih < IH && iiw >= 0 && iiw < IW) {
        const long ofs = in  * (is_2d ? p.src1.nb[3] : p.src1.nb[2])
                       + iic * (is_2d ? p.src1.nb[2] : p.src1.nb[1])
                       + iih * (is_2d ? p.src1.nb[1] : 0)
                       + iiw * p.src1.nb[0];
        v = *(device const float *) (b + ofs);
    }

    gk_mtl_set(d, p.dst, x.i0, x.i1, x.i2, x.i3, v);
}

// --------------------------------------------------------------------------
// row-wise kernels
//
// One threadgroup per destination row. The reduction is the usual two stages:
// SIMD-group first, then across the group's SIMD groups through threadgroup
// memory.
// --------------------------------------------------------------------------

#define GK_MTL_ROW_TG 256

static float gk_mtl_group_sum(float x, threadgroup float * scratch,
                              uint tiisg, uint sgitg, uint n_sg) {
    x = simd_sum(x);
    if (n_sg == 1) {
        return x;
    }

    if (tiisg == 0) {
        scratch[sgitg] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint i = 0; i < n_sg; ++i) {
        total += scratch[i];
    }
    return total;
}

static float gk_mtl_group_max(float x, threadgroup float * scratch,
                              uint tiisg, uint sgitg, uint n_sg) {
    x = simd_max(x);
    if (n_sg == 1) {
        return x;
    }

    if (tiisg == 0) {
        scratch[sgitg] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best = scratch[0];
    for (uint i = 1; i < n_sg; ++i) {
        best = max(best, scratch[i]);
    }
    return best;
}

// i[0]: 0 = rms_norm, 1 = norm, 2 = l2_norm; f[0] = eps
kernel void gk_mtl_norm(device const uchar * a [[buffer(0)]],
                        device uchar *       d [[buffer(2)]],
                        constant gk_mtl_params & p [[buffer(3)]],
                        threadgroup float * scratch [[threadgroup(0)]],
                        uint3 tgpig  [[threadgroup_position_in_grid]],
                        uint3 tpitg3 [[thread_position_in_threadgroup]],
                        uint3 ntg3   [[threads_per_threadgroup]],
                        uint  tiisg [[thread_index_in_simdgroup]],
                        uint  sgitg [[simdgroup_index_in_threadgroup]],
                        uint  nsg   [[simdgroups_per_threadgroup]]) {
    // the grid-position attributes must all be scalar or all be vectors of the
    // same width in one kernel; tgpig is uint3, so these are too
    const uint tpitg = tpitg3.x;
    const uint ntg   = ntg3.x;
    const long ir = (long) tgpig.x;
    const long i1 = ir % p.dst.ne[1];
    const long i2 = (ir / p.dst.ne[1]) % p.dst.ne[2];
    const long i3 = ir / (p.dst.ne[1] * p.dst.ne[2]);

    const long n = p.dst.ne[0];
    const float eps = p.f[0];

    if (p.i[0] == 1) {
        float sum = 0.0f;
        for (long i = tpitg; i < n; i += ntg) {
            sum += gk_mtl_get(a, p.src0, i, i1, i2, i3);
        }
        const float mean = gk_mtl_group_sum(sum, scratch, tiisg, sgitg, nsg) / (float) n;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float var = 0.0f;
        for (long i = tpitg; i < n; i += ntg) {
            const float c = gk_mtl_get(a, p.src0, i, i1, i2, i3) - mean;
            var += c * c;
        }
        const float scale = rsqrt(gk_mtl_group_sum(var, scratch, tiisg, sgitg, nsg) / (float) n + eps);

        for (long i = tpitg; i < n; i += ntg) {
            gk_mtl_set(d, p.dst, i, i1, i2, i3,
                       (gk_mtl_get(a, p.src0, i, i1, i2, i3) - mean) * scale);
        }
        return;
    }

    float sumsq = 0.0f;
    for (long i = tpitg; i < n; i += ntg) {
        const float v = gk_mtl_get(a, p.src0, i, i1, i2, i3);
        sumsq += v * v;
    }
    const float total = gk_mtl_group_sum(sumsq, scratch, tiisg, sgitg, nsg);

    const float scale = p.i[0] == 0 ? rsqrt(total / (float) n + eps)
                                    : rsqrt(max(total, eps));

    for (long i = tpitg; i < n; i += ntg) {
        gk_mtl_set(d, p.dst, i, i1, i2, i3, gk_mtl_get(a, p.src0, i, i1, i2, i3) * scale);
    }
}

// i[0] = n_groups, f[0] = eps
kernel void gk_mtl_group_norm(device const uchar * a [[buffer(0)]],
                              device uchar *       d [[buffer(2)]],
                              constant gk_mtl_params & p [[buffer(3)]],
                              threadgroup float * scratch [[threadgroup(0)]],
                              uint3 tgpig  [[threadgroup_position_in_grid]],
                              uint3 tpitg3 [[thread_position_in_threadgroup]],
                              uint3 ntg3   [[threads_per_threadgroup]],
                              uint  tiisg [[thread_index_in_simdgroup]],
                              uint  sgitg [[simdgroup_index_in_threadgroup]],
                              uint  nsg   [[simdgroups_per_threadgroup]]) {
    const uint tpitg = tpitg3.x;
    const uint ntg   = ntg3.x;

    const int n_groups = p.i[0];

    const long unit = (long) tgpig.x;
    const long i3   = unit / n_groups;
    const long g    = unit % n_groups;

    const long per_group = p.src0.ne[2] / n_groups;
    const long c0 = g * per_group;

    const long plane = p.src0.ne[0] * p.src0.ne[1];
    const long count = plane * per_group;

    float sum = 0.0f;
    for (long t = tpitg; t < count; t += ntg) {
        const long i2 = c0 + t / plane;
        const long r  = t % plane;
        sum += gk_mtl_get(a, p.src0, r % p.src0.ne[0], r / p.src0.ne[0], i2, i3);
    }
    const float mean = gk_mtl_group_sum(sum, scratch, tiisg, sgitg, nsg) / (float) count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float var = 0.0f;
    for (long t = tpitg; t < count; t += ntg) {
        const long i2 = c0 + t / plane;
        const long r  = t % plane;
        const float c = gk_mtl_get(a, p.src0, r % p.src0.ne[0], r / p.src0.ne[0], i2, i3) - mean;
        var += c * c;
    }
    const float scale = rsqrt(gk_mtl_group_sum(var, scratch, tiisg, sgitg, nsg) / (float) count + p.f[0]);

    for (long t = tpitg; t < count; t += ntg) {
        const long i2 = c0 + t / plane;
        const long r  = t % plane;
        const long i0 = r % p.src0.ne[0];
        const long i1 = r / p.src0.ne[0];
        gk_mtl_set(d, p.dst, i0, i1, i2, i3, (gk_mtl_get(a, p.src0, i0, i1, i2, i3) - mean) * scale);
    }
}

// f[0] = scale, f[1] = max_bias, i[0] = n_head_log2,
// flags bit 0 = has mask, bit 1 = has sinks
kernel void gk_mtl_soft_max(device const uchar * a     [[buffer(0)]],
                            device const uchar * mask  [[buffer(1)]],
                            device uchar *       d     [[buffer(2)]],
                            constant gk_mtl_params & p [[buffer(3)]],
                            device const float * sinks [[buffer(4)]],
                            threadgroup float * scratch [[threadgroup(0)]],
                            uint3 tgpig  [[threadgroup_position_in_grid]],
                            uint3 tpitg3 [[thread_position_in_threadgroup]],
                            uint3 ntg3   [[threads_per_threadgroup]],
                            uint  tiisg [[thread_index_in_simdgroup]],
                            uint  sgitg [[simdgroup_index_in_threadgroup]],
                            uint  nsg   [[simdgroups_per_threadgroup]]) {
    const uint tpitg = tpitg3.x;
    const uint ntg   = ntg3.x;

    const long ir = (long) tgpig.x;
    const long i1 = ir % p.dst.ne[1];
    const long i2 = (ir / p.dst.ne[1]) % p.dst.ne[2];
    const long i3 = ir / (p.dst.ne[1] * p.dst.ne[2]);

    const long n = p.dst.ne[0];

    const bool has_mask  = (p.flags & 1) != 0;
    const bool has_sinks = (p.flags & 2) != 0;

    const float scale    = p.f[0];
    const float max_bias = p.f[1];

    // the ALiBi slope for this head
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const float n_head_log2 = (float) p.i[0];
        const float m0 = pow(2.0f, -max_bias / n_head_log2);
        const float m1 = pow(2.0f, -(max_bias / 2.0f) / n_head_log2);
        slope = (float) i2 < n_head_log2
            ? pow(m0, (float) (i2 + 1))
            : pow(m1, (float) (2 * ((float) i2 - n_head_log2) + 1));
    }

    float local_max = -INFINITY;
    for (long i = tpitg; i < n; i += ntg) {
        float v = gk_mtl_get(a, p.src0, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_mtl_get(mask, p.src1, i, i1, i2 % p.src1.ne[2], i3 % p.src1.ne[3]);
        }
        local_max = max(local_max, v);
    }

    float row_max = gk_mtl_group_max(local_max, scratch, tiisg, sgitg, nsg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sink = has_sinks ? sinks[i2] : -INFINITY;
    if (has_sinks) {
        row_max = max(row_max, sink);
    }

    // a fully masked row is all -inf; define it as a uniform zero row rather
    // than the 0/0 that exponentiating would give
    if (!isfinite(row_max)) {
        for (long i = tpitg; i < n; i += ntg) {
            gk_mtl_set(d, p.dst, i, i1, i2, i3, 0.0f);
        }
        return;
    }

    float local_sum = 0.0f;
    for (long i = tpitg; i < n; i += ntg) {
        float v = gk_mtl_get(a, p.src0, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_mtl_get(mask, p.src1, i, i1, i2 % p.src1.ne[2], i3 % p.src1.ne[3]);
        }
        local_sum += exp(v - row_max);
    }

    float sum = gk_mtl_group_sum(local_sum, scratch, tiisg, sgitg, nsg);
    if (has_sinks) {
        sum += exp(sink - row_max);
    }

    const float inv = 1.0f / sum;

    // The exponentials are recomputed rather than stored and read back. The
    // destination may be f16, and rounding through it before the normalisation
    // would put the answer a visible distance from the CPU's, which normalises
    // in f32 and converts once at the end.
    for (long i = tpitg; i < n; i += ntg) {
        float v = gk_mtl_get(a, p.src0, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_mtl_get(mask, p.src1, i, i1, i2 % p.src1.ne[2], i3 % p.src1.ne[3]);
        }
        gk_mtl_set(d, p.dst, i, i1, i2, i3, exp(v - row_max) * inv);
    }
}

// i[0] = 1 for mean, 0 for sum_rows
kernel void gk_mtl_sum_rows(device const uchar * a [[buffer(0)]],
                            device uchar *       d [[buffer(2)]],
                            constant gk_mtl_params & p [[buffer(3)]],
                            threadgroup float * scratch [[threadgroup(0)]],
                            uint3 tgpig  [[threadgroup_position_in_grid]],
                            uint3 tpitg3 [[thread_position_in_threadgroup]],
                            uint3 ntg3   [[threads_per_threadgroup]],
                            uint  tiisg [[thread_index_in_simdgroup]],
                            uint  sgitg [[simdgroup_index_in_threadgroup]],
                            uint  nsg   [[simdgroups_per_threadgroup]]) {
    const uint tpitg = tpitg3.x;
    const uint ntg   = ntg3.x;

    const long ir = (long) tgpig.x;
    const long i1 = ir % p.dst.ne[1];
    const long i2 = (ir / p.dst.ne[1]) % p.dst.ne[2];
    const long i3 = ir / (p.dst.ne[1] * p.dst.ne[2]);

    float local = 0.0f;
    for (long i = tpitg; i < p.src0.ne[0]; i += ntg) {
        local += gk_mtl_get(a, p.src0, i, i1, i2, i3);
    }

    const float total = gk_mtl_group_sum(local, scratch, tiisg, sgitg, nsg);

    if (tpitg == 0) {
        gk_mtl_set(d, p.dst, 0, i1, i2, i3,
                   p.i[0] != 0 ? total / (float) p.src0.ne[0] : total);
    }
}

// --------------------------------------------------------------------------
// rotary position embedding
//
// One thread per rotated pair; the angle is recomputed rather than cached,
// which keeps every pair independent and costs two transcendentals.
//
// i[0] = n_dims, i[1] = mode, i[2..5] = sections,
// f[0] = freq_scale, f[1] = ext_factor, f[2] = attn_factor,
// f[3] = theta_scale, f[4..5] = corr_dims, flags bit 0 = has freq_factors
// --------------------------------------------------------------------------

kernel void gk_mtl_rope(device const uchar * a   [[buffer(0)]],
                        device const int *   pos [[buffer(1)]],
                        device uchar *       d   [[buffer(2)]],
                        constant gk_mtl_params & p [[buffer(3)]],
                        device const float * freq_factors [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const int  n_dims = p.i[0];
    const int  mode   = p.i[1];
    const bool neox   = (mode & 2) != 0;
    const bool mrope  = (mode & 8) != 0;
    const bool vision = mode == 24;
    const bool imrope = mode == 40;

    const long n_rot = vision ? p.dst.ne[0] : (long) n_dims;
    const long pairs_per_row = n_rot / 2;

    const long ir = k / pairs_per_row;
    const long ip = k % pairs_per_row;

    const long i1 = ir % p.dst.ne[1];
    const long i2 = (ir / p.dst.ne[1]) % p.dst.ne[2];
    const long i3 = ir / (p.dst.ne[1] * p.dst.ne[2]);

    const long i0 = ip * 2;

    float theta_base;
    if (!mrope) {
        theta_base = (float) pos[i2] * pow(p.f[3], (float) ip);
    } else {
        const long n_pos = p.dst.ne[2];
        const int sect_dims = p.i[2] + p.i[3] + p.i[4] + p.i[5];
        const int sec_w = p.i[3] + p.i[2];
        const int sec_e = p.i[4] + sec_w;

        const int sector = (int) (ip % sect_dims);

        int axis;
        if (imrope) {
            if (sector % 3 == 1 && sector < 3 * p.i[3])      axis = 1;
            else if (sector % 3 == 2 && sector < 3 * p.i[4]) axis = 2;
            else if (sector % 3 == 0 && sector < 3 * p.i[2]) axis = 0;
            else                                             axis = 3;
        } else {
            if (sector < p.i[2])              axis = 0;
            else if (sector < sec_w)          axis = 1;
            else if (sector < sec_w + p.i[4]) axis = 2;
            else                              axis = 3;
        }

        int step = (int) ip;
        if (vision) {
            // the vision rope restarts each axis's angle at its section
            const int base = axis == 0 ? 0 : axis == 1 ? p.i[2] : axis == 2 ? sec_w : sec_e;
            step = (int) ip - base;
        }

        theta_base = (float) pos[i2 + n_pos * axis] * pow(p.f[3], (float) step);
    }

    const float ff = (p.flags & 1) != 0 ? freq_factors[ip] : 1.0f;
    theta_base /= ff;

    float theta  = theta_base * p.f[0];
    float mscale = p.f[2];

    if (p.f[1] != 0.0f) {
        const float y = ((float) (i0 / 2) - p.f[4]) / max(0.001f, p.f[5] - p.f[4]);
        const float ramp = (1.0f - min(1.0f, max(0.0f, y))) * p.f[1];

        theta = theta * (1.0f - ramp) + theta_base * ramp;
        mscale *= 1.0f + 0.1f * log(1.0f / p.f[0]);
    }

    const float cos_t = cos(theta) * mscale;
    const float sin_t = sin(theta) * mscale;

    const long offset = vision ? (long) n_dims : ((neox || mrope) ? n_dims / 2 : 1);
    const long ic     = (neox || mrope) ? ip : i0;

    const float x0 = gk_mtl_get(a, p.src0, ic,          i1, i2, i3);
    const float x1 = gk_mtl_get(a, p.src0, ic + offset, i1, i2, i3);

    gk_mtl_set(d, p.dst, ic,          i1, i2, i3, x0 * cos_t - x1 * sin_t);
    gk_mtl_set(d, p.dst, ic + offset, i1, i2, i3, x0 * sin_t + x1 * cos_t);
}

// channels at or past n_dims pass through unrotated
kernel void gk_mtl_rope_passthrough(device const uchar * a [[buffer(0)]],
                                    device uchar *       d [[buffer(2)]],
                                    constant gk_mtl_params & p [[buffer(3)]],
                                    uint tid [[thread_position_in_grid]]) {
    const long k = (long) tid;
    if (k >= p.n) {
        return;
    }

    const long width = p.dst.ne[0] - p.i[0];
    const long ir = k / width;
    const long i0 = p.i[0] + k % width;

    const long i1 = ir % p.dst.ne[1];
    const long i2 = (ir / p.dst.ne[1]) % p.dst.ne[2];
    const long i3 = ir / (p.dst.ne[1] * p.dst.ne[2]);

    gk_mtl_set(d, p.dst, i0, i1, i2, i3, gk_mtl_get(a, p.src0, i0, i1, i2, i3));
}

// --------------------------------------------------------------------------
// matmul
//
// One threadgroup per output element, or per group of NC of them along the
// activation columns: the weight row is read once and used NC times, which is
// the reuse that makes a quantized matmul memory-bound rather than
// decode-bound. i[0] carries NC.
//
// The k-loops read through gk_mtl_row_elem rather than gk_mtl_get: the row
// address is computed once outside the loop, and the within-row math is
// 32-bit shifts and masks. The GPU has no integer divide - a 64-bit divide
// by a runtime block size costs more than the multiply it feeds, and
// gk_mtl_get would pay it on every element.
// --------------------------------------------------------------------------

// Per-row read state, resolved once per row outside the k-loop.
struct gk_mtl_row_view {
    device const uchar * row;
    int type;
    int nb0;    // element stride for the float types
    int shift;  // log2(block size) for the quantized types
    int tsz;    // block byte size for the quantized types
};

static gk_mtl_row_view gk_mtl_row_view_make(device const uchar * base,
                                            constant gk_mtl_tview & t,
                                            long i1, long i2, long i3) {
    gk_mtl_row_view v;
    v.row   = base + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
    v.type  = t.type;
    v.nb0   = (int) t.nb[0];
    v.shift = ctz((uint) gk_mtl_blck_size(t.type));
    v.tsz   = gk_mtl_type_size(t.type);
    return v;
}

static inline float gk_mtl_row_elem(const thread gk_mtl_row_view & v, int kk) {
    switch (v.type) {
        case GKT_F32:  return *(device const float *) (v.row + kk * v.nb0);
        case GKT_F16:  return (float) *(device const half *) (v.row + kk * v.nb0);
        case GKT_BF16: return gk_mtl_bf2f(*(device const ushort *) (v.row + kk * v.nb0));
        default:
            return gk_mtl_block_elem(v.row + (kk >> v.shift) * v.tsz, v.type,
                                     kk & ((1 << v.shift) - 1));
    }
}

#define GK_MTL_MM_NC 4

kernel void gk_mtl_mul_mat(device const uchar * a [[buffer(0)]],
                           device const uchar * b [[buffer(1)]],
                           device uchar *       d [[buffer(2)]],
                           constant gk_mtl_params & p [[buffer(3)]],
                           threadgroup float * scratch [[threadgroup(0)]],
                           uint3 tgpig  [[threadgroup_position_in_grid]],
                           uint3 tpitg3 [[thread_position_in_threadgroup]],
                           uint3 ntg3   [[threads_per_threadgroup]],
                           uint  tiisg [[thread_index_in_simdgroup]],
                           uint  sgitg [[simdgroup_index_in_threadgroup]],
                           uint  nsg   [[simdgroups_per_threadgroup]]) {
    const uint tpitg = tpitg3.x;
    const uint ntg   = ntg3.x;

    const long i0  = (long) tgpig.x;
    const long c0  = (long) tgpig.y * p.i[0];
    const long i23 = (long) tgpig.z;

    const long i2 = i23 % p.dst.ne[2];
    const long i3 = i23 / p.dst.ne[2];

    const long r2 = p.src1.ne[2] / p.src0.ne[2];
    const long r3 = p.src1.ne[3] / p.src0.ne[3];

    const long a2 = i2 / r2;
    const long a3 = i3 / r3;

    const int k_len = (int) p.src0.ne[0];

    const gk_mtl_row_view arow = gk_mtl_row_view_make(a, p.src0, i0, a2, a3);

    // A column past ne[1] is clamped to a valid row: its reads are in-bounds
    // garbage and its accumulator is never written back.
    gk_mtl_row_view brow[GK_MTL_MM_NC];
    for (int j = 0; j < p.i[0]; ++j) {
        const long col = c0 + j < p.dst.ne[1] ? c0 + j : p.dst.ne[1] - 1;
        brow[j] = gk_mtl_row_view_make(b, p.src1, col, i2, i3);
    }

    float acc[GK_MTL_MM_NC] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int kk = (int) tpitg; kk < k_len; kk += (int) ntg) {
        const float av = gk_mtl_row_elem(arow, kk);

        for (int j = 0; j < p.i[0]; ++j) {
            acc[j] += av * gk_mtl_row_elem(brow[j], kk);
        }
    }

    for (int j = 0; j < p.i[0]; ++j) {
        const float total = gk_mtl_group_sum(acc[j], scratch, tiisg, sgitg, nsg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const long col = c0 + j;
        if (tpitg == 0 && col < p.dst.ne[1]) {
            gk_mtl_set(d, p.dst, i0, col, i2, i3, total);
        }
    }
}

// Tiled matmul for wide destinations. One threadgroup owns a 32x32 output
// tile; the operands stream through threadgroup memory in K-slabs of 16, and
// each thread holds a 2x2 block of accumulators. The staging is the point:
// the row-per-threadgroup kernel below re-reads the activation columns once
// per output row, which makes a diffusion step bandwidth-bound by a factor of
// the tile width. Quantized operands are decoded once, on the way into the
// tile.
#define GK_MTL_MM_TILE 32
#define GK_MTL_MM_SLAB 16

kernel void gk_mtl_mul_mat_tiled(device const uchar * a [[buffer(0)]],
                                 device const uchar * b [[buffer(1)]],
                                 device uchar *       d [[buffer(2)]],
                                 constant gk_mtl_params & p [[buffer(3)]],
                                 threadgroup float * tile [[threadgroup(0)]],
                                 uint3 tgpig  [[threadgroup_position_in_grid]],
                                 uint3 tpitg3 [[thread_position_in_threadgroup]]) {
    const int t  = (int) tpitg3.x;   // 256 threads, laid out 16x16
    const int tx = t & 15;
    const int ty = t >> 4;

    const long row0 = (long) tgpig.x * GK_MTL_MM_TILE;
    const long col0 = (long) tgpig.y * GK_MTL_MM_TILE;
    const long i23  = (long) tgpig.z;

    const long i2 = i23 % p.dst.ne[2];
    const long i3 = i23 / p.dst.ne[2];

    const long r2 = p.src1.ne[2] / p.src0.ne[2];
    const long r3 = p.src1.ne[3] / p.src0.ne[3];

    const long a2 = i2 / r2;
    const long a3 = i3 / r3;

    const long M = p.dst.ne[0];
    const long N = p.dst.ne[1];
    const int  K = (int) p.src0.ne[0];

    threadgroup float * ta = tile;                                    // [SLAB][TILE]
    threadgroup float * tb = tile + GK_MTL_MM_TILE * GK_MTL_MM_SLAB;  // [SLAB][TILE]

    // Each thread stages the same tile row / column every slab, at two k
    // positions; the row views can therefore be resolved once. An
    // out-of-range row is staged as zeros and its accumulators never stored.
    const int  rs = t & (GK_MTL_MM_TILE - 1);
    const int  ks = t >> 5;   // this thread fills k slots ks and ks + 8

    const long ar = row0 + rs;
    const long bc = col0 + rs;
    const bool a_ok = ar < M;
    const bool b_ok = bc < N;

    const gk_mtl_row_view av = gk_mtl_row_view_make(a, p.src0, a_ok ? ar : 0, a2, a3);
    const gk_mtl_row_view bv = gk_mtl_row_view_make(b, p.src1, b_ok ? bc : 0, i2, i3);

    const int r0 = ty * 2;   // this thread's 2x2 block within the tile
    const int c0 = tx * 2;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    for (int k0 = 0; k0 < K; k0 += GK_MTL_MM_SLAB) {
        const int k1 = k0 + ks;
        const int k2 = k0 + ks + 8;

        ta[ ks      * GK_MTL_MM_TILE + rs] = a_ok && k1 < K ? gk_mtl_row_elem(av, k1) : 0.0f;
        ta[(ks + 8) * GK_MTL_MM_TILE + rs] = a_ok && k2 < K ? gk_mtl_row_elem(av, k2) : 0.0f;
        tb[ ks      * GK_MTL_MM_TILE + rs] = b_ok && k1 < K ? gk_mtl_row_elem(bv, k1) : 0.0f;
        tb[(ks + 8) * GK_MTL_MM_TILE + rs] = b_ok && k2 < K ? gk_mtl_row_elem(bv, k2) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < GK_MTL_MM_SLAB; ++kk) {
            const float a0 = ta[kk * GK_MTL_MM_TILE + r0];
            const float a1 = ta[kk * GK_MTL_MM_TILE + r0 + 1];
            const float b0 = tb[kk * GK_MTL_MM_TILE + c0];
            const float b1 = tb[kk * GK_MTL_MM_TILE + c0 + 1];

            acc00 += a0 * b0; acc01 += a0 * b1;
            acc10 += a1 * b0; acc11 += a1 * b1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const long orow0 = row0 + r0;
    const long ocol0 = col0 + c0;

    if (orow0 < M && ocol0 < N) {
        gk_mtl_set(d, p.dst, orow0, ocol0, i2, i3, acc00);
    }
    if (orow0 < M && ocol0 + 1 < N) {
        gk_mtl_set(d, p.dst, orow0, ocol0 + 1, i2, i3, acc01);
    }
    if (orow0 + 1 < M && ocol0 < N) {
        gk_mtl_set(d, p.dst, orow0 + 1, ocol0, i2, i3, acc10);
    }
    if (orow0 + 1 < M && ocol0 + 1 < N) {
        gk_mtl_set(d, p.dst, orow0 + 1, ocol0 + 1, i2, i3, acc11);
    }
}

kernel void gk_mtl_mul_mat_id(device const uchar * as  [[buffer(0)]],
                              device const uchar * b   [[buffer(1)]],
                              device uchar *       d   [[buffer(2)]],
                              constant gk_mtl_params & p [[buffer(3)]],
                              device const uchar * ids [[buffer(4)]],
                              threadgroup float * scratch [[threadgroup(0)]],
                              uint3 tgpig  [[threadgroup_position_in_grid]],
                              uint3 tpitg3 [[thread_position_in_threadgroup]],
                              uint3 ntg3   [[threads_per_threadgroup]],
                              uint  tiisg [[thread_index_in_simdgroup]],
                              uint  sgitg [[simdgroup_index_in_threadgroup]],
                              uint  nsg   [[simdgroups_per_threadgroup]]) {
    const uint tpitg = tpitg3.x;
    const uint ntg   = ntg3.x;

    const long i0 = (long) tgpig.x;
    const long is = (long) tgpig.y;
    const long it = (long) tgpig.z;

    const int expert = *(device const int *) (ids + it * p.src2.nb[1] + is * p.src2.nb[0]);

    const gk_mtl_row_view arow = gk_mtl_row_view_make(as, p.src0, i0, (long) expert, 0);
    const gk_mtl_row_view brow = gk_mtl_row_view_make(b,  p.src1, 0, it, 0);

    const int k_len = (int) p.src0.ne[0];

    float acc = 0.0f;
    for (int kk = (int) tpitg; kk < k_len; kk += (int) ntg) {
        acc += gk_mtl_row_elem(arow, kk) * gk_mtl_row_elem(brow, kk);
    }

    const float total = gk_mtl_group_sum(acc, scratch, tiisg, sgitg, nsg);

    if (tpitg == 0) {
        gk_mtl_set(d, p.dst, i0, is, it, 0, total);
    }
}
