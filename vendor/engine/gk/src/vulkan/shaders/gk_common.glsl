// What every gk compute shader shares: the parameter block, how a tensor is
// addressed, and how the GGUF block formats decode.
//
// Two constraints shape all of it, and both come from GLSL rather than from
// the ops:
//
//   * A storage buffer cannot be passed to a function. So the accessors are
//     macros that generate a set of functions bound to one buffer name, and a
//     shader instantiates the ones it needs. It reads worse than a parameter
//     would; there is no other way to write it once.
//
//   * There are no byte pointers. Every buffer is declared as an array of
//     32-bit words and narrower reads are shifts out of a word. That is why
//     the byte and half helpers exist, and why the destination is restricted
//     to f32: writing an f16 would be a read-modify-write of a word two
//     threads share, and the race would be real.
//
// Offsets are 32-bit, which caps a single Vulkan buffer at 4 GiB. That is at
// or below the maximum allocation most drivers report anyway, and the backend
// refuses larger allocations rather than wrapping silently.

#extension GL_EXT_control_flow_attributes : enable

// --------------------------------------------------------------------------
// the parameter block
//
// Laid out in vec4-sized pieces because std140 pads array elements to 16 bytes
// and a float[8] would otherwise cost 128.
// --------------------------------------------------------------------------

struct TView {
    uvec4 ne;
    uvec4 nb;
    uint  type;
    // Where this tensor starts inside the buffer bound for it. Bindings cover
    // a whole buffer - a tensor is a range within one - so the offset travels
    // here and every address below starts from it.
    uint  base;
    uint  pad1;
    uint  pad2;
};

layout(binding = 0, std140) uniform Params {
    TView src0;
    TView src1;
    TView src2;
    TView dst;

    vec4  f01; // f[0..3]
    vec4  f23; // f[4..7]
    ivec4 i0;  // i[0..3]
    ivec4 i1;  // i[4..7]
    ivec4 i2;  // i[8..11]
    ivec4 i3;  // i[12..15]

    uint  n;      // destination elements this dispatch covers
    uint  flags;
    uint  pad_a;
    uint  pad_b;
} P;

#define PF(k) ((k) < 4 ? P.f01[(k)] : P.f23[(k) - 4])
#define PI(k) ((k) < 4 ? P.i0[(k)] : (k) < 8 ? P.i1[(k) - 4] : (k) < 12 ? P.i2[(k) - 8] : P.i3[(k) - 12])

// --------------------------------------------------------------------------
// type codes, mirrored from gk.h
// --------------------------------------------------------------------------

#define GKT_F32     0u
#define GKT_F16     1u
#define GKT_Q4_0    2u
#define GKT_Q4_1    3u
#define GKT_Q5_0    6u
#define GKT_Q5_1    7u
#define GKT_Q8_0    8u
#define GKT_Q2_K   10u
#define GKT_Q3_K   11u
#define GKT_Q4_K   12u
#define GKT_Q5_K   13u
#define GKT_Q6_K   14u
#define GKT_IQ4_NL 20u
#define GKT_IQ4_XS 23u
#define GKT_I32    26u
#define GKT_BF16   30u
#define GKT_MXFP4  39u

#define GK_QK 256u

uint gk_blck_size(uint type) {
    switch (type) {
        case GKT_Q4_0: case GKT_Q4_1: case GKT_Q5_0: case GKT_Q5_1:
        case GKT_Q8_0: case GKT_IQ4_NL: case GKT_MXFP4:
            return 32u;
        case GKT_Q2_K: case GKT_Q3_K: case GKT_Q4_K: case GKT_Q5_K:
        case GKT_Q6_K: case GKT_IQ4_XS:
            return GK_QK;
        default:
            return 1u;
    }
}

uint gk_type_size(uint type) {
    switch (type) {
        case GKT_F32:    return 4u;
        case GKT_F16:    return 2u;
        case GKT_BF16:   return 2u;
        case GKT_I32:    return 4u;
        case GKT_Q4_0:   return 18u;
        case GKT_Q4_1:   return 20u;
        case GKT_Q5_0:   return 22u;
        case GKT_Q5_1:   return 24u;
        case GKT_Q8_0:   return 34u;
        case GKT_MXFP4:  return 17u;
        case GKT_Q2_K:   return 4u + GK_QK / 16u + GK_QK / 4u;
        case GKT_Q3_K:   return 2u + GK_QK / 4u + GK_QK / 8u + 12u;
        case GKT_Q4_K:   return 4u + 12u + GK_QK / 2u;
        case GKT_Q5_K:   return 4u + 12u + GK_QK / 2u + GK_QK / 8u;
        case GKT_Q6_K:   return 2u + GK_QK / 16u + 3u * GK_QK / 4u;
        case GKT_IQ4_NL: return 18u;
        case GKT_IQ4_XS: return 2u + 2u + GK_QK / 64u + GK_QK / 2u;
        default:         return 0u;
    }
}

// The IQ4 and E2M1 codebooks. Written as arrays rather than a switch because
// every entry is a constant the compiler can fold into the index.
const int gk_iq4_values[16] = int[16](
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

const int gk_e2m1_values[16] = int[16](
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);

float gk_e8m0_half(uint e) {
    // the MXFP4 codebook holds doubled values, so its scale is half the
    // shared exponent
    if (e >= 2u) {
        return uintBitsToFloat((e - 1u) << 23);
    }
    return uintBitsToFloat(0x00200000u << e);
}

// The byte offset of row (i1,i2,i3) within the bound buffer.
uint gk_row_off(TView t, uint i1, uint i2, uint i3) {
    return t.base + i1 * t.nb.y + i2 * t.nb.z + i3 * t.nb.w;
}

// --------------------------------------------------------------------------
// readers
//
// GK_FLOAT_READER covers the types an elementwise op can see. GK_QUANT_READER
// adds the block formats, and is instantiated only where a quantized operand
// can appear - the weight side of a matmul, a gather, a conversion.
// --------------------------------------------------------------------------

#define GK_FLOAT_READER(P_, BUF)                                                        \
    uint P_##_byte(uint off) {                                                          \
        return (BUF[off >> 2] >> (8u * (off & 3u))) & 0xffu;                             \
    }                                                                                   \
    float P_##_f32(uint off)  { return uintBitsToFloat(BUF[off >> 2]); }                \
    float P_##_f16(uint off)  {                                                         \
        return unpackHalf2x16(BUF[off >> 2])[(off >> 1) & 1u];                           \
    }                                                                                   \
    float P_##_bf16(uint off) {                                                         \
        return uintBitsToFloat((BUF[off >> 2] >> (16u * ((off >> 1) & 1u))) << 16);      \
    }                                                                                   \
    float P_##_plain(TView t, uint off) {                                               \
        switch (t.type) {                                                               \
            case GKT_F32:  return P_##_f32(off);                                        \
            case GKT_F16:  return P_##_f16(off);                                        \
            case GKT_BF16: return P_##_bf16(off);                                       \
            case GKT_I32:  return float(int(BUF[off >> 2]));                            \
            default:       return 0.0;                                                  \
        }                                                                               \
    }                                                                                   \
    float P_##_get(TView t, uint i0, uint i1, uint i2, uint i3) {                       \
        return P_##_plain(t, gk_row_off(t, i1, i2, i3) + i0 * t.nb.x);                   \
    }

// The 6-bit scale and min of group g out of q4_K/q5_K's twelve packed bytes.
#define GK_QUANT_READER(P_, BUF)                                                        \
    GK_FLOAT_READER(P_, BUF)                                                            \
    float P_##_h2f(uint off) {                                                          \
        const uint bits = P_##_byte(off) | (P_##_byte(off + 1u) << 8u);                  \
        return unpackHalf2x16(bits)[0];                                                  \
    }                                                                                   \
    void P_##_scale_min_6(uint s, uint g, out int scale, out int mn) {                   \
        if (g < 4u) {                                                                    \
            scale = int(P_##_byte(s + g) & 63u);                                         \
            mn    = int(P_##_byte(s + g + 4u) & 63u);                                    \
        } else {                                                                         \
            scale = int((P_##_byte(s + g + 4u) & 15u) | ((P_##_byte(s + g - 4u) >> 6u) << 4u)); \
            mn    = int((P_##_byte(s + g + 4u) >> 4u)  | ((P_##_byte(s + g)      >> 6u) << 4u)); \
        }                                                                                \
    }                                                                                    \
    float P_##_block(uint b, uint type, uint j) {                                        \
        switch (type) {                                                                  \
            case GKT_Q4_0: {                                                             \
                const float d = P_##_h2f(b);                                             \
                const uint q = j < 16u ? (P_##_byte(b + 2u + j) & 15u)                    \
                                       : (P_##_byte(b + 2u + j - 16u) >> 4u);             \
                return d * (float(q) - 8.0);                                             \
            }                                                                            \
            case GKT_Q4_1: {                                                             \
                const float d = P_##_h2f(b);                                             \
                const float m = P_##_h2f(b + 2u);                                        \
                const uint q = j < 16u ? (P_##_byte(b + 4u + j) & 15u)                    \
                                       : (P_##_byte(b + 4u + j - 16u) >> 4u);             \
                return d * float(q) + m;                                                 \
            }                                                                            \
            case GKT_Q5_0: {                                                             \
                const float d = P_##_h2f(b);                                             \
                const uint qh = P_##_byte(b + 2u) | (P_##_byte(b + 3u) << 8u)             \
                              | (P_##_byte(b + 4u) << 16u) | (P_##_byte(b + 5u) << 24u);  \
                const uint nib = j < 16u ? (P_##_byte(b + 6u + j) & 15u)                  \
                                         : (P_##_byte(b + 6u + j - 16u) >> 4u);           \
                const uint q = nib | (((qh >> j) & 1u) << 4u);                            \
                return d * (float(q) - 16.0);                                            \
            }                                                                            \
            case GKT_Q5_1: {                                                             \
                const float d = P_##_h2f(b);                                             \
                const float m = P_##_h2f(b + 2u);                                        \
                const uint qh = P_##_byte(b + 4u) | (P_##_byte(b + 5u) << 8u)             \
                              | (P_##_byte(b + 6u) << 16u) | (P_##_byte(b + 7u) << 24u);  \
                const uint nib = j < 16u ? (P_##_byte(b + 8u + j) & 15u)                  \
                                         : (P_##_byte(b + 8u + j - 16u) >> 4u);           \
                const uint q = nib | (((qh >> j) & 1u) << 4u);                            \
                return d * float(q) + m;                                                 \
            }                                                                            \
            case GKT_Q8_0: {                                                             \
                const float d = P_##_h2f(b);                                             \
                const int q = int(P_##_byte(b + 2u + j) << 24u) >> 24;                    \
                return d * float(q);                                                     \
            }                                                                            \
            case GKT_MXFP4: {                                                            \
                const float d = gk_e8m0_half(P_##_byte(b));                               \
                const uint code = j < 16u ? (P_##_byte(b + 1u + j) & 15u)                 \
                                          : (P_##_byte(b + 1u + j - 16u) >> 4u);          \
                return d * float(gk_e2m1_values[code]);                                   \
            }                                                                            \
            case GKT_Q2_K: {                                                             \
                const uint qs = b + GK_QK / 16u;                                          \
                const float d    = P_##_h2f(b + GK_QK / 16u + GK_QK / 4u);                \
                const float dmin = P_##_h2f(b + GK_QK / 16u + GK_QK / 4u + 2u);           \
                const uint half_i = j / 128u;                                             \
                const uint r      = j % 128u;                                             \
                const uint shift  = 2u * (r / 32u);                                       \
                const uint part   = (r % 32u) / 16u;                                      \
                const uint l      = r % 16u;                                              \
                const uint g  = half_i * 8u + (shift / 2u) * 2u + part;                   \
                const uint sc = P_##_byte(b + g);                                         \
                const uint q  = (P_##_byte(qs + half_i * 32u + part * 16u + l) >> shift) & 3u; \
                return d * float(sc & 15u) * float(q) - dmin * float(sc >> 4u);           \
            }                                                                            \
            case GKT_Q3_K: {                                                             \
                const uint hmask  = b;                                                    \
                const uint qs     = b + GK_QK / 8u;                                       \
                const uint scales = b + GK_QK / 8u + GK_QK / 4u;                          \
                const float d = P_##_h2f(scales + 12u);                                   \
                const uint half_i = j / 128u;                                             \
                const uint r      = j % 128u;                                             \
                const uint shift  = 2u * (r / 32u);                                       \
                const uint bit    = half_i * 4u + (r / 32u);                              \
                const uint part   = (r % 32u) / 16u;                                      \
                const uint l      = r % 16u;                                              \
                const uint idx    = part * 16u + l;                                       \
                const uint g = half_i * 8u + (shift / 2u) * 2u + part;                    \
                const uint low  = g < 8u ? (P_##_byte(scales + g) & 15u)                  \
                                         : (P_##_byte(scales + g - 8u) >> 4u);            \
                const uint high = (P_##_byte(scales + 8u + (g % 4u)) >> (2u * (g / 4u))) & 3u; \
                const int  sc   = int(low | (high << 4u)) - 32;                           \
                const int  lo   = int((P_##_byte(qs + half_i * 32u + idx) >> shift) & 3u); \
                const int  v    = lo - (((P_##_byte(hmask + idx) & (1u << bit)) != 0u) ? 0 : 4); \
                return d * float(sc) * float(v);                                          \
            }                                                                            \
            case GKT_Q4_K: {                                                             \
                const float d    = P_##_h2f(b);                                          \
                const float dmin = P_##_h2f(b + 2u);                                     \
                const uint scales = b + 4u;                                               \
                const uint qs     = b + 16u;                                              \
                const uint g = j / 32u;                                                   \
                const uint l = j % 32u;                                                   \
                int sc, mn;                                                               \
                P_##_scale_min_6(scales, g, sc, mn);                                      \
                const uint q = (P_##_byte(qs + (g / 2u) * 32u + l) >> ((g % 2u) * 4u)) & 15u; \
                return d * float(sc) * float(q) - dmin * float(mn);                       \
            }                                                                            \
            case GKT_Q5_K: {                                                             \
                const float d    = P_##_h2f(b);                                          \
                const float dmin = P_##_h2f(b + 2u);                                     \
                const uint scales = b + 4u;                                               \
                const uint qh     = b + 16u;                                              \
                const uint qs     = b + 16u + GK_QK / 8u;                                 \
                const uint g = j / 32u;                                                   \
                const uint l = j % 32u;                                                   \
                int sc, mn;                                                               \
                P_##_scale_min_6(scales, g, sc, mn);                                      \
                const uint lo = (P_##_byte(qs + (g / 2u) * 32u + l) >> ((g % 2u) * 4u)) & 15u; \
                const uint q  = lo | (((P_##_byte(qh + l) & (1u << g)) != 0u) ? 16u : 0u); \
                return d * float(sc) * float(q) - dmin * float(mn);                       \
            }                                                                            \
            case GKT_Q6_K: {                                                             \
                const uint ql = b;                                                        \
                const uint qh = b + GK_QK / 2u;                                           \
                const uint sc = b + GK_QK / 2u + GK_QK / 4u;                              \
                const float d = P_##_h2f(sc + GK_QK / 16u);                               \
                const uint half_i = j / 128u;                                             \
                const uint r      = j % 128u;                                             \
                const uint which  = r / 32u;                                              \
                const uint i      = r % 32u;                                              \
                const uint is     = i / 16u;                                              \
                const uint l = ql + half_i * 64u;                                         \
                const uint h = qh + half_i * 32u;                                         \
                const uint s = sc + half_i * 8u;                                          \
                uint q;                                                                   \
                int  scale;                                                               \
                if (which == 0u) {                                                        \
                    q = (P_##_byte(l + i) & 15u) | (((P_##_byte(h + i) >> 0u) & 3u) << 4u); \
                    scale = int(P_##_byte(s + is) << 24u) >> 24;                          \
                } else if (which == 1u) {                                                 \
                    q = (P_##_byte(l + i + 32u) & 15u) | (((P_##_byte(h + i) >> 2u) & 3u) << 4u); \
                    scale = int(P_##_byte(s + is + 2u) << 24u) >> 24;                     \
                } else if (which == 2u) {                                                 \
                    q = (P_##_byte(l + i) >> 4u) | (((P_##_byte(h + i) >> 4u) & 3u) << 4u); \
                    scale = int(P_##_byte(s + is + 4u) << 24u) >> 24;                     \
                } else {                                                                  \
                    q = (P_##_byte(l + i + 32u) >> 4u) | (((P_##_byte(h + i) >> 6u) & 3u) << 4u); \
                    scale = int(P_##_byte(s + is + 6u) << 24u) >> 24;                     \
                }                                                                         \
                return d * float(scale) * (float(q) - 32.0);                              \
            }                                                                            \
            case GKT_IQ4_NL: {                                                           \
                const float d = P_##_h2f(b);                                             \
                const uint code = j < 16u ? (P_##_byte(b + 2u + j) & 15u)                 \
                                          : (P_##_byte(b + 2u + j - 16u) >> 4u);          \
                return d * float(gk_iq4_values[code]);                                    \
            }                                                                            \
            case GKT_IQ4_XS: {                                                           \
                const float d = P_##_h2f(b);                                             \
                const uint scales_h = P_##_byte(b + 2u) | (P_##_byte(b + 3u) << 8u);      \
                const uint scales_l = b + 4u;                                             \
                const uint qs       = b + 4u + GK_QK / 64u;                               \
                const uint g = j / 32u;                                                   \
                const uint r = j % 32u;                                                   \
                const uint ls = ((P_##_byte(scales_l + g / 2u) >> (4u * (g % 2u))) & 15u) \
                              | (((scales_h >> (2u * g)) & 3u) << 4u);                    \
                const float dl = d * (float(ls) - 32.0);                                  \
                const uint code = r < 16u ? (P_##_byte(qs + g * 16u + r) & 15u)            \
                                          : (P_##_byte(qs + g * 16u + r - 16u) >> 4u);     \
                return dl * float(gk_iq4_values[code]);                                   \
            }                                                                            \
            default: return 0.0;                                                          \
        }                                                                                 \
    }                                                                                     \
    float P_##_getq(TView t, uint i0, uint i1, uint i2, uint i3) {                        \
        const uint row = gk_row_off(t, i1, i2, i3);                                       \
        if (t.type == GKT_F32 || t.type == GKT_F16 || t.type == GKT_BF16 || t.type == GKT_I32) { \
            return P_##_plain(t, row + i0 * t.nb.x);                                      \
        }                                                                                 \
        const uint blck = gk_blck_size(t.type);                                           \
        const uint tsz  = gk_type_size(t.type);                                            \
        return P_##_block(row + (i0 / blck) * tsz, t.type, i0 % blck);                     \
    }

// The destination is f32 only: a narrower write would be a read-modify-write
// of a word two threads share.
#define GK_WRITER(P_, BUF)                                                               \
    void P_##_set(TView t, uint i0, uint i1, uint i2, uint i3, float v) {                \
        BUF[(gk_row_off(t, i1, i2, i3) + i0 * t.nb.x) >> 2] = floatBitsToUint(v);         \
    }                                                                                    \
    float P_##_read(TView t, uint i0, uint i1, uint i2, uint i3) {                       \
        return uintBitsToFloat(BUF[(gk_row_off(t, i1, i2, i3) + i0 * t.nb.x) >> 2]);      \
    }

// --------------------------------------------------------------------------
// index decomposition
// --------------------------------------------------------------------------

uvec4 gk_decompose(uint k, TView t) {
    uvec4 x;
    x.x = k % t.ne.x;
    x.y = (k / t.ne.x) % t.ne.y;
    x.z = (k / (t.ne.x * t.ne.y)) % t.ne.z;
    x.w = k / (t.ne.x * t.ne.y * t.ne.z);
    return x;
}

// --------------------------------------------------------------------------
// activations, transcribed from the CPU pass
// --------------------------------------------------------------------------

float gk_erf_poly(float x) {
    const float s = x < 0.0 ? -1.0 : 1.0;
    x = abs(x);

    const float p  = 0.3275911;
    const float a1 = 0.254829592;
    const float a2 = -0.284496736;
    const float a3 = 1.421413741;
    const float a4 = -1.453152027;
    const float a5 = 1.061405429;

    const float t = 1.0 / (1.0 + p * x);
    const float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return s * y;
}

float gk_gelu(float x) {
    const float c = 0.797884560802865;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

float gk_gelu_erf(float x)   { return 0.5 * x * (1.0 + gk_erf_poly(x * 0.7071067811865475)); }
float gk_gelu_quick(float x) { return x * (1.0 / (1.0 + exp(-1.702 * x))); }
float gk_silu(float x)       { return x / (1.0 + exp(-x)); }

float gk_unary(int op, float x, float p1, float p2, float p3, float p4) {
    switch (op) {
        case 0:  return abs(x);
        case 1:  return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
        case 2:  return -x;
        case 3:  return x > 0.0 ? 1.0 : 0.0;
        case 4:  return tanh(x);
        case 5:  return x > 0.0 ? x : exp(x) - 1.0;
        case 6:  return max(x, 0.0);
        case 7:  return 1.0 / (1.0 + exp(-x));
        case 8:  return gk_gelu(x);
        case 9:  return gk_gelu_quick(x);
        case 10: return gk_silu(x);
        case 11: return x * clamp((x + 3.0) / 6.0, 0.0, 1.0);
        case 12: return clamp((x + 3.0) / 6.0, 0.0, 1.0);
        case 13: return exp(x);
        case 14: return exp(x) - 1.0;
        case 15: return x > 20.0 ? x : log(1.0 + exp(x));
        case 16: return gk_gelu_erf(x);
        case 17: {
            if (x > 0.0) {
                return p2 * x * x + p3 * x;
            }
            const float mx = min(x, p4);
            return (exp(mx) - 1.0 - x) * p1 + p3 * x;
        }
        case 18: return floor(x);
        case 19: return ceil(x);
        case 20: return roundEven(x);
        case 21: return trunc(x);
        default: return x;
    }
}

// --------------------------------------------------------------------------
// block reductions
//
// A macro rather than a function because the shared array has to be declared
// in the shader that uses it. Every use is preceded by a barrier, so a shader
// may reduce twice without an explicit one in between.
// --------------------------------------------------------------------------

#define GK_REDUCTION(SIZE)                                                  \
    shared float gk_red[SIZE];                                              \
    float gk_block_sum(float v, uint tid) {                                 \
        barrier();                                                          \
        gk_red[tid] = v;                                                    \
        barrier();                                                          \
        for (uint s = SIZE / 2u; s > 0u; s >>= 1) {                          \
            if (tid < s) { gk_red[tid] += gk_red[tid + s]; }                 \
            barrier();                                                      \
        }                                                                   \
        return gk_red[0];                                                   \
    }                                                                       \
    float gk_block_max(float v, uint tid) {                                 \
        barrier();                                                          \
        gk_red[tid] = v;                                                    \
        barrier();                                                          \
        for (uint s = SIZE / 2u; s > 0u; s >>= 1) {                          \
            if (tid < s) { gk_red[tid] = max(gk_red[tid], gk_red[tid + s]); } \
            barrier();                                                      \
        }                                                                   \
        return gk_red[0];                                                   \
    }
