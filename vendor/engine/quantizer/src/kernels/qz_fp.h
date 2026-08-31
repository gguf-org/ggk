#pragma once

// Scalar conversions between f32 and the narrow float formats that appear in
// GGUF blocks: IEEE binary16, bfloat16, the E8M0 shared exponent used by MXFP4
// and the unsigned E4M3 scale used by NVFP4.
//
// All encoders round to nearest, ties to even, which is what the formats
// prescribe and what makes an encode/decode round trip stable.

#include "qz_quant.h"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t qz_bits_of(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

static inline float qz_float_of(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

// --------------------------------------------------------------------------
// IEEE binary16
// --------------------------------------------------------------------------

static inline float qz_h2f(qz_fp16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t man  = h & 0x3ffu;

    if (exp == 0) {
        if (man == 0) {
            return qz_float_of(sign); // +-0
        }
        // subnormal: value = man * 2^-24, exact in f32
        const float v = (float) man * 5.9604644775390625e-8f;
        return sign ? -v : v;
    }

    if (exp == 0x1f) {
        // inf / nan - keep the payload so a NaN stays a NaN
        return qz_float_of(sign | 0x7f800000u | (man << 13));
    }

    return qz_float_of(sign | ((exp + 112u) << 23) | (man << 13));
}

static inline qz_fp16_t qz_f2h(float f) {
    const uint32_t b    = qz_bits_of(f);
    const uint16_t sign = (uint16_t) ((b >> 16) & 0x8000u);
    const uint32_t mag  = b & 0x7fffffffu;

    if (mag > 0x7f800000u) {
        return (uint16_t) (sign | 0x7e00u); // nan, forced quiet
    }
    if (mag >= 0x477ff000u) {
        // >= 65520 rounds up past the largest finite half, and infinities land
        // here too
        return (uint16_t) (sign | 0x7c00u);
    }
    if (mag <= 0x33000000u) {
        // <= 2^-25: rounds to zero (the tie at exactly 2^-25 goes to even)
        return sign;
    }

    const int32_t  e = (int32_t) (mag >> 23) - 127;
    const uint32_t m = (mag & 0x7fffffu) | 0x800000u; // 24-bit significand

    // number of low bits of `m` that do not survive; 13 for a normal result,
    // more when the result lands in the subnormal range
    const int shift = e < -14 ? 13 + (-14 - e) : 13;

    // round to nearest, ties to even
    const uint32_t half = 1u << (shift - 1);
    const uint32_t r    = (m + half - 1u + ((m >> shift) & 1u)) >> shift;

    if (e < -14) {
        return (uint16_t) (sign | r); // r == 0x400 carries into the smallest normal
    }
    // r == 0x800 carries into the next exponent, which this sum absorbs
    return (uint16_t) (sign | (uint16_t) (((uint32_t) (e + 15) << 10) + r - 0x400u));
}

// --------------------------------------------------------------------------
// bfloat16 - the top 16 bits of an f32
// --------------------------------------------------------------------------

static inline float qz_bf2f(qz_bf16_t h) {
    return qz_float_of((uint32_t) h.bits << 16);
}

static inline qz_bf16_t qz_f2bf(float f) {
    const uint32_t b = qz_bits_of(f);
    qz_bf16_t out;

    if ((b & 0x7fffffffu) > 0x7f800000u) {
        out.bits = (uint16_t) ((b >> 16) | 0x40u); // nan, forced quiet
        return out;
    }
    out.bits = (uint16_t) ((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
    return out;
}

// --------------------------------------------------------------------------
// E8M0: an 8-bit shared exponent, value = 2^(e-127), used by MXFP4
// --------------------------------------------------------------------------

// The MXFP4 codebook stores its values doubled (see qz_codebook.h), so the
// scale that pairs with it is half the shared exponent.
static inline float qz_e8m0_to_fp32_half(uint8_t e) {
    if (e >= 2) {
        return qz_float_of((uint32_t) (e - 1) << 23); // 2^(e-128)
    }
    // 2^-128 and 2^-127 are below the smallest normal f32 and have to be
    // spelled out as subnormals
    return qz_float_of(0x00200000u << e);
}

// shared exponent for a block whose largest magnitude is `amax`: the exponent
// that maps the largest codebook entry (6.0, i.e. 2^2 * 1.5) onto amax without
// clipping
static inline uint8_t qz_e8m0_from_amax(float amax) {
    const uint32_t b = qz_bits_of(amax);
    const int32_t  e = (int32_t) ((b >> 23) & 0xffu);

    if (e == 0) {
        return 0; // zero or subnormal input: nothing to scale
    }
    const int32_t shared = e - 2;
    return (uint8_t) (shared < 0 ? 0 : shared > 255 ? 255 : shared);
}

// --------------------------------------------------------------------------
// UE4M3: unsigned, 4 exponent bits (bias 7), 3 mantissa bits, used as the
// per-group scale of NVFP4. Decoding halves the value for the same reason as
// E8M0 above. 0x7f is the NaN slot and decodes as zero.
// --------------------------------------------------------------------------

static inline float qz_ue4m3_to_fp32(uint8_t v) {
    if (v == 0 || v == 0x7f) {
        return 0.0f;
    }
    const uint32_t exp = (v >> 3) & 0xfu;
    const uint32_t man = v & 0x7u;

    if (exp == 0) {
        return (float) man * (1.0f / 512.0f) * 0.5f; // subnormal: man * 2^-9, halved
    }
    return qz_float_of(((exp + 120u) << 23) | (man << 20)) * 0.5f;
}

static inline uint8_t qz_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x >= 464.0f) {
        return 0x7e; // 448 is the largest value; halfway to the NaN slot clamps
    }

    const uint32_t b = qz_bits_of(x);
    const int32_t  e = (int32_t) ((b >> 23) & 0xffu) - 127;

    if (e < -6) {
        // subnormal range: value = man * 2^-9
        const float scaled = x * 512.0f;
        int man = (int) (scaled + 0.5f);
        if ((float) man - scaled == 0.5f && (man & 1)) {
            man -= 1; // tie to even
        }
        if (man > 7) {
            man = 7;
        }
        return (uint8_t) (man < 0 ? 0 : man);
    }

    const uint32_t m     = (b & 0x7fffffu) | 0x800000u; // 24-bit significand
    const uint32_t half  = 1u << 19;
    const uint32_t r     = (m + half - 1u + ((m >> 20) & 1u)) >> 20; // 4 bits, 0x8..0x10

    // r == 0x10 means the rounding carried into the next exponent
    const int32_t  exp   = e + 7 + (r == 0x10 ? 1 : 0);
    const uint32_t man   = r == 0x10 ? 0u : (r & 0x7u);

    if (exp >= 15) {
        return 0x7e;
    }
    return (uint8_t) (((uint32_t) exp << 3) | man);
}

#ifdef __cplusplus
}
#endif
