#pragma once

// Fixed codebooks of the lattice and micro-scaling formats. See qz_codebook.c
// for what the entries mean and why they are pinned.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QZ_GRID_IQ2_XXS_SIZE 256
#define QZ_GRID_IQ2_XS_SIZE  512
#define QZ_GRID_IQ2_S_SIZE  1024
#define QZ_GRID_IQ3_XXS_SIZE 256
#define QZ_GRID_IQ3_S_SIZE   512
#define QZ_GRID_IQ1_SIZE    2048

// eight 8-bit magnitudes per entry
extern const uint64_t qz_grid_iq2_xxs[QZ_GRID_IQ2_XXS_SIZE];
extern const uint64_t qz_grid_iq2_xs [QZ_GRID_IQ2_XS_SIZE];
extern const uint64_t qz_grid_iq2_s  [QZ_GRID_IQ2_S_SIZE];

// four 8-bit magnitudes per entry
extern const uint32_t qz_grid_iq3_xxs[QZ_GRID_IQ3_XXS_SIZE];
extern const uint32_t qz_grid_iq3_s  [QZ_GRID_IQ3_S_SIZE];

// eight signed ternary values per entry
extern const uint64_t qz_grid_iq1[QZ_GRID_IQ1_SIZE];

// 16-entry non-linear 4-bit codebooks
extern const int8_t qz_iq4_values [16];
extern const int8_t qz_e2m1_values[16];  // doubled E2M1 values

// Every UE4M3 code decoded. qz_ue4m3_to_fp32 in qz_fp.h is the definition;
// this is that function tabulated, for the benefit of a decoder that has to
// evaluate it in an inner loop rather than once per block. NVFP4 carries one
// of these scales per 16 weights, so a dot product against an NVFP4 operand
// hits it that often, and doing the bit arithmetic there costs more than the
// vector work it feeds.
extern const float qz_ue4m3_values[256];

// The sign mask stored alongside a lattice index only carries seven bits; the
// eighth is implied by the rule that the mask has an even number of set bits.
static inline uint8_t qz_sign_mask_from_bits(uint8_t low7) {
    uint8_t m = (uint8_t) (low7 & 0x7f);
    uint8_t p = m;
    p ^= (uint8_t) (p >> 4);
    p ^= (uint8_t) (p >> 2);
    p ^= (uint8_t) (p >> 1);
    return (uint8_t) (p & 1 ? m | 0x80 : m);
}

#ifdef __cplusplus
}
#endif
