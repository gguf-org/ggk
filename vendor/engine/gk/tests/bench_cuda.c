// A per-op benchmark for the device backend.
//
// bench.c answers "is the SIMD worth it" for the CPU pass. This answers a
// different question: for each op, what does the device kernel actually cost,
// and is it better than the CPU doing the same work? That second half is the
// point. An op with no device kernel does not fail - the scheduler quietly
// runs it on the CPU, which costs a graph split and a round trip over the bus
// on top of the CPU time - so an op that is missing and an op that is slow
// look identical from outside. Here they are separate columns.
//
// Every case builds the same graph twice, once against each backend, and
// times both. What comes out is:
//
//   - GPU ms: the device kernel, or "-" when there is no kernel for the op
//   - CPU ms: the same graph on the CPU backend, all cores
//   - x:      how many times faster the device is; below 1.0 the device is
//             losing to the CPU, which for a GPU-shaped op means the kernel
//             is wrong rather than merely unoptimized
//
// The shapes are gemma-4-e2b's, read off the model file, so the numbers are
// about a model someone runs rather than a round number. The matmul cases
// name the tensor they stand for.
//
// Build:
//   cmake -B build-cuda -DGK_CUDA=ON -DGK_CUDA_ARCHITECTURES=89
//   cmake --build build-cuda --target bench-cuda --config Release
//
// Run with no arguments for everything, or pass a substring to filter:
//   bench-cuda mul_mat
//   bench-cuda flash

#include "gk_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

static double now_sec(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double) counter.QuadPart / (double) frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
#endif
}

static uint64_t g_rng = 0x243f6a8885a308d3ull;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float) (uint32_t) (g_rng >> 33) / (float) 0x7fffffffu * 2.0f - 1.0f;
}

// The model this is shaped after. Read from gemma-4-e2b-it-q4_0.gguf: 35
// blocks, MQA with one KV head, and a vocabulary large enough that the output
// projection is a different kind of matmul from everything before it.
#define N_EMBD    1536
#define N_FF      6144
#define N_HEAD       8
#define N_HEAD_KV    1
#define HEAD_DIM   256
#define N_VOCAB 262144

// A prefill batch and a decode batch, which are the two regimes every kernel
// here has to be good at and which stress opposite things: prefill has enough
// columns to reuse a decoded weight, decode has one and is pure bandwidth.
#define N_PREFILL  512
#define N_DECODE     1

// --------------------------------------------------------------------------
// filling inputs
//
// Timing is mostly data-independent, but not entirely: the CPU argsort is an
// insertion sort, whose cost depends on how far from sorted the input is, and
// a buffer left at whatever the allocator held can hold NaNs, which change
// what the CPU's vector paths do. So every leaf gets written before a case is
// timed. Leaves that mean something specific - positions, row ids, an
// attention mask - say so in their name, because a random int32 read as a row
// index is an out-of-bounds read rather than a slow one.
// --------------------------------------------------------------------------

// Reads the ":N" suffix a builder attaches to an index tensor's name, which is
// the range those indices have to stay inside.
static int64_t name_limit(const struct gk_tensor * t, int64_t fallback) {
    const char * colon = strchr(t->name, ':');
    if (colon == NULL) {
        return fallback;
    }
    const long long v = atoll(colon + 1);
    return v > 0 ? (int64_t) v : fallback;
}

// How many distinct rows the generic filler bothers to generate before it
// starts repeating them. The values only have to be plausible - no kernel
// here is faster or slower for one float than another - and a vocabulary-sized
// weight has a quarter of a million rows, so generating each one would cost
// more than every case in the table put together.
#define FILL_MAX_ROWS 64

// The generic fill: a bounded block of rows, encoded once, repeated across the
// tensor. Named tensors do not come here - a position, a row id or a mask
// means something at every index and is small enough to write out in full.
static void fill_generic(struct gk_tensor * t) {
    const int64_t row_n = t->ne[0];
    const int64_t rows  = gk_nelements(t) / row_n;
    const int64_t chunk = rows < FILL_MAX_ROWS ? rows : FILL_MAX_ROWS;

    const size_t row_bytes = gk_row_size(t->type, row_n);

    float * src = (float *) malloc((size_t) chunk * row_n * sizeof(float));
    char  * enc = (char  *) malloc((size_t) chunk * row_bytes);
    if (src == NULL || enc == NULL) {
        free(src);
        free(enc);
        return;
    }

    for (int64_t i = 0; i < chunk * row_n; ++i) {
        src[i] = frand();
    }

    if (t->type == GK_TYPE_F32) {
        memcpy(enc, src, (size_t) chunk * row_bytes);
    } else {
        const struct gk_type_traits * traits = gk_get_type_traits(t->type);
        if (traits == NULL || traits->from_float == NULL) {
            free(src);
            free(enc);
            return;
        }
        for (int64_t r = 0; r < chunk; ++r) {
            traits->from_float(src + r * row_n, enc + (size_t) r * row_bytes, row_n);
        }
    }

    for (int64_t r = 0; r < rows; r += chunk) {
        const int64_t take = rows - r < chunk ? rows - r : chunk;
        gk_backend_tensor_set(t, enc, (size_t) r * row_bytes, (size_t) take * row_bytes);
    }

    free(enc);
    free(src);
}

static void fill_tensor(struct gk_tensor * t) {
    const int64_t n = gk_nelements(t);
    if (n == 0) {
        return;
    }

    if (t->type == GK_TYPE_I32) {
        int32_t * buf = (int32_t *) malloc((size_t) n * sizeof(int32_t));
        if (buf == NULL) {
            return;
        }
        const int64_t limit = name_limit(t, 1);
        for (int64_t i = 0; i < n; ++i) {
            // positions are the token's place in the sequence; row ids have to
            // land inside the table they index
            buf[i] = strncmp(t->name, "pos", 3) == 0
                ? (int32_t) i
                : (int32_t) (i % limit);
        }
        gk_backend_tensor_set(t, buf, 0, (size_t) n * sizeof(int32_t));
        free(buf);
        return;
    }

    // Everything whose values are arbitrary goes through the bounded filler.
    // What is left below means something at every index, and all of it is
    // small: a mask is one row per query, a state is one per sequence.
    if (strncmp(t->name, "mask",  4) != 0 &&
        strncmp(t->name, "state", 5) != 0 &&
        strncmp(t->name, "decay", 5) != 0) {
        fill_generic(t);
        return;
    }

    float * src = (float *) malloc((size_t) n * sizeof(float));
    if (src == NULL) {
        return;
    }

    if (strncmp(t->name, "mask", 4) == 0) {
        // A causal mask, which is the shape attention actually sees: the
        // kernel skips fully masked positions, so a mask of all zeros would
        // time a longer sequence than the model ever runs.
        const int64_t n_kv = t->ne[0];
        for (int64_t j = 0; j < t->ne[1]; ++j) {
            for (int64_t i = 0; i < n_kv; ++i) {
                src[j * n_kv + i] = i <= j + (n_kv - t->ne[1]) ? 0.0f : -INFINITY;
            }
        }
    } else if (strncmp(t->name, "state", 5) == 0) {
        memset(src, 0, (size_t) n * sizeof(float));
    } else {
        // a recurrent decay: has to stay inside (0, 1) or the state diverges
        // to Inf partway through the sequence and the rest is denormal math
        for (int64_t i = 0; i < n; ++i) {
            src[i] = 0.9f + 0.05f * frand();
        }
    }

    if (t->type == GK_TYPE_F32) {
        gk_backend_tensor_set(t, src, 0, (size_t) n * sizeof(float));
        free(src);
        return;
    }

    const struct gk_type_traits * traits = gk_get_type_traits(t->type);
    if (traits == NULL || traits->from_float == NULL) {
        free(src);
        return;
    }

    // Encoded row by row, because a quantized row is a sequence of blocks and
    // from_float works a row at a time.
    const size_t nbytes = gk_nbytes(t);
    char * enc = (char *) malloc(nbytes);
    if (enc == NULL) {
        free(src);
        return;
    }
    const int64_t row_n = t->ne[0];
    const int64_t rows  = n / row_n;
    const size_t row_bytes = gk_row_size(t->type, row_n);
    for (int64_t r = 0; r < rows; ++r) {
        traits->from_float(src + r * row_n, enc + (size_t) r * row_bytes, row_n);
    }
    gk_backend_tensor_set(t, enc, 0, nbytes);

    free(enc);
    free(src);
}

// --------------------------------------------------------------------------
// the cases
//
// A builder gets an arena and returns the node to time. Inputs are created
// inside it and named where the name matters; everything else the harness
// works out from the graph.
// --------------------------------------------------------------------------

typedef struct gk_tensor * (*build_fn)(struct gk_ctx * ctx);

// One weight against `cols` activation columns - the shape of every linear
// layer in the model, and the op that owns most of inference.
static struct gk_tensor * mul_mat_case(struct gk_ctx * ctx, enum gk_type type,
                                       int64_t k, int64_t rows, int64_t cols) {
    struct gk_tensor * w = gk_new_tensor_2d(ctx, type, k, rows);
    struct gk_tensor * x = gk_new_tensor_2d(ctx, GK_TYPE_F32, k, cols);
    gk_set_name(w, "weight");
    gk_set_name(x, "x");
    return gk_mul_mat(ctx, w, x);
}

static struct gk_tensor * b_mm_q_dec   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_0, N_EMBD, 2048,    N_DECODE);  }
static struct gk_tensor * b_mm_kv_dec  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_0, N_EMBD, 256,     N_DECODE);  }
static struct gk_tensor * b_mm_out_dec (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_0, 2048,   N_EMBD,  N_DECODE);  }
static struct gk_tensor * b_mm_gate_dec(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_0, N_EMBD, N_FF,    N_DECODE);  }
static struct gk_tensor * b_mm_down_dec(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_1, N_FF,   N_EMBD,  N_DECODE);  }
static struct gk_tensor * b_mm_lm_dec  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_K, N_EMBD, N_VOCAB, N_DECODE);  }
static struct gk_tensor * b_mm_f16_dec (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16,  N_EMBD, N_FF,    N_DECODE);  }

// The same matmul in every weight format. One shape, one column, so the only
// thing that changes between rows is what it costs to read a weight element -
// which makes the rate column a direct measurement of the decoder rather than
// of the memory system. A format that moves less data and takes longer is
// paying for its decode, and the f32/f16 rows are the control: they have no
// decode at all and show what the shape is worth when the bytes are the only
// cost.
static struct gk_tensor * b_dec_f32  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F32,   N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_f16  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16,   N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_q8_0 (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q8_0,  N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_q4_0 (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_0,  N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_q4_K (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_K,  N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_q6_K (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q6_K,  N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_mxfp4(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_MXFP4, N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_nvfp4(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_NVFP4, N_EMBD, N_FF, N_DECODE); }

// The lattice formats, which decode through a codebook gather rather than
// arithmetic on the bits. They are the ones a 2-bit model is built from, and
// the question this row answers is what that gather costs against a format
// whose decode is a shift and a multiply.
static struct gk_tensor * b_dec_iq2_xxs(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_XXS, N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_iq2_s  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_S,   N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_iq3_xxs(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_XXS, N_EMBD, N_FF, N_DECODE); }
static struct gk_tensor * b_dec_iq3_s  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_S,   N_EMBD, N_FF, N_DECODE); }

// The same question at a 30B model's FFN width. The row above is small enough
// that every format lands on the launch-latency floor and the bandwidth column
// says nothing; this one moves enough bytes for the kernel to be the cost.
#define N_EMBD_30B 6656
#define N_FF_30B  19968
static struct gk_tensor * b_dec30_f16    (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16,     N_EMBD_30B, N_FF_30B, N_DECODE); }
static struct gk_tensor * b_dec30_q4_K   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_K,    N_EMBD_30B, N_FF_30B, N_DECODE); }
static struct gk_tensor * b_dec30_iq2_xxs(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_XXS, N_EMBD_30B, N_FF_30B, N_DECODE); }
static struct gk_tensor * b_dec30_iq2_s  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_S,   N_EMBD_30B, N_FF_30B, N_DECODE); }
static struct gk_tensor * b_dec30_iq3_xxs(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_XXS, N_EMBD_30B, N_FF_30B, N_DECODE); }
static struct gk_tensor * b_dec30_iq3_s  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_S,   N_EMBD_30B, N_FF_30B, N_DECODE); }
static struct gk_tensor * b_dec30_q6_K   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q6_K,    N_EMBD_30B, N_FF_30B, N_DECODE); }

// The lm_head of a 30B model with a 202k vocabulary, which is one matmul per
// generated token and the single largest read in the graph.
static struct gk_tensor * b_lm_q6_K(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q6_K, 6656, 202048, 1); }

// The same weights against four activation columns instead of one.
//
// Four is what speculative decoding makes every one of these shapes: the
// target verifies the drafted block in a single pass, so a run with a draft
// never sees a one-column matmul again. The weights are read once either way,
// so the pair of rows should differ by almost nothing - a fourth column adds
// an accumulator, not a pass over memory. Where it does not, the extra columns
// are being paid for in decode work or in indexing, and the model loses more
// on the wider verification pass than the draft wins back.
//
// The formats are muse-glimmer-30b-iq2_xxs's, taken from a profile rather than
// picked: attn_k is q5_K, attn_v and ffn_down iq3_xxs, ffn_gate/up iq2_s,
// attn_q iq2_xxs, attn_out iq3_s, and the lm_head q6_K over a 202k vocabulary.
// attn_k/v are also the two shapes narrow enough to miss the integer path, so
// they measure the float mat-vec where the rest measure the integer one.
static struct gk_tensor * b_ver1_gate(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_S,   6656, 19968,      1); }
// Three columns is a draft depth of two - two drafted tokens and the one
// that samples - so it is the width a `--spec-draft-n-max 2` run verifies at.
static struct gk_tensor * b_ver3_gate(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_S,   6656, 19968,      3); }
static struct gk_tensor * b_ver4_gate(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_S,   6656, 19968,      4); }
static struct gk_tensor * b_ver3_lm  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q6_K,    6656, 202048,     3); }
static struct gk_tensor * b_ver1_down(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_XXS, 19968, 6656,      1); }
static struct gk_tensor * b_ver4_down(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_XXS, 19968, 6656,      4); }
static struct gk_tensor * b_ver1_q   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_XXS, 6656,  4096,      1); }
static struct gk_tensor * b_ver4_q   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_XXS, 6656,  4096,      4); }
static struct gk_tensor * b_ver1_o   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_S,   4096,  6656,      1); }
static struct gk_tensor * b_ver4_o   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_S,   4096,  6656,      4); }
static struct gk_tensor * b_ver1_k   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q5_K,    6656,   256,      1); }
static struct gk_tensor * b_ver4_k   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q5_K,    6656,   256,      4); }
static struct gk_tensor * b_ver1_v   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_XXS, 6656,   256,      1); }
static struct gk_tensor * b_ver4_v   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ3_XXS, 6656,   256,      4); }
static struct gk_tensor * b_ver4_lm  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q6_K,    6656, 202048,     4); }

// The three nvfp4 mat-vecs a DFlash draft runs, taken from a profile of one
// rather than invented: the KV projections, the encoder's fusion matrix over
// five concatenated target layers, and an FFN matmul as the control. In that
// profile the first two ran at a thousandth of the third's bandwidth, so the
// shapes are here rather than a round number.
static struct gk_tensor * b_dft_kv  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_NVFP4,  6656,  1024, 4); }
static struct gk_tensor * b_dft_fc  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_NVFP4, 33280,  6656, 4); }
static struct gk_tensor * b_dft_ffn (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_NVFP4,  6656, 19968, 4); }
static struct gk_tensor * b_dft_o   (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_NVFP4,  4096,  6656, 4); }

// The same sweep at a batch, which is the regime a diffusion transformer runs
// in: hundreds of tokens per matmul, never one. It is a different kernel from
// the decode sweep above - the tiled one - so a format can be fine in one and
// poor in the other, and a model that only ever runs batched is only ever
// measured here.
static struct gk_tensor * b_pre_f16  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16,   N_EMBD, N_FF, N_PREFILL); }
static struct gk_tensor * b_pre_q8_0 (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q8_0,  N_EMBD, N_FF, N_PREFILL); }
static struct gk_tensor * b_pre_q4_0 (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_0,  N_EMBD, N_FF, N_PREFILL); }
static struct gk_tensor * b_pre_q4_K (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_K,  N_EMBD, N_FF, N_PREFILL); }
static struct gk_tensor * b_pre_mxfp4(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_MXFP4, N_EMBD, N_FF, N_PREFILL); }
static struct gk_tensor * b_pre_nvfp4(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_NVFP4, N_EMBD, N_FF, N_PREFILL); }
static struct gk_tensor * b_pre_iq2_xxs(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_XXS, N_EMBD, N_FF, N_PREFILL); }
static struct gk_tensor * b_pre_iq2_s  (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_IQ2_S,   N_EMBD, N_FF, N_PREFILL); }

// The six matmuls a SD 1.x UNet actually spends its time in, taken from a
// profile of a generation rather than invented. They are not the shapes above:
// a language model's are tall and narrow with a handful of columns, and these
// are short and very wide - 4096 columns is one 64x64 latent's worth of pixels,
// and k runs from 320 to 5120 as the UNet goes down its levels.
//
// Each is given twice, in nvfp4 and in f16. The f16 row is not a competitor, it
// is the ceiling: same shape, same tile machinery, no weight decode, so the
// difference between the pair is what the format costs and nothing else.
#define SD_MM(name, k, rows, cols)                                                       \
    static struct gk_tensor * b_sd_##name##_nv(struct gk_ctx * c) {                      \
        return mul_mat_case(c, GK_TYPE_NVFP4, (k), (rows), (cols));                      \
    }                                                                                    \
    static struct gk_tensor * b_sd_##name##_f16(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_F16, (k), (rows), (cols));                        \
    }

// The convolutions, which are a different matmul from every one above and were
// 29% of an SD 1.x step. `ggml_conv_2d` is
// `mul_mat(im2col[k, N*OH*OW], kernel[k, OC])`, so **both operands are f16 and
// the big one is the activation**: 4096 rows of 8640 halves is 71 MB against a
// 5 MB weight, the reverse of every other row in this file. k is the patch,
// 3x3 times the input channels.
static struct gk_tensor * conv_case(struct gk_ctx * ctx, int64_t k, int64_t pixels, int64_t oc) {
    struct gk_tensor * im = gk_new_tensor_2d(ctx, GK_TYPE_F16, k, pixels);
    struct gk_tensor * w  = gk_new_tensor_2d(ctx, GK_TYPE_F16, k, oc);
    gk_set_name(im, "im2col");
    gk_set_name(w,  "kernel");
    return gk_mul_mat(ctx, im, w);
}

#define SD_CONV(name, k, pixels, oc)                                                     \
    static struct gk_tensor * b_conv_##name(struct gk_ctx * c) {                         \
        return conv_case(c, (k), (pixels), (oc));                                        \
    }

SD_CONV(l1_320,   2880, 4096, 320)    // 64x64x320, 3x3 -> 320
SD_CONV(l1_640,   5760, 4096, 320)    // 64x64x640 (after a skip concat) -> 320
SD_CONV(l1_960,   8640, 4096, 320)
SD_CONV(l1_down,  5760, 4096, 640)
SD_CONV(l2_640,   5760, 1024, 640)
SD_CONV(l3_1280, 11520,  256, 1280)
SD_CONV(l4_1280, 11520,   64, 1280)

SD_MM(l1_attn,  320, 320,  4096)   // 64x64 level, attention projection
SD_MM(l1_ff,    320, 2560, 4096)   // 64x64 level, feed-forward in
SD_MM(l1_out,  1280, 320,  4096)   // 64x64 level, feed-forward out
SD_MM(l2_attn,  640, 640,  1024)   // 32x32 level
SD_MM(l2_ff,    640, 5120, 1024)
SD_MM(l3_ff,   1280, 10240, 256)   // 16x16 level, the widest rows
SD_MM(l3_out,  5120, 1280,  256)

// The three matmuls a MageFlow DiT step is made of, read off a profile of an
// actual generation rather than invented: 2048 tokens is a 512x512 latent, the
// hidden width is 3072 and the feed-forward opens to 12288. Between them they
// are 79% of a denoising step, so what these rows say about nvfp4 is very
// nearly what the whole engine's diffusion speed is.
//
// Each shape is given three ways. nvfp4 is the one that runs; f16 is the same
// shape with no weight decode at all, so it is the ceiling the format is being
// measured against; q4_K is the control that matters most, because it is a
// decoded format like nvfp4 but takes a different kernel - if q4_K is fast and
// nvfp4 is slow at one shape, the shape is not the problem.
#define DIT_MM(name, k, rows)                                                            \
    static struct gk_tensor * b_dit_##name##_nv (struct gk_ctx * c) {                    \
        return mul_mat_case(c, GK_TYPE_NVFP4, (k), (rows), 2048);                        \
    }                                                                                    \
    static struct gk_tensor * b_dit_##name##_f16(struct gk_ctx * c) {                    \
        return mul_mat_case(c, GK_TYPE_F16,   (k), (rows), 2048);                        \
    }                                                                                    \
    static struct gk_tensor * b_dit_##name##_q4k(struct gk_ctx * c) {                    \
        return mul_mat_case(c, GK_TYPE_Q4_K,  (k), (rows), 2048);                        \
    }

DIT_MM(proj,  3072,  3072)   // attention qkv / out projection - 48 per step
DIT_MM(ffup,  3072, 12288)   // feed-forward in  - 12 per step
DIT_MM(ffdn, 12288,  3072)   // feed-forward out - 12 per step

// The four matmuls a Z-Image DiT step is made of, read off a profile of an
// actual 8-step generation: 1056 tokens, hidden 3840, feed-forward opening to
// 10240 (up and gate fused), and a fused qkv of 11520 rows. Together they are
// 64% of the run's device time at nvfp4. Same three ways as above.
#define ZIMG_MM(name, k, rows)                                                           \
    static struct gk_tensor * b_zi_##name##_nv (struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_NVFP4, (k), (rows), 1056);                        \
    }                                                                                    \
    static struct gk_tensor * b_zi_##name##_f16(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_F16,   (k), (rows), 1056);                        \
    }                                                                                    \
    static struct gk_tensor * b_zi_##name##_q4k(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_Q4_K,  (k), (rows), 1056);                        \
    }

ZIMG_MM(qkv,   3840, 11520)
ZIMG_MM(proj,  3840,  3840)
ZIMG_MM(ffup,  3840, 10240)
ZIMG_MM(ffdn, 10240,  3840)

// The four matmuls a krea2 DiT step is made of, read off Investigation 9's
// profile of an actual 8-step generation: 1038 tokens, hidden 6144,
// feed-forward opening to 16384, and a fused 1536-row modulation projection.
// At q2_K they were 83% of the step's device time. q2_K is the point of this
// group: it is a split-scale format, so it drains the integer tile twice per
// group where q4_K drains once - q4_K at the same shape is the control that
// prices that drain, and f16 is the no-decode ceiling.
#define KREA2_MM(name, k, rows)                                                          \
    static struct gk_tensor * b_k2_##name##_q2k(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_Q2_K,  (k), (rows), 1038);                        \
    }                                                                                    \
    static struct gk_tensor * b_k2_##name##_f16(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_F16,   (k), (rows), 1038);                        \
    }                                                                                    \
    static struct gk_tensor * b_k2_##name##_q4k(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_Q4_K,  (k), (rows), 1038);                        \
    }

KREA2_MM(ffup,  6144, 16384)   // feed-forward in - the wide shape
KREA2_MM(ffdn, 16384,  6144)   // feed-forward out - the deep-k shape
KREA2_MM(proj,  6144,  6144)   // attention projection
KREA2_MM(mod,   6144,  1536)   // modulation - short of rows

// The four matmuls a MiniMax-H3 video DiT step is made of, read off a profile
// of an actual 480x480x124 generation: 8742 tokens, hidden 5376, feed-forward
// to 14336, and a fused qkv of 28672 rows. Between them they are 59% of a
// denoising step at nvfp4, so the same argument as above applies with a wider
// margin - and the shapes are five times the MageFlow ones in every direction,
// which is where a tile that re-streams its operands starts to show.
//
// Same three ways: nvfp4 is what runs, f16 is the no-decode ceiling, q4_K is
// the control that takes a *different* kernel at the same shape. That last one
// is the point of the group - the two formats have separate tiles, and their
// separation is the hypothesis being tested.
#define DIT_H3(name, k, rows)                                                            \
    static struct gk_tensor * b_h3_##name##_nv (struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_NVFP4, (k), (rows), 8742);                        \
    }                                                                                    \
    static struct gk_tensor * b_h3_##name##_f16(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_F16,   (k), (rows), 8742);                        \
    }                                                                                    \
    static struct gk_tensor * b_h3_##name##_q4k(struct gk_ctx * c) {                     \
        return mul_mat_case(c, GK_TYPE_Q4_K,  (k), (rows), 8742);                        \
    }

DIT_H3(qkv,   5376, 28672)   // fused qkv projection
DIT_H3(proj,  5376, 21504)   // the 21504-row projection
DIT_H3(ffdn, 14336,  5376)   // feed-forward out - the long-k shape
DIT_H3(ffo,   7168,  5376)   // the 7168-k projection

// The MiniMax-H3 *video VAE* decoder, which is a second 36-layer transformer
// and not the convolutional stack the name suggests: hidden 2048, 32 heads of
// 64, 1797 tokens a tile, and 28 tiles for a 480x480x124 clip. It runs 1008
// times per decode, so it is the same order of work as the DiT and lands on
// entirely different kernels - f16 weights rather than nvfp4, and an
// *unfused* attention whose two matmuls are f32 on both operands.
//
// The f32 pair is the point of this group. Everything else gk has tuned is a
// quantized or f16 tile; these two land on `tile-f32`, which has no
// tensor-core path at all, and they are what `--diffusion-fa` leaves unfused.
static struct gk_tensor * b_vae_ffup(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16, 2048, 16384, 1797); }
static struct gk_tensor * b_vae_ffdn(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16, 8192,  2048, 1797); }
static struct gk_tensor * b_vae_qkv (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16, 2048,  6144, 1797); }
static struct gk_tensor * b_vae_out (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16, 2048,  2048, 1797); }

// Batched over the 32 heads, which is how the graph issues them: one call per
// layer per tile with ne[2] = 32, not 32 separate calls.
static struct gk_tensor * mul_mat_heads(struct gk_ctx * ctx, enum gk_type type,
                                        int64_t k, int64_t rows, int64_t cols, int64_t heads) {
    struct gk_tensor * w = gk_new_tensor_3d(ctx, type, k, rows, heads);
    struct gk_tensor * x = gk_new_tensor_3d(ctx, GK_TYPE_F32, k, cols, heads);
    gk_set_name(w, "weight");
    gk_set_name(x, "x");
    return gk_mul_mat(ctx, w, x);
}

static struct gk_tensor * b_vae_kq (struct gk_ctx * c) { return mul_mat_heads(c, GK_TYPE_F32,   64, 1797, 1797, 32); }
static struct gk_tensor * b_vae_kqv(struct gk_ctx * c) { return mul_mat_heads(c, GK_TYPE_F32, 1797,   64, 1797, 32); }

// The same two shapes in f16, as the ceiling a tensor-core path would aim at.
static struct gk_tensor * b_vae_kq_f16 (struct gk_ctx * c) { return mul_mat_heads(c, GK_TYPE_F16,   64, 1797, 1797, 32); }
static struct gk_tensor * b_vae_kqv_f16(struct gk_ctx * c) { return mul_mat_heads(c, GK_TYPE_F16, 1797,   64, 1797, 32); }

static struct gk_tensor * b_mm_gate_pre(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_0, N_EMBD, N_FF,    N_PREFILL); }
static struct gk_tensor * b_mm_down_pre(struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_Q4_1, N_FF,   N_EMBD,  N_PREFILL); }
static struct gk_tensor * b_mm_f16_pre (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F16,  N_EMBD, N_FF,    N_PREFILL); }
static struct gk_tensor * b_mm_f32_pre (struct gk_ctx * c) { return mul_mat_case(c, GK_TYPE_F32,  N_EMBD, N_FF,    N_PREFILL); }

// Fused attention over an existing cache. `n_batch` is 1 for decode, where the
// kernel has one query row and a whole cache to walk, and N_PREFILL for the
// prompt pass, where it has many.
static struct gk_tensor * fattn_case(struct gk_ctx * ctx, int64_t n_batch, int64_t n_kv) {
    struct gk_tensor * q = gk_new_tensor_4d(ctx, GK_TYPE_F32, HEAD_DIM, n_batch, N_HEAD, 1);
    struct gk_tensor * k = gk_new_tensor_4d(ctx, GK_TYPE_F16, HEAD_DIM, n_kv, N_HEAD_KV, 1);
    struct gk_tensor * v = gk_new_tensor_4d(ctx, GK_TYPE_F16, HEAD_DIM, n_kv, N_HEAD_KV, 1);
    struct gk_tensor * m = gk_new_tensor_4d(ctx, GK_TYPE_F16, n_kv, n_batch, 1, 1);
    gk_set_name(q, "q");
    gk_set_name(k, "k");
    gk_set_name(v, "v");
    gk_set_name(m, "mask");
    return gk_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) HEAD_DIM), 0.0f, 0.0f);
}

static struct gk_tensor * b_fa_dec_512 (struct gk_ctx * c) { return fattn_case(c, N_DECODE,  512);  }
static struct gk_tensor * b_fa_dec_2k  (struct gk_ctx * c) { return fattn_case(c, N_DECODE,  2048); }
static struct gk_tensor * b_fa_dec_8k  (struct gk_ctx * c) { return fattn_case(c, N_DECODE,  8192); }
static struct gk_tensor * b_fa_pre_512 (struct gk_ctx * c) { return fattn_case(c, N_PREFILL, 512);  }

// Unfused attention: the path a model takes when flash attention is off or
// refused. Timed because it is the fallback the FA kernel is competing with.
// Self-attention over a whole image's worth of tokens, which is the shape a
// diffusion transformer runs: every token attends to every other, so both the
// query count and the cache grow together and the work grows with the square.
// Nothing in the decode cases above reaches this regime.
static struct gk_tensor * fattn_square(struct gk_ctx * ctx, int64_t n_tok, int64_t n_head) {
    const int64_t DK = 64;
    struct gk_tensor * q = gk_new_tensor_4d(ctx, GK_TYPE_F32, DK, n_tok, n_head, 1);
    struct gk_tensor * k = gk_new_tensor_4d(ctx, GK_TYPE_F16, DK, n_tok, n_head, 1);
    struct gk_tensor * v = gk_new_tensor_4d(ctx, GK_TYPE_F16, DK, n_tok, n_head, 1);
    gk_set_name(q, "q");
    gk_set_name(k, "k");
    gk_set_name(v, "v");
    return gk_flash_attn_ext(ctx, q, k, v, NULL, 1.0f / sqrtf((float) DK), 0.0f, 0.0f);
}

// Head width 128, which is what most diffusion transformers actually use and
// what the shared tiles have to be sized for.
static struct gk_tensor * fattn_square_d(struct gk_ctx * ctx, int64_t n_tok, int64_t n_head,
                                         int64_t DK) {
    struct gk_tensor * q = gk_new_tensor_4d(ctx, GK_TYPE_F32, DK, n_tok, n_head, 1);
    struct gk_tensor * k = gk_new_tensor_4d(ctx, GK_TYPE_F16, DK, n_tok, n_head, 1);
    struct gk_tensor * v = gk_new_tensor_4d(ctx, GK_TYPE_F16, DK, n_tok, n_head, 1);
    gk_set_name(q, "q");
    gk_set_name(k, "k");
    gk_set_name(v, "v");
    return gk_flash_attn_ext(ctx, q, k, v, NULL, 1.0f / sqrtf((float) DK), 0.0f, 0.0f);
}

static struct gk_tensor * b_fa_dit_2k_d128(struct gk_ctx * c) { return fattn_square_d(c, 2048, 16, 128); }
static struct gk_tensor * b_fa_dit_4k_d128(struct gk_ctx * c) { return fattn_square_d(c, 4096, 16, 128); }

// The MiniMax-H3 video DiT's attention, at its real size: 8742 tokens against
// itself, 56 heads of 128. It is 34% of a denoising step after Tier 2 - the
// single largest thing left - and none of the shapes above reach it: the
// widest is 4096 tokens and 16 heads, an eighth of the work, and this kernel's
// cost is dominated by how many times a block re-reads the cache, which is a
// function of exactly the token count the small cases shrink.
static struct gk_tensor * b_fa_h3(struct gk_ctx * c) { return fattn_square_d(c, 8742, 56, 128); }

// Z-Image's self-attention: 1056 tokens, 30 heads of 128.
static struct gk_tensor * b_fa_zimg(struct gk_ctx * c) { return fattn_square_d(c, 1056, 30, 128); }

static struct gk_tensor * b_fa_dit_1k(struct gk_ctx * c) { return fattn_square(c, 1024, 16); }
static struct gk_tensor * b_fa_dit_2k(struct gk_ctx * c) { return fattn_square(c, 2048, 16); }
static struct gk_tensor * b_fa_dit_4k(struct gk_ctx * c) { return fattn_square(c, 4096, 16); }

static struct gk_tensor * b_softmax_dec(struct gk_ctx * ctx) {
    struct gk_tensor * kq = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2048, N_DECODE, N_HEAD, 1);
    struct gk_tensor * m  = gk_new_tensor_4d(ctx, GK_TYPE_F16, 2048, N_DECODE, 1, 1);
    gk_set_name(kq, "kq");
    gk_set_name(m, "mask");
    return gk_soft_max_ext(ctx, kq, m, 1.0f / sqrtf((float) HEAD_DIM), 0.0f);
}

static struct gk_tensor * b_softmax_pre(struct gk_ctx * ctx) {
    struct gk_tensor * kq = gk_new_tensor_4d(ctx, GK_TYPE_F32, 512, N_PREFILL, N_HEAD, 1);
    struct gk_tensor * m  = gk_new_tensor_4d(ctx, GK_TYPE_F16, 512, N_PREFILL, 1, 1);
    gk_set_name(kq, "kq");
    gk_set_name(m, "mask");
    return gk_soft_max_ext(ctx, kq, m, 1.0f / sqrtf((float) HEAD_DIM), 0.0f);
}

static struct gk_tensor * b_rope_pre(struct gk_ctx * ctx) {
    struct gk_tensor * a   = gk_new_tensor_3d(ctx, GK_TYPE_F32, HEAD_DIM, N_HEAD, N_PREFILL);
    struct gk_tensor * pos = gk_new_tensor_1d(ctx, GK_TYPE_I32, N_PREFILL);
    gk_set_name(a, "a");
    gk_set_name(pos, "pos");
    return gk_rope_ext(ctx, a, pos, NULL, HEAD_DIM, 0, 131072,
                       1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

static struct gk_tensor * b_rms_norm_pre(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, N_EMBD, N_PREFILL);
    gk_set_name(a, "a");
    return gk_rms_norm(ctx, a, 1e-6f);
}

static struct gk_tensor * b_silu_pre(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, N_FF, N_PREFILL);
    gk_set_name(a, "a");
    return gk_unary(ctx, a, GK_UNARY_OP_SILU);
}

static struct gk_tensor * b_add_pre(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, N_EMBD, N_PREFILL);
    struct gk_tensor * b = gk_new_tensor_2d(ctx, GK_TYPE_F32, N_EMBD, N_PREFILL);
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    return gk_add(ctx, a, b);
}

// The elementwise ops a DiT's rope actually runs, at MageFlow's shapes:
// {2, d_head/2, L, n_head} against a table that is the same over the heads.
// Two elements to a row, so which kernel takes this is worth an order of
// magnitude - the flat one indexes off the destination and does not care, the
// row-mapped one gives a 256-thread block two elements of work.
#define ROPE_L      2208
#define ROPE_HALF   64
#define ROPE_HEADS  24

static struct gk_tensor * b_rope_mul_bcast(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, ROPE_HALF, ROPE_L, ROPE_HEADS);
    struct gk_tensor * b = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, ROPE_HALF, ROPE_L, 1);
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    return gk_mul(ctx, a, b);
}

static struct gk_tensor * b_rope_add(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, ROPE_HALF, ROPE_L, ROPE_HEADS);
    struct gk_tensor * b = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, ROPE_HALF, ROPE_L, ROPE_HEADS);
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    return gk_add(ctx, a, b);
}

// The same tensor with a strided innermost axis, which no flat kernel may
// take: this is the row-mapped path against the fully general one, and where
// the crossover between them sits decides GK_CU_ROWS_MIN.
static struct gk_tensor * b_rope_mul_strided(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_4d(ctx, GK_TYPE_F32, 4, ROPE_HALF, ROPE_L, ROPE_HEADS);
    struct gk_tensor * b = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, ROPE_HALF, ROPE_L, ROPE_HEADS);
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    struct gk_tensor * v = gk_view_4d(ctx, a, 2, ROPE_HALF, ROPE_L, ROPE_HEADS,
                                      a->nb[1], a->nb[2], a->nb[3], 2 * sizeof(float));
    return gk_mul(ctx, v, b);
}

// A 64-wide row, strided the same way: a UNet residual is this shape, and at
// 16 float4s a row it is the case either kernel could plausibly win.
static struct gk_tensor * b_res_mul_strided(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_4d(ctx, GK_TYPE_F32, 128, 64, 320, 4);
    struct gk_tensor * b = gk_new_tensor_4d(ctx, GK_TYPE_F32, 64, 64, 320, 4);
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    struct gk_tensor * v = gk_view_4d(ctx, a, 64, 64, 320, 4,
                                      a->nb[1], a->nb[2], a->nb[3], 64 * sizeof(float));
    return gk_mul(ctx, v, b);
}

// Narrower still: 32 and 16 elements to a row, so eight and four float4s.
// These bracket the crossover from above.
static struct gk_tensor * b_narrow32_mul_strided(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_4d(ctx, GK_TYPE_F32, 64, 128, 320, 4);
    struct gk_tensor * b = gk_new_tensor_4d(ctx, GK_TYPE_F32, 32, 128, 320, 4);
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    struct gk_tensor * v = gk_view_4d(ctx, a, 32, 128, 320, 4,
                                      a->nb[1], a->nb[2], a->nb[3], 32 * sizeof(float));
    return gk_mul(ctx, v, b);
}

static struct gk_tensor * b_narrow16_mul_strided(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_4d(ctx, GK_TYPE_F32, 32, 256, 320, 4);
    struct gk_tensor * b = gk_new_tensor_4d(ctx, GK_TYPE_F32, 16, 256, 320, 4);
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    struct gk_tensor * v = gk_view_4d(ctx, a, 16, 256, 320, 4,
                                      a->nb[1], a->nb[2], a->nb[3], 16 * sizeof(float));
    return gk_mul(ctx, v, b);
}

// The embedding lookup, which reads scattered rows out of the largest tensor
// in the model.
static struct gk_tensor * b_get_rows_pre(struct gk_ctx * ctx) {
    struct gk_tensor * w   = gk_new_tensor_2d(ctx, GK_TYPE_Q4_K, N_EMBD, N_VOCAB);
    struct gk_tensor * ids = gk_new_tensor_1d(ctx, GK_TYPE_I32, N_PREFILL);
    gk_set_name(w, "weight");
    gk_set_name(ids, "ids:262144");
    return gk_get_rows(ctx, w, ids);
}

// --------------------------------------------------------------------------
// the sort family
//
// Two regimes that look like one op. Expert routing sorts a row as wide as the
// expert count; sampling sorts a row as wide as the vocabulary. The device
// kernel switches strategy between them, so both are timed.
// --------------------------------------------------------------------------

static struct gk_tensor * b_top_k_moe(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, N_PREFILL);
    gk_set_name(a, "logits");
    return gk_top_k(ctx, a, 8);
}

static struct gk_tensor * b_top_k_4k(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 4096, 1);
    gk_set_name(a, "logits");
    return gk_top_k(ctx, a, 40);
}

// Deliberately one step past the bitonic path's limit, where the device falls
// back to ranking every element against every other. The cost of that fallback
// grows with the square of the row, so this case is the one to extrapolate
// from - a vocabulary row is 64x wider and 4096x more work.
static struct gk_tensor * b_top_k_8k(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 8192, 1);
    gk_set_name(a, "logits");
    return gk_top_k(ctx, a, 40);
}

// A vocabulary row, which is what the backend sampler's top_k sees. Before the
// rounds this shape was not measurable - the fallback's cost grows with the
// square of the row, and 262144 would have taken tens of seconds per call.
static struct gk_tensor * b_top_k_vocab(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 262144, 1);
    gk_set_name(a, "logits");
    return gk_top_k(ctx, a, 40);
}

static struct gk_tensor * b_argsort_moe(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, N_PREFILL);
    gk_set_name(a, "logits");
    return gk_argsort(ctx, a, GK_SORT_ORDER_DESC);
}

// A whole permutation of a row too wide for one network, which is what the
// backend sampler's top-p needs: it sorts the logits, sorts the candidates by
// the same order, and walks the cumulative distribution.
static struct gk_tensor * b_argsort_8k(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 8192, 1);
    gk_set_name(a, "logits");
    return gk_argsort(ctx, a, GK_SORT_ORDER_DESC);
}

static struct gk_tensor * b_argsort_vocab(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 262144, 1);
    gk_set_name(a, "logits");
    return gk_argsort(ctx, a, GK_SORT_ORDER_DESC);
}

// --------------------------------------------------------------------------
// ops with no device kernel
//
// Everything below is refused by the device backend today, so the harness
// reports the CPU cost and marks it. These are the candidates for new kernels,
// and the CPU column is what a kernel would be replacing.
// --------------------------------------------------------------------------

// The backend sampler's two ops, on a vocabulary row. Both run per token when
// --backend-sampling is on, which is the whole reason they matter.
static struct gk_tensor * b_argmax_vocab(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, N_VOCAB, 1);
    gk_set_name(a, "logits");
    return gk_argmax(ctx, a);
}

static struct gk_tensor * b_cumsum_vocab(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, N_VOCAB, 1);
    gk_set_name(a, "probs");
    return gk_cumsum(ctx, a);
}

// MobileNetV5's depthwise stage, which is Gemma 3n's vision tower: a 3x3
// depthwise convolution over a 128x128 feature map with 512 channels.
static struct gk_tensor * b_conv_2d_dw(struct gk_ctx * ctx) {
    struct gk_tensor * kern = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 1, 512);
    struct gk_tensor * img  = gk_new_tensor_4d(ctx, GK_TYPE_F32, 128, 128, 512, 1);
    gk_set_name(kern, "kernel");
    gk_set_name(img, "img");
    return gk_conv_2d_dw(ctx, kern, img, 1, 1, 1, 1, 1, 1);
}

// The same convolution through the direct kernel rather than the im2col one,
// which is the path the vision builders actually take.
static struct gk_tensor * b_conv_2d_dw_direct(struct gk_ctx * ctx) {
    struct gk_tensor * kern = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 1, 512);
    struct gk_tensor * img  = gk_new_tensor_4d(ctx, GK_TYPE_F32, 128, 128, 512, 1);
    gk_set_name(kern, "kernel");
    gk_set_name(img, "img");
    return gk_conv_2d_dw_direct(ctx, kern, img, 1, 1, 1, 1, 1, 1);
}

static struct gk_tensor * b_pad_reflect_1d(struct gk_ctx * ctx) {
    struct gk_tensor * a = gk_new_tensor_2d(ctx, GK_TYPE_F32, 3000, 128);
    gk_set_name(a, "a");
    return gk_pad_reflect_1d(ctx, a, 32, 32);
}

// A video patch embedding: 16 frames of 224x224 in 3 channels, unrolled
// against a 2x14x14 kernel.
static struct gk_tensor * b_im2col_3d(struct gk_ctx * ctx) {
    struct gk_tensor * kern = gk_new_tensor_4d(ctx, GK_TYPE_F16, 14, 14, 2, 3 * 64);
    struct gk_tensor * vol  = gk_new_tensor_4d(ctx, GK_TYPE_F32, 224, 224, 16, 3);
    gk_set_name(kern, "kernel");
    gk_set_name(vol, "vol");
    return gk_im2col_3d(ctx, kern, vol, 3, 14, 14, 2, 0, 0, 0, 1, 1, 1, GK_TYPE_F16);
}

// RWKV's recurrent core over a prefill batch: 32 heads of 64, which is the
// 7B configuration.
static struct gk_tensor * b_rwkv_wkv6(struct gk_ctx * ctx) {
    const int64_t S = 64, H = 32, T = N_PREFILL, n_seqs = 1;
    struct gk_tensor * k     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * v     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * r     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * tf    = gk_new_tensor_2d(ctx, GK_TYPE_F32, S, H);
    struct gk_tensor * td    = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * state = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    gk_set_name(k, "k");
    gk_set_name(v, "v");
    gk_set_name(r, "r");
    gk_set_name(tf, "tf");
    gk_set_name(td, "decay");
    gk_set_name(state, "state");
    return gk_rwkv_wkv6(ctx, k, v, r, tf, td, state);
}

static struct gk_tensor * b_rwkv_wkv7(struct gk_ctx * ctx) {
    const int64_t S = 64, H = 32, T = N_PREFILL, n_seqs = 1;
    struct gk_tensor * r     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * w     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * k     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * v     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * a     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * b     = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    struct gk_tensor * state = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    gk_set_name(r, "r");
    gk_set_name(w, "decay");
    gk_set_name(k, "k");
    gk_set_name(v, "v");
    gk_set_name(a, "a");
    gk_set_name(b, "b");
    gk_set_name(state, "state");
    return gk_rwkv_wkv7(ctx, r, w, k, v, a, b, state);
}

// Qwen3-Next's gated delta rule, over a prefill batch.
static struct gk_tensor * b_gated_delta_net(struct gk_ctx * ctx) {
    const int64_t S = 128, H = 16, T = N_PREFILL, n_seqs = 1;
    struct gk_tensor * q     = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);
    struct gk_tensor * k     = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);
    struct gk_tensor * v     = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);
    struct gk_tensor * g     = gk_new_tensor_4d(ctx, GK_TYPE_F32, 1, H, T, n_seqs);
    struct gk_tensor * beta  = gk_new_tensor_4d(ctx, GK_TYPE_F32, 1, H, T, n_seqs);
    struct gk_tensor * state = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, S, H, n_seqs);
    gk_set_name(q, "q");
    gk_set_name(k, "k");
    gk_set_name(v, "v");
    gk_set_name(g, "decay");
    gk_set_name(beta, "beta");
    gk_set_name(state, "state");
    return gk_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
}

// --------------------------------------------------------------------------

struct bench_case {
    const char * group;
    const char * name;
    const char * shape;
    build_fn     build;
    size_t       arena;   // 0 for the default
};

// A MiniMax-H3 output tile alone is 28672x8742 floats - one gigabyte - so the
// group below needs an arena the others would never touch.
#define ARENA_HUGE  ((size_t) 6144u << 20)
#define ARENA_BIG   ((size_t) 3072u << 20)
#define ARENA_MID   ((size_t) 1024u << 20)
#define ARENA_SMALL ((size_t)  256u << 20)

static const struct bench_case g_cases[] = {
    { "matmul (decode, 1 column)", "attn_q      q4_0", "1536x2048",       b_mm_q_dec,     ARENA_SMALL },
    { NULL,                        "attn_k/v    q4_0", "1536x256",        b_mm_kv_dec,    ARENA_SMALL },
    { NULL,                        "attn_out    q4_0", "2048x1536",       b_mm_out_dec,   ARENA_SMALL },
    { NULL,                        "ffn_gate/up q4_0", "1536x6144",       b_mm_gate_dec,  ARENA_SMALL },
    { NULL,                        "ffn_down    q4_1", "6144x1536",       b_mm_down_dec,  ARENA_SMALL },
    { NULL,                        "lm_head     q4_K", "1536x262144",     b_mm_lm_dec,    ARENA_BIG   },
    { NULL,                        "ffn_gate    f16",  "1536x6144",       b_mm_f16_dec,   ARENA_SMALL },

    { "decoder cost (one matmul shape, 1536x6144, 1 column)",
                       "f32   (no decode)", "18.9 MB",       b_dec_f32,   ARENA_SMALL },
    { NULL,            "f16   (no decode)", "18.9 MB",       b_dec_f16,   ARENA_SMALL },
    { NULL,            "q8_0",              "10.0 MB",       b_dec_q8_0,  ARENA_SMALL },
    { NULL,            "q4_0",              " 5.3 MB",       b_dec_q4_0,  ARENA_SMALL },
    { NULL,            "q4_K",              " 5.3 MB",       b_dec_q4_K,  ARENA_SMALL },
    { NULL,            "q6_K",              " 7.7 MB",       b_dec_q6_K,  ARENA_SMALL },
    { NULL,            "mxfp4",             " 5.0 MB",       b_dec_mxfp4, ARENA_SMALL },
    { NULL,            "nvfp4",             " 5.0 MB",       b_dec_nvfp4, ARENA_SMALL },
    { NULL,            "iq2_xxs",           " 2.4 MB",       b_dec_iq2_xxs, ARENA_SMALL },
    { NULL,            "iq2_s",             " 3.0 MB",       b_dec_iq2_s,   ARENA_SMALL },
    { NULL,            "iq3_xxs",           " 3.6 MB",       b_dec_iq3_xxs, ARENA_SMALL },
    { NULL,            "iq3_s",             " 4.1 MB",       b_dec_iq3_s,   ARENA_SMALL },

    { "decoder cost at a 30B FFN width (6656x19968, 1 column)",
                       "f16   (no decode)", "266 MB",        b_dec30_f16,     ARENA_BIG },
    { NULL,            "q4_K",              " 74 MB",        b_dec30_q4_K,    ARENA_BIG },
    { NULL,            "iq2_xxs",           " 34 MB",        b_dec30_iq2_xxs, ARENA_BIG },
    { NULL,            "iq2_s",             " 42 MB",        b_dec30_iq2_s,   ARENA_BIG },
    { NULL,            "iq3_xxs",           " 50 MB",        b_dec30_iq3_xxs, ARENA_BIG },
    { NULL,            "iq3_s",             " 56 MB",        b_dec30_iq3_s,   ARENA_BIG },
    { NULL,            "q6_K",              "109 MB",        b_dec30_q6_K,    ARENA_BIG },
    { NULL,            "q6_K lm_head 202k", "1.1 GB",        b_lm_q6_K,       ARENA_BIG },

    { "a 30B decode, verified 4 tokens at a time (speculative) against 1",
                       "ffn_gate iq2_s   x1", " 42 MB",       b_ver1_gate, ARENA_BIG },
    { NULL,            "ffn_gate iq2_s   x3", " 42 MB",       b_ver3_gate, ARENA_BIG },
    { NULL,            "ffn_gate iq2_s   x4", " 42 MB",       b_ver4_gate, ARENA_BIG },
    { NULL,            "ffn_down iq3_xxs x1", " 50 MB",       b_ver1_down, ARENA_BIG },
    { NULL,            "ffn_down iq3_xxs x4", " 50 MB",       b_ver4_down, ARENA_BIG },
    { NULL,            "attn_q   iq2_xxs x1", "  8 MB",       b_ver1_q,    ARENA_BIG },
    { NULL,            "attn_q   iq2_xxs x4", "  8 MB",       b_ver4_q,    ARENA_BIG },
    { NULL,            "attn_out iq3_s   x1", " 12 MB",       b_ver1_o,    ARENA_BIG },
    { NULL,            "attn_out iq3_s   x4", " 12 MB",       b_ver4_o,    ARENA_BIG },
    { NULL,            "attn_k   q5_K    x1", "1.2 MB",       b_ver1_k,    ARENA_BIG },
    { NULL,            "attn_k   q5_K    x4", "1.2 MB",       b_ver4_k,    ARENA_BIG },
    { NULL,            "attn_v   iq3_xxs x1", "0.6 MB",       b_ver1_v,    ARENA_BIG },
    { NULL,            "attn_v   iq3_xxs x4", "0.6 MB",       b_ver4_v,    ARENA_BIG },
    { NULL,            "lm_head  q6_K    x1", "1.1 GB",       b_lm_q6_K,   ARENA_BIG },
    { NULL,            "lm_head  q6_K    x3", "1.1 GB",       b_ver3_lm,   ARENA_BIG },
    { NULL,            "lm_head  q6_K    x4", "1.1 GB",       b_ver4_lm,   ARENA_BIG },

    { "a DFlash draft's nvfp4 mat-vecs (4 columns)",
                       "kv    6656->1024",  "  3.8 MB",      b_dft_kv,  ARENA_BIG },
    { NULL,            "o     4096->6656",  " 15.3 MB",      b_dft_o,   ARENA_BIG },
    { NULL,            "fc   33280->6656",  "124.6 MB",      b_dft_fc,  ARENA_BIG },
    { NULL,            "ffn   6656->19968", " 74.8 MB",      b_dft_ffn, ARENA_BIG },

    { "decoder cost at a batch (1536x6144, 512 columns)",
                       "f16   (no decode)", "18.9 MB",       b_pre_f16,   ARENA_MID },
    { NULL,            "q8_0",              "10.0 MB",       b_pre_q8_0,  ARENA_MID },
    { NULL,            "q4_0",              " 5.3 MB",       b_pre_q4_0,  ARENA_MID },
    { NULL,            "q4_K",              " 5.3 MB",       b_pre_q4_K,  ARENA_MID },
    { NULL,            "mxfp4",             " 5.0 MB",       b_pre_mxfp4, ARENA_MID },
    { NULL,            "nvfp4",             " 5.0 MB",       b_pre_nvfp4, ARENA_MID },
    { NULL,            "iq2_xxs",           " 2.4 MB",       b_pre_iq2_xxs, ARENA_MID },
    { NULL,            "iq2_s",             " 3.0 MB",       b_pre_iq2_s,   ARENA_MID },

    { "SD UNet convolutions (im2col GEMM, f16 x f16; ggml's cuBLAS time in the last column)",
                       "l1  3x3 320->320",  "k=2880  n=4096x320",  b_conv_l1_320,  ARENA_BIG },
    { NULL,            "l1  3x3 640->320",  "k=5760  n=4096x320",  b_conv_l1_640,  ARENA_BIG },
    { NULL,            "l1  3x3 960->320",  "k=8640  n=4096x320",  b_conv_l1_960,  ARENA_BIG },
    { NULL,            "l1  3x3 640->640",  "k=5760  n=4096x640",  b_conv_l1_down, ARENA_BIG },
    { NULL,            "l2  3x3 640->640",  "k=5760  n=1024x640",  b_conv_l2_640,  ARENA_BIG },
    { NULL,            "l3  3x3 1280->1280","k=11520 n=256x1280",  b_conv_l3_1280, ARENA_BIG },
    { NULL,            "l4  3x3 1280->1280","k=11520 n=64x1280",   b_conv_l4_1280, ARENA_BIG },

    { "SD UNet matmuls (nvfp4 against its f16 ceiling)",
                       "l1 attn  nvfp4", "320x320 n=4096",    b_sd_l1_attn_nv,  ARENA_MID },
    { NULL,            "l1 attn  f16",   "320x320 n=4096",    b_sd_l1_attn_f16, ARENA_MID },
    { NULL,            "l1 ff    nvfp4", "320x2560 n=4096",   b_sd_l1_ff_nv,    ARENA_BIG },
    { NULL,            "l1 ff    f16",   "320x2560 n=4096",   b_sd_l1_ff_f16,   ARENA_BIG },
    { NULL,            "l1 out   nvfp4", "1280x320 n=4096",   b_sd_l1_out_nv,   ARENA_MID },
    { NULL,            "l1 out   f16",   "1280x320 n=4096",   b_sd_l1_out_f16,  ARENA_MID },
    { NULL,            "l2 attn  nvfp4", "640x640 n=1024",    b_sd_l2_attn_nv,  ARENA_MID },
    { NULL,            "l2 attn  f16",   "640x640 n=1024",    b_sd_l2_attn_f16, ARENA_MID },
    { NULL,            "l2 ff    nvfp4", "640x5120 n=1024",   b_sd_l2_ff_nv,    ARENA_BIG },
    { NULL,            "l2 ff    f16",   "640x5120 n=1024",   b_sd_l2_ff_f16,   ARENA_BIG },
    { NULL,            "l3 ff    nvfp4", "1280x10240 n=256",  b_sd_l3_ff_nv,    ARENA_BIG },
    { NULL,            "l3 ff    f16",   "1280x10240 n=256",  b_sd_l3_ff_f16,   ARENA_BIG },
    { NULL,            "l3 out   nvfp4", "5120x1280 n=256",   b_sd_l3_out_nv,   ARENA_BIG },
    { NULL,            "l3 out   f16",   "5120x1280 n=256",   b_sd_l3_out_f16,  ARENA_BIG },

    { "MageFlow DiT matmuls (nvfp4, against f16 and q4_K at the same shape)",
                       "attn proj nvfp4", "3072x3072 n=2048",   b_dit_proj_nv,  ARENA_BIG },
    { NULL,            "attn proj f16",   "3072x3072 n=2048",   b_dit_proj_f16, ARENA_BIG },
    { NULL,            "attn proj q4_K",  "3072x3072 n=2048",   b_dit_proj_q4k, ARENA_BIG },
    { NULL,            "ff up     nvfp4", "3072x12288 n=2048",  b_dit_ffup_nv,  ARENA_BIG },
    { NULL,            "ff up     f16",   "3072x12288 n=2048",  b_dit_ffup_f16, ARENA_BIG },
    { NULL,            "ff up     q4_K",  "3072x12288 n=2048",  b_dit_ffup_q4k, ARENA_BIG },
    { NULL,            "ff down   nvfp4", "12288x3072 n=2048",  b_dit_ffdn_nv,  ARENA_BIG },
    { NULL,            "ff down   f16",   "12288x3072 n=2048",  b_dit_ffdn_f16, ARENA_BIG },
    { NULL,            "ff down   q4_K",  "12288x3072 n=2048",  b_dit_ffdn_q4k, ARENA_BIG },

    { "MiniMax-H3 video DiT matmuls (nvfp4, against f16 and q4_K at the same shape)",
                       "qkv       nvfp4", "5376x28672 n=8742",  b_h3_qkv_nv,   ARENA_HUGE },
    { NULL,            "qkv       f16",   "5376x28672 n=8742",  b_h3_qkv_f16,  ARENA_HUGE },
    { NULL,            "qkv       q4_K",  "5376x28672 n=8742",  b_h3_qkv_q4k,  ARENA_HUGE },
    { NULL,            "proj      nvfp4", "5376x21504 n=8742",  b_h3_proj_nv,  ARENA_HUGE },
    { NULL,            "proj      f16",   "5376x21504 n=8742",  b_h3_proj_f16, ARENA_HUGE },
    { NULL,            "proj      q4_K",  "5376x21504 n=8742",  b_h3_proj_q4k, ARENA_HUGE },
    { NULL,            "ff down   nvfp4", "14336x5376 n=8742",  b_h3_ffdn_nv,  ARENA_HUGE },
    { NULL,            "ff down   f16",   "14336x5376 n=8742",  b_h3_ffdn_f16, ARENA_HUGE },
    { NULL,            "ff down   q4_K",  "14336x5376 n=8742",  b_h3_ffdn_q4k, ARENA_HUGE },
    { NULL,            "ff out    nvfp4", "7168x5376 n=8742",   b_h3_ffo_nv,   ARENA_HUGE },
    { NULL,            "ff out    f16",   "7168x5376 n=8742",   b_h3_ffo_f16,  ARENA_HUGE },
    { NULL,            "ff out    q4_K",  "7168x5376 n=8742",   b_h3_ffo_q4k,  ARENA_HUGE },

    { "krea2 DiT matmuls (q2_K, against f16 and q4_K at the same shape)",
                       "ff up     q2_K",  "6144x16384 n=1038",  b_k2_ffup_q2k, ARENA_BIG },
    { NULL,            "ff up     f16",   "6144x16384 n=1038",  b_k2_ffup_f16, ARENA_BIG },
    { NULL,            "ff up     q4_K",  "6144x16384 n=1038",  b_k2_ffup_q4k, ARENA_BIG },
    { NULL,            "ff down   q2_K",  "16384x6144 n=1038",  b_k2_ffdn_q2k, ARENA_BIG },
    { NULL,            "ff down   f16",   "16384x6144 n=1038",  b_k2_ffdn_f16, ARENA_BIG },
    { NULL,            "ff down   q4_K",  "16384x6144 n=1038",  b_k2_ffdn_q4k, ARENA_BIG },
    { NULL,            "attn proj q2_K",  "6144x6144 n=1038",   b_k2_proj_q2k, ARENA_BIG },
    { NULL,            "attn proj f16",   "6144x6144 n=1038",   b_k2_proj_f16, ARENA_BIG },
    { NULL,            "attn proj q4_K",  "6144x6144 n=1038",   b_k2_proj_q4k, ARENA_BIG },
    { NULL,            "mod       q2_K",  "6144x1536 n=1038",   b_k2_mod_q2k,  ARENA_BIG },
    { NULL,            "mod       f16",   "6144x1536 n=1038",   b_k2_mod_f16,  ARENA_BIG },
    { NULL,            "mod       q4_K",  "6144x1536 n=1038",   b_k2_mod_q4k,  ARENA_BIG },

    { "Z-Image DiT matmuls (nvfp4, against f16 and q4_K at the same shape)",
                       "qkv       nvfp4", "3840x11520 n=1056",  b_zi_qkv_nv,   ARENA_BIG },
    { NULL,            "qkv       f16",   "3840x11520 n=1056",  b_zi_qkv_f16,  ARENA_BIG },
    { NULL,            "qkv       q4_K",  "3840x11520 n=1056",  b_zi_qkv_q4k,  ARENA_BIG },
    { NULL,            "proj      nvfp4", "3840x3840 n=1056",   b_zi_proj_nv,  ARENA_BIG },
    { NULL,            "proj      f16",   "3840x3840 n=1056",   b_zi_proj_f16, ARENA_BIG },
    { NULL,            "proj      q4_K",  "3840x3840 n=1056",   b_zi_proj_q4k, ARENA_BIG },
    { NULL,            "ff up     nvfp4", "3840x10240 n=1056",  b_zi_ffup_nv,  ARENA_BIG },
    { NULL,            "ff up     f16",   "3840x10240 n=1056",  b_zi_ffup_f16, ARENA_BIG },
    { NULL,            "ff up     q4_K",  "3840x10240 n=1056",  b_zi_ffup_q4k, ARENA_BIG },
    { NULL,            "ff down   nvfp4", "10240x3840 n=1056",  b_zi_ffdn_nv,  ARENA_BIG },
    { NULL,            "ff down   f16",   "10240x3840 n=1056",  b_zi_ffdn_f16, ARENA_BIG },
    { NULL,            "ff down   q4_K",  "10240x3840 n=1056",  b_zi_ffdn_q4k, ARENA_BIG },

    { "MiniMax-H3 video VAE decoder matmuls (36 layers x 28 tiles a decode)",
                       "ff up     f16",   "2048x16384 n=1797",  b_vae_ffup,    ARENA_BIG  },
    { NULL,            "ff down   f16",   "8192x2048 n=1797",   b_vae_ffdn,    ARENA_BIG  },
    { NULL,            "qkv       f16",   "2048x6144 n=1797",   b_vae_qkv,     ARENA_BIG  },
    { NULL,            "out proj  f16",   "2048x2048 n=1797",   b_vae_out,     ARENA_BIG  },
    { NULL,            "attn KQ   f32",   "64x1797 n=1797 x32", b_vae_kq,      ARENA_BIG  },
    { NULL,            "attn KQV  f32",   "1797x64 n=1797 x32", b_vae_kqv,     ARENA_BIG  },
    { NULL,            "attn KQ   f16",   "64x1797 n=1797 x32", b_vae_kq_f16,  ARENA_BIG  },
    { NULL,            "attn KQV  f16",   "1797x64 n=1797 x32", b_vae_kqv_f16, ARENA_BIG  },

    { "matmul (prefill, 512 columns)", "ffn_gate/up q4_0", "1536x6144",   b_mm_gate_pre,  ARENA_MID   },
    { NULL,                            "ffn_down    q4_1", "6144x1536",   b_mm_down_pre,  ARENA_MID   },
    { NULL,                            "ffn_gate    f16",  "1536x6144",   b_mm_f16_pre,   ARENA_MID   },
    { NULL,                            "ffn_gate    f32",  "1536x6144",   b_mm_f32_pre,   ARENA_MID   },

    { "attention", "flash_attn decode",  "n_kv=512",       b_fa_dec_512,  ARENA_SMALL },
    { NULL,        "flash_attn decode",  "n_kv=2048",      b_fa_dec_2k,   ARENA_SMALL },
    { NULL,        "flash_attn decode",  "n_kv=8192",      b_fa_dec_8k,   ARENA_MID   },
    { NULL,        "flash_attn prefill", "n_kv=512 nb=512", b_fa_pre_512, ARENA_MID   },
    { NULL,        "flash_attn DiT",     "1024 tok x16",   b_fa_dit_1k,   ARENA_MID   },
    { NULL,        "flash_attn DiT",     "2048 tok x16",   b_fa_dit_2k,   ARENA_BIG   },
    { NULL,        "flash_attn DiT",     "4096 tok x16",   b_fa_dit_4k,   ARENA_BIG   },
    { NULL,        "flash_attn DiT d128","2048 tok x16",   b_fa_dit_2k_d128, ARENA_BIG },
    { NULL,        "flash_attn DiT d128","4096 tok x16",   b_fa_dit_4k_d128, ARENA_BIG },
    { NULL,        "flash_attn H3 d128", "8742 tok x56",   b_fa_h3,          ARENA_HUGE },
    { NULL,        "flash_attn ZImg d128","1056 tok x30",  b_fa_zimg,        ARENA_BIG  },
    { NULL,        "soft_max decode",    "2048x1x8",       b_softmax_dec, ARENA_SMALL },
    { NULL,        "soft_max prefill",   "512x512x8",      b_softmax_pre, ARENA_MID   },

    { "elementwise and norms", "rope     prefill", "256x8x512",  b_rope_pre,     ARENA_SMALL },
    { NULL,                    "rms_norm prefill", "1536x512",   b_rms_norm_pre, ARENA_SMALL },
    { NULL,                    "silu     prefill", "6144x512",   b_silu_pre,     ARENA_SMALL },
    { NULL,                    "add      prefill", "1536x512",   b_add_pre,      ARENA_SMALL },
    { NULL,                    "get_rows q4_K",    "1536x512",   b_get_rows_pre, ARENA_BIG   },

    { "rope-shaped elementwise (2-element rows)",
                    "mul  bcast over heads", "2x64x2208x24",  b_rope_mul_bcast,   ARENA_BIG },
    { NULL,         "add  same shape",       "2x64x2208x24",  b_rope_add,         ARENA_BIG },
    { NULL,         "mul  strided rows",     "2x64x2208x24",  b_rope_mul_strided, ARENA_BIG },
    { NULL,         "mul  strided rows",     "64x64x320x4",   b_res_mul_strided,  ARENA_BIG },
    { NULL,         "mul  strided rows",     "32x128x320x4",  b_narrow32_mul_strided, ARENA_BIG },
    { NULL,         "mul  strided rows",     "16x256x320x4",  b_narrow16_mul_strided, ARENA_BIG },

    { "sort and select", "top_k   moe",   "128x512 k=8",  b_top_k_moe,   ARENA_SMALL },
    { NULL,              "top_k   4096",  "4096 k=40",    b_top_k_4k,    ARENA_SMALL },
    { NULL,              "top_k   8192",  "8192 k=40",    b_top_k_8k,    ARENA_SMALL },
    { NULL,              "top_k   vocab", "262144 k=40",  b_top_k_vocab, ARENA_MID   },
    { NULL,              "argsort moe",   "128x512",      b_argsort_moe,   ARENA_SMALL },
    { NULL,              "argsort 8192",  "8192",         b_argsort_8k,    ARENA_SMALL },
    { NULL,              "argsort vocab", "262144",       b_argsort_vocab, ARENA_MID   },

    { "convolution, recurrences and the sampler's helpers",
                                "argmax        vocab", "262144",         b_argmax_vocab,       ARENA_SMALL },
    { NULL,                     "cumsum        vocab", "262144",         b_cumsum_vocab,       ARENA_SMALL },
    { NULL,                     "conv_2d_dw",          "128x128x512 3x3", b_conv_2d_dw,        ARENA_MID   },
    { NULL,                     "conv_2d_dw_direct",   "128x128x512 3x3", b_conv_2d_dw_direct, ARENA_MID   },
    { NULL,                     "pad_reflect_1d",      "3000x128",       b_pad_reflect_1d,     ARENA_SMALL },
    { NULL,                     "im2col_3d",           "224x224x16x3",   b_im2col_3d,          ARENA_BIG   },
    { NULL,                     "rwkv_wkv6",           "S=64 H=32 T=512", b_rwkv_wkv6,         ARENA_MID   },
    { NULL,                     "rwkv_wkv7",           "S=64 H=32 T=512", b_rwkv_wkv7,         ARENA_MID   },
    { NULL,                     "gated_delta_net",     "S=128 H=16 T=512", b_gated_delta_net,  ARENA_MID   },
};

// --------------------------------------------------------------------------
// running one case
// --------------------------------------------------------------------------

struct run_result {
    double  seconds;      // per graph evaluation; < 0 when it did not run
    bool    unsupported;  // the backend has no kernel for some node
    double  traffic;      // see below
};

// The rate column is the graph's tensors divided by the time, which is an
// upper bound on what the kernel moved rather than a measurement of it. For
// the ops that read all of their operands - every matmul, every elementwise
// pass - the bound is tight and the number is the one to compare against the
// card's memory bandwidth. For an op that reads a few rows out of a large
// table, get_rows being the one here, it is meaningless: the weight counts in
// full and the kernel touched a thousandth of it. Read the milliseconds there.

// The budget each case gets. Long enough that the launch overhead and the
// clock's resolution are both noise, short enough that the whole table
// finishes while you are still looking at it.
#define BUDGET_SEC 0.25

// A single evaluation this slow means the case is pathological rather than
// merely expensive, and repeating it would only make you wait. The O(n^2)
// sort fallback is the one that trips this.
#define SLOW_SEC 0.5

static struct run_result run_case(const struct bench_case * bc, gk_backend_t backend,
                                  bool is_gpu) {
    struct run_result res = { -1.0, false, 0.0 };

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = bc->arena ? bc->arena : ARENA_SMALL,
        .mem_buffer = NULL, .no_alloc = true,
    });
    if (ctx == NULL) {
        return res;
    }

    struct gk_tensor * out = bc->build(ctx);
    gk_set_output(out);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    // Asked before anything is allocated, because the answer decides whether
    // there is anything to time. A node the backend refuses is not an error
    // here - it is the result.
    const int n_nodes = gk_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        if (!gk_backend_supports_op(backend, gk_graph_node(graph, i))) {
            res.unsupported = true;
            gk_free(ctx);
            return res;
        }
    }

    struct gk_gallocr * alloc =
        gk_gallocr_new(gk_backend_get_default_buffer_type(backend));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, graph)) {
        fprintf(stderr, "  %s: allocation failed\n", bc->name);
        gk_gallocr_free(alloc);
        gk_free(ctx);
        return res;
    }

    const int n_leafs = gk_graph_n_leafs(graph);
    for (int i = 0; i < n_leafs; ++i) {
        struct gk_tensor * leaf = gk_graph_leaf(graph, i);
        fill_tensor(leaf);
        res.traffic += (double) gk_nbytes(leaf);
    }
    for (int i = 0; i < n_nodes; ++i) {
        res.traffic += (double) gk_nbytes(gk_graph_node(graph, i));
    }

    // One untimed pass: the first launch of a kernel pays for its module load,
    // and the CPU pass pays for its threads starting.
    if (gk_backend_graph_compute(backend, graph) != GK_STATUS_SUCCESS) {
        fprintf(stderr, "  %s: graph execution failed\n", bc->name);
        gk_gallocr_free(alloc);
        gk_free(ctx);
        return res;
    }

    // GK_BENCH_COLD: time single passes with the cache evicted between them.
    //
    // The loop below queues a graph back to back and divides, which is the
    // right way to time a kernel and the wrong way to predict one. A model
    // runs each of its matmuls once against weights nothing has touched; this
    // harness runs one matmul many times against weights that after the first
    // pass are sitting in L2. Where a kernel's cost is its arithmetic the two
    // agree. Where it is its memory traffic they can differ by more than an
    // order of magnitude, and it is exactly then that the harness is most
    // confidently wrong - so the eviction is available on demand.
    if (is_gpu && getenv("GK_BENCH_COLD") != NULL) {
        const size_t flush_bytes = (size_t) 128u << 20;   // comfortably past L2
        void *       flush       = NULL;

        gk_backend_buffer_t fb = gk_backend_buft_alloc_buffer(
            gk_backend_get_default_buffer_type(backend), flush_bytes);

        double total = 0.0;
        const int cold_iters = 5;

        for (int i = 0; i < cold_iters; ++i) {
            if (fb != NULL) {
                gk_backend_buffer_clear(fb, (uint8_t) (i + 1));
                gk_backend_synchronize(backend);
            }
            const double c0 = now_sec();
            gk_backend_graph_compute(backend, graph);
            total += now_sec() - c0;
        }

        if (fb != NULL) {
            gk_backend_buffer_free(fb);
        }
        (void) flush;

        res.seconds = total / cold_iters;
        gk_gallocr_free(alloc);
        gk_free(ctx);
        return res;
    }

    double t0 = now_sec();
    gk_backend_graph_compute(backend, graph);
    const double one = now_sec() - t0;

    if (one > SLOW_SEC) {
        res.seconds = one;
    } else {
        int iters = (int) (BUDGET_SEC / (one > 1e-9 ? one : 1e-9));
        if (iters < 3)     { iters = 3; }
        if (iters > 100000) { iters = 100000; }

        if (is_gpu) {
            // Queued back to back with one wait at the end. A wait per
            // evaluation would measure the round trip to the driver as often
            // as it measured the kernel, which for the small ops here is most
            // of the number.
            t0 = now_sec();
            for (int i = 0; i < iters; ++i) {
                gk_backend_graph_compute_async(backend, graph);
            }
            gk_backend_synchronize(backend);
            res.seconds = (now_sec() - t0) / iters;
        } else {
            // The CPU pass has no queue, so the fastest of several passes is
            // the honest number: the slow ones are the scheduler, not the op.
            double best = 1e30;
            for (int i = 0; i < iters; ++i) {
                t0 = now_sec();
                gk_backend_graph_compute(backend, graph);
                const double dt = now_sec() - t0;
                if (dt < best) {
                    best = dt;
                }
            }
            res.seconds = best;
        }
    }

    gk_gallocr_free(alloc);
    gk_free(ctx);
    return res;
}

// --------------------------------------------------------------------------
// the verdict
//
// The table is the measurement; this is the reading of it. A kernel that
// exists but loses to the CPU is a different problem from one that does not
// exist, and both are different from one that is merely unexciting, so they
// are collected separately and each sorted by what it would be worth to fix.
// --------------------------------------------------------------------------

struct verdict {
    const char * name;
    const char * shape;
    double       gpu_ms;
    double       cpu_ms;
    double       ratio;   // cpu/gpu; how many times the device wins
    bool         missing;
};

static struct verdict g_verdicts[sizeof(g_cases) / sizeof(g_cases[0])];
static int g_n_verdicts = 0;

static int by_cpu_ms_desc(const void * a, const void * b) {
    const double x = ((const struct verdict *) a)->cpu_ms;
    const double y = ((const struct verdict *) b)->cpu_ms;
    return x < y ? 1 : x > y ? -1 : 0;
}

static int by_ratio_asc(const void * a, const void * b) {
    const double x = ((const struct verdict *) a)->ratio;
    const double y = ((const struct verdict *) b)->ratio;
    return x > y ? 1 : x < y ? -1 : 0;
}

int main(int argc, char ** argv) {
    const char * filter = argc > 1 ? argv[1] : NULL;

    gk_device_t device = gk_device_by_type(GK_DEVICE_TYPE_GPU);
    if (device == NULL) {
        fprintf(stderr, "no GPU was discovered; this benchmark needs one\n");
        return 1;
    }

    gk_backend_t gpu = gk_device_init_backend(device);
    if (gpu == NULL) {
        fprintf(stderr, "failed to initialize a backend for %s\n", gk_device_name(device));
        return 1;
    }

    gk_backend_t cpu = gk_backend_cpu_init(0);
    if (cpu == NULL) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }

    printf("device: %s (%s)\n", gk_device_name(device), gk_device_description(device));
    printf("shapes: gemma-4-e2b  n_embd=%d n_ff=%d n_head=%d/%d head_dim=%d vocab=%d\n",
           N_EMBD, N_FF, N_HEAD, N_HEAD_KV, HEAD_DIM, N_VOCAB);
    printf("        prefill batch %d, decode batch %d\n", N_PREFILL, N_DECODE);
    printf("\n");
    printf("x        = how many times faster the device is than the CPU backend\n");
    printf("GB/s max = graph bytes over time; an upper bound, and only tight for\n");
    printf("           ops that read all of their operands (see get_rows)\n\n");

    const int n_cases = (int) (sizeof(g_cases) / sizeof(g_cases[0]));
    const char * group = NULL;

    for (int i = 0; i < n_cases; ++i) {
        const struct bench_case * bc = &g_cases[i];
        if (bc->group != NULL) {
            group = bc->group;
        }
        if (filter != NULL && strstr(bc->name, filter) == NULL &&
            (group == NULL || strstr(group, filter) == NULL)) {
            continue;
        }

        if (bc->group != NULL) {
            printf("%s\n", bc->group);
            printf("  %-22s %-18s %10s %10s %8s %10s\n",
                   "op", "shape", "GPU ms", "CPU ms", "x", "GB/s max");
        }

        const struct run_result g = run_case(bc, gpu, true);
        const struct run_result c = run_case(bc, cpu, false);

        char gpu_col[32];
        char ratio_col[32];
        char rate_col[32];

        if (g.unsupported) {
            snprintf(gpu_col,   sizeof(gpu_col),   "%10s", "none");
            snprintf(ratio_col, sizeof(ratio_col), "%8s",  "-");
            snprintf(rate_col,  sizeof(rate_col),  "%10s", "-");
        } else if (g.seconds < 0.0) {
            snprintf(gpu_col,   sizeof(gpu_col),   "%10s", "fail");
            snprintf(ratio_col, sizeof(ratio_col), "%8s",  "-");
            snprintf(rate_col,  sizeof(rate_col),  "%10s", "-");
        } else {
            snprintf(gpu_col,   sizeof(gpu_col),   "%10.4f", g.seconds * 1e3);
            snprintf(rate_col,  sizeof(rate_col),  "%10.1f", g.traffic / g.seconds / 1e9);
            if (c.seconds > 0.0) {
                snprintf(ratio_col, sizeof(ratio_col), "%8.2f", c.seconds / g.seconds);
            } else {
                snprintf(ratio_col, sizeof(ratio_col), "%8s", "-");
            }
        }

        char cpu_col[32];
        if (c.seconds < 0.0) {
            snprintf(cpu_col, sizeof(cpu_col), "%10s", c.unsupported ? "none" : "fail");
        } else {
            snprintf(cpu_col, sizeof(cpu_col), "%10.4f", c.seconds * 1e3);
        }

        printf("  %-22s %-18s %s %s %s %s%s\n",
               bc->name, bc->shape, gpu_col, cpu_col, ratio_col, rate_col,
               g.seconds > SLOW_SEC ? "   (single pass; too slow to repeat)" : "");
        fflush(stdout); // a slow case should not look like a hang

        if (g_n_verdicts < n_cases && c.seconds > 0.0) {
            struct verdict * v = &g_verdicts[g_n_verdicts++];
            v->name    = bc->name;
            v->shape   = bc->shape;
            v->gpu_ms  = g.seconds * 1e3;
            v->cpu_ms  = c.seconds * 1e3;
            v->ratio   = g.seconds > 0.0 ? c.seconds / g.seconds : 0.0;
            v->missing = g.unsupported;
        }

        if (i + 1 < n_cases && g_cases[i + 1].group != NULL) {
            printf("\n");
        }
    }

    // ----------------------------------------------------------------------

    printf("\n");
    printf("========================================================================\n");
    printf("what to work on first\n");
    printf("========================================================================\n\n");

    struct verdict slow[sizeof(g_cases) / sizeof(g_cases[0])];
    struct verdict gone[sizeof(g_cases) / sizeof(g_cases[0])];
    int n_slow = 0, n_gone = 0;

    for (int i = 0; i < g_n_verdicts; ++i) {
        if (g_verdicts[i].missing) {
            gone[n_gone++] = g_verdicts[i];
        } else if (g_verdicts[i].ratio > 0.0) {
            slow[n_slow++] = g_verdicts[i];
        }
    }

    qsort(slow, (size_t) n_slow, sizeof(slow[0]), by_ratio_asc);
    qsort(gone, (size_t) n_gone, sizeof(gone[0]), by_cpu_ms_desc);

    printf("kernels that exist and are losing to the CPU, worst first.\n");
    printf("a ratio near or below 1 on a parallel op means the kernel's shape is\n");
    printf("wrong, not that it needs tuning.\n\n");
    printf("  %-22s %-18s %8s %10s %10s\n", "op", "shape", "x vs CPU", "GPU ms", "CPU ms");
    for (int i = 0; i < n_slow && i < 10; ++i) {
        printf("  %-22s %-18s %8.2f %10.4f %10.4f\n",
               slow[i].name, slow[i].shape, slow[i].ratio, slow[i].gpu_ms, slow[i].cpu_ms);
    }

    printf("\nops with no device kernel, by what the CPU spends on them.\n");
    printf("the real cost is higher: each one splits the graph and moves its\n");
    printf("operands across the bus in both directions.\n\n");
    if (n_gone == 0) {
        printf("  none - every op measured here has one.\n");
    } else {
        printf("  %-22s %-18s %10s\n", "op", "shape", "CPU ms");
        for (int i = 0; i < n_gone; ++i) {
            printf("  %-22s %-18s %10.4f\n", gone[i].name, gone[i].shape, gone[i].cpu_ms);
        }
    }
    printf("\n");

    gk_backend_free(cpu);
    gk_backend_free(gpu);
    return 0;
}
