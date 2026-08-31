// Differential tests against a reference tensor library.
//
// This is the harness the rest of the rewrite leans on. gk is an independent
// implementation, but the bytes it reads and writes are not free to differ:
// a GGUF file produced anywhere must decode here to the same numbers, and a
// tensor gk encodes must be readable by anything else that reads the format.
// Every check below is about that contract, not about matching another
// library's internals.
//
// The comparisons run in both directions on purpose. Agreeing with a reference
// decoder only proves the encoder is right if the decoder is independent of
// it, so each format is encoded on one side and decoded on the other, both
// ways round.
//
// Build with -DGK_REFERENCE_DIR=<path to a reference tree>.

#include "gk_impl.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fails  = 0;
static int g_checks = 0;

#define CHECK_MSG(cond, ...) \
    do { \
        g_checks++; \
        if (!(cond)) { \
            g_fails++; \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__); \
            printf("\n"); \
        } \
    } while (0)

// A deterministic generator, so a failure is reproducible from the seed alone.
static uint64_t g_rng = 0x853c49e6748fea9bull;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t x = (uint32_t) (g_rng >> 33);
    return (float) x / (float) 0x7fffffffu * 2.0f - 1.0f;
}

// Weight-like data: mostly small, with a few large outliers, which is the
// shape that separates a good scale search from a lazy one. Uniform noise
// would let a broken encoder look fine.
static void fill_weights(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        const float u = frand();
        x[i] = u * u * u; // cubed: concentrates near zero, keeps the tails
    }
    // a handful of genuine outliers
    for (int i = 0; i < n / 64 + 1; ++i) {
        const int j = (int) ((uint32_t) (g_rng >> 40) % (uint32_t) n);
        x[j] = frand() * 4.0f;
        g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    }
}

// The types both libraries carry. The ids are the format's, so they are equal
// on both sides by construction; this table is only the list of what to test.
static const int g_shared_types[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
    GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    GGML_TYPE_MXFP4,
};

#define N_SHARED (int) (sizeof(g_shared_types) / sizeof(g_shared_types[0]))

// --------------------------------------------------------------------------

// Block geometry is the format's own; if these disagree nothing else can be
// meaningful, so this runs first.
static void test_geometry(void) {
    printf("block geometry\n");

    for (int i = 0; i < N_SHARED; ++i) {
        const int t = g_shared_types[i];

        CHECK_MSG(gk_blck_size((enum gk_type) t) == ggml_blck_size((enum ggml_type) t),
                  "%s: block size %lld vs %lld", gk_type_name((enum gk_type) t),
                  (long long) gk_blck_size((enum gk_type) t),
                  (long long) ggml_blck_size((enum ggml_type) t));

        CHECK_MSG(gk_type_size((enum gk_type) t) == ggml_type_size((enum ggml_type) t),
                  "%s: type size %zu vs %zu", gk_type_name((enum gk_type) t),
                  gk_type_size((enum gk_type) t), ggml_type_size((enum ggml_type) t));

        const int64_t blck = gk_blck_size((enum gk_type) t);
        for (int64_t n = blck; n <= blck * 16; n += blck) {
            CHECK_MSG(gk_row_size((enum gk_type) t, n) == ggml_row_size((enum ggml_type) t, n),
                      "%s: row_size(%lld) %zu vs %zu", gk_type_name((enum gk_type) t),
                      (long long) n, gk_row_size((enum gk_type) t, n),
                      ggml_row_size((enum ggml_type) t, n));
        }
    }
}

// f16 and bf16 must agree bit for bit, not approximately: they are stored in
// files and a one-ulp difference in rounding compounds through a model.
static void test_narrow_floats(void) {
    printf("narrow float encodings\n");

    int f16_mismatch = 0;

    // every f16 bit pattern, decoded both ways
    for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
        const float a = gk_fp16_to_fp32((gk_fp16_t) bits);
        const float b = ggml_fp16_to_fp32((ggml_fp16_t) bits);

        const bool both_nan = isnan(a) && isnan(b);
        if (!both_nan && memcmp(&a, &b, sizeof(float)) != 0) {
            if (f16_mismatch++ < 4) {
                printf("  f16 0x%04x decodes to %g vs %g\n", bits, (double) a, (double) b);
            }
        }
    }
    CHECK_MSG(f16_mismatch == 0, "%d of 65536 f16 patterns decode differently", f16_mismatch);

    // encoding: sweep a wide range of f32 values through both
    int enc_mismatch = 0;
    for (int i = 0; i < 200000; ++i) {
        const float x = frand() * powf(10.0f, (float) (i % 12) - 6.0f);

        const uint16_t a = (uint16_t) gk_fp32_to_fp16(x);
        const uint16_t b = (uint16_t) ggml_fp32_to_fp16(x);

        if (a != b) {
            if (enc_mismatch++ < 4) {
                printf("  f16 encode of %g: 0x%04x vs 0x%04x\n", (double) x, a, b);
            }
        }
    }
    CHECK_MSG(enc_mismatch == 0, "%d f16 encodings differ", enc_mismatch);
}

// The interop contract, both directions:
//
//   encode here -> decode there    our writer produces bytes others can read
//   encode there -> decode here    our reader accepts bytes others produced
//
// The second direction is the one that matters for running published models,
// and it has to be exact - decoding is a pure function of the stored bits, so
// any difference at all is a bug, not a tolerance question.
static void test_codec_interop(void) {
    printf("codec interop\n");

    const int64_t n_per_row = 1024;
    const int64_t nrows     = 4;
    const int64_t n         = n_per_row * nrows;

    float * src    = (float *) malloc((size_t) n * sizeof(float));
    float * out_gk = (float *) malloc((size_t) n * sizeof(float));
    float * out_rf = (float *) malloc((size_t) n * sizeof(float));

    for (int i = 0; i < N_SHARED; ++i) {
        const int t = g_shared_types[i];
        const char * name = gk_type_name((enum gk_type) t);

        const size_t row_bytes = gk_row_size((enum gk_type) t, n_per_row);
        void * enc_gk = malloc(row_bytes * (size_t) nrows);
        void * enc_rf = malloc(row_bytes * (size_t) nrows);

        fill_weights(src, (int) n);

        // ---- encode on each side -------------------------------------------------
        const struct gk_type_traits * tr = gk_get_type_traits((enum gk_type) t);
        for (int64_t r = 0; r < nrows; ++r) {
            tr->from_float(src + r * n_per_row,
                           (char *) enc_gk + (size_t) r * row_bytes, n_per_row);
        }

        ggml_quantize_chunk((enum ggml_type) t, src, enc_rf, 0, nrows, n_per_row, NULL);

        // ---- cross-decode --------------------------------------------------------
        // our bytes, their decoder
        const struct ggml_type_traits * rtr = ggml_get_type_traits((enum ggml_type) t);
        for (int64_t r = 0; r < nrows; ++r) {
            rtr->to_float((char *) enc_gk + (size_t) r * row_bytes,
                          out_rf + r * n_per_row, (int) n_per_row);
        }
        // our bytes, our decoder
        for (int64_t r = 0; r < nrows; ++r) {
            tr->to_float((char *) enc_gk + (size_t) r * row_bytes,
                         out_gk + r * n_per_row, n_per_row);
        }

        int mismatch = 0;
        for (int64_t j = 0; j < n; ++j) {
            if (out_gk[j] != out_rf[j]) {
                mismatch++;
            }
        }
        CHECK_MSG(mismatch == 0,
                  "%s: bytes we wrote decode differently elsewhere (%d/%lld elements)",
                  name, mismatch, (long long) n);

        // their bytes, our decoder vs their decoder
        for (int64_t r = 0; r < nrows; ++r) {
            tr->to_float((char *) enc_rf + (size_t) r * row_bytes,
                         out_gk + r * n_per_row, n_per_row);
            rtr->to_float((char *) enc_rf + (size_t) r * row_bytes,
                          out_rf + r * n_per_row, (int) n_per_row);
        }

        mismatch = 0;
        for (int64_t j = 0; j < n; ++j) {
            if (out_gk[j] != out_rf[j]) {
                mismatch++;
            }
        }
        CHECK_MSG(mismatch == 0,
                  "%s: we decode foreign bytes differently (%d/%lld elements)",
                  name, mismatch, (long long) n);

        free(enc_gk);
        free(enc_rf);
    }

    free(src);
    free(out_gk);
    free(out_rf);
}

// Encoding quality. The two encoders are independent and will not agree
// element for element - they are solving the same fit with different searches.
// What must hold is that ours is not meaningfully worse, measured as the
// reconstruction error over the row.
static void test_encode_quality(void) {
    printf("encode quality\n");

    const int64_t n_per_row = 4096;

    float * src = (float *) malloc((size_t) n_per_row * sizeof(float));
    float * rec = (float *) malloc((size_t) n_per_row * sizeof(float));

    for (int i = 0; i < N_SHARED; ++i) {
        const int t = g_shared_types[i];
        const char * name = gk_type_name((enum gk_type) t);

        // named before any work, so a crash identifies the format
        printf("  %-8s ", name);
        fflush(stdout);

        const size_t row_bytes = gk_row_size((enum gk_type) t, n_per_row);
        void * enc = malloc(row_bytes);

        fill_weights(src, (int) n_per_row);

        double err_gk = 0.0;
        double err_rf = 0.0;
        double energy = 0.0;
        for (int64_t j = 0; j < n_per_row; ++j) {
            energy += (double) src[j] * src[j];
        }

        const struct gk_type_traits  * tr  = gk_get_type_traits((enum gk_type) t);
        const struct ggml_type_traits * rtr = ggml_get_type_traits((enum ggml_type) t);

        tr->from_float(src, enc, n_per_row);
        tr->to_float(enc, rec, n_per_row);
        for (int64_t j = 0; j < n_per_row; ++j) {
            const double d = (double) src[j] - rec[j];
            err_gk += d * d;
        }

        ggml_quantize_chunk((enum ggml_type) t, src, enc, 0, 1, n_per_row, NULL);
        rtr->to_float(enc, rec, (int) n_per_row);
        for (int64_t j = 0; j < n_per_row; ++j) {
            const double d = (double) src[j] - rec[j];
            err_rf += d * d;
        }

        const double rel_gk = sqrt(err_gk / energy);
        const double rel_rf = sqrt(err_rf / energy);

        printf("rel err  ours %.5f  reference %.5f\n", rel_gk, rel_rf);

        // 5% slack: the searches differ, and a format this coarse has real
        // run-to-run spread. A genuine regression shows up far larger.
        CHECK_MSG(rel_gk <= rel_rf * 1.05 + 1e-6,
                  "%s: our encoder is worse (%.5f vs %.5f)", name, rel_gk, rel_rf);

        free(enc);
    }

    free(src);
    free(rec);
}

// The dot product a matmul actually calls. Both sides are computing the same
// mathematical quantity from the same bytes, so they agree to float rounding;
// the tolerance is relative to the magnitude of the result.
static void test_vec_dot(void) {
    printf("quantized dot products\n");

    const int n = 4096;

    float * wf = (float *) malloc((size_t) n * sizeof(float));
    float * af = (float *) malloc((size_t) n * sizeof(float));

    for (int i = 0; i < N_SHARED; ++i) {
        const int t = g_shared_types[i];
        const char * name = gk_type_name((enum gk_type) t);

        printf("  %-8s ", name);
        fflush(stdout);

        const struct gk_type_traits      * tr  = gk_get_type_traits((enum gk_type) t);
        // the reference keeps its dot kernels in the CPU traits, not the base ones
        const struct ggml_type_traits_cpu * rtr = ggml_get_type_traits_cpu((enum ggml_type) t);

        if (tr->vec_dot == NULL || rtr->vec_dot == NULL) {
            printf("no dot on one side, skipped\n");
            continue;
        }

        fill_weights(wf, n);
        for (int j = 0; j < n; ++j) {
            af[j] = frand();
        }

        void * w = malloc(gk_row_size((enum gk_type) t, n));
        tr->from_float(wf, w, n);

        // Each side's dot takes its activations in whatever its own traits
        // name as `vec_dot_type` - f32 for a format still on the float path,
        // an integer type for one with a fast path. Driving each through its
        // own contract is the point; handing raw f32 to a dot that expects
        // q8_0 reads the scale bytes as data and produces nonsense.
        const enum gk_type our_vdt = tr->vec_dot_type;

        void * our_y = (void *) af;
        void * our_buf = NULL;

        if (our_vdt != GK_TYPE_F32) {
            const struct gk_type_traits * vtr = gk_get_type_traits(our_vdt);
            our_buf = malloc(gk_row_size(our_vdt, n));
            vtr->from_float(af, our_buf, n);
            our_y = our_buf;
        }

        float s_gk = 0.0f;
        tr->vec_dot(n, &s_gk, 0, w, 0, our_y, 0, 1);
        free(our_buf);

        // The activation encoder for an intermediate type like q8_K lives in
        // the reference's CPU traits; the base traits carry no encoder for it
        // at all, so ask the CPU side first.
        const enum ggml_type vdt = rtr->vec_dot_type;

        ggml_from_float_t to_vdt = ggml_get_type_traits_cpu(vdt)->from_float;
        if (to_vdt == NULL) {
            to_vdt = ggml_get_type_traits(vdt)->from_float_ref;
        }
        if (to_vdt == NULL) {
            printf("reference has no encoder for %s, skipped\n", ggml_type_name(vdt));
            free(w);
            continue;
        }

        void * ay = malloc(ggml_row_size(vdt, n));
        to_vdt(af, ay, n);

        float s_rf = 0.0f;
        rtr->vec_dot(n, &s_rf, 0, w, 0, ay, 0, 1);

        // The activation side is itself quantized on the reference path, so
        // the two results differ by that quantization, not just by rounding.
        // Scale the tolerance to the row's magnitude rather than the result,
        // which can sit near zero by cancellation.
        double mag = 0.0;
        for (int j = 0; j < n; ++j) {
            mag += fabs((double) wf[j] * af[j]);
        }

        const double tol = mag * 0.02;
        CHECK_MSG(fabs((double) s_gk - s_rf) <= tol,
                  "%s: dot %g vs %g (tolerance %g)", name,
                  (double) s_gk, (double) s_rf, tol);

        printf("dot  ours %+.4f  reference %+.4f  (tol %.4f)\n",
               (double) s_gk, (double) s_rf, tol);

        free(w);
        free(ay);
    }

    free(wf);
    free(af);
}

int main(void) {
    printf("gk differential tests against reference\n\n");

    // The reference builds its lookup tables - narrow-float conversion and the
    // lattice codebooks - inside its context init, and its dot kernels read
    // those tables directly. Without this they are all zero, which shows up as
    // dot products of exactly zero rather than as a crash, so it is worth
    // doing before anything is compared.
    struct ggml_init_params rp = {
        .mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true,
    };
    struct ggml_context * rctx = ggml_init(rp);
    if (rctx == NULL) {
        printf("could not initialise the reference library\n");
        return 1;
    }

    // The reference's CPU backend is a separate library with its own copy of
    // those tables, and its dot kernels read that copy. Initialising only the
    // core leaves it zeroed, which makes every quantized dot return exactly
    // zero - a silent wrong answer rather than a failure.
    ggml_cpu_init();

    test_geometry();
    test_narrow_floats();
    test_codec_interop();
    test_encode_quality();
    test_vec_dot();

    ggml_free(rctx);

    printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
