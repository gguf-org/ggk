// CUDA/ROCm backend smoke and numerical tests.
//
// The ordinary backend tests intentionally run without a GPU. This one is
// compiled only when the CUDA-family backend is enabled and exercises the
// complete device path: discovery, device allocation, host/device transfers,
// graph execution, synchronization, and result transfer back to the host.
//
// Quantized weights matter here. A float-only smoke test cannot catch a
// device decoder drifting from the on-disk GGUF block layout, which turns an
// otherwise healthy CUDA build into plausible-looking but incorrect output.
// Decoding is therefore checked bit-for-bit. Matmul is checked separately
// with a small numerical tolerance: like ggml CUDA, the device path and the
// CPU SIMD path reduce/quantize activations differently and are not expected
// to be bit-identical.

#include "gk_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const enum gk_type g_weight_types[] = {
    GK_TYPE_F32, GK_TYPE_F16, GK_TYPE_BF16,
    GK_TYPE_Q4_0, GK_TYPE_Q4_1, GK_TYPE_Q5_0, GK_TYPE_Q5_1, GK_TYPE_Q8_0,
    GK_TYPE_Q2_K, GK_TYPE_Q3_K, GK_TYPE_Q4_K, GK_TYPE_Q5_K, GK_TYPE_Q6_K,
    GK_TYPE_IQ4_NL, GK_TYPE_IQ4_XS,
    GK_TYPE_IQ1_S, GK_TYPE_IQ1_M,
    GK_TYPE_IQ2_XXS, GK_TYPE_IQ2_XS, GK_TYPE_IQ2_S,
    GK_TYPE_IQ3_XXS, GK_TYPE_IQ3_S,
    GK_TYPE_TQ1_0, GK_TYPE_TQ2_0,
    GK_TYPE_MXFP4, GK_TYPE_NVFP4, GK_TYPE_Q1_0, GK_TYPE_Q2_0,
};

static float input_value(int i) {
    return sinf((float) i * 0.071f) * 0.5f + cosf((float) i * 0.013f) * 0.25f;
}

static struct gk_tensor * build_graph(struct gk_ctx * ctx, enum gk_type weight_type,
                                      struct gk_tensor ** weight,
                                      struct gk_tensor ** input) {
    const int64_t k = 256;
    const int64_t m = 16;
    const int64_t n = 5;

    *weight = gk_new_tensor_2d(ctx, weight_type, k, m);
    *input  = gk_new_tensor_2d(ctx, GK_TYPE_F32, k, n);

    struct gk_tensor * out = gk_mul_mat(ctx, *weight, *input);
    gk_set_output(out);
    return out;
}

static void encode_weights(enum gk_type type, const float * src, void * dst,
                           int64_t k, int64_t rows) {
    const struct gk_type_traits * traits = gk_get_type_traits(type);
    const size_t row_bytes = gk_row_size(type, k);

    for (int64_t r = 0; r < rows; ++r) {
        traits->from_float(src + r * k, (char *) dst + r * row_bytes, k);
    }
}

static int run_decode_type(gk_backend_t gpu, enum gk_type type) {
    const int64_t k = 256;
    const int64_t rows = 4;

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * weight = gk_new_tensor_2d(ctx, type, k, rows);
    struct gk_tensor * ids = gk_new_tensor_1d(ctx, GK_TYPE_I32, rows);
    struct gk_tensor * out = gk_get_rows(ctx, weight, ids);
    gk_set_output(out);
    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    struct gk_gallocr * alloc =
        gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, graph)) {
        fprintf(stderr, "%s: failed to allocate decoder graph\n", gk_type_name(type));
        return 1;
    }

    const int64_t n = k * rows;
    float * src = (float *) malloc((size_t) n * sizeof(float));
    float * expected = (float *) malloc((size_t) n * sizeof(float));
    float * got = (float *) malloc((size_t) n * sizeof(float));
    void * encoded = malloc(gk_nbytes(weight));
    int32_t row_ids[4] = { 0, 1, 2, 3 };
    if (src == NULL || expected == NULL || got == NULL || encoded == NULL) {
        return 1;
    }
    for (int64_t i = 0; i < n; ++i) {
        src[i] = input_value((int) i) * 0.125f;
    }
    encode_weights(type, src, encoded, k, rows);
    const struct gk_type_traits * traits = gk_get_type_traits(type);
    const size_t row_bytes = gk_row_size(type, k);
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float((const char *) encoded + r * row_bytes, expected + r * k, k);
    }

    gk_backend_tensor_set(weight, encoded, 0, gk_nbytes(weight));
    gk_backend_tensor_set(ids, row_ids, 0, sizeof(row_ids));
    if (gk_backend_graph_compute(gpu, graph) != GK_STATUS_SUCCESS) {
        fprintf(stderr, "%s: decoder graph execution failed\n", gk_type_name(type));
        return 1;
    }
    gk_backend_synchronize(gpu);
    gk_backend_tensor_get(out, got, 0, (size_t) n * sizeof(float));

    int bad = 0;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float diff = fabsf(got[i] - expected[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (diff != 0.0f) {
            bad++;
        }
    }
    if (bad != 0) {
        printf("  %-8s decoder: max abs error %.8g, %d/%lld mismatches\n",
               gk_type_name(type), max_abs, bad, (long long) n);
    }

    free(encoded);
    free(got);
    free(expected);
    free(src);
    gk_gallocr_free(alloc);
    gk_free(ctx);
    return bad == 0 ? 0 : 1;
}

static int run_type(gk_backend_t gpu, enum gk_type weight_type) {
    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * gpu_w;
    struct gk_tensor * gpu_x;
    struct gk_tensor * gpu_out = build_graph(gpu_ctx, weight_type, &gpu_w, &gpu_x);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gpu_out);

    if (!gk_backend_supports_op(gpu, gpu_out)) {
        fprintf(stderr, "%s: backend unexpectedly rejects the matmul\n",
                gk_type_name(weight_type));
        return 1;
    }

    struct gk_gallocr * gpu_alloc =
        gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (gpu_alloc == NULL || !gk_gallocr_alloc_graph(gpu_alloc, gpu_graph)) {
        fprintf(stderr, "%s: failed to allocate the device graph\n",
                gk_type_name(weight_type));
        return 1;
    }

    const int64_t nw = gk_nelements(gpu_w);
    const int64_t nx = gk_nelements(gpu_x);
    const int64_t no = gk_nelements(gpu_out);
    const size_t encoded_size = gk_nbytes(gpu_w);
    float * w = (float *) malloc((size_t) nw * sizeof(float));
    float * x = (float *) malloc((size_t) nx * sizeof(float));
    void * encoded = malloc(encoded_size);
    float * got = (float *) malloc((size_t) no * sizeof(float));
    float * expected = (float *) malloc((size_t) no * sizeof(float));
    float * decoded = (float *) malloc((size_t) nw * sizeof(float));
    if (w == NULL || x == NULL || encoded == NULL || got == NULL || expected == NULL || decoded == NULL) {
        fprintf(stderr, "%s: host allocation failed\n", gk_type_name(weight_type));
        return 1;
    }
    for (int64_t i = 0; i < nw; ++i) {
        w[i] = input_value((int) i) * 0.125f;
    }
    for (int64_t i = 0; i < nx; ++i) {
        x[i] = input_value((int) (i + nw));
    }
    encode_weights(weight_type, w, encoded, gpu_w->ne[0], gpu_w->ne[1]);
    const struct gk_type_traits * traits = gk_get_type_traits(weight_type);
    const size_t row_bytes = gk_row_size(weight_type, gpu_w->ne[0]);
    for (int64_t r = 0; r < gpu_w->ne[1]; ++r) {
        traits->to_float((const char *) encoded + r * row_bytes,
                         decoded + r * gpu_w->ne[0], gpu_w->ne[0]);
    }

    gk_backend_tensor_set(gpu_w, encoded, 0, encoded_size);
    gk_backend_tensor_set(gpu_x, x, 0, (size_t) nx * sizeof(float));

    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        fprintf(stderr, "%s: device graph execution failed\n", gk_type_name(weight_type));
        return 1;
    }
    gk_backend_synchronize(gpu);
    gk_backend_tensor_get(gpu_out, got, 0, (size_t) no * sizeof(float));

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 4u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * cpu_w;
    struct gk_tensor * cpu_x;
    struct gk_tensor * cpu_out = build_graph(cpu_ctx, weight_type, &cpu_w, &cpu_x);
    memcpy(cpu_w->data, encoded, encoded_size);
    memcpy(cpu_x->data, x, (size_t) nx * sizeof(float));
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cpu_out);
    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        fprintf(stderr, "%s: CPU reference execution failed\n", gk_type_name(weight_type));
        return 1;
    }
    memcpy(expected, cpu_out->data, (size_t) no * sizeof(float));

    int bad = 0;
    float max_abs = 0.0f;
    int64_t i_worst = 0;
    float worst_over = 0.0f;
    for (int64_t i = 0; i < no; ++i) {
        const float diff = fabsf(got[i] - expected[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        const float tol = 5e-3f + 5e-3f * fabsf(expected[i]);
        if (!(diff <= tol)) {
            bad++;
            // by how much the element misses its own tolerance, which is what
            // picks the element worth printing - the largest absolute error
            // usually belongs to the largest output and is within tolerance
            if (diff - tol > worst_over) {
                worst_over = diff - tol;
                i_worst    = i;
            }
        }
    }

    printf("  %-8s %3lld outputs, max abs error %.8g, %d mismatches\n",
           gk_type_name(weight_type), (long long) no, max_abs, bad);
    if (bad != 0) {
        for (int64_t i = i_worst; i < no && i < i_worst + 4; ++i) {
            const int64_t row = i % gpu_w->ne[1];
            const int64_t col = i / gpu_w->ne[1];
            float scalar = 0.0f;
            double precise = 0.0;
            for (int64_t kk = 0; kk < gpu_w->ne[0]; ++kk) {
                scalar += decoded[row * gpu_w->ne[0] + kk] *
                          x[col * gpu_w->ne[0] + kk];
                precise += (double) decoded[row * gpu_w->ne[0] + kk] *
                           (double) x[col * gpu_w->ne[0] + kk];
            }
            printf("      [%lld] device %.8g cpu %.8g scalar %.8g double %.8g\n",
                   (long long) i, got[i], expected[i], scalar, (float) precise);
        }
    }

    free(decoded);
    free(expected);
    free(got);
    free(encoded);
    free(x);
    free(w);
    gk_free(cpu_ctx);
    gk_gallocr_free(gpu_alloc);
    gk_free(gpu_ctx);

    return bad == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------
// op parity
//
// The matmul tests above cover the path a language model spends its time in,
// which leaves the ops a multimodal graph reaches for - the vision tower's
// pooling, the audio tower's short convolution and its cyclic shift - checked
// only by whether the whole server produces sensible text. That is too coarse
// to localise anything, and an op the backend silently declines is not a wrong
// answer but a quiet fall back to the CPU, so both facts are asserted here:
// the backend claims the op, and the answer matches the CPU's.
// --------------------------------------------------------------------------

// The op under test, built twice against the same context-creation rules: once
// where the tensors are host memory and once where they are device memory.
typedef struct gk_tensor * (*op_builder)(struct gk_ctx * ctx, struct gk_tensor ** inputs,
                                         int * n_inputs);

static struct gk_tensor * build_roll(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 24, 5, 3);
    *n_in = 1;
    // a negative shift, a positive one and a zero: each takes a different
    // branch through the wrap
    return gk_roll(ctx, in[0], -7, 3, 0, 0);
}

static struct gk_tensor * build_ssm_conv(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t d_conv = 4, n_t = 11, d_inner = 9, n_seqs = 2;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, d_conv - 1 + n_t, d_inner, n_seqs);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, d_conv, d_inner);
    *n_in = 2;
    return gk_ssm_conv(ctx, in[0], in[1]);
}

static struct gk_tensor * build_pool_2d_max(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 13, 11, 3);
    *n_in = 1;
    return gk_pool_2d(ctx, in[0], GK_OP_POOL_MAX, 3, 3, 2, 2, 1.0f, 1.0f);
}

static struct gk_tensor * build_pool_2d_avg(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 16, 16, 4);
    *n_in = 1;
    // stride equal to the kernel and no padding: the patch-merging shape a
    // vision tower actually uses
    return gk_pool_2d(ctx, in[0], GK_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
}

// The regression this file exists for: a batch that asks for no logits gives
// the output matmul zero columns. There is no launch geometry that means "no
// work", so an empty node has to be recognised before the launch rather than
// rejected by the driver, and the graph it sits in has to still succeed.
static struct gk_tensor * build_empty_mul_mat(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 128);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 0);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// Padded, strided and dilated all at once, so none of the three index terms
// can be dropped without the comparison noticing.
static struct gk_tensor * build_conv_2d(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 3, 4, 5);  // [KW, KH, IC, OC]
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 14, 12, 4, 2); // [IW, IH, IC, N]
    *n_in = 2;
    return gk_conv_2d_direct(ctx, in[0], in[1], 2, 2, 1, 1, 2, 2);
}

// The half-precision kernel, which also rounds the input through f16 on the
// way in - a separate path in both the CPU pass and the kernel.
static struct gk_tensor * build_conv_2d_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 4, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 14, 12, 4, 2);
    *n_in = 2;
    return gk_conv_2d_direct(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1);
}

// The transposed convolution, which the device computes by gathering where the
// CPU pass scatters. A stride above one is the case that matters: it is what
// leaves gaps in the output that only some kernel taps reach, so a wrong
// divisibility test shows up as whole positions being wrong rather than as
// drift. K > s0 also makes the taps overlap, so several of them land on one
// output and the accumulation order is exercised too.
static struct gk_tensor * build_conv_transpose_1d(struct gk_ctx * ctx, struct gk_tensor ** in,
                                                  int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 5, 7, 4);   // [K, Cout, Cin]
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 13, 4, 3);  // [L, Cin, N]
    *n_in = 2;
    return gk_conv_transpose_1d(ctx, in[0], in[1], 3, 0, 1);
}

// Stride one, so every output position is reached by every tap - the opposite
// coverage to the case above.
static struct gk_tensor * build_conv_transpose_1d_s1(struct gk_ctx * ctx, struct gk_tensor ** in,
                                                     int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 4, 6, 5);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 17, 5, 2);
    *n_in = 2;
    return gk_conv_transpose_1d(ctx, in[0], in[1], 1, 0, 1);
}

// The half-precision kernel, which rounds the input through f16 on the way in.
// This is the shape MiniMax-H3's audio VAE decoder uses.
static struct gk_tensor * build_conv_transpose_1d_f16(struct gk_ctx * ctx, struct gk_tensor ** in,
                                                      int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F16, 5, 7, 4);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 13, 4, 3);
    *n_in = 2;
    return gk_conv_transpose_1d(ctx, in[0], in[1], 2, 0, 1);
}

// The two selective-scan variants. They differ only in A's first extent, but
// that picks between one decay per head and one per state element, which is a
// whole branch of the kernel each.
static struct gk_tensor * build_ssm_scan_common(struct gk_ctx * ctx, struct gk_tensor ** in,
                                                int * n_in, int64_t a_ne0) {
    const int64_t d_state = 8, head_dim = 4, n_head = 4, n_group = 2;
    const int64_t n_tok = 5, n_seqs = 2;

    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, d_state, head_dim, n_head, n_seqs);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, head_dim, n_head, n_tok, n_seqs);
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, n_head, n_tok, n_seqs);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, a_ne0, n_head);
    in[4] = gk_new_tensor_4d(ctx, GK_TYPE_F32, d_state, n_group, n_tok, n_seqs);
    in[5] = gk_new_tensor_4d(ctx, GK_TYPE_F32, d_state, n_group, n_tok, n_seqs);
    in[6] = gk_new_tensor_1d(ctx, GK_TYPE_I32, n_seqs);
    *n_in = 7;

    return gk_ssm_scan(ctx, in[0], in[1], in[2], in[3], in[4], in[5], in[6]);
}

static struct gk_tensor * build_ssm_scan_m2(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return build_ssm_scan_common(ctx, in, n_in, 1); // Mamba-2
}

static struct gk_tensor * build_ssm_scan_m1(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return build_ssm_scan_common(ctx, in, n_in, 8); // Mamba-1, d_state decays
}


// --------------------------------------------------------------------------
// The depthwise convolution and the linear-attention recurrences.
//
// The convolution is here in both layouts the builders emit: the ordinary
// WHCN one, and the channels-fastest CWHN a permuted input produces, which the
// kernel only gets right by reading through strides rather than assuming a
// packing. The recurrences are shaped like the models that use them - RWKV's
// heads are 64 wide, the delta rule's 128 - and are run over several tokens,
// because a single token would never exercise the state chaining forward.
// --------------------------------------------------------------------------

static struct gk_tensor * build_conv_2d_dw(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 3, 1, 5);  // kernel, one plane per channel
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 13, 11, 5, 2); // image
    *n_in = 2;
    return gk_conv_2d_dw_direct(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1);
}

static struct gk_tensor * build_conv_2d_dw_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 1, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 13, 11, 5, 2);
    *n_in = 2;
    return gk_conv_2d_dw_direct(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1);
}

// Stride 2 and no padding, which is what MobileNetV5's downsampling stages
// ask for and what makes the window arithmetic worth checking separately.
static struct gk_tensor * build_conv_2d_dw_s2(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 1, 4);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 16, 16, 4, 1);
    *n_in = 2;
    return gk_conv_2d_dw_direct(ctx, in[0], in[1], 2, 2, 0, 0, 1, 1);
}

// The channels-fastest layout: the image is built transposed and permuted back,
// so its channel stride is the innermost one.
static struct gk_tensor * build_conv_2d_dw_cwhn(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 3, 1, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 5, 13, 11, 2); // [C, W, H, N]
    *n_in = 2;

    // src axis 0 (C) becomes axis 2, W becomes 0, H becomes 1
    struct gk_tensor * img = gk_permute(ctx, in[1], 2, 0, 1, 3); // -> [W, H, C, N]
    return gk_conv_2d_dw_direct(ctx, in[0], img, 1, 1, 1, 1, 1, 1);
}

static struct gk_tensor * build_rwkv_wkv6(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t S = 64, H = 3, T = 7, n_seqs = 1;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // k
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // v
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // r
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S, H);          // time-mix first
    in[4] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // decay
    in[5] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    *n_in = 6;
    return gk_rwkv_wkv6(ctx, in[0], in[1], in[2], in[3], in[4], in[5]);
}

// Two sequences, so the state has to be re-read from the input at each
// sequence boundary rather than carried across it.
static struct gk_tensor * build_rwkv_wkv6_seqs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t S = 64, H = 2, T = 8, n_seqs = 2;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S, H);
    in[4] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[5] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    *n_in = 6;
    return gk_rwkv_wkv6(ctx, in[0], in[1], in[2], in[3], in[4], in[5]);
}

// The decay and the two feedback vectors are scaled down before they reach the
// recurrence. Left at the harness's own fill they are order 1, the state grows
// token over token, and the outputs come out in the tens of thousands - where
// the relative tolerance is tens of thousands of times looser than it looks and
// would wave through a kernel that was merely close. Scaled, the recurrence
// settles and the comparison is worth something: this case lands at 1e-6
// rather than at 2.5.
static struct gk_tensor * build_rwkv_wkv7(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t S = 64, H = 3, T = 7, n_seqs = 1;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // r
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // w, the decay
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // k
    in[3] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // v
    in[4] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // a
    in[5] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // b
    in[6] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    *n_in = 7;

    return gk_rwkv_wkv7(ctx, in[0],
                        gk_scale(ctx, in[1], 0.1f),
                        in[2], in[3],
                        gk_scale(ctx, in[4], 0.1f),
                        gk_scale(ctx, in[5], 0.1f),
                        in[6]);
}

static struct gk_tensor * gdn_common(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in,
                                     int64_t S, int64_t H, int64_t T, int64_t n_seqs,
                                     bool kda, int64_t K) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // q
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // k
    in[2] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // v
    in[3] = gk_new_tensor_4d(ctx, GK_TYPE_F32, kda ? S : 1, H, T, n_seqs); // gate
    in[4] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 1, H, T, n_seqs);        // beta
    in[5] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, S, H, n_seqs);        // state
    *n_in = 6;
    return gk_gated_delta_net(ctx, in[0], in[1], in[2], in[3], in[4], in[5], K);
}

// The scalar gate: one decay for the whole head, which is Qwen3-Next's form.
static struct gk_tensor * build_gdn_scalar(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 3, 6, 1, false, 1);
}

// The per-channel gate, which is KDA's, and the only path with a barrier.
static struct gk_tensor * build_gdn_kda(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 3, 6, 1, true, 1);
}

// K above one, so older states are snapshotted as the token loop passes them.
static struct gk_tensor * build_gdn_snapshots(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 2, 6, 1, false, 3);
}

static struct gk_tensor * build_gdn_seqs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 2, 5, 2, true, 1);
}

// The layout the delta-net graphs actually hand in: S = 128 like Qwen3-Next,
// and v a strided slice of the fused qkv projection rather than its own
// tensor. Snapshots on top, so the register form's slot writes are covered at
// full width too.
static struct gk_tensor * build_gdn_strided(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t S = 128, H = 2, T = 5, n_seqs = 1;
    const int64_t qkv = 3 * S * H + 32; // wider than the slices it carries
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // q
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // k
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, qkv, T, n_seqs);         // fused projection
    in[3] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 1, H, T, n_seqs);        // gate
    in[4] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 1, H, T, n_seqs);        // beta
    in[5] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, S, H, n_seqs);        // state
    *n_in = 6;
    struct gk_tensor * v = gk_view_4d(ctx, in[2], S, H, T, n_seqs,
        S * sizeof(float),                  // heads packed inside the slice
        qkv * sizeof(float),                // token stride is the full projection
        (size_t) qkv * T * sizeof(float),
        2 * S * H * sizeof(float));         // v sits after q and k
    return gk_gated_delta_net(ctx, in[0], in[1], v, in[3], in[4], in[5], 2);
}

// A width off every power of two, so the generic fallback kernel keeps its
// own coverage now that the ordinary widths take the register form.
static struct gk_tensor * build_gdn_odd(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 48, 2, 4, 1, false, 1);
}

// The per-channel gate at the register form's full width.
static struct gk_tensor * build_gdn_kda_wide(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 64, 2, 5, 1, true, 1);
}

// The padding, reduction and scan kernels. The shapes here are chosen to land
// off every boundary the implementations care about: a row that is not a
// multiple of the scan's chunk, a pad wider than a warp, a reduction row that
// is not a multiple of the block.
static struct gk_tensor * build_pad_reflect_1d(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 97, 5);
    *n_in = 1;
    return gk_pad_reflect_1d(ctx, in[0], 13, 7);
}

// The largest reflection the op allows: a pad one short of the row, which is
// where an off-by-one in the mirror shows up.
static struct gk_tensor * build_pad_reflect_edge(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 40, 3);
    *n_in = 1;
    return gk_pad_reflect_1d(ctx, in[0], 39, 39);
}

static struct gk_tensor * build_argmax(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 1000, 7);
    *n_in = 1;
    return gk_argmax(ctx, in[0]);
}

// A row wider than one block can cover in a pass, so the reduction has to
// combine partial winners rather than just scan.
static struct gk_tensor * build_argmax_wide(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 262144, 2);
    *n_in = 1;
    return gk_argmax(ctx, in[0]);
}

static struct gk_tensor * build_cumsum(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 333, 4, 2);
    *n_in = 1;
    return gk_cumsum(ctx, in[0]);
}

// Several chunks, the last one short: the three-pass scan has to carry a
// prefix across chunk boundaries and stop at the row's real end.
static struct gk_tensor * build_cumsum_wide(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 20011, 3);
    *n_in = 1;
    return gk_cumsum(ctx, in[0]);
}

static struct gk_tensor * build_im2col_3d(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t IC = 2, N = 2;
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 2, IC * 4); // kernel, shape only
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 5, IC * N); // volume
    *n_in = 2;
    return gk_im2col_3d(ctx, in[0], in[1], IC, 1, 1, 1, 0, 0, 0, 1, 1, 1, GK_TYPE_F32);
}

// f16 output, strided and padded, which is the form a video patch embedding
// actually asks for.
static struct gk_tensor * build_im2col_3d_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t IC = 3, N = 1;
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 2, 2, 2, IC * 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 11, 9, 6, IC * N);
    *n_in = 2;
    return gk_im2col_3d(ctx, in[0], in[1], IC, 2, 2, 2, 1, 1, 1, 1, 1, 1, GK_TYPE_F16);
}

// --------------------------------------------------------------------------
// The diffusion op set.
//
// A transformer decoder exercises a narrow slice of this backend; an image
// model asks for a different one - normalisations over spatial groups, the
// broadcasts a residual stack builds, the im2col a convolution decomposes to.
// These are the ops the diffusion graphs actually contain, and the geometry is
// theirs: channel counts that are not multiples of the block, a token count
// that is not a multiple of the warp.
// --------------------------------------------------------------------------

static struct gk_tensor * build_norm(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 37, 2);
    *n_in = 1;
    return gk_norm(ctx, in[0], 1e-5f);
}

static struct gk_tensor * build_group_norm(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    // [W, H, C, N] with 32 groups over C, the shape every VAE block uses
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 64, 2);
    *n_in = 1;
    return gk_group_norm(ctx, in[0], 32, 1e-6f);
}

static struct gk_tensor * build_mean(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 133, 5, 3);
    *n_in = 1;
    return gk_mean(ctx, in[0]);
}

static struct gk_tensor * build_repeat(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    // repeat_4d rather than repeat: the shape argument of the two-tensor form
    // is a template, not a graph input, and the harness only fills inputs.
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 64, 1, 3);
    *n_in = 1;
    return gk_repeat_4d(ctx, in[0], 64, 37, 3, 1);
}

static struct gk_tensor * build_concat(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 48, 37, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 48, 11, 2);
    *n_in = 2;
    return gk_concat(ctx, in[0], in[1], 1);
}

static struct gk_tensor * build_timestep_embedding(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_1d(ctx, GK_TYPE_F32, 3);
    *n_in = 1;
    return gk_timestep_embedding(ctx, in[0], 256, 10000);
}

static struct gk_tensor * build_im2col_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    // the destination type a composite conv_2d asks for, which is the route
    // every convolution in these models actually takes
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 4, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 14, 12, 4, 2);
    *n_in = 2;
    return gk_im2col(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1, true, GK_TYPE_F16);
}

// im2col decomposes a flat index into (input channel, kernel row, kernel
// column) and (batch, output row, output column), and the case above has every
// one of stride, padding and dilation set to one and a square kernel - so it
// passes whether or not those six are told apart correctly. These do not.
//
// A non-square kernel separates KW from KH, unequal strides separate the two
// spatial axes, and a dilation that differs per axis separates those again.
static struct gk_tensor * build_im2col_asym(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 5, 3, 6, 2);   // KW=5, KH=3, IC=6
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 23, 17, 6, 3); // IW=23, IH=17, N=3
    *n_in = 2;
    return gk_im2col(ctx, in[0], in[1], 2, 3, 2, 1, 2, 1, true, GK_TYPE_F32);
}

// The one-dimensional form, where the kernel height, the output height and the
// batch are all one. Three of the five divisors are then 1, which the
// multiply-shift has no magic number for and carries as a separate case.
static struct gk_tensor * build_im2col_1d(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F16, 4, 7, 2);      // KW=4, IC=7
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 31, 7, 1);     // IW=31, IC=7, N=1
    *n_in = 2;
    return gk_im2col(ctx, in[0], in[1], 3, 1, 1, 0, 2, 1, false, GK_TYPE_F16);
}

// A single input channel and a single batch, so the patch dimension is only
// the kernel area - the shape where an off-by-one between the patch stride and
// the kernel stride still divides evenly and so still looks right.
static struct gk_tensor * build_im2col_ic1(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 1, 1);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 9, 1, 1);
    *n_in = 2;
    return gk_im2col(ctx, in[0], in[1], 1, 1, 0, 0, 1, 1, true, GK_TYPE_F16);
}

// The volume form with all three axes told apart: different kernel extents,
// strides and dilations per axis.
static struct gk_tensor * build_im2col_3d_asym(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t IC = 2;
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 2, 4, IC);  // KW=3, KH=2, KD=4
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 11, 9, 7, IC * 2);
    *n_in = 2;
    return gk_im2col_3d(ctx, in[0], in[1], IC, 2, 1, 2, 1, 1, 0, 1, 2, 1, GK_TYPE_F32);
}

static struct gk_tensor * build_soft_max(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 37, 37, 5);
    *n_in = 1;
    return gk_soft_max(ctx, in[0]);
}

static struct gk_tensor * build_silu(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 133, 7);
    *n_in = 1;
    return gk_silu(ctx, in[0]);
}

static struct gk_tensor * build_gelu(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 133, 7);
    *n_in = 1;
    return gk_gelu(ctx, in[0]);
}

// A residual add where the bias broadcasts over rows - the shape a linear
// layer's bias arrives in, and the one a per-element kernel gets wrong.
static struct gk_tensor * build_add_broadcast(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 37, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 1, 1);
    *n_in = 2;
    return gk_add(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mul_broadcast(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 37, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 1, 2);
    *n_in = 2;
    return gk_mul(ctx, in[0], in[1]);
}

// The rope broadcast: an activation of {2, d_head/2, L, n_head} times a table
// that is the same over the heads. Both operands are contiguous and the second
// agrees with the first over every dimension but the outermost, so its flat
// index is the destination's modulo its own length - the shape that used to
// fall through to a row-mapped kernel with two elements in a row.
static struct gk_tensor * build_mul_wrap_outer(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, 64, 37, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, 64, 37, 1);
    *n_in = 2;
    return gk_mul(ctx, in[0], in[1]);
}

// The same wrap over the two outermost dimensions, and with a length that is
// not a whole number of float4s - so the scalar kernel takes it and the divisor
// is in elements.
static struct gk_tensor * build_add_wrap_odd(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 5, 4, 2);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 5, 1, 1);
    *n_in = 2;
    return gk_add(ctx, in[0], in[1]);
}

// A two-element row that is *not* contiguous, so neither the flat kernel nor
// the row-mapped one may take it: the fully general kernel has to, and this is
// the case that says so.
static struct gk_tensor * build_mul_short_rows_strided(struct gk_ctx * ctx,
                                                       struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 4, 64, 37, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 2, 64, 37, 5);
    *n_in = 2;

    struct gk_tensor * v = gk_view_4d(ctx, in[0], 2, 64, 37, 5,
                                      in[0]->nb[1], in[0]->nb[2], in[0]->nb[3],
                                      2 * sizeof(float));
    return gk_mul(ctx, v, in[1]);
}

// cont of a permuted view: the attention stacks do this between every
// projection, and it is where a kernel that assumes contiguity shows up.
static struct gk_tensor * build_cont_permuted(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 64, 5, 37, 2);
    *n_in = 1;
    return gk_cont(ctx, gk_permute(ctx, in[0], 0, 2, 1, 3));
}

// A matmul whose activations are a permuted view rather than a packed buffer.
static struct gk_tensor * build_mul_mat_permuted(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 96);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 64, 37, 2);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], gk_cont(ctx, gk_permute(ctx, in[1], 0, 2, 1, 3)));
}

// The tiled matmul path: wide enough to take it, and deliberately not a
// multiple of the tile in either direction.
static struct gk_tensor * build_mul_mat_wide(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 130, 100);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 130, 145);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// The batched integer tile, which needs all of: a format with an integer dot,
// enough columns to take the tiled path at all, and enough rows to be worth
// quantizing the activations for. Nothing else in this file meets all three -
// build_mul_mat_wide_q is nvfp4 and a hundred rows, so it takes whichever tile
// the device has for that format rather than the batched integer one - so
// these exist to reach that.
//
// The shapes are deliberately off every tile boundary: 64 is the tile in both
// directions, so rows and columns that are not multiples of it are what
// exercise the edge guards.
static struct gk_tensor * mmq_case(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in,
                                   enum gk_type type, int64_t k, int64_t rows, int64_t cols) {
    in[0] = gk_new_tensor_2d(ctx, type, k, rows);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, k, cols);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mmq_q4_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_0, 256, 600, 40);
}

static struct gk_tensor * build_mmq_q4_1(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_1, 256, 1024, 64);
}

static struct gk_tensor * build_mmq_q8_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q8_0, 256, 576, 33);
}

static struct gk_tensor * build_mmq_q4_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_K, 512, 700, 70);
}

// One super-block rather than two, so the group-to-super-block indexing never
// has to cross a boundary. If this agrees and the wider one drifts, the drift
// is accumulation, not indexing.
static struct gk_tensor * build_mmq_q4_K_1sb(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_K, 256, 700, 70);
}

// The integer *mat-vec*, which is a different kernel from the tile above and
// needs the opposite shape to reach: enough rows to be worth quantizing the
// activations for, few enough columns to stay off the tiled path.
//
// This is the path lm_head takes, and it went untested the moment the row
// threshold was introduced - run_type's sixteen rows fall below it, so the
// per-weight-type sweep quietly stopped covering the integer kernel it used to.
static struct gk_tensor * build_mmv_q4_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_0, 256, 600, 1);
}

static struct gk_tensor * build_mmv_q4_1(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_1, 256, 1024, 2);
}

static struct gk_tensor * build_mmv_q8_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q8_0, 256, 576, 1);
}

static struct gk_tensor * build_mmv_q4_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_K, 512, 700, 3);
}

// The narrow tensor-core tile: 2..23 columns, with enough rows and k for its
// occupancy gate to say yes on any part. One split-scale format and one
// whole-group format, because the tile drains the two differently; the q2_K
// rows sit off the 256 boundary so the row guard runs, and the q4_K columns
// are odd so the column padding does.
static struct gk_tensor * build_mmn_q2_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q2_K, 2048, 4000, 4);
}

static struct gk_tensor * build_mmn_q4_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_K, 2048, 4096, 5);
}

// The lattice formats, against a reference that is exact.
//
// Two things make this harness different from `run_op_tol`, and both are
// forced by the formats.
//
// The weights are not encoded from floats. iq2_xxs, iq2_xs and iq1_s want an
// importance matrix (qz_quantize_requires_imatrix) and this test has none; run
// without one, the iq2_xxs encoder reconstructs weights up to 1e7 with
// thousands of non-finite values among them, and every comparison downstream
// then measures the encoder rather than the kernel. So the block payload is
// filled with pseudo-random bytes instead - every index, sign mask and
// sub-scale in these formats is a bit field with no invalid values - and only
// the block's f16 scale, which is its first two bytes in every format this
// runs on, is overwritten with something sane. The result is a weight row that exercises
// the whole codebook rather than the corner of it a fitted encode picks.
//
// The reference is the host decoder dotted in double, not gk's CPU matmul: the
// CPU dots these against Q8_K activations, one scale per 256, and carries an
// error of its own about the size of the one being looked for. Activations are
// drawn so the device's own activation quantizer reproduces them exactly -
// that quantizer is `d = amax/127, q = rint(v/d)` per 32 values, so values in
// {-1, 0, +1} with at least one non-zero per group come back as themselves.
// Both operands are then exact and the only thing between the device and the
// reference is the order f32 adds things up in.
static int run_lattice_exact(gk_backend_t gpu, const char * name, enum gk_type type,
                             int64_t k, int64_t m, int64_t n) {
    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * w   = gk_new_tensor_2d(ctx, type,        k, m);
    struct gk_tensor * x   = gk_new_tensor_2d(ctx, GK_TYPE_F32, k, n);
    struct gk_tensor * out = gk_mul_mat(ctx, w, x);
    gk_set_output(out);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    if (!gk_backend_supports_op(gpu, out)) {
        printf("  %-14s FAIL: the backend declines the op\n", name);
        gk_free(ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, graph)) {
        printf("  %-14s FAIL: could not allocate the device graph\n", name);
        gk_free(ctx);
        return 1;
    }

    const int64_t nw = k * m;
    const int64_t nx = k * n;
    const int64_t no = m * n;

    float * decoded = (float *) malloc((size_t) nw * sizeof(float));
    float * xf      = (float *) malloc((size_t) nx * sizeof(float));
    float * got     = (float *) malloc((size_t) no * sizeof(float));
    void  * enc     = malloc(gk_nbytes(w));
    double * ref    = (double *) malloc((size_t) no * sizeof(double));

    if (decoded == NULL || xf == NULL || got == NULL || enc == NULL || ref == NULL) {
        return 1;
    }

    for (int64_t i = 0; i < nx; ++i) {
        const float v = input_value((int) (i + 7919));
        xf[i] = v > 0.25f ? 1.0f : (v < -0.25f ? -1.0f : 0.0f);
    }
    // one saturating value per 32, so every group's amax is exactly 1
    for (int64_t i = 0; i < nx; i += 32) {
        xf[i] = 1.0f;
    }

    // Random payload, fixed scale. The scale is the block's leading f16 in
    // every format this runs on; a random one would be a random power of two
    // and the dot would then be decided by whichever block drew the largest.
    const size_t   enc_bytes = gk_nbytes(w);
    const size_t   blk_bytes = gk_row_size(type, gk_blck_size(type));
    uint8_t      * eb        = (uint8_t *) enc;
    uint32_t       rng       = 0x9e3779b9u;
    for (size_t i = 0; i < enc_bytes; ++i) {
        rng = rng * 1664525u + 1013904223u;
        eb[i] = (uint8_t) (rng >> 24);
    }
    for (size_t off = 0; off + blk_bytes <= enc_bytes; off += blk_bytes) {
        const uint16_t d_bits = 0x2E66; // 0.05 in f16
        if (type == GK_TYPE_IQ1_M) {
            // the one lattice block whose scale is not a leading f16: it is
            // spread over the top nibble of each of the four scale words, so
            // pin those nibbles instead
            uint8_t * s = eb + off + 48; // past qs (32) and qh (16)
            for (int w = 0; w < 4; ++w) {
                const uint8_t nib = (uint8_t) ((d_bits >> (4 * w)) & 0xf);
                s[2 * w + 1] = (uint8_t) ((s[2 * w + 1] & 0x0f) | (nib << 4));
            }
        } else if (type == GK_TYPE_Q3_K) {
            // q3_K trails its f16 scale rather than leading with it
            memcpy(eb + off + 108, &d_bits, sizeof(d_bits));
        } else {
            memcpy(eb + off, &d_bits, sizeof(d_bits));
        }
    }

    const struct gk_type_traits * tr = gk_get_type_traits(type);
    tr->to_float(enc, decoded, nw);

    gk_backend_tensor_set(w, enc, 0, gk_nbytes(w));
    gk_backend_tensor_set(x, xf, 0, (size_t) nx * sizeof(float));

    if (gk_backend_graph_compute(gpu, graph) != GK_STATUS_SUCCESS) {
        printf("  %-14s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);
    gk_backend_tensor_get(out, got, 0, (size_t) no * sizeof(float));

    double sum_sq = 0.0;
    for (int64_t col = 0; col < n; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            double precise = 0.0;
            for (int64_t kk = 0; kk < k; ++kk) {
                precise += (double) decoded[row * k + kk] * (double) xf[col * k + kk];
            }
            ref[col * m + row] = precise;
            sum_sq += precise * precise;
        }
    }

    const double rms = sqrt(sum_sq / (double) no);

    int   bad     = 0;
    float max_rel = 0.0f;
    for (int64_t i = 0; i < no; ++i) {
        // scaled by the output's own RMS rather than by each element, because
        // a dot of signed data has outputs near zero and those read as huge
        // relative errors for an absolute difference that is nothing
        const float rel = (float) (fabs((double) got[i] - ref[i]) / (rms > 0.0 ? rms : 1.0));
        if (rel > max_rel) {
            max_rel = rel;
        }
        if (!(rel <= 1e-3f)) {
            bad++;
        }
    }

    printf("  %-14s %5lld outputs, max rel error %.8g, %d mismatches%s\n",
           name, (long long) no, max_rel, bad, bad == 0 ? "" : "  FAIL");
    if (bad != 0) {
        float wmax = 0.0f; int nbadw = 0;
        for (int64_t i = 0; i < nw; ++i) {
            if (!isfinite(decoded[i])) nbadw++;
            else if (fabsf(decoded[i]) > wmax) wmax = fabsf(decoded[i]);
        }
        int64_t iw = 0; float wr = 0.0f;
        for (int64_t i = 0; i < no; ++i) {
            const float r = (float) (fabs((double) got[i] - ref[i]) / (rms > 0.0 ? rms : 1.0));
            if (r > wr) { wr = r; iw = i; }
        }
        printf("      rms %.6g  max|w| %.6g  nonfinite w %d  worst [%lld] got %.8g ref %.8g\n",
               rms, wmax, nbadw, (long long) iw, got[iw], ref[iw]);
    }

    free(ref);
    free(enc);
    free(got);
    free(xf);
    free(decoded);
    gk_gallocr_free(alloc);
    gk_free(ctx);
    return bad == 0 ? 0 : 1;
}

// The lattice formats on the integer path. Both shapes matter: the tile stages
// a whole 32-element group per weight row, the mat-vec walks the row a group
// at a time, and the two find the group inside a 256-element super-block by
// different arithmetic.


// Below GK_CU_MM_Q8_MIN_ROWS, so the same weights take the float decoder
// instead. If this agrees and the one above does not, the difference is the
// integer staging rather than the format's layout.

static struct gk_tensor * build_mmq_iq3_xxs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ3_XXS, 512, 700, 70);
}

static struct gk_tensor * build_mmv_iq3_xxs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ3_XXS, 256, 576, 3);
}

// q6_K on the integer path. It cannot go through run_lattice_exact - its f16
// scale is at the *end* of the block, not the start, so that harness cannot pin
// its magnitude - but its encoder needs no importances, so the CPU comparison
// is sound. Both shapes: past GK_CU_MM_Q8_MIN_ROWS for the tile and the
// mat-vec, since q6_K's scale changes every sixteen and each drains twice.
static struct gk_tensor * build_mmq_q6_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q6_K, 512, 700, 70);
}

static struct gk_tensor * build_mmv_q6_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q6_K, 512, 700, 1);
}

// The wide tile for q2_K - the shape class a q2_K DiT runs all step long.
// q2_K carries a per-16 offset as well as a per-16 scale, so this is the case
// that exercises the two-window drain's `Adl` term at full width.
static struct gk_tensor * build_mmq_q2_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q2_K, 512, 700, 70);
}

// The same tile at its tall (256-row) shape: the dispatch takes it above
// 4096 rows, and the 700-row case above never reaches it. Rows off the 256
// boundary so the tall guard runs.
static struct gk_tensor * build_mmq_q2_K_tall(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q2_K, 512, 4200, 70);
}

// One super-block of k, so the group-to-super-block indexing never crosses a
// boundary; if this agrees and the wider one drifts, the drift is accumulation.
static struct gk_tensor * build_mmv_q6_K_1sb(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q6_K, 256, 576, 3);
}

// The shapes a 30B decode actually runs, which differ from the ones above in
// the dimension that indexes the super-block: k = 6656 is twenty-six of them,
// where 512 is two. A group-to-super-block bug that stays inside the first
// couple of blocks does not show up until k is this long.
static struct gk_tensor * build_mmq_iq2_s_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ2_S, 6656, 1024, 50);
}

static struct gk_tensor * build_mmv_iq2_s_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ2_S, 6656, 1024, 1);
}

static struct gk_tensor * build_mmq_q6_K_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q6_K, 6656, 1024, 50);
}

static struct gk_tensor * build_mmv_q6_K_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q6_K, 6656, 1024, 1);
}

static struct gk_tensor * build_mmq_iq3_s(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ3_S, 512, 700, 70);
}

static struct gk_tensor * build_mmv_iq3_s(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ3_S, 512, 700, 1);
}

// q3_K and iq4_xs on the integer path. Encoders need no importances, so the
// CPU comparison is sound; both shapes reach the tile and the mat-vec, and the
// deep ones make the group-to-super-block indexing cross many boundaries.
static struct gk_tensor * build_mmq_q3_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q3_K, 512, 700, 70);
}

static struct gk_tensor * build_mmv_q3_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q3_K, 512, 700, 1);
}

static struct gk_tensor * build_mmq_q3_K_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q3_K, 6656, 1024, 50);
}

static struct gk_tensor * build_mmv_q3_K_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q3_K, 6656, 1024, 1);
}

static struct gk_tensor * build_mmq_iq4_xs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ4_XS, 512, 700, 70);
}

static struct gk_tensor * build_mmv_iq4_xs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ4_XS, 512, 700, 1);
}

static struct gk_tensor * build_mmq_iq4_xs_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ4_XS, 6656, 1024, 50);
}

static struct gk_tensor * build_mmv_iq4_xs_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_IQ4_XS, 6656, 1024, 1);
}

// Eight columns: still short of the tiled path, but past the point where the
// mat-vec kernel serves several columns from one pass over a weight row.
static struct gk_tensor * build_mmv_q4_0_nc(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_0, 256, 600, 8);
}

// The tensor-core pilot: nvfp4, batched, with k a whole number of 64-element
// blocks. Shapes off the 64x32 tile in both directions so the edge guards are
// exercised, and one that is exactly on it.
static struct gk_tensor * build_mma_nvfp4(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_NVFP4, 256, 600, 40);
}

static struct gk_tensor * build_mma_nvfp4_exact(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_NVFP4, 512, 640, 64);
}

// An *odd* number of 64-element groups: 320 of k is five of them.
//
// The tile stages GK_CU_MMA_FP4_KSTEP groups per round, so a k whose group
// count is not a multiple of KSTEP leaves a partial last round that has to
// stage zeros rather than read past the end of the row. Every other nvfp4 case
// here has k in {256, 512, 1024} - four, eight and sixteen groups - so all of
// them divide by any KSTEP anyone is likely to pick, and none of them would
// notice if the tail were wrong.
static struct gk_tensor * build_mma_nvfp4_odd_k(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_NVFP4, 320, 600, 40);
}

// The FP4 tensor-core tile, against a reference that is actually exact.
//
// This one does not go through `run_op_tol`, and the reason is the whole point
// of it. That harness compares the device against gk's *CPU* matmul, and gk's
// CPU nvfp4 dot quantizes the activation side to 8 bits before dotting. So the
// reference carries an error of its own, of about the size being looked for -
// and worse, the integer device tile makes the *same* 8-bit approximation, so
// it agrees with the reference partly by being wrong in the same direction
// while the FP4 tile disagrees by being more accurate. A tolerance over that
// comparison cannot tell a correct kernel from a scrambled one.
//
// So: decode the weights on the host, dot in double, and feed activations the
// FP4 quantizer reproduces exactly - every value on the e2m1 grid, and one
// value of 6 per group of sixteen so the group's scale quantizes to exactly 1.
// Both operands are then exact and the only thing left between the device and
// the reference is the order f32 adds things up in, which at k=1024 is worth
// about 1e-4 relative. A wrong fragment layout or a misplaced scale is worth
// 100%.
static int run_nvfp4_tile_exact(gk_backend_t gpu) {
    const int64_t k = 1024;   // sixteen 64-element blocks
    const int64_t m = 288;    // not a multiple of the 128-row tile
    const int64_t n = 200;    // wide enough for a tile, ragged against 128

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * w   = gk_new_tensor_2d(ctx, GK_TYPE_NVFP4, k, m);
    struct gk_tensor * x   = gk_new_tensor_2d(ctx, GK_TYPE_F32,   k, n);
    struct gk_tensor * out = gk_mul_mat(ctx, w, x);
    gk_set_output(out);

    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    if (!gk_backend_supports_op(gpu, out)) {
        printf("  %-14s FAIL: the backend declines the op\n", "mma nvfp4 exact");
        gk_free(ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, graph)) {
        printf("  %-14s FAIL: could not allocate the device graph\n", "mma nvfp4 exact");
        gk_free(ctx);
        return 1;
    }

    const int64_t nw = k * m;
    const int64_t nx = k * n;
    const int64_t no = m * n;

    float * wf      = (float *) malloc((size_t) nw * sizeof(float));
    float * decoded = (float *) malloc((size_t) nw * sizeof(float));
    float * xf      = (float *) malloc((size_t) nx * sizeof(float));
    float * got     = (float *) malloc((size_t) no * sizeof(float));
    void  * enc     = malloc(gk_nbytes(w));

    if (wf == NULL || decoded == NULL || xf == NULL || got == NULL || enc == NULL) {
        return 1;
    }

    for (int64_t i = 0; i < nw; ++i) {
        wf[i] = input_value((int) i) * 0.125f;
    }

    // Activations on the e2m1 grid, times a power of two that *changes from
    // one group of sixteen to the next*.
    //
    // The varying scale is the point. A group's amax fixes its ue4m3 scale, so
    // giving every group the same one leaves the activation scale register
    // constant across the whole block - and a kernel that mapped those four
    // scale bytes to the wrong sixteen elements, or read them from the wrong
    // lane, would still be exactly right. That is not hypothetical: it is the
    // bug this case was extended to catch, after a version that held every
    // group at 1.0 measured zero error and still produced noise in a real
    // model.
    //
    // Exactness survives because both parts are exact: a power of two is a
    // ue4m3 value, and 6 times it is the group's amax, so the scale quantizes
    // to itself and the codes are the grid.
    //
    // The cycle is four long and the spread only 2^-1..2^2, for two reasons.
    // Four means every one of the four scale bytes in a 64-element block is
    // distinct, so any permutation of them changes the answer. Narrow means
    // the *integer* tile can still pass this case: it carries one activation
    // scale per thirty-two, so two sub-groups share it, and a wide spread
    // would cost the smaller one its precision and fail the case for a reason
    // that has nothing to do with the kernel.
    static const float grid[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    for (int64_t i = 0; i < nx; ++i) {
        const float v     = input_value((int) (i + 7919));
        const float sign  = v < 0.0f ? -1.0f : 1.0f;
        const float gscl  = ldexpf(1.0f, (int) ((i / 16) % 4) - 1);   // 2^-1 .. 2^2

        xf[i] = sign * grid[(int) (fabsf(v) * 9.0f) & 7] * gscl;
    }
    for (int64_t i = 0; i < nx; i += 16) {
        const float gscl = ldexpf(1.0f, (int) ((i / 16) % 4) - 1);

        xf[i] = (xf[i] < 0.0f ? -6.0f : 6.0f) * gscl;
    }

    const struct gk_type_traits * tr = gk_get_type_traits(GK_TYPE_NVFP4);
    tr->from_float(wf, enc, nw);
    tr->to_float(enc, decoded, nw);

    gk_backend_tensor_set(w, enc, 0, gk_nbytes(w));
    gk_backend_tensor_set(x, xf, 0, (size_t) nx * sizeof(float));

    if (gk_backend_graph_compute(gpu, graph) != GK_STATUS_SUCCESS) {
        printf("  %-14s FAIL: the device graph failed\n", "mma nvfp4 exact");
        return 1;
    }
    gk_backend_synchronize(gpu);
    gk_backend_tensor_get(out, got, 0, (size_t) no * sizeof(float));

    int   bad     = 0;
    float max_rel = 0.0f;

    // The reference first, so the error can be scaled by the output's own RMS.
    //
    // Dividing each error by its own `1 + |expected|` was the obvious thing and
    // it is wrong here: a dot product of signed data has outputs near zero, and
    // an ordinary absolute error against one of those reads as a huge relative
    // one. That is a property of the metric, not of the kernel - it flagged the
    // integer tile on outputs whose true value was a rounding error away from
    // nothing. The RMS of the whole output is the scale to judge against.
    double * ref = (double *) malloc((size_t) no * sizeof(double));
    if (ref == NULL) {
        return 1;
    }

    double sum_sq = 0.0;
    for (int64_t col = 0; col < n; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            double precise = 0.0;
            for (int64_t kk = 0; kk < k; ++kk) {
                precise += (double) decoded[row * k + kk] * (double) xf[col * k + kk];
            }
            ref[col * m + row] = precise;
            sum_sq += precise * precise;
        }
    }

    const double rms = sqrt(sum_sq / (double) no);

    for (int64_t col = 0; col < n; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            const double precise = ref[col * m + row];

            const float g   = got[col * m + row];
            const float rel = (float) (fabs((double) g - precise) / (rms > 0.0 ? rms : 1.0));

            if (rel > max_rel) {
                max_rel = rel;
            }
            // 4e-2 rather than something tight, because this same case has
            // to pass on the integer tile too - a pre-Blackwell part, or
            // `GK_MM_NVFP4_FP4=0` - and that one quantizes the activations to
            // eight bits and lands at about 2e-2 here. The bound is set to
            // catch a wrong layout, which is worth ~100%, not to distinguish
            // the two paths; the printed figure does that, and the difference
            // is stark. The FP4 tile measures exactly 0.
            if (!(rel <= 4e-2f)) {
                bad++;
            }
        }
    }

    printf("  %-14s %5lld outputs, max rel error %.8g, %d mismatches%s\n",
           "mma nvfp4 exact", (long long) no, max_rel, bad, bad == 0 ? "" : "  FAIL");

    free(ref);
    free(enc);
    free(got);
    free(xf);
    free(decoded);
    free(wf);
    gk_gallocr_free(alloc);
    gk_free(ctx);
    return bad == 0 ? 0 : 1;
}

// Several k-blocks and a column count that leaves the last tile ragged.
static struct gk_tensor * build_mma_nvfp4_deep(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_NVFP4, 1024, 130, 33);
}

// Higher dimensions, so the per-slice activation indexing has to be right.
static struct gk_tensor * build_mma_nvfp4_batched(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_NVFP4, 256, 100, 3);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32,   256, 40,  3);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// The f16 tensor-core tile. Its tile is 128 rows by 128 columns by 32 of k,
// and it has two instantiations picked by row count, so what these vary is
// which side of every one of those boundaries the shape falls on.
//
// src1's type is varied too, and that is not incidental: a convolution reaches
// this kernel with its weights as src1, and those are f16, while f32 is what
// activations arrive as everywhere else. Only those two, and bf16 - supports_op
// requires src1 to be a float type, so a quantized src1 never reaches here.
static struct gk_tensor * mma_f16_case(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in,
                                       int64_t k, int64_t rows, int64_t cols, enum gk_type btype) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F16, k, rows);
    in[1] = gk_new_tensor_2d(ctx, btype,       k, cols);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// These are larger than the rest of the file's cases, and they have to be: the
// kernel declines any shape whose grid would leave multiprocessors idle, so a
// case built at the scale of the others would quietly measure the float tile
// instead and pass without ever running the code it names. Every shape here
// clears that gate on a twenty-multiprocessor part.

// Off the tile in every direction, on the wide instantiation.
static struct gk_tensor * build_mma_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mma_f16_case(ctx, in, n_in, 256, 2600, 140, GK_TYPE_F32);
}

// Exactly on it, which is the case where an off-by-one in the edge guards
// hides rather than showing.
static struct gk_tensor * build_mma_f16_exact(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mma_f16_case(ctx, in, n_in, 256, 2560, 128, GK_TYPE_F32);
}

// Too few rows for the wide tile, so the narrow instantiation runs, and enough
// columns that it still fills the device. A UNet's deepest levels are this
// shape - a hundred-odd pixels against a thousand-odd channels.
static struct gk_tensor * build_mma_f16_narrow(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mma_f16_case(ctx, in, n_in, 256, 100, 1400, GK_TYPE_F32);
}

// A k that is neither a whole number of the staged 32 nor a multiple of the
// run of 8, so the last pass of the k loop is partly padding and both operands
// fall to the scalar staging arm.
static struct gk_tensor * build_mma_f16_ragged(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mma_f16_case(ctx, in, n_in, 100, 2600, 140, GK_TYPE_F32);
}

// f16 weights on the src1 side: a convolution's, and the one combination
// where nothing is rounded on the way to the fragment.
static struct gk_tensor * build_mma_f16_wf16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mma_f16_case(ctx, in, n_in, 288, 2600, 140, GK_TYPE_F16);
}

// bf16 on the src1 side, which is neither the fragment's type nor f32 and so
// takes its own arm of the staging pack.
static struct gk_tensor * build_mma_f16_wbf16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mma_f16_case(ctx, in, n_in, 256, 2600, 140, GK_TYPE_BF16);
}

// Higher dimensions, and src0 with fewer of them than src1 so the broadcast
// divisors are exercised rather than left at one.
static struct gk_tensor * build_mma_f16_batched(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F16, 128, 700, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 128, 140, 4);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// Too small in both output directions to cover the device, with a k long
// enough to be cut up - so this is the one case that splits k across blocks
// and sums the pieces afterwards. Nothing else here reaches that path, and
// what it gets wrong if the combine is wrong is a partial sum, which looks
// like a plausible number rather than like a failure.
static struct gk_tensor * build_mma_f16_split(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mma_f16_case(ctx, in, n_in, 8192, 100, 256, GK_TYPE_F16);
}

// Activations that arrive as a permuted view rather than in the order they
// were written, which is the normal shape of an attention projection.
static struct gk_tensor * build_mma_f16_permuted(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F16, 128, 2600);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 128, 2, 140);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], gk_cont(ctx, gk_permute(ctx, in[1], 0, 2, 1, 3)));
}

// A k that is not a whole number of 32-element groups, which the integer path
// declines - so this checks that whichever path an f16 weight falls to at this
// k still computes the same thing.
static struct gk_tensor * build_mmq_ragged_k(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_F16, 130, 600, 40);
}

// Higher dimensions, so the per-slice indexing of the quantized activations
// has to be right rather than accidentally zero.
static struct gk_tensor * build_mmq_batched(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_Q4_0, 256, 520, 3);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32,  256, 40,  3);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mul_mat_wide_q(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_NVFP4, 128, 100);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, 145);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// Batched, so the broadcast of the weight's higher dimensions onto the
// activations' is exercised on the tiled path too.
// The same quantized weight through the mat-vec path, as a control: the CPU
// dots a quantized weight against quantized activations and the device decodes
// to float, so the two differ by the activation quantization on *both* paths.
// If this control mismatches too, the tolerance is what is wrong, not the tile.
static struct gk_tensor * build_mul_mat_narrow_q(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_NVFP4, 128, 100);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, 8);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mul_mat_wide_batched(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 96, 70, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 96, 133, 4);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_rope(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 64, 8, 37);
    in[1] = gk_new_tensor_1d(ctx, GK_TYPE_I32, 37);
    *n_in = 2;
    return gk_rope(ctx, in[0], in[1], 64, 0);
}

static struct gk_tensor * build_get_rows(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 96, 40);
    in[1] = gk_new_tensor_1d(ctx, GK_TYPE_I32, 13);
    *n_in = 2;
    return gk_get_rows(ctx, in[0], in[1]);
}

// The sampler ops: a top-k sampler gathers token ids out of an i32 table, a
// dist sampler casts its computed index to i32 and sums a mask. The gather's
// table is the harness's 0..n fill, which happens to be exactly what a
// candidate list is.
static struct gk_tensor * build_get_rows_i32(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_I32, 1, 40);
    in[1] = gk_new_tensor_1d(ctx, GK_TYPE_I32, 13);
    *n_in = 2;
    return gk_get_rows(ctx, in[0], in[1]);
}

static struct gk_tensor * build_cast_i32(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_1d(ctx, GK_TYPE_F32, 1000);
    *n_in = 1;
    return gk_cast(ctx, gk_scale(ctx, in[0], 1000.0f), GK_TYPE_I32);
}

static struct gk_tensor * build_sum(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 513, 7);
    *n_in = 1;
    return gk_sum(ctx, in[0]);
}

static struct gk_tensor * build_scale(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 133, 7);
    *n_in = 1;
    return gk_scale(ctx, in[0], 0.375f);
}

static struct gk_tensor * build_pad(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 5, 2);
    *n_in = 1;
    return gk_pad(ctx, in[0], 3, 2, 1, 0);
}

static struct gk_tensor * build_upscale(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 5, 2);
    *n_in = 1;
    return gk_upscale(ctx, in[0], 2, GK_SCALE_MODE_NEAREST);
}


// --------------------------------------------------------------------------
// Composite graphs.
//
// Every op above passes on its own, which is not the same as a graph passing.
// A deep chain is where buffer reuse in the allocator, aliasing between a node
// and its source, and any dependence on evaluation order actually show up - a
// single-op test allocates two live tensors and never reuses anything. These
// mirror the two shapes the engine really runs.
// --------------------------------------------------------------------------

static struct gk_tensor * build_vae_stack(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t W = 16, H = 16, C = 64;

    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, W, H, C, 1);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, C, C);
    in[2] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, C, C);
    in[3] = gk_new_tensor_1d(ctx, GK_TYPE_F32, C);
    *n_in = 4;

    struct gk_tensor * x = in[0];
    for (int layer = 0; layer < 3; ++layer) {
        struct gk_tensor * h = gk_group_norm(ctx, x, 32, 1e-6f);
        h = gk_silu(ctx, h);
        h = gk_conv_2d(ctx, in[1], h, 1, 1, 1, 1, 1, 1);
        h = gk_add(ctx, h, gk_reshape_4d(ctx, in[3], 1, 1, C, 1));
        h = gk_group_norm(ctx, h, 32, 1e-6f);
        h = gk_silu(ctx, h);
        h = gk_conv_2d(ctx, in[2], h, 1, 1, 1, 1, 1, 1);
        x = gk_add(ctx, x, h);
    }
    return x;
}

static struct gk_tensor * build_transformer_block(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t D = 128, T = 37, HD = 32, NH = 4;

    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, T);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[2] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[4] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[5] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D * 2);
    in[6] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D * 2, D);
    *n_in = 7;

    struct gk_tensor * x = in[0];
    for (int layer = 0; layer < 2; ++layer) {
        struct gk_tensor * h = gk_norm(ctx, x, 1e-5f);

        struct gk_tensor * q = gk_mul_mat(ctx, in[1], h);
        struct gk_tensor * k = gk_mul_mat(ctx, in[2], h);
        struct gk_tensor * v = gk_mul_mat(ctx, in[3], h);

        q = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, q, HD, NH, T), 0, 2, 1, 3));
        k = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, k, HD, NH, T), 0, 2, 1, 3));
        v = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, v, HD, NH, T), 0, 2, 1, 3));

        struct gk_tensor * att = gk_mul_mat(ctx, k, q);
        att = gk_soft_max_ext(ctx, att, NULL, 1.0f / sqrtf((float) HD), 0.0f);

        struct gk_tensor * o = gk_mul_mat(ctx, gk_cont(ctx, gk_transpose(ctx, v)), att);
        o = gk_cont(ctx, gk_permute(ctx, o, 0, 2, 1, 3));
        o = gk_reshape_2d(ctx, o, D, T);
        o = gk_mul_mat(ctx, in[4], o);

        x = gk_add(ctx, x, o);

        h = gk_norm(ctx, x, 1e-5f);
        h = gk_mul_mat(ctx, in[5], h);
        h = gk_gelu(ctx, h);
        h = gk_mul_mat(ctx, in[6], h);
        x = gk_add(ctx, x, h);
    }
    return x;
}

// anima's two attentions, built the way the diffusion engine builds them with
// flash attention on: the cache cast to f16, V permuted out of [d, head, key]
// and made contiguous, and the result read back through a view. The kernels
// answer these shapes correctly when they are handed plain contiguous
// operands - so if the graph disagrees with the CPU, the disagreement is in
// what feeds the kernel rather than in the kernel.
static struct gk_tensor * anima_attention(struct gk_ctx * ctx,
                                          struct gk_tensor * q,
                                          struct gk_tensor * k_in,
                                          struct gk_tensor * v_in,
                                          int64_t D, int64_t NH, int64_t LQ, int64_t LK) {
    struct gk_tensor * k = gk_cast(ctx, k_in, GK_TYPE_F16);

    struct gk_tensor * v = gk_cont(ctx, gk_permute(ctx, v_in, 0, 2, 1, 3));
    v = gk_reshape_3d(ctx, v, D, LK, NH);
    v = gk_cast(ctx, v, GK_TYPE_F16);

    struct gk_tensor * out = gk_flash_attn_ext(ctx, q, k, v, NULL,
                                               1.0f / sqrtf((float) D), 0.0f, 0.0f);
    gk_flash_attn_ext_set_prec(out, GK_PREC_F32);

    out = gk_view_4d(ctx, out, D, NH, LQ, 1,
                     out->nb[1], out->nb[2], out->nb[1] * NH, 0);
    out = gk_cont(ctx, out);
    return gk_reshape_3d(ctx, out, D * NH, LQ, 1);
}

// The llm adapter's cross-attention: four query rows against a cache of two,
// which is a prompt of two tokens.
static struct gk_tensor * build_anima_adapter_fa(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t D = 64, NH = 16, LQ = 4, LK = 2;

    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, D, LQ, NH);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, D, LK, NH);
    in[2] = gk_new_tensor_4d(ctx, GK_TYPE_F32, D, NH, LK, 1);
    *n_in = 3;

    return anima_attention(ctx, in[0], in[1], in[2], D, NH, LQ, LK);
}

// The transformer's cross-attention: the image against the padded context.
static struct gk_tensor * build_anima_dit_fa(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t D = 128, NH = 4, LQ = 64, LK = 512;

    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, D, LQ, NH);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, D, LK, NH);
    in[2] = gk_new_tensor_4d(ctx, GK_TYPE_F32, D, NH, LK, 1);
    *n_in = 3;

    return anima_attention(ctx, in[0], in[1], in[2], D, NH, LQ, LK);
}

static int run_op_tol(gk_backend_t gpu, const char * name, op_builder build, float tol);
static int run_op(gk_backend_t gpu, const char * name, op_builder build);

// Most ops are exact to f32 rounding against the CPU and are held to 1e-4. Two
// families legitimately are not, and loosening those here is the difference
// between a suite that catches a regression and one that always prints
// failures:
//
//   * a quantized weight: the CPU dots it against activations it has
//     quantized to 8 bits, the device decodes to float and does not. The
//     device answer is the more accurate one; they differ by the activation
//     quantization, not by a bug.
//   * an f16 intermediate: a composite conv_2d hands its im2col result on as
//     f16 and the CPU keeps it there, while the device widens to f32 to
//     accumulate. Again the device is closer to the truth.
//
// Both bounds are set just above what the difference actually measures, so a
// real regression in either still trips them.
static int run_op(gk_backend_t gpu, const char * name, op_builder build) {
    return run_op_tol(gpu, name, build, 1e-4f);
}

static int run_op_tol(gk_backend_t gpu, const char * name, op_builder build, float tol) {
    struct gk_tensor * gpu_in[GK_MAX_SRC] = { NULL };
    struct gk_tensor * cpu_in[GK_MAX_SRC] = { NULL };
    int n_in = 0;

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 4u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * gpu_out = build(gpu_ctx, gpu_in, &n_in);
    gk_set_output(gpu_out);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gpu_out);

    if (!gk_backend_supports_op(gpu, gpu_out)) {
        printf("  %-14s FAIL: the backend declines the op (it would fall back to the CPU)\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-14s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    // The same context on the host, filled with the same numbers.
    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 8u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * cpu_out = build(cpu_ctx, cpu_in, &n_in);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cpu_out);

    int seed = 0;
    for (int i = 0; i < n_in; ++i) {
        const int64_t n     = gk_nelements(cpu_in[i]);
        const size_t bytes  = gk_nbytes(cpu_in[i]);
        void * values = malloc(bytes > 0 ? bytes : 1);
        if (values == NULL) {
            return 1;
        }

        if (cpu_in[i]->type == GK_TYPE_I32) {
            // The only integer input any of these ops takes is ssm_scan's
            // sequence ids, which have to be valid rows of the state cache.
            for (int64_t j = 0; j < n; ++j) {
                ((int32_t *) values)[j] = (int32_t) j;
            }
        } else {
            float * f = (float *) malloc((size_t) (n > 0 ? n : 1) * sizeof(float));
            if (f == NULL) {
                return 1;
            }
            for (int64_t j = 0; j < n; ++j) {
                f[j] = input_value(seed++);
            }
            // A half-precision input is rounded once, here, so both sides read
            // exactly the same bits rather than each rounding its own way.
            if (cpu_in[i]->type == GK_TYPE_F32) {
                memcpy(values, f, (size_t) n * sizeof(float));
            } else {
                gk_get_type_traits(cpu_in[i]->type)->from_float(f, values, n);
            }
            free(f);
        }

        memcpy(cpu_in[i]->data, values, bytes);
        gk_backend_tensor_set(gpu_in[i], values, 0, bytes);
        free(values);
    }

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-14s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-14s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = gk_nelements(gpu_out);
    int   bad     = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float * dump_got = NULL;
    float * dump_exp = NULL;

    if (no > 0) {
        // The output need not be f32 - im2col hands a convolution an f16
        // buffer - so both sides are read as raw bytes and widened through the
        // type's own converter rather than assumed to be floats already.
        const size_t out_bytes = gk_nbytes(gpu_out);
        void  * raw = malloc(out_bytes);
        float * got = (float *) malloc((size_t) no * sizeof(float));
        float * exp_buf = (float *) malloc((size_t) no * sizeof(float));
        if (raw == NULL || got == NULL || exp_buf == NULL) {
            return 1;
        }
        gk_backend_tensor_get(gpu_out, raw, 0, out_bytes);

        if (gpu_out->type == GK_TYPE_F32) {
            memcpy(got, raw, (size_t) no * sizeof(float));
            memcpy(exp_buf, cpu_out->data, (size_t) no * sizeof(float));
        } else {
            const struct gk_type_traits * tr = gk_get_type_traits(gpu_out->type);
            tr->to_float(raw, got, no);
            tr->to_float(cpu_out->data, exp_buf, no);
        }
        free(raw);

        const float * expected = exp_buf;
        for (int64_t i = 0; i < no; ++i) {
            const float diff = fabsf(got[i] - expected[i]);
            if (diff > max_abs) {
                max_abs = diff;
            }
            const float rel = diff / (1.0f + fabsf(expected[i]));
            if (rel > max_rel) {
                max_rel = rel;
            }
            if (!(diff <= tol + tol * fabsf(expected[i]))) {
                bad++;
            }
        }
        dump_got = got;
        dump_exp = exp_buf;
    }

    printf("  %-14s %5lld outputs, max abs error %.8g (rel %.4g), %d mismatches%s\n",
           name, (long long) no, max_abs, max_rel, bad, bad == 0 ? "" : "  FAIL");

    if (bad != 0 && getenv("GK_TEST_DUMP") != NULL) {
        // The first few values either side. A tolerance count says something
        // is wrong; this says whether it is noise on top of the right answer
        // or a different answer altogether, which are not fixed the same way.
        const int64_t show = no < 8 ? no : 8;
        for (int64_t i = 0; i < show; ++i) {
            printf("      [%lld] device %12.6f  cpu %12.6f\n",
                   (long long) i, dump_got[i], dump_exp[i]);
        }
    }
    free(dump_got);
    free(dump_exp);

    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);

    return bad == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------
// fused attention
//
// This op gets its own harness rather than joining run_op's list, for one
// reason: run_op fills every input from the same generator, and an attention
// mask filled that way holds arbitrary finite numbers. The interesting values
// in a mask are the infinities - a position the kernel must skip entirely -
// and whole regions of them, because the device splits the cache across blocks
// and a block whose entire slice is masked contributes nothing to the merge.
// That path cannot be reached with a mask of ordinary floats.
//
// So the mask is built here, in three shapes: absent, causal, and a suffix
// window that leaves early slices completely masked.
// --------------------------------------------------------------------------

enum fa_mask_mode {
    FA_MASK_NONE = 0,
    FA_MASK_CAUSAL,   // position ic visible to query iq1 if it precedes it
    FA_MASK_SUFFIX,   // only the last quarter of the cache is visible
};

struct fa_shape {
    int64_t n_batch;
    int64_t n_head;
    int64_t n_head_kv;   // fewer than n_head is grouped-query attention
    int64_t n_kv;
    int64_t DK;
    int64_t DV;
    enum fa_mask_mode mask;
    bool    sinks;
    // Keys past this one are written as exact zeros in K and V, with no mask
    // to go with them - which is not a contrived pattern but what a diffusion
    // transformer's cross-attention is fed: a prompt of a few tokens in a
    // context the model pads to a fixed 512. A zero key scores exactly zero
    // against every query, so the padding carries real softmax weight, and an
    // error in how the running maximum or the sum treats it costs the
    // conditioning without costing anything else. Zero means the whole cache
    // is live.
    int64_t n_live;
};

// Builds the graph into `ctx`, and hands back the inputs so both sides can be
// given identical bytes.
static struct gk_tensor * fa_build(struct gk_ctx * ctx, const struct fa_shape * s,
                                   struct gk_tensor ** q, struct gk_tensor ** k,
                                   struct gk_tensor ** v, struct gk_tensor ** m,
                                   struct gk_tensor ** sk) {
    *q = gk_new_tensor_4d(ctx, GK_TYPE_F32, s->DK, s->n_batch, s->n_head,    1);
    *k = gk_new_tensor_4d(ctx, GK_TYPE_F16, s->DK, s->n_kv,    s->n_head_kv, 1);
    *v = gk_new_tensor_4d(ctx, GK_TYPE_F16, s->DV, s->n_kv,    s->n_head_kv, 1);
    *m = s->mask == FA_MASK_NONE
        ? NULL
        : gk_new_tensor_4d(ctx, GK_TYPE_F16, s->n_kv, s->n_batch, 1, 1);

    struct gk_tensor * out = gk_flash_attn_ext(ctx, *q, *k, *v, *m,
                                               1.0f / sqrtf((float) s->DK), 0.0f, 0.0f);
    *sk = NULL;
    if (s->sinks) {
        *sk = gk_new_tensor_1d(ctx, GK_TYPE_F32, s->n_head);
        gk_flash_attn_ext_add_sinks(out, *sk);
    }
    return out;
}

// The mask, as f16, in whichever shape the case asked for.
static void fa_fill_mask(const struct fa_shape * s, gk_fp16_t * dst) {
    for (int64_t j = 0; j < s->n_batch; ++j) {
        for (int64_t i = 0; i < s->n_kv; ++i) {
            bool visible = true;
            if (s->mask == FA_MASK_CAUSAL) {
                visible = i <= j + (s->n_kv - s->n_batch);
            } else if (s->mask == FA_MASK_SUFFIX) {
                visible = i >= (s->n_kv * 3) / 4;
            }
            dst[j * s->n_kv + i] = gk_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }
}

static void fa_fill(struct gk_tensor * gpu_t, struct gk_tensor * cpu_t, int * seed) {
    if (gpu_t == NULL) {
        return;
    }
    const int64_t n = gk_nelements(cpu_t);
    const size_t bytes = gk_nbytes(cpu_t);

    float * f = (float *) malloc((size_t) n * sizeof(float));
    void  * raw = malloc(bytes);
    for (int64_t i = 0; i < n; ++i) {
        f[i] = input_value((*seed)++);
    }
    if (cpu_t->type == GK_TYPE_F32) {
        memcpy(raw, f, bytes);
    } else {
        gk_get_type_traits(cpu_t->type)->from_float(f, raw, n);
    }

    memcpy(cpu_t->data, raw, bytes);
    gk_backend_tensor_set(gpu_t, raw, 0, bytes);

    free(raw);
    free(f);
}

// Zero every cache row from `n_live` on, in both copies of the tensor.
static void fa_zero_tail(struct gk_tensor * gpu_t, struct gk_tensor * cpu_t,
                         int64_t n_live) {
    const int64_t d    = cpu_t->ne[0];
    const int64_t n_kv = cpu_t->ne[1];
    const size_t  bytes = gk_nbytes(cpu_t);

    char * buf = (char *) malloc(bytes);
    memcpy(buf, cpu_t->data, bytes);

    for (int64_t i3 = 0; i3 < cpu_t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < cpu_t->ne[2]; ++i2) {
            for (int64_t i1 = n_live; i1 < n_kv; ++i1) {
                memset(buf + i3 * cpu_t->nb[3] + i2 * cpu_t->nb[2] + i1 * cpu_t->nb[1],
                       0, (size_t) d * cpu_t->nb[0]);
            }
        }
    }

    memcpy(cpu_t->data, buf, bytes);
    gk_backend_tensor_set(gpu_t, buf, 0, bytes);
    free(buf);
}

static int run_flash_attn(gk_backend_t gpu, const char * name,
                          struct fa_shape s, float tol) {
    struct gk_tensor *gq, *gk_, *gv, *gm, *gs;
    struct gk_tensor *cq, *ck, *cv, *cm, *cs;

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * gpu_out = fa_build(gpu_ctx, &s, &gq, &gk_, &gv, &gm, &gs);
    gk_set_output(gpu_out);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gpu_out);

    if (!gk_backend_supports_op(gpu, gpu_out)) {
        printf("  %-16s FAIL: the backend declines the op\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-16s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 128u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * cpu_out = fa_build(cpu_ctx, &s, &cq, &ck, &cv, &cm, &cs);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cpu_out);

    int seed = 0;
    fa_fill(gq,  cq,  &seed);
    fa_fill(gk_, ck,  &seed);
    fa_fill(gv,  cv,  &seed);
    fa_fill(gs,  cs,  &seed);

    if (s.n_live > 0 && s.n_live < s.n_kv) {
        fa_zero_tail(gk_, ck, s.n_live);
        fa_zero_tail(gv,  cv, s.n_live);
    }

    if (cm != NULL) {
        const size_t bytes = gk_nbytes(cm);
        gk_fp16_t * mbuf = (gk_fp16_t *) malloc(bytes);
        fa_fill_mask(&s, mbuf);
        memcpy(cm->data, mbuf, bytes);
        gk_backend_tensor_set(gm, mbuf, 0, bytes);
        free(mbuf);
    }

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-16s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-16s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = gk_nelements(gpu_out);
    float * got = (float *) malloc((size_t) no * sizeof(float));
    gk_backend_tensor_get(gpu_out, got, 0, (size_t) no * sizeof(float));
    const float * want = (const float *) cpu_out->data;

    int   bad = 0;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < no; ++i) {
        const float diff = fabsf(got[i] - want[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (!(diff <= tol)) { // catches NaN, which a comparison the other way lets past
            bad++;
        }
    }

    printf("  %-16s %6lld outputs, max abs error %.8g, %d mismatches%s\n",
           name, (long long) no, max_abs, bad, bad == 0 ? "" : "  FAIL");

    free(got);
    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);
    return bad == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------
// the quantized-activation claim
//
// The integer mat-vec quantizes its activations into scratch and leaves a
// claim on them, so that the projections which share one activation - q, k
// and v, or gate and up - quantize it once. What the claim may not be is the
// activation's *address*: the graph allocator hands a dead tensor's storage
// to a later tensor of the same size, so within one execution an address is
// many tensors, and a later matmul whose activation happens to land there was
// handed the earlier tensor's numbers.
//
// A tolerance against the CPU cannot see this. Chained quantized matmuls
// disagree with the CPU by percent already - the two sides quantize the
// activations differently and the products cancel - and the wrong answer here
// is the same size as that noise. So the comparison is device against device:
// the same graph twice, once with the intermediate pinned as an output so
// that nothing can be placed on top of it, and once without. Same kernels,
// same inputs, same arithmetic; the only difference is where the tensors sit,
// so the two runs have to agree bit for bit.
// --------------------------------------------------------------------------

static struct gk_tensor * aq_claim_build(struct gk_ctx * ctx, struct gk_tensor ** in, bool pin) {
    const int64_t K = 1024, N = 4;   // anima's llm adapter: 1024 wide, four tokens

    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_Q4_K, K, K);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_Q4_K, K, K);
    in[2] = gk_new_tensor_2d(ctx, GK_TYPE_Q4_K, K, K);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32,  K, N);

    struct gk_tensor * x = gk_mul_mat(ctx, in[0], in[3]);
    if (pin) {
        // A reference the allocator's walk never gives back, so x keeps its
        // storage to the end of the graph and the norm below is placed
        // somewhere else.
        gk_set_output(x);
    }
    struct gk_tensor * a = gk_mul_mat(ctx, in[1], x);   // quantizes x, claims it
    struct gk_tensor * b = gk_rms_norm(ctx, a, 1e-6f);  // placed where x was
    return gk_mul_mat(ctx, in[2], b);                   // must quantize b, not reuse x
}

static int run_aq_claim(gk_backend_t gpu) {
    struct gk_tensor * in[2][4] = { { NULL } };
    struct gk_tensor * out[2]   = { NULL, NULL };
    struct gk_ctx *    ctx[2]   = { NULL, NULL };
    struct gk_gallocr * alloc[2] = { NULL, NULL };
    float * got[2] = { NULL, NULL };
    int64_t n_out  = 0;
    bool    landed = false;
    int     rc     = 0;

    for (int pass = 0; pass < 2; ++pass) {
        ctx[pass] = gk_init((struct gk_init_params) {
            .mem_size = 4u << 20, .mem_buffer = NULL, .no_alloc = true,
        });
        out[pass] = aq_claim_build(ctx[pass], in[pass], pass == 0);
        gk_set_output(out[pass]);

        struct gk_cgraph * graph = gk_new_graph(ctx[pass]);
        gk_build_forward_expand(graph, out[pass]);

        alloc[pass] = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
        if (alloc[pass] == NULL || !gk_gallocr_alloc_graph(alloc[pass], graph)) {
            printf("  %-14s FAIL: could not allocate the device graph\n", "aq claim");
            rc = 1;
            goto done;
        }

        int seed = 0;
        for (int i = 0; i < 4; ++i) {
            const int64_t n     = gk_nelements(in[pass][i]);
            const size_t  bytes = gk_nbytes(in[pass][i]);
            float * f   = (float *) malloc((size_t) n * sizeof(float));
            void  * raw = malloc(bytes);
            for (int64_t j = 0; j < n; ++j) {
                f[j] = input_value(seed++);
            }
            if (in[pass][i]->type == GK_TYPE_F32) {
                memcpy(raw, f, bytes);
            } else {
                gk_get_type_traits(in[pass][i]->type)->from_float(f, raw, n);
            }
            gk_backend_tensor_set(in[pass][i], raw, 0, bytes);
            free(raw);
            free(f);
        }

        if (pass == 1) {
            // Whether the case is exercised at all: the norm has to be placed
            // exactly where the freed intermediate was, or there is no
            // recycled address and the run proves nothing.
            struct gk_tensor * b = out[pass]->src[1];
            struct gk_tensor * x = b->src[0]->src[1];
            landed = b->data == x->data;
        }

        if (gk_backend_graph_compute(gpu, graph) != GK_STATUS_SUCCESS) {
            printf("  %-14s FAIL: the device graph failed\n", "aq claim");
            rc = 1;
            goto done;
        }
        gk_backend_synchronize(gpu);

        n_out   = gk_nelements(out[pass]);
        got[pass] = (float *) malloc((size_t) n_out * sizeof(float));
        gk_backend_tensor_get(out[pass], got[pass], 0, (size_t) n_out * sizeof(float));
    }

    {
        int   bad     = 0;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < n_out; ++i) {
            const float diff = fabsf(got[0][i] - got[1][i]);
            if (diff > max_abs) {
                max_abs = diff;
            }
            if (!(diff == 0.0f)) {
                bad++;
            }
        }
        printf("  %-14s %5lld outputs, max abs error %.8g, %d mismatches%s%s\n",
               "aq claim", (long long) n_out, max_abs, bad,
               landed ? "" : "  (no address was recycled: unexercised)",
               bad == 0 ? "" : "  FAIL");
        rc = bad == 0 ? 0 : 1;
    }

done:
    for (int pass = 0; pass < 2; ++pass) {
        free(got[pass]);
        if (alloc[pass] != NULL) {
            gk_gallocr_free(alloc[pass]);
        }
        if (ctx[pass] != NULL) {
            gk_free(ctx[pass]);
        }
    }
    return rc;
}

// --------------------------------------------------------------------------
// top_k
//
// Also its own harness, and for a sharper reason than attention's: top_k's
// result is a set of indices with a fully specified order - descending by
// value, index breaking ties, and then the first two deliberately swapped - so
// the device and the CPU must agree exactly, not approximately. An int
// comparison catches what a tolerance would hide.
//
// The data patterns matter more than the shapes. A row of distinct values
// never exercises the tie-break; a row that is entirely -inf is what a fully
// masked router produces, and is where padding slots would displace real
// indices if their sentinel index were not chosen to lose the tie.
// --------------------------------------------------------------------------

enum tk_data {
    TK_DISTINCT = 0,
    TK_TIES,        // few distinct levels, so most comparisons are ties
    TK_ALL_NEG_INF, // a fully masked row
    TK_SOME_NEG_INF,
};

static void tk_fill(float * p, int64_t n, int64_t rows, enum tk_data mode) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t at = r * n + i;
            switch (mode) {
                case TK_DISTINCT:
                    p[at] = input_value((int) (at * 7 + 1));
                    break;
                case TK_TIES:
                    // eight levels over the row: every top-k slot is contested
                    p[at] = (float) (((at * 2654435761u) >> 8) % 8u);
                    break;
                case TK_ALL_NEG_INF:
                    p[at] = -INFINITY;
                    break;
                default:
                    p[at] = (i % 3 == 0) ? -INFINITY : input_value((int) (at * 5 + 3));
                    break;
            }
        }
    }
}

static int run_top_k(gk_backend_t gpu, const char * name,
                     int64_t n, int64_t k, int64_t rows, enum tk_data mode) {
    const size_t in_bytes = (size_t) n * rows * sizeof(float);

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * ga  = gk_new_tensor_2d(gpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * gout = gk_top_k(gpu_ctx, ga, (int) k);
    gk_set_output(gout);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gout);

    if (!gk_backend_supports_op(gpu, gout)) {
        printf("  %-20s FAIL: the backend declines the op\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-20s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * ca   = gk_new_tensor_2d(cpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * cout = gk_top_k(cpu_ctx, ca, (int) k);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cout);

    float * values = (float *) malloc(in_bytes);
    tk_fill(values, n, rows, mode);
    memcpy(ca->data, values, in_bytes);
    gk_backend_tensor_set(ga, values, 0, in_bytes);
    free(values);

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = k * rows;
    int32_t * got = (int32_t *) malloc((size_t) no * sizeof(int32_t));
    gk_backend_tensor_get(gout, got, 0, (size_t) no * sizeof(int32_t));
    const int32_t * want = (const int32_t *) cout->data;

    int bad = 0;
    int first_at = -1;
    for (int64_t i = 0; i < no; ++i) {
        if (got[i] != want[i]) {
            if (first_at < 0) {
                first_at = (int) i;
            }
            bad++;
        }
    }

    if (bad == 0) {
        printf("  %-20s %5lld indices, exact\n", name, (long long) no);
    } else {
        printf("  %-20s %5lld indices, %d differ (first at %d: got %d, want %d)  FAIL\n",
               name, (long long) no, bad, first_at, got[first_at], want[first_at]);
    }

    free(got);
    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);
    return bad == 0 ? 0 : 1;
}

// argsort, which shares top_k's data patterns but returns a whole permutation
// rather than a selection - so a wrong tie-break shows up as a transposed pair
// somewhere in the middle rather than as a missing index, and the comparison
// has to cover every slot.
static int run_argsort(gk_backend_t gpu, const char * name,
                       int64_t n, int64_t rows, enum gk_sort_order order,
                       enum tk_data mode) {
    const size_t in_bytes = (size_t) n * rows * sizeof(float);

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * ga   = gk_new_tensor_2d(gpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * gout = gk_argsort(gpu_ctx, ga, order);
    gk_set_output(gout);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gout);

    if (!gk_backend_supports_op(gpu, gout)) {
        printf("  %-20s FAIL: the backend declines the op\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-20s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * ca   = gk_new_tensor_2d(cpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * cout = gk_argsort(cpu_ctx, ca, order);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cout);

    float * values = (float *) malloc(in_bytes);
    tk_fill(values, n, rows, mode);
    memcpy(ca->data, values, in_bytes);
    gk_backend_tensor_set(ga, values, 0, in_bytes);
    free(values);

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = n * rows;
    int32_t * got = (int32_t *) malloc((size_t) no * sizeof(int32_t));
    gk_backend_tensor_get(gout, got, 0, (size_t) no * sizeof(int32_t));
    const int32_t * want = (const int32_t *) cout->data;

    int bad = 0;
    int first_at = -1;
    for (int64_t i = 0; i < no; ++i) {
        if (got[i] != want[i]) {
            if (first_at < 0) {
                first_at = (int) i;
            }
            bad++;
        }
    }

    if (bad == 0) {
        printf("  %-20s %7lld indices, exact\n", name, (long long) no);
    } else {
        printf("  %-20s %7lld indices, %d differ (first at %d: got %d, want %d)  FAIL\n",
               name, (long long) no, bad, first_at, got[first_at], want[first_at]);
    }

    free(got);
    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);
    return bad == 0 ? 0 : 1;
}

int main(void) {
    gk_device_t device = gk_device_by_type(GK_DEVICE_TYPE_GPU);
    if (device == NULL) {
        fprintf(stderr, "CUDA-family backend was built but no GPU was discovered\n");
        return 1;
    }

    gk_backend_t gpu = gk_device_init_backend(device);
    if (gpu == NULL) {
        fprintf(stderr, "failed to initialize backend for %s\n", gk_device_name(device));
        return 1;
    }

    printf("%s: %s\n", gk_device_name(device), gk_device_description(device));

    int failures = 0;
    const int n_types = (int) (sizeof(g_weight_types) / sizeof(g_weight_types[0]));
    for (int i = 0; i < n_types; ++i) {
        // Both arms below encode from floats, and three of the lattice formats
        // want an importance matrix to do that (qz_quantize_requires_imatrix).
        // Without one their encoders produce weights the format never intended
        // - iq2_xs and iq2_xxs emit non-finite values outright, iq1_s a dynamic
        // range wide enough that the CPU reference lands 10% off its own scalar
        // and double-precision references while the device matches both to
        // seven digits. Either way the comparison would be measuring the
        // encoder rather than the kernel. run_lattice_exact covers all three
        // instead, on blocks built rather than fitted.
        const enum gk_type t = g_weight_types[i];
        if (t == GK_TYPE_IQ1_S || t == GK_TYPE_IQ2_XS || t == GK_TYPE_IQ2_XXS) {
            continue;
        }

        failures += run_decode_type(gpu, t);
        failures += run_type(gpu, t);
    }

    printf("op parity against the CPU:\n");
    failures += run_op(gpu, "roll",          build_roll);
    failures += run_op(gpu, "ssm_conv",      build_ssm_conv);
    failures += run_op(gpu, "pool_2d max",   build_pool_2d_max);
    failures += run_op(gpu, "pool_2d avg",   build_pool_2d_avg);
    failures += run_op(gpu, "conv_2d",       build_conv_2d);
    failures += run_op(gpu, "conv_2d f16",   build_conv_2d_f16);
    failures += run_op(gpu, "convT_1d s3",   build_conv_transpose_1d);
    failures += run_op(gpu, "convT_1d s1",   build_conv_transpose_1d_s1);
    failures += run_op(gpu, "convT_1d f16",  build_conv_transpose_1d_f16);
    failures += run_op(gpu, "ssm_scan m2",   build_ssm_scan_m2);
    failures += run_op(gpu, "ssm_scan m1",   build_ssm_scan_m1);
    failures += run_op(gpu, "mul_mat empty", build_empty_mul_mat);

    // The recurrences chain a state forward and each token's error feeds the
    // next, so they are held to a looser bound than a stateless op: 1e-4 is
    // above the drift a few tokens of f32 accumulation produce and well below
    // anything a wrong recurrence gives.
    printf("depthwise convolution and recurrences:\n");
    failures += run_op(gpu, "conv_2d_dw",     build_conv_2d_dw);
    failures += run_op(gpu, "conv_2d_dw f16", build_conv_2d_dw_f16);
    failures += run_op(gpu, "conv_2d_dw s2",  build_conv_2d_dw_s2);
    failures += run_op(gpu, "conv_dw cwhn",   build_conv_2d_dw_cwhn);
    failures += run_op(gpu, "rwkv_wkv6",      build_rwkv_wkv6);
    failures += run_op(gpu, "rwkv_wkv6 seqs", build_rwkv_wkv6_seqs);
    failures += run_op(gpu, "rwkv_wkv7",      build_rwkv_wkv7);
    failures += run_op(gpu, "gdn scalar",     build_gdn_scalar);
    failures += run_op(gpu, "gdn kda",        build_gdn_kda);
    failures += run_op(gpu, "gdn snapshots",  build_gdn_snapshots);
    failures += run_op(gpu, "gdn seqs",       build_gdn_seqs);
    failures += run_op(gpu, "gdn strided",    build_gdn_strided);
    failures += run_op(gpu, "gdn odd",        build_gdn_odd);
    failures += run_op(gpu, "gdn kda wide",   build_gdn_kda_wide);

    printf("padding, reductions and the scan:\n");
    failures += run_op(gpu, "pad_reflect_1d", build_pad_reflect_1d);
    failures += run_op(gpu, "pad_reflect max", build_pad_reflect_edge);
    failures += run_op(gpu, "argmax",         build_argmax);
    failures += run_op(gpu, "argmax wide",    build_argmax_wide);
    failures += run_op(gpu, "im2col_3d",      build_im2col_3d);
    failures += run_op(gpu, "im2col_3d f16",  build_im2col_3d_f16);
    // The scan sums in a different order from the CPU's straight walk - that
    // is what makes it parallel - so it is held to a relative bound rather
    // than to equality. 1e-4 is far above the reassociation and far below a
    // dropped or double-counted chunk.
    failures += run_op(gpu, "cumsum",         build_cumsum);
    failures += run_op(gpu, "cumsum wide",    build_cumsum_wide);

    printf("diffusion op set:\n");
    failures += run_op(gpu, "norm",           build_norm);
    failures += run_op(gpu, "group_norm",     build_group_norm);
    failures += run_op(gpu, "mean",           build_mean);
    failures += run_op(gpu, "repeat",         build_repeat);
    failures += run_op(gpu, "concat",         build_concat);
    failures += run_op(gpu, "timestep_emb",   build_timestep_embedding);
    failures += run_op(gpu, "im2col f16",     build_im2col_f16);
    failures += run_op(gpu, "im2col asym",    build_im2col_asym);
    failures += run_op(gpu, "im2col 1d",      build_im2col_1d);
    failures += run_op(gpu, "im2col ic1",     build_im2col_ic1);
    failures += run_op(gpu, "im2col_3d asym", build_im2col_3d_asym);
    failures += run_op(gpu, "soft_max",       build_soft_max);
    failures += run_op(gpu, "silu",           build_silu);
    failures += run_op(gpu, "gelu",           build_gelu);
    failures += run_op(gpu, "add broadcast",  build_add_broadcast);
    failures += run_op(gpu, "mul broadcast",  build_mul_broadcast);
    failures += run_op(gpu, "mul wrap outer", build_mul_wrap_outer);
    failures += run_op(gpu, "add wrap odd",   build_add_wrap_odd);
    failures += run_op(gpu, "mul short rows", build_mul_short_rows_strided);
    failures += run_op(gpu, "cont permuted",  build_cont_permuted);
    failures += run_op(gpu, "mul_mat permut", build_mul_mat_permuted);
    failures += run_op(gpu, "mul_mat wide",   build_mul_mat_wide);
    // nvfp4 and 145 columns, so on a Blackwell part this is the FP4 tile and
    // is bounded by the format rather than by the kernel - see the nvfp4 block
    // further down for why that number is what it is.
    failures += run_op_tol(gpu, "mul_mat wide q", build_mul_mat_wide_q, 6e-1f);
    failures += run_op_tol(gpu, "mul_mat narw q", build_mul_mat_narrow_q, 4e-2f);
    failures += run_op(gpu, "mul_mat wide b", build_mul_mat_wide_batched);

    // The batched integer tile. Held to the same bound as the other quantized
    // matmuls: both sides quantize, so they differ by the activation rounding
    // rather than by anything structural.
    failures += run_op_tol(gpu, "mmq q4_0",      build_mmq_q4_0,     4e-2f);
    failures += run_op_tol(gpu, "mmq q4_1",      build_mmq_q4_1,     4e-2f);
    failures += run_op_tol(gpu, "mmq q8_0",      build_mmq_q8_0,     4e-2f);
    // q4_K is held looser than the rest, and for a reason that is about the
    // CPU rather than the device: gk's q4_K vec_dot converts the activation
    // side to Q8_K, which carries one scale per 256 values, while the device
    // carries one per 32. The device is the more accurate of the two and the
    // gap widens with k - 0.040 at one super-block, 0.056 at two - so this
    // bound is set above where that lands rather than where a bug would.
    failures += run_op_tol(gpu, "mmq q4_K 1sb",  build_mmq_q4_K_1sb, 8e-2f);
    failures += run_op_tol(gpu, "mmq q4_K",      build_mmq_q4_K,     8e-2f);
    failures += run_op_tol(gpu, "mmq batched",   build_mmq_batched,  4e-2f);
    failures += run_op_tol(gpu, "mmq ragged k",  build_mmq_ragged_k, 4e-2f);

    failures += run_op_tol(gpu, "mmv q4_0",      build_mmv_q4_0,     4e-2f);
    failures += run_op_tol(gpu, "mmv q4_1",      build_mmv_q4_1,     4e-2f);
    failures += run_op_tol(gpu, "mmv q8_0",      build_mmv_q8_0,     4e-2f);
    failures += run_op_tol(gpu, "mmv q4_K",      build_mmv_q4_K,     8e-2f);
    failures += run_op_tol(gpu, "mmv q4_0 nc",   build_mmv_q4_0_nc,  4e-2f);

    // The narrow tensor-core tile. q2_K is held at the deep-k bound: the CPU
    // reference carries one activation scale per 256 against the device's
    // per-32, and a 2.5-bit format at k=2048 widens that spread.
    failures += run_op_tol(gpu, "mmn q2_K",      build_mmn_q2_K,     2e-1f);
    failures += run_op_tol(gpu, "mmn q4_K",      build_mmn_q4_K,     8e-2f);

    // The lattice formats that have an integer path. iq3_xxs and iq3_s encode
    // sanely without importances, so they can take the CPU comparison at the
    // same loose bound as q4_K - the reference dots them against Q8_K
    // activations, one scale per 256, while the device carries one per 32.
    // q6_K is held at the same loose bound and for the same reason as q4_K:
    // gk's CPU q6_K dot converts the activation side to Q8_K, one scale per
    // 256, while the device carries one per 32.
    failures += run_op_tol(gpu, "mmq q6_K",      build_mmq_q6_K,     8e-2f);
    failures += run_op_tol(gpu, "mmv q6_K",      build_mmv_q6_K,     8e-2f);
    failures += run_op_tol(gpu, "mmv q6_K 1sb",  build_mmv_q6_K_1sb, 8e-2f);
    // The 2-bit encoder is noisy, so like mmn q2_K this carries the loose
    // bound; what it checks is the wide tile's two-window drain, whose
    // exactness against the dp4a tile GK_MM_MMA_SPLIT=0 can bisect.
    failures += run_op_tol(gpu, "mmq q2_K",      build_mmq_q2_K,     2e-1f);
    failures += run_op_tol(gpu, "mmq q2_K tall", build_mmq_q2_K_tall, 2e-1f);

    // The deep mmq bounds are looser than the shallow ones: at k = 6656 the
    // CPU reference's per-256 activation scales cost ~0.1 relative on outputs
    // near zero, and the kernel's own exactness is what run_lattice_exact
    // establishes below - these only check the encoder round-trip still lands
    // in the right place.
    failures += run_op_tol(gpu, "mmq iq2_s deep", build_mmq_iq2_s_deep, 2e-1f);
    failures += run_op_tol(gpu, "mmv iq2_s deep", build_mmv_iq2_s_deep, 8e-2f);
    failures += run_op_tol(gpu, "mmq q6_K deep", build_mmq_q6_K_deep, 8e-2f);
    failures += run_op_tol(gpu, "mmv q6_K deep", build_mmv_q6_K_deep, 8e-2f);

    failures += run_op_tol(gpu, "mmq iq3_xxs",   build_mmq_iq3_xxs,  8e-2f);
    failures += run_op_tol(gpu, "mmv iq3_xxs",   build_mmv_iq3_xxs,  8e-2f);
    failures += run_op_tol(gpu, "mmq iq3_s",     build_mmq_iq3_s,    8e-2f);
    failures += run_op_tol(gpu, "mmv iq3_s",     build_mmv_iq3_s,    8e-2f);

    // q3_K and iq4_xs at the loose bound and for the same reason as q4_K: the
    // CPU reference dots them against Q8_K activations, one scale per 256,
    // while the device carries one per 32.
    failures += run_op_tol(gpu, "mmq q3_K",      build_mmq_q3_K,      8e-2f);
    failures += run_op_tol(gpu, "mmv q3_K",      build_mmv_q3_K,      8e-2f);
    failures += run_op_tol(gpu, "mmq q3_K deep", build_mmq_q3_K_deep, 2e-1f);
    failures += run_op_tol(gpu, "mmv q3_K deep", build_mmv_q3_K_deep, 8e-2f);
    failures += run_op_tol(gpu, "mmq iq4_xs",    build_mmq_iq4_xs,    8e-2f);
    failures += run_op_tol(gpu, "mmv iq4_xs",    build_mmv_iq4_xs,    8e-2f);
    failures += run_op_tol(gpu, "mmq iq4_xs deep", build_mmq_iq4_xs_deep, 2e-1f);
    failures += run_op_tol(gpu, "mmv iq4_xs deep", build_mmv_iq4_xs_deep, 8e-2f);

    // All three against an exact reference, which is the only way iq2_xxs can
    // be checked at all - and the shapes are chosen to reach both kernels: 700
    // rows is past GK_CU_MM_Q8_MIN_ROWS and takes the integer path, 300 is
    // short of it and takes the float decoder.
    failures += run_lattice_exact(gpu, "iq2_xxs mmq",  GK_TYPE_IQ2_XXS, 512, 700, 70);
    failures += run_lattice_exact(gpu, "iq2_xxs mmv",  GK_TYPE_IQ2_XXS, 512, 700,  1);
    failures += run_lattice_exact(gpu, "iq2_xxs f32",  GK_TYPE_IQ2_XXS, 512, 300,  1);
    failures += run_lattice_exact(gpu, "iq3_xxs mmq",  GK_TYPE_IQ3_XXS, 512, 700, 70);
    failures += run_lattice_exact(gpu, "iq3_s mmq",    GK_TYPE_IQ3_S,   512, 700, 70);
    // k = 6656 is twenty-six super-blocks, which is what a 30B model runs and
    // what the 512-of-k cases above cannot reach.
    failures += run_lattice_exact(gpu, "iq2_s deep",   GK_TYPE_IQ2_S,  6656, 2048, 50);
    failures += run_lattice_exact(gpu, "iq2_s deep mv",GK_TYPE_IQ2_S,  6656, 2048,  1);
    failures += run_lattice_exact(gpu, "iq2_s mmq",    GK_TYPE_IQ2_S,   512, 700, 70);
    failures += run_lattice_exact(gpu, "iq2_s mmv",    GK_TYPE_IQ2_S,   512, 700,  1);
    failures += run_lattice_exact(gpu, "iq2_s f32",    GK_TYPE_IQ2_S,   512, 300,  1);
    failures += run_lattice_exact(gpu, "iq2_xs mmq",   GK_TYPE_IQ2_XS,  512, 700, 70);
    failures += run_lattice_exact(gpu, "iq2_xs mmv",   GK_TYPE_IQ2_XS,  512, 700,  1);
    failures += run_lattice_exact(gpu, "iq2_xs deep",  GK_TYPE_IQ2_XS, 6656, 1024, 50);
    failures += run_lattice_exact(gpu, "iq2_xs deep mv", GK_TYPE_IQ2_XS, 6656, 1024, 1);
    failures += run_lattice_exact(gpu, "iq1_s f32",    GK_TYPE_IQ1_S,   512, 700, 70);
    // iq1_m's scale is not a leading f16 - it is assembled from the top nibble
    // of each of four scale words, which the harness pins specially.
    failures += run_lattice_exact(gpu, "iq1_m mmq",    GK_TYPE_IQ1_M,   512, 700, 70);
    failures += run_lattice_exact(gpu, "iq1_m mmv",    GK_TYPE_IQ1_M,   512, 700,  1);
    failures += run_lattice_exact(gpu, "iq1_m deep",   GK_TYPE_IQ1_M,  6656, 1024, 50);
    failures += run_lattice_exact(gpu, "iq1_m deep mv", GK_TYPE_IQ1_M, 6656, 1024, 1);
    // q3_K and iq4_xs against the exact reference too - the deep run_op_tol
    // cases above cannot separate kernel drift from the CPU reference's own.
    failures += run_lattice_exact(gpu, "q3_K exact",    GK_TYPE_Q3_K,   6656, 1024, 50);
    failures += run_lattice_exact(gpu, "q3_K exact mv", GK_TYPE_Q3_K,   6656, 1024,  1);
    failures += run_lattice_exact(gpu, "iq4_xs exact",  GK_TYPE_IQ4_XS, 6656, 1024, 50);
    failures += run_lattice_exact(gpu, "iq4_xs exact mv", GK_TYPE_IQ4_XS, 6656, 1024, 1);

    // nvfp4, and the bound here needs its reasoning spelled out because it is
    // far looser than anything else in this file and that is not slack.
    //
    // On a device with FP4 tensor cores these four run on a kernel that
    // quantizes the *activations* to e2m1. Four bits on both operands is about
    // 7% relative error on a dot product of random data - and unlike weight
    // error it does not shrink with k, because signal and noise both grow as
    // sqrt(k). The CPU reference meanwhile quantizes activations to eight
    // bits. So this comparison is between two different approximations and its
    // spread is a property of the formats rather than of the kernel: no bound
    // over it separates a correct kernel from a broken one.
    //
    // `run_nvfp4_tile_exact` is what does that, against a double-precision
    // reference and inputs both sides represent exactly, where the FP4 tile
    // measures *zero* error. These four are kept because they cover shapes it
    // does not - ragged tiles, batching, several k-blocks - and a gross
    // failure still trips them. `GK_MM_NVFP4_FP4=0` puts them back on the
    // integer tile, where they hold to 4e-2 as they always did.
    failures += run_op_tol(gpu, "mma nvfp4",     build_mma_nvfp4,         6e-1f);
    failures += run_nvfp4_tile_exact(gpu);
    failures += run_op_tol(gpu, "mma nvfp4 exct", build_mma_nvfp4_exact,  6e-1f);
    failures += run_op_tol(gpu, "mma nvfp4 deep", build_mma_nvfp4_deep,   6e-1f);
    failures += run_op_tol(gpu, "mma nvfp4 oddk", build_mma_nvfp4_odd_k,  6e-1f);
    failures += run_op_tol(gpu, "mma nvfp4 batch", build_mma_nvfp4_batched, 6e-1f);

    // The f16 tensor-core tile. Held looser than the float paths for one
    // reason, and it is worth being explicit about it: the fragments are f16,
    // so an f32 src1 is rounded to half on the way in and the products are
    // half-precision even though the accumulator is not. That is the same
    // trade cuBLAS makes for an f16 GEMM, and a caller who does not want it
    // says so with GK_PREC_F32, which sends the matmul to the float tile. The
    // bound below is set just above what the rounding measures at these k;
    // the cases whose src1 is already f16 or quantized round nothing and sit
    // an order of magnitude inside it.
    failures += run_op_tol(gpu, "mma f16",       build_mma_f16,          5e-3f);
    failures += run_op_tol(gpu, "mma f16 exact", build_mma_f16_exact,    5e-3f);
    failures += run_op_tol(gpu, "mma f16 narrow", build_mma_f16_narrow,  5e-3f);
    failures += run_op_tol(gpu, "mma f16 ragged", build_mma_f16_ragged,  5e-3f);
    failures += run_op_tol(gpu, "mma f16 w f16", build_mma_f16_wf16,     5e-3f);
    failures += run_op_tol(gpu, "mma f16 w bf16", build_mma_f16_wbf16,   5e-3f);
    failures += run_op_tol(gpu, "mma f16 batch", build_mma_f16_batched,  5e-3f);
    failures += run_op_tol(gpu, "mma f16 split", build_mma_f16_split,    5e-3f);
    failures += run_op_tol(gpu, "mma f16 permut", build_mma_f16_permuted, 5e-3f);
    failures += run_op(gpu, "rope",           build_rope);
    failures += run_op(gpu, "get_rows",       build_get_rows);
    failures += run_op(gpu, "get_rows i32",   build_get_rows_i32);
    failures += run_op(gpu, "cast f32->i32",  build_cast_i32);
    // The sum reduces a few thousand signed values into one scalar in float
    // against the CPU's double, so it is held to an absolute rather than an
    // exact bound.
    failures += run_op_tol(gpu, "sum",        build_sum, 1e-3f);
    failures += run_op(gpu, "scale",          build_scale);
    failures += run_op(gpu, "pad",            build_pad);
    failures += run_op(gpu, "upscale",        build_upscale);

    // Attention, both ways the device can spread it. "prefill" has enough
    // query rows to fill the card, so the cache is walked whole; "decode" has
    // one token, so the cache is cut into slices and merged. The two paths sum
    // in different orders and are held to a looser bound than an exact op -
    // 2e-3 is above what the reordering measures and below anything a real
    // merge bug would produce.
    printf("fused attention:\n");
    {
        const int64_t DK = 64, DV = 64;

        // one block per row: the unsplit path
        failures += run_flash_attn(gpu, "prefill",
            (struct fa_shape) { 256, 4, 4, 256, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // few rows and a long cache: the split path
        failures += run_flash_attn(gpu, "decode",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_NONE, false }, 2e-3f);
        failures += run_flash_attn(gpu, "decode causal",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // grouped-query: four query heads share one key head
        failures += run_flash_attn(gpu, "decode gqa",
            (struct fa_shape) { 1, 8, 2, 1024, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // the sink belongs to the row, not to a slice, and must be counted
        // once however many slices there are
        failures += run_flash_attn(gpu, "decode sinks",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_CAUSAL, true }, 2e-3f);
        failures += run_flash_attn(gpu, "prefill sinks",
            (struct fa_shape) { 256, 4, 4, 256, DK, DV, FA_MASK_CAUSAL, true }, 2e-3f);

        // only the tail of the cache is visible, so the early slices are
        // entirely masked and contribute nothing - the merge has to skip them
        // rather than let their empty maximum poison the common one
        failures += run_flash_attn(gpu, "decode masked slices",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_SUFFIX, false }, 2e-3f);
        failures += run_flash_attn(gpu, "masked slices+sinks",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_SUFFIX, true }, 2e-3f);

        // a cache the split cannot divide evenly
        failures += run_flash_attn(gpu, "decode ragged",
            (struct fa_shape) { 1, 4, 4, 1000, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // a few tokens rather than one: still short of filling the card, so
        // still split, but with more than one query row per head
        failures += run_flash_attn(gpu, "small batch",
            (struct fa_shape) { 8, 4, 4, 1024, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // The tiled path, which many query rows and a narrow head select. A
        // cache that is not a whole number of 32-key tiles is what checks the
        // tail, and no mask at all is the shape a diffusion transformer runs.
        failures += run_flash_attn(gpu, "tiled ragged kv",
            (struct fa_shape) { 40, 4, 4, 1000, DK, DV, FA_MASK_NONE, false }, 2e-3f);
        failures += run_flash_attn(gpu, "tiled square",
            (struct fa_shape) { 128, 4, 4, 128, DK, DV, FA_MASK_NONE, false }, 2e-3f);
        failures += run_flash_attn(gpu, "tiled gqa mask",
            (struct fa_shape) { 96, 8, 2, 320, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);
        failures += run_flash_attn(gpu, "tiled sinks",
            (struct fa_shape) { 64, 4, 4, 200, DK, DV, FA_MASK_CAUSAL, true }, 2e-3f);
        failures += run_flash_attn(gpu, "tiled masked tail",
            (struct fa_shape) { 48, 4, 4, 300, DK, DV, FA_MASK_SUFFIX, false }, 2e-3f);

        // Wider heads. 128 is what a diffusion transformer usually runs, and
        // the shared tiles grow with it - this is the case that decides
        // whether the tiled path can be taken at all.
        failures += run_flash_attn(gpu, "tiled d96",
            (struct fa_shape) { 64, 4, 4, 256, 96, 96, FA_MASK_NONE, false }, 2e-3f);
        failures += run_flash_attn(gpu, "tiled d128",
            (struct fa_shape) { 64, 4, 4, 256, 128, 128, FA_MASK_NONE, false }, 2e-3f);
        failures += run_flash_attn(gpu, "tiled d128 mask",
            (struct fa_shape) { 40, 8, 2, 300, 128, 128, FA_MASK_CAUSAL, false }, 2e-3f);

        // Past what a lane's accumulator array covers, so this has to fall
        // back rather than drop the dimensions it cannot hold.
        failures += run_flash_attn(gpu, "very wide dv",
            (struct fa_shape) { 32, 4, 4, 128, 256, 256, FA_MASK_CAUSAL, false }, 2e-3f);

        // The head widths a UNet runs, which are not the ones a language
        // model does and were not covered until the tiled path was rewritten
        // on mma. Three things about them break kernels written for 64 and
        // 128: d_head 40 is not a multiple of the 16 a tensor-core window
        // reduces over, so the operands have to be zero-padded rather than
        // assumed whole; d_head 160 is past every accumulator that was sized
        // for 128, and used to fall off the tiled path onto the split kernel
        // at a twentieth of the speed; and cross-attention's cache is 77
        // deep, so the last tile is a ragged 13 whose padding must score as
        // -inf and not as a zero logit that the softmax then counts.
        //
        // The mma path multiplies in f16 - the accumulator stays f32, and K
        // and V were already f16 on both sides, so what is new is Q and the
        // probabilities being rounded before the product, where the CPU
        // reference keeps both in f32. That measures at just over 1e-4 across
        // every case here; 1e-3 leaves an order of magnitude for a different
        // card's rounding and is still two orders below what dropping a tile
        // or mis-scaling a row would produce.
        const float sd_tol = 1e-3f;

        failures += run_flash_attn(gpu, "sd d40 self",
            (struct fa_shape) { 1024, 2, 2, 1024, 40, 40, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "sd d40 cross",
            (struct fa_shape) { 1024, 2, 2,   77, 40, 40, FA_MASK_NONE, false }, sd_tol);

        // the real 64x64 layer, one head of it: the shape that was 42% of a
        // sampling step
        failures += run_flash_attn(gpu, "sd d40 4096",
            (struct fa_shape) { 4096, 1, 1, 4096, 40, 40, FA_MASK_NONE, false }, sd_tol);

        failures += run_flash_attn(gpu, "sd d80 self",
            (struct fa_shape) { 1024, 2, 2, 1024, 80, 80, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "sd d80 cross",
            (struct fa_shape) { 1024, 2, 2,   77, 80, 80, FA_MASK_NONE, false }, sd_tol);

        failures += run_flash_attn(gpu, "sd d160 self",
            (struct fa_shape) {  256, 2, 2,  256, 160, 160, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "sd d160 cross",
            (struct fa_shape) {  256, 2, 2,   77, 160, 160, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "sd d160 8x8",
            (struct fa_shape) {   64, 2, 2,   64, 160, 160, FA_MASK_NONE, false }, sd_tol);

        // Neither the query count nor the cache is a whole number of tiles,
        // so both tails are ragged at once.
        failures += run_flash_attn(gpu, "sd d40 ragged",
            (struct fa_shape) {  300, 2, 2,  205, 40, 40, FA_MASK_NONE, false }, sd_tol);

        // The features that are easy to keep working at 64 and easy to lose
        // at 40 and 160: a mask, grouped-query broadcast, sinks, and a tile
        // that the mask empties completely.
        failures += run_flash_attn(gpu, "sd d40 gqa mask",
            (struct fa_shape) {  300, 4, 2,  205, 40, 40, FA_MASK_CAUSAL, false }, sd_tol);
        failures += run_flash_attn(gpu, "sd d160 sinks",
            (struct fa_shape) {  128, 2, 2,  192, 160, 160, FA_MASK_SUFFIX, true }, sd_tol);
        failures += run_flash_attn(gpu, "sd d80 causal",
            (struct fa_shape) {  192, 4, 4,  192, 80, 80, FA_MASK_CAUSAL, false }, sd_tol);

        // Anima's four attention shapes, taken from GK_FA_DUMP on a 512x512
        // txt2img step. The DiT self-attention (d128, square) is the one the
        // mma path was tuned on; the other three are what the prompt travels
        // through, and a wrong answer in any of them costs the conditioning
        // without costing image quality.
        failures += run_flash_attn(gpu, "anima dit self",
            (struct fa_shape) { 1024, 16, 16, 1024, 128, 128, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "anima dit cross",
            (struct fa_shape) { 1024, 16, 16,  512, 128, 128, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "anima adapter self",
            (struct fa_shape) {    4, 16, 16,    4,  64,  64, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "anima adapter cross",
            (struct fa_shape) {    4, 16, 16,    2,  64,  64, FA_MASK_NONE, false }, sd_tol);
        failures += run_flash_attn(gpu, "anima adapter cross 7",
            (struct fa_shape) {    4, 16, 16,    7,  64,  64, FA_MASK_NONE, false }, sd_tol);

        // The same two shapes with the cache padded the way anima pads it:
        // ten real tokens and 502 zero keys in the DiT's cross-attention, two
        // real and two zero in the adapter's.
        failures += run_flash_attn(gpu, "anima dit cross pad",
            (struct fa_shape) { 1024, 16, 16,  512, 128, 128, FA_MASK_NONE, false, 10 }, sd_tol);
        failures += run_flash_attn(gpu, "anima adapter cross pad",
            (struct fa_shape) {    4, 16, 16,    4,  64,  64, FA_MASK_NONE, false,  2 }, sd_tol);
    }

    // top_k on both sides of the width where one network stops fitting: 4096
    // slots. Below it the row is sorted whole; above it the selection is done
    // in rounds, and the two must give identical indices.
    printf("top_k:\n");
    {
        // the in-network path, which the rounds have to agree with
        failures += run_top_k(gpu, "moe 128 k=8",        128,    8,  17, TK_DISTINCT);
        failures += run_top_k(gpu, "narrow 4096 k=40",   4096,   40,  1, TK_DISTINCT);

        // one past the network's width: the rounds start here
        failures += run_top_k(gpu, "wide 4097 k=40",     4097,   40,  1, TK_DISTINCT);
        failures += run_top_k(gpu, "wide 8192 k=40",     8192,   40,  1, TK_DISTINCT);

        // a real vocabulary row, which is what the old path could not do
        failures += run_top_k(gpu, "vocab 262144 k=40",  262144, 40,  1, TK_DISTINCT);
        failures += run_top_k(gpu, "vocab k=1",          262144,  1,  1, TK_DISTINCT);

        // not a multiple of the chunk, so the last chunk is short and pads
        failures += run_top_k(gpu, "ragged 100000 k=64", 100000, 64,  1, TK_DISTINCT);

        // several rows at once, so the per-row candidate spans have to be
        // indexed apart
        failures += run_top_k(gpu, "wide rows",          20000,  16,  5, TK_DISTINCT);

        // where the tie-break earns its keep
        failures += run_top_k(gpu, "ties narrow",        2048,   32,  3, TK_TIES);
        failures += run_top_k(gpu, "ties wide",          50000,  32,  2, TK_TIES);

        // a fully masked row: every value -inf, so the padding sentinel is the
        // only thing keeping real indices in the answer
        failures += run_top_k(gpu, "all -inf wide",      50000,  24,  2, TK_ALL_NEG_INF);
        failures += run_top_k(gpu, "some -inf wide",     50000,  24,  2, TK_SOME_NEG_INF);

        // a k large enough to need more than one round
        failures += run_top_k(gpu, "big k rounds",       262144, 1024, 1, TK_DISTINCT);
    }

    // argsort across the same boundary. Unlike top_k this returns the whole
    // permutation, so every slot is compared - a tie-break that disagrees with
    // the CPU shows up as a transposed pair in the middle, not as a missing
    // index at the front.
    printf("argsort:\n");
    {
        // the in-network path both directions, as the reference for the rest
        failures += run_argsort(gpu, "narrow desc",   1024,  3, GK_SORT_ORDER_DESC, TK_DISTINCT);
        failures += run_argsort(gpu, "narrow asc",    1024,  3, GK_SORT_ORDER_ASC,  TK_DISTINCT);

        // one past the network's width: out of global memory from here
        failures += run_argsort(gpu, "wide 4097",     4097,  1, GK_SORT_ORDER_DESC, TK_DISTINCT);
        failures += run_argsort(gpu, "wide 8192 asc", 8192,  1, GK_SORT_ORDER_ASC,  TK_DISTINCT);

        // not a power of two, so the padding has to sort past every real slot
        failures += run_argsort(gpu, "ragged 20000",  20000, 2, GK_SORT_ORDER_DESC, TK_DISTINCT);
        failures += run_argsort(gpu, "ragged asc",    20000, 2, GK_SORT_ORDER_ASC,  TK_DISTINCT);

        // where the tie-break decides most of the answer
        failures += run_argsort(gpu, "wide ties",     50000, 2, GK_SORT_ORDER_DESC, TK_TIES);
        failures += run_argsort(gpu, "wide ties asc", 50000, 2, GK_SORT_ORDER_ASC,  TK_TIES);

        // every value the same, so the whole permutation is the tie-break
        failures += run_argsort(gpu, "all -inf",      50000, 2, GK_SORT_ORDER_DESC, TK_ALL_NEG_INF);
        failures += run_argsort(gpu, "some -inf",     50000, 2, GK_SORT_ORDER_ASC,  TK_SOME_NEG_INF);

        // a real vocabulary row, which is the shape the sampler argsorts
        failures += run_argsort(gpu, "vocab 262144",  262144, 1, GK_SORT_ORDER_DESC, TK_DISTINCT);
    }

    printf("composite graphs:\n");
    failures += run_aq_claim(gpu);
    failures += run_op_tol(gpu, "vae stack",      build_vae_stack, 4e-2f);
    failures += run_op_tol(gpu, "transformer",    build_transformer_block, 4e-3f);
    failures += run_op_tol(gpu, "anima adapter fa", build_anima_adapter_fa, 1e-3f);
    failures += run_op_tol(gpu, "anima dit fa",     build_anima_dit_fa,     1e-3f);

    gk_backend_free(gpu);

    printf("%d failures across %d weight types\n", failures, n_types);
    return failures == 0 ? 0 : 1;
}
