#pragma once

// What the backend file needs from the kernel files, and the one conversion
// every launcher starts with.

#include "gk_cuda_common.cuh"

extern "C" {
#include "gk_impl.h"

// --------------------------------------------------------------------------
// activations
//
// Transcribed from gk_compute.c, including the polynomial error function: the
// device has erff(), but it is not the same polynomial, and GELU_ERF differing
// in the last bits between the CPU and the GPU is exactly the kind of thing
// that makes a split graph look subtly broken.
// --------------------------------------------------------------------------

static __device__ __forceinline__ float gk_cu_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabsf(x);

    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    const float t = 1.0f / (1.0f + p * x);
    const float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * expf(-x * x);

    return sign * y;
}

static __device__ __forceinline__ float gk_cu_gelu(float x) {
    const float c = 0.797884560802865f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static __device__ __forceinline__ float gk_cu_gelu_erf(float x) {
    return 0.5f * x * (1.0f + gk_cu_erf(x * 0.7071067811865475f));
}

static __device__ __forceinline__ float gk_cu_gelu_quick(float x) {
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

static __device__ __forceinline__ float gk_cu_silu(float x) {
    return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float gk_cu_unary(int op, float x,
                                                    float p1, float p2, float p3, float p4) {
    switch (op) {
        case GK_UNARY_OP_ABS:         return fabsf(x);
        case GK_UNARY_OP_SGN:         return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        case GK_UNARY_OP_NEG:         return -x;
        case GK_UNARY_OP_STEP:        return x > 0.0f ? 1.0f : 0.0f;
        case GK_UNARY_OP_TANH:        return tanhf(x);
        case GK_UNARY_OP_ELU:         return x > 0.0f ? x : expm1f(x);
        case GK_UNARY_OP_RELU:        return x > 0.0f ? x : 0.0f;
        case GK_UNARY_OP_SIGMOID:     return 1.0f / (1.0f + expf(-x));
        case GK_UNARY_OP_GELU:        return gk_cu_gelu(x);
        case GK_UNARY_OP_GELU_QUICK:  return gk_cu_gelu_quick(x);
        case GK_UNARY_OP_GELU_ERF:    return gk_cu_gelu_erf(x);
        case GK_UNARY_OP_SILU:        return gk_cu_silu(x);
        case GK_UNARY_OP_HARDSWISH:   return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
        case GK_UNARY_OP_HARDSIGMOID: return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
        case GK_UNARY_OP_EXP:         return expf(x);
        case GK_UNARY_OP_EXPM1:       return expm1f(x);
        case GK_UNARY_OP_SOFTPLUS:    return x > 20.0f ? x : logf(1.0f + expf(x));
        case GK_UNARY_OP_FLOOR:       return floorf(x);
        case GK_UNARY_OP_CEIL:        return ceilf(x);
        case GK_UNARY_OP_ROUND:       return rintf(x);
        case GK_UNARY_OP_TRUNC:       return truncf(x);
        case GK_UNARY_OP_XIELU: {
            // p1..p4 are alpha_n, alpha_p, beta, eps, already softplus'd where
            // the builder said so
            if (x > 0.0f) {
                return p2 * x * x + p3 * x;
            }
            const float mx = fminf(x, p4);
            return (expm1f(mx) - x) * p1 + p3 * x;
        }
        default:              return x;
    }
}


}

static __host__ __forceinline__ gk_tview gk_cu_view(const struct gk_tensor * t) {
    gk_tview v;
    v.data = (const char *) t->data;
    for (int i = 0; i < 4; ++i) {
        v.ne[i] = t->ne[i];
        v.nb[i] = (int64_t) t->nb[i];
    }
    v.type = (int) t->type;
    return v;
}

static __host__ __forceinline__ gk_tview_mut gk_cu_view_mut(struct gk_tensor * t) {
    gk_tview_mut v;
    v.data = (char *) t->data;
    for (int i = 0; i < 4; ++i) {
        v.ne[i] = t->ne[i];
        v.nb[i] = (int64_t) t->nb[i];
    }
    v.type = (int) t->type;
    return v;
}

static __host__ __forceinline__ int64_t gk_cu_nelements(const struct gk_tensor * t) {
    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

// Evaluates one node. Returns false for an op this backend has no kernel for,
// which never happens for a graph the scheduler placed here - supports_op is
// asked first - and is a loud failure rather than a wrong answer if it does.
//
// `scratch` is the backend's, and belongs to the same stream. It is passed
// rather than reached for globally so that two backends on two devices cannot
// end up sharing one buffer.
bool gk_cuda_compute_op(gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * node);

// Whether this backend can evaluate the node, operand types included. The
// scheduler calls this before placing anything.
bool gk_cuda_supports_op(const struct gk_tensor * op);

// The head widths the attention kernel can hold in shared memory. supports_op
// checks against these, so they live here rather than beside the kernel.
#define GK_CUDA_FA_MAX_DK 640
#define GK_CUDA_FA_MAX_DV 512

// The widest head the linear-attention recurrences take. They give each slot
// of a head's state its own thread, so this is a block's thread limit.
#define GK_CUDA_RECURRENT_MAX_S 1024

// The matmuls live in their own translation unit: they are the only kernels
// here with a performance story worth separating out.
void gk_cuda_mul_mat   (gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * dst);

// Which of gk_cuda_mul_mat's kernels the last call picked. A shape and a rate
// say a matmul was slow; they do not say whether the fast path declined it, and
// every one of those paths can decline silently - a type it does not cover, a
// scratch allocation that failed, a tile too wide for the output. The profile
// keys on this so the two questions are answered by one row.
const char * gk_cuda_mm_last_path(void);
void gk_cuda_fp4_stats(double * sq_err, double * sq_ref,
                       unsigned long long * zero_groups, unsigned long long * groups);
void gk_cuda_mul_mat_id(gkStream_t stream, struct gk_tensor * dst);

// Launches the fused (rms_norm, mul) pair the backend's fusion plan
// approved: one kernel, writing only the mul's destination.
void gk_cuda_fused_rms_mul(gkStream_t stream, const struct gk_tensor * norm,
                           const struct gk_tensor * mul);

// The three-op residual chain (add, rms_norm, mul): one kernel writing both
// the add's and the mul's destinations.
// The tail fusions: same contracts as the pair above, but the parts need not
// be adjacent - the plan proved the gap safe. All are bit-exact against the
// chains they replace.
void gk_cuda_fused_rms_mul_x(gkStream_t stream, const struct gk_tensor * norm,
                             const struct gk_tensor * mul);
void gk_cuda_fused_add_rms_mul_x(gkStream_t stream, const struct gk_tensor * add,
                                 const struct gk_tensor * norm, const struct gk_tensor * mul);
void gk_cuda_fused_madd(gkStream_t stream, const struct gk_tensor * mul,
                        const struct gk_tensor * add, const struct gk_tensor * add2);
void gk_cuda_fused_unary_mul(gkStream_t stream, const struct gk_tensor * un,
                             const struct gk_tensor * mul);

// The fused activation epilogue: the matmul writes the unary's destination
// with the activation applied to the finished accumulator. `fusable` is the
// plan's gate and mirrors the dispatch exactly, so a fused matmul can only
// land on a kernel that carries the epilogue.
bool gk_cuda_mm_act_fusable(const struct gk_cuda_scratch * scratch,
                            const struct gk_tensor * mm);
void gk_cuda_mul_mat_act(gkStream_t stream, struct gk_cuda_scratch * scratch,
                         struct gk_tensor * mm, struct gk_tensor * un);
void gk_cuda_fused_rope_pair(gkStream_t stream, const struct gk_tensor * m1,
                             const struct gk_tensor * m2, const struct gk_tensor * add);

void gk_cuda_fused_add_rms_mul(gkStream_t stream, const struct gk_tensor * add,
                               const struct gk_tensor * norm, const struct gk_tensor * mul);
void gk_cuda_flash_attn(gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * dst);
