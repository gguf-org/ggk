// Apple Metal accelerator backend: f32 -> quant kernels for the block formats
// that fit one thread per block, plus f16/bf16. The super-block and lattice
// formats have no GPU kernel and fall back to the CPU path.
//
// The shaders below are the Metal counterparts of src/kernels/qz_pack.c and
// follow the same fits, so a tensor comes out the same whichever path ran.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device_metal.h"

#include <algorithm>
#include <cstring>

namespace qz {
namespace {

// The block layouts are restated in MSL because shader source cannot include
// the C headers; they must stay in step with src/kernels/qz_format.h (MSL half
// is 2 bytes, so the layouts line up)
constexpr const char * QZ_METAL_SOURCE = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define QK 32 // block size of all supported quant formats

typedef struct { half d;         uint8_t qs[QK/2]; } blk_q4_0;
typedef struct { half d; half m; uint8_t qs[QK/2]; } blk_q4_1;
typedef struct { half d;         uint8_t qh[4]; uint8_t qs[QK/2]; } blk_q5_0;
typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[QK/2]; } blk_q5_1;
typedef struct { half d;         int8_t qs[QK]; } blk_q8_0;
typedef struct { half d;         uint8_t qs[QK/2]; } blk_iq4_nl;

constant int8_t k_iq4_values[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                        1,   13,  25,  38,  53,  69,  89, 113 };

// starting scales, as fractions of the one that just fits the largest element
#define QZ_STARTS 3
#define QZ_ROUNDS 5
constant float k_shrink[QZ_STARTS] = { 1.0f, 0.94f, 0.88f };

// ----------------------------------------------------------------------
// The fits below mirror src/kernels/qz_common.h: weight each element by
// sqrt(x^2 + floor), then alternate between rounding at the current scale
// and re-solving the scale for that rounding, from several starting points.
// Keeping them in step with the CPU code is what makes the choice of device
// invisible in the output.
// ----------------------------------------------------------------------

static inline void qz_weights(device const float * x, thread float * w) {
    float sumx2 = 0.0f;
    for (int i = 0; i < QK; ++i) {
        sumx2 += x[i] * x[i];
    }
    const float floor2 = 0.5f * sumx2 / (float) QK;
    for (int i = 0; i < QK; ++i) {
        w[i] = sqrt(x[i] * x[i] + floor2);
    }
}

// x[i] ~= s * q[i], q integer in [qlo, qhi]
static inline float qz_fit_sym(device const float * x, thread const float * w, int qlo, int qhi,
                               thread int * q) {
    float amax = 0.0f, xext = 0.0f;
    for (int i = 0; i < QK; ++i) {
        const float a = fabs(x[i]);
        if (a > amax) {
            amax = a;
            xext = x[i];
        }
    }
    if (!(amax > 0.0f)) {
        for (int i = 0; i < QK; ++i) {
            q[i] = 0;
        }
        return 0.0f;
    }

    const float ends[2] = { xext / (float) qlo, xext / (float) qhi };
    int   cur[QK];
    float best_s = 0.0f, best_sse = -1.0f;

    for (int t = 0; t < 2 * QZ_STARTS; ++t) {
        float s = ends[t % 2] * k_shrink[t / 2];
        if (!(fabs(s) > 0.0f) || !isfinite(s)) {
            continue;
        }

        for (int it = 0; it < QZ_ROUNDS; ++it) {
            const float is = 1.0f / s;
            float sse = 0.0f, num = 0.0f, den = 0.0f;

            for (int i = 0; i < QK; ++i) {
                const int k = clamp((int) rint(x[i] * is), qlo, qhi);
                cur[i] = k;
                const float e = x[i] - s * (float) k;
                sse += w[i] * e * e;
                num += w[i] * x[i] * (float) k;
                den += w[i] * (float) k * (float) k;
            }

            if (best_sse < 0.0f || sse < best_sse) {
                best_sse = sse;
                best_s = s;
                for (int i = 0; i < QK; ++i) {
                    q[i] = cur[i];
                }
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

// x[i] ~= s * q[i] - o, q integer in [0, qmax], s >= 0, o >= 0
static inline float qz_fit_min(device const float * x, thread const float * w, int qmax,
                               thread int * q, thread float * offset) {
    float lo = x[0], hi = x[0];
    for (int i = 1; i < QK; ++i) {
        lo = min(lo, x[i]);
        hi = max(hi, x[i]);
    }
    lo = min(lo, 0.0f);
    hi = max(hi, 0.0f);

    if (!(hi > lo)) {
        for (int i = 0; i < QK; ++i) {
            q[i] = 0;
        }
        *offset = 0.0f;
        return 0.0f;
    }

    int   cur[QK];
    float best_s = 0.0f, best_o = -lo, best_sse = -1.0f;

    for (int t = 0; t < QZ_STARTS * QZ_STARTS; ++t) {
        const float lo_t = lo * k_shrink[t % QZ_STARTS];
        const float hi_t = hi * k_shrink[t / QZ_STARTS];

        float s = (hi_t - lo_t) / (float) qmax;
        float o = max(-lo_t, 0.0f);
        if (!(s > 0.0f)) {
            continue;
        }

        for (int it = 0; it < QZ_ROUNDS; ++it) {
            const float is = 1.0f / s;
            float sse = 0.0f;
            float sq2 = 0.0f, sq = 0.0f, sw = 0.0f, sxq = 0.0f, sx = 0.0f;

            for (int i = 0; i < QK; ++i) {
                const int k = clamp((int) rint((x[i] + o) * is), 0, qmax);
                cur[i] = k;
                const float e = x[i] - (s * (float) k - o);
                sse += w[i] * e * e;
                sq2 += w[i] * (float) k * (float) k;
                sq  += w[i] * (float) k;
                sw  += w[i];
                sxq += w[i] * x[i] * (float) k;
                sx  += w[i] * x[i];
            }

            if (best_sse < 0.0f || sse < best_sse) {
                best_sse = sse;
                best_s = s;
                best_o = o;
                for (int i = 0; i < QK; ++i) {
                    q[i] = cur[i];
                }
            }

            const float det = sq2 * sw - sq * sq;
            float ns, no;
            if (fabs(det) > 1e-30f) {
                ns = (sxq * sw - sq * sx) / det;
                no = -(sq2 * sx - sq * sxq) / det;
            } else {
                ns = sq2 > 0.0f ? sxq / sq2 : s;
                no = o;
            }
            if (no < 0.0f) {
                no = 0.0f;
                ns = sq2 > 0.0f ? sxq / sq2 : s;
            }
            if (!(ns > 0.0f) || !isfinite(ns) || !isfinite(no) || (ns == s && no == o)) {
                break;
            }
            s = ns;
            o = no;
        }
    }

    *offset = best_o;
    return best_s;
}

static inline int qz_nearest_sorted(constant const int8_t * tab, int n, float v) {
    if (v <= (float) tab[0]) return 0;
    if (v >= (float) tab[n-1]) return n-1;
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (v < (float) tab[mid]) hi = mid; else lo = mid;
    }
    return (v - (float) tab[lo] <= (float) tab[hi] - v) ? lo : hi;
}

kernel void quantize_f16(device const float * x [[buffer(0)]],
                         device       half  * y [[buffer(1)]],
                         constant   int64_t & n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < (uint64_t) n) {
        y[i] = (half) x[i];
    }
}

kernel void quantize_bf16(device const float    * x [[buffer(0)]],
                          device       uint16_t * y [[buffer(1)]],
                          constant   int64_t    & n [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < (uint64_t) n) {
        uint u = as_type<uint>(x[i]);
        if ((u & 0x7fffffff) > 0x7f800000) { // nan
            y[i] = (uint16_t) ((u >> 16) | 64); // force to quiet
        } else {
            y[i] = (uint16_t) ((u + (0x7fff + ((u >> 16) & 1))) >> 16);
        }
    }
}

kernel void quantize_q4_0(device const float    * x  [[buffer(0)]],
                          device       blk_q4_0 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q4_0 * y = yy + i;

    float w[QK];
    int   q[QK];
    qz_weights(xb, w);
    const float d = qz_fit_sym(xb, w, -8, 7, q);

    y->d = (half) d;
    for (int j = 0; j < QK/2; ++j) {
        y->qs[j] = (uint8_t) ((q[j] + 8) | ((q[j + QK/2] + 8) << 4));
    }
}

kernel void quantize_q4_1(device const float    * x  [[buffer(0)]],
                          device       blk_q4_1 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q4_1 * y = yy + i;

    float w[QK];
    int   q[QK];
    float o;
    qz_weights(xb, w);
    const float d = qz_fit_min(xb, w, 15, q, &o);

    y->d = (half) d;
    y->m = (half) (-o);
    for (int j = 0; j < QK/2; ++j) {
        y->qs[j] = (uint8_t) (q[j] | (q[j + QK/2] << 4));
    }
}

kernel void quantize_q5_0(device const float    * x  [[buffer(0)]],
                          device       blk_q5_0 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q5_0 * y = yy + i;

    float w[QK];
    int   q[QK];
    qz_weights(xb, w);
    const float d = qz_fit_sym(xb, w, -16, 15, q);

    y->d = (half) d;

    uint32_t qh = 0;
    for (int j = 0; j < QK/2; ++j) {
        const uint32_t q0 = (uint32_t) (q[j] + 16);
        const uint32_t q1 = (uint32_t) (q[j + QK/2] + 16);
        y->qs[j] = (uint8_t) ((q0 & 0xf) | ((q1 & 0xf) << 4));
        qh |= ((q0 >> 4) & 1u) << j;
        qh |= ((q1 >> 4) & 1u) << (j + QK/2);
    }
    for (int j = 0; j < 4; ++j) {
        y->qh[j] = (qh >> (8*j)) & 0xff;
    }
}

kernel void quantize_q5_1(device const float    * x  [[buffer(0)]],
                          device       blk_q5_1 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q5_1 * y = yy + i;

    float w[QK];
    int   q[QK];
    float o;
    qz_weights(xb, w);
    const float d = qz_fit_min(xb, w, 31, q, &o);

    y->d = (half) d;
    y->m = (half) (-o);

    uint32_t qh = 0;
    for (int j = 0; j < QK/2; ++j) {
        const uint32_t q0 = (uint32_t) q[j];
        const uint32_t q1 = (uint32_t) q[j + QK/2];
        y->qs[j] = (uint8_t) ((q0 & 0xf) | ((q1 & 0xf) << 4));
        qh |= ((q0 >> 4) & 1u) << j;
        qh |= ((q1 >> 4) & 1u) << (j + QK/2);
    }
    for (int j = 0; j < 4; ++j) {
        y->qh[j] = (qh >> (8*j)) & 0xff;
    }
}

kernel void quantize_q8_0(device const float    * x  [[buffer(0)]],
                          device       blk_q8_0 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q8_0 * y = yy + i;

    float amax = 0.0f;
    for (int j = 0; j < QK; ++j) {
        amax = max(amax, fabs(xb[j]));
    }

    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f/d : 0.0f;

    y->d = (half) d;
    for (int j = 0; j < QK; ++j) {
        y->qs[j] = (int8_t) rint(xb[j]*id);
    }
}

kernel void quantize_iq4_nl(device const float      * x  [[buffer(0)]],
                            device       blk_iq4_nl * yy [[buffer(1)]],
                            constant   int64_t      & nb [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_iq4_nl * y = yy + i;

    float w[QK];
    qz_weights(xb, w);

    float amax = 0.0f, xext = 0.0f;
    for (int j = 0; j < QK; ++j) {
        const float a = fabs(xb[j]);
        if (a > amax) {
            amax = a;
            xext = xb[j];
        }
    }

    int   k[QK];
    float best_d = 0.0f;

    if (amax > 0.0f) {
        const float ends[2] = { xext / (float) k_iq4_values[0], xext / (float) k_iq4_values[15] };
        int   cur[QK];
        float best_sse = -1.0f;

        for (int t = 0; t < 2 * QZ_STARTS; ++t) {
            float s = ends[t % 2] * k_shrink[t / 2];
            if (!(fabs(s) > 0.0f) || !isfinite(s)) {
                continue;
            }

            for (int it = 0; it < QZ_ROUNDS; ++it) {
                const float is = 1.0f / s;
                float sse = 0.0f, num = 0.0f, den = 0.0f;

                for (int j = 0; j < QK; ++j) {
                    const int idx = qz_nearest_sorted(k_iq4_values, 16, xb[j] * is);
                    const float v = (float) k_iq4_values[idx];
                    const float e = xb[j] - s * v;
                    cur[j] = idx;
                    sse += w[j] * e * e;
                    num += w[j] * xb[j] * v;
                    den += w[j] * v * v;
                }

                if (best_sse < 0.0f || sse < best_sse) {
                    best_sse = sse;
                    best_d = s;
                    for (int j = 0; j < QK; ++j) {
                        k[j] = cur[j];
                    }
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
    } else {
        for (int j = 0; j < QK; ++j) {
            k[j] = 8; // codebook entry closest to zero
        }
    }

    y->d = (half) best_d;
    for (int j = 0; j < QK/2; ++j) {
        y->qs[j] = (uint8_t) (k[j] | (k[j + QK/2] << 4));
    }
}
)MSL";

constexpr int QK_SIMPLE = 32;

struct type_kernel {
    qz_type    type;
    const char * kernel_name;
    size_t       block_bytes;   // output bytes per QK_SIMPLE input values (or per element for f16/bf16)
    bool         elementwise;
};

const type_kernel k_type_kernels[] = {
    { QZ_TYPE_F16,    "quantize_f16",    2,          true  },
    { QZ_TYPE_BF16,   "quantize_bf16",   2,          true  },
    { QZ_TYPE_Q4_0,   "quantize_q4_0",   2 + 16,     false },
    { QZ_TYPE_Q4_1,   "quantize_q4_1",   4 + 16,     false },
    { QZ_TYPE_Q5_0,   "quantize_q5_0",   2 + 4 + 16, false },
    { QZ_TYPE_Q5_1,   "quantize_q5_1",   4 + 4 + 16, false },
    { QZ_TYPE_Q8_0,   "quantize_q8_0",   2 + 32,     false },
    { QZ_TYPE_IQ4_NL, "quantize_iq4_nl", 2 + 16,     false },
};

const type_kernel * find_kernel(qz_type type) {
    for (const type_kernel & k : k_type_kernels) {
        if (k.type == type) {
            return &k;
        }
    }
    return nullptr;
}

class metal_converter : public device_converter {
public:
    metal_converter() : name_("metal") {}

    bool init(std::string & error) {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) {
                error = "no Metal device available";
                return false;
            }
            NSError * ns_error   = nil;
            id<MTLLibrary> library = [device_ newLibraryWithSource:@(QZ_METAL_SOURCE) options:nil error:&ns_error];
            if (!library) {
                error = "failed to compile Metal kernels";
                if (ns_error) {
                    error += std::string(": ") + ns_error.localizedDescription.UTF8String;
                }
                return false;
            }
            for (size_t i = 0; i < sizeof(k_type_kernels) / sizeof(k_type_kernels[0]); ++i) {
                id<MTLFunction> fn = [library newFunctionWithName:@(k_type_kernels[i].kernel_name)];
                if (!fn) {
                    error = std::string("missing Metal kernel ") + k_type_kernels[i].kernel_name;
                    return false;
                }
                pipelines_[i] = [device_ newComputePipelineStateWithFunction:fn error:&ns_error];
                if (!pipelines_[i]) {
                    error = std::string("failed to create pipeline for ") + k_type_kernels[i].kernel_name;
                    if (ns_error) {
                        error += std::string(": ") + ns_error.localizedDescription.UTF8String;
                    }
                    return false;
                }
            }
            queue_ = [device_ newCommandQueue];
            if (!queue_) {
                error = "failed to create Metal command queue";
                return false;
            }
            return true;
        }
    }

    const std::string & name() const override {
        return name_;
    }

    bool supports(qz_type dst_type) const override {
        return find_kernel(dst_type) != nullptr;
    }

    bool convert(const float * src, qz_type dst_type, void * dst, int64_t nrows, int64_t n_per_row,
                 std::string & error) override {
        const type_kernel * kernel = find_kernel(dst_type);
        if (kernel == nullptr) {
            error = "unsupported type";
            return false;
        }
        size_t pipeline_index = kernel - k_type_kernels;

        // bound temporary buffer memory; a row is never split because
        // quantized blocks are row-oriented
        constexpr size_t max_chunk_input_bytes = 64u * 1024u * 1024u;
        const size_t  row_bytes = (size_t) n_per_row * sizeof(float);
        const size_t  dst_row_bytes =
            kernel->elementwise ? (size_t) n_per_row * kernel->block_bytes
                                : (size_t) n_per_row / QK_SIMPLE * kernel->block_bytes;
        const int64_t rows_per_chunk =
            std::max<int64_t>(1, std::min<int64_t>(nrows, max_chunk_input_bytes / std::max<size_t>(1, row_bytes)));

        @autoreleasepool {
            id<MTLBuffer> src_buf = [device_ newBufferWithLength:rows_per_chunk * row_bytes
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> dst_buf = [device_ newBufferWithLength:rows_per_chunk * dst_row_bytes
                                                         options:MTLResourceStorageModeShared];
            if (!src_buf || !dst_buf) {
                error = "failed to allocate Metal buffers";
                return false;
            }

            for (int64_t row0 = 0; row0 < nrows; row0 += rows_per_chunk) {
                const int64_t chunk_rows = std::min(rows_per_chunk, nrows - row0);
                const int64_t n          = chunk_rows * n_per_row;
                const int64_t n_items    = kernel->elementwise ? n : n / QK_SIMPLE;

                memcpy(src_buf.contents, src + row0 * n_per_row, n * sizeof(float));

                id<MTLCommandBuffer>         cmd = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                id<MTLComputePipelineState>  pso = pipelines_[pipeline_index];

                [enc setComputePipelineState:pso];
                [enc setBuffer:src_buf offset:0 atIndex:0];
                [enc setBuffer:dst_buf offset:0 atIndex:1];
                [enc setBytes:&n_items length:sizeof(n_items) atIndex:2];

                const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
                [enc dispatchThreadgroups:MTLSizeMake((n_items + tg - 1) / tg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                if (cmd.status != MTLCommandBufferStatusCompleted) {
                    error = "Metal command buffer failed";
                    if (cmd.error) {
                        error += std::string(": ") + cmd.error.localizedDescription.UTF8String;
                    }
                    return false;
                }

                memcpy((char *) dst + row0 * dst_row_bytes, dst_buf.contents, chunk_rows * dst_row_bytes);
            }
        }
        return true;
    }

private:
    std::string                 name_;
    id<MTLDevice>               device_ = nil;
    id<MTLCommandQueue>         queue_  = nil;
    id<MTLComputePipelineState> pipelines_[sizeof(k_type_kernels) / sizeof(k_type_kernels[0])] = {};
};

} // namespace

bool metal_device_available() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

std::string metal_device_description() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device ? device.name.UTF8String : "no Metal device";
    }
}

std::unique_ptr<device_converter> metal_device_open(std::string & error) {
    auto converter = std::make_unique<metal_converter>();
    if (!converter->init(error)) {
        return nullptr;
    }
    return converter;
}

} // namespace qz
