// CUDA / ROCm(HIP) accelerator backend: f32 -> quant kernels for the block
// formats that fit one thread per block, plus f16/bf16. The super-block and
// lattice formats have no GPU kernel and fall back to the CPU path.
//
// The encoders here are the device counterparts of src/kernels/qz_pack.c and
// follow the same fits, so a tensor comes out the same whichever path ran.
//
// The same file compiles as CUDA (default) or HIP (with QZ_USE_HIP defined and
// the source language set to HIP in CMake).

#include "device_gpu.h"

#include "kernels/qz_format.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(QZ_USE_HIP)
#  include <hip/hip_runtime.h>
#  include <hip/hip_fp16.h>
#  define gpuDeviceProp           hipDeviceProp_t
#  define gpuError_t              hipError_t
#  define gpuFree                 hipFree
#  define gpuGetDeviceCount       hipGetDeviceCount
#  define gpuGetDeviceProperties  hipGetDeviceProperties
#  define gpuGetErrorString       hipGetErrorString
#  define gpuGetLastError         hipGetLastError
#  define gpuMalloc               hipMalloc
#  define gpuMemcpyAsync          hipMemcpyAsync
#  define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
#  define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#  define gpuSetDevice            hipSetDevice
#  define gpuStreamCreate         hipStreamCreate
#  define gpuStreamDestroy        hipStreamDestroy
#  define gpuStreamSynchronize    hipStreamSynchronize
#  define gpuStream_t             hipStream_t
#  define gpuSuccess              hipSuccess
#  define QZ_GPU_NAME_PREFIX      "rocm"
#else
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>
#  define gpuDeviceProp           cudaDeviceProp
#  define gpuError_t              cudaError_t
#  define gpuFree                 cudaFree
#  define gpuGetDeviceCount       cudaGetDeviceCount
#  define gpuGetDeviceProperties  cudaGetDeviceProperties
#  define gpuGetErrorString       cudaGetErrorString
#  define gpuGetLastError         cudaGetLastError
#  define gpuMalloc               cudaMalloc
#  define gpuMemcpyAsync          cudaMemcpyAsync
#  define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#  define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#  define gpuSetDevice            cudaSetDevice
#  define gpuStreamCreate         cudaStreamCreate
#  define gpuStreamDestroy        cudaStreamDestroy
#  define gpuStreamSynchronize    cudaStreamSynchronize
#  define gpuStream_t             cudaStream_t
#  define gpuSuccess              cudaSuccess
#  define QZ_GPU_NAME_PREFIX      "cuda"
#endif

namespace qz {
namespace {

// ------------------------------------------------------------------
// block layouts
//
// The structs come straight from the CPU headers, so there is only one
// description of each layout in the tree and the two paths cannot drift.
// ------------------------------------------------------------------

constexpr int QK4_0  = QZ_QK4_0;
constexpr int QK4_1  = QZ_QK4_1;
constexpr int QK5_0  = QZ_QK5_0;
constexpr int QK5_1  = QZ_QK5_1;
constexpr int QK8_0  = QZ_QK8_0;
constexpr int QK4_NL = QZ_QK4_NL;

typedef qz_blk_q4_0   blk_q4_0;
typedef qz_blk_q4_1   blk_q4_1;
typedef qz_blk_q5_0   blk_q5_0;
typedef qz_blk_q5_1   blk_q5_1;
typedef qz_blk_q8_0   blk_q8_0;
typedef qz_blk_iq4_nl blk_iq4_nl;

__constant__ int8_t k_iq4_values[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                            1,   13,  25,  38,  53,  69,  89, 113 };

__device__ __forceinline__ uint16_t f32_to_f16_bits(float f) {
    return __half_as_ushort(__float2half(f));
}

// round-to-nearest-even fp32 -> bf16, matching qz_f2bf() on the CPU side
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffff) > 0x7f800000) { // nan
        return (uint16_t) ((u >> 16) | 64); // force to quiet
    }
    return (uint16_t) ((u + (0x7fff + ((u >> 16) & 1))) >> 16);
}

// ------------------------------------------------------------------
// per-block encoders
//
// One thread encodes one block. The fits below are the device versions of
// the ones in src/kernels/qz_common.h - same weighting, same alternation
// between rounding and re-solving the scale, same set of starting points -
// so selecting an accelerator does not change how a tensor is encoded.
// ------------------------------------------------------------------

#define QZ_DEV_STARTS 3
#define QZ_DEV_ROUNDS 5

__constant__ float k_shrink[QZ_DEV_STARTS] = { 1.0f, 0.94f, 0.88f };

__device__ __forceinline__ int dev_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// magnitude weighting with a floor at half the block's mean square
template <int N>
__device__ void dev_weights(const float * x, float * w) {
    float sumx2 = 0.0f;
    for (int i = 0; i < N; ++i) {
        sumx2 += x[i] * x[i];
    }
    const float floor2 = 0.5f * sumx2 / (float) N;
    for (int i = 0; i < N; ++i) {
        w[i] = sqrtf(x[i] * x[i] + floor2);
    }
}

// x[i] ~= s * q[i], q integer in [qlo, qhi]
template <int N>
__device__ float dev_fit_sym(const float * x, const float * w, int qlo, int qhi, int8_t * q) {
    float amax = 0.0f, xext = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) {
            amax = a;
            xext = x[i];
        }
    }
    if (!(amax > 0.0f)) {
        for (int i = 0; i < N; ++i) {
            q[i] = 0;
        }
        return 0.0f;
    }

    const float ends[2] = { xext / (float) qlo, xext / (float) qhi };
    int8_t cur[N];
    float best_s = 0.0f, best_sse = -1.0f;

    for (int t = 0; t < 2 * QZ_DEV_STARTS; ++t) {
        float s = ends[t % 2] * k_shrink[t / 2];
        if (!(fabsf(s) > 0.0f) || !isfinite(s)) {
            continue;
        }

        for (int it = 0; it < QZ_DEV_ROUNDS; ++it) {
            const float is = 1.0f / s;
            float sse = 0.0f, num = 0.0f, den = 0.0f;

            for (int i = 0; i < N; ++i) {
                const int k = dev_clamp(__float2int_rn(x[i] * is), qlo, qhi);
                cur[i] = (int8_t) k;
                const float e = x[i] - s * (float) k;
                sse += w[i] * e * e;
                num += w[i] * x[i] * (float) k;
                den += w[i] * (float) k * (float) k;
            }

            if (best_sse < 0.0f || sse < best_sse) {
                best_sse = sse;
                best_s = s;
                for (int i = 0; i < N; ++i) {
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
template <int N>
__device__ float dev_fit_min(const float * x, const float * w, int qmax, uint8_t * q, float * offset) {
    float lo = x[0], hi = x[0];
    for (int i = 1; i < N; ++i) {
        lo = fminf(lo, x[i]);
        hi = fmaxf(hi, x[i]);
    }
    lo = fminf(lo, 0.0f);
    hi = fmaxf(hi, 0.0f);

    if (!(hi > lo)) {
        for (int i = 0; i < N; ++i) {
            q[i] = 0;
        }
        *offset = 0.0f;
        return 0.0f;
    }

    uint8_t cur[N];
    float best_s = 0.0f, best_o = -lo, best_sse = -1.0f;

    for (int t = 0; t < QZ_DEV_STARTS * QZ_DEV_STARTS; ++t) {
        const float lo_t = lo * k_shrink[t % QZ_DEV_STARTS];
        const float hi_t = hi * k_shrink[t / QZ_DEV_STARTS];

        float s = (hi_t - lo_t) / (float) qmax;
        float o = fmaxf(-lo_t, 0.0f);
        if (!(s > 0.0f)) {
            continue;
        }

        for (int it = 0; it < QZ_DEV_ROUNDS; ++it) {
            const float is = 1.0f / s;
            float sse = 0.0f;
            float sq2 = 0.0f, sq = 0.0f, sw = 0.0f, sxq = 0.0f, sx = 0.0f;

            for (int i = 0; i < N; ++i) {
                const int k = dev_clamp(__float2int_rn((x[i] + o) * is), 0, qmax);
                cur[i] = (uint8_t) k;
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
                for (int i = 0; i < N; ++i) {
                    q[i] = cur[i];
                }
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

__device__ __forceinline__ int dev_nearest_sorted(const int8_t * tab, int n, float v) {
    if (v <= (float) tab[0]) {
        return 0;
    }
    if (v >= (float) tab[n - 1]) {
        return n - 1;
    }
    int lo = 0, hi = n - 1;
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

__device__ void quantize_block_q4_0(const float * x, blk_q4_0 * y) {
    float  w[QK4_0];
    int8_t q[QK4_0];
    dev_weights<QK4_0>(x, w);
    const float d = dev_fit_sym<QK4_0>(x, w, -8, 7, q);

    y->d = f32_to_f16_bits(d);
    for (int j = 0; j < QK4_0 / 2; ++j) {
        y->qs[j] = (uint8_t) ((q[j] + 8) | ((q[j + QK4_0 / 2] + 8) << 4));
    }
}

__device__ void quantize_block_q5_0(const float * x, blk_q5_0 * y) {
    float  w[QK5_0];
    int8_t q[QK5_0];
    dev_weights<QK5_0>(x, w);
    const float d = dev_fit_sym<QK5_0>(x, w, -16, 15, q);

    y->d = f32_to_f16_bits(d);

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0 / 2; ++j) {
        const uint32_t q0 = (uint32_t) (q[j] + 16);
        const uint32_t q1 = (uint32_t) (q[j + QK5_0 / 2] + 16);
        y->qs[j] = (uint8_t) ((q0 & 0xf) | ((q1 & 0xf) << 4));
        qh |= ((q0 >> 4) & 1u) << j;
        qh |= ((q1 >> 4) & 1u) << (j + QK5_0 / 2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

__device__ void quantize_block_q4_1(const float * x, blk_q4_1 * y) {
    float   w[QK4_1];
    uint8_t q[QK4_1];
    float   o;
    dev_weights<QK4_1>(x, w);
    const float d = dev_fit_min<QK4_1>(x, w, 15, q, &o);

    y->d = f32_to_f16_bits(d);
    y->m = f32_to_f16_bits(-o);
    for (int j = 0; j < QK4_1 / 2; ++j) {
        y->qs[j] = (uint8_t) (q[j] | (q[j + QK4_1 / 2] << 4));
    }
}

__device__ void quantize_block_q5_1(const float * x, blk_q5_1 * y) {
    float   w[QK5_1];
    uint8_t q[QK5_1];
    float   o;
    dev_weights<QK5_1>(x, w);
    const float d = dev_fit_min<QK5_1>(x, w, 31, q, &o);

    y->d = f32_to_f16_bits(d);
    y->m = f32_to_f16_bits(-o);

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1 / 2; ++j) {
        const uint32_t q0 = q[j];
        const uint32_t q1 = q[j + QK5_1 / 2];
        y->qs[j] = (uint8_t) ((q0 & 0xf) | ((q1 & 0xf) << 4));
        qh |= ((q0 >> 4) & 1u) << j;
        qh |= ((q1 >> 4) & 1u) << (j + QK5_1 / 2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

__device__ void quantize_block_q8_0(const float * x, blk_q8_0 * y) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    y->d = f32_to_f16_bits(d);
    for (int j = 0; j < QK8_0; ++j) {
        y->qs[j] = (int8_t) __float2int_rn(x[j] * id);
    }
}

__device__ void quantize_block_iq4_nl(const float * x, blk_iq4_nl * y) {
    float w[QK4_NL];
    dev_weights<QK4_NL>(x, w);

    float amax = 0.0f, xext = 0.0f;
    for (int j = 0; j < QK4_NL; ++j) {
        const float a = fabsf(x[j]);
        if (a > amax) {
            amax = a;
            xext = x[j];
        }
    }

    uint8_t k[QK4_NL];
    float best_d = 0.0f;

    if (amax > 0.0f) {
        const float ends[2] = { xext / (float) k_iq4_values[0], xext / (float) k_iq4_values[15] };
        uint8_t cur[QK4_NL];
        float best_sse = -1.0f;

        for (int t = 0; t < 2 * QZ_DEV_STARTS; ++t) {
            float s = ends[t % 2] * k_shrink[t / 2];
            if (!(fabsf(s) > 0.0f) || !isfinite(s)) {
                continue;
            }

            for (int it = 0; it < QZ_DEV_ROUNDS; ++it) {
                const float is = 1.0f / s;
                float sse = 0.0f, num = 0.0f, den = 0.0f;

                for (int j = 0; j < QK4_NL; ++j) {
                    const int idx = dev_nearest_sorted(k_iq4_values, 16, x[j] * is);
                    const float v = (float) k_iq4_values[idx];
                    const float e = x[j] - s * v;
                    cur[j] = (uint8_t) idx;
                    sse += w[j] * e * e;
                    num += w[j] * x[j] * v;
                    den += w[j] * v * v;
                }

                if (best_sse < 0.0f || sse < best_sse) {
                    best_sse = sse;
                    best_d = s;
                    for (int j = 0; j < QK4_NL; ++j) {
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
        for (int j = 0; j < QK4_NL; ++j) {
            k[j] = 8; // codebook entry closest to zero
        }
    }

    y->d = f32_to_f16_bits(best_d);
    for (int j = 0; j < QK4_NL / 2; ++j) {
        y->qs[j] = (uint8_t) (k[j] | (k[j + QK4_NL / 2] << 4));
    }
}

// ------------------------------------------------------------------
// kernels: one thread per quant block / element
// ------------------------------------------------------------------

template <typename block_t, int qk, void (*quantize_block)(const float *, block_t *)>
__global__ void k_quantize_blocks(const float * __restrict__ x, block_t * __restrict__ y, int64_t nblocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nblocks) {
        quantize_block(x + i * qk, y + i);
    }
}

__global__ void k_f32_to_f16(const float * __restrict__ x, uint16_t * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = f32_to_f16_bits(x[i]);
    }
}

__global__ void k_f32_to_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = f32_to_bf16_bits(x[i]);
    }
}

size_t gpu_row_size(qz_type type, int64_t n_per_row) {
    switch (type) {
        case QZ_TYPE_F16:
        case QZ_TYPE_BF16:   return (size_t) n_per_row * 2;
        case QZ_TYPE_Q4_0:   return (size_t) n_per_row / QK4_0  * sizeof(blk_q4_0);
        case QZ_TYPE_Q4_1:   return (size_t) n_per_row / QK4_1  * sizeof(blk_q4_1);
        case QZ_TYPE_Q5_0:   return (size_t) n_per_row / QK5_0  * sizeof(blk_q5_0);
        case QZ_TYPE_Q5_1:   return (size_t) n_per_row / QK5_1  * sizeof(blk_q5_1);
        case QZ_TYPE_Q8_0:   return (size_t) n_per_row / QK8_0  * sizeof(blk_q8_0);
        case QZ_TYPE_IQ4_NL: return (size_t) n_per_row / QK4_NL * sizeof(blk_iq4_nl);
        default:               return 0;
    }
}

class gpu_converter : public device_converter {
public:
    gpu_converter(int device, std::string name) : device_(device), name_(std::move(name)) {}

    ~gpu_converter() override {
        if (d_src_) gpuFree(d_src_);
        if (d_dst_) gpuFree(d_dst_);
        if (stream_) gpuStreamDestroy(stream_);
    }

    bool init(std::string & error) {
        if (!check(gpuSetDevice(device_), "set device", error)) {
            return false;
        }
        return check(gpuStreamCreate(&stream_), "create stream", error);
    }

    const std::string & name() const override {
        return name_;
    }

    bool supports(qz_type dst_type) const override {
        return gpu_row_size(dst_type, 32) != 0;
    }

    bool convert(const float * src, qz_type dst_type, void * dst, int64_t nrows, int64_t n_per_row,
                 std::string & error) override {
        if (!supports(dst_type)) {
            error = "unsupported type";
            return false;
        }
        if (!check(gpuSetDevice(device_), "set device", error)) {
            return false;
        }

        // bound temporary device memory; a row is never split because
        // quantized blocks are row-oriented
        constexpr size_t max_chunk_input_bytes = 64u * 1024u * 1024u;
        const size_t  row_bytes     = (size_t) n_per_row * sizeof(float);
        const size_t  dst_row_bytes = gpu_row_size(dst_type, n_per_row);
        const int64_t rows_per_chunk =
            std::max<int64_t>(1, std::min<int64_t>(nrows, max_chunk_input_bytes / std::max<size_t>(1, row_bytes)));

        if (!reserve(rows_per_chunk * row_bytes, rows_per_chunk * dst_row_bytes, error)) {
            return false;
        }

        for (int64_t row0 = 0; row0 < nrows; row0 += rows_per_chunk) {
            const int64_t chunk_rows = std::min(rows_per_chunk, nrows - row0);
            const int64_t n          = chunk_rows * n_per_row;

            if (!check(gpuMemcpyAsync(d_src_, src + row0 * n_per_row, n * sizeof(float), gpuMemcpyHostToDevice,
                                      stream_), "copy to device", error)) {
                return false;
            }

            constexpr int threads = 256;
            switch (dst_type) {
                case QZ_TYPE_F16:
                    k_f32_to_f16<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
                        (const float *) d_src_, (uint16_t *) d_dst_, n);
                    break;
                case QZ_TYPE_BF16:
                    k_f32_to_bf16<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
                        (const float *) d_src_, (uint16_t *) d_dst_, n);
                    break;
                case QZ_TYPE_Q4_0:
                    launch_blocks<blk_q4_0, QK4_0, quantize_block_q4_0>(n);
                    break;
                case QZ_TYPE_Q4_1:
                    launch_blocks<blk_q4_1, QK4_1, quantize_block_q4_1>(n);
                    break;
                case QZ_TYPE_Q5_0:
                    launch_blocks<blk_q5_0, QK5_0, quantize_block_q5_0>(n);
                    break;
                case QZ_TYPE_Q5_1:
                    launch_blocks<blk_q5_1, QK5_1, quantize_block_q5_1>(n);
                    break;
                case QZ_TYPE_Q8_0:
                    launch_blocks<blk_q8_0, QK8_0, quantize_block_q8_0>(n);
                    break;
                case QZ_TYPE_IQ4_NL:
                    launch_blocks<blk_iq4_nl, QK4_NL, quantize_block_iq4_nl>(n);
                    break;
                default:
                    error = "unsupported type";
                    return false;
            }
            if (!check(gpuGetLastError(), "launch kernel", error)) {
                return false;
            }

            if (!check(gpuMemcpyAsync((char *) dst + row0 * dst_row_bytes, d_dst_, chunk_rows * dst_row_bytes,
                                      gpuMemcpyDeviceToHost, stream_), "copy from device", error)) {
                return false;
            }
            if (!check(gpuStreamSynchronize(stream_), "synchronize", error)) {
                return false;
            }
        }
        return true;
    }

private:
    template <typename block_t, int qk, void (*fn)(const float *, block_t *)>
    void launch_blocks(int64_t n) {
        constexpr int threads = 256;
        const int64_t nblocks = n / qk;
        k_quantize_blocks<block_t, qk, fn><<<(nblocks + threads - 1) / threads, threads, 0, stream_>>>(
            (const float *) d_src_, (block_t *) d_dst_, nblocks);
    }

    bool reserve(size_t src_bytes, size_t dst_bytes, std::string & error) {
        if (src_bytes > src_cap_) {
            if (d_src_) gpuFree(d_src_);
            d_src_   = nullptr;
            src_cap_ = 0;
            if (!check(gpuMalloc(&d_src_, src_bytes), "allocate device memory", error)) {
                return false;
            }
            src_cap_ = src_bytes;
        }
        if (dst_bytes > dst_cap_) {
            if (d_dst_) gpuFree(d_dst_);
            d_dst_   = nullptr;
            dst_cap_ = 0;
            if (!check(gpuMalloc(&d_dst_, dst_bytes), "allocate device memory", error)) {
                return false;
            }
            dst_cap_ = dst_bytes;
        }
        return true;
    }

    static bool check(gpuError_t err, const char * what, std::string & error) {
        if (err == gpuSuccess) {
            return true;
        }
        error = std::string(what) + ": " + gpuGetErrorString(err);
        return false;
    }

    int         device_;
    std::string name_;
    gpuStream_t stream_ = nullptr;
    void *      d_src_  = nullptr;
    void *      d_dst_  = nullptr;
    size_t      src_cap_ = 0;
    size_t      dst_cap_ = 0;
};

} // namespace

int gpu_device_count() {
    int count = 0;
    if (gpuGetDeviceCount(&count) != gpuSuccess) {
        return 0;
    }
    return count;
}

std::string gpu_device_name(int index) {
    return QZ_GPU_NAME_PREFIX + std::to_string(index);
}

std::string gpu_device_description(int index) {
    gpuDeviceProp prop;
    if (gpuGetDeviceProperties(&prop, index) != gpuSuccess) {
        return "unknown GPU";
    }
    return prop.name;
}

std::unique_ptr<device_converter> gpu_device_open(int index, std::string & error) {
    auto converter = std::make_unique<gpu_converter>(index, gpu_device_name(index));
    if (!converter->init(error)) {
        return nullptr;
    }
    return converter;
}

} // namespace qz
