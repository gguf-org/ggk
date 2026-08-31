#pragma once

// Internal device/accelerator abstraction.
//
// The CPU path (the kernels in src/kernels/) is always available
// and needs no converter. Accelerator backends (CUDA/ROCm/Metal) implement
// device_converter with fast f32 -> dst quantization kernels for a subset of
// types; tensors with unsupported target types transparently fall back to the
// CPU kernels.

#include "kernels/qz_quant.h"

#include <memory>
#include <string>
#include <vector>

namespace qz {

struct device_info {
    std::string name;        // "cpu", "cuda0", "rocm0", "metal", ...
    std::string description;
};

class device_converter {
public:
    virtual ~device_converter() = default;

    virtual const std::string & name() const = 0;

    // true if this device has a f32 -> dst_type quantization kernel
    virtual bool supports(qz_type dst_type) const = 0;

    // quantize `nrows` rows of `n_per_row` f32 values into dst.
    // returns false and sets `error` on a runtime failure (the caller then
    // falls back to the CPU kernels for this tensor).
    virtual bool convert(const float * src, qz_type dst_type, void * dst, int64_t nrows,
                         int64_t n_per_row, std::string & error) = 0;
};

// devices available in this build/runtime; index 0 is always "cpu"
const std::vector<device_info> & device_list();

// open an accelerator by its exact name from device_list().
// returns nullptr with empty error for "cpu" (no converter needed),
// nullptr with non-empty error on failure.
std::unique_ptr<device_converter> device_open(const std::string & name, std::string & error);

} // namespace qz
