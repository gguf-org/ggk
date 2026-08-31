#pragma once

// CUDA / ROCm(HIP) accelerator backend (implemented in device_gpu.cu, which is
// compiled once per enabled GPU runtime). Only linked when QZ_USE_CUDA or
// QZ_USE_HIP is defined.

#include "device.h"

#include <memory>
#include <string>

namespace qz {

int         gpu_device_count();
std::string gpu_device_name(int index);        // "cuda0"/"rocm0", ...
std::string gpu_device_description(int index);

std::unique_ptr<device_converter> gpu_device_open(int index, std::string & error);

} // namespace qz
