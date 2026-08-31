#include "device.h"

#if defined(QZ_USE_CUDA) || defined(QZ_USE_HIP)
#  include "device_gpu.h"
#endif
#if defined(QZ_USE_METAL)
#  include "device_metal.h"
#endif

namespace qz {

const std::vector<device_info> & device_list() {
    static const std::vector<device_info> devices = [] {
        std::vector<device_info> list;
        list.push_back({ "cpu", "CPU (built-in quantization kernels)" });
#if defined(QZ_USE_CUDA) || defined(QZ_USE_HIP)
        const int n_gpus = gpu_device_count();
        for (int i = 0; i < n_gpus; ++i) {
            list.push_back({ gpu_device_name(i), gpu_device_description(i) });
        }
#endif
#if defined(QZ_USE_METAL)
        if (metal_device_available()) {
            list.push_back({ "metal", metal_device_description() });
        }
#endif
        return list;
    }();
    return devices;
}

std::unique_ptr<device_converter> device_open(const std::string & name, std::string & error) {
    error.clear();
    if (name == "cpu") {
        return nullptr;
    }
#if defined(QZ_USE_CUDA) || defined(QZ_USE_HIP)
    {
        const int n_gpus = gpu_device_count();
        for (int i = 0; i < n_gpus; ++i) {
            if (gpu_device_name(i) == name) {
                return gpu_device_open(i, error);
            }
        }
    }
#endif
#if defined(QZ_USE_METAL)
    if (name == "metal" && metal_device_available()) {
        return metal_device_open(error);
    }
#endif
    error = "device '" + name + "' is not available in this build";
    return nullptr;
}

} // namespace qz
