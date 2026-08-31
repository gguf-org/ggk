#pragma once

// Apple Metal accelerator backend (implemented in device_metal.mm).
// Only linked when QZ_USE_METAL is defined.

#include "device.h"

#include <memory>
#include <string>

namespace qz {

bool        metal_device_available();
std::string metal_device_description();

std::unique_ptr<device_converter> metal_device_open(std::string & error);

} // namespace qz
