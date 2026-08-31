# diffusion runtime + CLI, built as part of the unified ggk engine.
#
# This used to be a standalone project with its own vendored ggml; in the
# unified tree it links the shared `ggml` target that <engine>/gk/compat
# builds on top of gk. Backend selection (CUDA/HIP/Metal/Vulkan) is global to
# the engine via the GK_* options, so the old SD_CUDA-style switches are gone.
# GGML_MAX_NAME/GK_MAX_NAME=128 (long diffusion tensor names) are set
# engine-wide so every consumer agrees on the tensor struct layout.

option(SD_BUILD_CLI         "diffusion: build the diffusion executable"   ON)
option(SD_BUILD_SHARED_LIBS "diffusion: build shared library"             OFF)

if(APPLE)
    function(sd_set_macos_rpaths target)
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL "EXECUTABLE")
            set(runtime_paths "@executable_path" "@executable_path/../lib")
        elseif(target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY")
            set(runtime_paths "@loader_path" "@loader_path/../lib")
            set_target_properties(${target} PROPERTIES
                MACOSX_RPATH ON
                INSTALL_NAME_DIR "@rpath"
                BUILD_WITH_INSTALL_NAME_DIR ON
            )
        else()
            return()
        endif()
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${runtime_paths}"
            INSTALL_RPATH "${runtime_paths}"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endfunction()
endif()

set(SD_LIB diffusion)

file(GLOB SD_LIB_SOURCES CONFIGURE_DEPENDS
    "src/*.h"
    "src/*.cpp"
    "src/*.hpp"
    "src/conditioning/*.h"
    "src/conditioning/*.cpp"
    "src/conditioning/*.hpp"
    "src/core/*.h"
    "src/core/*.cpp"
    "src/core/*.hpp"
    "src/extensions/*.h"
    "src/extensions/*.cpp"
    "src/extensions/*.hpp"
    "src/model/*/*.h"
    "src/model/*/*.cpp"
    "src/model/*/*.hpp"
    "src/runtime/*.h"
    "src/runtime/*.cpp"
    "src/runtime/*.hpp"
    "src/model_io/*.h"
    "src/model_io/*.cpp"
    "src/tokenizers/*.h"
    "src/tokenizers/*.cpp"
    "src/tokenizers/vocab/*.h"
    "src/tokenizers/vocab/*.cpp"
)

set(DIFFUSION_BUILD_VERSION 2026.8.31)
set(DIFFUSION_BUILD_COMMIT ggk)
message(STATUS "diffusion version ${DIFFUSION_BUILD_VERSION} (${DIFFUSION_BUILD_COMMIT})")

set_property(
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp
  APPEND PROPERTY COMPILE_DEFINITIONS
  SDCPP_BUILD_COMMIT=${DIFFUSION_BUILD_COMMIT} SDCPP_BUILD_VERSION=${DIFFUSION_BUILD_VERSION}
)

if(SD_BUILD_SHARED_LIBS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_library(${SD_LIB} SHARED ${SD_LIB_SOURCES})
    add_definitions(-DSD_BUILD_SHARED_LIB)
    target_compile_definitions(${SD_LIB} PRIVATE -DSD_BUILD_DLL)
else()
    add_library(${SD_LIB} STATIC ${SD_LIB_SOURCES})
endif()

if(APPLE)
    sd_set_macos_rpaths(${SD_LIB})
endif()

target_link_libraries(${SD_LIB} PUBLIC ggml)
target_include_directories(${SD_LIB} PUBLIC . src include)
target_include_directories(${SD_LIB} PRIVATE src/core)
target_include_directories(${SD_LIB} PUBLIC thirdparty)
target_compile_features(${SD_LIB} PUBLIC c_std_11 cxx_std_17)

if(SD_BUILD_CLI)
    add_subdirectory(cli)
endif()
