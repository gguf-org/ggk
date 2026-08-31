cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.
project("ggk" C CXX)

#
# ggk: the unified engine — LLM server + diffusion + quantizer, all compiled
# from one tree against one shared copy of the gk compute kernels.
#
#   gk/          the gk compute kernels (CPU + optional GPU backends), and the
#                ggml compatibility layer the runtimes above still speak
#   src/         the GGUF model runtime (libllama)
#   common/      argument parsing, chat templates, sampling, logging
#   mtmd/        multimodal (vision/audio) projector support
#   app/         the gguf-server HTTP server
#   ui/          embedded web UI assets (optional, empty by default)
#   diffusion/   the diffusion runtime + `diffusion` CLI (links the same gk)
#   quantizer/   the standalone quantizer shared library (own quant kernels)
#   thirdparty/  cpp-httplib, nlohmann/json, stb, miniaudio, subprocess.h
#
# There is no vendored llama.cpp checkout and no ggml at all: the graphs the
# runtimes build are evaluated by gk, and everything is compiled from the
# sources in this directory, in a single build.
#
# The quantizer keeps its own qz_* codec rather than sharing the runtimes'
# code path — encoding is a different job from inference — but gk compiles
# those same qz_ sources in, so the two can never disagree about a block.
#

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_STANDARD          11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if (MINGW OR EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
else()
    set(BUILD_SHARED_LIBS_DEFAULT ON)
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

#
# Options
#

# backends - these simply forward to the matching gk option, and apply to the
# whole engine: the server, the diffusion runtime and the multimodal
# projectors all evaluate their graphs on the one gk build below.
option(GGUF_SERVER_CUDA     "gguf-server: CUDA backend"                     OFF)
option(GGUF_SERVER_HIP      "gguf-server: ROCm/HIP backend"                 OFF)
option(GGUF_SERVER_METAL    "gguf-server: Metal backend"                    ${APPLE})
option(GGUF_SERVER_VULKAN   "gguf-server: Vulkan backend"                   OFF)

# features
option(GGUF_SERVER_OPENSSL  "gguf-server: use OpenSSL for HTTPS model downloads"    ON)
option(GGUF_SERVER_INSTALL  "gguf-server: install the binary on `cmake --install`"  ON)

# subprocess spawning powers `--models` router mode; not available everywhere
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID
        OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR EMSCRIPTEN)
    set(GGUF_SERVER_SUBPROCESS_DEFAULT OFF)
else()
    set(GGUF_SERVER_SUBPROCESS_DEFAULT ON)
endif()
option(GGUF_SERVER_SUBPROCESS "gguf-server: support router mode (spawns child servers)" ${GGUF_SERVER_SUBPROCESS_DEFAULT})

# diagnostics
option(GGUF_SERVER_ALL_WARNINGS      "gguf-server: enable all compiler warnings" ON)
option(GGUF_SERVER_FATAL_WARNINGS    "gguf-server: enable -Werror flag"          OFF)
option(GGUF_SERVER_SANITIZE_THREAD    "gguf-server: enable thread sanitizer"     OFF)
option(GGUF_SERVER_SANITIZE_ADDRESS   "gguf-server: enable address sanitizer"    OFF)
option(GGUF_SERVER_SANITIZE_UNDEFINED "gguf-server: enable undefined sanitizer"  OFF)

# the vendored sources still use the LLAMA_* names internally
set(LLAMA_ALL_WARNINGS       ${GGUF_SERVER_ALL_WARNINGS})
set(LLAMA_FATAL_WARNINGS     ${GGUF_SERVER_FATAL_WARNINGS})
set(LLAMA_SANITIZE_THREAD    ${GGUF_SERVER_SANITIZE_THREAD})
set(LLAMA_SANITIZE_ADDRESS   ${GGUF_SERVER_SANITIZE_ADDRESS})
set(LLAMA_SANITIZE_UNDEFINED ${GGUF_SERVER_SANITIZE_UNDEFINED})
set(LLAMA_SUBPROCESS         ${GGUF_SERVER_SUBPROCESS})
set(LLAMA_OPENSSL            ${GGUF_SERVER_OPENSSL})

if (GGUF_SERVER_CUDA)
    message(STATUS "ggk: CUDA backend enabled")
    set(GK_CUDA ON)
endif()
if (GGUF_SERVER_HIP)
    message(STATUS "ggk: HIP backend enabled")
    set(GK_HIP ON)
    # the device objects must be position-independent, or the default-PIE
    # executable link fails on distros that default to PIE
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if (GGUF_SERVER_METAL)
    message(STATUS "ggk: Metal backend enabled")
    set(GK_METAL ON)
endif()
if (GGUF_SERVER_VULKAN)
    message(STATUS "ggk: Vulkan backend enabled")
    set(GK_VULKAN ON)
endif()

#
# Version / build info
#

include(build-info)
include(common)

if (NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER ${BUILD_NUMBER})
endif()
if (NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT ${BUILD_COMMIT})
endif()
set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})

#
# gk options
#
# gk's own test programs check differentially against a reference ggml, which
# this tree does not carry; they are built from the standalone gk checkout.
#

set(GK_BUILD_TESTS OFF CACHE BOOL "gk: build the test programs" FORCE)

#
# Build
#

# Diffusion models carry tensor names longer than ggml's default 64-byte cap.
# The compat layer's ggml_tensor and gk's gk_tensor are layout-identical by
# construction — a static assert in ggml-compat-impl.h checks it — so the name
# field has to be widened on both sides together, globally, before anything
# compiles. Widening only one is what makes that assert fire.
add_compile_definitions(GGML_MAX_NAME=128 GK_MAX_NAME=128)

add_subdirectory(gk/compat)          # gk, and the ggml API on top of it
add_subdirectory(src)                # llama
add_subdirectory(thirdparty/cpp-httplib)
add_subdirectory(common)             # llama-common
add_subdirectory(mtmd)               # multimodal
add_subdirectory(ui)                 # embedded web UI assets
add_subdirectory(app)                # the gguf-server binary
add_subdirectory(diffusion)          # diffusion runtime + CLI
add_subdirectory(quantizer)          # quantizer shared library
