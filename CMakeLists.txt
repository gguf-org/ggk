# Builds the unified ggk engine (LLM server + diffusion CLI + quantizer
# library) from the vendored source tree in ONE configure/build, and installs
# each artifact into the panel package that spawns/loads it:
#
#   gguf-server     -> src/ggk/server/bin      (spawned by the server panel)
#   diffusion       -> src/ggk/diffuser/bin    (spawned by the diffuser panel)
#   gguf_quantizer  -> src/ggk/editor/lib      (ctypes-loaded by the editor panel)
#
# The engine tree shares a single gk/ compute library between all three, so
# the kernels compile once. Modeled on the wrappers the three standalone
# packages (gguf-server, gguf-diffusion, gguf-editor) used.
cmake_minimum_required(VERSION 3.15)
project(ggk C CXX)

option(GGK_BUILD "Build the unified engine and install it alongside the python package" ON)

if(NOT GGK_BUILD)
    return()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Locate the engine source tree ────────────────────────────────────────────
set(GGK_ENGINE_DIR "" CACHE PATH "Path to the unified ggk engine source tree")
if(NOT GGK_ENGINE_DIR AND DEFINED ENV{GGK_ENGINE_DIR})
    set(GGK_ENGINE_DIR "$ENV{GGK_ENGINE_DIR}")
endif()
if(NOT GGK_ENGINE_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/engine/CMakeLists.txt")
    set(GGK_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/engine")
endif()
if(NOT GGK_ENGINE_DIR OR NOT EXISTS "${GGK_ENGINE_DIR}/app/CMakeLists.txt")
    message(FATAL_ERROR
        "ggk engine source not found. Provide it via one of:\n"
        "  - the vendored copy at vendor/engine\n"
        "  - -DGGK_ENGINE_DIR=/path/to/engine or the same env var")
endif()
get_filename_component(GGK_ENGINE_DIR "${GGK_ENGINE_DIR}" ABSOLUTE)
message(STATUS "ggk: engine source at ${GGK_ENGINE_DIR}")

# ── Option helper ────────────────────────────────────────────────────────────
# Values are read from the environment as well as -D, because `pip install`
# makes environment variables far easier to pass than CMake defines:
#   GGK_CUDA=1 pip install .
#   CMAKE_ARGS="-DGGK_CUDA=ON" pip install .
function(ggk_option name docstring default)
    if(DEFINED ENV{${name}} AND NOT "$ENV{${name}}" STREQUAL "")
        set(default "$ENV{${name}}")
    endif()
    option(${name} "${docstring}" "${default}")
endfunction()

# ── Windows / MinGW toolchain fixes ──────────────────────────────────────────
if(WIN32 AND NOT MSVC)
    # MinGW-w64's headers pin _WIN32_WINNT to a Windows 7-era value before the
    # engine gets a chance to set it; cpp-httplib then hard-errors and
    # ::CreateFile2 goes undeclared. Set it up front, matching what the engine
    # expects everywhere else.
    add_compile_definitions(_WIN32_WINNT=0x0A00 WINVER=0x0A00 NTDDI_VERSION=0x0A000000)

    # A MinGW-built binary otherwise needs libstdc++-6.dll, libgcc_s_seh-1.dll
    # and libwinpthread-1.dll from the toolchain at run time. The wheel ships
    # self-contained binaries and Python spawns them without MSYS2 on PATH, so
    # link the runtime in.
    ggk_option(GGK_MINGW_STATIC
        "Statically link the MinGW runtime into the engine binaries" ON)
    if(GGK_MINGW_STATIC)
        # gk runs its own thread pool rather than OpenMP, so -static has no
        # libgomp import library to fight here — unlike the ggml build this
        # tree used to carry, where OpenMP had to be redirected or dropped.
        add_link_options(-static-libgcc -static-libstdc++ -static)
    endif()
endif()

# ── Shape the engine build for a single-file-per-artifact wheel ──────────────
# Static libs (gk, ggml-compat, llama, llama-common, mtmd, diffusion, …)
# linked straight into the executables keep the wheel to standalone binaries
# with no rpath/DLL handling. The quantizer stays a shared library by design
# (ctypes).
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "ggk: static engine for single-file installs" FORCE)

# The engine's own install() rules would put artifacts in the wheel's data
# directory; ours below place them inside the package instead.
set(GGUF_SERVER_INSTALL OFF CACHE BOOL "ggk: install rules are ours" FORCE)

# Router mode (`--models`) spawns child servers via subprocess.h — harmless to
# keep and the GUI's custom-command box can use it.
ggk_option(GGK_SUBPROCESS
    "Support router mode (the server engine spawns child servers)" ON)
set(GGUF_SERVER_SUBPROCESS ${GGK_SUBPROCESS} CACHE BOOL "" FORCE)

# ── HTTPS / OpenSSL ──────────────────────────────────────────────────────────
# Off by default: OpenSSL is only needed to *download* models over HTTPS; the
# panels always hand the engine local file paths, so the dependency buys
# nothing and costs portability (a foreign-ABI OpenSSL next to the toolchain
# breaks the build on Windows).
ggk_option(GGK_OPENSSL
    "Link the server engine against OpenSSL for HTTPS model downloads" OFF)
set(GGUF_SERVER_OPENSSL ${GGK_OPENSSL} CACHE BOOL "" FORCE)

# ── GPU / accelerator backends ───────────────────────────────────────────────
# One switch per backend drives the whole engine: the shared gk kernels
# (server + diffusion) and, for CUDA/HIP, the quantizer's own device kernels.
# Anything else gk supports can still be passed straight through as -DGK_*.
#
# A CUDA build works its own architecture list out from nvcc and the installed
# GPUs, and embeds PTX for the newest, so an unlisted card JITs rather than
# failing. A wheel built on one machine for another should still say what it is
# targeting: -DGK_CUDA_ARCHITECTURES="75;86;89;120".
ggk_option(GGK_CUDA   "Build the engine with the CUDA backend"   OFF)
ggk_option(GGK_HIP    "Build the engine with the HIP backend"    OFF)
ggk_option(GGK_VULKAN "Build the engine with the Vulkan backend" OFF)
if(APPLE)  # matches the engine's own default; APPLE is "" elsewhere, not a valid default
    set(_gguf_metal_default ON)
else()
    set(_gguf_metal_default OFF)
endif()
ggk_option(GGK_METAL  "Build the engine with the Metal backend"  ${_gguf_metal_default})

set(GGUF_SERVER_CUDA   ${GGK_CUDA}   CACHE BOOL "" FORCE)
set(GGUF_SERVER_HIP    ${GGK_HIP}    CACHE BOOL "" FORCE)
set(GGUF_SERVER_VULKAN ${GGK_VULKAN} CACHE BOOL "" FORCE)
set(GGUF_SERVER_METAL  ${GGK_METAL}  CACHE BOOL "" FORCE)

# The quantizer's GPU path is its own small kernel set, not gk's; CUDA/HIP
# follow the engine-wide switch. Its Metal path stays opt-in (QUANTIZER_METAL),
# matching the standalone gguf-editor defaults.
set(QUANTIZER_CUDA ${GGK_CUDA} CACHE BOOL "" FORCE)
set(QUANTIZER_HIP  ${GGK_HIP}  CACHE BOOL "" FORCE)

set(GGK_ENABLED_BACKENDS "")
foreach(backend IN ITEMS CUDA HIP VULKAN METAL)
    if(GGK_${backend})
        list(APPEND GGK_ENABLED_BACKENDS ${backend})
    endif()
endforeach()
if(GGK_ENABLED_BACKENDS)
    string(REPLACE ";" ", " _backend_list "${GGK_ENABLED_BACKENDS}")
    message(STATUS "ggk: accelerator backends requested: ${_backend_list}")
else()
    message(STATUS "ggk: CPU-only build (enable e.g. -DGGK_CUDA=ON for GPU)")
endif()

# ── Embedded web UI ──────────────────────────────────────────────────────────
# The server engine embeds whatever static assets live in GGUF_SERVER_UI_DIR
# (default <engine>/ui/dist). The tree ships none, so the engine is API-only —
# irrelevant here, because this package serves its own GUI.
if(NOT DEFINED CACHE{GGUF_SERVER_UI_DIR} AND DEFINED ENV{GGUF_SERVER_UI_DIR})
    set(GGUF_SERVER_UI_DIR "$ENV{GGUF_SERVER_UI_DIR}" CACHE PATH
        "directory with web UI assets to embed into the server engine")
endif()

# EXCLUDE_FROM_ALL keeps every target the engine defines out of the default
# build — the dependencies below pull in the three shipped artifacts and just
# the libraries they link against.
add_subdirectory("${GGK_ENGINE_DIR}" "${CMAKE_BINARY_DIR}/ggk-engine" EXCLUDE_FROM_ALL)

foreach(required_target IN ITEMS gguf-server diffusion-cli gguf_quantizer)
    if(NOT TARGET ${required_target})
        message(FATAL_ERROR "engine did not define the ${required_target} target — is ${GGK_ENGINE_DIR} the unified ggk source tree?")
    endif()
endforeach()

add_custom_target(ggk_engine ALL)
add_dependencies(ggk_engine gguf-server diffusion-cli gguf_quantizer)

# ── Install into the panel packages ──────────────────────────────────────────
# Install into the wheel (SKBUILD_PLATLIB_DIR) and also into the source tree
# so editable installs (`pip install -e .`) find the artifacts — the same dual
# install the three standalone packages used.
install(
    PROGRAMS $<TARGET_FILE:gguf-server>
    DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/ggk/server/bin"
)
install(
    PROGRAMS $<TARGET_FILE:diffusion-cli>
    DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/ggk/diffuser/bin"
)
install(
    TARGETS gguf_quantizer
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/ggk/editor/lib"
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/ggk/editor/lib"
    ARCHIVE DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/ggk/editor/lib"
)
if(DEFINED SKBUILD_PLATLIB_DIR)
    install(
        PROGRAMS $<TARGET_FILE:gguf-server>
        DESTINATION "${SKBUILD_PLATLIB_DIR}/ggk/server/bin"
    )
    install(
        PROGRAMS $<TARGET_FILE:diffusion-cli>
        DESTINATION "${SKBUILD_PLATLIB_DIR}/ggk/diffuser/bin"
    )
    install(
        TARGETS gguf_quantizer
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/ggk/editor/lib"
        RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/ggk/editor/lib"
        ARCHIVE DESTINATION "${SKBUILD_PLATLIB_DIR}/ggk/editor/lib"
    )
endif()
