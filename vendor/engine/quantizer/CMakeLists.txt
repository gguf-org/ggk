# Standalone GGUF quantizer, built as part of the unified ggk engine as a
# shared library (the editor panel loads it via ctypes).
#
# The quant kernels are the quantizer's own (src/kernels/qz_*), written against
# the GGUF on-disk formats rather than shared with the runtimes: encoding is a
# different job from inference, and keeping the two apart means this library
# pulls in nothing from the engine tree. The formats cannot drift apart even
# so, because gk compiles those same qz_* sources in for its decode side - so
# anything produced here loads in the server and the diffusion runtime,
# including the fork's own types (q1_0, q2_0, mxfp4, nvfp4).

option(QUANTIZER_CUDA  "quantizer: CUDA support (NVIDIA GPUs)" OFF)
option(QUANTIZER_HIP   "quantizer: HIP/ROCm support (AMD GPUs)" OFF)
option(QUANTIZER_METAL "quantizer: Metal support (Apple GPUs)" OFF)

if(QUANTIZER_CUDA AND QUANTIZER_HIP)
    message(FATAL_ERROR "QUANTIZER_CUDA and QUANTIZER_HIP are mutually exclusive")
endif()

# Statically link the MSVC CRT so the wheel works on machines without the
# VC++ redistributable (same as the standalone quantizer build).
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
endif()

set(QZ_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(QUANTIZER_SOURCES
    ${QZ_DIR}/src/quantize.cpp
    ${QZ_DIR}/src/quantize.h
    ${QZ_DIR}/src/gguf_io.cpp
    ${QZ_DIR}/src/gguf_io.h
    ${QZ_DIR}/src/safetensors_io.cpp
    ${QZ_DIR}/src/safetensors_io.h
    ${QZ_DIR}/src/device.cpp
    ${QZ_DIR}/src/device.h
    ${QZ_DIR}/src/kernels/qz_pack.c
    ${QZ_DIR}/src/kernels/qz_super.c
    ${QZ_DIR}/src/kernels/qz_lattice.c
    ${QZ_DIR}/src/kernels/qz_decode.c
    ${QZ_DIR}/src/kernels/qz_codebook.c
    ${QZ_DIR}/src/kernels/qz_traits.c
    ${QZ_DIR}/src/kernels/qz_init.cpp
)

if(QUANTIZER_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;120")
    endif()
    list(APPEND QUANTIZER_SOURCES ${QZ_DIR}/src/device_gpu.cu ${QZ_DIR}/src/device_gpu.h)
endif()

if(QUANTIZER_HIP)
    enable_language(HIP)
    list(APPEND QUANTIZER_SOURCES ${QZ_DIR}/src/device_gpu.cu ${QZ_DIR}/src/device_gpu.h)
    set_source_files_properties(${QZ_DIR}/src/device_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(QUANTIZER_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "QUANTIZER_METAL requires macOS")
    endif()
    enable_language(OBJCXX)
    list(APPEND QUANTIZER_SOURCES ${QZ_DIR}/src/device_metal.mm ${QZ_DIR}/src/device_metal.h)
    set_source_files_properties(${QZ_DIR}/src/device_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

add_library(gguf_quantizer SHARED ${QUANTIZER_SOURCES})
target_include_directories(gguf_quantizer PUBLIC ${QZ_DIR}/src)
target_compile_definitions(gguf_quantizer PRIVATE GGUF_QUANTIZER_BUILD PUBLIC GGUF_QUANTIZER_SHARED)

if(QUANTIZER_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(gguf_quantizer PRIVATE QZ_USE_CUDA)
    target_link_libraries(gguf_quantizer PRIVATE CUDA::cudart)
endif()

if(QUANTIZER_HIP)
    find_package(hip REQUIRED)
    target_compile_definitions(gguf_quantizer PRIVATE QZ_USE_HIP)
    target_link_libraries(gguf_quantizer PRIVATE hip::host)
endif()

if(QUANTIZER_METAL)
    target_compile_definitions(gguf_quantizer PRIVATE QZ_USE_METAL)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    target_link_libraries(gguf_quantizer PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})
endif()

find_package(Threads REQUIRED)
target_link_libraries(gguf_quantizer PRIVATE Threads::Threads)

set_target_properties(gguf_quantizer PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(NOT MSVC)
    target_compile_options(gguf_quantizer PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

if(MINGW)
    # -static-libgcc/-static-libstdc++ alone still leave a dynamic dependency
    # on libwinpthread-1.dll, which is not shipped in the wheel; -static folds
    # it (and any other MinGW runtime) into the DLL.
    target_link_options(gguf_quantizer PRIVATE -static -static-libgcc -static-libstdc++)
elseif(UNIX AND NOT APPLE)
    target_link_options(gguf_quantizer PRIVATE -static-libgcc -static-libstdc++)
endif()

if(WIN32 AND TARGET gguf_quantizer)
    set_target_properties(gguf_quantizer PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Keep any shipped runtime deps resolvable next to the library.
if(APPLE)
    set_target_properties(gguf_quantizer PROPERTIES
        INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH TRUE)
elseif(UNIX)
    set_target_properties(gguf_quantizer PROPERTIES
        INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE)
endif()
