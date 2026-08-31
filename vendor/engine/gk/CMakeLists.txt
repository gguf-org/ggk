cmake_minimum_required(VERSION 3.14)

project("gk" C CXX)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GK_BUILD_TESTS "gk: build the test programs" ON)

# Device backends. Each is off by default and each is entirely optional: with
# none of them on, gk is the CPU library it has always been, and the device
# registry reports one device. Turning one on adds its sources and its
# discovery hook; nothing above this directory changes.
option(GK_CUDA   "gk: CUDA backend"       OFF)
option(GK_HIP    "gk: ROCm/HIP backend"   OFF)
option(GK_METAL  "gk: Metal backend"      OFF)
option(GK_VULKAN "gk: Vulkan backend"     OFF)

# Which architectures the device code is compiled for. Left empty, gk works
# the list out from nvcc and the installed GPUs and adds PTX for the newest -
# see cmake/gk_cuda_arch.cmake for why the toolchain defaults are not good
# enough. Set it explicitly for a build that has to run somewhere else.
set(GK_CUDA_ARCHITECTURES "" CACHE STRING "gk: CUDA architectures (e.g. 75;86;89;120)")
set(GK_HIP_ARCHITECTURES  "" CACHE STRING "gk: HIP architectures (e.g. gfx1030;gfx1100)")

# Which instruction set the SIMD paths may use.
#
# GK_NATIVE targets the building machine, which is right for development and
# wrong for anything shipped to another machine. A distributable build should
# turn it off and set GK_ARCH_FLAGS to the baseline it promises - or, better,
# grow runtime dispatch, which this does not have yet.
option(GK_NATIVE "gk: build for the host CPU (-march=native)" ON)
set(GK_ARCH_FLAGS "" CACHE STRING "gk: explicit architecture flags, used when GK_NATIVE is off")

# MSVC has no equivalent of -march=native, so GK_NATIVE means nothing there and
# the baseline is a choice instead. AVX2 is the default; this raises it. Both
# are overridden by GK_ARCH_FLAGS.
option(GK_AVX512 "gk: target AVX512 instead of AVX2 (MSVC only)" OFF)

# The block formats come from the quantizer, which is the one place in this
# tree where the GGUF on-disk layouts are implemented. Compiling those sources
# in rather than duplicating them is what keeps the engine and the quantizer
# from ever disagreeing about a block.
set(QZ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../quantizer/src/kernels)

set(GK_QZ_SOURCES
    ${QZ_DIR}/qz_codebook.c
    ${QZ_DIR}/qz_decode.c
    ${QZ_DIR}/qz_init.cpp
    ${QZ_DIR}/qz_lattice.c
    ${QZ_DIR}/qz_pack.c
    ${QZ_DIR}/qz_super.c
    ${QZ_DIR}/qz_traits.c
)

set(GK_SOURCES
    src/gk_alloc.c
    src/gk_backend.c
    src/gk_compute.c
    src/gk_ctx.c
    src/gk_device.c
    src/gk_graph.c
    src/gk_names.c
    src/gk_pool.c
    src/gk_sched.c
    src/gk_ops.c
    src/gk_traits.c
    src/gk_vecdot.c
)

add_library(gk STATIC ${GK_SOURCES} ${GK_QZ_SOURCES})

target_include_directories(gk
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${QZ_DIR}
)

if (NOT MSVC)
    target_compile_options(gk PRIVATE -Wall -Wextra -Wno-unused-function)

    if (GK_NATIVE)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-march=native" GK_HAS_MARCH_NATIVE)
        if (GK_HAS_MARCH_NATIVE)
            target_compile_options(gk PUBLIC -march=native)
        endif()
    elseif (GK_ARCH_FLAGS)
        separate_arguments(GK_ARCH_FLAGS_LIST NATIVE_COMMAND "${GK_ARCH_FLAGS}")
        target_compile_options(gk PUBLIC ${GK_ARCH_FLAGS_LIST})
    endif()
    target_link_libraries(gk PUBLIC m)
    find_package(Threads REQUIRED)
    target_link_libraries(gk PUBLIC Threads::Threads)
else()
    # MSVC has no -march=native, and left alone it targets plain SSE2: none of
    # the SIMD paths in gk_simd.h compile in, and every f16 weight is converted
    # in software. That is worth a factor of ten or more on a quantized model,
    # so a default build has to ask for something newer.
    #
    # It also never defines __FMA__ or __F16C__, even with /arch:AVX2 enabling
    # exactly those instructions - they are GCC/Clang spellings. gk_simd.h
    # gates on them, so they have to be defined here or the AVX2 paths stay
    # dark with the flag on. AVX2 hardware always has FMA and F16C; they
    # shipped together in Haswell, so this is the ISA rather than a guess.
    # The flag is scoped to C/C++: nvcc does not understand /arch and would
    # read it as an input file. The device code has no use for it anyway - it
    # only governs which host SIMD paths compile in.
    if (GK_ARCH_FLAGS)
        separate_arguments(GK_ARCH_FLAGS_LIST NATIVE_COMMAND "${GK_ARCH_FLAGS}")
        target_compile_options(gk PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:${GK_ARCH_FLAGS_LIST}>)
    elseif (GK_AVX512)
        # /arch:AVX512 implies AVX2's instructions as well, so the AVX512 path
        # in gk_simd.h and its f16 helpers both need the same manual defines.
        target_compile_options(gk PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX512>)
        target_compile_definitions(gk PUBLIC __FMA__ __F16C__)
    else()
        target_compile_options(gk PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>)
        target_compile_definitions(gk PUBLIC __FMA__ __F16C__)
    endif()
endif()

set_target_properties(gk PROPERTIES POSITION_INDEPENDENT_CODE ON)

#
# device backends
#
# Each one adds its sources to the same `gk` target rather than producing a
# library of its own. They are not independently useful - a backend is only
# reachable through the registry in gk_device.c - and one target keeps the
# discovery hooks from needing weak symbols or a plugin mechanism.
#

set(GK_CUDA_SOURCES
    src/cuda/gk_cuda.cu
    src/cuda/gk_cuda_ops.cu
    src/cuda/gk_cuda_mmul.cu
)

if (GK_CUDA AND GK_HIP)
    message(FATAL_ERROR "gk: GK_CUDA and GK_HIP compile the same sources for different vendors; enable one")
endif()

if (GK_CUDA)
    enable_language(CUDA)

    target_sources(gk PRIVATE ${GK_CUDA_SOURCES})
    target_compile_definitions(gk PUBLIC GK_USE_CUDA)

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gk_cuda_arch.cmake)

    if (GK_CUDA_ARCHITECTURES)
        set(GK_CUDA_ARCH_VALUE "${GK_CUDA_ARCHITECTURES}")
    elseif (DEFINED CMAKE_CUDA_ARCHITECTURES)
        # An explicit CMAKE_CUDA_ARCHITECTURES is somebody's decision; leave it.
        set(GK_CUDA_ARCH_VALUE "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        gk_choose_cuda_architectures(GK_CUDA_ARCH_VALUE)
    endif()

    if (GK_CUDA_ARCH_VALUE)
        set_property(TARGET gk PROPERTY CUDA_ARCHITECTURES ${GK_CUDA_ARCH_VALUE})

        # Handed to the device code so that a device with no kernel image can
        # be told what the binary *was* built for. Without it the run-time
        # message can say a launch failed but not why, and the answer is
        # always in this list.
        gk_cuda_architectures_to_list("${GK_CUDA_ARCH_VALUE}" GK_CUDA_ARCH_LIST)
        target_compile_definitions(gk PRIVATE GK_CUDA_ARCH_LIST="${GK_CUDA_ARCH_LIST}")
    endif()

    set_property(TARGET gk PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

    find_package(CUDAToolkit REQUIRED)
    # gk is a static library.  Its CUDA objects are linked into the final
    # server/diffusion executables, so the CUDA runtime must propagate to those
    # consumers; PRIVATE would only make it available while building gk itself.
    target_link_libraries(gk PUBLIC CUDA::cudart)

    message(STATUS "gk: CUDA backend enabled")
endif()

if (GK_HIP)
    # The CUDA sources compile unchanged as HIP; src/cuda/gk_cuda_vendor.h is
    # the only file that knows the difference.
    enable_language(HIP)

    set_source_files_properties(${GK_CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
    target_sources(gk PRIVATE ${GK_CUDA_SOURCES})
    target_compile_definitions(gk PUBLIC GK_USE_CUDA GK_USE_HIP)

    if (GK_HIP_ARCHITECTURES)
        set_property(TARGET gk PROPERTY HIP_ARCHITECTURES ${GK_HIP_ARCHITECTURES})
    endif()

    find_package(hip REQUIRED)
    target_link_libraries(gk PRIVATE hip::device)

    message(STATUS "gk: ROCm/HIP backend enabled")
endif()

if (GK_METAL)
    enable_language(OBJC)

    target_sources(gk PRIVATE src/metal/gk_metal.m)
    target_compile_definitions(gk PUBLIC GK_USE_METAL)

    # The Metal objects are held by ARC; the backend's own bookkeeping is
    # plain C, which is why only this one file gets the flag.
    set_source_files_properties(src/metal/gk_metal.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")

    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)

    target_link_libraries(gk PRIVATE
        ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})

    # The shader source is embedded in the binary and compiled at load time.
    # Shipping the .metal file next to the library instead would make the
    # library depend on a file it cannot guarantee is there; a few hundred
    # kilobytes of source in .rodata is the cheaper promise.
    set(GK_METAL_EMBED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/gk_metal_shaders.h)

    add_custom_command(
        OUTPUT  ${GK_METAL_EMBED_HEADER}
        COMMAND ${CMAKE_COMMAND}
                -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/src/metal/gk_metal.metal
                -DOUTPUT=${GK_METAL_EMBED_HEADER}
                -DSYMBOL=gk_metal_shader_source
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/gk_metal.metal
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
        COMMENT "gk: embedding the Metal shader source")

    add_custom_target(gk-metal-shaders DEPENDS ${GK_METAL_EMBED_HEADER})
    add_dependencies(gk gk-metal-shaders)

    target_include_directories(gk PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    message(STATUS "gk: Metal backend enabled")
endif()

if (GK_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)

    target_sources(gk PRIVATE src/vulkan/gk_vulkan.cpp)
    target_compile_definitions(gk PUBLIC GK_USE_VULKAN)
    target_link_libraries(gk PRIVATE Vulkan::Vulkan)

    # Every shader is compiled to SPIR-V at build time and the words are
    # embedded in the library. Compiling GLSL at run time would mean shipping a
    # compiler; loading .spv files from disk would mean the library depending on
    # its own install layout. Neither is worth it for a few tens of kilobytes.
    file(GLOB GK_VULKAN_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/*.comp)

    set(GK_VULKAN_SPV_HEADER ${CMAKE_CURRENT_BINARY_DIR}/gk_vulkan_shaders.h)

    add_custom_command(
        OUTPUT  ${GK_VULKAN_SPV_HEADER}
        COMMAND ${CMAKE_COMMAND}
                -DGLSLC=${Vulkan_GLSLC_EXECUTABLE}
                -DSHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders
                -DOUTPUT=${GK_VULKAN_SPV_HEADER}
                -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/gk-spv
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_shaders.cmake
        DEPENDS ${GK_VULKAN_SHADERS}
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_shaders.cmake
        COMMENT "gk: compiling the Vulkan shaders")

    add_custom_target(gk-vulkan-shaders DEPENDS ${GK_VULKAN_SPV_HEADER})
    add_dependencies(gk gk-vulkan-shaders)

    target_include_directories(gk PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    message(STATUS "gk: Vulkan backend enabled")
endif()

if (GK_BUILD_TESTS)
    enable_testing()
    # The tests need the codec headers as well as gk's own: the accuracy sweep
    # encodes weights through qz_quantize_chunk directly, because the formats
    # that require an importance matrix cannot be produced through the traits'
    # from_float, which has nowhere to pass one.
    set(GK_QZ_INCLUDE_DIR ${QZ_DIR})
    enable_testing()
    add_subdirectory(tests)
endif()
