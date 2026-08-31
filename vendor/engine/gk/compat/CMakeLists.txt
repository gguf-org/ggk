# The ggml compatibility layer: the historical ggml API implemented on gk.
#
# This directory produces the `ggml` target the rest of the engine links.
# Everything below it - libllama, the server, the diffusion runtime, the
# multimodal projectors - compiles against the ggml headers in include/ and
# runs on gk without knowing it.

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/gk)

add_library(ggml STATIC
    src/ggml.c
    src/ggml-backend.c
    src/ggml-cpu.c
    src/ggml-opt.c
    src/gguf.cpp
)

target_include_directories(ggml
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../quantizer/src/kernels
)

target_link_libraries(ggml PUBLIC gk)

set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT MSVC)
    target_compile_options(ggml PRIVATE -Wall -Wextra -Wno-unused-function)
endif()

# The historical build split ggml into base/cpu libraries; anything that still
# names those pieces gets the same target.
add_library(ggml-base ALIAS ggml)
add_library(ggml-cpu  ALIAS ggml)
