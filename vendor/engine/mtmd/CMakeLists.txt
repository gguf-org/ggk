# mtmd: multimodal (vision / audio) projector support for the server

set(MTMD_VIDEO_HELP "enable video support in mtmd (requires ffmpeg binary in PATH)")

set(MTMD_VIDEO ON CACHE BOOL "${MTMD_VIDEO_HELP}")

if (MTMD_VIDEO AND NOT LLAMA_SUBPROCESS)
    message(STATUS "Disabling MTMD_VIDEO because subprocess support is OFF")
    set(MTMD_VIDEO OFF CACHE BOOL "${MTMD_VIDEO_HELP}" FORCE)
endif()

find_package(Threads REQUIRED)

file(GLOB MTMD_MODEL_SOURCES CONFIGURE_DEPENDS "models/*.cpp")

add_library(mtmd
            mtmd.cpp
            mtmd.h
            mtmd-audio.cpp
            mtmd-audio.h
            mtmd-image.cpp
            mtmd-image.h
            mtmd-helper.cpp
            mtmd-helper.h
            clip.cpp
            clip.h
            clip-impl.h
            clip-model.h
            clip-graph.h
            models/models.h
            ${MTMD_MODEL_SOURCES}
            )

set_target_properties(mtmd PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC  ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
# for <stb/stb_image.h> and <miniaudio/miniaudio.h>
target_include_directories(mtmd PRIVATE ../thirdparty)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (MTMD_VIDEO)
    target_compile_definitions(mtmd PRIVATE MTMD_VIDEO)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)
endif()

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (ANDROID)
    # miniaudio.h defines ma_android_sdk_version() without a prior prototype
    target_compile_options(mtmd PRIVATE -Wno-missing-prototypes)
endif()
