# gguf-server-ui: embeds the web UI assets into the binary.
#
# The assets are plain static files (index.html, js, css, ...) taken from
# GGUF_SERVER_UI_DIR, which defaults to ui/dist in this tree. If that directory
# is missing the target still builds, but with an empty asset table: the server
# then only exposes its HTTP API, and `--path <dir>` can still serve a web UI
# from disk at runtime.

set(TARGET gguf-server-ui)

set(GGUF_SERVER_UI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE PATH
    "directory with the web UI assets to embed (empty table if it does not exist)")

set(UI_CPP "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(UI_H   "${CMAKE_CURRENT_BINARY_DIR}/ui.h")

# the embedder is a build-time tool, so under cross-compilation it has to be
# built with the host compiler
if (CMAKE_CROSSCOMPILING)
    find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    if (NOT HOST_CXX_COMPILER)
        message(FATAL_ERROR "ui: no host C++ compiler (g++/clang++) found to build the asset embedder; set -DHOST_CXX_COMPILER=<path>")
    endif()

    if (CMAKE_HOST_WIN32)
        set(UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/gguf-server-ui-embed-host.exe")
    else()
        set(UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/gguf-server-ui-embed-host")
    endif()

    add_custom_command(
        OUTPUT  "${UI_EMBED_EXE}"
        COMMAND "${HOST_CXX_COMPILER}" -O2 -std=c++17
                -o "${UI_EMBED_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        COMMENT "Building gguf-server-ui-embed (host)"
        VERBATIM
    )
    add_custom_target(gguf-server-ui-embed DEPENDS "${UI_EMBED_EXE}")
else()
    add_executable(gguf-server-ui-embed embed.cpp)
    target_compile_features(gguf-server-ui-embed PRIVATE cxx_std_17)
    set_target_properties(gguf-server-ui-embed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    set(UI_EMBED_EXE "$<TARGET_FILE:gguf-server-ui-embed>")
endif()

if (EXISTS "${GGUF_SERVER_UI_DIR}/index.html")
    message(STATUS "gguf-server: embedding web UI assets from ${GGUF_SERVER_UI_DIR}")
    set(UI_EMBED_ARGS "${UI_CPP}" "${UI_H}" "${GGUF_SERVER_UI_DIR}")
    file(GLOB_RECURSE UI_ASSET_FILES CONFIGURE_DEPENDS "${GGUF_SERVER_UI_DIR}/*")
else()
    message(STATUS "gguf-server: no web UI assets at ${GGUF_SERVER_UI_DIR}, building API-only server")
    set(UI_EMBED_ARGS "${UI_CPP}" "${UI_H}")
    set(UI_ASSET_FILES "")
endif()

add_custom_command(
    OUTPUT  "${UI_CPP}" "${UI_H}"
    COMMAND "${UI_EMBED_EXE}" ${UI_EMBED_ARGS}
    DEPENDS gguf-server-ui-embed ${UI_ASSET_FILES}
    COMMENT "Embedding web UI assets"
    VERBATIM
)

add_library(${TARGET} STATIC "${UI_CPP}" "${UI_H}")
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
