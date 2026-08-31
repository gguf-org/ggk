find_package(Threads REQUIRED)

llama_add_compile_flags()

#
# llama-common-base: build info, generated from the CMake variables
#

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_BINARY_DIR}/build-info.cpp")

configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})

set(TARGET llama-common-base)
add_library(${TARGET} STATIC ${OUTPUT_FILE})

target_include_directories(${TARGET} PUBLIC .)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#
# llama-common
#

set(TARGET llama-common)

add_library(${TARGET}
    arg.cpp
    arg.h
    base64.hpp
    chat-auto-parser-generator.cpp
    chat-auto-parser-helpers.cpp
    chat-auto-parser.h
    chat-diff-analyzer.cpp
    chat-peg-parser.cpp
    chat-peg-parser.h
    chat.cpp
    chat.h
    common.cpp
    common.h
    console.cpp
    console.h
    debug.cpp
    debug.h
    download.cpp
    download.h
    fit.cpp
    fit.h
    hf-cache.cpp
    hf-cache.h
    http.h
    imatrix-loader.cpp
    imatrix-loader.h
    json-schema-to-grammar.cpp
    json-schema-to-grammar.h
    llguidance.cpp
    log.cpp
    log.h
    ngram-cache.cpp
    ngram-cache.h
    ngram-map.cpp
    ngram-map.h
    ngram-mod.cpp
    ngram-mod.h
    peg-parser.cpp
    peg-parser.h
    preset.cpp
    preset.h
    reasoning-budget.cpp
    reasoning-budget.h
    sampling.cpp
    sampling.h
    speculative.cpp
    speculative.h
    subproc.cpp
    subproc.h
    trie.cpp
    trie.h
    unicode.cpp
    unicode.h
    jinja/lexer.cpp
    jinja/lexer.h
    jinja/parser.cpp
    jinja/parser.h
    jinja/runtime.cpp
    jinja/runtime.h
    jinja/value.cpp
    jinja/value.h
    jinja/string.cpp
    jinja/string.h
    jinja/caps.cpp
    jinja/caps.h
    )

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

# `.` for the common headers, `../thirdparty` for <nlohmann/json.hpp> & friends
target_include_directories(${TARGET} PUBLIC . ../thirdparty)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

if (LLAMA_SUBPROCESS)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_SUBPROCESS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  llama-common-base llama Threads::Threads)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)
