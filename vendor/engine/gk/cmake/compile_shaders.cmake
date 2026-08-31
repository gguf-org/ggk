# Compiles every GLSL compute shader in a directory to SPIR-V and writes them
# into one header as arrays of 32-bit words, with a table mapping names to
# spans.
#
# The Vulkan backend looks a shader up by name at pipeline-creation time, so
# the table is what it reads; the arrays are what the driver gets.
#
#   cmake -DGLSLC=... -DSHADER_DIR=... -DOUTPUT=... -DWORK_DIR=... -P compile_shaders.cmake

file(MAKE_DIRECTORY ${WORK_DIR})
# Only the .comp files are entry points; gk_common.glsl is included by them.
file(GLOB shaders ${SHADER_DIR}/*.comp)
list(SORT shaders)

set(decls "")
set(table "")

foreach(shader ${shaders})
    get_filename_component(name ${shader} NAME_WE)
    set(spv ${WORK_DIR}/${name}.spv)

    execute_process(
        COMMAND ${GLSLC} -O --target-env=vulkan1.1 -I ${SHADER_DIR} -o ${spv} ${shader}
        RESULT_VARIABLE rc
        ERROR_VARIABLE  err)

    if (NOT rc EQUAL 0)
        message(FATAL_ERROR "gk: compiling ${shader} failed:\n${err}")
    endif()

    file(READ ${spv} words HEX)

    # SPIR-V is a stream of little-endian 32-bit words; the hex dump is
    # byte-order as stored, so each group of four bytes is reversed back.
    string(REGEX MATCHALL "([0-9a-f][0-9a-f])" bytes "${words}")
    list(LENGTH bytes n_bytes)

    set(literals "")
    math(EXPR last "${n_bytes} / 4 - 1")
    foreach(i RANGE ${last})
        math(EXPR b0 "${i} * 4")
        math(EXPR b1 "${b0} + 1")
        math(EXPR b2 "${b0} + 2")
        math(EXPR b3 "${b0} + 3")
        list(GET bytes ${b0} v0)
        list(GET bytes ${b1} v1)
        list(GET bytes ${b2} v2)
        list(GET bytes ${b3} v3)
        string(APPEND literals "0x${v3}${v2}${v1}${v0},")
        math(EXPR nl "${i} % 8")
        if (nl EQUAL 7)
            string(APPEND literals "\n")
        endif()
    endforeach()

    math(EXPR n_words "${n_bytes} / 4")

    string(APPEND decls "static const uint32_t gk_spv_${name}[] = {\n${literals}\n};\n\n")
    string(APPEND table  "    { \"${name}\", gk_spv_${name}, ${n_words} },\n")
endforeach()

file(WRITE ${OUTPUT}
"// Generated from ${SHADER_DIR}. Do not edit.\n"
"#pragma once\n"
"\n"
"#include <stdint.h>\n"
"#include <stddef.h>\n"
"\n"
"${decls}"
"struct gk_spv_entry {\n"
"    const char *     name;\n"
"    const uint32_t * words;\n"
"    size_t           n_words;\n"
"};\n"
"\n"
"static const struct gk_spv_entry gk_spv_shaders[] = {\n"
"${table}"
"};\n")
