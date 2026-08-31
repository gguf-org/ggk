# Turns a text file into a C string literal in a header.
#
# Used for the Metal shader source, which the backend hands to the driver at
# load time. Written as a script rather than as inline CMake so it can be a
# proper build step with a dependency on the input.
#
#   cmake -DINPUT=... -DOUTPUT=... -DSYMBOL=... -P embed_text.cmake

file(READ ${INPUT} contents)

# Escape what a C string cannot hold literally. Order matters: backslashes
# first, or the escapes introduced below would themselves be escaped.
string(REPLACE "\\" "\\\\" contents "${contents}")
string(REPLACE "\"" "\\\"" contents "${contents}")
string(REPLACE "\n" "\\n\"\n\"" contents "${contents}")

file(WRITE ${OUTPUT}
"// Generated from ${INPUT}. Do not edit.\n"
"#pragma once\n"
"\n"
"static const char * ${SYMBOL} =\n"
"\"${contents}\";\n")
