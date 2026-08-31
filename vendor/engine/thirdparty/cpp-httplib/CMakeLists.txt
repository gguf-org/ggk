set(TARGET cpp-httplib)

find_package(Threads REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${TARGET} STATIC httplib.cpp httplib.h)

# disable warnings in 3rd party code
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${TARGET} PRIVATE /w)
else()
    target_compile_options(${TARGET} PRIVATE -w)
endif()

target_link_libraries(${TARGET} PRIVATE Threads::Threads)

if (WIN32 AND NOT MSVC)
    target_link_libraries(${TARGET} PUBLIC ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET} PRIVATE
    # increase max payload length to allow use of larger context size
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
    # increase backlog size to avoid connection resets for >> 1 slots
    CPPHTTPLIB_LISTEN_BACKLOG=512
    # increase max URI length to handle longer prompts in query string
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    # disable Nagle's algorithm
    CPPHTTPLIB_TCP_NODELAY=1
)

# HTTPS support - only needed for downloading models over https:// URLs
if (LLAMA_OPENSSL)
    # MSVC must not pick up a MinGW/MSYS2 OpenSSL: those prefixes are on PATH on
    # many Windows dev machines, and dragging C:/msys64/mingw64/include into an
    # MSVC compile makes its GCC-only winnt.h fail with "No supported target
    # architecture" (the import libraries would not link either).
    if (MSVC)
        foreach (root "$ENV{MSYS2_ROOT}" "$ENV{MSYS_ROOT}" "C:/msys64" "C:/msys32" "C:/cygwin64")
            if (root)
                # the prefix a find_* actually lands on is the subsystem dir, not the root
                foreach (sub "" "/mingw64" "/mingw32" "/ucrt64" "/clang64" "/clang32" "/usr")
                    list(APPEND CMAKE_IGNORE_PREFIX_PATH "${root}${sub}")
                endforeach()
            endif()
        endforeach()
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "C:/MinGW" "C:/mingw64" "C:/mingw32")
    endif()

    find_package(OpenSSL)

    # catch any other toolchain-incompatible hit (a prefix not listed above, or
    # one handed to us via OPENSSL_ROOT_DIR)
    if (MSVC AND OpenSSL_FOUND)
        set(OPENSSL_TOOLCHAIN_MISMATCH FALSE)
        foreach (path ${OPENSSL_INCLUDE_DIR} ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
            if (path MATCHES "(msys|mingw|MSYS|MinGW|Mingw|cygwin|Cygwin)" OR path MATCHES "\\.a$")
                set(OPENSSL_TOOLCHAIN_MISMATCH TRUE)
            endif()
        endforeach()
        if (OPENSSL_TOOLCHAIN_MISMATCH)
            message(WARNING
                "Ignoring OpenSSL at ${OPENSSL_INCLUDE_DIR}: it is a MinGW/MSYS/Cygwin build and "
                "cannot be used from MSVC. HTTPS model downloads are disabled. Install an MSVC "
                "OpenSSL (e.g. `vcpkg install openssl:x64-windows`) and point CMake at it with "
                "-DOPENSSL_ROOT_DIR=..., or configure with -DGGUF_SERVER_OPENSSL=OFF to silence this.")
            set(OpenSSL_FOUND FALSE)
        endif()
    endif()

    if (OpenSSL_FOUND)
        include(CheckCSourceCompiles)
        set(SAVED_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
        set(CMAKE_REQUIRED_INCLUDES ${OPENSSL_INCLUDE_DIR})
        check_c_source_compiles("
        #include <openssl/opensslv.h>
        #if defined(OPENSSL_IS_BORINGSSL) || defined(LIBRESSL_VERSION_NUMBER)
        #    if OPENSSL_VERSION_NUMBER < 0x1010107f
        #        error bad version
        #    endif
        #else
        #    if OPENSSL_VERSION_NUMBER < 0x30000000L
        #        error bad version
        #    endif
        #endif
        int main() { return 0; }
        " OPENSSL_VERSION_SUPPORTED)
        set(CMAKE_REQUIRED_INCLUDES ${SAVED_CMAKE_REQUIRED_INCLUDES})
        if (OPENSSL_VERSION_SUPPORTED)
            message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
            set(CPPHTTPLIB_OPENSSL_SUPPORT TRUE)
            target_link_libraries(${TARGET} PUBLIC OpenSSL::SSL OpenSSL::Crypto)
        endif()
    else()
        message(WARNING "OpenSSL not found, HTTPS support disabled")
    endif()
endif()

if (CPPHTTPLIB_OPENSSL_SUPPORT)
    target_compile_definitions(${TARGET} PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
        find_library(SECURITY_FRAMEWORK Security REQUIRED)
        target_link_libraries(${TARGET} PUBLIC ${CORE_FOUNDATION_FRAMEWORK} ${SECURITY_FRAMEWORK})
    endif()
    if (WIN32 AND NOT MSVC)
        target_link_libraries(${TARGET} PUBLIC crypt32)
    endif()
endif()

target_include_directories(${TARGET} PUBLIC ..)
