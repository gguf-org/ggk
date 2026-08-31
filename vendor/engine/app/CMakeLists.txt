# the gguf-server binary: HTTP layer + request handling on top of llama-common

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(TARGET gguf-server)

add_executable(${TARGET}
    main.cpp
    server.cpp
    server-chat.cpp
    server-chat.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-cors-proxy.h
    server-http.cpp
    server-http.h
    server-mcp.cpp
    server-mcp.h
    server-models.cpp
    server-models.h
    server-queue.cpp
    server-queue.h
    server-schema.cpp
    server-schema.h
    server-stream.cpp
    server-stream.h
    server-task.cpp
    server-task.h
    server-tools.cpp
    server-tools.h
)

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../mtmd)
target_link_libraries     (${TARGET} PRIVATE llama-common mtmd gguf-server-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

if (GGUF_SERVER_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
