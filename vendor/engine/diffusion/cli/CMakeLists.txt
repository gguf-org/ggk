set(TARGET diffusion-cli)

add_executable(${TARGET}
    common/common.cpp
    common/log.cpp
    common/media_io.cpp
    image_metadata.cpp
    main.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME diffusion)
if(APPLE)
    sd_set_macos_rpaths(${TARGET})
endif()
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
