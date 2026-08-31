# Test programs.
#
# test-foundation is self-contained: it checks the invariants the rest of the
# library is allowed to assume (block geometry, stride rules, arena and graph
# behaviour) without reference to anything outside gk.
#
# test-vs-ggml is the differential harness. It builds only when a reference
# tree is pointed at with -DGK_REFERENCE_DIR, so an ordinary build does not
# depend on the reference existing.

add_executable(test-foundation test_foundation.c)
target_link_libraries(test-foundation PRIVATE gk)
target_include_directories(test-foundation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${GK_QZ_INCLUDE_DIR}
)
add_test(NAME foundation COMMAND test-foundation)

add_executable(test-threading test_threading.c)
target_link_libraries(test-threading PRIVATE gk)
target_include_directories(test-threading PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME threading COMMAND test-threading)

add_executable(test-backend test_backend.c)
target_link_libraries(test-backend PRIVATE gk)
target_include_directories(test-backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME backend COMMAND test-backend)

add_executable(test-sched test_sched.c)
target_link_libraries(test-sched PRIVATE gk)
target_include_directories(test-sched PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME sched COMMAND test-sched)

if (GK_CUDA OR GK_HIP)
    add_executable(test-cuda test_cuda.c)
    target_link_libraries(test-cuda PRIVATE gk)
    target_include_directories(test-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    add_test(NAME cuda COMMAND test-cuda)

    # The pair, rather than the device: a graph whose weights are spread over
    # every GPU present, and the copies between them. Passes on a machine with
    # one GPU, where it reports that the crossing went unexercised.
    add_executable(test-cuda-multi test_cuda_multi.c)
    target_link_libraries(test-cuda-multi PRIVATE gk)
    target_include_directories(test-cuda-multi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    add_test(NAME cuda-multi COMMAND test-cuda-multi)

    # Not a test either: times each op on the device against the same op on
    # the CPU, and reports which ops have no device kernel at all. Never
    # fails, so it is not registered with ctest.
    add_executable(bench-cuda bench_cuda.c)
    target_link_libraries(bench-cuda PRIVATE gk)
    target_include_directories(bench-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()

# Not a test: reports rates, never fails. Build the library with GK_NATIVE on
# and off and compare to see what the SIMD paths are actually worth.
add_executable(bench-gk bench.c)
target_link_libraries(bench-gk PRIVATE gk)
target_include_directories(bench-gk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${GK_QZ_INCLUDE_DIR}
)

if (GK_REFERENCE_DIR)
    if (NOT EXISTS ${GK_REFERENCE_DIR}/include/ggml.h)
        message(FATAL_ERROR "GK_REFERENCE_DIR=${GK_REFERENCE_DIR} has no include/ggml.h")
    endif()
    message(STATUS "gk: differential tests will run against ${GK_REFERENCE_DIR}")

    add_executable(test-vs-ggml test_vs_ggml.c)
    target_link_libraries(test-vs-ggml PRIVATE gk ${GK_REFERENCE_LIB})
    target_include_directories(test-vs-ggml PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${GK_REFERENCE_DIR}/include
    )
    add_test(NAME vs-ggml COMMAND test-vs-ggml)

    add_executable(test-ops-vs-ggml test_ops_vs_ggml.c)
    target_link_libraries(test-ops-vs-ggml PRIVATE gk ${GK_REFERENCE_LIB})
    target_include_directories(test-ops-vs-ggml PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${GK_REFERENCE_DIR}/include
    )
    add_test(NAME ops-vs-ggml COMMAND test-ops-vs-ggml)
endif()
