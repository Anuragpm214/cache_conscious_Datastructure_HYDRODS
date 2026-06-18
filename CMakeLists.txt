cmake_minimum_required(VERSION 3.16)
project(HydroDS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Compiler Flags ----
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Test executable ----
add_executable(hydrods_test hydrods_test.cpp)
target_include_directories(hydrods_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(hydrods_test PRIVATE -Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(hydrods_test PRIVATE -fsanitize=address,undefined)
endif()

# ---- Original benchmark (for reference/comparison) ----
add_executable(hydrods_original hydrods.cpp)
target_compile_options(hydrods_original PRIVATE -O3 -march=native -mtune=native)

# ---- Comparison benchmark (HydroDS vs Baselines) ----
add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tlx_baseline
)
target_compile_options(benchmark PRIVATE -Wall -Wextra -O3 -march=native -mtune=native)

# ---- Concurrent Phase 2 Test ----
add_executable(test_concurrent test_concurrent.cpp)
target_include_directories(test_concurrent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_concurrent PRIVATE -Wall -Wextra -O2 -g -fsanitize=thread)
target_link_options(test_concurrent PRIVATE -fsanitize=thread)
target_link_libraries(test_concurrent PRIVATE pthread)

# ---- Phase 3 Scaling Benchmark ----
add_executable(benchmark_scaling benchmark_scaling.cpp)
target_include_directories(benchmark_scaling PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tlx_baseline
)
target_compile_options(benchmark_scaling PRIVATE -Wall -Wextra -O3 -march=native -mtune=native)
target_link_libraries(benchmark_scaling PRIVATE pthread)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags:  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
