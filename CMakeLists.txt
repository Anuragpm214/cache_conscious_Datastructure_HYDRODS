cmake_minimum_required(VERSION 3.10)
project(hydrods)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimize for cache architecture
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -march=native -mtune=native")

# Main unified benchmark executable
add_executable(master_benchmark master_benchmark.cpp)

target_include_directories(master_benchmark PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/tlx_baseline
    ${CMAKE_CURRENT_SOURCE_DIR}/alex_baseline/src/core
)

find_package(OpenMP REQUIRED)

# Enforce hardware optimizations and OpenMP
target_compile_options(master_benchmark PRIVATE -mavx2 -mfma -O3 -pthread -fopenmp)
target_link_libraries(master_benchmark PRIVATE pthread OpenMP::OpenMP_CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags:  ${CMAKE_CXX_FLAGS}")
