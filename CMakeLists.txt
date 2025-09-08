# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(VectorDatabase)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable ARM NEON for ARM architectures
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    # For Apple Silicon, NEON is always available and enabled by default
    # For other ARM systems, we might need additional flags
    if(NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()
endif()

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.hpp"
)

# Create library
add_library(vector_database STATIC ${SOURCES})

# Add examples
add_executable(basic_usage examples/basic_usage.cpp)
target_link_libraries(basic_usage vector_database)

add_executable(advanced_features examples/advanced_features.cpp)
target_link_libraries(advanced_features vector_database)

# Add benchmarks
add_executable(insertion_benchmark benchmarks/insertion_benchmark.cpp)
target_link_libraries(insertion_benchmark vector_database)

add_executable(search_benchmark benchmarks/search_benchmark.cpp)
target_link_libraries(search_benchmark vector_database)

add_executable(simd_benchmark benchmarks/simd_benchmark.cpp)
target_link_libraries(simd_benchmark vector_database)

add_executable(quick_simd_test benchmarks/quick_simd_test.cpp)
target_link_libraries(quick_simd_test vector_database)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vector_database OpenMP::OpenMP_CXX)
endif()
