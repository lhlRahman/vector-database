# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(VectorDatabase)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# An unset CMAKE_BUILD_TYPE silently compiles at -O0 (unoptimized) — which, with
# the SIMD paths, produced meaningless performance. Default to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# No per-ISA flags needed: SIMD paths are feature-guarded and runtime-dispatched.
# On Apple Silicon, tune for the host core (M1..M4) when supported.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm" AND APPLE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mcpu=native" HAVE_MCPU_NATIVE)
    if(HAVE_MCPU_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    endif()
endif()

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
)

# Create library
add_library(vector_database STATIC ${SOURCES})

# Add examples (if present)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_usage.cpp")
    add_executable(basic_usage examples/basic_usage.cpp)
    target_link_libraries(basic_usage vector_database)
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/advanced_features.cpp")
    add_executable(advanced_features examples/advanced_features.cpp)
    target_link_libraries(advanced_features vector_database)
endif()

# Add benchmarks (if present)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/insertion_benchmark.cpp")
    add_executable(insertion_benchmark benchmarks/insertion_benchmark.cpp)
    target_link_libraries(insertion_benchmark vector_database)
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/search_benchmark.cpp")
    add_executable(search_benchmark benchmarks/search_benchmark.cpp)
    target_link_libraries(search_benchmark vector_database)
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/simd_benchmark.cpp")
    add_executable(simd_benchmark benchmarks/simd_benchmark.cpp)
    target_link_libraries(simd_benchmark vector_database)
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/quick_simd_test.cpp")
    add_executable(quick_simd_test benchmarks/quick_simd_test.cpp)
    target_link_libraries(quick_simd_test vector_database)
endif()

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vector_database OpenMP::OpenMP_CXX)
endif()
