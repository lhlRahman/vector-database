# Compiler
CXX = clang++

# Base flags, common to both release and debug
BASE_CXXFLAGS = -std=c++20 -Iinclude -Isrc -Wno-psabi -I$(SRC_DIR)

# Detect architecture and set appropriate flags
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    # ARM64 (Apple Silicon) - use NEON
    ARCH_FLAGS = -mcpu=apple-m1
else ifeq ($(UNAME_M),x86_64)
    # x86_64 - use AVX
    ARCH_FLAGS = -mavx -mavx2
else
    # Default fallback
    ARCH_FLAGS =
endif

# --- Build Mode Flags ---
# Release flags: -O2 for optimization
RELEASE_CXXFLAGS = $(BASE_CXXFLAGS) $(ARCH_FLAGS) -O2
# Debug flags: -g for debug symbols, -O0 to disable optimization, -Wall for all warnings
DEBUG_CXXFLAGS = $(BASE_CXXFLAGS) $(ARCH_FLAGS) -g -O0 -Wall

# Objective-C++ flags (for .mm files)
OBJCXXFLAGS = -fobjc-arc

# --- Platform-Specific Tools & Frameworks ---
# Detect OS and set the appropriate debugger and frameworks
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    DEBUGGER = lldb
    # Metal frameworks for GPU acceleration (macOS only)
    METAL_FRAMEWORKS = -framework Metal -framework MetalPerformanceShaders -framework Foundation
else
    DEBUGGER = gdb
    METAL_FRAMEWORKS =
endif

# Set default flags to release mode
CXXFLAGS = $(RELEASE_CXXFLAGS)

# Directories
SRC_DIR = src
BUILD_DIR = build
SHADER_DIR = $(SRC_DIR)/optimizations/shaders

# Source files (C++ and Objective-C++)
CPP_SRCS = $(shell find $(SRC_DIR) -name '*.cpp')
MM_SRCS = $(shell find $(SRC_DIR) -name '*.mm')
CPP_OBJS = $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
MM_OBJS = $(MM_SRCS:$(SRC_DIR)/%.mm=$(BUILD_DIR)/%.o)
OBJS = $(CPP_OBJS) $(MM_OBJS)

# Metal shader files
METAL_SRCS = $(shell find $(SHADER_DIR) -name '*.metal' 2>/dev/null)
METALLIB = $(BUILD_DIR)/vector_ops.metallib

# Executable
TARGET = $(BUILD_DIR)/tcp_server

# Default target when you just run "make"
all: tcp-server

# --- Build Rules ---

# Rule to compile a .cpp source file into a .o object file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile a .mm (Objective-C++) source file into a .o object file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.mm
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OBJCXXFLAGS) -c $< -o $@

# Rule to compile Metal shaders into a .metallib
$(METALLIB): $(METAL_SRCS)
ifneq ($(METAL_SRCS),)
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c $(METAL_SRCS) -o $(BUILD_DIR)/vector_ops.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/vector_ops.air -o $@
	@echo "Metal shaders compiled to $@"
else
	@echo "No Metal shaders found, skipping metallib compilation"
endif

# --- Convenience Targets ---

# Run TCP server
run-server: tcp-server
	./$(TARGET)

# Compile only Metal shaders
metal: $(METALLIB)

# Build GPU benchmark
BENCHMARK_GPU = $(BUILD_DIR)/benchmark_gpu
benchmark-gpu: $(BUILD_DIR)/core/vector_database.o $(BUILD_DIR)/core/vector.o $(BUILD_DIR)/core/kd_tree.o \
               $(BUILD_DIR)/features/query_cache.o $(BUILD_DIR)/features/atomic_batch_insert.o \
               $(BUILD_DIR)/features/atomic_file_writer.o $(BUILD_DIR)/features/atomic_persistence.o \
               $(BUILD_DIR)/features/commit_log.o $(BUILD_DIR)/algorithms/approximate_nn.o \
               $(BUILD_DIR)/algorithms/lsh_index.o $(BUILD_DIR)/algorithms/hnsw_index.o \
               $(BUILD_DIR)/utils/distance_metrics.o $(BUILD_DIR)/utils/random_generator.o \
               $(BUILD_DIR)/optimizations/simd_operations.o $(BUILD_DIR)/optimizations/parallel_processing.o \
               $(BUILD_DIR)/optimizations/gpu_operations.o
	$(CXX) $(CXXFLAGS) -c test/benchmark_gpu.cpp -o $(BUILD_DIR)/benchmark_gpu.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(BUILD_DIR)/benchmark_gpu.o \
		$(BUILD_DIR)/core/vector_database.o $(BUILD_DIR)/core/vector.o $(BUILD_DIR)/core/kd_tree.o \
		$(BUILD_DIR)/features/query_cache.o $(BUILD_DIR)/features/atomic_batch_insert.o \
		$(BUILD_DIR)/features/atomic_file_writer.o $(BUILD_DIR)/features/atomic_persistence.o \
		$(BUILD_DIR)/features/commit_log.o $(BUILD_DIR)/algorithms/approximate_nn.o \
		$(BUILD_DIR)/algorithms/lsh_index.o $(BUILD_DIR)/algorithms/hnsw_index.o \
		$(BUILD_DIR)/utils/distance_metrics.o $(BUILD_DIR)/utils/random_generator.o \
		$(BUILD_DIR)/optimizations/simd_operations.o $(BUILD_DIR)/optimizations/parallel_processing.o \
		$(BUILD_DIR)/optimizations/gpu_operations.o \
		$(METAL_FRAMEWORKS) -o $(BENCHMARK_GPU)
	@echo "Benchmark built: $(BENCHMARK_GPU)"

# Build SIMD tail test
SIMD_TAIL_TEST = $(BUILD_DIR)/simd_tail_test
simd-tail-test: $(BUILD_DIR)/core/vector.o $(BUILD_DIR)/optimizations/simd_operations.o
	$(CXX) $(CXXFLAGS) -c test/simd_tail_test.cpp -o $(BUILD_DIR)/simd_tail_test.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(BUILD_DIR)/simd_tail_test.o \
		$(BUILD_DIR)/core/vector.o $(BUILD_DIR)/optimizations/simd_operations.o \
		-o $(SIMD_TAIL_TEST)
	@echo "SIMD tail test built: $(SIMD_TAIL_TEST)"

# Common library objects (everything except main.o)
LIB_OBJS = $(BUILD_DIR)/core/vector_database.o $(BUILD_DIR)/core/vector.o $(BUILD_DIR)/core/kd_tree.o \
           $(BUILD_DIR)/features/query_cache.o $(BUILD_DIR)/features/atomic_batch_insert.o \
           $(BUILD_DIR)/features/atomic_file_writer.o $(BUILD_DIR)/features/atomic_persistence.o \
           $(BUILD_DIR)/features/commit_log.o $(BUILD_DIR)/algorithms/approximate_nn.o \
           $(BUILD_DIR)/algorithms/lsh_index.o $(BUILD_DIR)/algorithms/hnsw_index.o \
           $(BUILD_DIR)/utils/distance_metrics.o $(BUILD_DIR)/utils/random_generator.o \
           $(BUILD_DIR)/optimizations/simd_operations.o $(BUILD_DIR)/optimizations/parallel_processing.o \
           $(BUILD_DIR)/optimizations/gpu_operations.o

# Unit tests
UNIT_TEST = $(BUILD_DIR)/unit_tests
unit-test: $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -c test/unit_tests.cpp -o $(BUILD_DIR)/unit_tests.o
	$(CXX) $(CXXFLAGS) -pthread $(BUILD_DIR)/unit_tests.o $(LIB_OBJS) $(METAL_FRAMEWORKS) -o $(UNIT_TEST)
	@echo "Running unit tests..."
	@./$(UNIT_TEST)

# End-to-end tests
E2E_TEST = $(BUILD_DIR)/e2e_tests
e2e-test: $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -c test/e2e_tests.cpp -o $(BUILD_DIR)/e2e_tests.o
	$(CXX) $(CXXFLAGS) -pthread $(BUILD_DIR)/e2e_tests.o $(LIB_OBJS) $(METAL_FRAMEWORKS) -o $(E2E_TEST)
	@echo "Running end-to-end tests..."
	@./$(E2E_TEST)

# Performance tests
PERF_TEST = $(BUILD_DIR)/perf_tests
perf-test: $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -c test/perf_tests.cpp -o $(BUILD_DIR)/perf_tests.o
	$(CXX) $(CXXFLAGS) -pthread $(BUILD_DIR)/perf_tests.o $(LIB_OBJS) $(METAL_FRAMEWORKS) -o $(PERF_TEST)
	@echo "Running performance tests..."
	@./$(PERF_TEST)

# TCP server
TCP_SERVER = $(BUILD_DIR)/tcp_server
tcp-server: $(LIB_OBJS)
	@mkdir -p $(BUILD_DIR)/api
	$(CXX) $(CXXFLAGS) -c src/api/tcp_server.cpp -o $(BUILD_DIR)/api/tcp_server.o
	$(CXX) $(CXXFLAGS) -c src/api/tcp_main.cpp -o $(BUILD_DIR)/api/tcp_main.o
	$(CXX) $(CXXFLAGS) -pthread $(BUILD_DIR)/api/tcp_main.o $(BUILD_DIR)/api/tcp_server.o $(LIB_OBJS) $(METAL_FRAMEWORKS) -o $(TCP_SERVER)
	@echo "TCP server built: $(TCP_SERVER)"

# TCP transport tests
TCP_TEST = $(BUILD_DIR)/tcp_tests
tcp-test: $(LIB_OBJS)
	@mkdir -p $(BUILD_DIR)/api
	$(CXX) $(CXXFLAGS) -c src/api/tcp_server.cpp -o $(BUILD_DIR)/api/tcp_server.o
	$(CXX) $(CXXFLAGS) -c src/api/tcp_client.cpp -o $(BUILD_DIR)/api/tcp_client.o
	$(CXX) $(CXXFLAGS) -c test/test_tcp.cpp -o $(BUILD_DIR)/test_tcp.o
	$(CXX) $(CXXFLAGS) -pthread $(BUILD_DIR)/test_tcp.o $(BUILD_DIR)/api/tcp_server.o $(BUILD_DIR)/api/tcp_client.o $(LIB_OBJS) $(METAL_FRAMEWORKS) -o $(TCP_TEST)
	@echo "Running TCP transport tests..."
	@./$(TCP_TEST)

# TCP network benchmark (direct API vs TCP transport)
TCP_BENCH = $(BUILD_DIR)/bench_tcp
bench-tcp: $(LIB_OBJS)
	@mkdir -p $(BUILD_DIR)/api
	$(CXX) $(CXXFLAGS) -c src/api/tcp_server.cpp -o $(BUILD_DIR)/api/tcp_server.o
	$(CXX) $(CXXFLAGS) -c src/api/tcp_client.cpp -o $(BUILD_DIR)/api/tcp_client.o
	$(CXX) $(CXXFLAGS) -c test/bench_tcp.cpp -o $(BUILD_DIR)/bench_tcp.o
	$(CXX) $(CXXFLAGS) -pthread $(BUILD_DIR)/bench_tcp.o $(BUILD_DIR)/api/tcp_server.o $(BUILD_DIR)/api/tcp_client.o $(LIB_OBJS) $(METAL_FRAMEWORKS) -o $(TCP_BENCH)
	@echo "Running TCP benchmark..."
	@./$(TCP_BENCH)

# Run all tests
test: unit-test e2e-test

# Clean up
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean run-server metal benchmark-gpu simd-tail-test unit-test e2e-test perf-test test tcp-server tcp-test bench-tcp