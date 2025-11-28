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
TARGET = $(BUILD_DIR)/vector_db_server

# Default target when you just run "make"
all: release

# --- Build Targets ---

# Release target (skip metallib if metal compiler not available)
release: CXXFLAGS = $(RELEASE_CXXFLAGS)
release: $(TARGET)
	@echo "Note: Metal shader compiler not found. Using runtime shader compilation."

# Debug target (skip metallib if metal compiler not available)
debug: CXXFLAGS = $(DEBUG_CXXFLAGS)
debug: $(TARGET)
	@echo "Note: Metal shader compiler not found. Using runtime shader compilation."

# --- Build Rules ---

# Rule to link the final executable
$(TARGET): $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -pthread $(OBJS) $(METAL_FRAMEWORKS) -o $@

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

# Run API server (release mode)
run-server: release
	./$(TARGET)

# Run API server with the appropriate debugger (gdb or lldb)
run-debug: debug
	$(DEBUGGER) ./$(TARGET)

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

# Clean up
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all release debug clean run-server run-debug metal benchmark-gpu