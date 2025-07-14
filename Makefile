# makefile
# Compiler
CXX = g++

# Detect architecture and set appropriate flags
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    # ARM64 (Apple Silicon) - use NEON
    CXXFLAGS = -std=c++20 -Iinclude -Isrc -O2 -mcpu=apple-m1 -Wno-psabi
else ifeq ($(UNAME_M),x86_64)
    # x86_64 - use AVX
    CXXFLAGS = -std=c++20 -Iinclude -Isrc -O2 -mavx -mavx2 -Wno-psabi
else
    # Default fallback
    CXXFLAGS = -std=c++20 -Iinclude -Isrc -O2 -Wno-psabi
endif

# Directories
SRC_DIR = src
INC_DIR = include
EXAMPLES_DIR = examples
BENCHMARKS_DIR = benchmarks
BUILD_DIR = build

# Source files
SRCS = $(shell find $(SRC_DIR) -name '*.cpp' ! -path '$(SRC_DIR)/api/*')
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Example and benchmark files
EXAMPLES = $(shell find $(EXAMPLES_DIR) -name '*.cpp')
BENCHMARKS = $(shell find $(BENCHMARKS_DIR) -name '*.cpp')

# Executables
EXAMPLES_EXE = $(EXAMPLES:$(EXAMPLES_DIR)/%.cpp=$(BUILD_DIR)/%)
BENCHMARKS_EXE = $(BENCHMARKS:$(BENCHMARKS_DIR)/%.cpp=$(BUILD_DIR)/%)
API_SERVER = $(BUILD_DIR)/vector_db_server
API_CLIENT = $(BUILD_DIR)/api_client_demo

# Default target
all: $(EXAMPLES_EXE) $(BENCHMARKS_EXE) $(API_SERVER) $(API_CLIENT)

# Build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build example executables
$(BUILD_DIR)/%: $(EXAMPLES_DIR)/%.cpp $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@

# Build benchmark executables
$(BUILD_DIR)/%: $(BENCHMARKS_DIR)/%.cpp $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@

# Build API server
$(API_SERVER): src/api/vector_db_server.cpp $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -pthread $< $(OBJS) -o $@

# Build API client demo
$(API_CLIENT): examples/api_client_demo.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

# Run API server
run-server: $(API_SERVER)
	./$(API_SERVER)

# Run API client demo
run-client: $(API_CLIENT)
	./$(API_CLIENT)

# Clean up
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
