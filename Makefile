# makefile
# Compiler
CXX = g++
CXXFLAGS = -std=c++20 -Iinclude -Isrc -O2

# Directories
SRC_DIR = src
INC_DIR = include
EXAMPLES_DIR = examples
BENCHMARKS_DIR = benchmarks
BUILD_DIR = build

# Source files
SRCS = $(shell find $(SRC_DIR) -name '*.cpp')
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Example and benchmark files
EXAMPLES = $(shell find $(EXAMPLES_DIR) -name '*.cpp')
BENCHMARKS = $(shell find $(BENCHMARKS_DIR) -name '*.cpp')

# Executables
EXAMPLES_EXE = $(EXAMPLES:$(EXAMPLES_DIR)/%.cpp=$(BUILD_DIR)/%)
BENCHMARKS_EXE = $(BENCHMARKS:$(BENCHMARKS_DIR)/%.cpp=$(BUILD_DIR)/%)

# Default target
all: $(EXAMPLES_EXE) $(BENCHMARKS_EXE)

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

# Clean up
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
