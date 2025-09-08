# Compiler
CXX = g++

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

# --- Platform-Specific Tools ---
# Detect OS and set the appropriate debugger
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	DEBUGGER = lldb
else
	DEBUGGER = gdb
endif

# Set default flags to release mode
CXXFLAGS = $(RELEASE_CXXFLAGS)

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source files
SRCS = $(shell find $(SRC_DIR) -name '*.cpp')
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Executable
TARGET = $(BUILD_DIR)/vector_db_server

# Default target when you just run "make"
all: release

# --- Build Targets ---

# Release target
release: CXXFLAGS = $(RELEASE_CXXFLAGS)
release: $(TARGET)

# Debug target
debug: CXXFLAGS = $(DEBUG_CXXFLAGS)
debug: $(TARGET)

# --- Build Rules ---

# Rule to link the final executable
$(TARGET): $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -pthread $(OBJS) -o $@

# Rule to compile a .cpp source file into a .o object file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@


# --- Convenience Targets ---

# Run API server (release mode)
run-server: release
	./$(TARGET)

# Run API server with the appropriate debugger (gdb or lldb)
run-debug: debug
	$(DEBUGGER) ./$(TARGET)

# Clean up
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all release debug clean run-server run-debug