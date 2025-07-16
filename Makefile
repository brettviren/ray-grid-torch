# Makefile for WireCell::Spng::RayGrid C++/LibTorch modules

# C++ Compiler
CXX = g++

# C++ Standard
CXX_STANDARD = -std=c++17

# Compiler Flags
# -fPIC: Generate position-independent code for shared libraries
# -Wall: Enable all standard warnings
# -O2: Optimization level 2
# -D_GLIBCXX_USE_CXX11_ABI=0: Optional, uncomment if you face ABI compatibility issues with LibTorch
CXXFLAGS = $(CXX_STANDARD) -fPIC -Wall -O2 -ggdb3

## note: something, probably libtorch will print a stack trace on assert().
## It lacks detail.  Use "where" or "bt full" in gdb to see line numbers, etc.



# LibTorch Paths (Dynamically found using Python)
# Ensure you have a Python environment with PyTorch installed
# LIBTORCH_INCLUDE = $(shell python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")
# LIBTORCH_LIB = $(shell python -c "import torch; print(torch.utils.cpp_extension.library_paths()[0])")
TORCH = /home/bv/dev/wire-cell-python/.venv/lib/python3.12/site-packages/torch
#LIBTORCH_INCLUDE 
#LIBTORCH_LIB

# Include Paths
INC_FLAGS = -I. -I$(TORCH)/include -I$(TORCH)/include/torch/csrc/api/include

# Library Paths and Linker Flags
# -L: Add directory to library search path
# -l: Link with specified library
# -Wl,-rpath: Add a runtime search path for shared libraries (important for running tests)
LDFLAGS = -L$(TORCH)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath=$(TORCH)/lib

# Source files for the shared library
RAYGRID_SRCS = RayGrid.cpp RayTiling.cpp RayTest.cpp
RAYGRID_OBJS = $(RAYGRID_SRCS:.cpp=.o)

# Test source files
TEST_SRCS = test_raygrid.cpp test_raytiling.cpp
TEST_BINS = $(TEST_SRCS:.cpp=)

# Shared library name
SHARED_LIB = libraygrid.so

.PHONY: all clean run_tests

# Default target: build shared library and all test executables
all: $(SHARED_LIB) $(TEST_BINS)

# Rule to build the shared library
$(SHARED_LIB): $(RAYGRID_OBJS)
	$(CXX) $(CXXFLAGS) $(RAYGRID_OBJS) -shared -o $@ $(LDFLAGS)

# Rule to compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

# Rule to build test_raygrid executable
test_raygrid: test_raygrid.o $(SHARED_LIB)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -L. -lraygrid -Wl,-rpath=.

# Rule to build test_raytiling executable
test_raytiling: test_raytiling.o $(SHARED_LIB)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -L. -lraygrid -Wl,-rpath=.

# Clean up generated files
clean:
	rm -f $(RAYGRID_OBJS) $(TEST_BINS) $(SHARED_LIB) *.o

# Run all tests
run_tests: $(TEST_BINS)
	@echo "Running tests..."
	./test_raygrid
	./test_raytiling
	@echo "All tests finished."

