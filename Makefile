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
CXXFLAGS = $(CXX_STANDARD) -fPIC -Wall -O2 -MMD -g

## note: something, probably libtorch will print a stack trace on assert().
## It lacks detail.  Use "where" or "bt full" in gdb to see line numbers, etc.

TOP := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
VENV = $(TOP)/.venv

TORCH = $(VENV)/lib/python3.13/site-packages/torch
TORCH_LDFLAGS = -L$(TORCH)/lib -Wl,-rpath=$(TORCH)/lib
TORCH_LIBS = -ltorch -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda
TORCH_INCFLAGS = -I$(TORCH)/include -I$(TORCH)/include/torch/csrc/api/include

CUDA = $(VENV)/lib/python3.13/site-packages/nvidia/cuda_runtime
CUDA_LDFLAGS = -L$(CUDA)/lib -Wl,-rpath=$(CUDA)/lib
CUDA_LIBS = -lcudart

INC_FLAGS = -I. $(TORCH_INCFLAGS)
LDFLAGS = $(TORCH_LDFLAGS) $(CUDA_LDFLAGS)
LIBS = $(TORCH_LIBS) $(CUDA_LIBS)

# Source files for the shared library
RAYGRID_SRCS = RayGrid.cpp RayTiling.cpp RayTest.cpp
RAYGRID_OBJS = $(RAYGRID_SRCS:.cpp=.o)

# Test source files
TEST_SRCS = $(wildcard test_*.cpp)
# TEST_SRCS = test_raygrid.cpp test_raytiling.cpp test_raytiling_speed.cpp test_raytest.cpp
# TEST_BINS = $(TEST_SRCS:.cpp=)
TEST_BINS = $(patsubst %.cpp,%,$(TEST_SRCS))

DEPS = $(patsubst %.cpp,%.d,$(TEST_SRCS))

# Shared library name
SHARED_LIB = libraygrid.so

.PHONY: all clean run_tests

# Default target: build shared library and all test executables
all: $(SHARED_LIB) $(TEST_BINS)

# Rule to build the shared library
$(SHARED_LIB): $(RAYGRID_OBJS)
	$(CXX) $(CXXFLAGS) $(RAYGRID_OBJS) -shared -o $@ $(LDFLAGS) $(LIBS)

# Rule to compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

test_%: test_%.o $(SHARED_LIB)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS) -L. -lraygrid -Wl,-rpath=.


# Clean up generated files
clean:
	rm -f $(RAYGRID_OBJS) $(TEST_BINS) $(SHARED_LIB) *.o *.d

# Run all tests
run_tests: $(TEST_BINS)
	@echo "Running tests..."
	./test_raygrid
	./test_raytest
	./test_raytiling cpu
	./test_raytiling gpu
	@echo "All tests finished."

-include $(TEST_DEPS)
