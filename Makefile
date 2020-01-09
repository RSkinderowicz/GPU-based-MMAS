GOOD_WARN = -std=c++14
CUDA_DIR = /usr/local/cuda/
NVCC = nvcc
CXX = g++

CC_KEPLER=-gencode arch=compute_30,code=sm_30
CC_MAXWELL=-gencode arch=compute_50,code=sm_50
CC_PASCAL=-gencode arch=compute_60,code=sm_60
CC_VOLTA=-gencode arch=compute_70,code=sm_70

# IMPORTANT!
# The following should be adjusted to the target GPU architecture:
COMPUTE_CAPABILITY=$(CC_MAXWELL)

mode = release

ifeq ($(mode),release)
	CXXFLAGS = -pipe -std=c++14 -Wall -pedantic -O2 -mtune=native -march=native -DNDEBUG  -flto
	NVCCFLAGS = -std=c++14 -Xptxas="-v"  $(COMPUTE_CAPABILITY) -DNDEBUG 
else
	CXXFLAGS = -g -pipe -std=c++14 -Wall -O0
	NVCCFLAGS = -G -std=c++14 -m64  $(COMPUTE_CAPABILITY)
endif

# This should point to a valid CUDA lib64/ location
LDFLAGS  = -L$(CUDA_DIR)/lib64 -lcuda -lcudart

BUILDDIR=obj
TARGET=mmas
SRCDIR=src

SOURCES=main.cc\
	    docopt.cc\
		utils.cc\
		tsp.cc
OBJS = $(SOURCES:.cc=.o)

CUDA_SOURCES=mmas.cu
CUDA_OBJS = $(CUDA_SOURCES:.cu=.o)

OUT_OBJS = $(addprefix $(BUILDDIR)/,$(OBJS))
OUT_CUDA_OBJS = $(addprefix $(BUILDDIR)/,$(CUDA_OBJS))

.PHONY: clean all

all: $(TARGET)

$(TARGET): $(OUT_OBJS) $(OUT_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $(OUT_CUDA_OBJS) -dlink -o $(BUILDDIR)/link.o
	$(CXX) $(CXXFLAGS) $(Ommas -DNDEBUGUT_OBJS) $(OUT_OBJS) $(OUT_CUDA_OBJS) $(BUILDDIR)/link.o $(LDFLAGS) -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc $(SRCDIR)/tsp.h
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

clean:
	rm -f $(OUT_OBJS) $(OUT_CUDA_OBJS) $(TARGET) $(BUILDDIR)/link.o