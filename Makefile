CUDA_PATH ?= /usr/local/cuda-8.0
HOST_COMPILER ?= g++ 
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) -g -G
LFLAGS := -lcufft -lcublas -lm -lcuda
INCLUDES := -I. -I$(CUDA_PATH)/targets/x86_64-linux/include

all:
	gcc beta.c betafx.c -o beta  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -lblas
	gcc makedata.c gasdev.c ran1.c -lm -o makedata

beta: beta.c betafx.c
	gcc beta.c betafx.c -o $@  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -g -lblas

gamma.o : gamma.cu
	$(NVCC) -c $? -o $@ $(INCLUDES) 

gamma : gamma.o 
	$(NVCC) $? -o $@ $(LFLAGS)


makedata:
	gcc makedata.c gasdev.c ran1.c -lm -o $@ 
clean:
	-rm -f beta makedata gamma.o gamma
