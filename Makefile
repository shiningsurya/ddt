CUDA_PATH ?= /usr/local/cuda-8.0
HOST_COMPILER ?= g++ 
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
LFLAGS := -lcufft -lcublas -lm -lcuda 
INCLUDES := -I.

all:
	gcc beta.c betafx.c -o beta  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -lblas
	gcc makedata.c gasdev.c ran1.c -lm -o makedata

beta: beta.c betafx.c
	gcc beta.c betafx.c -o $@  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -g -lblas

gamma_kernel.o : gamma_kernel.cu
	$(NVCC) -c $? -o $@ $(INCLUDES) 
gamma.o : gamma.c
	$(NVCC) -c $? -o $@ $(INCLUDES) 

gamma : gamma.o gamma_kernel.o 
	$(NVCC) $? -o $@ $(LFLAGS)


makedata:
	gcc makedata.c gasdev.c ran1.c -lm -o $@ 
clean:
	-rm -f beta makedata
