CUDA_PATH ?= /usr/local/cuda-8.0
HOST_COMPILER ?= gcc 
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
FLAGS := -lcufft -lcublas -lm -lcuda 


all:
	gcc beta.c betafx.c -o beta  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -lblas
	gcc makedata.c gasdev.c ran1.c -lm -o makedata

beta: beta.c betafx.c
	gcc beta.c betafx.c -o $@  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -g -lblas

gamma: gamma.cu gammafx.c
	$(NVCC) gamma.cu gammafx.c -o $@ -lcublas -lcufft -g -L/usr/lib/"nvidia-367"

gamma_kernel.o : gamma_kernel.cu
	$(NVCC) $? -o $@ $(FLAGS)
gamma.o : gamma.c
	$(NVCC) $? -o $@ $(FLAGS)

gamma : gamma.o gamma_kernel.o 
	$(NVCC) $? -o $@


makedata:
	gcc makedata.c gasdev.c ran1.c -lm -o $@ 
clean:
	-rm -f beta makedata
