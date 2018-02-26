all:
	gcc beta.c betafx.c -o beta  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -lblas
	gcc makedata.c gasdev.c ran1.c -lm -o makedata

beta: beta.c betafx.c
	gcc beta.c betafx.c -o $@  -lm -I/home/shining/pulsoft/include -L/home/shining/pulsoft/lib -lfftw3 -g -lblas

gamma: gamma.cu gammafx.c gamma.h
	nvcc gammafx.c gamma.cu -o $@ -lcublas -g 

makedata:
	gcc makedata.c gasdev.c ran1.c -lm -o $@ 
clean:
	-rm -f beta makedata
