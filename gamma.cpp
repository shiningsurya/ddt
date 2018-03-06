// This is Gammafx.
// This is the original code

// Standard includes
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"

// CUDA includes
#include "cuda.h"
#include "cufft.h"
#include "cublas_v2.h"
#include <helper_cuda.h>
#include "cuda_runtime.h"

// typedef 
// because I am lazy to write float2 everywhere
typedef float2 Complex; 

// constants 
__global__ void ddtchirp(Complex * chirp, float delta, long N) {
__global__ void fftchirp(Complex * chirp, float delta, long N){
__device__ Complex ComplexMult(Complex one, Complex two) {
__global__ void vecpro(Complex * i1, Complex * i2, Complex * out){
// starting Main
int main(int argc, char * argv[]){
		if(argc < 2) {
				printf("Direct De-dispersion Transform\n");
				printf("Usage : <program> <DM> <exponent of two> <filename>\n");
				printf("Written by Suryarao Bethapudi[ep14btech11008@iith.ac.in]\n");
				return 0;
		}
		// check for CUDA 
		int dev = findCudaDevice(argc, (const char **) argv);
		if(dev == -1) {
				fprintf(stderr,"Couldn't find CUDA device\n");
				return 1;
		}
		// basic part
		//
		long N = (long)strtol(argv[2],NULL,10);
		double dm;
		FILE * fp;
		fp = fopen((const char*)argv[3],"r");
		N = pow(2,N);
		dm = (double)strtod(argv[1],NULL);

		// complex pointers for host and device memory
		Complex * h_in, * d_in;
		Complex * h_out, * ddt_out, * fft_out;
		Complex * ddtchirp;
		long memsize = N * sizeof(Complex);	
		// Allocating in Host
		h_in = (Complex*)malloc(memsize);
		h_out = (Complex*)malloc(memsize);
		// FILE IO 
		long i;
		float dx;
		Complex t;
		for(i = 0; i < N;i++){
				fscanf(fp,"%f\n",&dx);
				t.x = dx;
				t.y = 0.0f;
				h_in[i] = t;
		}
		///////////////////////////////////////////////////////////////////
		// Allocating in Device
		// NOTE: For now, I am creating three N-arrays. 
		checkCudaErrors(cudaMalloc((void**)&d_in,memsize));
		checkCudaErrors(cudaMalloc((void**)&ddt_out,memsize));
		checkCudaErrors(cudaMalloc((void**)&ddtchirp,N*memsize)); 
		// Need N^2 elements
		checkCudaErrors(cudaMalloc((void**)&fft_out,memsize));
		// These are compressed commands
	
		// Copying from Host to Device 
		checkCudaErrors(cudaMemCpy(h_in, d_in, memsize, cudaMemcpyHostToDevice));

	    // cuda variables, types
	    cudaError_t cuerr;
	    // measure time
	    cudaEvent_t estart, estop; 
		checkCudaErrors( cudaEventCreate(&estart) );
		checkCudaErrors( cudaEventCreate(&estop) );
		// elasped time
		float t_fftchirp, t_fft, t_ddtchrip, t_ddt; 
		// plans and handles	
	    cublasStatus_t cstat;
	    cufftHandle cplan;
	    cublasHandle_t candle;
	    cufftResult cufftres;
	    cstat = cublasCreate(&candle); // creating handle
		if(cstat != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr,"CUBLAS Initialization failed...\n");
				return 1;
		}
		///////////////////////////////////////////////////////////////////
		//The FFT heart
		cufftres = cufftPlan1d(&cplan, N, CUFFT_C2C, 1);
		if(cufftres != CUFFT_SUCCESS) {
				fprintf(stderr,"CUFFT Error: Plan creation failed!.\n");
				return 1;
		}
		checkCudaErrors( cudaEventRecord(estart,0) );
		// This is the actual kernel call
		dim3 grid1 = {4,4,4};
		dim3 block1= {32,32,1};
		fftchirp<<<grid1,block1>>>(fft_out, delta, N);
		checkCudaErrors( cudaEventRecord(estop,0) );
		checkCudaErrors( cudaEventSynchronize(estop) );
		checkCudaErrors( cudaEventElapsedTime(&t_fftchirp, estart, estop) );
		// FFT CHIRP timed
		////////////////////////////////////////////////////////////////////
		checkCudaErrors( cudaEventRecord(estart,0) );
		cufftres = cufftExecC2C(cplan, (cufftComplex*)d_in, (cufftComplex*)fft_out, CUFFT_FORWARD); 
		if(cufftres != CUFFT_SUCCESS) {
				fprintf(stderr,"CUFFT Error: Transform failed!.\n");
				return 1;
		}
		dim3 grid2 = {4,4,4};
		dim3 block2= {32,32,1};
		vecpro<<<grid2,block2>>>(fft_out,d_in,fft_out);
		cufftres = cufftExecC2C(cplan,(cufftComplex*)fft_out, (cufftComplex*)fft_out, CUFFT_INVERSE); 
		if(cufftres != CUFFT_SUCCESS) {
				fprintf(stderr,"CUFFT Error: Transform failed!.\n");
				return 1;
		}
		checkCudaErrors( cudaEventRecord(estop,0) );
		checkCudaErrors( cudaEventSynchronize(estop) );
		checkCudaErrors( cudaEventElapsedTime(&t_fft, estart, estop) );
		// FFT timed 
		//////////////////////////////////////////////////////////////////
		
		//////////////////////////////////////////////////////////////////
		// The DDT heart.
		checkCudaErrors( cudaEventRecord(estart,0) );
		dim3 grid3 = {4,4,4};
		dim3 block3= {32,32,1};
		ddtchirp<<<grid3,block3>>>(ddtchirp,delta,N);
		checkCudaErrors( cudaEventRecord(estop,0) );
		checkCudaErrors( cudaEventSynchronize(estop) );
		checkCudaErrors( cudaEventElapsedTime(&t_ddtchrip, estart, estop) );
		// DDT chirp timed 
		//////////////////////////////////////////////////////////////////
		t.x = 1.0;
		t.y = 0.0;
		checkCudaErrors( cudaEventRecord(estart,0) );
		cstat = cublasCgemv(candle, CUBLAS_OP_N, N, N, &t, (cuComplex*)ddtchirp, N, (cuComplex*)d_in, 1, 0, (cuComplex*)ddt_out, 1);
		if(cstat != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr,"CUBLAS Error: GEMV failed!.\n");
				return 1;
		}
		checkCudaErrors( cudaEventRecord(estop,0) );
		checkCudaErrors( cudaEventSynchronize(estop) );
		checkCudaErrors( cudaEventElapsedTime(&t_ddt, estart, estop) );
		// DDT timed 
		//////////////////////////////////////////////////////////////////
		
		// Blocking 
		if(cudaDeviceSynchronize() != cudaSuccess){
				fprintf(stderr,"CUDA Error: Failed to synchronize..\n");
				return 1;
		}
		// Compute MSE
		// Result goes in fft_out
		cstat = cublasCaxpy(candle, N, &t, ddt_out, 1, fft_out, 1);
		if(cstat != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr,"CUBLAS Error: AXPY failed!.\n");
				return 1;
		}
		float mse;
		cstat = cublasScnrm2(candle, N, fft_out, 1, &mse);
		if(cstat != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr,"CUBLAS Error: NRM2 failed!.\n");
				return 1;
		}
		/////////////////////////////////////////////////////////////////
		printf("%ld, %f, %f. %f, %f, %f\n",N, t_fftchirp, t_ddtchrip, t_fft, t_ddt, mse);
		// Exit
		cudaEventDestory(estart); // destroying events
		cudaEventDestory(estop);
		cufftDestroy(cplan); // destorying plan 
		cublasDestroy(candle); // destorying handle
		return 0;
}
