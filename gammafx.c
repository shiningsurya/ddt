// This is Gammafx.
//
// This is the original code
//
int main(int argc, char * argv[]){
		if(argc < 2) {
				printf("Direct De-dispersion Transform\n");
				printf("Usage : <program> <DM> <exponent of two> <filename>\n");
				printf("Written by Suryarao Bethapudi[ep14btech11008@iith.ac.in]\n");
				return 0;
		}
		// basic part
		bw = 120E6; // 120 Mhz
		fsky = 1300.3333; // Mhz
		sideband = 1;
		//
		long long int N;
		double dm;
		FILE * fp;
		fp = fopen(argv[3],"r");
		N = pow(2,numsamp);
		dm = (double)strtod(argv[1],NULL);
		// FILE IO 
		for(i = 0; i < N;i++){
				fscanf(fp,"%lf\n",&dx);
		}


	    // cuda variables, types
	    cudaError_t cuerr;
	    cudaEvent_t estart, estop; // to measure time

	    cublasStatus_t cstat;
	    cufftHandle cplan;
	    cublasHandle_t candle;
	    cstat = cublasCreate(&candle); // creating handle
		if(cstat != CUBLAS_STATUS_SUCCESS) {
				printf("CUBLAS Initialization failed...\n");
				return 1;
		}

		//The FFT heart
		cufftres = cufftPlan1d(cplan, N, CUFFT_C2C, 1);
		if(cufftres ! = CUFFT_SUCCESS) {
				fprintf(stderr,"CUFFT Error: Plan creation failed!.\n");
				return 1;
		}
		fftchirp<<<,>>>(dm,fftdm,..);
		cufftres = cufftExecC2C(&cplan, in, in, CUFFT_FORWARD); 
		if(cufftres ! = CUFFT_SUCCESS) {
				fprintf(stderr,"CUFFT Error: Transform failed!.\n");
				return 1;
		}
		vecpro<<<,>>>(in,fftdm,out);
		cufftres = cufftExecC2C(&cplan, out, out, CUFFT_INVERSE); 
		if(cufftres ! = CUFFT_SUCCESS) {
				fprintf(stderr,"CUFFT Error: Transform failed!.\n");
				return 1;
		}

		// The DDT heart.
		ddtchirp<<<,>>>(dm,ddtdm);
		cublasCgemv(candle, CUBLAS_OP_N, N, N, alpha, N, in, 1, beta, out, 1);

		// Blocking 
		if(cudaDeviceSynchronize() != cudaSuccess){
				fprintf(stderr,"CUDA Error: Failed to synchronize..\n");
				return 1;
		}


		cudaEventDestory(estart); // destroying events
		cudaEventDestory(estop);
		cufftDestroy(cplan); // destorying plan 
		cublasDestroy(candle); // destorying handle
		return 0;
}
