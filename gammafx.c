// This is Gammafx.
//
// This is the original code
//
int main(int argc, char * argv[]){
		if(argc < 2) {
				printf("----DDT---FFT----\n");
				printf("Usage : <program> <DM> <numsamp> <filename>\n");
				return 0;
		}
	    // cuda variables, types
	    cudaError_t cuerr;
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
		//
		cufftres = cufftExecC2C(&cplan, in, out, CUFFT_FORWARD); 
		if(cufftres ! = CUFFT_SUCCESS) {
				fprintf(stderr,"CUFFT Error: Transform failed!.\n");
				return 1;
		}
		cufftres = cufftExecC2C(&cplan, in, out, CUFFT_INVERSE); 



		// The DDT heart. 
		cublasCgemv(candle, CUBLAS_OP_N, N, N, alpha, N, in, 1, beta, out, 1);





		cufftDestroy(cplan); // destorying plan 
		cublasDestroy(candle); // destorying handle
		return 0;
}
