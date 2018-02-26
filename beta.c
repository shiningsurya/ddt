#ifndef _BETA_H_
#include "beta.h"
#endif 

int main(int argc, char * argv[]){
    long long int i,j;
    if(argc < 2){ 
    	   printf("DDT\n(Written by Suryarao Bethapudi)\n");
    	   printf("[ep14btech11008@iith.ac.in]\n");
    	   printf("-----------------------------------\n");
    	   printf("usage : beta numsamp dm filename\n");
    	   exit(0);
    }
    bw = 120E6;
    fsky = 1300.3333;
    sideband = 1;
    // printf("Number of samples to generate:");scanf("%ld",&numsam);fflush(stdin);
    // printf("DM to use:");scanf("%ld",&numsam);fflush(stdin);
    FILE *fp;
    fp = fopen(argv[3],"r");
    fftw_complex *in, *outfft, *outddt, *rdm;
    double dm;
    double * in_i, * in_r, * y_1, *y_2, * y_3, *y_4;
    double *mdm_i, *mdm_r;
    long long int numsamp = (long long int)atol(argv[1]);
    dm = (double)strtod(argv[2],NULL);
    // declaration done
    // Malloc
    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*numsamp);
    in_r = (double*)malloc(sizeof(double)*numsamp);
    in_i = (double*)malloc(sizeof(double)*numsamp);
    outfft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*numsamp);
    outddt = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*numsamp);
    mdm_r = (double*)malloc(sizeof(double)*numsamp*numsamp);
    mdm_i = (double*)malloc(sizeof(double)*numsamp*numsamp);
    rdm = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*numsamp);
    double dx;
    for(i = 0 ; i < numsamp ; i++) { 
		fscanf(fp,"%lf\n",&dx);
		in[i][0] = dx;
		in[i][1] = 0.0;
		in_r[i] = dx;
		in_i[i] = 0.0;
	}
	y_1 = (double*)malloc(sizeof(double)*numsamp);
	y_2 = (double*)malloc(sizeof(double)*numsamp);
	y_3 = (double*)malloc(sizeof(double)*numsamp);
	y_4 = (double*)malloc(sizeof(double)*numsamp);
    // Main ring
    clock_t start,end;
    clock_t fft_ini, fft_dd, ddt_ini, ddt_dd;
    double d1,d2,d3,d4;
    // 
    start = clock();
    fft_dm(dm, rdm, numsamp);
    end = clock();
    d1 = (double)(end - start)/CLOCKS_PER_SEC;
    // ---
    start = clock();
    ddt_dm(dm, mdm_r,mdm_i, numsamp);
    end = clock();
    d2 = (double)(end - start)/CLOCKS_PER_SEC;
    // --- 
    start = clock();
    test_fft(in, numsamp, rdm, outfft);
    end = clock();
    d3 = (double)(end - start)/CLOCKS_PER_SEC;
    // --- 
    start = clock();
    test_ddt(in_r,in_i, numsamp, mdm_r,mdm_i,y_1,y_2,y_3,y_4, outddt);
    end = clock();
    d4 = (double)(end - start)/CLOCKS_PER_SEC;
    // ---
    // Compute differences 
    double mean, var;
    for(i = 0; i < numsamp;i++) {
    		mean += (outddt[i] - outfft[i])^2;
	}
	var = mean / numsamp;
    // IO
    printf("----------------SUMMARY----------------\n");
    printf("Number of samples.................  %lld\n",numsamp);
    printf("Dispersion Measure................  %lf\n",dm);
    printf("FFT SETUP TIME   .................  %lf\n",d1);
    printf("DDT SETUP TIME   .................  %lf\n",d2);
    printf("FFT WORK TIME    .................  %lf\n",d3);
    printf("DDT WORK TIME    .................  %lf\n",d4);
    printf("MSE ..............................  %lf\n",var);
    printf("SE ...............................  %lf\n",mean);
    // End
    fftw_free(in);
    fftw_free(outddt);
    fftw_free(outfft);
    fftw_free(rdm);
    free(mdm_i);
    free(mdm_r);
    free(y_1);
    free(y_2);
    free(y_3);
    free(y_4);
    free(in_i);
    free(in_r);
    fclose(fp);
    return 0;
}


