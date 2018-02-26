#ifndef _BETA_H_
#include "beta.h"
#endif



/*******************
 * I just realized the structure of that chrip matrix
 * We can exploit it like anything 
 * it just depends on (n-l)
 *
 * My lazy ass has made me write code which uses global variables
 * They will be defined in the include file. 
 * bw, taper, eta will few in them.
 * fsky , sideband 
 * ***************/

void fft_dm(double dm, fftw_complex * rdm, long long int numsamp){
		double r,s,f;
		long long int i,i0 = numsamp/2;
		double taper;
		bw = bw * 1e-6;
		s = TWOPI * dm / DFFAC;

		for( i = 0 ; i < numsamp ;i++) {
				f = i * bw / i0;
				if(f > bw)  {
						f -= bw; 
						f = bw - f;
				} 
				if(i <= i0) r = -1 * f * f * s / ( (fsky + sideband * f) * fsky * fsky );  
				else r = f * f * s / ( (fsky + sideband * f) * fsky * fsky);

				if (f > 0.5*bw) taper = 1.0 / sqrt(1.0 + pow( (f/(0.94 * bw)) , 80));
				else  taper = 1.0 / sqrt(1.0 + pow( ( (bw - f)/( 0.84 * bw)), 80 ) );

				rdm[i][0] = (double)( cos(r) * taper );
				rdm[i][1] = ( -1.0* (double)( sin(r) * taper));

		}
		rdm[0][0] = 0.0;
		rdm[0][1] = 0.0;
}

void ddt_dm(double dm, double * rdm_r,double * rdm_i, long long int numsamp){
		/*******************************
		 * The exponential part depends only on (n-l)
		 * thankfully for us, both n,l vary over [0,N_F-1] 
		 * for a given k. 
		 * We will use this fact in optimizing this up
		 * ****************************
		 * Somewhere along this code I lost my mind. 
		 * Left the optimization for a later stage. 
		 * Sorry CPU. 
		 * ****************************/
		long long int i, k, n,l,i0;
		double kappa;
		double taper,f0,f;
		double Delta = dm * DFFAC;
		i0 = numsamp / 2;
		f0 = bw/2;
		fftw_complex te;
		for(k = 0; k < numsamp; k++){
				f = k * bw / i0;
				if(f > bw)  {
						f -= bw; 
						f = bw - f;
				} 
				if(k > i0) kappa = Delta * f * f / (f + f0);
				else kappa = Delta * f * f / (f0 + f);
				if (f > 0.5*bw) taper = 1.0 / sqrt(1.0 + pow( (f/(0.94 * bw)) , 80));
				else  taper = 1.0 / sqrt(1.0 + pow( ( (bw - f)/( 0.84 * bw)), 80 ) );
				for(n = 0; n < numsamp ; n++){
						for(l = 0; l < numsamp; l++) {
								te[0] = taper * cos(TWOPI * (kappa + ( f * (n-l)/numsamp)));
								te[1] = taper * sin(TWOPI * (kappa + ( f * (n-l)/numsamp)));
								//printf("%lf %lf %lf %lf\n%lld\t%lld\n",te[0],te[1],rdm[n*numsamp +l][0],rdm[n*numsamp + l][1],n,l);
								// double 
								rdm_r[n*numsamp + l] = rdm_r[n*numsamp + l] + te[0];
								rdm_i[n*numsamp + l] = rdm_i[n*numsamp + l] + te[1];

						}
				}
		}
}

//void test_ddt(fftw_complex * in, long long int numsamp, fftw_complex * dm, fftw_complex * out){
void test_ddt(double * in_r, double * in_i, long long int numsamp, double * dm_r, double * dm_i, double * y_1, double * y_2, double * y_3, double * y_4, fftw_complex * out) {
		/************************************************
		 * in --> fftw_complex pointer 
		 * out --> fftw_complex pointer 
		 * dm --> fftw_complex pointer 
		 * numsamp --> number of samples, size of the 2D matrix
		 * *********************************************
		 * This is basically matrix multiplication
		 * *********************************************/
		// long long int n,l;
		// double resumx, resumy;
		cblas_dgemv(CblasRowMajor,CblasNoTrans,numsamp,numsamp,1.0,dm_r,numsamp,in_r,1,0,y_1,1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,numsamp,numsamp,1.0,dm_r,numsamp,in_i,1,0,y_2,1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,numsamp,numsamp,1.0,dm_i,numsamp,in_r,1,0,y_3,1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,numsamp,numsamp,1.0,dm_i,numsamp,in_i,1,0,y_4,1);
		long long int n;
		for(n = 0; n < numsamp;n++){
				out[n][0] = y_1[n]  - y_4[n];
				out[n][1] = y_2[n]  + y_3[n];
		}
		/****************************
		  for(n = 0; n < numsamp;n++) {
		// resumx is real resumy is imaginary
		resumx = 0.0;
		resumy = 0.0;
		for(l = 0; l < numsamp; l++) {
		// resum += in[l] * dm[n][l] 
		resumx += (in[l][0] * dm[n*numsamp + l][0]) - (in[l][1] * dm[n*numsamp + l][1]);
		resumy += (in[l][0] * dm[n*numsamp + l][1]) + (in[l][1] * dm[n*numsamp + l][0]);
		out[n][0] = resumx;
		out[n][1] = resumy;
		}
		}
		// Done
		// This is not at all optimized. Hence, 
		 ****************************/
		return;
}

void test_fft(fftw_complex * in, long long int numsamp, fftw_complex * dm, fftw_complex * out){
		/************************************************
		 * in --> fftw_complex pointer 
		 * out --> fftw_complex pointer 
		 * dm --> fftw_complex pointer 
		 * numsamp --> number of samples, size of the 2D matrix
		 * *********************************************
		 * This is basically fft
		 * *********************************************/
		fftw_plan pf, pd;
		pf = fftw_plan_dft_1d(numsamp, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
		pd = fftw_plan_dft_1d(numsamp, out, out, FFTW_BACKWARD, FFTW_ESTIMATE);
		long long int i;
		// Start main algo
		fftw_execute(pf);
		for(i = 0 ; i < numsamp; i++) {
				out[i][0] = (in[i][0]) * dm[i][0] - (in[i][1] * dm[i][1]);
				out[i][1] = (in[i][0]) * dm[i][1] + (in[i][1] * dm[i][0]);
		}
		fftw_execute(pd);
		// Done
		return;
}


