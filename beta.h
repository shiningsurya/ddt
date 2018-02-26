#define _BETA_H_ 
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>

#define TWOPI 6.2831853071796 
#define DFFAC 2.41e-10 /*DM (pc cm-3) = DFFAC*D (MHz) */
#include "time.h"
#include "fftw3.h"
void fft_dm(double dm, fftw_complex * rdm, long long int numsamp);
void ddt_dm(double dm, double * rdm_r,double * rdm_i, long long int numsamp);
// void test_ddt(fftw_complex * in, long long int numsamp, fftw_complex * dm, fftw_complex * out);
void test_fft(fftw_complex * in, long long int numsamp, fftw_complex * dm, fftw_complex * out);
void test_ddt(double * in_r, double * in_i, long long int numsamp, double * dm_r, double * dm_i, double * y_1, double * y_2, double * y_3, double * y_4, fftw_complex * out); 
double bw, fsky;
int sideband;

