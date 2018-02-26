// When you want to get some data 
// And you don't know how 
//
#include <stdio.h>
#include <math.h>
#include<stdlib.h>
#include<time.h>

float gasdev(long *idum);
float ran1(long *idum);

int main(int argc, char * argv[]){ 
  if(argc < 2) {
  	  printf("This program creates a simple time-series data to \n test the integrity of DDT.\n");
  	  printf("Usage : makedata filename\n");
  	  exit(0);
  }
  FILE * fp;
  fp = fopen(argv[1],"w+");
  double sampletime_base = 250.0E-6; // Base is 250 microsecond time samples
  double downsamp    = 1.0;
  double Tobs        = 6.0;    // Observation duration in seconds
  double dt          = downsamp*sampletime_base;     // s (0.25 ms sampling)
  double f0          = 1300.3333;    // MHz (highest channel!)
  double bw          = 120.0; // MHz
  long  nchans       = 1;
  double df          = -1.0*bw/nchans;   // MHz   (This must be negative!)

  long  nsamps       = Tobs / dt;
  double datarms     = 25.0;
  double sigDM       = 42.420; // The answer to every problem in life #42 
  double sigT        = 3.14159; // seconds into time series (at f0)
  double sigamp      = 28.0; // amplitude of signal

  double dm_start    = 2.0;    // pc cm^-3
  double dm_end      = 100.0;    // pc cm^-3
  double pulse_width = 4.0;   // ms
  double dm_tol      = 1.25;
  long  in_nbits    = 8;
  long  out_nbits   = 32;  // DON'T CHANGE THIS FROM 32, since that signals it to use floats
        
  long  dm_count;
  long  max_delay;
  long  nsamps_computed;
  double *output = 0;

  int i,nc,ns,nd;
  const double *dmlist;
  //const long *dt_factors;
  double *delay_s;
  long idum=-1*time(NULL);


  double *rawdata;

  printf("----------------------------- INPUT DATA ---------------------------------\n");
  printf("Frequency of highest chanel (MHz)            : %.4f\n",f0);
  printf("Bandwidth (MHz)                              : %.2f\n",bw);
  printf("NCHANS (Channel Width [MHz])                 : %lu (%f)\n",nchans,df);
  printf("Sample time (after downsampling by %.0f)        : %f\n",downsamp,dt);
  printf("Observation duration (s)                     : %f (%lu samples)\n",Tobs,nsamps);
  printf("Data RMS (%2lu bit input data)                 : %f\n",in_nbits,datarms);
  printf("Input data array size                        : %lu MB\n",(nsamps*nchans*sizeof(float))/(1<<20));
  printf("\n");

  /* First build 2-D array of floats with our signal in it */
  rawdata = malloc(nsamps*nchans*sizeof(double));
  for (ns=0; ns<nsamps; ns++) {
    for (nc=0; nc<nchans; nc++) {
      rawdata[ns*nchans+nc] = datarms*gasdev(&idum);
    }
  }
  double a,b;
  /* Now embed a dispersed pulse signal in it */
  delay_s = malloc(nchans*sizeof(double));
for (nc=0; nc<nchans; nc++) {
    a = 1.f/(f0+nc*df);
    b = 1.f/f0;
    delay_s[nc] = sigDM * 4.15e3 * (a*a - b*b);
  }
  printf("Embedding signal\n");
  for (nc=0; nc<nchans; nc++) {
    ns = (int)((sigT + delay_s[nc])/dt);
    if (ns > nsamps) {
      printf("ns too big %u\n",ns);
      exit(1);
    }
    rawdata[ns*nchans + nc] += sigamp;
  }

  printf("----------------------------- INJECTED SIGNAL  ----------------------------\n");
  printf("Pulse time at f0 (s)                      : %.6f (sample %lu)\n",sigT,(long)(sigT/dt));
  printf("Pulse DM (pc/cm^3)                        : %f \n",sigDM);
  printf("Signal Delays : %f, %f, %f ... %f\n",delay_s[0],delay_s[1],delay_s[2],delay_s[nchans-1]);

  printf("Writing to file....");
  for(ns = 0; ns < nsamps; ns++) {
  	  for(nc = 0 ; nc < nchans ; nc++) {
  	  	  fprintf(fp,"%lf ",rawdata[ns*nchans + nc]);
	  }
	  fprintf(fp,"\n");
  }
  printf("Done....Exiting...\n");
  fclose(fp);
  return 0;
}
