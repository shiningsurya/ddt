// The chirp matrix is here 
typedef float2 Complex; 
#define TWOPI 6.2831853071796 
#define DFFAC 2.41e-10 /*DM (pc cm-3) = DFFAC*D (MHz) */
// Very Very Bad Programming Practise Here
#define fsky 1300.3333E6f // Mhz
#define bw 120E6f // Mhz
#define sideband 1

__global__ void ddtchirp(Complex * chirp, float delta, long N) {
		delta = DFFAC * delta;
		int n, l, k;
		n = blockIdx.x * blockDim.x + threadIdx.x;
		l = blockIdx.y * blockDim.y + threadIdx.y;
		k = threadIdx.z;
		//
		float kappa, taper, f;
		f = (n - N/2) * bw / (N-1);
		f += fsky;
		if(f > 0.5 * bw)
				taper = 1.0 / sqrt(1.0 + pow( (f/(0.94 * bw)), 80));
		else
				taper = 1.0 / sqrt(1.0 + pow( ((bw -f)/(0.94 * bw)), 80));
		if(n <= N/2)
				kappa = -1 * delta * f * f /(f + f*fsky);
		else
				kappa = delta * f * f /(f + f*fsky);
		//
		/*
		 *TWOPI * ( k * (n-l)/N) 
		 *TWOPI * ( kappa)
		 */
		chirp[n * N + l].x += taper * cos ( TWOPI * ( ( k * (n-l)/N) + kappa ));
		chirp[n * N + l].y -= taper * sin ( TWOPI * ( ( k * (n-l)/N) + kappa ));
}

__global__ void fftchirp(Complex * chirp, float delta, long N){
		delta = DFFAC * delta;
		int n;
		n = blockIdx.x * blockDim.x + threadIdx.x;
		//
		float kappa, taper, f;
		f = (n - N/2) * bw / (N-1);
		f += fsky;
		if(f > 0.5 * bw)
				taper = 1.0 / sqrt(1.0 + pow( (f/(0.94 * bw)), 80));
		else
				taper = 1.0 / sqrt(1.0 + pow( ((bw -f)/(0.94 * bw)), 80));
		if(n <= N/2)
				kappa = -1 * delta * f * f /(f + f*fsky);
		else
				kappa = delta * f * f /(f + f*fsky);
		//
		/*
		 *TWOPI * ( k * (n-l)/N) 
		 *TWOPI * ( kappa)
		 *****************
		 * Negative to take inverse filter
		 */
		chirp[n].x += taper * cos ( TWOPI * ( kappa ));
		chirp[n].y -= taper * sin ( TWOPI * ( kappa ));
}

__device__ Complex ComplexMult(Complex one, Complex two) {
		Complex ret; 
		ret.x = one.x * two.x - one.y * two.y;
		ret.y = one.x * two.y + one.y * two.x;
		return ret;
}

__global__ void vecpro(Complex * i1, Complex * i2, Complex * out){
		int n;
		n = blockIdx.x * blockDim.x + threadIdx.x;
		out[n] = ComplexMult(i1[n],i2[n]); 
}
