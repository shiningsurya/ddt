// The chirp matrix is here 

__global__
void ddtchirp(double * chirp1, double * chirp2, double delta, long long int N) {
		int n, l, k;
		n = blockIdx.x * TILE_WIDTH + threadIdx.x;
		l = blockIdx.y * TILE_WIDTH + threadIdx.y;
		k = threadIdx.z;
		//
		double kappa, taper, f;
		f = (n - N/2) * bw / (N-1);
		f += fsky;
		if(f > 0.5 * bw)
				taper = 1.0 / sqrt(1.0 + pow( (f/(0.94 * bw)), 80);
		else
				taper = 1.0 / sqrt(1.0 + pow( ((bw -f)/(0.94 * bw)), 80);
		if(n <= N/2)
				kappa = -1 * delta * f * f /(f + f*fsky);
		else
				kappa = delta * f * f /(f + f*fsky);
		//
		/*
		 *TWOPI * ( k * (n-l)/N) 
		 *TWOPI * ( kappa)
		 */
		chirp1[n * N + l] += taper * cos ( TWOPI * ( ( k * (n-l)/N) + kappa ));
		chirp2[n * N + l] -= taper * sin ( TWOPI * ( ( k * (n-l)/N) + kappa ));
}

__global__
void fftchirp(double * chirp1, double * chirp2, double delta, long long int N){
		int n;
		n = blockIdx.x + TILE_WIDTH + threadIdx.x;
		//
		double kappa, taper, f;
		f = (n - N/2) * bw / (N-1);
		f += fsky;
		if(f > 0.5 * bw)
				taper = 1.0 / sqrt(1.0 + pow( (f/(0.94 * bw)), 80);
		else
				taper = 1.0 / sqrt(1.0 + pow( ((bw -f)/(0.94 * bw)), 80);
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
		chirp1[n] += taper * cos ( TWOPI * ( kappa ));
		chirp2[n] -= taper * sin ( TWOPI * ( kappa ));
}

__global__
void vecpro(double * i1, double * i2, double * out){
		int n;
		n = blockIdx.x * TILE_WIDTH + threadIdx.x;
		out[n] = i1[n] * i2[n];
}
