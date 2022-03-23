
//thermostat

//Making a gamma shape distribuation:
__device__ void gamrnd_d(double* x, double2* params, curandState_t* d_localstates)
{
    double alpha = params->x;
    double beta = params->y;

    if (alpha >= 1){
        curandState_t localState = *d_localstates; // Be careful the change in localState variable needs to be reflected back to d_localStates
        double d = alpha - 1 / 3.0, c = 1 / sqrt(9 * d);
        do{
            double z = curand_normal(&localState);
            double u = curand_uniform(&localState);
            double v = pow((double) 1.0f + c*z, (double) 3.0f);
            double extra = 0;
            if (z > -1 / c && log(u) < (z*z / 2 + d - d*v + d*log(v))){
                *x = d*v / beta;
                *d_localstates = localState;
                return;
            }
        } while (true);
    }
    else{
        double r;
        params->x += 1;
        gamrnd_d(&r, params, d_localstates);

        curandState_t localState = *d_localstates;
        double u = curand_uniform(&localState);
        *x = r*pow((double)u, (double)1 / alpha);
        params->x -= 1;
        return;
    }
}

//Setup kernel for random generators (in parallel)
__global__ void setup_kernel(unsigned int seed, curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(4294967296ULL^seed , id ,0 , &state[id]);
}

// calculating dE of the cell in order to scale the velocity of the particles and set the temprature the desired value
__global__ void E_cell(double* rvx , double* rvy , double* rvz , double *e , int* index, int N, int mass)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];
        atomicAdd(&e[idxx] , mass*rvx[tid]*rvx[tid]/2 );
        atomicAdd(&e[idxx] , mass*rvy[tid]*rvy[tid]/2 );
        atomicAdd(&e[idxx] , mass*rvz[tid]*rvz[tid]/2 );
    }

}


//Thermostat it self: it's calculate the scale based of d_energy 
//(considering the fact if they are nor enough d_energy as a result of low number of particle or low floctuation 
//it won't scale the cell at all, the scaling array after calculation would be considered in calculating the new velocity)
__global__ void MBS (double* scalefactor , int* n , double* e, curandState *state,int Nc )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid<Nc)
    {
        double f = 1.5 * double (n[tid] - 1);

        if (e[tid] > 1e-2)
        {
            double2 params { f , 1}; //1/(kT) =1
            curandState localState = state[tid];
            double ek;
            gamrnd_d(&ek, &params, &localState);
            scalefactor[tid] = sqrt(ek/ e[tid]);
        }
        else 
        {
            scalefactor[tid] = 1;
        }
        //scalefactor[tid] = 1; //nve check
    }
}

__host__ double temp_calc(double *d_vx,
double *d_vy , 
double *d_vz , 
double *d_mdVx,
double *d_mdVy, 
double *d_mdVz,
int mass,
int N, 
int Nmd,
int grid_size = 32)
{
        double *d_tmp;
        double E1, E2;
        cudaMalloc((void**)&d_tmp, sizeof(double)*grid_size);
        sumsquared3arrayCommMultiBlock<<<grid_size, blockSize>>>(d_vx, d_vy, d_vz, N, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy(&E1, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        sumsquared3arrayCommMultiBlock<<<grid_size, blockSize>>>(d_mdVx, d_mdVy, d_mdVz, Nmd, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy(&E2, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_tmp);
        return (E1 + E2*mass)/(N+Nmd)/3;
}


__host__ void thermo_test(FILE *file,
double *d_vx,
double *d_vy, 
double *d_vz,
int mass,
int N)
{
    double *h_vx , *h_vy , *h_vz;
    h_vx = (double*)malloc(sizeof(double) * N);
    h_vy = (double*)malloc(sizeof(double) * N);
    h_vz = (double*)malloc(sizeof(double) * N);
    cudaMemcpy(h_vx, d_vx, sizeof(double) * N , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy, d_vy, sizeof(double) * N , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vz, d_vz, sizeof(double) * N , cudaMemcpyDeviceToHost);

    for (int i =0 ; i<N-1 ; ++i)
    {
        fprintf(file , "%f," , h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i] +h_vz[i]*h_vz[i]);
    }
    int i = N-1;
    fprintf(file , "%f\n" , h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i] +h_vz[i]*h_vz[i]);

    free(h_vx);
    free(h_vy);
    free(h_vz);
}