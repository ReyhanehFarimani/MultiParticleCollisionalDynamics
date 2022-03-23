
//cell sort, calculating the index of each particle there is a unique ID for each cell  based on their position 
__global__ void cellSort(double* x,double* y,double* z, double *L, int* index, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
    index[tid] = int(x[tid] + L[0] / 2) + L[0] * int(y[tid] + L[1] / 2) + L[0] * L[1] * int(z[tid] + L[2] / 2);
    }

}

__global__ void MakeCellReady(double* ux , double* uy , double* uz,double* e, int* n,int Nc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nc)
    {
        ux[tid] = 0;
        uy[tid] = 0;
        uz[tid] = 0;
        n[tid] = 0;
        e[tid]=0;
    }

}
__global__ void MeanVelCell(double* ux, double* vx,double* uy, double* vy,double* uz, double* vz,int* index, int *n,int *m, int mass, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];
        double tmp =vx[tid] *mass;
        atomicAdd(&ux[idxx] , tmp );
        tmp =vy[tid] *mass;
        atomicAdd(&uy[idxx] , tmp );
        tmp =vz[tid] *mass;
        atomicAdd(&uz[idxx] , tmp );
        atomicAdd(&n[idxx] , 1 );
        atomicAdd(&m[idxx], mass);
    }
}

__global__ void RotationStep1(double *ux , double *uy ,double *uz,double *rot, int *m ,double *phi , double *theta, int Nc)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double alpha = 13.0 / 18.0 * M_PI;
    double co = cos(alpha), si = sin(alpha);
    if (tid<Nc)
    {
        theta[tid] = theta[tid]* 2 -1; 
        phi[tid] = phi[tid]* M_PI*2;
        ux[tid] = ux[tid]/m[tid];
        uy[tid] = uy[tid]/m[tid];
        uz[tid] = uz[tid]/m[tid];
        double n1 = std::sqrt(1 - theta[tid] * theta[tid]) * cos(phi[tid]);
        double n2 = std::sqrt(1 - theta[tid] * theta[tid]) * sin(phi[tid]);
        double n3 = theta[tid];
        
        rot[tid*9+0] =n1 * n1 + (1 - n1 * n1) * co ;
        rot[tid*9+1] =n1 * n2 * (1 - co) - n3 * si;
        rot[tid*9+2] =n1 * n3 * (1 - co) + n2 * si;
        rot[tid*9+3] =n1 * n2 * (1 - co) + n3 * si;
        rot[tid*9+4] =n2 * n2 + (1 - n2 * n2) * co;
        rot[tid*9+5] =n2 * n3 * (1 - co) - n1 * si;
        rot[tid*9+6] =n1 * n3 * (1 - co) - n2 * si;
        rot[tid*9+7] =n2 * n3 * (1 - co) + n1 * si;
        rot[tid*9+8] =n3 * n3 + (1 - n3 * n3) * co;
        
    }
}

__global__ void RotationStep2(double *rvx , double *rvy, double *rvz , double *rot , int *index,int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if(tid<N)
    {
        const unsigned int idxx = index[tid];

        double RV[3] = {rvx[tid] , rvy[tid] , rvz[tid]};
        double rv[3] = {0.};

        for (unsigned int i = 0; i < 3; i++)
        {
            for (unsigned int j = 0; j < 3; j++)
                rv[i] += rot[idxx*9+3*j+i] * RV[j];
        }
        
        rvx[tid] = rv[0];
        rvy[tid] = rv[1];
        rvz[tid] = rv[2];   
    }
}

__global__ void MakeCellReady(double* ux , double* uy , double* uz,double* e, int* n,int* m,int Nc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nc)
    {
        ux[tid] = 0;
        uy[tid] = 0;
        uz[tid] = 0;
        n[tid] = 0;
        e[tid]=0;
        m[tid] = 0;
    }

}

__global__ void UpdateVelocity(double* vx, double *vy, double *vz , double* ux, double *uy , double *uz ,double *factor,int *index, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];
        vx[tid] = ux[idxx] + vx[tid]*factor[idxx]; 
        vy[tid] = uy[idxx] + vy[tid]*factor[idxx];
        vz[tid] = uz[idxx] + vz[tid]*factor[idxx];
    }

}

__global__ void relativeVelocity(double* ux , double* uy , double* uz, int* n, double* vx, double* vy, double* vz, int* index,int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];
        vx[tid] = vx[tid] - ux[idxx] ;
        vy[tid] = vy[tid] - uy[idxx] ;
        vz[tid] = vz[tid] - uz[idxx] ;
    }

}

__host__ void MPCD_MD_collision(double* d_vx ,double*  d_vy ,double*  d_vz , int* d_index,
double* d_mdVx ,double*  d_mdVy,double*  d_mdVz , int *d_mdIndex,
double* d_ux ,double*  d_uy ,double*  d_uz ,
double *d_e ,double *d_scalefactor, int *d_n , int* d_m,
double *d_rot, double *d_theta, double *d_phi ,
int N , int Nmd, int Nc,
curandState *devStates, int grid_size)
{
            MakeCellReady<<<grid_size,blockSize>>>(d_ux , d_uy, d_uz ,d_e, d_n,d_m,Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );            

            MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_vx , d_uy, d_vy, d_uz, d_vz, d_index, d_n , d_m, 1 ,N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_mdVx , d_uy, d_mdVy, d_uz, d_mdVz, d_mdIndex, d_n , d_m, density ,Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            RotationStep1<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_rot, d_m, d_phi, d_theta, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_vx, d_vy, d_vz, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            RotationStep2<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_rot, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            RotationStep2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_rot, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            E_cell<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_e, d_index, N, 1);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            E_cell<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_e, d_mdIndex, Nmd , density);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            MBS<<<grid_size,blockSize>>>(d_scalefactor,d_n,d_e,devStates, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_ux, d_uy, d_uz, d_scalefactor, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_ux, d_uy, d_uz, d_scalefactor, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
}
