__global__ void grid_shift(double *x , double *y , double *z , double *r, int N )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        x[tid] += r[0];
        y[tid] += r[1];
        z[tid] += r[2];
    }
}
__global__ void de_grid_shift(double *x , double *y , double *z , double *r , int N )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        x[tid] -= r[0];
        y[tid] -= r[1];
        z[tid] -= r[2];
    }
}

__host__ void Sort_begin(double *d_x , double *d_y , double *d_z ,double *d_vx, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ ,double *d_mdVx, int *d_mdIndex , double ux,
    double *d_L , double *d_r ,int N ,int Nmd , double real_time , int grid_size)
{
    grid_shift<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_r, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );            

    LEBC<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_vx, ux , d_L, real_time, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    grid_shift<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_r, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx , ux , d_L, real_time , Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    
    cellSort<<<grid_size,blockSize>>>(d_x,d_y,d_z,d_L,d_index,N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cellSort<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_L, d_mdIndex, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__host__ void Sort_finish(double *d_x , double *d_y , double *d_z ,double *d_vx, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ ,double *d_mdVx, int *d_mdIndex , double ux,
    double *d_L , double *d_r ,int N ,int Nmd , double real_time , int grid_size)
{
    de_grid_shift<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_r,N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    LEBC<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_vx, ux , d_L, real_time, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    de_grid_shift<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_r, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx , ux , d_L, real_time , Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
}
