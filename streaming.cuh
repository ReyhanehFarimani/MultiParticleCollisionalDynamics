
//streaming: the first function no force field is considered while calculating the new postion of the fluid 
__global__ void mpcd_streaming(double* x,double* y ,double* z,double* vx ,double* vy,double* vz ,double timestep, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        
        x[tid] += timestep * vx[tid];
        y[tid] += timestep * vy[tid];
        z[tid] += timestep * vz[tid];
        
    }
}

__host__ void MPCD_streaming(double* d_x,double* d_y ,double* d_z,double* d_vx ,double* d_vy,double* d_vz ,double h_mpcd, int N , int grid_size)
{
    mpcd_streaming<<<grid_size,blockSize>>>(d_x, d_y, d_z , d_vx, d_vy, d_vz, h_mpcd ,N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
}