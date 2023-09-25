
//streaming: the first function no force field is considered while calculating the new postion of the fluid 
__global__ void mpcd_streaming(double* x, double* v, double timestep, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        
        x[tid] += timestep * v[tid];
        x[tid + N] += timestep * v[tid + N];
        x[tid + 2 * N] += timestep * v[tid + 2 * N];
        
    }
}

__host__ void MPCD_streaming(double* d_x, double* d_v,double h_mpcd, int N , int grid_size)
{
    mpcd_streaming<<<grid_size,blockSize>>>(d_x, d_v, h_mpcd ,N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
}