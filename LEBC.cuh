
//Bondry condition:
__global__ void LEBC(double *x1 ,double *x2 , double *x3, double *v1 ,double ux,double *L, double t, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N)
    {
        x1[tid] -= ux * t * round(x3[tid] / L[2]);
        x1[tid] -= L[0] * (round(x1[tid] / L[0]));
        v1[tid] -= ux * round(x3[tid] / L[2]);
        x2[tid] -= L[1] * (round(x2[tid] / L[1]));
        x3[tid] -= L[2] * (round(x3[tid] / L[2]));
    }
}

// This function suppose to calculate the nearest image accounting Lees Edwards:
__device__ __host__ void LeeEdwNearestImage(double x0,double x1, double x2, double y0, double y1, double y2, double *r,double *L,double ux, double t)
{
    //double ux = shear_rate * L[2];
    r[0] = x0 - y0;  r[1] = x1 - y1;    r[2] = x2 - y2;

    r[0] -= ux * t * round(r[2] / L[2]);
    r[0] -= L[0] * (round(r[0] / L[0]));
    r[1] -= L[1] * (round(r[1] / L[1]));
    r[2] -= L[2] * (round(r[2] / L[2]));
}
