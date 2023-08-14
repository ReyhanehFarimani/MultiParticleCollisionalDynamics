double L[3] = {100, 60, 60};
int n_md = 1;
int m_md = 100;
int swapsize=100;
int simuationtime=1000;
double shear_rate = 0.00;
int density =10;
double h_md = 0.001;
double h_mpcd = 0.1;
int TIME = 0;
int topology = 1;
__device__ int Morse_Potential_variable = 1;       // This is added for deattaching the rings if their distance
                                       // is more than a treshould like 2\sigma.  
std::default_random_engine generator(time(0));
std::uniform_real_distribution<float> realdistribution(0, 1);
static const int blockSize = 512;
//this is a function for error checking in kernels:
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


