
// position and velocity of MD particles are initialized here ( currenty it only supports rings)
__host__ void initMD(double *d_mdX, double *d_mdY , double *d_mdZ ,
 double *d_mdVx , double *d_mdVy , double *d_mdVz, 
 double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *d_Fx_holder , double *d_Fy_holder, double *d_Fz_holder,
 double *d_L, double ux, double xx[3], int n, int m, int topology, int mass)
{
    int Nmd = n * m;
    double *mdX, *mdY, *mdZ, *mdVx, *mdVy , *mdVz, *mdAx , *mdAy, *mdAz;
    //host allocation:
    mdX = (double*)malloc(sizeof(double) * Nmd);  
    mdY = (double*)malloc(sizeof(double) * Nmd);  
    mdZ = (double*)malloc(sizeof(double) * Nmd);
    mdVx = (double*)malloc(sizeof(double) * Nmd); 
    mdVy = (double*)malloc(sizeof(double) * Nmd); 
    mdVz = (double*)malloc(sizeof(double) * Nmd);
    mdAx = (double*)malloc(sizeof(double) * Nmd); 
    mdAy = (double*)malloc(sizeof(double) * Nmd); 
    mdAz = (double*)malloc(sizeof(double) * Nmd);
    std::normal_distribution<double> normaldistribution(0, 0.44);
    double theta;
    double r;
    theta = 4 * M_PI_2 / m;
    r=m/(4 * M_PI_2);
    if (topology == 0) //linear polymer topology
    {
        for (unsigned int i = 0 ; i<Nmd ; i++)
        {
                
            mdAx[i]=0;
            mdAy[i]=0;
            mdAz[i]=0;
            //monomer[i].init(kT ,box, mass);
            mdVx[i] = normaldistribution(generator);
            mdVy[i] = normaldistribution(generator);
            mdVz[i] = normaldistribution(generator);
            mdX[i]  = i;
            mdY[i]  = 0;
            mdZ[i]  = 0;
        }   
            
    }
    if (topology == 1) //poly[2]catnane topology
    {
        for (unsigned int j = 0 ; j< n ; j++)
        { 
            for (unsigned int i =0 ; i<m ; i++)
            {
                mdAx[i+j*m]=0;
                mdAy[i+j*m]=0;
                mdAz[i+j*m]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*m] = normaldistribution(generator);
                mdVy[i+j*m] = normaldistribution(generator);
                mdVz[i+j*m] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*m]  = xx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);
                if ( j%2 == 0 )
                {
                    mdY[i+j*m]  = xx[1] + r * cos(i *theta);
                    //monomer[i].x[2]  = xx[2];
                    mdZ[i+j*m]  = xx[2];
                }
                if(j%2==1)
                {
                    mdZ[i+j*m]  = xx[2] + r * cos(i *theta);
                    //monomer[i].x[2]  = xx[2];
                    mdY[i+j*m]  = xx[1];

                }            
            }   
            xx[0]+=1.2*r;
        }
    }            
    if (topology == 2) //bonded ring topology
    {
        for (unsigned int j = 0 ; j< n ; j++)
        {
            
            for (unsigned int i =0 ; i<m ; i++)
            {
                
                mdAx[i+j*m]=0;
                mdAy[i+j*m]=0;
                mdAz[i+j*m]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*m] = normaldistribution(generator);
                mdVy[i+j*m] = normaldistribution(generator);
                mdVz[i+j*m] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*m]  = xx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);

                mdY[i+j*m]  = xx[1] + r * cos(i *theta);
                //monomer[i].x[2]  = xx[2];
                mdZ[i+j*m]  = xx[2];

            
            }
            
            xx[0]+=(2*r+1) ;
        }
    } 
    
    if (topology == 3) //physically bonded ring topology
    {
        for (unsigned int j = 0 ; j< n ; j++)
        {
            
            for (unsigned int i =0 ; i<m ; i++)
            {
                
                mdAx[i+j*m]=0;
                mdAy[i+j*m]=0;
                mdAz[i+j*m]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*m] = normaldistribution(generator);
                mdVy[i+j*m] = normaldistribution(generator);
                mdVz[i+j*m] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*m]  = xx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);

                mdY[i+j*m]  = xx[1] + r * cos(i *theta);
                //monomer[i].x[2]  = xx[2];
                mdZ[i+j*m]  = xx[2];

            
            }
            
            xx[0]+=(2*r+1) ;
        }
    }
    

        double px =0 , py =0 ,pz =0;
        for (unsigned int i =0 ; i<Nmd ; i++)
        {
            px+=mdVx[i] ; 
            py+=mdVy[i] ; 
            pz+=mdVz[i] ;
        }

        for (unsigned int i =0 ; i<Nmd ; i++)
        {
            mdVx[i]-=px/Nmd ;
            mdVy[i]-=py/Nmd ;
            mdVz[i]-=pz/Nmd ;
        }
    cudaMemcpy(d_mdX ,mdX, Nmd*sizeof(double), cudaMemcpyHostToDevice);   
    cudaMemcpy(d_mdY ,mdY, Nmd*sizeof(double), cudaMemcpyHostToDevice);   
    cudaMemcpy(d_mdZ ,mdZ, Nmd*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mdVx ,mdVx, Nmd*sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mdVy ,mdVy, Nmd*sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mdVz ,mdVz, Nmd*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mdAx ,mdAx, Nmd*sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mdAy ,mdAy, Nmd*sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mdAz ,mdAz, Nmd*sizeof(double), cudaMemcpyHostToDevice);



    free(mdX);    
    free(mdY);    
    free(mdZ);
    free(mdVx);   
    free(mdVy);   
    free(mdVz);
    free(mdAx);   
    free(mdAy);   
    free(mdAz);

}


// a tool for resetting a vector to zero!
__global__ void reset_vector_to_zero(double *F1 , double *F2 , double *F3 , int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size)
    {
        F1[tid] = 0 ;
        F2[tid] = 0 ;
        F3[tid] = 0 ;
    }
}
//sum kernel: is used for sum the interaxtion matrix:F in one axis and calculate acceleration.
__global__ void sum_kernel(double *F1 ,double *F2 , double *F3,
 double *A1 ,double *A2 , double *A3,
  int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size)
    {
        double sum =0;
        for (int i = 0 ; i<size ; ++i)
        {
            int index = tid *size +i;
            sum += F1[index];
        }
        A1[tid] = sum;
    }
    if (size<tid+1 && tid<2*size)
    {
        tid -=size;
        double sum =0;
        for (int i = 0 ; i<size ; ++i)
        {
            int index = tid *size +i ;
            sum += F2[index];
        }
        A2[tid] = sum;        
    }
    if (2*size<tid+1 && tid<3*size)
    {
        tid -=2*size;
        double sum =0;
        for (int i = 0 ; i<size ; ++i)
        {
            int index = tid *size +i;
            sum += F3[index];
        }
        A3[tid] = sum;        
    }

}

//calculating interaction matrix of the system in the given time
__global__ void nb_b_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *L,int size , double ux, int mass, double real_time, int m , int topology)
{
    int size2 = size*(size);

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size2)
    {
        int ID1 = int(tid /size);
        int ID2 = tid%(size);
        if(ID1 != ID2) 
        {
        double r[3];
        LeeEdwNearestImage(mdX[ID1], mdY[ID1], mdZ[ID1] , mdX[ID2] , mdY[ID2] , mdZ[ID2] , r,L, ux, real_time);
        double r_sqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        double f =0;

 
        //lennard jones:
       
        if (r_sqr < 1.258884)
        {
                double r8 = 1/r_sqr* 1/r_sqr; //r^{-4}
                r8 *= r8; //r^{-8}
                double r14 = r8 *r8; //r^{-16}
                r14 *= r_sqr; //r^{-14}
                f = 24 * (2 * r14 - r8);
        }
        
        //FENE:

        if (topology == 0)
        {
            if (ID2 - ID1 == 1)
            {
                f -= 30/(1 - r_sqr/2.25);
            }
            if (ID1 - ID2 == 1)
            {
                f -= 30/(1 - r_sqr/2.25);
            }
        }
        
        //FENE:

        if (topology == 1)
        {
            if (int(ID1/m) == int(ID2/m))
            {
                if( ID2 - ID1 == 1 || ID2 - ID1 == m-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }

                if( ID1 - ID2 == 1 || ID1 - ID2 == m-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }
            }   
        }
        
        //FENE:
        if (topology == 2)
        {
            if (int(ID1/m) == int(ID2/m))
            {
                if( ID2 - ID1 == 1 || ID2 - ID1 == m-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }

                if( ID1 - ID2 == 1 || ID1 - ID2 == m-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }
            }
            
            if (ID1==int(m/4) && ID2 ==m+int(3*m/4))
            {
                
                f -= 30/(1 - r_sqr/2.25);
            }
                
            if (ID2==int(m/4) && ID1 ==m+int(3*m/4))
            {
                f -= 30/(1 - r_sqr/2.25);
            }
        }

        if (topology == 3)
        {
            if (int(ID1/m) == int(ID2/m))
            {
                if( ID2 - ID1 == 1 || ID2 - ID1 == m-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }

                if( ID1 - ID2 == 1 || ID1 - ID2 == m-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }
            }
            
            // if (ID1==int(m/4) && ID2 ==m+int(3*m/4))
            // {
                
            //     f -= 30/(1 - r_sqr/2.25);
            // }
                
            // if (ID2==int(m/4) && ID1 ==m+int(3*m/4))
            // {
            //     f -= 30/(1 - r_sqr/2.25);
            // }
        }
        f/=mass;

        fx[tid] = f * r[0] ;
        fy[tid] = f * r[1] ;
        fz[tid] = f * r[2] ;
        }
        else
        {
            fx[tid] = 0;
            fy[tid] = 0;
            fz[tid] = 0;
        }
    }

}

__host__ void calc_accelaration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz,
double *Ax , double *Ay , double *Az,
double *L,int size ,int m ,int topology, double ux,double real_time, int grid_size)
{
    nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux,density, real_time , m , topology);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sum_kernel<<<grid_size,blockSize>>>(Fx ,Fy,Fz, Ax ,Ay, Az, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}



//second Kernel of velocity verelt: v += 0.5ha(old)
__global__ void velocityVerletKernel2(double *mdVx , double *mdVy , double *mdVz,
double *mdAx , double *mdAy , double *mdAz,
 double h, int Nmd)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < Nmd)
    {
        mdVx[particleID] += 0.5 * h * mdAx[particleID];
        mdVy[particleID] += 0.5 * h * mdAy[particleID];
        mdVz[particleID] += 0.5 * h * mdAz[particleID];
    }
}




//first kernel: x+= hv(half time) + 0.5hha(new) ,v += 0.5ha(new)

__global__ void velocityVerletKernel1(double *mdX, double *mdY , double *mdZ , 
double *mdVx , double *mdVy , double *mdVz,
double *mdAx , double *mdAy , double *mdAz,
 double h, int Nmd)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < Nmd)
    {
        mdVx[particleID] += 0.5 * h * mdAx[particleID];
        mdVy[particleID] += 0.5 * h * mdAy[particleID];
        mdVz[particleID] += 0.5 * h * mdAz[particleID];

        mdX[particleID] = mdX[particleID] + h * mdVx[particleID] ;
        mdY[particleID] = mdY[particleID] + h * mdVy[particleID] ;
        mdZ[particleID] = mdZ[particleID] + h * mdVz[particleID] ;


    }
}



__host__ void MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ,
    double *d_mdVx, double *d_mdVy, double *d_mdVz,
    double *d_mdAx, double *d_mdAy, double *d_mdAz,
    double *d_Fx, double *d_Fy, double *d_Fz,
    double h_md ,int Nmd, int density, double *d_L ,double ux,int grid_size ,int delta, double real_time)
{
    for (int tt = 0 ; tt < delta ; tt++)
    {

        
        velocityVerletKernel1<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz , h_md,Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        
        LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx , ux , d_L, real_time , Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        calc_accelaration(d_mdX, d_mdY , d_mdZ , d_Fx , d_Fy , d_Fz , d_mdAx , d_mdAy , d_mdAz, d_L , Nmd ,m_md ,topology, ux ,real_time, grid_size);
        
        
        velocityVerletKernel2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz, h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        real_time += h_md;


        
    }
}




