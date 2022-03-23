#inluce "md_analyser.cuh"

__host__ double momentum(int mass , double *v , int N)
{
    double p=0;
    for (int i =0 ; i<N ; ++i)
        p+= v[i];
    return(p*mass);
}

__host__ double angular_momentum(int mass, double *v1 , double *v2 , double *x1 , double *x2, int N)
{
    double l = 0;
    for (int i =0 ; i<N ; ++i)
        l+= (x1[i]*v2[i] - x2[i]*v1[i]);
    return(l*mass);
}
__host__ double harmonic_p(const double *r ,double H)
{
    double r_sqr = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    return H*r_sqr/2;
}
__host__ double FENE_p(const double *r ,double H, double L_max_sqr)
{
    double r_sqr = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    if (r_sqr> L_max_sqr)
    {
        std::cout<<"error!"<<std::endl;
        return 10000000000;
    }
    double E = log(1 - (r_sqr/L_max_sqr));
    E *= -H*L_max_sqr/2;
    return E;
}
__host__ double LJ_p(const double *r)
{
    double r_sqr = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    if (r_sqr>=1.25992105)
        return 0;
    double r6 = r_sqr* r_sqr* r_sqr;
    r6 = 1/r6;
    double r12 = r6* r6;
    return 4*(r12 - r6) +1 ;
    
}


__host__ double kinetinc(int mass ,int N , double *vx , double *vy , double *vz)
{
    double E=0;
    for (int i =0 ;i<N ; i++)
    {
        E+= vx[i]*vx[i];
        E+= vy[i]*vy[i];
        E+= vz[i]*vz[i];
    }
    E /= 2;
    E *= mass;
    return (E);
}


__host__ double potential(int Nmd,
double *mdX ,double *mdY , double *mdZ,
double *L, double ux)
{
    double E = 0;
    
    for (int i = 0 ; i<Nmd ; ++i)
    {
        for (int j =i+1 ; j<Nmd ; ++j)
        {
            double r[3];
            LeeEdwNearestImage(mdX[i], mdY[i], mdZ[i] , mdX[j] , mdY[j] , mdZ[j] , r, L, ux, h_md);
            
            E +=LJ_p(r);

            if( j - i== 1 || j - i == Nmd-1 ) 
                E+= FENE_p(r,30,2.25);
                
            
        }
    }
    return E;
}


__host__ void xyz_traj(FILE *traj , double *d_mdX, double *d_mdY , double *d_mdZ, int Nmd)
{
    double *h_mdX, *h_mdY, *h_mdZ;
    h_mdX = (double*)malloc(sizeof(double) * Nmd);  
    h_mdY = (double*)malloc(sizeof(double) * Nmd);  
    h_mdZ = (double*)malloc(sizeof(double) * Nmd);
    cudaMemcpy(h_mdX, d_mdX, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mdY, d_mdY, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mdZ, d_mdZ, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    fprintf(traj , "%i\n\n" , Nmd);
    for (int i =0 ; i< Nmd ; i++)
    {
        fprintf(traj , "C      %.8f      %.8f      %.8f\n", h_mdX[i] , h_mdY[i] , h_mdZ[i]);
    }

}
