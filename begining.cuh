__global__ void kerenlInit(double *r, double *v, double *L,double px , double py, double pz, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        r[3 * tid + 0] *= L[0];
        r[3 * tid + 1] *= L[1];
        r[3 * tid + 2] *= L[2];
        r[3 * tid + 0] -= L[0]/2;
        r[3 * tid + 1] -= L[1]/2;
        r[3 * tid + 2] -= L[2]/2;
        v[3 * tid + 0] -= px;
        v[3 * tid + 1] -= py;
        v[3 * tid + 2] -= pz;
    }

}
__host__ void mpcd_init(curandGenerator_t gen,
double *d_r,
double *d_v,
int grid_size,
int N)
{

        //initialisation postions:
        curandGenerateUniformDouble(gen, d_r, 3 * N);
        
        //initialisation velocity:
        // For changing tempreture this part of the code should change
        curandGenerateNormalDouble(gen, d_v, 3 * N, 0, 1);

}
__host__ void start_simulation(std::string file_name, int simulationtime , int swapsize , double *d_L,
double *d_mdX , double *d_mdY , double *d_mdZ,
double *d_mdVx , double *d_mdVy , double *d_mdVz,
double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *d_Fx_holder , double *d_Fy_holder, double *d_Fz_holder,
double *d_r_mpcd
double *d_v_mpcd,
curandGenerator_t gen, int grid_size)
{
    std::string log_name = file_name + "_log.log";
    std::string trj_name = file_name + "_traj.xyz";
    std::ofstream log (log_name);
    std::ofstream trj (trj_name);
    double ux =shear_rate * L[2];
    int Nc = L[0]*L[1]*L[2];
    int N =density* Nc;
    int Nmd = n_md * m_md;
    printf( "***WELCOME TO MPCD CUDA CODE!***\nBy: Reyhaneh A. Farimani,
        reyhaneh.afghahi.farimani@univie.ac.at . \n
            This code comes with a python code to analyse the results***");
    printf("\ninput system:\nensemble:NVT, 
        thermostat= cell_level_Maxwell_Boltzaman_thermostat, 
            Lx=%i,Ly=%i,Lz=%i,shear_rate=%f,density=%i\n", 
                int(L[0]), int(L[1]),int(L[2]), shear_rate, density);
    if (ux != 0)
        printf( "SHEAR_FLOW is produced using Lees_Edwards Periodic Boundry Condition:
            shear direction:x , gradiant direction:z , vorticity direction: y\n");
    if (topology == 0)
    printf(" A linear polymer with %i
         monomers is embeded in the MPCD fluid.\n", n_md * m_md);
    if (topology == 1)
        printf("A poly[%i]catenane with %i
         monomer in each ring is embeded in the MPCD fluid.\n" , n_md , m_md);
    if (topology==2)
    printf("A linked[%i]ring with %i
         monomer in each ring is embeded in the MPCD fluid.\n
            Warning: the code currently support only linked[2]rings.\n" , n_md , m_md);
    printf("simulation time = %i, measurments accur every %i step.\n", simuationtime, swapsize);
    

    log<<"***WELCOME TO MPCD CUDA CODE!***\nBy: Reyhaneh A. Farimani,
    reyhaneh.afghahi.farimani@univie.ac.at . \n
        This code comes with a python code to analyse the results***";
    log<< "\ninput system:\nensemble:NVT, 
    thermostat= cell_level_Maxwell_Boltzaman_thermostat, 
        Lx="<<int(L[0])<<",Ly="<<int(L[1])<<",Lz="<<int(L[2])<<",shear_rate
            = "<<shear_rate<<",density="<<density<<std::endl;
    if (topology==1)
        log<<"A poly["<<n_md<<"]catenane with "<<m_md<<"
            monomer in each ring is embeded in the MPCD fluid.\n";
    if (topology==2)
        log<<"A linked["<<n_md<<"]ring with "<<m_md<<"
            monomer in each ring is embeded in the MPCD fluid.\n";
    if (ux != 0)
        log<< "SHEAR_FLOW is produced using Lees_Edwards Periodic Boundry Condition:
            shear direction:x , gradiant direction:z , vorticity direction: y\n";
    log<<"simulation time ="<<simulationtime<<", measurments accur every "<<swapsize<<" step.\n" ;



    //help variable:
    double *d_tmp;
    double px,py,pz;
    cudaMalloc((void**)&d_tmp, sizeof(double)*grid_size);
    mpcd_init(gen, d_x, d_y, d_z, d_vx, d_vy, d_vz, grid_size, N);
    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vx, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&px, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vy, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&py, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vz, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&pz, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_L, &L, 3*sizeof(double) , cudaMemcpyHostToDevice);
    kerenlInit<<<grid_size,blockSize>>>(d_x,d_y,d_z,d_vx, d_vy, d_vz,d_L,px/N,py/N,pz/N,N);

    double pos[3] ={0,0,0};
    //initMD:
    initMD(d_mdX, d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , d_Fx_holder , d_Fy_holder , d_Fz_holder , d_L , ux ,pos , n_md , m_md , topology , density);
    calc_accelaration(d_mdX , d_mdY, d_mdZ , d_Fx_holder , d_Fy_holder , d_Fz_holder , d_mdAx , d_mdAy , d_mdAz ,d_L , Nmd , n_md ,topology ,  ux ,h_md, grid_size);
    cudaFree(d_tmp);


}




__host__ void restarting_simulation(std::string file_name,std::string filename2, int simulationtime , int swapsize , double *d_L,
double *d_mdX , double *d_mdY , double *d_mdZ,
double *d_mdVx , double *d_mdVy , double *d_mdVz,
double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *d_Fx_holder , double *d_Fy_holder, double *d_Fz_holder,
double *d_x , double *d_y , double *d_z ,
double *d_vx , double *d_vy , double *d_vz, double ux,
int N, int Nmd, int last_step, int grid_size)
{

    cudaMemcpy(d_L, &L, 3*sizeof(double) , cudaMemcpyHostToDevice);
    std::ofstream log (file_name + "_log.log", std::ios_base::app); 
    log<<"[INFO]Restarting your simulation from step "<<last_step<<":"<<std::endl<<"reading MPCD particle data:"<<std::endl;
    std::cout<<"[INFO]Restarting your simulation from step "<<last_step<<":"<<std::endl<<"reading MPCD particle data:"<<std::endl;
    mpcd_read_restart_file(filename2 , d_x , d_y , d_z , d_vx , d_vy , d_vz , N);
    std::cout<<"[INFO]MPCD data entered into device memory succesfully!"<<std::endl<<"Reading MD particle data:"<<std::endl;
    log<<"[INFO]MPCD data entered into device memory succesfully!"<<std::endl<<"Reading MD particle data:"<<std::endl;
    md_read_restart_file(filename2 , d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
    std::cout<<"[INFO]MD data entered into device memory succesfully!"<<std::endl;
    log<<"[INFO]MD data entered into device memory succesfully!"<<std::endl;
    reset_vector_to_zero<<<grid_size,blockSize>>>(d_mdAx, d_mdAy, d_mdAz, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    calc_accelaration(d_mdX , d_mdY, d_mdZ , d_Fx_holder , d_Fy_holder , d_Fz_holder , d_mdAx , d_mdAy , d_mdAz ,d_L , Nmd ,m_md , topology, ux ,h_md, grid_size);
    
}