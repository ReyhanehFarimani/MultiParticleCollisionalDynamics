#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <exception>
#include <unistd.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "mpcd_macro.cuh"
#include "LEBC.cuh"
#include "reduction_sum.cuh"
#include "thermostat.cuh"
#include "streaming.cuh"
#include "collision.cuh"
#include "gallileain_inv.cuh"
#include "rerstart_file.cuh"
#include "gpu_md.cuh"
#include "md_analyser.cuh"
#include "begining.cuh"
#include "logging.cuh"

int main(int argc, const char* argv[])
{
    //Check for number of parsed argument:
    if( argc !=16 )
    {
        std::cout<<"Argument parsing failed!\n";
        std::string exeName = argv[0];
        std::cout<<exeName<<"\n";
        std::cout<<"Number of given arguments: "<<argc<<"\n";
        return 1;
    }

    // Setting the parsed argument:
    std::string inputfile= argv[1];         // Restart file name(The one reading from it!)
                                            // If the simulation start with t=0, put some 
                                            // dummy argument.
    std::string basename = argv[2];         // Output base name
    L[0] = atof(argv[3]);                   // Dimension of the simulation in x direction
    L[1] = atof(argv[4]);                   // Dimension of the simulation in y direction
    L[2] = atof(argv[5]);                   // Dimension of the simulation in z direction
    density = atoi(argv[6]);                // Density of the particles
    n_md = atoi(argv[7]);                   // Number of rings
    m_md = atof(argv[8]);                   // Number of monomer in each ring
    shear_rate = atof(argv[9]);             // Shear rate
    h_md = atof(argv[10]);                  // Md time step
    h_mpcd = atof(argv[11]);                // Mpcd time step
    swapsize = atoi(argv[12]);              // Output interval
    simuationtime = atoi(argv[13]);         // Final simulation step count
    TIME = atoi(argv[14]);                  // Starting time 
    topology = atoi(argv[15]);              // System topology 1 is a poly[n]catenane
                                            // 2 is the bonded ring.

    // Setting some constarint based on parsed argument  
    double ux =shear_rate * L[2];
    int Nc = L[0] * L[1] * L[2];            // Number of cells
    int N =density * Nc;                    // Number of MPCD particles
    int Nmd = n_md * m_md;                  // Number of MD particles

    // Setting the number of grid for parallel simulation,
    // It can be optimised based on GPU attributes, I did not care much about it!
    int grid_size = ((N + blockSize) / blockSize);
    
     // Random generator
     curandGenerator_t gen;
     curandCreateGenerator(&gen, 
         CURAND_RNG_PSEUDO_DEFAULT);
     // Setting seed for the simulation
     /* !NEVER EVER CHANGE THIS PART! */
     curandSetPseudoRandomGeneratorSeed(gen, 
         4294967296ULL^time(NULL));
     curandState *devStates;
     cudaMalloc((void **)&devStates,
         blockSize * grid_size *sizeof(curandState));
     setup_kernel<<<grid_size, blockSize>>>(time(NULL), devStates);
    

    /*** Allocate device memory for mpcd particle ***/
    // position and velocity of MPCD particles:
    double *d_r_mpcd;
    cudaMalloc((void**)&d_r_mpcd, sizeof(double) * N * 3);   

    double *d_v_mpcd;
    cudaMalloc((void**)&d_v_mpcd, sizeof(double) * N * 3);

    // The mpcd index is used for sorting the particles,
    // into cell, more on collision module.
    int *d_index;
    cudaMalloc((void**)&d_index, sizeof(int) * N);
    
    /*** Allocate device memory for box attributes ***/
    // d_L is a copy of box dimention in the GPU.
    // d_r is a ranfom vector for grid shifting.
    double *d_L, *d_r;   
    cudaMalloc((void**)&d_L, sizeof(double) *3);
    cudaMalloc((void**)&d_r, sizeof(double) *3); 
    
    // Allocate device memory for cells, 
    // d_u is for the mean momentum of cells
    double *d_v_cell;
    cudaMalloc((void**)&d_v_cell, sizeof(double) * Nc * 3);
    
    // d_n is for saving number of MPCD particle in a cell
    // (important for the thermostat)
    // d_m is for saving the whole mass stored in each cell
    // (important for computing the mean momentum)
    int  *d_n, *d_m;
    cudaMalloc((void**)&d_n, sizeof(int) * Nc);     
    cudaMalloc((void**)&d_m, sizeof(int) * Nc);

    // The random angles attribated for each cell for Rotation step: 
    double *d_phi , *d_theta,*d_rot;
    cudaMalloc((void**)&d_phi, sizeof(double) * Nc);    
    cudaMalloc((void**)&d_theta, sizeof(double) * Nc);  

    // The rotation matrix of the cell:
    cudaMalloc((void**)&d_rot, sizeof(double) * Nc * 9);

    // Allocate device memory for cell level thermostat atributes:
    double* d_e, *d_scalefactor;
    cudaMalloc((void**)&d_e, sizeof(double) * Nc);             //kinetic energy of the cell particles.
    cudaMalloc((void**)&d_scalefactor, sizeof(double) * Nc);    // scale factor to set the velocities distibuation
                                                                // to a desired gamma distribuation

    /*** Allocate device memory for md particle ***/
    /// postions, velocity, and acceleration:
    double *d_mdX, *d_mdY, *d_mdZ;
    cudaMalloc((void**)&d_mdX, sizeof(double) * Nmd);    
    cudaMalloc((void**)&d_mdY, sizeof(double) * Nmd);    
    cudaMalloc((void**)&d_mdZ, sizeof(double) * Nmd);
    double *d_mdVx, *d_mdVy, *d_mdVz;
    cudaMalloc((void**)&d_mdVx, sizeof(double) * Nmd);   
    cudaMalloc((void**)&d_mdVy, sizeof(double) * Nmd);   
    cudaMalloc((void**)&d_mdVz, sizeof(double) * Nmd);
    double *d_mdAx, *d_mdAy, *d_mdAz;
    cudaMalloc((void**)&d_mdAx, sizeof(double) * Nmd);   
    cudaMalloc((void**)&d_mdAy, sizeof(double) * Nmd);   
    cudaMalloc((void**)&d_mdAz, sizeof(double) * Nmd);
    // This index will be used for sorting the MD particle in to the cells:
    int *d_mdIndex;
    cudaMalloc((void**)&d_mdIndex, sizeof(int) * Nmd);

    // This attribute, is for matrix of polymer interaction in each direction
    double *md_Fx_holder , *md_Fy_holder , *md_Fz_holder;
    cudaMalloc((void**)&md_Fx_holder, sizeof(double) * Nmd * Nmd);    
    cudaMalloc((void**)&md_Fy_holder, sizeof(double) * Nmd * Nmd);    
    cudaMalloc((void**)&md_Fz_holder, sizeof(double) * Nmd * Nmd);
    // These attributes can be changed and removed.
    // I know it is the worst way to compute force between particles,
    // But it is bugless and it does not affect my running speed much!
    // If you ever wanted to simulate a system with more MD particles,
    // You should modify this part!
    

    /* Simulation stars in this part! */
    if (TIME ==0)
    {
        start_simulation(basename, simuationtime, swapsize, d_L, d_mdX, d_mdY, d_mdZ,
                         d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz,
                         md_Fx_holder, md_Fy_holder, md_Fz_holder,
                         d_r_mpcd, d_v_mpcd, gen, grid_size);
    }
    // else 
    // {
    //     restarting_simulation(basename, inputfile, simuationtime, swapsize,
    //                          d_L, d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz,
    //                          d_mdAx, d_mdAy, d_mdAz, md_Fx_holder, md_Fy_holder, md_Fz_holder,
    //                          d_r_mpcd, d_v_mpcd, ux, N, Nmd, TIME, grid_size);
    // }

    /* Setting time for the simulation! */
    double real_time = TIME;                        // It is imprtant for us because of Lees Edwards PBC
    int T =simuationtime/swapsize +TIME/swapsize;   // Computing time for the loop based on logging frequency
    int delta = h_mpcd / h_md;                      // MD step calcilation based on MPCD time loop.  
    

    // Loop based on sampling frequncy"
    for (int t = TIME/swapsize; t<T; t++)
    {
        // Loop for calculation:
        for (int i =0; i<int(swapsize/h_mpcd); i++)
        {
            curandGenerateUniformDouble(gen, d_phi, Nc);
            curandGenerateUniformDouble(gen, d_theta, Nc);
            curandGenerateUniformDouble(gen, d_r, 3);

            // MPCD_streaming(d_x, d_y, d_z, d_vx, d_vy, d_vz, h_mpcd, N, grid_size);            

            // MD_streaming(d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz,
            //              d_mdAx , d_mdAy , d_mdAz ,md_Fx_holder, md_Fy_holder, md_Fz_holder,
            //              h_md , Nmd , density , d_L , ux , grid_size, delta,real_time);
            // // xyz_trj(basename + "_force.xyz", d_mdAx, d_mdAy, d_mdAz, Nmd);
            // Sort_begin(d_x, d_y, d_z, d_vx, d_index, d_mdX, d_mdY, d_mdZ,
            //             d_mdVx, d_mdIndex, ux, d_L, d_r, N, Nmd, real_time, grid_size);

            // MPCD_MD_collision(d_vx, d_vy, d_vz, d_index, d_mdVx, d_mdVy, d_mdVz,
            //                  d_mdIndex, d_ux, d_uy, d_uz, d_e, d_scalefactor, d_n, d_m,
            //                  d_rot, d_theta, d_phi, N, Nmd, Nc, devStates, grid_size);
            
            // Sort_finish(d_x, d_y, d_z,d_vx, d_index , 
            //              d_mdX, d_mdY, d_mdZ ,d_mdVx, d_mdIndex, ux, 
            //              d_L, d_r, N, Nmd, real_time, grid_size);
            
            real_time += h_mpcd;
                 

        }
        //logging:
        logging(basename + "_log.log", real_time, d_mdVx, d_mdVy, d_mdVz,
                 d_vx, d_vy, d_vz, N, Nmd, grid_size);
        xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY, d_mdZ, Nmd);
        xyz_trj(basename + "_vel.xyz", d_mdVx, d_mdVy, d_mdVz, Nmd);
        xyz_trj(basename + "_force.xyz", d_mdAx, d_mdAy, d_mdAz, Nmd);
       
    }


    // // End of simualtion:
    // md_write_restart_file(basename, d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
    // mpcd_write_restart_file(basename ,d_x , d_y , d_z , d_vx , d_vy , d_vz , N);

    // Free memory of the MPCD particles and cells:
    cudaFree(d_r_mpcd);
    cudaFree(d_v_mpcd); 
    cudaFree(d_v_cell); 
    cudaFree(d_rot); 
    cudaFree(d_phi); 
    cudaFree(d_theta);
    cudaFree(devStates); 
    cudaFree(d_e); 
    cudaFree(d_scalefactor);
    // Free memory of the MD particles:
    cudaFree(d_mdX);    
    cudaFree(d_mdY);    
    cudaFree(d_mdZ);
    cudaFree(d_mdVx);   
    cudaFree(d_mdVy);   
    cudaFree(d_mdVz);
    cudaFree(d_mdAx);   
    cudaFree(d_mdAy);   
    cudaFree(d_mdAz);
    cudaFree(md_Fx_holder); 
    cudaFree(md_Fy_holder); 
    cudaFree(md_Fz_holder);
    curandDestroyGenerator(gen);

    std::cout<<"The program has terminated succesffuly at time:"<<real_time<<std::endl;
}
