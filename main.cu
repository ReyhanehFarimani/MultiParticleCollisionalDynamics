#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
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

int main(int argc, const char* argv[]) {
    // Check if correct number of arguments were passed in
    if (argc != 16) {
        std::cout << "Argument parsing failed!\n";
        std::string exeName = argv[0];
        std::cout << exeName << "\n";
        std::cout << "Number of given arguments: " << argc << "\n";
        return 1;
    }

    // Set simulation parameters based on command line arguments
    std::string inputfile = argv[1];  // restart file name
    std::string basename = argv[2];   // output base name
    L[0] = atof(argv[3]);             // dimension of the simulation in x direction
    L[1] = atof(argv[4]);             // dimension of the simulation in y direction
    L[2] = atof(argv[5]);             // dimension of the simulation in z direction
    density = atoi(argv[6]);          // density of the particles
    n_md = atoi(argv[7]);             // number of rings
    m_md = atof(argv[8]);             // number of monomer in each ring
    shear_rate = atof(argv[9]);       // shear rate
    h_md = atof(argv[10]);            // md time step
    h_mpcd = atof(argv[11]);          // mpcd time step
    swapsize = atoi(argv[12]);        // output interval
    simuationtime = atoi(argv[13]);   // final simulation step count
    TIME = atoi(argv[14]);            // starting time
    topology = atoi(argv[15]);        // Topology of the system for now 
                                      //  0 means polycatenane and 1 is the bonded ring

      
    double ux = shear_rate * L[2];
    int Nc = L[0] * L[1] * L[2];
    int N = density * Nc;
    int Nmd = n_md * m_md;
    int grid_size = ((N + blockSize) / blockSize);

    // Random generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, 4294967296ULL ^ time(NULL));
    curandState *devStates;
    cudaMalloc((void **)&devStates, blockSize * grid_size * sizeof(curandState));
    setup_kernel<<<grid_size, blockSize>>>(time(NULL), devStates);

    // Allocate device memory for mpcd particle:
    double *d_x, *d_vx, *d_y, *d_vy, *d_z, *d_vz;
    int *d_index;
    cudaMalloc((void**)&d_x, sizeof(double) * N);
    cudaMalloc((void**)&d_y, sizeof(double) * N);
    cudaMalloc((void**)&d_z, sizeof(double) * N);
    cudaMalloc((void**)&d_vx, sizeof(double) * N);
    cudaMalloc((void**)&d_vy, sizeof(double) * N);
    cudaMalloc((void**)&d_vz, sizeof(double) * N); 
    
    // Allocate device memory for box attributes:
    double *d_L, *d_r;
    cudaMalloc((void**)&d_L, sizeof(double) * 3);
    cudaMalloc((void**)&d_r, sizeof(double) * 3);

    // Allocate device memory for cells:
    double *d_ux, *d_uy, *d_uz;
    int *d_n, *d_m;
    cudaMalloc((void**)&d_ux, sizeof(double) * Nc);
    cudaMalloc((void**)&d_uy, sizeof(double) * Nc);
    cudaMalloc((void**)&d_uz, sizeof(double) * Nc);
    cudaMalloc((void**)&d_n, sizeof(int) * Nc);
    cudaMalloc((void**)&d_m, sizeof(int) * Nc);

    // Allocate device memory for rotating angles and matrix:
    double *d_phi, *d_theta, *d_rot;
    cudaMalloc((void**)&d_phi, sizeof(double) * Nc);
    cudaMalloc((void**)&d_theta, sizeof(double) * Nc);
    cudaMalloc((void**)&d_rot, sizeof(double) * Nc * 9);

    // Allocate device memory for cell level thermostat attributes:
    double *d_e, *d_scalefactor;
    cudaMalloc((void**)&d_e, sizeof(double) * Nc);
    cudaMalloc((void**)&d_scalefactor, sizeof(double) * Nc);

    // Allocate device memory for MD particles:
    double *d_mdX, *d_mdY, *d_mdZ, *d_mdVx, *d_mdVy, *d_mdVz, *d_mdAx, *d_mdAy, *d_mdAz;
    int *d_mdIndex;
    cudaMalloc((void**)&d_mdX, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdY, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdZ, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdVx, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdVy, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdVz, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdAx, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdAy, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdAz, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdIndex, sizeof(int) * Nmd);

    // Allocate device memory for MD forces:
    double *md_Fx_holder, *md_Fy_holder, *md_Fz_holder;
    cudaMalloc((void**)&md_Fx_holder, sizeof(double) * Nmd * Nmd);
    cudaMalloc((void**)&md_Fy_holder, sizeof(double) * Nmd * Nmd);
    cudaMalloc((void**)&md_Fz_holder, sizeof(double) * Nmd * Nmd);

    if (TIME == 0) {
        start_simulation(basename, simuationtime, swapsize, d_L, d_mdX, d_mdY, d_mdZ,
                        d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz, md_Fx_holder,
                        md_Fy_holder, md_Fz_holder, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                        gen, grid_size);
    } else {
        restarting_simulation(basename, inputfile, simuationtime, swapsize, d_L, d_mdX, d_mdY,
                            d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz,
                            md_Fx_holder, md_Fy_holder, md_Fz_holder, d_x, d_y, d_z, d_vx,
                            d_vy, d_vz, ux, N, Nmd, TIME, grid_size);
    }

    double real_time = TIME;
    int T = simuationtime / swapsize + TIME / swapsize;
    int delta = h_mpcd / h_md;

    xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY, d_mdZ, Nmd);

    for (int t = TIME/swapsize; t < T; t++) {

        for (int i = 0; i < int(swapsize/h_mpcd); i++) {

            curandGenerateUniformDouble(gen, d_phi, Nc);
            curandGenerateUniformDouble(gen, d_theta, Nc);
            curandGenerateUniformDouble(gen, d_r, 3);

            MPCD_streaming(d_x, d_y, d_z, d_vx, d_vy, d_vz, h_mpcd, N, grid_size);

            MD_streaming(d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz,
                        md_Fx_holder, md_Fy_holder, md_Fz_holder, h_md, Nmd, density, d_L, ux, grid_size,
                        delta, real_time);

            Sort_begin(d_x, d_y, d_z, d_vx, d_index, d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdIndex, ux, d_L, d_r,
                    N, Nmd, real_time, grid_size);

            MPCD_MD_collision(d_vx, d_vy, d_vz, d_index, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, d_ux,
                            d_uy, d_uz, d_e, d_scalefactor, d_n, d_m, d_rot, d_theta, d_phi,
                            N, Nmd, Nc, devStates, grid_size);

            Sort_finish(d_x, d_y, d_z, d_vx, d_index, d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdIndex, ux,
                        d_L, d_r, N, Nmd, real_time, grid_size);

            real_time += h_mpcd;
        }

        //logging:
        logging(basename + "_log.log" , (t+1)*swapsize , d_mdVx , d_mdVy , d_mdVz , d_vx, d_vy , d_vz, N , Nmd, grid_size );
        xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
        xyz_trj(basename + "_vel.xyz", d_mdVx, d_mdVy , d_mdVz, Nmd);
       
    }

    // Write restart files
    md_write_restart_file(basename, d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
    mpcd_write_restart_file(basename ,d_x , d_y , d_z , d_vx , d_vy , d_vz , N);

    // Free memory
    cudaFree(d_x); 
    cudaFree(d_y); 
    cudaFree(d_z);
    cudaFree(d_vx); 
    cudaFree(d_vy); 
    cudaFree(d_vz);
    cudaFree(d_ux); 
    cudaFree(d_uy); 
    cudaFree(d_uz);
    cudaFree(d_rot); 
    cudaFree(d_phi); 
    cudaFree(d_theta);
    cudaFree(devStates); 
    cudaFree(d_e); 
    cudaFree(d_scalefactor);

    // Free memory for MD particles
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

    // Destroy generator
    curandDestroyGenerator(gen);

    // Print termination message
    std::cout<<"The program has terminated successfully at time: "<<real_time<<std::endl;

}
