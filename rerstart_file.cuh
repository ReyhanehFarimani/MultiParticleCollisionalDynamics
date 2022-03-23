// reading a text file
__host__ void read (std::ifstream &myfile, double *array, int size) {
  std::string line;
  if (myfile.is_open())
  {
    std::cout << "reading restart file"<<std::endl;
    int i = 0;
    while ( getline (myfile,line) )
    {
       array[i] = std::stod(line);
       i +=1;
    }
    
    if (i != size) 
    {
        std::cout<< "Error:: The operation remained incompelete as the file is incompelete!\n";
    }
    myfile.close();
  }

  else std::cout << "Unable to open restart file"<<std::endl; 

}

__host__ void write (std::ofstream &myfile, double *array, int size) {

  if (myfile.is_open())
  {
    std::cout << "Writing restart file"<<std::endl; 
    for(int i = 0 ; i<size-1; ++i)
        myfile <<array[i]<<std::endl;
    myfile <<array[size-1];
    myfile.close();
  }

  else std::cout << "Unable to open restart file"<<std::endl; 

}

__host__ void mpcd_read_restart_file(std::string name ,double *x, double *y, double *z , double *vx , double *vy , double *vz, int N)
{
    std::ifstream file_x (name + "_x_mpcd.txt");
    std::ifstream file_y (name + "_y_mpcd.txt");
    std::ifstream file_z (name + "_z_mpcd.txt");
    std::ifstream file_vx (name + "_vx_mpcd.txt");
    std::ifstream file_vy (name + "_vy_mpcd.txt");
    std::ifstream file_vz (name + "_vz_mpcd.txt");
    double *tmp;
    tmp = (double*)malloc(sizeof(double) * N); 
    read(file_x , tmp, N);
    cudaMemcpy(x ,tmp, N*sizeof(double), cudaMemcpyHostToDevice);  
    read(file_y , tmp, N);
    cudaMemcpy(y ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_z , tmp, N);
    cudaMemcpy(z ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_vx , tmp, N);
    cudaMemcpy(vx ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_vy , tmp, N);
    cudaMemcpy(vy ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_vz , tmp, N);
    cudaMemcpy(vz ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    free(tmp);

}

__host__ void md_read_restart_file(std::string name ,double *x, double *y, double *z , double *vx , double *vy , double *vz, int N)
{
  std::ifstream file_x (name + "_x_md.txt");
  std::ifstream file_y (name + "_y_md.txt");
  std::ifstream file_z (name + "_z_md.txt");
  std::ifstream file_vx (name + "_vx_md.txt");
  std::ifstream file_vy (name + "_vy_md.txt");
  std::ifstream file_vz (name + "_vz_md.txt");
    double *tmp;
    tmp = (double*)malloc(sizeof(double) * N); 
    read(file_x , tmp, N);
    cudaMemcpy(x ,tmp, N*sizeof(double), cudaMemcpyHostToDevice);  
    read(file_y , tmp, N);
    cudaMemcpy(y ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_z , tmp, N);
    cudaMemcpy(z ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_vx , tmp, N);
    cudaMemcpy(vx ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_vy , tmp, N);
    cudaMemcpy(vy ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    read(file_vz , tmp, N);
    cudaMemcpy(vz ,tmp, N*sizeof(double), cudaMemcpyHostToDevice); 
    free(tmp);

}

__host__ void mpcd_write_restart_file(std::string name , double *x, double *y, double *z , double *vx , double *vy , double *vz, int N)
{
  std::ofstream file_x (name + "_x_mpcd.txt");
  std::ofstream file_y (name + "_y_mpcd.txt");
  std::ofstream file_z (name + "_z_mpcd.txt");
  std::ofstream file_vx (name + "_vx_mpcd.txt");
  std::ofstream file_vy (name + "_vy_mpcd.txt");
  std::ofstream file_vz (name + "_vz_mpcd.txt");
    double *tmp;
    tmp = (double*)malloc(sizeof(double) * N);
    cudaMemcpy(tmp , x , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_x , tmp, N);
    cudaMemcpy(tmp , y , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_y , tmp, N);
    cudaMemcpy(tmp , z , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_z , tmp, N);
    cudaMemcpy(tmp , vx , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_vx , tmp, N);
    cudaMemcpy(tmp , vy , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_vy , tmp, N);
    cudaMemcpy(tmp , vz , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_vz , tmp, N);
    free(tmp);

}

__host__ void md_write_restart_file(std::string name ,double *x, double *y, double *z , double *vx , double *vy , double *vz, int N)
{
  std::ofstream file_x (name + "_x_md.txt");
  std::ofstream file_y (name + "_y_md.txt");
  std::ofstream file_z (name + "_z_md.txt");
  std::ofstream file_vx (name + "_vx_md.txt");
  std::ofstream file_vy (name + "_vy_md.txt");
  std::ofstream file_vz (name + "_vz_md.txt");
    double *tmp;
    tmp = (double*)malloc(sizeof(double) * N);
    cudaMemcpy(tmp , x , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_x , tmp, N);
    cudaMemcpy(tmp , y , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_y , tmp, N);
    cudaMemcpy(tmp , z , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_z , tmp, N);
    cudaMemcpy(tmp , vx , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_vx , tmp, N);
    cudaMemcpy(tmp , vy , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_vy , tmp, N);
    cudaMemcpy(tmp , vz , N*sizeof(double), cudaMemcpyDeviceToHost);   
    write(file_vz , tmp, N);
    free(tmp);

}


