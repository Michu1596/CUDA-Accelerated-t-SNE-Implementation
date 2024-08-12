/* 
żeby intlisense od vsighta działało potrzben sa te pliki co sa w .vscode
*/
// CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_cuda.h>  // helper functions for CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <cuda.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <random>

#include "helper_functions.h"
#include "consts.h"

double data[N * DIMENSIONS];
#define TRIANGLE(X, Y) ( X < Y ? (Y * (Y + 1) / 2 + X) : (X * (X + 1) / 2 + Y) )
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void init_fake_data(){
  // cluster 1 of 5 points near (0, ..., 0) with std 1.0
  std::normal_distribution<float> distribution(0.0, 1.0);
  std::default_random_engine generator;
  for (int i = 0; i < N / 3; i++) {
    for (int j = 0; j < DIMENSIONS; j++) {
      data[i * DIMENSIONS + j] = distribution(generator);
    }
  }

  // cluster 2 of 5 points near (1000, ..., 1000) with std 0.01
  for (int i = N / 3; i < (2 * N) / 3; i++) {
    for (int j = 0; j < DIMENSIONS; j++) {
      data[i * DIMENSIONS + j] = distribution(generator)/100 + 1000;
    }
  }

  // cluster 3 of 6 points near (20000, ..., 20000) with std 500
  for (int i = (2 * N) / 3; i < N; i++) {
    for (int j = 0; j < DIMENSIONS; j++) {
      data[i * DIMENSIONS + j] = distribution(generator)*500 + 20000;
    }
  }
}

void sample_initial_solution(double *solution) {
  std::normal_distribution<float> distribution(0.0, 0.0001);
  std::default_random_engine generator;

  for(int i = 0; i < N * DIMENSIONS_LOWER; i++) {
    solution[i] = distribution(generator);
  }
}

int main(int argc, char **argv) {
  double *dData;
  double *distances_device;
  double *sigmas_device;
  double *sigmas_host;
  double *denominators_device; // for calculating pji - we can calculate it in the same kernel as sigmas
  double *p_asym_device; // p_i|j
  double *p_sym_device;  // p_ij
  double *p_asym_host;
  double *p_sym_host;

  double* solution;
  checkCudaErrors(cudaMalloc(&dData, N * DIMENSIONS * sizeof(double)));

  checkCudaErrors(cudaMalloc(&distances_device, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMemset(distances_device, -1, N * (N + 1) / 2 * sizeof(double)));

  checkCudaErrors(cudaMalloc(&denominators_device, N * sizeof(double)));
  checkCudaErrors(cudaMemset(denominators_device, -1, N * sizeof(double)));

  checkCudaErrors(cudaMalloc(&sigmas_device, N * sizeof(double)));
  checkCudaErrors(cudaMemset(sigmas_device, -1, N * sizeof(double)));

  checkCudaErrors(cudaMalloc(&p_asym_device, N * N * sizeof(double)));
  checkCudaErrors(cudaMemset(p_asym_device, 0, N * N * sizeof(double)));

  checkCudaErrors(cudaMalloc(&p_sym_device, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMemset(p_sym_device, 0, N * (N + 1) / 2 * sizeof(double)));

  solution = (double *)malloc(N * DIMENSIONS_LOWER * sizeof(double));
  p_sym_host = (double *)malloc(N * (N + 1) / 2 * sizeof(double));
  p_asym_host = (double *)malloc(N * N * sizeof(double));
  sigmas_host = (double *)malloc(N * sizeof(double));
  // ^ triangle matrix of distances between points
  // 
  //   0 1 2 3 4
  // 0 0
  // 1 1 2
  // 2 3 4 5
  // 3 6 7 8 9
  

  // init data and copy to device
  init_fake_data();
  checkCudaErrors(cudaMemcpy(dData, data, N * DIMENSIONS * sizeof(double),
                             cudaMemcpyHostToDevice));
                            
  // make timer
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  // calculate distances
  sdkStartTimer(&timer);
  calculate_distances<<<(N + 1) / 2, THREADS>>>(dData, distances_device, N);  
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  std::cout << "Kernel time: " << sdkGetTimerValue(&timer) << std::endl;

  // debug Distance
  double *distances_host = (double *)malloc(N * (N + 1) / 2 * sizeof(double));
  checkCudaErrors(cudaMemcpy(distances_host, distances_device, N * (N + 1) / 2 * sizeof(double),
                             cudaMemcpyDeviceToHost));

  // calculating sigmas
  double perplexity = 5;
  double tolerance = 0.1;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  calculate_sigmas<<<N, THREADS>>>(distances_device, sigmas_device, perplexity, tolerance, denominators_device, N);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  std::cout << "Kernel sigma time: " << sdkGetTimerValue(&timer) << std::endl;

  // calculating p_asym
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  calculate_p_asym<<<N, THREADS>>>(distances_device, sigmas_device, denominators_device, p_asym_device, N);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  std::cout << "Kernel p_asym time: " << sdkGetTimerValue(&timer) << std::endl;

  // now we can free distances_device and denominators_device
  checkCudaErrors(cudaFree(distances_device));
  checkCudaErrors(cudaFree(denominators_device));


  // calculating p_sym
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  calculate_p_sym<<<(N + 1) / 2, THREADS>>>(p_asym_device, p_sym_device, N);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  std::cout << "Kernel p_sym time: " << sdkGetTimerValue(&timer) << std::endl;
  checkCudaErrors(cudaMemcpy(p_sym_host, p_sym_device, N * (N + 1) / 2 * sizeof(double),
                             cudaMemcpyDeviceToHost)); // 


  // grtadient descent
  for(int i = 0; i < 1; i++) {
    
  }

  // free memory
  checkCudaErrors(cudaFree(dData));
  // checkCudaErrors(cudaFree(distances_device));
  checkCudaErrors(cudaFree(sigmas_device));
  checkCudaErrors(cudaFree(p_asym_device));
  checkCudaErrors(cudaFree(p_sym_device));
  free(p_sym_host);
  free(p_asym_host);
  free(sigmas_host);
  free(solution);
  
  sdkDeleteTimer(&timer);
  // finish
  exit(0);
}
