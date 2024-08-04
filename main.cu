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

int main(int argc, char **argv) {

  // call kernel
  double *dData;
  double *distances_device;
  double *sigmas_device;
  double *sigmas_host;
  checkCudaErrors(cudaMalloc(&dData, N * DIMENSIONS * sizeof(double)));
  checkCudaErrors(cudaMalloc(&distances_device, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMemset(distances_device, -1, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMalloc(&sigmas_device, N * sizeof(double)));
  sigmas_host = (double *)malloc(N * sizeof(double));
  // ^ triangle matrix of distances between points
  // 
  //   0 1 2 3 4
  // 0 0
  // 1 1 2
  // 2 3 4 5
  // 3 6 7 8 9
  
  init_fake_data();
  checkCudaErrors(cudaMemcpy(dData, data, N * DIMENSIONS * sizeof(double),
                             cudaMemcpyHostToDevice));
                            
  // measure kernel time
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  calculate_distances<<<(N + 1) / 2, THREADS>>>(dData, distances_device, N);  
  checkCudaErrors(cudaDeviceSynchronize());

  sdkStopTimer(&timer);
  std::cout << "Kernel time: " << sdkGetTimerValue(&timer) << std::endl;

  // debug Distance
  double *distances_host = (double *)malloc(N * (N + 1) / 2 * sizeof(double));
  checkCudaErrors(cudaMemcpy(distances_host, distances_device, N * (N + 1) / 2 * sizeof(double),
                             cudaMemcpyDeviceToHost));
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     if(i == 50){
  //       std::cout<< "distances for i:" << i << " j:" << j << " distance: "  << distances_host[TRIANGLE(i, j)] << std::endl;
  //   }
  //   }
  // }

  // calculating sigmas
  double perplexity = 5;
  double tolerance = 0.1;
  sdkStartTimer(&timer);
  calculate_sigmas<<<N, THREADS>>>(distances_device, sigmas_device, perplexity, tolerance, N);
  checkCudaErrors(cudaDeviceSynchronize());

  sdkStopTimer(&timer);
  std::cout << "Kernel sigma time: " << sdkGetTimerValue(&timer) << std::endl;

  checkCudaErrors(cudaMemcpy(sigmas_host, sigmas_device, N * sizeof(double),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    std::cout << "i: " << i << " sigma: " << sigmas_host[i] << std::endl;
  }
  
  sdkDeleteTimer(&timer);


  checkCudaErrors(cudaFree(dData));
  checkCudaErrors(cudaFree(distances_device));
  checkCudaErrors(cudaFree(sigmas_device));
  // free(distances_host);
  free(sigmas_host);
  
  // finish
  exit(0);
}
