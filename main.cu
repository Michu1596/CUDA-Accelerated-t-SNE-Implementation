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

// float data[N * DIMENSIONS];
#define TRIANGLE(X, Y) ( X < Y ? (Y * (Y + 1) / 2 + X) : (X * (X + 1) / 2 + Y) )
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void sample_initial_solution(float *solution, int seed) {
  std::normal_distribution<float> distribution(0.0, 1.0);
  std::default_random_engine generator;
  // set seed
  generator.seed(seed);


  for(int i = 0; i < N * DIMENSIONS_LOWER; i++) {
    solution[i] = distribution(generator) / 10000;
  }
}

float sum_arr_from_device(float* device_arr, int size) {
  float* host_arr = (float *)malloc(size * sizeof(float));
  checkCudaErrors(cudaMemcpy(host_arr, device_arr, size * sizeof(float),
                             cudaMemcpyDeviceToHost));
  float sum = 0;
  for(int i = 0; i < size; i++) {
    sum += host_arr[i];
  }
  free(host_arr);
  return sum;
}

void set_lerning_rates_device(float* d_lerning_rates, float initial_rate, int size) {
  float* host_lerning_rates = (float *)malloc(size * sizeof(float));
  for(int i = 0; i < size; i++) {
    host_lerning_rates[i] = initial_rate;
  }
  checkCudaErrors(cudaMemcpy(d_lerning_rates, host_lerning_rates, size * sizeof(float),
                             cudaMemcpyHostToDevice));
  free(host_lerning_rates);
}

void read_lines_from_file(const char* filename, float* arr, int size) {
  FILE *f = fopen(filename, "r");
  if(f == NULL) {
    std::cout << "Error opening file" << std::endl;
    exit(1);
  }
  char line[256];
  int i = 0;
  while(fgets(line, sizeof(line), f)) {
    float x;
    sscanf(line, "%f", &x); // lf for double, f for float
    // printf("%lf\n", x);
    arr[i++] = x;
    if(i == size)
      break;
  }
  fclose(f);
}

void modify_p_sym(float* p_sym_device, int n) {
  float* p_sym_host = (float *)malloc(n * (n + 1) / 2 * sizeof(float));
  // copy p_sym to host
  checkCudaErrors(cudaMemcpy(p_sym_host, p_sym_device, n * (n + 1) / 2 * sizeof(float),
                         cudaMemcpyDeviceToHost));
  float before = sum_arr_from_device(p_sym_device, n * (n + 1) / 2);
  // modify p_sym 
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < i; j++) {
      float p = p_sym_host[TRIANGLE(i, j)];
      p /= P_MULTIPLIER;
      p_sym_host[TRIANGLE(i, j)] = p;
    }
  }
  // copy back to device
  checkCudaErrors(cudaMemcpy(p_sym_device, p_sym_host, n * (n + 1) / 2 * sizeof(float),
                             cudaMemcpyHostToDevice));
  float after = sum_arr_from_device(p_sym_device, n * (n + 1) / 2);
  std::cout << "p_sym sum before: " << before << " after: " << after << std::endl;
  free(p_sym_host);
}

void calculate_distances_wrapper(float* d_data, float* distances_device, int dimensions, int n) {
  if(dimensions < DIMENSION_LIMIT_FOR_USING_TILED_KERNEL)
    calculate_distances<<<(n + 1) / 2, THREADS>>>(d_data, distances_device, dimensions, n);
  else {
    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    calculate_distances_tiled<<<blocks, block_size,
               dimensions * TILE_WIDTH * 2 * sizeof(float)>>>
               (d_data, distances_device, dimensions, n);
  }
}

int main(int argc, char **argv) {
  
  float *d_data;
  float *distances_device;
  float *distances_device2;
  float *sigmas_device;
  float *sigmas_host;
  float *denominators_device; // for calculating pji - we can calculate it in the same kernel as sigmas
  float *p_asym_device; // p_i|j
  float *p_sym_device;  // p_ij
  float *p_asym_host;
  float *p_sym_host;

  float* solution;
  float* data = (float *)malloc(N * DIMENSIONS * sizeof(float));
  checkCudaErrors(cudaMalloc(&d_data, N * DIMENSIONS * sizeof(float)));

  checkCudaErrors(cudaMalloc(&distances_device, N * (N + 1) / 2 * sizeof(float)));
  checkCudaErrors(cudaMemset(distances_device, -1, N * (N + 1) / 2 * sizeof(float)));

  checkCudaErrors(cudaMalloc(&distances_device2, N * (N + 1) / 2 * sizeof(float)));
  checkCudaErrors(cudaMemset(distances_device2, -1, N * (N + 1) / 2 * sizeof(float)));

  checkCudaErrors(cudaMalloc(&denominators_device, N * sizeof(float)));
  checkCudaErrors(cudaMemset(denominators_device, -1, N * sizeof(float)));

  checkCudaErrors(cudaMalloc(&sigmas_device, N * sizeof(float)));
  checkCudaErrors(cudaMemset(sigmas_device, -1, N * sizeof(float)));

  checkCudaErrors(cudaMalloc(&p_asym_device, N * N * sizeof(float)));
  checkCudaErrors(cudaMemset(p_asym_device, 0, N * N * sizeof(float)));

  checkCudaErrors(cudaMalloc(&p_sym_device, N * (N + 1) / 2 * sizeof(float)));
  checkCudaErrors(cudaMemset(p_sym_device, 0, N * (N + 1) / 2 * sizeof(float)));

  solution = (float *)malloc(N * DIMENSIONS_LOWER * sizeof(float));
  p_sym_host = (float *)malloc(N * (N + 1) / 2 * sizeof(float));
  p_asym_host = (float *)malloc(N * N * sizeof(float));
  sigmas_host = (float *)malloc(N * sizeof(float));
  // ^ triangle matrix of distances between points
  // 
  //   0 1 2 3 4 - columns
  // 0 0
  // 1 1 2
  // 2 3 4 5
  // 3 6 7 8 9
  // 4 10 11 12 13 14
  // ^ rows

  // init data and copy to device
  int seed = 0;
  if(argc > 2) {
    seed = atoi(argv[2]);
  }
    
  // for calculating sigmas
  float perplexity = 40.0;
  float tolerance = 0.1;

  if(argc > 3) {
    perplexity = atof(argv[3]);
  }

  if(argc > 4) {
    tolerance = atof(argv[4]);
  }

  // default result_filename = "solution.txt"
  char result_filename[256];
  if(argc > 5) {
    strcpy(result_filename, argv[5]);
  }
  else {
    strcpy(result_filename, "solution.txt");
  }

  printf("Seed: %d\n", seed);
  printf("Perplexity: %f\n", perplexity);
  printf("Tolerance: %f\n", tolerance);

  sample_initial_solution(solution, seed);  
                            
  // read data from file and copy to device
  read_lines_from_file("/home/micha-nowicki/Dokumenty/tSNE1/transformed6k.csv", data, N * DIMENSIONS);
  checkCudaErrors(cudaMemcpy(d_data, data, N * DIMENSIONS * sizeof(float),
                             cudaMemcpyHostToDevice));

  // calculate distances
  calculate_distances_wrapper(d_data, distances_device, DIMENSIONS, N);

  
  calculate_sigmas<<<N, THREADS>>>(distances_device, sigmas_device, perplexity, tolerance, denominators_device, N);

  // calculating p_asym
  calculate_p_asym<<<N, THREADS>>>(distances_device, sigmas_device, denominators_device, p_asym_device, N);

  // now we can free distances_device and denominators_device
  checkCudaErrors(cudaFree(distances_device));
  checkCudaErrors(cudaFree(denominators_device));


  // calculating p_sym
  calculate_p_sym<<<(N + 1) / 2, THREADS>>>(p_asym_device, p_sym_device, N);

  // grtadient descent
  float* d_processed_distances;   // divided by q denoinator gives q (low dim affinites)
  float* d_solution;              // low dim solution
  float* d_solution_old;          // this will become handy for momentum
  float* d_denominator_for_block; // for calculating q
  float* d_grad;                  // gradient
  float* d_lerning_rates;         // learning rates for each parameter
  float* d_delta_bar;             // exponential average of partial derivatives
  float* d_kullback_leibler;      // for calculating kullback leibler divergence - just for curiosity
  checkCudaErrors(cudaMalloc(&d_processed_distances, N * (N + 1) / 2 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_solution, N * DIMENSIONS_LOWER * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_solution_old, N * DIMENSIONS_LOWER * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_denominator_for_block, ((N + 1) / 2) * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_grad, N * DIMENSIONS_LOWER * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_lerning_rates, N * DIMENSIONS_LOWER * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_delta_bar, N * DIMENSIONS_LOWER * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_kullback_leibler, (N + 1) / 2 * sizeof(float)));

  checkCudaErrors(cudaMemset(d_processed_distances, 0, N * (N + 1) / 2 * sizeof(float)));
  checkCudaErrors(cudaMemset(d_solution, 3, N * DIMENSIONS_LOWER * sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_solution, solution, N * DIMENSIONS_LOWER * sizeof(float),
                             cudaMemcpyHostToDevice));

  set_lerning_rates_device(d_lerning_rates, 1000.0, N * DIMENSIONS_LOWER);
  float alpha = 0.9; // momentum

  // for delta bar delta optimization technique
  float kappa = 3.75;
  float fi = 0.1;
  float theta = 0.7;


  for(int i = 0; i < ITERATIONS; i++) {
    if(i == 50){
      modify_p_sym(p_sym_device, N);
    }
    // calculate_distances_wrapper(d_solution, d_processed_distances, DIMENSIONS_LOWER, N);
    calculate_and_process_distances<<<(N + 1) / 2, THREADS>>>(d_solution, d_processed_distances,d_denominator_for_block, DIMENSIONS_LOWER, N);
    checkCudaErrors(cudaDeviceSynchronize());    
    // process_distances<<<(N + 1) / 2, THREADS>>>(d_processed_distances, d_denominator_for_block, N);
    // checkCudaErrors(cudaDeviceSynchronize());

    float denominator = 2 * sum_arr_from_device(d_denominator_for_block, (N + 1) / 2); // its important to
    // multiply by 2 because we are operating on half of the matrix, we want our array to sum up to 0.5 so whole matrix sums up to 1
    // just like in p_ij

    calculate_gradient<<<N, THREADS>>>(p_sym_device, d_processed_distances, d_solution, denominator, d_grad, N);
    checkCudaErrors(cudaDeviceSynchronize());

    if(i > 250)
      alpha = 0.8;
    // update solution

    make_step_and_update_learning_rate<<<(N + 255) / 256, 256>>>(d_solution, d_solution_old, d_grad, d_lerning_rates, alpha,
                                                    theta, d_delta_bar, kappa, fi, DIMENSIONS_LOWER, N);
    checkCudaErrors(cudaDeviceSynchronize());


  }


  checkCudaErrors(cudaMemcpy(solution, d_solution, N * DIMENSIONS_LOWER * sizeof(float),
                             cudaMemcpyDeviceToHost));
  // open file to save solution
  FILE *f = fopen(result_filename, "w");
  fprintf(f, "x, y\n");
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < DIMENSIONS_LOWER; j++) {
      fprintf(f, "%f", solution[i * DIMENSIONS_LOWER + j]);
      if(j != DIMENSIONS_LOWER - 1)
        fprintf(f, ", ");
    }
    fprintf(f, "\n");
    
  }
  // free memory
  checkCudaErrors(cudaFree(d_data));
  // checkCudaErrors(cudaFree(distances_device));
  checkCudaErrors(cudaFree(sigmas_device));
  checkCudaErrors(cudaFree(p_asym_device));
  checkCudaErrors(cudaFree(p_sym_device));
  free(p_sym_host);
  free(p_asym_host);
  free(sigmas_host);
  free(solution);
  free(data);
  // finish
  exit(0);
}
