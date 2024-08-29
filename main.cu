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

// double data[N * DIMENSIONS];
#define TRIANGLE(X, Y) ( X < Y ? (Y * (Y + 1) / 2 + X) : (X * (X + 1) / 2 + Y) )
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void init_fake_data(double *data) {
  // cluster 1 of 5 points near (-1000, ..., -1000) with std 1.0
  std::normal_distribution<float> distribution(0.0, 1.0);
  std::default_random_engine generator;
  for (int i = 0; i < N / 3; i++) {
    for (int j = 0; j < DIMENSIONS; j++) {
      data[i * DIMENSIONS + j] = distribution(generator) - 1000;
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

void sample_initial_solution(double *solution, int seed) {
  std::normal_distribution<float> distribution(0.0, 1.0);
  std::default_random_engine generator;
  // set seed
  generator.seed(seed);


  for(int i = 0; i < N * DIMENSIONS_LOWER; i++) {
    solution[i] = distribution(generator) / 10000;
  }
}

double sum_arr_from_device(double* device_arr, int size) {
  double* host_arr = (double *)malloc(size * sizeof(double));
  checkCudaErrors(cudaMemcpy(host_arr, device_arr, size * sizeof(double),
                             cudaMemcpyDeviceToHost));
  double sum = 0;
  for(int i = 0; i < size; i++) {
    sum += host_arr[i];
  }
  free(host_arr);
  return sum;
}

void set_lerning_rates_device(double* d_lerning_rates, double initial_rate, int size) {
  double* host_lerning_rates = (double *)malloc(size * sizeof(double));
  for(int i = 0; i < size; i++) {
    host_lerning_rates[i] = initial_rate;
  }
  checkCudaErrors(cudaMemcpy(d_lerning_rates, host_lerning_rates, size * sizeof(double),
                             cudaMemcpyHostToDevice));
  free(host_lerning_rates);
}

void read_lines_from_file(const char* filename, double* arr, int size) {
  FILE *f = fopen(filename, "r");
  if(f == NULL) {
    std::cout << "Error opening file" << std::endl;
    exit(1);
  }
  char line[256];
  int i = 0;
  while(fgets(line, sizeof(line), f)) {
    double x;
    sscanf(line, "%lf", &x);
    // printf("%lf\n", x);
    arr[i++] = x;
    if(i == size)
      break;
  }
  fclose(f);
}

void modify_p_sym(double* p_sym_device, int n) {
  double* p_sym_host = (double *)malloc(n * (n + 1) / 2 * sizeof(double));
  // copy p_sym to host
  checkCudaErrors(cudaMemcpy(p_sym_host, p_sym_device, n * (n + 1) / 2 * sizeof(double),
                         cudaMemcpyDeviceToHost));
  double before = sum_arr_from_device(p_sym_device, n * (n + 1) / 2);
  // modify p_sym 
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < i; j++) {
      double p = p_sym_host[TRIANGLE(i, j)];
      p /= P_MULTIPLIER;
      p_sym_host[TRIANGLE(i, j)] = p;
    }
  }
  // copy back to device
  checkCudaErrors(cudaMemcpy(p_sym_device, p_sym_host, n * (n + 1) / 2 * sizeof(double),
                             cudaMemcpyHostToDevice));
  double after = sum_arr_from_device(p_sym_device, n * (n + 1) / 2);
  std::cout << "p_sym sum before: " << before << " after: " << after << std::endl;
  free(p_sym_host);
}

void calculate_distances_wrapper(double* d_data, double* distances_device, int dimensions, int n) {
  if(dimensions < DIMENSION_LIMIT_FOR_USING_TILED_KERNEL)
    calculate_distances<<<(n + 1) / 2, THREADS>>>(d_data, distances_device, dimensions, n);
  else {
    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    calculate_distances_tiled<<<blocks, block_size,
               dimensions * TILE_WIDTH * 2 * sizeof(double)>>>
               (d_data, distances_device, dimensions, n);
  }
}

int main(int argc, char **argv) {
  
  double *d_data;
  double *distances_device;
  double *distances_device2;
  double *sigmas_device;
  double *sigmas_host;
  double *denominators_device; // for calculating pji - we can calculate it in the same kernel as sigmas
  double *p_asym_device; // p_i|j
  double *p_sym_device;  // p_ij
  double *p_asym_host;
  double *p_sym_host;

  double* solution;
  double* data = (double *)malloc(N * DIMENSIONS * sizeof(double));
  checkCudaErrors(cudaMalloc(&d_data, N * DIMENSIONS * sizeof(double)));

  checkCudaErrors(cudaMalloc(&distances_device, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMemset(distances_device, -1, N * (N + 1) / 2 * sizeof(double)));

  checkCudaErrors(cudaMalloc(&distances_device2, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMemset(distances_device2, -1, N * (N + 1) / 2 * sizeof(double)));

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
  // 4 10 11 12 13 14
  

  // init data and copy to device
  // init_fake_data();
  int seed = 0;
  if(argc > 1) {
    seed = atoi(argv[1]);
  }
    
  sample_initial_solution(solution, seed);
  
                            
  // make timer
  StopWatchInterface *timer = NULL;

  // read data from file and copy to device

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  read_lines_from_file("/home/micha-nowicki/Dokumenty/tSNE1/transformed6k.csv", data, N * DIMENSIONS);
  checkCudaErrors(cudaMemcpy(d_data, data, N * DIMENSIONS * sizeof(double),
                             cudaMemcpyHostToDevice));
  sdkStopTimer(&timer);
  std::cout << "Reading data time: " << sdkGetTimerValue(&timer) << std::endl;

  // calculate distances
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  calculate_distances_wrapper(d_data, distances_device, DIMENSIONS, N);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  std::cout << "distances time: " << sdkGetTimerValue(&timer) << std::endl;


  // debug Distance
  double *distances_host = (double *)malloc(N * (N + 1) / 2 * sizeof(double));
  checkCudaErrors(cudaMemcpy(distances_host, distances_device, N * (N + 1) / 2 * sizeof(double),
                             cudaMemcpyDeviceToHost));

  double *distances_host2 = (double *)malloc(N * (N + 1) / 2 * sizeof(double));
  checkCudaErrors(cudaMemcpy(distances_host2, distances_device2, N * (N + 1) / 2 * sizeof(double),
                             cudaMemcpyDeviceToHost));
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < i; j++) {
      if(distances_host2[TRIANGLE(i, j)] - distances_host[TRIANGLE(i, j)] > 0.0001 
        || distances_host2[TRIANGLE(i, j)] - distances_host[TRIANGLE(i, j)] < -0.0001) {
        std::cout << "distances[" << i << "][" << j << "] = " << distances_host[TRIANGLE(i, j)] << " distances2[" << i << "][" << j << "] = " << distances_host2[TRIANGLE(i, j)] << std::endl;
      }
      // else
        // std::cout << "distances[" << i << "][" << j << "] = " << distances_host[TRIANGLE(i, j)] << " distances2[" << i << "][" << j << "] = " << distances_host2[TRIANGLE(i, j)] << " OK" << std::endl;
    }
  }

  // calculating sigmas
  double perplexity = 40.0;
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
  std::cout << "p_sym summed: " << sum_arr_from_device(p_sym_device, N * (N + 1) / 2) << std::endl;

  // grtadient descent
  double* d_processed_distances;   // divided by q denoinator gives q (low dim affinites)
  double* d_solution;              // low dim solution
  double* d_solution_old;          // this will become handy for momentum
  double* d_denominator_for_block; // for calculating q
  double* d_grad;                  // gradient
  double* d_lerning_rates;         // learning rates for each parameter
  double* d_delta_bar;             // exponential average of partial derivatives
  double* d_kullback_leibler;      // for calculating kullback leibler divergence - just for curiosity
  checkCudaErrors(cudaMalloc(&d_processed_distances, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_solution, N * DIMENSIONS_LOWER * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_solution_old, N * DIMENSIONS_LOWER * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_denominator_for_block, ((N + 1) / 2) * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_grad, N * DIMENSIONS_LOWER * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_lerning_rates, N * DIMENSIONS_LOWER * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_delta_bar, N * DIMENSIONS_LOWER * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_kullback_leibler, (N + 1) / 2 * sizeof(double)));

  checkCudaErrors(cudaMemset(d_processed_distances, 0, N * (N + 1) / 2 * sizeof(double)));
  checkCudaErrors(cudaMemset(d_solution, 3, N * DIMENSIONS_LOWER * sizeof(double)));

  checkCudaErrors(cudaMemcpy(d_solution, solution, N * DIMENSIONS_LOWER * sizeof(double),
                             cudaMemcpyHostToDevice));

  set_lerning_rates_device(d_lerning_rates, 1000.0, N * DIMENSIONS_LOWER);
  // print initial (random) solution to check if it was copied correctly
  // for(int i = 0; i < N ; i++) {
  //   for(int j = 0; j < DIMENSIONS_LOWER; j++) {
  //     std::cout << "solution[" << i << "][" << j << "] = " << solution[i * DIMENSIONS_LOWER + j] << std::endl;
  //   }__global__ void calculate_Kullback_Leibler(double *p, double *processed_distances, double denominator, double* partial_ans, int n);
  double alpha = 0.5; // momentum

  // for delta bar delta
  double kappa = 3.75;
  double fi = 0.1;
  double theta = 0.7;

  // reset timer
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  StopWatchInterface *timer_distances = NULL;
  StopWatchInterface *timer_process_distances = NULL;
  StopWatchInterface *timer_sum_arr = NULL;
  StopWatchInterface *timer_calculate_gradient = NULL;
  StopWatchInterface *timer_make_step = NULL;

  sdkCreateTimer(&timer_distances);
  sdkCreateTimer(&timer_process_distances);
  sdkCreateTimer(&timer_sum_arr);
  sdkCreateTimer(&timer_calculate_gradient);
  sdkCreateTimer(&timer_make_step);

  for(int i = 0; i < ITERATIONS; i++) {
    if(i == 50){
      modify_p_sym(p_sym_device, N);
    }
    
    sdkStartTimer(&timer_distances);
    calculate_distances_wrapper(d_solution, d_processed_distances, DIMENSIONS_LOWER, N);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer_distances);

    sdkStartTimer(&timer_process_distances);
    process_distances<<<(N + 1) / 2, THREADS>>>(d_processed_distances, d_denominator_for_block, N);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer_process_distances);


    sdkStartTimer(&timer_sum_arr);
    // TODO add this to second stream and run in parallel EDIT: naah
    double denominator = 2 * sum_arr_from_device(d_denominator_for_block, (N + 1) / 2); // its important to
    // multiply by 2 because we are operating on half of the matrix, we want our array to sum up to 0.5 so whole matrix sums up to 1
    // just like in p_ij
    sdkStopTimer(&timer_sum_arr);


    sdkStartTimer(&timer_calculate_gradient);
    calculate_gradient<<<N, THREADS>>>(p_sym_device, d_processed_distances, d_solution, denominator, d_grad, N);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer_calculate_gradient);

    // just for curiosity - calculate kulback leibler divergence
    // calculate_Kullback_Leibler<<<(N + 1) / 2, THREADS>>>(p_sym_device, d_processed_distances, denominator,d_kullback_leibler,  N);
    // checkCudaErrors(cudaDeviceSynchronize());
    // double kullback_leibler = 2 * sum_arr_from_device(d_kullback_leibler, (N + 1) / 2); // cuz its half of the matrix
    // std::cout << "Kullback Leibler divergence: " << kullback_leibler << std::endl;
    // std::cout << "q summed: " << sum_arr_from_device(d_processed_distances, N * (N + 1) / 2) / denominator << std::endl;
    // std::cout << "denominator: " << denominator << std::endl;
    // std::cout << "p summed: " << sum_arr_from_device(p_sym_device, N * (N + 1) / 2) << std::endl;
    // std::cout << "iteration: " << i << std::endl;

    if(i > 250)
      alpha = 0.8;
    // update solution

    sdkStartTimer(&timer_make_step);
    make_step_and_update_learning_rate<<<(N + 255) / 256, 256>>>(d_solution, d_solution_old, d_grad, d_lerning_rates, alpha,
                                                    theta, d_delta_bar, kappa, fi, DIMENSIONS_LOWER, N);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer_make_step);


  }
  sdkStopTimer(&timer);
  std::cout << "Total gradient descent time: " << sdkGetTimerValue(&timer) << std::endl;
  std::cout << "avg. distances time: " << sdkGetTimerValue(&timer_distances) / ITERATIONS << std::endl;
  std::cout << "avg. process_distances time: " << sdkGetTimerValue(&timer_process_distances) / ITERATIONS  << std::endl;
  std::cout << "avg.sum_arr time: " << sdkGetTimerValue(&timer_sum_arr) / ITERATIONS << std::endl;
  std::cout << "avg. calculate_gradient time: " << sdkGetTimerValue(&timer_calculate_gradient) / ITERATIONS << std::endl;
  std::cout << "avg. make_step time: " << sdkGetTimerValue(&timer_make_step) / ITERATIONS << std::endl;

  // print gradient
  double* grad = (double *)malloc(N * DIMENSIONS_LOWER * sizeof(double));
  checkCudaErrors(cudaMemcpy(grad, d_grad, N * DIMENSIONS_LOWER * sizeof(double),
                             cudaMemcpyDeviceToHost));

  // for(int i = 0; i < N ; i++) {
  //   for(int j = 0; j < DIMENSIONS_LOWER; j++) {
  //     std::cout << "grad[" << i << "][" << j << "] = " << grad[i * DIMENSIONS_LOWER + j] << std::endl;
  //   }
  // }

  checkCudaErrors(cudaMemcpy(solution, d_solution, N * DIMENSIONS_LOWER * sizeof(double),
                             cudaMemcpyDeviceToHost));
  // open file to save solution
  FILE *f = fopen("solution.txt", "w");
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
  
  sdkDeleteTimer(&timer);
  // finish
  exit(0);
}
