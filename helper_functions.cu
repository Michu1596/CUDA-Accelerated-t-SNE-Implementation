#include <cuda_runtime.h>

// includes
#include <helper_cuda.h>  // helper functions for CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <cuda.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <stdio.h>

#include "helper_functions.h"
#include "consts.h"
#define TRIANGLE(X, Y) ( X < Y ? (Y * (Y + 1) / 2 + X) : (X * (X + 1) / 2 + Y) )
//   0 1 2 3 4
// 0 0
// 1 1 2
// 2 3 4 5
// 3 6 7 8 9

__device__ double l2_dist_sq(double *a, double *b, int n) {
  double sum = 0;
  int index;
  for (int i = 0; i < n; i++) {
    index = i;
    sum += (a[index] - b[index]) * (a[index] - b[index]); // TODO try to diverge memory access like a[(i+c)%n] edit: tried and no difference
  }
  return sum;
}


// each block is responsible a column x and N - x so each block calculates N + 1 distances
__global__ void calculate_distances(double *d_data,double* distances, int n) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x;
  // calculate distance to each data point with lower index

  int triangle_index = 0;

  // column myPoint
  int myPoint = blockIdx.x;
  int otherPoint = myPoint - 1 - threadIdx.x; // this version with divergent memory access is 3 x faster (wow)
  while(otherPoint >= 0 && myPoint < n){
    triangle_index = TRIANGLE(myPoint, otherPoint);
    distances[triangle_index] = l2_dist_sq(d_data + myPoint * DIMENSIONS, d_data + otherPoint * DIMENSIONS, DIMENSIONS);
    otherPoint -= stride;
  }

  // column N - myPoint
  myPoint = n - myPoint - 1;
  otherPoint = myPoint - 1 - threadIdx.x;
  while(otherPoint >= 0 && myPoint < n){
    triangle_index = TRIANGLE(myPoint, otherPoint);
    distances[triangle_index] = l2_dist_sq(d_data + myPoint * DIMENSIONS, d_data + otherPoint * DIMENSIONS, DIMENSIONS);
    otherPoint -= stride;
  }

}

// each block is responsible for a single value of sigma for p_{j|blockId} for all j
__global__ void calculate_sigmas(double *distances_sq, double *sigmas, double perp, double tolerance, int n, double *shannon_entropies, double *perplexities, double *sigmas_out) 
{
  int blockId = blockIdx.x;
  int stride = blockDim.x;
  __shared__ double sigma;
  __shared__ double shared_denominator[THREADS];
  __shared__ double sum_of_numerators_logs[THREADS];
  __shared__ bool done;
  int max_iter = 10;

  //only for the first thread in the block
  if(threadIdx.x == 0){
    done = false;
    sigma = 1; // initial guess
  }
  double sigma_upper_bound = sigma;
  double sigma_lower_bound = sigma;
  bool lower_bound_found = false;
  bool upper_bound_found = false;

  __syncthreads(); // BUG FIXED HERE
  while (!done)
  {  
    __syncthreads();
    double my_denominator = 0;
    double my_sum_of_numerators_logs = 0;
    int i = threadIdx.x;
    double temp = 0;
    double temp_exp = 0;
    while(i < n){
      if(blockIdx.x == 50){
        printf("thread: %d in while\n",threadIdx.x);
      }
      if(i != blockId){
        int index = TRIANGLE(blockId, i);
        temp = -(distances_sq[index] / (2 * sigma * sigma));
        temp_exp = exp(temp);
        if(blockIdx.x == 50){
          printf("thread: %d \n index: %d\n distances_sq[index]: %f\n temp: %f\n temp_exp: %f\n",threadIdx.x, index, distances_sq[index], temp, temp_exp);
        }
        // compiler would probably optimize this to a single exp call 
        my_denominator += temp_exp;
        my_sum_of_numerators_logs += temp_exp * temp; 
      }
      i += stride;
    }

    if(blockIdx.x == 50){
      printf("threadId: %d\n my_denominator: %f\n my_sum_of_numerators_logs: %f\n", threadIdx.x, my_denominator, my_sum_of_numerators_logs);
    }
    shared_denominator[threadIdx.x] = my_denominator;
    sum_of_numerators_logs[threadIdx.x] = my_sum_of_numerators_logs;

    if(std::isnan(shared_denominator[threadIdx.x])){
      printf("Before reduction\n");
      printf("BlockId: %d, threadId: %d\n", blockId, threadIdx.x);
      printf("shared_denominator[%d]: %f\n", threadIdx.x, shared_denominator[threadIdx.x]);
      printf("sum_of_numerators_logs[%d]: %f\n", threadIdx.x, sum_of_numerators_logs[threadIdx.x]);
      printf("temp: %f\n", temp);
      printf("temp_exp: %f\n", temp_exp);
      printf("index: %d\n", TRIANGLE(blockId, i));
    }

    int limit = THREADS / 2;
    __syncthreads();

    while ( limit > 0)
    {
      if(threadIdx.x < limit){
        if(threadIdx.x == 0 && blockIdx.x == 50){
          for(int i = 0; i < limit * 2; i++){
            printf("shared_denominator[%d]: %f\n", i, shared_denominator[i]);
          }
          printf("\n");
        }
        shared_denominator[threadIdx.x] += shared_denominator[threadIdx.x + limit];
        sum_of_numerators_logs[threadIdx.x] += sum_of_numerators_logs[threadIdx.x + limit];
        // assert(shared_denominator[threadIdx.x] >= 0);
      }
      limit /= 2;
      __syncthreads();
    }

    if(threadIdx.x == 0 and false){
      printf("After reduction\n BlockId: %d\n shared_denominator[0]: %f\n", blockId, shared_denominator[0]);
    }

    __syncthreads();

    if(threadIdx.x == 0){
      
      double shannon_entropy;
      if(shared_denominator[0] != 0){
       shannon_entropy = - (sum_of_numerators_logs[0] / shared_denominator[0]) + log(shared_denominator[0]);
      } else {
        shannon_entropy = 0;
      }

      double perplexity = exp(shannon_entropy * log(2.0));
      double diff = perplexity - perp;
      double diff_abs = diff > 0 ? diff : -diff;

      // DEBUG
      if(blockId == 50){
        printf("BlockId: %d\n sigma: %e\n perplexity: %f\n shannon_entropy: %f\n sum_of_numerators_logs[0]: %f\n shared_denominator[0]: %f\n", 
                blockId, sigma, perplexity, shannon_entropy, sum_of_numerators_logs[0], shared_denominator[0]);
      }

      if(diff_abs < tolerance 
        // || max_iter-- == 0
        ){
        sigmas[blockId] = sigma;
        done = true; // this will break the while loop
      } else {
        if(lower_bound_found && upper_bound_found){
          if(diff > 0){
            sigma_upper_bound = sigma;
          } else {
            sigma_lower_bound = sigma;
          }
          sigma = (sigma_upper_bound + sigma_lower_bound) / 2;
        }
        else if(diff > 0){
          sigma_upper_bound = sigma;
          upper_bound_found = true;
          sigma /= 2;
        } else {
          sigma_lower_bound = sigma;
          lower_bound_found = true;
          sigma *= 2;
        }

      }
      
    }
    // __syncthreads();

  }
}