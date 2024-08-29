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
#define TRIANGLE(X, Y) ( X < Y ? ( (Y) * ( (Y)  + 1) / 2 + (X) ) : ( (X) * ( (X) + 1) / 2 + (Y) ) ) 
// they must be in prenthesis because if X = var1 + var2 * var3, then the macro would expand to X < var1 + var2 * var3 < Y

//   0 1 2 3 4
// 0 0
// 1 1 2
// 2 3 4 5
// 3 6 7 8 9
// 4 10 11 12 13 14
// 5 15 16 17 18 19 20

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
__global__ void calculate_distances(double *d_data,double* distances, int dim, int n) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x;
  // calculate distance to each data point with lower index

  int triangle_index = 0;

  // column myPoint
  int myPoint = blockIdx.x;
  int otherPoint = myPoint - 1 - threadIdx.x; // this version with divergent memory access is 3 x faster (wow)
  while(otherPoint >= 0 && myPoint < n){
    triangle_index = TRIANGLE(myPoint, otherPoint);
    double result = l2_dist_sq(d_data + myPoint * dim, d_data + otherPoint * dim, dim);
    distances[triangle_index] = result;
    otherPoint -= stride;
  }

  // column N - myPoint
  myPoint = n - myPoint - 1;
  otherPoint = myPoint - 1 - threadIdx.x;
  while(otherPoint >= 0 && myPoint < n){
    triangle_index = TRIANGLE(myPoint, otherPoint);
    distances[triangle_index] = l2_dist_sq(d_data + myPoint * dim, d_data + otherPoint * dim, dim);
    otherPoint -= stride;
  }

}

__global__ void calculate_distances_tiled(double *d_data,double* distances, int dim, int n){
  extern __shared__ double s[];
  double *chunk_x = s;
  double *chunk_y = s + dim * TILE_WIDTH;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int stride = blockDim.x * blockDim.y;
  int in_block_linear_index = tx + ty * blockDim.x;

  int global_x = bx * blockDim.x + tx;
  int global_y = by * blockDim.y + ty;

  // not all blocks are needed
  if(bx > by){
    return; 
  }

  // colaborative loading
  for(int i = in_block_linear_index; i < dim * TILE_WIDTH; i += stride){
    //  chunk x
    int index_x =(bx * blockDim.x * dim) + i; // index in global memory
    if(index_x < n * dim)
      chunk_x[i] = d_data[index_x];

    // chunk y
    int index_y = (by * blockDim.y * dim) + i; // index in global memory
    if(index_y < n * dim)
      chunk_y[i] = d_data[index_y];
  }


  __syncthreads();

  // thread divergence saddly
  if(bx == by && tx >= ty){
    return;
  }

  if(global_x < n && global_y < n){
    // calculate distance
    double distance = l2_dist_sq(chunk_x + tx * dim, chunk_y + ty * dim, dim);
    int triangle_index = TRIANGLE(bx * blockDim.x + tx, by * blockDim.y + ty);

    // write to global memory
    distances[triangle_index] = distance;
  }
}

// each block is responsible for a single value of sigma for p_{j|blockId} for all j
__global__ void calculate_sigmas(double *distances_sq, double *sigmas, double perp, double tolerance, double* dominators, int n) 
{
  int blockId = blockIdx.x;
  int stride = blockDim.x;
  __shared__ double sigma;
  __shared__ double shared_denominator[THREADS];
  __shared__ double sum_of_numerators_logs[THREADS];
  __shared__ bool done;

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
      if(i != blockId){
        int index = TRIANGLE(blockId, i);
        temp = -(distances_sq[index] / (2 * sigma * sigma));
        temp_exp = exp(temp);
        // compiler would probably optimize this to a single exp call 
        my_denominator += temp_exp;
        my_sum_of_numerators_logs += temp_exp * temp; 
      }
      i += stride;
    }

    shared_denominator[threadIdx.x] = my_denominator;
    sum_of_numerators_logs[threadIdx.x] = my_sum_of_numerators_logs;

    int limit = THREADS / 2;
    __syncthreads();

    while ( limit > 0)
    {
      if(threadIdx.x < limit){
        shared_denominator[threadIdx.x] += shared_denominator[threadIdx.x + limit];
        sum_of_numerators_logs[threadIdx.x] += sum_of_numerators_logs[threadIdx.x + limit];
        // assert(shared_denominator[threadIdx.x] >= 0);
      }
      limit /= 2;
      __syncthreads();
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

      if(diff_abs < tolerance ){
        sigmas[blockId] = sigma;
        dominators[blockId] = shared_denominator[0];
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

  }
}

__global__ void calculate_p_asym(double *distances, double *sigmas, double *denominators, double *p_asym, int n){
  int i = blockIdx.x;
  int stride = blockDim.x;

  double denominator = denominators[i];
  double sigma = sigmas[i];
  int j = threadIdx.x;


  while (j < n){
    if(i != j){
      int triangle_index = TRIANGLE(i, j);
      p_asym[i * n + j] = exp(-distances[triangle_index] / (2 * sigma * sigma)) / denominator;
    }
    j += stride;
  }
  
}

__global__ void calculate_p_sym(double *p_asym, double *p_sym, int n){
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x;
  // calculate distance to each data point with lower index

  int triangle_index = 0;

  // column myPoint
  int i = blockIdx.x;
  int j = i - 1 - threadIdx.x; // this version with divergent memory access is 3 x faster (wow)
  while(j >= 0 && i < n){
    triangle_index = TRIANGLE(i, j);
    double p = ((p_asym[i * n + j] + p_asym[j * n + i]) / (2 * n));
    p *= P_MULTIPLIER;
    
    p_sym[triangle_index] = p;
    j -= stride;
  }

  // column N - myPoint
  i = n - i - 1;
  j = i - 1 - threadIdx.x;
  // yes this is the same code as above but I don't want to make a function for this
  while(j >= 0 && i < n){
    triangle_index = TRIANGLE(i, j);
    double p = ((p_asym[i * n + j] + p_asym[j * n + i]) / (2 * n));
    p *= P_MULTIPLIER;

    p_sym[triangle_index] = p;
    j -= stride;
  }
}

// two times slower than calculate_distances
__global__ void process_distances(double *distances, double*denominator_for_block, int n){
  int stride = blockDim.x;
  int triangle_index = 0;
  __shared__ double shared_denominator[THREADS];
  shared_denominator[threadIdx.x] = 0;

  // inverting distances
  // column i
  int i = blockIdx.x;
  int j = i - 1 - threadIdx.x; // this version with divergent memory access is 3 x faster (wow)
  while(j >= 0 && i < n){
    triangle_index = TRIANGLE(i, j);
    
    distances[triangle_index] = 1 / (1 + distances[triangle_index]); // this processing is very expensive it takes about half of the time
    shared_denominator[threadIdx.x] += distances[triangle_index];

    j -= stride;
  }

  // column N - i
  i = n - i - 1;
  j = i - 1 - threadIdx.x;
  // yes this is the same code as above but I don't want to make a function for this
  while(j >= 0 && i < n){
    triangle_index = TRIANGLE(i, j);
    
    distances[triangle_index] = 1 / (1 + distances[triangle_index]);
    shared_denominator[threadIdx.x] += distances[triangle_index];
    
    j -= stride;
  }
  
  // calculate denominator 
  int limit = THREADS / 2;
   __syncthreads();

    while ( limit > 0)
    {
      if(threadIdx.x < limit){
        shared_denominator[threadIdx.x] += shared_denominator[threadIdx.x + limit];
      }
      limit /= 2;
      __syncthreads();
    }

    __syncthreads();
  
  // save denominator
  if(threadIdx.x == 0){
    denominator_for_block[blockIdx.x] = shared_denominator[0];
  }

}


__global__ void calculate_gradient(double *p, double *processed_distances, double *y, double denominator, double *grad, int n){
  int i = blockIdx.x;
  int stride = blockDim.x;

  __shared__ double shared_grad[THREADS * DIMENSIONS_LOWER];

  // each thread zeroes its part of the shared memory so there is no need for syncthreads
  for(int i = 0; i < DIMENSIONS_LOWER; i++){
    shared_grad[threadIdx.x * DIMENSIONS_LOWER + i] = 0;
  }

  int j = threadIdx.x;
  
  while(j < n){
    if(i != j){
      int triangle_index = TRIANGLE(i, j);
      double q = processed_distances[triangle_index] / denominator;

      for(int k = 0; k < DIMENSIONS_LOWER; k++){
        shared_grad[threadIdx.x * DIMENSIONS_LOWER + k] += 
                4 
                * (p[triangle_index] - q) 
                * (y[i * DIMENSIONS_LOWER + k] - y[j * DIMENSIONS_LOWER + k])
                * processed_distances[triangle_index];
      }
    }
    j += stride;
  }

  int limit = THREADS / 2;
   __syncthreads();

    while ( limit > 0)
    {
      if(threadIdx.x < limit){
        for(int i = 0; i < DIMENSIONS_LOWER; i++){
           shared_grad[threadIdx.x * DIMENSIONS_LOWER + i] += shared_grad[(threadIdx.x + limit) * DIMENSIONS_LOWER + i];
        }
      }
      limit /= 2;
      __syncthreads();
    }

    __syncthreads();

    if(threadIdx.x == 0){
      for(int i = 0; i < DIMENSIONS_LOWER; i++){
        grad[blockIdx.x * DIMENSIONS_LOWER + i] = shared_grad[i];
      }
    }

}

__global__ void calculate_Kullback_Leibler(double *p, double *processed_distances, double denominator, double* partial_ans, int n){
  int stride = blockDim.x;
  int triangle_index = 0;
  __shared__ double shared_kullback[THREADS];
  shared_kullback[threadIdx.x] = 0;

  // column i
  int i = blockIdx.x;
  int j = i - 1 - threadIdx.x; 
  while(j >= 0 && i < n){
    triangle_index = TRIANGLE(i, j);
    
    double q = processed_distances[triangle_index] / denominator;
    if(p[triangle_index] != 0)
      shared_kullback[threadIdx.x] += p[triangle_index] * log(p[triangle_index] / q);

    j -= stride;
  }

  // column N - i
  i = n - i - 1;
  j = i - 1 - threadIdx.x;
  // yes this is the same code as above but I don't want to make a function for this
  while(j >= 0 && i < n){
    triangle_index = TRIANGLE(i, j);
    
    double q = processed_distances[triangle_index] / denominator;
    if(p[triangle_index] != 0)
      shared_kullback[threadIdx.x] += p[triangle_index] * log(p[triangle_index] / q);
    
    j -= stride;
  }
  

  int limit = THREADS / 2;
   __syncthreads();

    while ( limit > 0)
    {
      if(threadIdx.x < limit){
        shared_kullback[threadIdx.x] += shared_kullback[threadIdx.x + limit];
      }
      limit /= 2;
      __syncthreads();
    }

    __syncthreads();
  
  if(threadIdx.x == 0){
    partial_ans[blockIdx.x] = shared_kullback[0];
  }

}

__global__ void make_step_and_update_learning_rate(double *y, double *old_y, double *grad, double *learning_rates, double alpha,
                                                    double theta, double *d_delta_bar,double kappa, double fi, int dim_lower, int n)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;

  if(j < n){
    for(int k = 0; k < dim_lower; k++){
      // index of the element in the y array
      int index = j * dim_lower + k;

      // update y
      double momentum = alpha * (y[index] - old_y[index]);
      old_y[index] = y[index]; // update old_y
      y[index] = y[index] - learning_rates[j] * grad[index] + momentum; // TODO add noise

      // update learning rate
      if(grad[index] * d_delta_bar[index] > 0){
        learning_rates[index] = learning_rates[index] + kappa < MAX_LEARNING_RATE ? learning_rates[index] + kappa : MAX_LEARNING_RATE;
      }
      else{
        learning_rates[index] = learning_rates[index] * fi; 
      }

      // update average gradient
      d_delta_bar[index] = (1 - theta) * grad[index] + theta * d_delta_bar[index];
    }
  }
  // if(j == 309){
  //   printf("y[0] = %f\n", y[0]);
  //   printf("learning_rates[0] = %f\n", learning_rates[0]);
  //   printf("d_delta_bar[0] = %f\n", d_delta_bar[0]);
  //   printf("grad[0] = %f\n", grad[0]);
  // }
}

__global__ void square_matrix_from_triangle(double *matrix, double *triangle, int n){

}