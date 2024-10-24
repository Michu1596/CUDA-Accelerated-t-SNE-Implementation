#pragma once

static bool bDontUseGPUTiming;

__global__ void calculate_distances(float *d_data,float* distances, int dim, int n);
__global__ void calculate_distances_tiled(float *d_data,float* distances, int dim, int n);
__global__ void calculate_sigmas(float *distances_sq, float *sigmas, float perp, float tolerance, float* denominators,int n);
__global__ void calculate_p_asym(float *distances, float *sigmas, float *denominators, float *p_asym, int n);
__global__ void calculate_p_sym(float *p_asym, float *p_sym, int n);
__global__ void process_distances(float *distances, float*denominator_for_block, int n);
__global__ void calculate_and_process_distances(float *d_data,float* distances, float*denominator_for_block, int dim, int n);
__global__ void calculate_gradient(float *p, float *processed_distances, float *y, float denominator, float *grad, int n); 
__global__ void calculate_Kullback_Leibler(float *p, float *processed_distances, float denominator, float* partial_ans, int n);
__global__ void make_step_and_update_learning_rate(float *y, float *old_y, float *grad, float *learning_rates, float alpha,
                                                    float theta,float *d_delta_bar, float kappa, float fi, int dim_lower, int n);