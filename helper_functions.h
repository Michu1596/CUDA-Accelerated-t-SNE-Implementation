#pragma once

static bool bDontUseGPUTiming;

__global__ void calculate_distances(double *d_data,double* distances, int dim, int n);
__global__ void calculate_distances_tiled(double *d_data,double* distances, int dim, int n);
__global__ void calculate_sigmas(double *distances_sq, double *sigmas, double perp, double tolerance, double* denominators,int n);
__global__ void calculate_p_asym(double *distances, double *sigmas, double *denominators, double *p_asym, int n);
__global__ void calculate_p_sym(double *p_asym, double *p_sym, int n);
__global__ void process_distances(double *distances, double*denominator_for_block, int n);
__global__ void calculate_gradient(double *p, double *processed_distances, double *y, double denominator, double *grad, int n); 
__global__ void calculate_Kullback_Leibler(double *p, double *processed_distances, double denominator, double* partial_ans, int n);
__global__ void make_step_and_update_learning_rate(double *y, double *old_y, double *grad, double *learning_rates, double alpha,
                                                    double theta,double *d_delta_bar, double kappa, double fi, int dim_lower, int n);
__global__ void square_matrix_from_triangle(double *matrix, double *triangle, int dim, int n);