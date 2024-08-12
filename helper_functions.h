#pragma once

static bool bDontUseGPUTiming;

__global__ void calculate_distances(double *d_data,double* distances, int n);
__global__ void calculate_sigmas(double *distances_sq, double *sigmas, double perp, double tolerance, double* denominators,int n);
__global__ void calculate_p_asym(double *distances, double *sigmas, double *denominators, double *p_asym, int n);
__global__ void calculate_p_sym(double *p_asym, double *p_sym, int n);