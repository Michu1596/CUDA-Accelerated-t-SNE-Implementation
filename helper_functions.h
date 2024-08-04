#pragma once

static bool bDontUseGPUTiming;

__global__ void calculate_distances(double *d_data,double* distances, int n);
__global__ void calculate_sigmas(double *distances_sq, double *sigmas, double perp, double tolerance, int n);