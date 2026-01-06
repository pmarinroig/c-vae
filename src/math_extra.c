#include <stdlib.h>
#include <math.h>
#include "math_extra.h"

/*
C = A * B
A = (dim1, dim2)
B = (dim2, dim3)
C = (dim1, dim3)
*/
void matmul(const float* A, const float* B, float* C, int dim1, int dim2, int dim3) {
    for (int i=0; i<dim1; ++i) {
        for (int j=0; j<dim3; ++j) {
            float sum = 0.0;
            for (int k=0; k<dim2; ++k) {
                sum += A[i + k*dim1] * B[k + j*dim2];
            }
            C[i + j*dim1] = sum;
        }
    }
}

/*
A = A + B where B is reshaped acordingly
A = (dim1, dim2)
B = (dim2)
*/
void add_mat_vec(float* A, const float* B, int dim1, int dim2) {
    for (int i=0; i<dim1; ++i) {
        for (int j=0; j<dim2; ++j) {
            A[i + j*dim1] = A[i + j*dim1] + B[j];
        }
    }
}

/*
Generate number sampled from standard normal distribution with Box-Muller transform
*/
double rand_normal() {
    double u, v, s;
    // Sample point uniformly from unit circle with rejection sampling
    do {
        u = (double)rand() / RAND_MAX * 2.0 - 1.0; // Range [-1, 1]
        v = (double)rand() / RAND_MAX * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    // Convert point (u,v) to sample from normal distribution
    double multiplier = sqrt(-2.0 * log(s) / s);
    return u * multiplier;
}

/*
Xavier init, suitable for linear.
A has length N
*/
void xavier_init(float* A, size_t N, size_t fan_in, size_t fan_out) {
    double std_dev = sqrt(2.0 / (double)(fan_in + fan_out));
    for (size_t i = 0; i < N; i++) {
        double z = rand_normal();
        A[i] = (float)(z * std_dev);
    }
}

/*
Kaiming He init, suitable for ReLUs.
A has length N
*/
void kaiming_init(float* A, size_t N, size_t fan_in) {
    double std_dev = sqrt(2.0 / (double)fan_in);
    for (size_t i = 0; i < N; i++) {
        double z = rand_normal();
        A[i] = (float)(z * std_dev);
    }
}