#include <stdlib.h>
#include <math.h>
#include "math_extra.h"

/*
C = A * B
A = (dim1, dim2)
B = (dim2, dim3)
C = (dim1, dim3)
Matrices are stored in row-major order.
*/
void matmul(const float* A, const float* B, float* C, int dim1, int dim2, int dim3) {
    for (int i=0; i<dim1; ++i) {
        for (int j=0; j<dim3; ++j) {
            float sum = 0.0;
            for (int k=0; k<dim2; ++k) {
                sum += A[i*dim2 + k] * B[k*dim3 + j];
            }
            C[i*dim3 + j] = sum;
        }
    }
}

/*
C = A^T * B
A = (dim2, dim1)
B = (dim2, dim3)
C = (dim1, dim3)
*/
void matmul_1t(const float* A, const float* B, float* C, int dim1, int dim2, int dim3) {
    for (int i=0; i<dim1; ++i) {
        for (int j=0; j<dim3; ++j) {
            float sum = 0.0;
            for (int k=0; k<dim2; ++k) {
                sum += A[k*dim1 + i] * B[k*dim3 + j];
            }
            C[i*dim3 + j] = sum;
        }
    }
}

/*
C = A * B^T
A = (dim1, dim2)
B = (dim3, dim2)
C = (dim1, dim3)
*/
void matmul_2t(const float* A, const float* B, float* C, int dim1, int dim2, int dim3) {
    for (int i=0; i<dim1; ++i) {
        for (int j=0; j<dim3; ++j) {
            float sum = 0.0;
            for (int k=0; k<dim2; ++k) {
                sum += A[i*dim2 + k] * B[j*dim2 + k];
            }
            C[i*dim3 + j] = sum;
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
            A[i*dim2 + j] = A[i*dim2 + j] + B[j];
        }
    }
}

/*
B = sum(A, axis=0)
A = (dim1, dim2)
B = (dim2)
*/
void sum_axis0(const float* A, float* B, int dim1, int dim2) {
    for (int j=0; j<dim2; ++j) {
        float sum = 0.0;
        for (int i=0; i<dim1; ++i) {
            sum += A[i*dim2 + j];
        }
        B[j] = sum;
    }
}

/*
Adam update step for a parameter vector.
params: parameter vector to update
grads: gradient vector
m: first moment vector
v: second moment vector
size: size of the vectors
lr: learning rate
b1: beta1 (momentum decay)
b2: beta2 (velocity decay)
*/
void adam_update(float* params, const float* grads, float* m, float* v, size_t size, float lr, float b1, float b2) {
    const float epsilon = 1e-8;
    for (size_t i = 0; i < size; ++i) {
        // Update biased first moment estimate
        m[i] = b1 * m[i] + (1.0f - b1) * grads[i];
        
        // Update biased second raw moment estimate
        v[i] = b2 * v[i] + (1.0f - b2) * grads[i] * grads[i];
        
        // Compute bias-corrected first moment estimate
        // Note: Assuming fully converged or ignoring bias correction for simplicity as 't' is not provided.
        // For a rigorous implementation, 't' (timestep) is needed: m_hat = m / (1 - b1^t), v_hat = v / (1 - b2^t)
        // Here we proceed with the simplified version often used in these contexts:
        
        params[i] -= lr * m[i] / (sqrtf(v[i]) + epsilon);
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
Xavier init, suitable for zero-centered activations.
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

float mse_loss(const float* pred, const float* target, size_t size) {
    float loss = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return 0.5f * loss / size;
}

void mse_loss_backward(const float* pred, const float* target, float* dloss, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dloss[i] = (pred[i] - target[i]) / size;
    }
}