#ifndef MATH_EXTRA_H
#define MATH_EXTRA_H

void matmul(const float* A, const float* B, float* C, int dim1, int dim2, int dim3);
void matmul_1t(const float* A, const float* B, float* C, int dim1, int dim2, int dim3);
void matmul_2t(const float* A, const float* B, float* C, int dim1, int dim2, int dim3);
void add_mat_vec(float* A, const float* B, int dim1, int dim2);
void sum_axis0(const float* A, float* B, int dim1, int dim2);
void adam_update(float* params, const float* grads, float* m, float* v, size_t size, float lr, float b1, float b2);
void xavier_init(float* A, size_t N, size_t fan_in, size_t fan_out);
void kaiming_init(float* A, size_t N, size_t fan_in);
float mse_loss(const float* pred, const float* target, size_t size);
void mse_loss_backward(const float* pred, const float* target, float* dloss, size_t size);

#endif