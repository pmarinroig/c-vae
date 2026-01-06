#ifndef MATH_EXTRA_H
#define MATH_EXTRA_H

void matmul(const float* A, const float* B, float* C, int dim1, int dim2, int dim3);
void add_mat_vec(float* A, const float* B, int dim1, int dim2);
void xavier_init(float* A, size_t N, size_t fan_in, size_t fan_out);
void kaiming_init(float* A, size_t N, size_t fan_in);

#endif