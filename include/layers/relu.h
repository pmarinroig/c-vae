#ifndef RELU_H
#define RELU_H

#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    float* output;
    size_t batch_size;
    size_t input_dim;
    size_t size; // Total number of elements (batch_size * input_dim)
    
    // For backward pass
    const float* input_cache;
    float* din; // Gradient w.r.t input
} ReluLayer;

ReluLayer* relu_create(size_t input_dim);
void relu_prepare_inference(ReluLayer* this, size_t batch_size);
void relu_prepare_training(ReluLayer* this);
void relu_zero_grad(ReluLayer* this);
void relu_forward(ReluLayer* this, const float* X);
void relu_backward(ReluLayer* this, const float* dout);
void relu_free(ReluLayer* this);

#endif