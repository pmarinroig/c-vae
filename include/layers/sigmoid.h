#ifndef SIGMOID_H
#define SIGMOID_H

#include <stdlib.h>

typedef struct {
    size_t input_dim; // elements per batch item
    size_t batch_size;
    size_t size; // batch_size * input_dim

    float* output;
    float* din;
} SigmoidLayer;

SigmoidLayer* sigmoid_create(size_t input_dim);
void sigmoid_prepare_inference(SigmoidLayer* this, size_t batch_size);
void sigmoid_prepare_training(SigmoidLayer* this);
void sigmoid_zero_grad(SigmoidLayer* this);
void sigmoid_forward(SigmoidLayer* this, const float* X);
void sigmoid_backward(SigmoidLayer* this, const float* dout);
void sigmoid_free(SigmoidLayer* this);

#endif
