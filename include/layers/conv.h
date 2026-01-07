#ifndef CONV_H
#define CONV_H

#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    float* weights; // [out_channels * in_channels * k * k]
    float* biases;  // [out_channels]

    // Buffers
    float* output;
    size_t batch_size;
    size_t in_width;
    size_t in_height;
    size_t out_width;
    size_t out_height;

    // Training
    const float* in_cache;
    float* din;
    float* dweights;
    float* dbiases;

    // Adam
    float* weights_p;
    float* weights_q;
    float* biases_p;
    float* biases_q;
} ConvLayer;

ConvLayer* conv_create(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding);
void conv_init_parameters(ConvLayer* this, bool relu);
void conv_prepare_inference(ConvLayer* this, size_t batch_size, size_t in_width, size_t in_height);
void conv_prepare_training(ConvLayer* this);
void conv_zero_grad(ConvLayer* this);
void conv_forward(ConvLayer* this, const float* X);
void conv_backward(ConvLayer* this, const float* dout);
void conv_adam_step(ConvLayer* this, float lr, float b1, float b2);
void conv_free(ConvLayer* this);

#endif
