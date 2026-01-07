#ifndef CONV_TRANSPOSE_H
#define CONV_TRANSPOSE_H

#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t output_padding;

    float* weights; // [in_channels * out_channels * k * k]
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
} ConvTransposeLayer;

ConvTransposeLayer* conv_transpose_create(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding, size_t output_padding);
void conv_transpose_init_parameters(ConvTransposeLayer* this, bool relu);
void conv_transpose_prepare_inference(ConvTransposeLayer* this, size_t batch_size, size_t in_width, size_t in_height);
void conv_transpose_prepare_training(ConvTransposeLayer* this);
void conv_transpose_zero_grad(ConvTransposeLayer* this);
void conv_transpose_forward(ConvTransposeLayer* this, const float* X);
void conv_transpose_backward(ConvTransposeLayer* this, const float* dout);
void conv_transpose_adam_step(ConvTransposeLayer* this, float lr, float b1, float b2);
void conv_transpose_free(ConvTransposeLayer* this);

#endif
