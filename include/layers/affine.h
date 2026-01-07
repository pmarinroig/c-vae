#ifndef AFFINE_H
#define AFFINE_H

#include<stdlib.h>
#include "layer.h"

typedef struct {
    Layer layer;
    float* weights;     // (in_dim, out_dim)
    float* biases;
    size_t in_dim;
    size_t out_dim;

    // Used for returning forward pass
    float* output;
    size_t batch_size;

    // Training variables. Null if not training.
    const float* in_cache;
    float* din;         // (B, in_dim)
    float* dweights;    // (in_dim, out_dim)
    float* dbiases;     // (out_dim)

    // Adam
    float* weights_p;
    float* weights_q;
    float* biases_p;
    float* biases_q;
} AffineLayer;

AffineLayer* affine_create(size_t in_dim, size_t out_dim);
void affine_init_parameters(AffineLayer* this, bool relu);
void affine_prepare_inference(AffineLayer* this, size_t batch_size);
void affine_prepare_training(AffineLayer* this);
void affine_zero_grad(AffineLayer* this);
void affine_forward(AffineLayer* this, const float* X);
void affine_backward(AffineLayer* this, const float* dout);
void affine_adam_step(AffineLayer* this, float lr, float b1, float b2);

#endif