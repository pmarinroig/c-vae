#ifndef AFFINE_H
#define AFFINE_H

#include<stdlib.h>
#include "layer.h"

typedef struct {
    Layer layer;
    float* weights;
    float* biases;
    const size_t in_dim;
    const size_t out_dim;

    // Used for returning forward pass
    float* output;
    size_t batch_size;

    // Training variables. Null if not training.
    const float* in_cache;
    float* din;
    float* dweights;
    float* dbiases;

    // Adam
    float* weights_p;
    float* weights_q;
    float* biases_p;
    float* biases_q;
} AffineLayer;

AffineLayer* affine_create(size_t in_dim, size_t out_dim);
void affine_init_parameters(AffineLayer* this, bool relu);
void affine_prepare(AffineLayer* this, size_t batch_size);
void affine_forward(AffineLayer* this, const float* X, size_t batch_size);

#endif