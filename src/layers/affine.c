#include <stdbool.h>
#include <string.h>
#include "layers/affine.h"
#include "math_extra.h"

AffineLayer* affine_create(size_t in_dim, size_t out_dim) {
    AffineLayer* this = malloc(sizeof(AffineLayer));
    this->weights = malloc(sizeof(float) * in_dim * out_dim);
    this->biases = malloc(sizeof(float) * out_dim);
    this->in_dim = in_dim;
    this->out_dim = out_dim;
    this->output = 0;
    this->batch_size = 0;
    return this;
}

void affine_init_parameters(AffineLayer* this, bool relu) {
    if(relu) {
        kaiming_init(this->weights, this->in_dim * this->out_dim, this->in_dim);
    } else {
        xavier_init(this->weights, this->in_dim * this->out_dim, this->in_dim, this->out_dim);
    }
    memset(this->biases, 0, sizeof(float) * this->out_dim);
}

void affine_prepare(AffineLayer* this, size_t batch_size) {
    if(this->output) {
        free(this->output);
    }
    this->output = malloc(sizeof(float) * batch_size * this->out_dim);
    this->batch_size = batch_size;
}

void affine_forward(AffineLayer* this, const float* X, size_t batch_size) {
    matmul(X, this->weights, this->output, batch_size, this->in_dim, this->out_dim);
    this->batch_size = batch_size;
    add_mat_vec(this->output, this->biases, this->batch_size, this->out_dim);
    this->in_cache = X;
}
