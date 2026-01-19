#include <stdbool.h>
#include <string.h>
#include "layers/affine.h"
#include "math_extra.h"

AffineLayer* affine_create(const size_t in_dim, const size_t out_dim) {
    AffineLayer* this = calloc(1, sizeof(AffineLayer));
    this->weights = malloc(sizeof(float) * in_dim * out_dim);
    this->biases = malloc(sizeof(float) * out_dim);
    this->in_dim = in_dim;
    this->out_dim = out_dim;
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

// Must run once if using the layer for inference
void affine_prepare_inference(AffineLayer* this, size_t batch_size) {
    if(this->output) {
        free(this->output);
    }
    this->output = malloc(sizeof(float) * batch_size * this->out_dim);
    this->batch_size = batch_size;
}

// Must run once if training is desired
void affine_prepare_training(AffineLayer* this) {
    if (this->din) return;
    this->din = malloc(sizeof(float) * this->batch_size * this->in_dim);
    this->dweights = malloc(sizeof(float) * this->in_dim * this->out_dim);
    this->dbiases = malloc(sizeof(float) * this->out_dim);
    this->weights_p = calloc(this->in_dim * this->out_dim, sizeof(float));
    this->weights_q = calloc(this->in_dim * this->out_dim, sizeof(float));
    this->biases_p = calloc(this->out_dim, sizeof(float));
    this->biases_q = calloc(this->out_dim, sizeof(float));
}

void affine_zero_grad(AffineLayer* this) {
    // If we aren't in training mode (pointers are null), do nothing
    if (!this->din) return;

    memset(this->din, 0, sizeof(float) * this->batch_size * this->in_dim);
    memset(this->dweights, 0, sizeof(float) * this->in_dim * this->out_dim);
    memset(this->dbiases, 0, sizeof(float) * this->out_dim);
}

void affine_forward(AffineLayer* this, const float* X) {
    matmul(X, this->weights, this->output, this->batch_size, this->in_dim, this->out_dim);
    add_mat_vec(this->output, this->biases, this->batch_size, this->out_dim);
    this->in_cache = X;
}

// dout has dimension (B, out_dim)
// weights has dimension (in_dim, out_dim)
// din has dimensinon (B, in_dim)
// dweights has dimension (in_dim, out_dim)
// dbias has dimension (d_out)
void affine_backward(AffineLayer* this, const float* dout) {
    // dL/dX = dL/dY * W^T
    matmul_2t(dout, this->weights, this->din, this->batch_size, this->out_dim, this->in_dim);
    
    // dL/dW = X^T * dL/dY
    matmul_1t(this->in_cache, dout, this->dweights, this->in_dim, this->batch_size, this->out_dim);
    
    // dL/db = sum(dL/dY, axis=0)
    sum_axis0(dout, this->dbiases, this->batch_size, this->out_dim);
}

void affine_adam_step(AffineLayer* this, float lr, float b1, float b2, int t) {
    adam_update(this->weights, this->dweights, this->weights_p, this->weights_q, this->in_dim * this->out_dim, lr, b1, b2, t);
    adam_update(this->biases, this->dbiases, this->biases_p, this->biases_q, this->out_dim, lr, b1, b2, t);
}

void affine_free(AffineLayer* this) {
    free(this->weights);
    free(this->biases);
    if (this->output) free(this->output);
    if (this->din) free(this->din);
    if (this->dweights) free(this->dweights);
    if (this->dbiases) free(this->dbiases);
    if (this->weights_p) free(this->weights_p);
    if (this->weights_q) free(this->weights_q);
    if (this->biases_p) free(this->biases_p);
    if (this->biases_q) free(this->biases_q);
    free(this);
}