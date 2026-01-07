#include "layers/relu.h"
#include <string.h>

ReluLayer* relu_create(size_t input_dim) {
    ReluLayer* this = malloc(sizeof(ReluLayer));
    this->input_dim = input_dim;
    this->output = NULL;
    this->din = NULL;
    return this;
}

void relu_prepare_inference(ReluLayer* this, size_t batch_size) {
    this->batch_size = batch_size;
    this->size = batch_size * this->input_dim;
    
    if (this->output) free(this->output);
    this->output = malloc(sizeof(float) * this->size);
}

void relu_prepare_training(ReluLayer* this) {
    if (this->din) return;
    this->din = malloc(sizeof(float) * this->size);
}

void relu_zero_grad(ReluLayer* this) {
    if (!this->din) return;
    memset(this->din, 0, sizeof(float) * this->size);
}

void relu_forward(ReluLayer* this, const float* X) {
    this->input_cache = X;
    for (size_t i = 0; i < this->size; i++) {
        this->output[i] = X[i] > 0 ? X[i] : 0.0f;
    }
}

void relu_backward(ReluLayer* this, const float* dout) {
    for (size_t i = 0; i < this->size; i++) {
        this->din[i] = (this->input_cache[i] > 0) ? dout[i] : 0.0f;
    }
}