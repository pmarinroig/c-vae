#include "layers/sigmoid.h"
#include <math.h>
#include <string.h>

SigmoidLayer* sigmoid_create(size_t input_dim) {
    SigmoidLayer* this = calloc(1, sizeof(SigmoidLayer));
    this->input_dim = input_dim;
    return this;
}

void sigmoid_prepare_inference(SigmoidLayer* this, size_t batch_size) {
    this->batch_size = batch_size;
    this->size = batch_size * this->input_dim;
    
    if (this->output) free(this->output);
    this->output = malloc(sizeof(float) * this->size);
}

void sigmoid_prepare_training(SigmoidLayer* this) {
    if (this->din) return;
    this->din = malloc(sizeof(float) * this->size);
}

void sigmoid_zero_grad(SigmoidLayer* this) {
    if (!this->din) return;
    memset(this->din, 0, sizeof(float) * this->size);
}

void sigmoid_forward(SigmoidLayer* this, const float* X) {
    for (size_t i = 0; i < this->size; i++) {
        this->output[i] = 1.0f / (1.0f + expf(-X[i]));
    }
}

void sigmoid_backward(SigmoidLayer* this, const float* dout) {
    // dL/dX = dL/dY * Y * (1 - Y)
    // Y is stored in this->output
    for (size_t i = 0; i < this->size; i++) {
        float y = this->output[i];
        this->din[i] = dout[i] * y * (1.0f - y);
    }
}

void sigmoid_free(SigmoidLayer* this) {
    if (this->output) free(this->output);
    if (this->din) free(this->din);
    free(this);
}
