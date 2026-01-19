#include "layers/conv.h"
#include "math_extra.h"
#include <string.h>

ConvLayer* conv_create(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding) {
    ConvLayer* this = calloc(1, sizeof(ConvLayer));
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size;
    this->weights = malloc(sizeof(float) * weight_size);
    this->biases = malloc(sizeof(float) * out_channels);

    return this;
}

void conv_init_parameters(ConvLayer* this, bool relu) {
    size_t weight_size = this->out_channels * this->in_channels * this->kernel_size * this->kernel_size;
    // Fan-in: in_channels * k * k
    size_t fan_in = this->in_channels * this->kernel_size * this->kernel_size;
    // Fan-out: out_channels * k * k
    size_t fan_out = this->out_channels * this->kernel_size * this->kernel_size;

    if (relu) {
        kaiming_init(this->weights, weight_size, fan_in);
    } else {
        xavier_init(this->weights, weight_size, fan_in, fan_out);
    }
    memset(this->biases, 0, sizeof(float) * this->out_channels);
}

void conv_prepare_inference(ConvLayer* this, size_t batch_size, size_t in_width, size_t in_height) {
    this->batch_size = batch_size;
    this->in_width = in_width;
    this->in_height = in_height;

    this->out_width = (in_width + 2 * this->padding - this->kernel_size) / this->stride + 1;
    this->out_height = (in_height + 2 * this->padding - this->kernel_size) / this->stride + 1;

    size_t out_size = batch_size * this->out_channels * this->out_width * this->out_height;
    
    if (this->output) free(this->output);
    this->output = malloc(sizeof(float) * out_size);
}

void conv_prepare_training(ConvLayer* this) {
    if (this->din) return;

    size_t in_size = this->batch_size * this->in_channels * this->in_width * this->in_height;
    size_t weight_size = this->out_channels * this->in_channels * this->kernel_size * this->kernel_size;
    size_t bias_size = this->out_channels;

    this->din = malloc(sizeof(float) * in_size);
    this->dweights = malloc(sizeof(float) * weight_size);
    this->dbiases = malloc(sizeof(float) * bias_size);

    this->weights_p = calloc(weight_size, sizeof(float));
    this->weights_q = calloc(weight_size, sizeof(float));
    this->biases_p = calloc(bias_size, sizeof(float));
    this->biases_q = calloc(bias_size, sizeof(float));
}

void conv_zero_grad(ConvLayer* this) {
    if (!this->din) return;
    size_t in_size = this->batch_size * this->in_channels * this->in_width * this->in_height;
    size_t weight_size = this->out_channels * this->in_channels * this->kernel_size * this->kernel_size;
    size_t bias_size = this->out_channels;

    memset(this->din, 0, sizeof(float) * in_size);
    memset(this->dweights, 0, sizeof(float) * weight_size);
    memset(this->dbiases, 0, sizeof(float) * bias_size);
}

void conv_forward(ConvLayer* this, const float* X) {
    this->in_cache = X;
    size_t in_map_size = this->in_channels * this->in_width * this->in_height;
    size_t out_map_size = this->out_channels * this->out_width * this->out_height;
    size_t out_spatial = this->out_width * this->out_height;
    size_t col_size = (this->in_channels * this->kernel_size * this->kernel_size) * out_spatial;
    
    // Temporary buffer for im2col
    float* col = malloc(sizeof(float) * col_size);

    for (size_t b = 0; b < this->batch_size; b++) {
        const float* img_ptr = X + b * in_map_size;
        float* out_ptr = this->output + b * out_map_size;

        // 1. im2col -> col [C_in*k*k, H_out*W_out]
        im2col(img_ptr, col, this->in_channels, this->in_height, this->in_width, this->kernel_size, this->stride, this->padding);

        // 2. matmul: weights [C_out, C_in*k*k] * col [C_in*k*k, H_out*W_out] = output [C_out, H_out*W_out]
        // This matches NCHW layout for the output pointer
        matmul(this->weights, col, out_ptr, this->out_channels, this->in_channels * this->kernel_size * this->kernel_size, out_spatial);

        // 3. Add bias
        for (size_t c = 0; c < this->out_channels; c++) {
            float bias = this->biases[c];
            for (size_t i = 0; i < out_spatial; i++) {
                out_ptr[c * out_spatial + i] += bias;
            }
        }
    }
    
    free(col);
}

void conv_backward(ConvLayer* this, const float* dout) {
    size_t in_map_size = this->in_channels * this->in_width * this->in_height;
    size_t out_map_size = this->out_channels * this->out_width * this->out_height;
    size_t out_spatial = this->out_width * this->out_height;
    size_t kernel_dim = this->in_channels * this->kernel_size * this->kernel_size;
    size_t col_size = kernel_dim * out_spatial;

    float* col = malloc(sizeof(float) * col_size); // For input im2col
    float* dcol = malloc(sizeof(float) * col_size); // For dL/d(col)

    for (size_t b = 0; b < this->batch_size; b++) {
        const float* img_ptr = this->in_cache + b * in_map_size;
        const float* dout_ptr = dout + b * out_map_size;
        float* din_ptr = this->din + b * in_map_size;

        // Recompute im2col
        im2col(img_ptr, col, this->in_channels, this->in_height, this->in_width, this->kernel_size, this->stride, this->padding);

        // dL/dW = dout * col^T
        // dout: [C_out, out_spatial]
        // col: [kernel_dim, out_spatial]
        // dW: [C_out, kernel_dim] = dout * col^T
        // Use matmul_2t(A, B) -> A * B^T
        
        float* batch_dweights = malloc(sizeof(float) * this->out_channels * kernel_dim);
        matmul_2t(dout_ptr, col, batch_dweights, this->out_channels, out_spatial, kernel_dim);
        
        // Accumulate dweights
        size_t total_weights = this->out_channels * kernel_dim;
        for (size_t i = 0; i < total_weights; i++) {
            this->dweights[i] += batch_dweights[i];
        }
        free(batch_dweights);

        // dL/db = sum(dout, axis=(spatial, batch))
        // Here we sum over spatial for this batch
        for (size_t c = 0; c < this->out_channels; c++) {
            float sum = 0.0f;
            for (size_t i = 0; i < out_spatial; i++) {
                sum += dout_ptr[c * out_spatial + i];
            }
            this->dbiases[c] += sum;
        }

        // dL/d(col) = W^T * dout
        // W: [C_out, kernel_dim]
        // dout: [C_out, out_spatial]
        // d(col): [kernel_dim, out_spatial]
        // W^T * dout -> matmul_1t(W, dout)
        matmul_1t(this->weights, dout_ptr, dcol, kernel_dim, this->out_channels, out_spatial);

        // dL/dX = col2im(dL/d(col))
        col2im(dcol, din_ptr, this->in_channels, this->in_height, this->in_width, this->kernel_size, this->stride, this->padding);
    }

    free(col);
    free(dcol);
}

void conv_adam_step(ConvLayer* this, float lr, float b1, float b2, int t) {
    size_t weight_size = this->out_channels * this->in_channels * this->kernel_size * this->kernel_size;
    adam_update(this->weights, this->dweights, this->weights_p, this->weights_q, weight_size, lr, b1, b2, t);
    adam_update(this->biases, this->dbiases, this->biases_p, this->biases_q, this->out_channels, lr, b1, b2, t);
}

void conv_free(ConvLayer* this) {
    if (this->weights) free(this->weights);
    if (this->biases) free(this->biases);
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
