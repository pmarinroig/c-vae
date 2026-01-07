#include "layers/conv_transpose.h"
#include "math_extra.h"
#include <string.h>

ConvTransposeLayer* conv_transpose_create(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding, size_t output_padding) {
    ConvTransposeLayer* this = calloc(1, sizeof(ConvTransposeLayer));
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;
    this->output_padding = output_padding;

    size_t weight_size = in_channels * out_channels * kernel_size * kernel_size;
    this->weights = malloc(sizeof(float) * weight_size);
    this->biases = malloc(sizeof(float) * out_channels);

    return this;
}

void conv_transpose_init_parameters(ConvTransposeLayer* this, bool relu) {
    size_t weight_size = this->in_channels * this->out_channels * this->kernel_size * this->kernel_size;
    // Fan-in: in_channels * k * k (Actually for ConvTranspose, the forward pass behaves like backward Conv)
    // But conceptually, fan-in is number of inputs.
    size_t fan_in = this->in_channels * this->kernel_size * this->kernel_size;
    size_t fan_out = this->out_channels * this->kernel_size * this->kernel_size;

    if (relu) {
        kaiming_init(this->weights, weight_size, fan_in);
    } else {
        xavier_init(this->weights, weight_size, fan_in, fan_out);
    }
    memset(this->biases, 0, sizeof(float) * this->out_channels);
}

void conv_transpose_prepare_inference(ConvTransposeLayer* this, size_t batch_size, size_t in_width, size_t in_height) {
    this->batch_size = batch_size;
    this->in_width = in_width;
    this->in_height = in_height;

    this->out_width = (in_width - 1) * this->stride - 2 * this->padding + this->kernel_size + this->output_padding;
    this->out_height = (in_height - 1) * this->stride - 2 * this->padding + this->kernel_size + this->output_padding;

    size_t out_size = batch_size * this->out_channels * this->out_width * this->out_height;
    
    if (this->output) free(this->output);
    this->output = malloc(sizeof(float) * out_size);
}

void conv_transpose_prepare_training(ConvTransposeLayer* this) {
    if (this->din) return;

    size_t in_size = this->batch_size * this->in_channels * this->in_width * this->in_height;
    size_t weight_size = this->in_channels * this->out_channels * this->kernel_size * this->kernel_size;
    size_t bias_size = this->out_channels;

    this->din = malloc(sizeof(float) * in_size);
    this->dweights = malloc(sizeof(float) * weight_size);
    this->dbiases = malloc(sizeof(float) * bias_size);

    this->weights_p = calloc(weight_size, sizeof(float));
    this->weights_q = calloc(weight_size, sizeof(float));
    this->biases_p = calloc(bias_size, sizeof(float));
    this->biases_q = calloc(bias_size, sizeof(float));
}

void conv_transpose_zero_grad(ConvTransposeLayer* this) {
    if (!this->din) return;
    size_t in_size = this->batch_size * this->in_channels * this->in_width * this->in_height;
    size_t weight_size = this->in_channels * this->out_channels * this->kernel_size * this->kernel_size;
    size_t bias_size = this->out_channels;

    memset(this->din, 0, sizeof(float) * in_size);
    memset(this->dweights, 0, sizeof(float) * weight_size);
    memset(this->dbiases, 0, sizeof(float) * bias_size);
}

void conv_transpose_forward(ConvTransposeLayer* this, const float* X) {
    this->in_cache = X;
    size_t in_map_size = this->in_channels * this->in_width * this->in_height;
    size_t out_map_size = this->out_channels * this->out_width * this->out_height;
    size_t out_spatial = this->out_width * this->out_height;
    size_t in_spatial = this->in_width * this->in_height;
    size_t col_size = (this->out_channels * this->kernel_size * this->kernel_size) * in_spatial;
    
    // Temporary buffer for col (result of W^T * X)
    float* col = malloc(sizeof(float) * col_size);

    for (size_t b = 0; b < this->batch_size; b++) {
        const float* img_ptr = X + b * in_map_size;
        float* out_ptr = this->output + b * out_map_size;

        // Zero output because col2im accumulates
        memset(out_ptr, 0, sizeof(float) * out_map_size);

        // 1. matmul_1t: W^T [C_out*k*k, C_in] * X [C_in, H_in*W_in] = col [C_out*k*k, H_in*W_in]
        // W is [C_in, C_out*k*k]
        // img_ptr is [C_in, H_in*W_in]
        size_t w_rows = this->in_channels;
        size_t w_cols = this->out_channels * this->kernel_size * this->kernel_size;
        matmul_1t(this->weights, img_ptr, col, w_cols, w_rows, in_spatial);

        // 2. col2im(col, out_ptr)
        // col2im expects out_ptr to be [C_out, H_out, W_out]
        // It uses in_width/in_height logic to determine col size, but here we pass out_width/out_height
        // col2im(col, im, channels, height, width, ...)
        // Here channels=out_channels, height=out_height, width=out_width
        // It calculates out_h_conv = (out_height + 2p - k)/s + 1 = in_height. Correct.
        col2im(col, out_ptr, this->out_channels, this->out_height, this->out_width, this->kernel_size, this->stride, this->padding);

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

void conv_transpose_backward(ConvTransposeLayer* this, const float* dout) {
    size_t in_map_size = this->in_channels * this->in_width * this->in_height;
    size_t out_map_size = this->out_channels * this->out_width * this->out_height;
    size_t out_spatial = this->out_width * this->out_height;
    size_t in_spatial = this->in_width * this->in_height;
    size_t kernel_dim = this->out_channels * this->kernel_size * this->kernel_size;
    size_t col_size = kernel_dim * in_spatial;

    float* col = malloc(sizeof(float) * col_size); 

    for (size_t b = 0; b < this->batch_size; b++) {
        const float* img_ptr = this->in_cache + b * in_map_size;
        const float* dout_ptr = dout + b * out_map_size;
        float* din_ptr = this->din + b * in_map_size;

        // im2col on dout
        // im2col(im, col, channels, height, width, ...)
        // Here im=dout [C_out, H_out, W_out]
        // col will be [C_out*k*k, H_in*W_in]
        im2col(dout_ptr, col, this->out_channels, this->out_height, this->out_width, this->kernel_size, this->stride, this->padding);

        // dL/dX = W * col
        // W: [C_in, C_out*k*k]
        // col: [C_out*k*k, H_in*W_in]
        // dX: [C_in, H_in*W_in]
        // Use matmul(W, col, din)
        matmul(this->weights, col, din_ptr, this->in_channels, kernel_dim, in_spatial);

        // dL/dW = X * col^T
        // X: [C_in, H_in*W_in]
        // col: [kernel_dim, H_in*W_in]
        // dW: [C_in, kernel_dim]
        // Use matmul_2t(X, col, batch_dW)
        
        float* batch_dweights = malloc(sizeof(float) * this->in_channels * kernel_dim);
        matmul_2t(img_ptr, col, batch_dweights, this->in_channels, in_spatial, kernel_dim);
        
        // Accumulate dweights
        size_t total_weights = this->in_channels * kernel_dim;
        for (size_t i = 0; i < total_weights; i++) {
            this->dweights[i] += batch_dweights[i];
        }
        free(batch_dweights);

        // dL/db = sum(dout, axis=(spatial, batch))
        for (size_t c = 0; c < this->out_channels; c++) {
            float sum = 0.0f;
            for (size_t i = 0; i < out_spatial; i++) {
                sum += dout_ptr[c * out_spatial + i];
            }
            this->dbiases[c] += sum;
        }
    }

    free(col);
}

void conv_transpose_adam_step(ConvTransposeLayer* this, float lr, float b1, float b2) {
    size_t weight_size = this->in_channels * this->out_channels * this->kernel_size * this->kernel_size;
    adam_update(this->weights, this->dweights, this->weights_p, this->weights_q, weight_size, lr, b1, b2);
    adam_update(this->biases, this->dbiases, this->biases_p, this->biases_q, this->out_channels, lr, b1, b2);
}

void conv_transpose_free(ConvTransposeLayer* this) {
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
