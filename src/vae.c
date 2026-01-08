#include "vae.h"
#include "math_extra.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

VAE* vae_create() {
    VAE* this = calloc(1, sizeof(VAE));
    
    // Encoder
    this->enc_conv1 = conv_create(3, 64, 3, 2, 1);
    this->enc_relu1 = relu_create(64 * 8 * 8); // Input dim per batch item
    this->enc_conv2 = conv_create(64, 128, 3, 2, 1);
    this->enc_relu2 = relu_create(128 * 4 * 4);
    
    // Latent
    this->fc_mu = affine_create(2048, LATENT_DIM);
    this->fc_logvar = affine_create(2048, LATENT_DIM);
    
    // Decoder
    this->dec_fc = affine_create(LATENT_DIM, 2048);
    this->dec_convt1 = conv_transpose_create(128, 64, 3, 2, 1, 1);
    this->dec_relu1 = relu_create(64 * 8 * 8);
    this->dec_convt2 = conv_transpose_create(64, 64, 3, 2, 1, 1);
    this->dec_relu2 = relu_create(64 * 16 * 16);
    this->dec_final = conv_create(64, 3, 3, 1, 1);
    this->dec_sigmoid = sigmoid_create(3 * 16 * 16);
    
    return this;
}

void vae_init(VAE* this) {
    conv_init_parameters(this->enc_conv1, true);
    conv_init_parameters(this->enc_conv2, true);
    affine_init_parameters(this->fc_mu, false);
    affine_init_parameters(this->fc_logvar, false);
    affine_init_parameters(this->dec_fc, true); 
    conv_transpose_init_parameters(this->dec_convt1, true);
    conv_transpose_init_parameters(this->dec_convt2, true);
    conv_init_parameters(this->dec_final, false); 
}

void vae_prepare(VAE* this, size_t batch_size, bool training) {
    this->batch_size = batch_size;
    
    conv_prepare_inference(this->enc_conv1, batch_size, 16, 16);
    relu_prepare_inference(this->enc_relu1, batch_size);
    conv_prepare_inference(this->enc_conv2, batch_size, 8, 8);
    relu_prepare_inference(this->enc_relu2, batch_size);
    
    affine_prepare_inference(this->fc_mu, batch_size);
    affine_prepare_inference(this->fc_logvar, batch_size);
    
    if (this->z) free(this->z);
    this->z = malloc(sizeof(float) * batch_size * LATENT_DIM);
    if (this->eps) free(this->eps);
    this->eps = malloc(sizeof(float) * batch_size * LATENT_DIM);
    
    affine_prepare_inference(this->dec_fc, batch_size);
    conv_transpose_prepare_inference(this->dec_convt1, batch_size, 4, 4);
    relu_prepare_inference(this->dec_relu1, batch_size);
    conv_transpose_prepare_inference(this->dec_convt2, batch_size, 8, 8);
    relu_prepare_inference(this->dec_relu2, batch_size);
    conv_prepare_inference(this->dec_final, batch_size, 16, 16);
    sigmoid_prepare_inference(this->dec_sigmoid, batch_size);
    
    this->output = this->dec_sigmoid->output;
    
    if (training) {
        conv_prepare_training(this->enc_conv1);
        relu_prepare_training(this->enc_relu1);
        conv_prepare_training(this->enc_conv2);
        relu_prepare_training(this->enc_relu2);
        affine_prepare_training(this->fc_mu);
        affine_prepare_training(this->fc_logvar);
        
        affine_prepare_training(this->dec_fc);
        conv_transpose_prepare_training(this->dec_convt1);
        relu_prepare_training(this->dec_relu1);
        conv_transpose_prepare_training(this->dec_convt2);
        relu_prepare_training(this->dec_relu2);
        conv_prepare_training(this->dec_final);
        sigmoid_prepare_training(this->dec_sigmoid);
    }
}

void vae_zero_grad(VAE* this) {
    conv_zero_grad(this->enc_conv1);
    relu_zero_grad(this->enc_relu1);
    conv_zero_grad(this->enc_conv2);
    relu_zero_grad(this->enc_relu2);
    affine_zero_grad(this->fc_mu);
    affine_zero_grad(this->fc_logvar);
    affine_zero_grad(this->dec_fc);
    conv_transpose_zero_grad(this->dec_convt1);
    relu_zero_grad(this->dec_relu1);
    conv_transpose_zero_grad(this->dec_convt2);
    relu_zero_grad(this->dec_relu2);
    conv_zero_grad(this->dec_final);
    sigmoid_zero_grad(this->dec_sigmoid);
}

void vae_forward(VAE* this, const float* X) {
    // Encoder
    conv_forward(this->enc_conv1, X);
    relu_forward(this->enc_relu1, this->enc_conv1->output);
    conv_forward(this->enc_conv2, this->enc_relu1->output);
    relu_forward(this->enc_relu2, this->enc_conv2->output);
    
    // Latent
    affine_forward(this->fc_mu, this->enc_relu2->output);
    affine_forward(this->fc_logvar, this->enc_relu2->output);
    
    // Reparameterization
    size_t size = this->batch_size * LATENT_DIM;
    for (size_t i = 0; i < size; i++) {
        float mu = this->fc_mu->output[i];
        float logvar = this->fc_logvar->output[i];
        float std = expf(0.5f * logvar);
        float eps = rand_normal(); 
        this->eps[i] = eps;
        this->z[i] = mu + eps * std;
    }
    
    // Decoder
    affine_forward(this->dec_fc, this->z);
    conv_transpose_forward(this->dec_convt1, this->dec_fc->output);
    relu_forward(this->dec_relu1, this->dec_convt1->output);
    conv_transpose_forward(this->dec_convt2, this->dec_relu1->output);
    relu_forward(this->dec_relu2, this->dec_convt2->output);
    conv_forward(this->dec_final, this->dec_relu2->output);
    sigmoid_forward(this->dec_sigmoid, this->dec_final->output);
}

float vae_backward(VAE* this, const float* target) {
    size_t out_size = this->batch_size * 3 * 16 * 16;
    float* dloss = malloc(sizeof(float) * out_size);
    
    // 1. Reconstruction Loss (MSE)
    float recon_loss = mse_loss(this->output, target, out_size);
    mse_loss_backward(this->output, target, dloss, out_size);
    
    // 2. Backprop through decoder
    sigmoid_backward(this->dec_sigmoid, dloss);
    conv_backward(this->dec_final, this->dec_sigmoid->din);
    relu_backward(this->dec_relu2, this->dec_final->din);
    conv_transpose_backward(this->dec_convt2, this->dec_relu2->din);
    relu_backward(this->dec_relu1, this->dec_convt2->din);
    conv_transpose_backward(this->dec_convt1, this->dec_relu1->din);
    affine_backward(this->dec_fc, this->dec_convt1->din);
    
    // 3. Backprop through reparameterization & KL Divergence
    // dL/dmu = dL/dz + mu
    // dL/dlogvar = dL/dz * eps * 0.5 * exp(0.5*logvar) + 0.5 * (exp(logvar) - 1)
    
    size_t latent_size = this->batch_size * LATENT_DIM;
    float kld_loss = 0.0f;
    
    float* dout_mu = malloc(sizeof(float) * latent_size);
    float* dout_logvar = malloc(sizeof(float) * latent_size);
    
    const float* dL_dz = this->dec_fc->din;
    
    for (size_t i = 0; i < latent_size; i++) {
        float mu = this->fc_mu->output[i];
        float logvar = this->fc_logvar->output[i];
        float eps = this->eps[i];
        float std = expf(0.5f * logvar);
        
        kld_loss += -0.5f * (1.0f + logvar - mu * mu - expf(logvar));
        
        // Gradients
        float dRecon_dmu = dL_dz[i];
        float dRecon_dlogvar = dL_dz[i] * eps * 0.5f * std;
        
        float dKLD_dmu = mu;
        float dKLD_dlogvar = 0.5f * (expf(logvar) - 1.0f);
        
        // Scale KLD gradients to match MSE scale (averaged over output size)
        dout_mu[i] = dRecon_dmu + (dKLD_dmu / out_size);
        dout_logvar[i] = dRecon_dlogvar + (dKLD_dlogvar / out_size);
    }
    
    kld_loss /= out_size; 
    
    // 4. Backprop through Encoder
    affine_backward(this->fc_logvar, dout_logvar);
    affine_backward(this->fc_mu, dout_mu);
    
    // Sum gradients at enc_relu2 output (fork)
    size_t flat_size = this->batch_size * 2048;
    float* combined_din = malloc(sizeof(float) * flat_size);
    for (size_t i = 0; i < flat_size; i++) {
        combined_din[i] = this->fc_mu->din[i] + this->fc_logvar->din[i];
    }
    
    relu_backward(this->enc_relu2, combined_din);
    conv_backward(this->enc_conv2, this->enc_relu2->din);
    relu_backward(this->enc_relu1, this->enc_conv2->din);
    conv_backward(this->enc_conv1, this->enc_relu1->din);
    
    free(dloss);
    free(dout_mu);
    free(dout_logvar);
    free(combined_din);
    
    return recon_loss + kld_loss;
}

void vae_step(VAE* this, float lr, float b1, float b2) {
    conv_adam_step(this->enc_conv1, lr, b1, b2);
    conv_adam_step(this->enc_conv2, lr, b1, b2);
    
    affine_adam_step(this->fc_mu, lr, b1, b2);
    affine_adam_step(this->fc_logvar, lr, b1, b2);
    
    affine_adam_step(this->dec_fc, lr, b1, b2);
    conv_transpose_adam_step(this->dec_convt1, lr, b1, b2);
    conv_transpose_adam_step(this->dec_convt2, lr, b1, b2);
    conv_adam_step(this->dec_final, lr, b1, b2);
}

void vae_free(VAE* this) {
    conv_free(this->enc_conv1);
    relu_free(this->enc_relu1);
    conv_free(this->enc_conv2);
    relu_free(this->enc_relu2);
    
    affine_free(this->fc_mu);
    affine_free(this->fc_logvar);
    
    if (this->z) free(this->z);
    if (this->eps) free(this->eps);
    
    affine_free(this->dec_fc);
    conv_transpose_free(this->dec_convt1);
    relu_free(this->dec_relu1);
    conv_transpose_free(this->dec_convt2);
    relu_free(this->dec_relu2);
    conv_free(this->dec_final);
    sigmoid_free(this->dec_sigmoid);
    
    free(this);
}

void vae_encode(VAE* this, const float* X, float* mu) {
    conv_forward(this->enc_conv1, X);
    relu_forward(this->enc_relu1, this->enc_conv1->output);
    conv_forward(this->enc_conv2, this->enc_relu1->output);
    relu_forward(this->enc_relu2, this->enc_conv2->output);
    
    affine_forward(this->fc_mu, this->enc_relu2->output);
    
    size_t size = this->batch_size * LATENT_DIM;
    memcpy(mu, this->fc_mu->output, sizeof(float) * size);
}

void vae_decode(VAE* this, const float* z, float* output) {
    affine_forward(this->dec_fc, z);
    conv_transpose_forward(this->dec_convt1, this->dec_fc->output);
    relu_forward(this->dec_relu1, this->dec_convt1->output);
    conv_transpose_forward(this->dec_convt2, this->dec_relu1->output);
    relu_forward(this->dec_relu2, this->dec_convt2->output);
    conv_forward(this->dec_final, this->dec_relu2->output);
    sigmoid_forward(this->dec_sigmoid, this->dec_final->output);
    
    size_t size = this->batch_size * 3 * 16 * 16;
    memcpy(output, this->dec_sigmoid->output, sizeof(float) * size);
}

void vae_save(VAE* this, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    
    size_t s;
    s = this->enc_conv1->out_channels * this->enc_conv1->in_channels * this->enc_conv1->kernel_size * this->enc_conv1->kernel_size;
    fwrite(this->enc_conv1->weights, sizeof(float), s, f);
    fwrite(this->enc_conv1->biases, sizeof(float), this->enc_conv1->out_channels, f);
    
    s = this->enc_conv2->out_channels * this->enc_conv2->in_channels * this->enc_conv2->kernel_size * this->enc_conv2->kernel_size;
    fwrite(this->enc_conv2->weights, sizeof(float), s, f);
    fwrite(this->enc_conv2->biases, sizeof(float), this->enc_conv2->out_channels, f);
    
    s = this->fc_mu->in_dim * this->fc_mu->out_dim;
    fwrite(this->fc_mu->weights, sizeof(float), s, f);
    fwrite(this->fc_mu->biases, sizeof(float), this->fc_mu->out_dim, f);
    
    s = this->fc_logvar->in_dim * this->fc_logvar->out_dim;
    fwrite(this->fc_logvar->weights, sizeof(float), s, f);
    fwrite(this->fc_logvar->biases, sizeof(float), this->fc_logvar->out_dim, f);
    
    s = this->dec_fc->in_dim * this->dec_fc->out_dim;
    fwrite(this->dec_fc->weights, sizeof(float), s, f);
    fwrite(this->dec_fc->biases, sizeof(float), this->dec_fc->out_dim, f);
    
    s = this->dec_convt1->in_channels * this->dec_convt1->out_channels * this->dec_convt1->kernel_size * this->dec_convt1->kernel_size;
    fwrite(this->dec_convt1->weights, sizeof(float), s, f);
    fwrite(this->dec_convt1->biases, sizeof(float), this->dec_convt1->out_channels, f);
    
    s = this->dec_convt2->in_channels * this->dec_convt2->out_channels * this->dec_convt2->kernel_size * this->dec_convt2->kernel_size;
    fwrite(this->dec_convt2->weights, sizeof(float), s, f);
    fwrite(this->dec_convt2->biases, sizeof(float), this->dec_convt2->out_channels, f);
    
    s = this->dec_final->out_channels * this->dec_final->in_channels * this->dec_final->kernel_size * this->dec_final->kernel_size;
    fwrite(this->dec_final->weights, sizeof(float), s, f);
    fwrite(this->dec_final->biases, sizeof(float), this->dec_final->out_channels, f);
    
    fclose(f);
}

void vae_load(VAE* this, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return;
    
    size_t s;
    int err = 0;

    #define READ_CHECK(ptr, size, count, stream) \
        if (fread(ptr, size, count, stream) != count) err = 1;

    s = this->enc_conv1->out_channels * this->enc_conv1->in_channels * this->enc_conv1->kernel_size * this->enc_conv1->kernel_size;
    READ_CHECK(this->enc_conv1->weights, sizeof(float), s, f);
    READ_CHECK(this->enc_conv1->biases, sizeof(float), this->enc_conv1->out_channels, f);
    
    s = this->enc_conv2->out_channels * this->enc_conv2->in_channels * this->enc_conv2->kernel_size * this->enc_conv2->kernel_size;
    READ_CHECK(this->enc_conv2->weights, sizeof(float), s, f);
    READ_CHECK(this->enc_conv2->biases, sizeof(float), this->enc_conv2->out_channels, f);
    
    s = this->fc_mu->in_dim * this->fc_mu->out_dim;
    READ_CHECK(this->fc_mu->weights, sizeof(float), s, f);
    READ_CHECK(this->fc_mu->biases, sizeof(float), this->fc_mu->out_dim, f);
    
    s = this->fc_logvar->in_dim * this->fc_logvar->out_dim;
    READ_CHECK(this->fc_logvar->weights, sizeof(float), s, f);
    READ_CHECK(this->fc_logvar->biases, sizeof(float), this->fc_logvar->out_dim, f);
    
    s = this->dec_fc->in_dim * this->dec_fc->out_dim;
    READ_CHECK(this->dec_fc->weights, sizeof(float), s, f);
    READ_CHECK(this->dec_fc->biases, sizeof(float), this->dec_fc->out_dim, f);
    
    s = this->dec_convt1->in_channels * this->dec_convt1->out_channels * this->dec_convt1->kernel_size * this->dec_convt1->kernel_size;
    READ_CHECK(this->dec_convt1->weights, sizeof(float), s, f);
    READ_CHECK(this->dec_convt1->biases, sizeof(float), this->dec_convt1->out_channels, f);
    
    s = this->dec_convt2->in_channels * this->dec_convt2->out_channels * this->dec_convt2->kernel_size * this->dec_convt2->kernel_size;
    READ_CHECK(this->dec_convt2->weights, sizeof(float), s, f);
    READ_CHECK(this->dec_convt2->biases, sizeof(float), this->dec_convt2->out_channels, f);
    
    s = this->dec_final->out_channels * this->dec_final->in_channels * this->dec_final->kernel_size * this->dec_final->kernel_size;
    READ_CHECK(this->dec_final->weights, sizeof(float), s, f);
    READ_CHECK(this->dec_final->biases, sizeof(float), this->dec_final->out_channels, f);
    
    if (err) {
        fprintf(stderr, "Warning: Failed to read all weights from %s\n", path);
    }

    fclose(f);
    #undef READ_CHECK
}