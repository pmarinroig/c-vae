#ifndef VAE_H
#define VAE_H

#include "layers/conv.h"
#include "layers/conv_transpose.h"
#include "layers/affine.h"
#include "layers/relu.h"
#include "layers/sigmoid.h"

#define LATENT_DIM 16

typedef struct {
    // Encoder
    ConvLayer* enc_conv1; // 3->64
    ReluLayer* enc_relu1;
    ConvLayer* enc_conv2; // 64->128
    ReluLayer* enc_relu2;
    
    // Latent
    // Input to affine: 128 * 4 * 4 = 2048
    AffineLayer* fc_mu;     // 2048 -> 16
    AffineLayer* fc_logvar; // 2048 -> 16
    
    // Sampler buffers
    float* z;       // [batch, 16]
    float* eps;     // [batch, 16]
    
    // Decoder
    AffineLayer* dec_fc;    // 16 -> 2048
    // Unflatten implicit: 2048 -> 128x4x4
    ConvTransposeLayer* dec_convt1; // 128->64
    ReluLayer* dec_relu1;
    ConvTransposeLayer* dec_convt2; // 64->64
    ReluLayer* dec_relu2;
    ConvLayer* dec_final;   // 64->3 (Refining conv)
    SigmoidLayer* dec_sigmoid;

    // Output pointer (points to dec_sigmoid->output)
    float* output; 
    
    // State
    size_t batch_size;
} VAE;

VAE* vae_create();
void vae_init(VAE* this); 
void vae_prepare(VAE* this, size_t batch_size, bool training);
void vae_zero_grad(VAE* this);
void vae_forward(VAE* this, const float* X);
// Backward returns total loss
float vae_backward(VAE* this, const float* target); 
void vae_step(VAE* this, float lr, float b1, float b2);
void vae_free(VAE* this);

// Inference helpers
void vae_encode(VAE* this, const float* X, float* mu);
void vae_decode(VAE* this, const float* z, float* output);

void vae_save(VAE* this, const char* path);
void vae_load(VAE* this, const char* path);

#endif
