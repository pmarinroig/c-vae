#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "layers/affine.h"
#include "layers/relu.h"
#include "math_extra.h"

void dummy_net() {
    printf("Setting up network for XOR training...\n");

    const size_t batch_size = 4;
    const size_t input_dim = 2;
    const size_t hidden_dim = 64;
    const size_t output_dim = 1;
    const size_t iterations = 180;
    const float lr = 0.01f;

    // XOR Data
    float X[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    float Y[] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    // Create Layers
    AffineLayer* fc1 = affine_create(input_dim, hidden_dim);
    ReluLayer* relu = relu_create(hidden_dim);
    AffineLayer* fc2 = affine_create(hidden_dim, output_dim);

    // Init Parameters
    affine_init_parameters(fc1, true); // true for ReLU init
    affine_init_parameters(fc2, false);

    // Prepare for batch size
    affine_prepare_inference(fc1, batch_size);
    relu_prepare_inference(relu, batch_size);
    affine_prepare_inference(fc2, batch_size);

    // Prepare for training
    affine_prepare_training(fc1);
    relu_prepare_training(relu);
    affine_prepare_training(fc2);

    float* dloss = malloc(sizeof(float) * batch_size * output_dim);

    printf("Starting training loop (%zu iterations)...\n", iterations);

    for (size_t i = 0; i < iterations; ++i) {
        // 1. Zero Gradients
        affine_zero_grad(fc1);
        relu_zero_grad(relu);
        affine_zero_grad(fc2);

        // 2. Forward
        affine_forward(fc1, X);
        relu_forward(relu, fc1->output);
        affine_forward(fc2, relu->output);

        // 3. Loss
        float loss = mse_loss(fc2->output, Y, batch_size * output_dim);
        if (i % 30 == 0 || i == iterations - 1) {
            printf("Iter %zu: Loss = %.6f\n", i, loss);
        }

        // 4. Backward
        mse_loss_backward(fc2->output, Y, dloss, batch_size * output_dim);
        affine_backward(fc2, dloss);
        relu_backward(relu, fc2->din);
        affine_backward(fc1, relu->din);

        // 5. Update
        affine_adam_step(fc1, lr, 0.9f, 0.999f);
        affine_adam_step(fc2, lr, 0.9f, 0.999f);
    }

    // Check predictions
    printf("\nFinal Predictions:\n");
    for(size_t i=0; i<batch_size; ++i) {
        printf("Input: (%.1f, %.1f) -> Target: %.1f, Prediction: %.4f\n", 
               X[i*2], X[i*2+1], Y[i], fc2->output[i]);
    }

    free(dloss);
}