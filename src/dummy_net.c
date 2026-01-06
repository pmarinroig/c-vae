#include <stdio.h>
#include <stdbool.h>
#include "layers/affine.h"

void dummy_net() {
    const float X[] = {1.0f, 2.0f, 1.0f};
    AffineLayer* layer1 = affine_create(1, 10);
    AffineLayer* layer2 = affine_create(10, 2);
    affine_init_parameters(layer1, false);
    affine_prepare(layer1, 3);
    affine_init_parameters(layer2, false);
    affine_prepare(layer2, 3);

    affine_forward(layer1, X, 3);
    affine_forward(layer2, layer1->output, 3);

    for (int i = 0; i < 3; i++) {
        printf("Batch %d (%.5f, %.5f)\n", i, layer2->output[i], layer2->output[i + 3]);
    }
}