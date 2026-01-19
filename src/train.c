#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vae.h"
#include "dataset.h"

#define BATCH_SIZE 32
#define EPOCHS 500
#define LR 0.001f
#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define SEED 0

// Helper to shuffle indices
void shuffle(size_t* array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            size_t t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(void) {
    srand(SEED);
    
    printf("Loading dataset...\n");
    Dataset* ds = dataset_load("pytorch_poc/mc_items.bin", "pytorch_poc/mc_items.txt");
    if (!ds) {
        fprintf(stderr, "Failed to load dataset\n");
        return 1;
    }
    printf("Dataset loaded: %zu images.\n", ds->count);
    
    VAE* vae = vae_create();
    vae_init(vae);
    
    // Check if weights exist and load them
    FILE* fcheck = fopen("vae_weights.bin", "rb");
    if (fcheck) {
        fclose(fcheck);
        printf("Loading existing weights from 'vae_weights.bin'...\n");
        vae_load(vae, "vae_weights.bin");
    } else {
        printf("No existing weights found. Training from scratch.\n");
    }

    vae_prepare(vae, BATCH_SIZE, true);
    
    size_t* indices = malloc(sizeof(size_t) * ds->count);
    for(size_t i=0; i<ds->count; ++i) indices[i] = i;
    
    // Buffer for batch data
    size_t img_size = 3 * 16 * 16;
    float* batch_data = malloc(sizeof(float) * BATCH_SIZE * img_size);
    
    printf("Starting training for %d epochs...\n", EPOCHS);
    
    time_t start_time = time(NULL);
    int t = 1; // Adam timestep

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle(indices, ds->count);
        float total_loss = 0.0f;
        int batches = 0;
        
        for (size_t i = 0; i < ds->count; i += BATCH_SIZE) {
            size_t current_batch_size = (i + BATCH_SIZE <= ds->count) ? BATCH_SIZE : (ds->count - i);
            if (current_batch_size < BATCH_SIZE) break; // Drop last incomplete batch
            
            // Prepare batch
            #pragma omp parallel for
            for (size_t b = 0; b < BATCH_SIZE; b++) {
                float* img = dataset_get_image(ds, indices[i + b]);
                const float* src = img; 
                // HWC -> CHW
                float* dst = batch_data + b * img_size;
                
                for (int h = 0; h < 16; h++) {
                    for (int w = 0; w < 16; w++) {
                        for (int c = 0; c < 3; c++) {
                            // src: h*16*3 + w*3 + c
                            // dst: c*16*16 + h*16 + w
                            dst[c * 16 * 16 + h * 16 + w] = src[(h * 16 + w) * 3 + c];
                        }
                    }
                }
            }
            
            vae_zero_grad(vae);
            vae_forward(vae, batch_data);
            float loss = vae_backward(vae, batch_data);
            total_loss += loss;
            vae_step(vae, LR, ADAM_B1, ADAM_B2, t);
            t++;
            
            batches++;
        }
        
        printf("Epoch %d: Avg Loss = %.4f\n", epoch + 1, total_loss / batches);
    }
    
    time_t end_time = time(NULL);
    double duration = difftime(end_time, start_time);
    printf("Training complete in %.0f seconds (%d min %d sec).\n", 
           duration, (int)duration / 60, (int)duration % 60);

    printf("Saving model...\n");
    vae_save(vae, "vae_weights.bin");
    
    free(batch_data);
    free(indices);
    vae_free(vae);
    dataset_free(ds);
    return 0;
}
