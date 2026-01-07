#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vae.h"
#include "dataset.h"

void save_ppm(const char* filename, float* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        unsigned char r = (unsigned char)(data[i*3 + 0] * 255.0f);
        unsigned char g = (unsigned char)(data[i*3 + 1] * 255.0f);
        unsigned char b = (unsigned char)(data[i*3 + 2] * 255.0f);
        fputc(r, f);
        fputc(g, f);
        fputc(b, f);
    }
    fclose(f);
}

void chw_to_hwc(const float* chw, float* hwc, int width, int height) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < 3; c++) {
                hwc[h*width*3 + w*3 + c] = chw[c*height*width + h*width + w];
            }
        }
    }
}

void get_average_vector(VAE* vae, Dataset* ds, const char* query, float* avg_mu) {
    int count = 0;
    float* sum_mu = calloc(LATENT_DIM, sizeof(float));
    float* temp_mu = malloc(sizeof(float) * LATENT_DIM);
    float* input_buf = malloc(sizeof(float) * 3 * 16 * 16);
    
    for (size_t i = 0; i < ds->count; i++) {
        if (strstr(ds->filenames[i], query)) {
            float* img = dataset_get_image(ds, i);
            for (int h = 0; h < 16; h++) {
                for (int w = 0; w < 16; w++) {
                    for (int c = 0; c < 3; c++) {
                        input_buf[c*16*16 + h*16 + w] = img[(h*16+w)*3 + c];
                    }
                }
            }
            vae_encode(vae, input_buf, temp_mu);
            for (int k = 0; k < LATENT_DIM; k++) {
                sum_mu[k] += temp_mu[k];
            }
            count++;
        }
    }
    
    if (count > 0) {
        for (int k = 0; k < LATENT_DIM; k++) {
            avg_mu[k] = sum_mu[k] / count;
        }
        printf("Averaged %d items for query '%s'\n", count, query);
    } else {
        printf("No items found for query '%s'\n", query);
    }
    
    free(sum_mu);
    free(temp_mu);
    free(input_buf);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <base_item> <minus_concept> <plus_concept>\n", argv[0]);
        return 1;
    }
    
    const char* base_name = argv[1];
    const char* minus_name = argv[2];
    const char* plus_name = argv[3];
    
    Dataset* ds = dataset_load("pytorch_poc/mc_items.bin", "pytorch_poc/mc_items.txt");
    if (!ds) return 1;
    
    VAE* vae = vae_create();
    vae_init(vae);
    vae_prepare(vae, 1, false);
    vae_load(vae, "vae_weights.bin");
    
    // 1. Base
    int base_idx = dataset_find_index(ds, base_name);
    if (base_idx == -1) {
        printf("Base item '%s' not found.\n", base_name);
        return 1;
    }
    float* base_mu = malloc(sizeof(float) * LATENT_DIM);
    float* input_buf = malloc(sizeof(float) * 3 * 16 * 16);
    float* base_img = dataset_get_image(ds, base_idx);
    
    // Save Base Image (already HWC)
    save_ppm("base.ppm", base_img, 16, 16);
    
    // HWC -> CHW for encoding
    for (int h = 0; h < 16; h++) {
        for (int w = 0; w < 16; w++) {
            for (int c = 0; c < 3; c++) {
                input_buf[c*16*16 + h*16 + w] = base_img[(h*16+w)*3 + c];
            }
        }
    }
    vae_encode(vae, input_buf, base_mu);
    printf("Encoded base item '%s' (Index %d)\n", ds->filenames[base_idx], base_idx);
    
    // 2. Minus
    float* minus_mu = calloc(LATENT_DIM, sizeof(float));
    get_average_vector(vae, ds, minus_name, minus_mu);
    
    // 3. Plus
    float* plus_mu = calloc(LATENT_DIM, sizeof(float));
    get_average_vector(vae, ds, plus_name, plus_mu);
    
    // 4. Arithmetic
    float* result_mu = malloc(sizeof(float) * LATENT_DIM);
    for (int k = 0; k < LATENT_DIM; k++) {
        result_mu[k] = base_mu[k] - minus_mu[k] + plus_mu[k];
    }
    
    // 5. Decode & Save
    float* output_chw = malloc(sizeof(float) * 3 * 16 * 16);
    float* output_hwc = malloc(sizeof(float) * 3 * 16 * 16);
    
    // Save Minus Reconstruction
    vae_decode(vae, minus_mu, output_chw);
    chw_to_hwc(output_chw, output_hwc, 16, 16);
    save_ppm("minus.ppm", output_hwc, 16, 16);
    
    // Save Plus Reconstruction
    vae_decode(vae, plus_mu, output_chw);
    chw_to_hwc(output_chw, output_hwc, 16, 16);
    save_ppm("plus.ppm", output_hwc, 16, 16);
    
    // Save Result
    vae_decode(vae, result_mu, output_chw);
    chw_to_hwc(output_chw, output_hwc, 16, 16);
    save_ppm("result.ppm", output_hwc, 16, 16);
    
    printf("Saved base.ppm, minus.ppm, plus.ppm, result.ppm\n");
    
    free(base_mu);
    free(minus_mu);
    free(plus_mu);
    free(result_mu);
    free(input_buf);
    free(output_chw);
    free(output_hwc);
    vae_free(vae);
    dataset_free(ds);
    
    return 0;
}