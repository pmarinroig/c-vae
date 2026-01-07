#ifndef DATASET_H
#define DATASET_H

#include <stdlib.h>

typedef struct {
    size_t count;
    size_t width;
    size_t height;
    size_t channels;
    float* data;      // All images flattened: [count * width * height * channels]
    char** filenames; // Array of strings [count]
} Dataset;

/**
 * Loads dataset from binary file and text manifest.
 * bin_path: path to .bin file
 * txt_path: path to .txt manifest
 * Returns pointer to Dataset, or NULL on failure.
 */
Dataset* dataset_load(const char* bin_path, const char* txt_path);

/**
 * Returns a pointer to the start of the image at given index.
 */
float* dataset_get_image(Dataset* this, size_t index);

/**
 * Finds the index of the first image whose filename contains the query.
 * Returns -1 if not found.
 */
int dataset_find_index(Dataset* this, const char* query);

void dataset_free(Dataset* this);

#endif
