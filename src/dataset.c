#include "dataset.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

Dataset* dataset_load(const char* bin_path, const char* txt_path) {
    FILE* fb = fopen(bin_path, "rb");
    if (!fb) {
        perror("Failed to open binary dataset");
        return NULL;
    }

    char magic[4];
    if (fread(magic, 1, 4, fb) != 4 || memcmp(magic, "MCVA", 4) != 0) {
        fprintf(stderr, "Invalid magic number in binary dataset\n");
        fclose(fb);
        return NULL;
    }

    Dataset* this = malloc(sizeof(Dataset));
    uint32_t count, width, height, channels;
    
    if (fread(&count, 4, 1, fb) != 1 ||
        fread(&width, 4, 1, fb) != 1 ||
        fread(&height, 4, 1, fb) != 1 ||
        fread(&channels, 4, 1, fb) != 1) {
        fprintf(stderr, "Failed to read header from binary dataset\n");
        fclose(fb);
        free(this);
        return NULL;
    }

    this->count = count;
    this->width = width;
    this->height = height;
    this->channels = channels;

    size_t num_elements = (size_t)count * width * height * channels;
    this->data = malloc(sizeof(float) * num_elements);
    
    uint8_t* raw_bytes = malloc(num_elements);
    if (fread(raw_bytes, 1, num_elements, fb) != num_elements) {
        fprintf(stderr, "Failed to read data from binary dataset\n");
        free(raw_bytes);
        free(this->data);
        free(this);
        fclose(fb);
        return NULL;
    }
    fclose(fb);

    // Convert to float 0-1
    for (size_t i = 0; i < num_elements; i++) {
        this->data[i] = (float)raw_bytes[i] / 255.0f;
    }
    free(raw_bytes);

    // Load Manifest
    FILE* ft = fopen(txt_path, "r");
    if (!ft) {
        perror("Failed to open manifest");
        this->filenames = NULL;
        return this; // Still return dataset, just no names
    }

    this->filenames = malloc(sizeof(char*) * count);
    char buffer[256];
    for (size_t i = 0; i < count; i++) {
        if (fgets(buffer, sizeof(buffer), ft)) {
            // Remove newline
            buffer[strcspn(buffer, "\n")] = 0;
            this->filenames[i] = strdup(buffer);
        } else {
            this->filenames[i] = strdup("unknown");
        }
    }
    fclose(ft);

    return this;
}

float* dataset_get_image(Dataset* this, size_t index) {
    if (index >= this->count) return NULL;
    return this->data + (index * this->width * this->height * this->channels);
}

int dataset_find_index(Dataset* this, const char* query) {
    if (!this->filenames) return -1;
    for (size_t i = 0; i < this->count; i++) {
        if (strstr(this->filenames[i], query)) {
            return (int)i;
        }
    }
    return -1;
}

void dataset_free(Dataset* this) {
    if (!this) return;
    free(this->data);
    if (this->filenames) {
        for (size_t i = 0; i < this->count; i++) {
            free(this->filenames[i]);
        }
        free(this->filenames);
    }
    free(this);
}
