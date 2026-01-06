#ifndef LAYER_H
#define LAYER_H

// This struct defines the interface that all layers must implement
struct Layer;

typedef struct {
    float* (*affine_forward)(struct Layer* self);
    double (*area)(struct Layer* self);
} LayerVTable;

typedef struct {
    const LayerVTable* vptr;
} Layer;

#endif